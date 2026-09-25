"""viam:isaac-sim-devin:palletizer - the service that drives
``viam:pack-sequencer:sequencer`` through a full pack: it asks the
sequencer for the next slot, moves the arm there, reports what happened,
and publishes the box's measured pose once physics has settled. The
sequencer owns the pack order, the placement cursor and every place
target; this service owns none of that, it only executes and reports.

Attributes:
  world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
  arm (string, required)      - name of the arm component (boot ordering only; every
                                 motion goes through "motion")
  gripper (string, required)  - name of the gripper component (driven only through the
                                 Viam Gripper API: open/grab/is_holding_something, so
                                 either a jaw or a vacuum gripper model works unchanged)
  motion (string, required)   - name of the motion service ("builtin" works)
  sequencer (string, required) - name of the viam:pack-sequencer:sequencer service
  box_props (list of strings, required, non-empty) - prop names in pick order; box_props[i]
                                 fills the sequencer's seq i + 1
  place_support_prop (string, optional, default "frame_pallet") - the prop whose airspace
                                 the arm keeps out of except while placing onto it. A
                                 frame.geometry collider spawns as frame_<component>
  obstacle_source (string, optional, default "world_state_store") - where the motion
                                 service's obstacles come from: "world_state_store" leaves
                                 obstacle assembly to the frame system and the store, the
                                 only other accepted value "prop_geometries" keeps this
                                 service's own hand-built WorldState

DoCommand:
  {"command": "start"} -> runs the whole pack. {"ok": true, "state": "running"}, or
    {"ok": false, "state": "running"} unchanged when already running |
  {"command": "stop"} -> cancels between motions, never mid-motion. {"ok": true} |
  {"command": "status"} -> {"state": "idle|running|stopping|complete|failed",
    "records": [PalletizerRecord.to_dict(), ...], "reason": "<why a failed run died>"}

Only one box_props entry sits on the pick station at a time - eight 150 mm boxes touching
would not fit its 1100 mm length. box_props[0]'s pose on the first prop_geometries read of
a run is captured as the infeed pose, sourced from the sim rather than from a configured
pick pose (deleting one invented pose only to add another would be a straight trade).
Every later box_props[i] is re-posed there, through the world's set_prop_pose verb, before
its own pick.

The sequence per box, every motion through the motion service, never through the arm
directly: (box_props[i], i >= 1, first) re-pose the prop to the infeed pose, then move
above the box at a standoff, descend onto its top face (straight line), grab() on the
gripper - a False return fails the record with a reason - lift (straight line), move to
the sequencer's place_start_in_world, descend to place_end_in_world (straight line, the
sequencer's own approach standoff already carries the clearance), open(), retreat to
place_start_in_world. The outcome is reported to the sequencer via report_placement; a box
that fails twice in a row is skip_box'ed rather than retried a third time, since a retry
only happens because the sequencer's cursor stayed put.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from grpclib import Status
from grpclib.exceptions import GRPCError
from typing_extensions import Self
from viam.components.arm import Arm
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    ResourceName,
    Transform,
    Vector3,
    WorldState,
)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import MotionClient
from viam.utils import ValueTypes, struct_to_dict

from pickcell.movers import RealMover
from pickcell.obstacles import (
    obstacles_from_prop_geometries,
    pick_area_keepout,
    support_obstacle,
    world_state,
)
from pickcell.pipeline import GripperApi, Mover, WorldApi
from pickcell.poses import PRE_GRASP_STANDOFF_MM, _pointing_down, _pose_to_dict

from .. import DEFAULT_WORLD_NAME, FAMILY, NAMESPACE
from ..asset_catalog import CUP_APPROACH_GAP_MM
from ..sequencer_client import NextBox, SequencerClient
from ..sort_plan import OUTCOME_FAILED, OUTCOME_PLACED
from ..stack_stability import DEFAULT_SETTLE_POLICY, Sample, settle_time_s, trailing_window_settled
from ..store_relay import RelayReading, WorldStateStoreLike, read_store_obstacles

LOGGER = getLogger(__name__)

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_COMPLETE = "complete"
STATE_FAILED = "failed"

_DEPENDENCY_ATTRS = ("world", "arm", "gripper", "motion", "sequencer")

_OBSTACLE_SOURCE_WORLD_STATE_STORE = "world_state_store"
_OBSTACLE_SOURCE_PROP_GEOMETRIES = "prop_geometries"
_OBSTACLE_SOURCES = (_OBSTACLE_SOURCE_WORLD_STATE_STORE, _OBSTACLE_SOURCE_PROP_GEOMETRIES)
_DEFAULT_OBSTACLE_SOURCE = _OBSTACLE_SOURCE_WORLD_STATE_STORE

# the floor everything else stands on, so a discrete collision check cannot
# step a link through it mid-swing (the same reasoning as
# pickcell.obstacles.support_obstacle's own docstring). Only used on the
# "prop_geometries" obstacle path.
_FLOOR_Z_MM = 0.0

# how far above its slot the box's bottom is when the cup lets go. The
# sequencer's place_end puts the cup where the box's top face rides when the
# box sits flat on its slot, and the surface gripper draws the box's top face
# up to the cup on close, so a descent to place_end itself sets the box down
# and the arm stops when the deck stops the box. Zero means release at
# contact, the way a real palletizer sets a box down, and _descend_to_release
# takes that stop as the touchdown it is.
#
# Measured on 2026-09-22 with the EPick held through the surface gripper: the
# descent to place_end stopped with the box's bottom 0.00 mm over the deck and
# the released box settled 0.10 to 0.16 mm from its slot, in both arm
# configurations, with no wrist fold. The 25 mm drop this replaces landed
# 0.35 to 0.64 mm out and covered a wrist fold the compliant cups no longer cause.
PLACE_RELEASE_CLEARANCE_MM = 0.0

# how far from the release pose the cup may stop, box on the deck, and still
# be released. A place descent that stalls with the cup this close to its
# target has been stopped by the surface it was placing onto, or by a
# neighbouring box, and letting go is what the descent was for. Measured on
# 2026-09-22: two descents onto a slot flush with the pallet's corner stalled
# on their last waypoint with the cup 7 mm above the release height and the
# box's bottom edge caught on the deck's top edge, and each box released flat
# once let go. Anything further away is a stall against something else, and
# is reported as one.
PLACE_STALL_TOLERANCE_MM = 25.0

# how far above the sequencer's own place_start the carry arrives, before a
# straight descent onto place_start begins the placement proper. The
# sequencer's standoff is box_height_mm + 10, sized for a load that hangs
# rigid. A box on the cups swings: 7.5 to 10 degrees during a carry, measured
# 2026-09-23, settling to 0.001 within a second of stopping. At 10 degrees a
# 125 mm half-diagonal dips 22 mm, past the 10 mm the standoff leaves over a
# same-layer neighbour, and on 2026-09-24 box 3 arrived swinging over boxes
# 1 and 2 and stalled on them. 50 mm clears that swing by 28 mm; under half a
# box height, so the straight settle stays short.
PLACE_APPROACH_CLEARANCE_MM = 50.0

# the planner seeds every plan with 0 unless told otherwise, so a free plan
# that timed out times out again on identical inputs. Measured on 2026-10-02
# on viam-server 1.9.0: the level carry timed out at the 60 s bound once in 33
# attempts. Seed 1 is the one other draw taken.
RETRY_PLANNER_SEED = 1

# the first attempt of a free move gets this much planning time, the retry
# keeps the mover's own bound. On 2026-10-02 the slowest of 57 successful free
# plans across three packs took 0.27 s, so 10 s is about 37 times that. A first
# attempt that times out anyway costs 10 s before the re-seeded retry, where
# the default bound would have spent 60 s.
FIRST_FREE_PLAN_TIMEOUT_S = 10.0

# the arm model's own words for a move that planned but could not be
# executed, `ArmMoveStalledError` and `ArmMoveTimeoutError` in models/arm.py.
# They cross the wire as text inside the motion service's error, so text is
# what tells them from a planner that refused to plan at all.
_EXECUTION_FAILURE_MARKERS = ("stalled at waypoint", "did not reach final waypoint")

# a seq that fails twice running is skip_box'ed rather than retried a third
# time - the sequencer's own cursor already gave it one retry for free by
# leaving itself put on the first failure.
MAX_ATTEMPTS_PER_SEQ = 2

# how still a box has to be before the cup is sent down onto it, and how long
# to wait for that.
#
# A pose read while the box is still moving puts the grasp pose where the box
# WAS. On the GPU run of 2026-09-16 the box was released from a 100 mm lift and
# read a moment later, mid-fall, so the descent stopped 5 mm under a top face
# the box had already left and the cup closed on air. Every restage has the
# same shape: `set_prop_pose` returns before physics has moved the prop.
SETTLE_POLICY = DEFAULT_SETTLE_POLICY

# how close a restaged box's centre has to read to the position it was sent
# to before the teleport counts as landed. The checklist's own reset uses
# this as its tolerance for the same question - has a prop actually reached
# where it was told to go.
RESTAGE_TOLERANCE_MM = 10.0

# how long one teleport gets to land before it is sent again, and how long the
# retry gets before the restage gives up. On the GPU run of 2026-09-25 the
# read straight after `set_prop_pose` still showed the box at its parked spot
# for the whole settle window, seven packs out of ten, and the pick that
# followed drove the cup down at the parked spot's x rather than the infeed
# pose's.
RESTAGE_TIMEOUT_S = 2.0

# how many teleports one restage gets before giving up: one send, two retries.
# Across seven GPU packs on 2026-10-01 and 2026-10-02 a restage needed a second
# teleport in six of them, and on 2026-10-02 one box missed both and still read
# at its parked spot. Each of the other second teleports landed.
RESTAGE_MAX_ATTEMPTS = 3


class RestageError(ValueError):
    """A box did not read back within `RESTAGE_TOLERANCE_MM` of the pose it
    was `set_prop_pose`'d to, after a retry."""


def _box_top_face_xyz_mm(
    geometries: Sequence[Mapping[str, Any]], box_prop: str
) -> tuple[float, float, float] | None:
    """``box_prop``'s current (x, y, top-face z) in the world, from a
    ``prop_geometries`` reading, or None when the prop is not there."""
    for geometry in geometries:
        if geometry.get("name") != box_prop:
            continue
        pose = geometry["pose_in_world_mm"]
        _dim_x, _dim_y, dim_z = geometry["box_dims_mm"]
        return (float(pose["x"]), float(pose["y"]), float(pose["z"]) + float(dim_z) / 2.0)
    return None


def _box_pose_mm(
    geometries: Sequence[Mapping[str, Any]], box_prop: str
) -> Mapping[str, Any] | None:
    """``box_prop``'s full ``pose_in_world_mm`` from a ``prop_geometries``
    reading, or None when the prop is not there."""
    for geometry in geometries:
        if geometry.get("name") == box_prop:
            return cast("Mapping[str, Any]", geometry["pose_in_world_mm"])
    return None


def _landed_within(
    pose_mm: Mapping[str, Any], target_mm: tuple[float, float, float], tolerance_mm: float
) -> bool:
    """Whether ``pose_mm``'s centre reads within ``tolerance_mm`` of
    ``target_mm``, straight-line."""
    x, y, z = target_mm
    return (
        math.sqrt(
            (float(pose_mm["x"]) - x) ** 2
            + (float(pose_mm["y"]) - y) ** 2
            + (float(pose_mm["z"]) - z) ** 2
        )
        <= tolerance_mm
    )


def _pose_from_mm(pose_mm: Mapping[str, Any]) -> Pose:
    """A ``pose_in_world_mm`` reading as a ``viam.proto.common.Pose``, with no
    top-face correction: the world state store centres its geometry on
    exactly the pose it is given, so this is the box's CENTRE."""
    return Pose(
        x=float(pose_mm["x"]),
        y=float(pose_mm["y"]),
        z=float(pose_mm["z"]),
        o_x=float(pose_mm["o_x"]),
        o_y=float(pose_mm["o_y"]),
        o_z=float(pose_mm["o_z"]),
        theta=float(pose_mm["theta"]),
    )


def _measured_box_pose(geometries: Sequence[Mapping[str, Any]], box_prop: str) -> Pose | None:
    """``box_prop``'s pose as physics settled it, read straight off
    ``prop_geometries`` rather than where the plan intended it - the whole
    point of feeding it back to ``set_box_transform``.

    In the sequencer's convention, the same one its ``place_end_in_world``
    and every ``target_pose_mm`` here use: the pose of the box's TOP face.
    ``prop_geometries`` reports the body's centre, so z is raised by half
    the box's own height. Publishing the centre drew every placed box half
    sunk into the one below it and read as a 50 mm error on 2026-09-24."""
    pose_mm = _box_pose_mm(geometries, box_prop)
    box = _prop_box_mm(geometries, box_prop)
    if pose_mm is None or box is None:
        return None
    _centre_mm, (_dim_x, _dim_y, dim_z) = box
    return Pose(
        x=float(pose_mm["x"]),
        y=float(pose_mm["y"]),
        z=float(pose_mm["z"]) + float(dim_z) / 2.0,
        o_x=float(pose_mm["o_x"]),
        o_y=float(pose_mm["o_y"]),
        o_z=float(pose_mm["o_z"]),
        theta=float(pose_mm["theta"]),
    )


def pick_grasp_pose(top_face_xyz_mm: tuple[float, float, float]) -> Pose:
    """The gripper TCP pose just above the box's top face, pointing straight
    down. CUP_APPROACH_GAP_MM short of the face rather than on it, since a cup
    driven onto a box stalls the arm against it."""
    x, y, top_face_z = top_face_xyz_mm
    return _pointing_down(x, y, top_face_z + CUP_APPROACH_GAP_MM)


def pick_grasp_standoff_pose(top_face_xyz_mm: tuple[float, float, float]) -> Pose:
    """The stationary pose above the box the arm descends from and lifts
    back to, by PRE_GRASP_STANDOFF_MM above its top face."""
    x, y, top_face_z = top_face_xyz_mm
    return _pointing_down(x, y, top_face_z + CUP_APPROACH_GAP_MM + PRE_GRASP_STANDOFF_MM)


def place_approach_pose(place_start: Pose) -> Pose:
    """Where the carry ends: the sequencer's `place_start_in_world` raised by
    PLACE_APPROACH_CLEARANCE_MM, so the box has stopped swinging before it is
    lowered to place_start in a straight line and from there onto its slot.
    Only z changes."""
    return Pose(
        x=place_start.x,
        y=place_start.y,
        z=place_start.z + PLACE_APPROACH_CLEARANCE_MM,
        o_x=place_start.o_x,
        o_y=place_start.o_y,
        o_z=place_start.o_z,
        theta=place_start.theta,
    )


def carry_height_pose(top_face_xyz_mm: tuple[float, float, float], carry_z_mm: float) -> Pose:
    """Straight above the pick, at the height the carry runs at: the
    approach's z, so the held box clears the place keep-out before the
    sweep. Pointing down, like every pick pose."""
    standoff = pick_grasp_standoff_pose(top_face_xyz_mm)
    return Pose(
        x=standoff.x,
        y=standoff.y,
        z=carry_z_mm,
        o_x=standoff.o_x,
        o_y=standoff.o_y,
        o_z=standoff.o_z,
        theta=standoff.theta,
    )


def place_release_pose(place_end: Pose) -> Pose:
    """Where the cup lets go: the sequencer's `place_end_in_world` raised by
    the release clearance.

    `place_end` is the cup pose with the box's top face at the cup and the
    box on its slot, so a descent all the way to `place_end` sets the box
    on its slot. `PLACE_RELEASE_CLEARANCE_MM` is how far above that the cup
    lets go, zero for release at contact. Only z changes: the slot's x, y
    and orientation are the sequencer's business."""
    return Pose(
        x=place_end.x,
        y=place_end.y,
        z=place_end.z + PLACE_RELEASE_CLEARANCE_MM,
        o_x=place_end.o_x,
        o_y=place_end.o_y,
        o_z=place_end.o_z,
        theta=place_end.theta,
    )


def held_box_transform(
    box_prop: str, box_dims_mm: tuple[float, float, float], gripper_name: str
) -> Transform:
    """The box the cup is carrying, as geometry the planner moves with the
    gripper: a frame named after the box, parented to the gripper frame,
    carrying the box's own dimensions.

    Without it the planner knows the arm and nothing hanging from it. On
    2026-09-22 it carried a box from the pick to the pallet through a
    one-segment plan that flipped the elbow 183 degrees and wrist 2 180
    degrees, and the box swung through the arm's own forearm and stalled
    it mid-flip. With it, that segment is a collision and is never planned.

    The gripper frame's z axis is the tool axis and points down at a grasp
    (`_pointing_down`'s orientation vector is (0, 0, -1)), so the box hangs
    along the gripper's +z, its top face at the cup: half the box's height
    to its centre."""
    _length_mm, _width_mm, height_mm = box_dims_mm
    centre_below_cup_mm = height_mm / 2.0
    return Transform(
        reference_frame=box_prop,
        pose_in_observer_frame=PoseInFrame(
            reference_frame=gripper_name,
            pose=Pose(x=0.0, y=0.0, z=centre_below_cup_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        ),
        physical_object=Geometry(
            center=Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
            box=RectangularPrism(
                dims_mm=Vector3(x=box_dims_mm[0], y=box_dims_mm[1], z=box_dims_mm[2])
            ),
            label=box_prop,
        ),
    )


def touched_down(
    cup_pose: Pose, release_pose: Pose, tolerance_mm: float = PLACE_STALL_TOLERANCE_MM
) -> bool:
    """Whether a cup that stopped short of `release_pose` is close enough to
    it that what stopped it was the surface under the box."""
    return (
        math.sqrt(
            (cup_pose.x - release_pose.x) ** 2
            + (cup_pose.y - release_pose.y) ** 2
            + (cup_pose.z - release_pose.z) ** 2
        )
        <= tolerance_mm
    )


def is_execution_failure(error: BaseException) -> bool:
    """Whether a refused move planned fine and failed on the arm, blocked by
    contact or out of time, rather than being refused by the planner."""
    message = str(error)
    return any(marker in message for marker in _EXECUTION_FAILURE_MARKERS)


# the prop whose airspace the arm keeps out of until it is placing. A
# `frame.geometry` collider spawns as `frame_<component>`, and a prim name
# cannot hold a hyphen, so the pallet component's collider is this. Named as an
# attribute rather than hardcoded, since which component carries the place
# support is the cell's business and not this service's.
DEFAULT_PLACE_SUPPORT_PROP = "frame_pallet"

# how far a keep-out's ceiling sits above what it protects. The pick keep-out
# has to stay BELOW the standoff it guards, or the arm cannot reach the pose it
# descends from, so this is headroom the approach still has.
PICK_KEEPOUT_HEADROOM_MM = 20.0

# The place keep-out's ceiling is the top face of the layer being placed
# (`place_end.z`, the sequencer's own pose), which is at least as high as every
# box already on the pallet, so the stack is covered whole. Until 2026-09-24 the
# ceiling stopped 20 mm under it and the placed boxes' top 20 mm stood out of
# the zone, which is where a carried box's corner or a retreating cup clipped
# them. The held box at `place_start` hangs 10 mm above this ceiling (the
# sequencer's standoff is box height plus 10), and the carry arrives from
# PLACE_APPROACH_CLEARANCE_MM above that, so the approach stays plannable.
# The legs the arm runs empty near the stack use a second keep-out raised by
# PLACE_APPROACH_CLEARANCE_MM, since a ceiling flush with the top face leaves
# the planner no margin over it (see `_place_one_box`).


def _prop_box_mm(
    geometries: Sequence[Mapping[str, Any]], name: str
) -> tuple[Mapping[str, Any], tuple[float, float, float]] | None:
    for geometry in geometries:
        if geometry.get("name") == name:
            dims = geometry["box_dims_mm"]
            return geometry["pose_in_world_mm"], (
                float(dims[0]),
                float(dims[1]),
                float(dims[2]),
            )
    return None


def _box_footprint_keepout(
    geometries: Sequence[Mapping[str, Any]], box_prop: str, label: str
) -> Geometry | None:
    """A keep-out standing in for ``box_prop`` itself: its own footprint,
    from the deck up PICK_KEEPOUT_HEADROOM_MM past its top face. Shared by
    `box_keepout` and `placed_box_keepout`, which differ only in which box
    this is and what a planner violation should call it."""
    found = _prop_box_mm(geometries, box_prop)
    if found is None:
        return None
    pose, (dim_x, dim_y, dim_z) = found
    bottom_z = float(pose["z"]) - dim_z / 2.0
    return pick_area_keepout(
        (
            (float(pose["x"]) - dim_x / 2.0, float(pose["y"]) - dim_y / 2.0, bottom_z),
            (float(pose["x"]) + dim_x / 2.0, float(pose["y"]) + dim_y / 2.0, bottom_z),
        ),
        height_mm=dim_z + PICK_KEEPOUT_HEADROOM_MM,
        label=label,
    )


def box_keepout(geometries: Sequence[Mapping[str, Any]], box_prop: str) -> Geometry | None:
    """The airspace over the box waiting to be picked, which the arm keeps out
    of until it is descending onto it.

    The box cannot be an obstacle in its own right, since the cup has to reach
    it. Without something in its place the planner routes a link straight
    through it: on the GPU runs of 2026-09-16 the swing to the standoff
    knocked the box off the station before the descent had begun, and every
    later item measured a box that was no longer where it had been put.

    The ceiling stops PICK_KEEPOUT_HEADROOM_MM above the box's own top, which
    is well under the standoff, so the pose the arm descends from stays
    reachable."""
    return _box_footprint_keepout(geometries, box_prop, "pick_box_keepout")


def placed_box_keepout(geometries: Sequence[Mapping[str, Any]], box_prop: str) -> Geometry | None:
    """The airspace over the box just released onto its slot, which the
    retreat keeps out of.

    Same footprint and headroom as `box_keepout` - the box just placed is as
    solid as the one about to be picked - under its own label, so a refusal
    names the placed box rather than the pick station's. On the GPU run of
    2026-09-25 a straight retreat clipped the box it had just set down once
    the free fallback (which cannot see this keep-out either) took over."""
    return _box_footprint_keepout(geometries, box_prop, "placed_box_keepout")


def support_keepout(
    geometries: Sequence[Mapping[str, Any]],
    support_prop: str,
    ceiling_z_mm: float,
    label: str = "place_support_keepout",
) -> Geometry | None:
    """The airspace over the place support, which the arm keeps out of until
    it is descending onto it.

    Covers the whole support rather than one slot, since what it protects is
    the boxes already stacked on it, and where those are is the sequencer's
    business rather than this service's. Returns None when the support is not
    in the scene, or when `ceiling_z_mm` leaves no room above its top face.

    `label` is what a planner violation calls the zone, and what tells two
    keep-outs over the same support apart: the legs that carry a box run
    under one that reaches only the top of the layer being placed, and the
    empty-armed legs near the stack under one that reaches
    PLACE_APPROACH_CLEARANCE_MM higher (see `_place_one_box`)."""
    found = _prop_box_mm(geometries, support_prop)
    if found is None:
        return None
    pose, (dim_x, dim_y, dim_z) = found
    top_z = float(pose["z"]) + dim_z / 2.0
    height_mm = ceiling_z_mm - top_z
    if height_mm <= 0.0:
        return None
    return pick_area_keepout(
        (
            (float(pose["x"]) - dim_x / 2.0, float(pose["y"]) - dim_y / 2.0, top_z),
            (float(pose["x"]) + dim_x / 2.0, float(pose["y"]) + dim_y / 2.0, top_z),
        ),
        height_mm=height_mm,
        label=label,
    )


def is_planner_timeout(error: BaseException) -> bool:
    """Whether a refused move ran the planner out of time, so that another
    random seed is a fresh draw. The arm's own timeout carries the same
    status, which is why an execution failure is excluded: that plan was fine."""
    return (
        isinstance(error, GRPCError)
        and error.status == Status.DEADLINE_EXCEEDED
        and not is_execution_failure(error)
    )


async def move_linear_or_free(
    move: Callable[[Pose, bool], Awaitable[bool]], pose: Pose, what: str
) -> bool:
    """Move to `pose` in a straight line, or by any path the planner will give
    when the straight line is refused.

    A straight line is what keeps a cup off a box's edges on the way down and a
    carried box level on the way up, so it is tried first. It is not always
    available. A free move returns whichever inverse-kinematics solution the
    planner picks for that cup pose, and from some of them no straight-line
    plan exists at all.

    Measured on 2026-09-16 from the configuration the pick standoff kept
    landing in: the straight line was refused at 100 mm and at 50 mm, at
    10 mm / 10 degree and at 50 mm / 30 degree tolerances, with the cell's
    obstacles and with none, while an unconstrained move to the same pose
    planned first time. Approaching again does not help, since the standoff
    returns the same solution every time it is asked, so the fallback is the
    recovery rather than a retry.

    The fallback is for a planner that refused. A straight line that planned
    and then stalled on the arm is a different thing: the arm is blocked by
    contact, and a free retry from the blocked pose plans a move of under a
    degree, stalls the same way, and buries the first message under the
    second. On 2026-09-22 that turned "the box touched the deck 5 mm early"
    into a two-waypoint plan that read like a failed approach.
    """
    try:
        return await move(pose, True)
    # a planner refusal is logged and the fallback below is what the caller
    # gets. A fallback that is also refused raises on its own.
    except Exception as error:
        if is_execution_failure(error):
            # the leg's name is the one fact the arm's own message lacks
            LOGGER.error("straight-line %s planned and then stalled on the arm: %s", what, error)
            raise
        LOGGER.warning("straight-line %s refused (%s), moving without the constraint", what, error)
    return await move(pose, False)


async def approach_and_descend(
    approach: Callable[[Pose, bool], Awaitable[bool]],
    descend: Callable[[Pose, bool], Awaitable[bool]],
    standoff: Pose,
    grasp: Pose,
) -> bool:
    """Drive to `standoff`, then descend onto `grasp`.

    The two callers differ in one thing: whether the box about to be picked is
    in the planner's obstacle set. It has to be out for the descent, since
    reaching the box is what the descent is for. Leaving it out for the
    APPROACH as well lets the planner route a link straight through it, and on
    the GPU runs of 2026-09-16 that is what happened. The arm swung across the
    station on its way to the standoff and knocked the box onto the floor
    before the descent had begun, so every later item measured a box that was
    no longer where it had been put.

    Returns whether both moves happened. False is a caller-side stop landing
    between motions, never a planning failure: a descent the planner will not
    give at all raises."""
    if not await approach(standoff, False):
        return False
    return await move_linear_or_free(descend, grasp, "grasp descent")


@contextlib.asynccontextmanager
async def _leg_timeline(seq: int, name: str) -> AsyncIterator[None]:
    """Times one motion leg of one box's pick and place: logs when it
    starts, how long it took once it finishes, or how long it ran before it
    raised - the error is logged and re-raised, never swallowed.

    On the GPU run of 2026-09-28, every leg planned straight and executed
    without a stall, and the second box of the second layer was still
    knocked 524 mm off the pallet while the far row was placed beside it.
    The pose data at the end could not say which leg did it; this can."""
    LOGGER.info("palletizer seq %d leg %s: start", seq, name)
    start = time.monotonic()
    try:
        yield
    except Exception as error:
        LOGGER.warning(
            "palletizer seq %d leg %s: failed after %.2f s: %s",
            seq,
            name,
            time.monotonic() - start,
            error,
        )
        raise
    LOGGER.info("palletizer seq %d leg %s: done in %.2f s", seq, name, time.monotonic() - start)


@dataclass(frozen=True)
class PalletizerRecord:
    """One box's terminal outcome for one attempt: the sequencer's ``seq``
    and target pose, the box's measured pose once placed (None when it
    never got that far), the outcome, duration and any failure reason."""

    box_prop: str
    seq: int
    target_pose_mm: Mapping[str, float]
    outcome: str  # one of the OUTCOME_* literals from sort_plan
    duration_s: float
    reason: str | None = None
    measured_pose_mm: Mapping[str, float] | None = None
    planner_obstacles: Mapping[str, int] | None = None
    settle_s: float | None = None
    swing_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "box_prop": self.box_prop,
            "seq": self.seq,
            "target_pose_mm": dict(self.target_pose_mm),
            "outcome": self.outcome,
            "duration_s": self.duration_s,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        if self.measured_pose_mm is not None:
            result["measured_pose_mm"] = dict(self.measured_pose_mm)
        if self.planner_obstacles is not None:
            result["planner_obstacles"] = dict(self.planner_obstacles)
        if self.settle_s is not None:
            result["settle_s"] = self.settle_s
        if self.swing_s is not None:
            result["swing_s"] = self.swing_s
        return result


def _outcome_record(
    box_prop: str,
    seq: int,
    target_pose_mm: Mapping[str, float],
    outcome: str,
    duration_s: float,
    reason: str | None = None,
    measured_pose_mm: Mapping[str, float] | None = None,
    planner_obstacles: Mapping[str, int] | None = None,
    settle_s: float | None = None,
    swing_s: float | None = None,
) -> PalletizerRecord:
    """One box's terminal outcome."""
    return PalletizerRecord(
        box_prop=box_prop,
        seq=seq,
        target_pose_mm=target_pose_mm,
        outcome=outcome,
        duration_s=duration_s,
        reason=reason,
        measured_pose_mm=measured_pose_mm,
        planner_obstacles=planner_obstacles,
        settle_s=settle_s,
        swing_s=swing_s,
    )


def _valid_box_props(config_name: str, attrs: Mapping[str, Any]) -> list[str]:
    box_props = attrs.get("box_props")
    if (
        not isinstance(box_props, Sequence)
        or isinstance(box_props, (str, bytes))
        or not box_props
        or not all(isinstance(prop, str) for prop in box_props)
    ):
        raise ValueError(
            f'{config_name}: set the "box_props" attribute to a non-empty list of prop names'
        )
    return [str(prop) for prop in box_props]


def _valid_obstacle_source(config_name: str, attrs: Mapping[str, Any]) -> str:
    obstacle_source = str(attrs.get("obstacle_source", _DEFAULT_OBSTACLE_SOURCE))
    if obstacle_source not in _OBSTACLE_SOURCES:
        raise ValueError(
            f'{config_name}: set the "obstacle_source" attribute to one of {_OBSTACLE_SOURCES}'
        )
    return obstacle_source


def _valid_world_state_stores(config_name: str, attrs: Mapping[str, Any]) -> list[str]:
    """The world state store resource names to relay obstacles from, in
    ``world_state_store`` obstacle mode: the configured ``world_state_stores``
    list, or just the sequencer when it is not set - the sequencer's own
    store is the one every cell already has."""
    stores = attrs.get("world_state_stores")
    if stores is None:
        return [str(attrs["sequencer"])]
    if (
        not isinstance(stores, Sequence)
        or isinstance(stores, (str, bytes))
        or not stores
        or not all(isinstance(name, str) and name for name in stores)
    ):
        raise ValueError(
            f'{config_name}: set the "world_state_stores" attribute to a non-empty list of '
            "resource names"
        )
    return [str(name) for name in stores]


def _validated_stop_after_seq(raw: ValueTypes | None) -> int | None:
    """The `start` command's `stop_after_seq`, an int of at least 1, or None
    for an unbounded run.

    A protobuf Struct carries every number as a double, so an int sent
    through DoCommand crosses the wire as a float with no fractional part -
    a caller's 4 arrives here as 4.0, and is stored back as 4. A float with a
    fractional part, a string, a bool or anything under 1 raises."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f'"stop_after_seq" must be an int of at least 1, got {raw!r}')
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f'"stop_after_seq" must be an int of at least 1, got {raw!r}')
    value = int(raw)
    if value < 1:
        raise ValueError(f'"stop_after_seq" must be an int of at least 1, got {raw!r}')
    return value


class IsaacPalletizer(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the service, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "palletizer")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._state: str = STATE_IDLE
        self._task: asyncio.Task[None] | None = None
        self._cancel_requested = False
        self._records: list[PalletizerRecord] = []
        self._failure_reason: str | None = None
        # box_props[0]'s pose on seq 1's first prop_geometries read - every
        # later box_props[i] is re-posed there before its own pick. Captured
        # once and left in place across every later start: only a part
        # restart parks a fresh box_props[0] on the station, and a run that
        # reset this on every start forgot it the moment a bounded run
        # stopped, leaving the start that resumed the pack nothing to
        # restage onto (see _place_one_box's seq > 1, pose-still-None case).
        self._infeed_position_mm: tuple[float, float, float] | None = None
        # the seq a bounded run stops after, or None for unbounded. Reset per
        # run (_handle_start).
        self._stop_after_seq: int | None = None
        self._stopped_after_seq: int | None = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        palletizer = cls(config.name)
        palletizer.reconfigure(config, dependencies)
        return palletizer

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        dependencies: list[str] = []
        for key in _DEPENDENCY_ATTRS:
            default = DEFAULT_WORLD_NAME if key == "world" else None
            value = attrs.get(key, default)
            if not value or not isinstance(value, str):
                raise ValueError(f'{config.name}: set the "{key}" attribute to a resource name')
            dependencies.append(value)

        _valid_box_props(config.name, attrs)
        _valid_obstacle_source(config.name, attrs)
        for store_name in _valid_world_state_stores(config.name, attrs):
            if store_name not in dependencies:
                dependencies.append(store_name)
        return dependencies, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        attrs.setdefault("world", DEFAULT_WORLD_NAME)
        by_name: dict[str, ResourceBase] = {
            rn.name: resource for rn, resource in dependencies.items()
        }

        def dep(key: str) -> ResourceBase:
            resource_name = str(attrs[key])
            if resource_name not in by_name:
                raise ValueError(
                    f"{config.name}: dependency {resource_name!r} for {key!r} was not resolved"
                )
            return by_name[resource_name]

        self._world = cast("WorldApi", dep("world"))
        self._arm = cast(Arm, dep("arm"))
        self._gripper = cast("GripperApi", dep("gripper"))
        self._gripper_name = str(attrs["gripper"])
        self._motion = cast(MotionClient, dep("motion"))
        self._sequencer = SequencerClient(dep("sequencer"))
        self._box_props = _valid_box_props(config.name, attrs)
        self._obstacle_source = _valid_obstacle_source(config.name, attrs)
        self._place_support_prop = str(attrs.get("place_support_prop", DEFAULT_PLACE_SUPPORT_PROP))

        def store(name: str) -> WorldStateStoreLike:
            if name not in by_name:
                raise ValueError(
                    f"{config.name}: dependency {name!r} for 'world_state_stores' was not resolved"
                )
            return cast("WorldStateStoreLike", by_name[name])

        store_names = _valid_world_state_stores(config.name, attrs)
        self._stores: list[tuple[str, WorldStateStoreLike]] = [
            (name, store(name)) for name in store_names
        ]

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Mapping[str, ValueTypes]:
        cmd = str(command.get("command", ""))
        if cmd == "start":
            return self._handle_start(command)
        if cmd == "stop":
            return self._handle_stop()
        if cmd == "status":
            return cast("Mapping[str, ValueTypes]", self._status_snapshot())
        raise ValueError(f"unknown command {cmd!r}; supported: start, stop, status")

    def _handle_start(self, command: Mapping[str, ValueTypes]) -> dict[str, ValueTypes]:
        if self._state in (STATE_RUNNING, STATE_STOPPING):
            return {"ok": False, "state": self._state}
        stop_after_seq = _validated_stop_after_seq(command.get("stop_after_seq"))
        self._state = STATE_RUNNING
        self._records = []
        self._failure_reason = None
        self._cancel_requested = False
        self._stop_after_seq = stop_after_seq
        self._stopped_after_seq = None
        self._task = asyncio.create_task(self._run())
        return {"ok": True, "state": "running"}

    def _handle_stop(self) -> dict[str, ValueTypes]:
        if self._state == STATE_RUNNING:
            self._state = STATE_STOPPING
        self._cancel_requested = True
        return {"ok": True}

    def _status_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "state": self._state,
            "records": [record.to_dict() for record in self._records],
        }
        # a run that died carries WHY. Without this, `status` answered
        # `{"state": "failed", "records": []}` and a caller had to go read the
        # machine's logs to learn anything at all.
        if self._failure_reason is not None:
            snapshot["reason"] = self._failure_reason
        if self._stopped_after_seq is not None:
            snapshot["stopped_after_seq"] = self._stopped_after_seq
        return snapshot

    async def wait_until_done(self) -> None:
        """Test-only join on the background pack task started by ``start``.
        Never awaited from ``do_command``: production callers poll ``status``."""
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        try:
            await self._pack_all_boxes()
            self._state = STATE_IDLE if self._cancel_requested else STATE_COMPLETE
        # the reason is recorded on the service and logged, and `status` is
        # what a caller reads it from
        except Exception as error:
            self._state = STATE_FAILED
            self._failure_reason = f"{type(error).__name__}: {error}"
            LOGGER.exception("palletizer run failed")
        finally:
            self._cancel_requested = False

    async def _descend_to_release(
        self, move: Callable[[Pose, bool], Awaitable[bool]], release: Pose
    ) -> bool:
        """The place descent, with a stall near the release pose taken as the
        box touching down rather than as a failure.

        A rigid arm holding a rigid box against a rigid deck cannot reach a
        pose the box has already stopped it short of, and the last few
        millimetres of a descent onto a slot flush with the pallet's edge are
        where a box's bottom edge meets the deck's top edge. Real palletizers
        release on contact. A stall further away is something else in the way,
        and it is raised as the stall it was."""
        try:
            return await move_linear_or_free(move, release, "place descent")
        except Exception as error:
            if not is_execution_failure(error):
                raise
            cup = await self._cup_pose_in_world()
            if cup is None or not touched_down(cup, release):
                raise
            LOGGER.warning(
                "place descent stalled with the cup at %s, within %.0f mm of the release pose "
                "%s: the box is down, releasing",
                _pose_to_dict(cup),
                PLACE_STALL_TOLERANCE_MM,
                _pose_to_dict(release),
            )
            return True

    async def _cup_pose_in_world(self) -> Pose | None:
        """Where the motion service says the gripper's frame is, in the world,
        or None when it cannot say."""
        try:
            in_frame = await self._motion.get_pose(
                component_name=self._gripper_name, destination_frame="world"
            )
        # the pose is diagnostic for a stall already in hand: failing to read
        # it leaves the stall to be reported, never hides it
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("could not read the cup's pose after a stall: %s", error)
            return None
        return cast(Pose, in_frame.pose)

    def _build_mover(self) -> Mover:
        # RealMover's camera_name only matters for look_from, which this
        # service never calls (there is no scan, the box pose comes from the
        # sequencer's own slot) - the gripper name is a harmless stand-in
        # rather than a fourth dependency the config would otherwise have to
        # name.
        return RealMover(self._motion, self._gripper_name, self._gripper_name)

    async def _prop_geometries(self) -> Sequence[Mapping[str, Any]]:
        response = await self._world.do_command({"command": "prop_geometries"})
        return cast("Sequence[Mapping[str, Any]]", response.get("geometries", []))

    async def _settled_geometries(
        self, box_prop: str
    ) -> tuple[Sequence[Mapping[str, Any]], float | None]:
        """`prop_geometries`, read once `box_prop` has stopped moving, and how
        long that took.

        The grasp pose is derived from this reading, and a cup sent to where a
        falling box was closes on air. Returns the last reading either way, so
        a box that never settles is picked against its most recent pose and
        the attempt fails on its own terms rather than here. The settle time
        is None when the box never settled inside the policy's timeout."""
        geometries = await self._prop_geometries()
        start = time.monotonic()
        samples: list[Sample] = [(0.0, _box_pose_mm(geometries, box_prop))]
        if samples[0][1] is None:
            # a box that is not in the world cannot settle. Waiting the full
            # timeout for one only delays the "no known geometry" the caller
            # is about to report.
            return geometries, None
        deadline = start + SETTLE_POLICY.timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(SETTLE_POLICY.poll_s)
            geometries = await self._prop_geometries()
            samples.append((time.monotonic() - start, _box_pose_mm(geometries, box_prop)))
            if trailing_window_settled(samples, SETTLE_POLICY):
                return geometries, settle_time_s(samples, SETTLE_POLICY)
        LOGGER.warning(
            "%r was still moving after %.1f s; picking against its last reading",
            box_prop,
            SETTLE_POLICY.timeout_s,
        )
        return geometries, None

    async def _settle_after_release(
        self, box_prop: str, released_at: float
    ) -> tuple[Sequence[Mapping[str, Any]], float | None]:
        """`prop_geometries`, polled from the moment `box_prop` was released
        rather than from whenever the caller gets around to reading it.

        The retreat leg that follows a release runs one to two seconds, and a
        box does most of its settling during it. Reading only once the retreat
        finishes would put the whole retreat's motion inside `settle_s` at 0,
        which is why this starts sampling immediately and runs concurrently
        with the retreat rather than after it."""
        geometries = await self._prop_geometries()
        samples: list[Sample] = [
            (time.monotonic() - released_at, _box_pose_mm(geometries, box_prop))
        ]
        while True:
            if trailing_window_settled(samples, SETTLE_POLICY):
                return geometries, settle_time_s(samples, SETTLE_POLICY)
            if time.monotonic() - released_at > SETTLE_POLICY.timeout_s:
                LOGGER.warning(
                    "%r was still moving after %.1f s; picking against its last reading",
                    box_prop,
                    SETTLE_POLICY.timeout_s,
                )
                return geometries, None
            await asyncio.sleep(SETTLE_POLICY.poll_s)
            geometries = await self._prop_geometries()
            samples.append((time.monotonic() - released_at, _box_pose_mm(geometries, box_prop)))

    async def _wait_for_the_carried_box_to_stop_swinging(
        self, seq: int, box_prop: str
    ) -> float | None:
        """Samples `box_prop`'s own centre pose at the place approach, where
        it hangs on the cups and swings (see `PLACE_APPROACH_CLEARANCE_MM`),
        until `trailing_window_settled` says the swing has stopped, or
        `SETTLE_POLICY.timeout_s` runs out. Mirrors `_settle_after_release`'s
        own sampler, started at the carry's own arrival rather than at a
        release."""
        start = time.monotonic()
        geometries = await self._prop_geometries()
        samples: list[Sample] = [(0.0, _box_pose_mm(geometries, box_prop))]
        while True:
            if trailing_window_settled(samples, SETTLE_POLICY):
                swing_s = settle_time_s(samples, SETTLE_POLICY)
                LOGGER.info("palletizer seq %d: carried box still after %.2f s", seq, swing_s)
                return swing_s
            if time.monotonic() - start > SETTLE_POLICY.timeout_s:
                LOGGER.warning(
                    "%r was still swinging after %.1f s; descending anyway. Last two readings: %s",
                    box_prop,
                    SETTLE_POLICY.timeout_s,
                    samples[-2:],
                )
                return None
            await asyncio.sleep(SETTLE_POLICY.poll_s)
            geometries = await self._prop_geometries()
            samples.append((time.monotonic() - start, _box_pose_mm(geometries, box_prop)))

    async def _poll_until_restaged(
        self, box_prop: str, target_mm: tuple[float, float, float]
    ) -> tuple[bool, Mapping[str, Any] | None]:
        """Whether `box_prop` reads within RESTAGE_TOLERANCE_MM of
        `target_mm` inside RESTAGE_TIMEOUT_S, polling at the settle policy's
        own rate, and the last pose read either way."""
        start = time.monotonic()
        geometries = await self._prop_geometries()
        pose_mm = _box_pose_mm(geometries, box_prop)
        if pose_mm is not None and _landed_within(pose_mm, target_mm, RESTAGE_TOLERANCE_MM):
            return True, pose_mm
        deadline = start + RESTAGE_TIMEOUT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(SETTLE_POLICY.poll_s)
            geometries = await self._prop_geometries()
            pose_mm = _box_pose_mm(geometries, box_prop)
            if pose_mm is not None and _landed_within(pose_mm, target_mm, RESTAGE_TOLERANCE_MM):
                return True, pose_mm
        return False, pose_mm

    async def _restage_box(self, box_prop: str) -> None:
        """Re-poses ``box_prop`` onto the pick station's infeed pose, so a
        later box_props entry arrives where box_props[0] started rather than
        wherever it was parked. The caller (`_place_one_box`) only calls this
        once `self._infeed_position_mm` is known to be set.

        `set_prop_pose` returns before physics has moved the prop - the same
        problem `_settled_geometries` exists for - so this reads the box's
        own pose back rather than trusting the command took. On the GPU run
        of 2026-09-25 the read straight after `set_prop_pose` still showed
        the box parked for the whole settle window, seven packs out of ten,
        and the pick that followed drove the cup down over nothing. A miss
        is retried once with the same teleport; a second miss raises
        `RestageError` rather than picking against a pose the box was never
        at."""
        assert self._infeed_position_mm is not None
        target_mm = self._infeed_position_mm
        x, y, z = target_mm
        start = time.monotonic()
        attempts = 0
        last_pose_mm: Mapping[str, Any] | None = None
        while attempts < RESTAGE_MAX_ATTEMPTS:
            attempts += 1
            await self._world.do_command(
                {"command": "set_prop_pose", "name": box_prop, "position": [x, y, z]}
            )
            landed, last_pose_mm = await self._poll_until_restaged(box_prop, target_mm)
            if landed:
                LOGGER.info(
                    "%r restaged to %s in %d teleport(s), %.0f ms",
                    box_prop,
                    target_mm,
                    attempts,
                    (time.monotonic() - start) * 1000.0,
                )
                return
            if attempts < RESTAGE_MAX_ATTEMPTS:
                LOGGER.warning(
                    "%r sent to %s had not landed within %.0f mm after %.1f s (last read "
                    "%s); sending the teleport again",
                    box_prop,
                    target_mm,
                    RESTAGE_TOLERANCE_MM,
                    RESTAGE_TIMEOUT_S,
                    last_pose_mm,
                )
        raise RestageError(
            f"{box_prop!r} did not land within {RESTAGE_TOLERANCE_MM:.0f} mm of {target_mm} "
            f"after {attempts} attempts; last read {last_pose_mm}"
        )

    def _world_state_for(
        self,
        geometries: Sequence[Mapping[str, Any]],
        box_prop: str,
        keep_outs: Sequence[Geometry] = (),
        holding: Transform | None = None,
        reading: RelayReading | None = None,
    ) -> WorldState:
        """The planner's obstacles for one leg of one box's pick.

        `keep_outs` is the airspace this leg may not enter, on top of whatever
        the obstacle source already carries. The box being picked is never an
        obstacle in its own right: the cup's job is to reach it, and a plan
        that treats it as solid cannot. Keeping the arm off it on the legs that
        are not the descent is the keep-out's job instead. `holding` is the
        box on the cup, geometry that travels with the gripper, on the legs
        between the grab and the release. `reading` is this box's read of the
        configured world state stores, only consulted in ``world_state_store``
        mode: the frame system already carries the store's other fixtures as
        colliders, but a store's placed boxes reach the planner only through
        this relay."""
        transforms = [holding] if holding is not None else []
        if self._obstacle_source == _OBSTACLE_SOURCE_PROP_GEOMETRIES:
            obstacles = [
                *obstacles_from_prop_geometries(geometries, {box_prop}),
                *keep_outs,
            ]
            return world_state(None, obstacles, support_obstacle(_FLOOR_Z_MM), transforms)
        # world_state_store: the cell's own fixtures come from the frame
        # system, and the placed boxes from the relayed reading, so the
        # keep-outs, the relayed boxes and the held box are the only things
        # left to state. Neither carries props.
        relayed = list(reading.obstacles) if reading is not None else []
        obstacles = [*relayed, *keep_outs]
        if not obstacles and not transforms:
            return WorldState()
        return world_state(None, obstacles, None, transforms)

    async def _pack_all_boxes(self) -> None:
        """Runs the pack, appending each box's record to `self._records` as it
        finishes rather than returning them at the end.

        A run that raises used to lose every record it had already made, so
        `status` answered `failed` with an empty list and said nothing about
        how far it got. Appending live means the boxes that did complete are
        still reportable."""
        mover = self._build_mover()
        # a run that failed between grab and release left its box on the cup,
        # and a cup holding something can neither pick nor let a restage move
        # the prop it holds. Every run starts with an empty cup.
        await self._gripper.open()
        tracked_seq: int | None = None
        consecutive_failures = 0
        while not self._cancel_requested:
            next_box = await self._sequencer.next_box()
            if next_box.is_complete:
                break
            seq = next_box.seq
            assert seq is not None

            if seq != tracked_seq:
                tracked_seq = seq
                consecutive_failures = 0

            record, stopped = await self._place_one_box(mover, next_box)
            self._records.append(record)
            if stopped:
                break

            success = record.outcome == OUTCOME_PLACED
            await self._sequencer.report_placement(seq, success=success, error=record.reason or "")
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= MAX_ATTEMPTS_PER_SEQ:
                    await self._sequencer.skip_box(seq, reason=record.reason or "")
            if self._stop_after_seq is not None and record.seq >= self._stop_after_seq:
                self._stopped_after_seq = self._stop_after_seq
                break

    async def _place_one_box(
        self, mover: Mover, next_box: NextBox
    ) -> tuple[PalletizerRecord, bool]:
        """Runs one pick and place for ``next_box``. Returns the outcome
        record and whether a stop landed mid-motion - a stop is reported to
        neither the records' outcome tally nor the sequencer, since nothing
        was actually attempted for the sequencer to hear about."""
        start = time.monotonic()
        seq = next_box.seq
        assert seq is not None
        box_prop = self._box_props[seq - 1]
        place_start = next_box.place_start_in_world
        place_end = next_box.place_end_in_world
        assert place_start is not None
        assert place_end is not None
        target_pose_mm = _pose_to_dict(place_end)

        def failed_record(
            reason: str, planner_obstacles: Mapping[str, int] | None = None
        ) -> PalletizerRecord:
            return _outcome_record(
                box_prop,
                seq,
                target_pose_mm,
                OUTCOME_FAILED,
                time.monotonic() - start,
                reason=reason,
                planner_obstacles=planner_obstacles,
            )

        def stopped_record(planner_obstacles: Mapping[str, int] | None = None) -> PalletizerRecord:
            return failed_record("stopped before the next motion", planner_obstacles)

        # a stop that lands after the previous box and before this one must not
        # put this box on the station. The restage is a teleport, not a motion,
        # so the between-motions check below it comes too late: on 2026-09-22
        # a run stopped after its first record had already restaged box 2 onto
        # the pick spot, and every later reset of box 1 landed on top of it.
        if self._cancel_requested:
            return stopped_record(), True
        if seq > 1:
            if self._infeed_position_mm is None:
                # seq 1's own read never captured an infeed pose to restage
                # onto - a run started at seq > 1 without one first (a part
                # restart moved the sequencer's cursor on without a fresh
                # box_props[0] ever being read) has nothing to send this box
                # to, and restaging nothing would pick it wherever it happens
                # to be parked.
                return (
                    failed_record(
                        "no infeed pose captured yet: a run must start at seq 1 after a "
                        "part restart"
                    ),
                    False,
                )
            try:
                await self._restage_box(box_prop)
            except RestageError as error:
                return failed_record(str(error)), False

        geometries, _pick_settle_s = await self._settled_geometries(box_prop)
        reading: RelayReading | None = None
        if self._obstacle_source == _OBSTACLE_SOURCE_WORLD_STATE_STORE:
            reading = await read_store_obstacles(self._stores)
        if seq == 1 and self._infeed_position_mm is None:
            infeed_pose_mm = _box_pose_mm(geometries, box_prop)
            if infeed_pose_mm is not None:
                self._infeed_position_mm = (
                    float(infeed_pose_mm["x"]),
                    float(infeed_pose_mm["y"]),
                    float(infeed_pose_mm["z"]),
                )

        top_face_xyz_mm = _box_top_face_xyz_mm(geometries, box_prop)
        if top_face_xyz_mm is None:
            return failed_record(f"{box_prop!r} has no known geometry in the world"), False

        # the two airspaces the arm stays out of except while working in them.
        # Neither box nor pallet can be an obstacle in its own right, since the
        # cup has to reach both, so a keep-out stands in their place on every
        # leg that is not the descent onto them.
        pick_zone = box_keepout(geometries, box_prop)
        place_zone = support_keepout(geometries, self._place_support_prop, place_end.z)
        # the same airspace raised by PLACE_APPROACH_CLEARANCE_MM, for the two
        # legs the arm runs with nothing on the cups and the stack nearby. On
        # the GPU run of 2026-10-01 both hits on the second layer began inside
        # the next box's free approach to the pick standoff, the arm empty,
        # 110 mm above the box it had just released, that box present as an
        # obstacle and under the place keep-out, both reaching exactly to its
        # top face: the planner had no margin and the cups swept the box. The
        # legs that carry a box cannot use it. With the box on the cups its
        # bottom rides PLACE_APPROACH_CLEARANCE_MM + 10 mm above the layer top
        # (the sequencer's standoff is box height plus 10, the carry adds the
        # approach clearance), so a keep-out raised by the clearance would
        # leave the held box 10 mm clear and the planner refusing.
        place_zone_empty = support_keepout(
            geometries,
            self._place_support_prop,
            place_end.z + PLACE_APPROACH_CLEARANCE_MM,
            label="place_support_keepout_empty",
        )
        keepout_count = sum(1 for zone in (pick_zone, place_zone) if zone is not None)
        prop_count = (
            len(obstacles_from_prop_geometries(geometries, {box_prop}))
            if self._obstacle_source == _OBSTACLE_SOURCE_PROP_GEOMETRIES
            else 0
        )
        planner_obstacles: Mapping[str, int] = {
            "relayed": reading.relayed if reading is not None else 0,
            "skipped_not_in_world": reading.skipped_not_in_world if reading is not None else 0,
            "skipped_no_geometry": reading.skipped_no_geometry if reading is not None else 0,
            "keepouts": keepout_count,
            "props": prop_count,
        }
        LOGGER.info(
            "palletizer planner obstacles for seq %d: %d relayed from %s, %d skipped "
            "(parent not world), %d keep-outs, %d props",
            seq,
            planner_obstacles["relayed"],
            ", ".join(name for name, _store in self._stores),
            planner_obstacles["skipped_not_in_world"],
            planner_obstacles["keepouts"],
            planner_obstacles["props"],
        )

        # the box on the cup, for the legs between the grab and the release.
        # The planner knows the arm and nothing hanging from it unless told.
        box_in_scene = _prop_box_mm(geometries, box_prop)
        held_box = (
            held_box_transform(box_prop, box_in_scene[1], self._gripper_name)
            if box_in_scene is not None
            else None
        )

        def mover_avoiding(
            *keep_outs: Geometry | None, holding: Transform | None = None
        ) -> Callable[[Pose, bool], Awaitable[bool]]:
            """A move that stays out of `keep_outs`, carrying `holding` on the
            cup, and that reports a stop landing since the previous motion
            rather than making the move.

            A move with something on the cup is level as well: its free paths
            keep the tool pointing the way it started. The planner has been
            told about the box, and on 2026-09-22 it still joined two
            pointing-down poses whose wrist solutions differed by 180 degrees
            with one segment that turned the box over the top of the arm and
            into the forearm. Forbidding the turn is what carrying means."""
            state = self._world_state_for(
                geometries,
                box_prop,
                [zone for zone in keep_outs if zone is not None],
                holding,
                reading,
            )
            level = holding is not None

            async def move(pose: Pose, linear: bool) -> bool:
                if self._cancel_requested:
                    return False
                try:
                    await mover.move_to(
                        pose,
                        state,
                        linear=linear,
                        level=level,
                        plan_timeout_s=None if linear else FIRST_FREE_PLAN_TIMEOUT_S,
                    )
                except Exception as error:
                    if linear or not is_planner_timeout(error):
                        raise
                    LOGGER.warning(
                        "free move for seq %d to %s timed out in the planner (%s), "
                        "planning once more with seed %d",
                        seq,
                        _pose_to_dict(pose),
                        error,
                        RETRY_PLANNER_SEED,
                    )
                    if self._cancel_requested:
                        return False
                    await mover.move_to(
                        pose, state, linear=linear, level=level, planner_seed=RETRY_PLANNER_SEED
                    )
                return True

            return move

        # the pick zone is open only while descending onto and lifting off the
        # box, the place zone only while descending onto and retreating from
        # the pallet. Every other leg has both closed. The box rides the cup
        # from the lift to the place descent. The place zone is the raised one
        # on the two empty-armed legs near the stack, the approach to the pick
        # standoff and the fallback retreat, and flush with the layer's top
        # on every other leg.
        cross_cell = mover_avoiding(pick_zone, place_zone_empty)
        over_the_pick = mover_avoiding(place_zone)
        over_the_place = mover_avoiding(pick_zone, place_zone_empty)
        lift_holding = mover_avoiding(place_zone, holding=held_box)
        cross_cell_holding = mover_avoiding(pick_zone, place_zone, holding=held_box)
        place_holding = mover_avoiding(pick_zone, holding=held_box)

        async with _leg_timeline(seq, "approach to the pick standoff"):
            approached = await cross_cell(pick_grasp_standoff_pose(top_face_xyz_mm), False)
        if not approached:
            return stopped_record(planner_obstacles), True

        async with _leg_timeline(seq, "descent onto the box"):
            descended = await move_linear_or_free(
                over_the_pick, pick_grasp_pose(top_face_xyz_mm), "grasp descent"
            )
        if not descended:
            return stopped_record(planner_obstacles), True

        if self._cancel_requested:
            return stopped_record(planner_obstacles), True
        async with _leg_timeline(seq, "grab"):
            grabbed = await self._gripper.grab()
        if not grabbed:
            # with the poses: a cup that found nothing is either in the wrong
            # place or over a box that moved, and the two read identically
            # without them
            return failed_record(
                "gripper reported no object grasped, cup sent to "
                f"{_pose_to_dict(pick_grasp_pose(top_face_xyz_mm))} for a box whose top face "
                f"read {top_face_xyz_mm}",
                planner_obstacles,
            ), False

        # the lift runs straight up to the height the carry runs at, the
        # approach's, so the box hangs above the place keep-out before the
        # sweep begins. On 2026-09-24 a carry from the standoff, box bottom
        # 371 mm up, had to cross the keep-out's airspace, and from one base
        # branch the planner found no way round it in five minutes. One
        # straight leg rather than a lift to the standoff and a climb: the
        # standoff is the pick's own pose and means nothing to the carry, and
        # the extra stop cost 10 s a box on the 17:50 run
        approach = place_approach_pose(place_start)
        async with _leg_timeline(seq, "lift to carry height"):
            lifted = await move_linear_or_free(
                lift_holding, carry_height_pose(top_face_xyz_mm, approach.z), "lift to carry height"
            )
        if not lifted:
            return stopped_record(planner_obstacles), True
        # the carry ends above the sequencer's standoff and the box settles
        # there; the straight leg down to place_start is what brings it over
        # its neighbours, and by then it hangs still
        async with _leg_timeline(seq, "carry to the place approach"):
            carried = await cross_cell_holding(approach, False)
        if not carried:
            return stopped_record(planner_obstacles), True

        # the box swings on the cups through the carry - PLACE_APPROACH_CLEARANCE_MM's
        # own docstring measures it at 7.5 to 10 degrees - and a straight descent onto
        # a same-layer neighbour started before that swing stops is what knocked a box
        # 524 mm off the pallet on the GPU run of 2026-09-28.
        swing_s = await self._wait_for_the_carried_box_to_stop_swinging(seq, box_prop)

        async with _leg_timeline(seq, "settle over the slot"):
            settled_over_the_slot = await move_linear_or_free(
                place_holding, place_start, "settle over the slot"
            )
        if not settled_over_the_slot:
            return stopped_record(planner_obstacles), True
        async with _leg_timeline(seq, "place descent"):
            descended_to_release = await self._descend_to_release(
                place_holding, place_release_pose(place_end)
            )
        if not descended_to_release:
            return stopped_record(planner_obstacles), True

        if self._cancel_requested:
            return stopped_record(planner_obstacles), True
        async with _leg_timeline(seq, "release"):
            await self._gripper.open()
        released_at = time.monotonic()
        sampler = asyncio.create_task(self._settle_after_release(box_prop, released_at))
        try:
            async with _leg_timeline(seq, "retreat off the place"):
                # the box just went down where the pick keep-out used to be
                # empty air - the retreat keeps out of it too, or the straight
                # line off the slot clips the box it just placed. Built from a
                # fresh read rather than the stale `geometries` above, since
                # that reading predates the box ever landing there.
                released_geometries = await self._prop_geometries()
                placed_zone = placed_box_keepout(released_geometries, box_prop)
                retreat_off_the_place = mover_avoiding(pick_zone, placed_zone)
                try:
                    await move_linear_or_free(
                        retreat_off_the_place, place_start, "retreat off the place"
                    )
                except Exception as error:
                    if is_execution_failure(error):
                        raise
                    LOGGER.warning(
                        "retreat off the place with %r as an obstacle was refused (%s); "
                        "retreating without it, which can clip that box",
                        box_prop,
                        error,
                    )
                    await move_linear_or_free(over_the_place, place_start, "retreat off the place")
        except BaseException:
            sampler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler
            raise

        # the sampler has been polling since the release, through the retreat
        # that follows it, so settle_s measures from the release rather than
        # from whenever the retreat happens to finish
        settled_geometries, settle_s = await sampler
        measured_pose = _measured_box_pose(settled_geometries, box_prop)
        measured_pose_mm = _pose_to_dict(measured_pose) if measured_pose is not None else None
        # the store centres its own geometry on the pose it is given, so the
        # transform published to it is the box's CENTRE, not the top face
        # measured_pose_mm keeps for the record
        centre_pose_mm = _box_pose_mm(settled_geometries, box_prop)
        if centre_pose_mm is not None:
            await self._sequencer.set_box_transform(seq, _pose_from_mm(centre_pose_mm))

        return (
            _outcome_record(
                box_prop,
                seq,
                target_pose_mm,
                OUTCOME_PLACED,
                time.monotonic() - start,
                measured_pose_mm=measured_pose_mm,
                planner_obstacles=planner_obstacles,
                settle_s=settle_s,
                swing_s=swing_s,
            ),
            False,
        )
