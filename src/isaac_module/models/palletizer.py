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
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

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
BOX_SETTLE_TOLERANCE_MM = 1.0
BOX_SETTLE_POLL_S = 0.05
BOX_SETTLE_TIMEOUT_S = 5.0


def box_has_settled(
    before_mm: Mapping[str, Any] | None,
    after_mm: Mapping[str, Any] | None,
    tolerance_mm: float = BOX_SETTLE_TOLERANCE_MM,
) -> bool:
    """Whether two consecutive readings of a box put it in the same place.

    A box that has not registered a pose at all has not settled: reaching for
    one that is not there is the failure this guards, not a special case of
    it."""
    if before_mm is None or after_mm is None:
        return False
    return (
        sum((float(after_mm[axis]) - float(before_mm[axis])) ** 2 for axis in ("x", "y", "z"))
        ** 0.5
    ) <= tolerance_mm


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


def _measured_box_pose(geometries: Sequence[Mapping[str, Any]], box_prop: str) -> Pose | None:
    """``box_prop``'s pose as physics settled it, read straight off
    ``prop_geometries`` rather than where the plan intended it - the whole
    point of feeding it back to ``set_box_transform``."""
    pose_mm = _box_pose_mm(geometries, box_prop)
    if pose_mm is None:
        return None
    return Pose(
        x=float(pose_mm["x"]),
        y=float(pose_mm["y"]),
        z=float(pose_mm["z"]),
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

# how far a keep-out's ceiling stays below the pose the arm approaches it from.
# Same reasoning as PICK_KEEPOUT_HEADROOM_MM, from the other side: a place
# keep-out that reaches the approach pose makes the approach unplannable.
PLACE_KEEPOUT_HEADROOM_MM = 20.0


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
        label="pick_box_keepout",
    )


def support_keepout(
    geometries: Sequence[Mapping[str, Any]], support_prop: str, ceiling_z_mm: float
) -> Geometry | None:
    """The airspace over the place support, which the arm keeps out of until
    it is descending onto it.

    Covers the whole support rather than one slot, since what it protects is
    the boxes already stacked on it, and where those are is the sequencer's
    business rather than this service's. Returns None when the support is not
    in the scene, or when `ceiling_z_mm` leaves no room above its top face."""
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
        label="place_support_keepout",
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
        return result


def _outcome_record(
    box_prop: str,
    seq: int,
    target_pose_mm: Mapping[str, float],
    outcome: str,
    duration_s: float,
    reason: str | None = None,
    measured_pose_mm: Mapping[str, float] | None = None,
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


class IsaacPalletizer(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the service, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "palletizer")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._state: str = STATE_IDLE
        self._task: asyncio.Task[None] | None = None
        self._cancel_requested = False
        self._records: list[PalletizerRecord] = []
        self._failure_reason: str | None = None
        # box_props[0]'s pose on the first prop_geometries read of a run -
        # every later box_props[i] is re-posed there before its own pick.
        # Reset per run (_handle_start), not per box.
        self._infeed_position_mm: tuple[float, float, float] | None = None

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

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Mapping[str, ValueTypes]:
        cmd = str(command.get("command", ""))
        if cmd == "start":
            return self._handle_start()
        if cmd == "stop":
            return self._handle_stop()
        if cmd == "status":
            return cast("Mapping[str, ValueTypes]", self._status_snapshot())
        raise ValueError(f"unknown command {cmd!r}; supported: start, stop, status")

    def _handle_start(self) -> dict[str, ValueTypes]:
        if self._state in (STATE_RUNNING, STATE_STOPPING):
            return {"ok": False, "state": self._state}
        self._state = STATE_RUNNING
        self._records = []
        self._failure_reason = None
        self._cancel_requested = False
        self._infeed_position_mm = None
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

    async def _settled_geometries(self, box_prop: str) -> Sequence[Mapping[str, Any]]:
        """`prop_geometries`, read once `box_prop` has stopped moving.

        The grasp pose is derived from this reading, and a cup sent to where a
        falling box was closes on air. Returns the last reading either way, so
        a box that never settles is picked against its most recent pose and
        the attempt fails on its own terms rather than here."""
        geometries = await self._prop_geometries()
        previous = _box_pose_mm(geometries, box_prop)
        if previous is None:
            # a box that is not in the world cannot settle. Waiting the full
            # timeout for one only delays the "no known geometry" the caller
            # is about to report.
            return geometries
        deadline = time.monotonic() + BOX_SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(BOX_SETTLE_POLL_S)
            geometries = await self._prop_geometries()
            current = _box_pose_mm(geometries, box_prop)
            if box_has_settled(previous, current):
                return geometries
            previous = current
        LOGGER.warning(
            "%r was still moving after %.1f s; picking against its last reading",
            box_prop,
            BOX_SETTLE_TIMEOUT_S,
        )
        return geometries

    async def _restage_box(self, box_prop: str) -> None:
        """Re-poses ``box_prop`` onto the pick station's infeed pose, so a
        later box_props entry arrives where box_props[0] started rather than
        wherever it was parked. A no-op when the infeed pose was never
        captured (box_props[0] was missing from the world on seq 1's read),
        so a bad first read fails the pick that reads it rather than
        crashing every later one."""
        if self._infeed_position_mm is None:
            return
        x, y, z = self._infeed_position_mm
        await self._world.do_command(
            {"command": "set_prop_pose", "name": box_prop, "position": [x, y, z]}
        )

    def _world_state_for(
        self,
        geometries: Sequence[Mapping[str, Any]],
        box_prop: str,
        keep_outs: Sequence[Geometry] = (),
        holding: Transform | None = None,
    ) -> WorldState:
        """The planner's obstacles for one leg of one box's pick.

        `keep_outs` is the airspace this leg may not enter, on top of whatever
        the obstacle source already carries. The box being picked is never an
        obstacle in its own right: the cup's job is to reach it, and a plan
        that treats it as solid cannot. Keeping the arm off it on the legs that
        are not the descent is the keep-out's job instead. `holding` is the
        box on the cup, geometry that travels with the gripper, on the legs
        between the grab and the release."""
        transforms = [holding] if holding is not None else []
        if self._obstacle_source == _OBSTACLE_SOURCE_PROP_GEOMETRIES:
            obstacles = [
                *obstacles_from_prop_geometries(geometries, {box_prop}),
                *keep_outs,
            ]
            return world_state(None, obstacles, support_obstacle(_FLOOR_Z_MM), transforms)
        # world_state_store: the cell's own obstacles come from the frame
        # system and the store the motion service already consults, so the
        # keep-outs and the held box are the only things left to state.
        # Neither carries props.
        if not keep_outs and not transforms:
            return WorldState()
        return world_state(None, list(keep_outs), None, transforms)

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
            if success:
                continue
            consecutive_failures += 1
            if consecutive_failures >= MAX_ATTEMPTS_PER_SEQ:
                await self._sequencer.skip_box(seq, reason=record.reason or "")

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

        def failed_record(reason: str) -> PalletizerRecord:
            return _outcome_record(
                box_prop,
                seq,
                target_pose_mm,
                OUTCOME_FAILED,
                time.monotonic() - start,
                reason=reason,
            )

        def stopped_record() -> PalletizerRecord:
            return failed_record("stopped before the next motion")

        # a stop that lands after the previous box and before this one must not
        # put this box on the station. The restage is a teleport, not a motion,
        # so the between-motions check below it comes too late: on 2026-09-22
        # a run stopped after its first record had already restaged box 2 onto
        # the pick spot, and every later reset of box 1 landed on top of it.
        if self._cancel_requested:
            return stopped_record(), True
        if seq > 1:
            await self._restage_box(box_prop)

        geometries = await self._settled_geometries(box_prop)
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
        place_zone = support_keepout(
            geometries,
            self._place_support_prop,
            min(place_start.z, place_end.z) - PLACE_KEEPOUT_HEADROOM_MM,
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
                geometries, box_prop, [zone for zone in keep_outs if zone is not None], holding
            )
            level = holding is not None

            async def move(pose: Pose, linear: bool) -> bool:
                if self._cancel_requested:
                    return False
                await mover.move_to(pose, state, linear=linear, level=level)
                return True

            return move

        # the pick zone is open only while descending onto and lifting off the
        # box, the place zone only while descending onto and retreating from
        # the pallet. Every other leg has both closed. The box rides the cup
        # from the lift to the place descent.
        cross_cell = mover_avoiding(pick_zone, place_zone)
        over_the_pick = mover_avoiding(place_zone)
        over_the_place = mover_avoiding(pick_zone)
        lift_holding = mover_avoiding(place_zone, holding=held_box)
        cross_cell_holding = mover_avoiding(pick_zone, place_zone, holding=held_box)
        place_holding = mover_avoiding(pick_zone, holding=held_box)

        if not await approach_and_descend(
            cross_cell,
            over_the_pick,
            pick_grasp_standoff_pose(top_face_xyz_mm),
            pick_grasp_pose(top_face_xyz_mm),
        ):
            return stopped_record(), True

        if self._cancel_requested:
            return stopped_record(), True
        if not await self._gripper.grab():
            # with the poses: a cup that found nothing is either in the wrong
            # place or over a box that moved, and the two read identically
            # without them
            return failed_record(
                "gripper reported no object grasped, cup sent to "
                f"{_pose_to_dict(pick_grasp_pose(top_face_xyz_mm))} for a box whose top face "
                f"read {top_face_xyz_mm}"
            ), False

        if not await move_linear_or_free(
            lift_holding, pick_grasp_standoff_pose(top_face_xyz_mm), "lift off the pick"
        ):
            return stopped_record(), True
        if not await cross_cell_holding(place_start, False):
            return stopped_record(), True
        if not await self._descend_to_release(place_holding, place_release_pose(place_end)):
            return stopped_record(), True

        if self._cancel_requested:
            return stopped_record(), True
        await self._gripper.open()
        await move_linear_or_free(over_the_place, place_start, "retreat off the place")

        # the box settles onto its slot after the release, so the pose
        # reported back is read once it has stopped
        settled_geometries = await self._settled_geometries(box_prop)
        measured_pose = _measured_box_pose(settled_geometries, box_prop)
        measured_pose_mm = _pose_to_dict(measured_pose) if measured_pose is not None else None
        if measured_pose is not None:
            await self._sequencer.set_box_transform(seq, measured_pose)

        return (
            _outcome_record(
                box_prop,
                seq,
                target_pose_mm,
                OUTCOME_PLACED,
                time.monotonic() - start,
                measured_pose_mm=measured_pose_mm,
            ),
            False,
        )
