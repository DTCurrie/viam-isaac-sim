"""GPU acceptance checklist for the palletizing cell. Four suites, each
testing one stage this cell was built up in: first-box (vacuum gripper,
one box picked and placed by the box-palletizer service), workcell (the
vendored `viam:workcell-components` workcell adopted under the same
mechanism), epick (the Robotiq EPick adopted as the cell's gripper, driven
directly rather than through box-palletizer), and pack (the full
eight-box pack, planned by `viam:pack-sequencer:sequencer` and executed
by `box-palletizer`). Pass `--suite first-box` (default), `--suite
workcell`, `--suite epick` or `--suite pack`.

Connects to a running Viam machine (the module running on the Isaac GPU box):
`arm-1`, `gripper-1`, `builtin` motion, the `box-palletizer` service, and
(the pack suite only) the `pack-sequencer` service. Item 3 is the
first-box suite's own done-when, so it drives the actual pick and place
through `box-palletizer`'s DoCommand rather than re-driving the arm and
gripper itself. Items 1, 2 and 4 drive the arm and gripper directly to test
the underlying grab mechanism, reading the box's pose from the world at run
time rather than the service's own arithmetic, so a green result there
can't just be agreement with the code under test. The pack suite drives
the whole pack through `box-palletizer`'s own `start`/`status` surface
throughout, since the sequencer, not this script, owns every target pose.

For the first-box and workcell suites, the box's known resting pose
before a run and the target place position are facts about the deployed
cell, so they come in as CLI arguments (`--pick-*-mm`, `--place-*-mm`)
rather than an invented constant in this file. The pack suite needs
neither: `box-palletizer` sources the infeed pose from the sim itself and
the sequencer owns every place target, so its checklist items read poses
off the service's own status records instead.

Prints PASS/FAIL and the raw numbers for each item.

First-box suite checklist items:

1. the vacuum tool renders on the wrist and the arm reaches the box
2. `grab` attaches, the box rides the tool through a carry, and release drops it
3. the box lands within tolerance of the place position and is still there after 5 s
4. a grab attempted with nothing under the tool reports not holding rather
   than hanging
5. cost: cold and warm `ready` and the 10 s step rate

Workcell suite checklist items, the cell having been adopted from Viam's own
palletizer workcell:

1. every component in the fragment renders at the pose its frame declares,
   compared against the component's own group frame primitive
2. the pick station and pallet stop a dropped box at the height their own
   geometry implies, and the pedestal's collider matches what it declares
3. the arm cannot plan through the fences or the scan tunnel, shown by a
   plan that detours rather than intersecting
4. the single-box pick and place runs end to end in the workcell
5. whether `scan-tunnel` needs a derived collider the way `robot-pedestal`
   does, answered by driving the arm at it
6. cost against the first-box suite's recorded run: cold and warm `ready`
   and the 10 s step rate, since this cell has far more geometry
7. implementation risk, not a plan requirement: render-only scenery has no
   collider proven on hardware

EPick suite checklist items, the cell's tool having been adopted from the
Robotiq EPick vacuum gripper. Items 2 through 6 drive the arm and gripper
directly, the same reasoning as the first-box suite's items 1, 2 and 4:
reading the box's pose from the world at run time rather than a service's
own arithmetic. `--grab-delay-ms` and `--retry-interval-s` bound item 2's
refusal timing; both default to the deployed overlay's own values.

0. smoke: one attachment point on the tool body grips a box prop, then four
   on the cup pattern
1. the EPick renders on the wrist at the module's dimensions and
   get_kinematics returns the module's own epick_model.json, with the 26 mm
   approach gap free of colliders
2. gating: a grab with the cups 5 mm over the box holds, one 40 mm over it
   does not, and the refusal arrives within grab_delay_ms plus the retry
   interval
3. swing: during the level cross-cell carry the box's tilt relative to the
   tool is nonzero and bounded, the grip holds, max and residual tilt printed
4. tear-off: a load past the cups' holding force drops on lift, one under it
   holds, and the 2 kg box holds
5. the place descent in both arm configurations by the deterministic
   reproduction, place error per configuration
6. release at contact: descend until the box stops the arm on the deck,
   open, landing error against the 25 mm drop
7. cost against the workcell suite's recorded run: cold and warm ready and
   the 10 s step rate

Depends only on the stdlib and viam-sdk: it runs on a laptop against a remote
machine, not inside the module process.

Usage::

    python examples/gpu_checklist_palletizer.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id> --box-prop <prop-name> \\
        --pick-x-mm <x> --pick-y-mm <y> --pick-z-mm <z> \\
        --place-x-mm <x> --place-y-mm <y> --run-label cold

Prints one heading plus the raw observations per item, then a summary table.
The pure helpers at the top take plain mappings so they are unit-tested on a
laptop (see tests/test_gpu_checklist_palletizer.py). Item 5's cost numbers
are only worth reading paired with `--run-label`: run the script once right
after a module restart with `--run-label cold`, then again without
restarting with `--run-label warm`, and compare the two printed rows.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# python examples/gpu_checklist_palletizer.py (standalone, no PYTHONPATH set) needs the
# repo's src/ on sys.path before isaac_module is importable. pytest already adds
# it (pyproject pythonpath = ["src"]), so this is a no-op there.
try:
    import isaac_module  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu_checklist_world import WorldApi, prop_pose_mm, run_step_rate_measurement

from isaac_module.asset_catalog import EPICK, EPICK_PRIM
from isaac_module.models.palletizer import (
    CUP_APPROACH_GAP_MM,
    PLACE_RELEASE_CLEARANCE_MM,
    PLACE_STALL_TOLERANCE_MM,
    approach_and_descend,
    held_box_transform,
    is_execution_failure,
    move_linear_or_free,
    pick_grasp_pose,
    pick_grasp_standoff_pose,
    place_release_pose,
    touched_down,
)
from isaac_module.prim_paths import prim_name
from isaac_module.sequencer_client import SequencerClient
from isaac_module.sort_plan import OUTCOME_PLACED
from isaac_module.spatial import (
    Quat,
    Vec3,
    compose_pose,
    ov_to_quat,
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_to_ov,
)
from isaac_module.surface_gripper import (
    COAXIAL_LOAD_WINDOW_S,
    DEFAULT_COAXIAL_FORCE_LIMIT_N,
    STANDARD_GRAVITY_M_S2,
)
from isaac_module.workcell_scenery import parse_visuals

MM_PER_M = 1000.0

FIRST_BOX_ITEMS: tuple[str, ...] = (
    "1. the vacuum tool renders on the wrist and the arm reaches the box",
    "2. `grab` attaches, the box rides the tool through a carry, and release drops it",
    "3. the box lands within tolerance of the place position and is still there after 5 s",
    "4. a grab attempted with nothing under the tool reports not holding rather than hanging",
    "5. cost: cold and warm `ready` and the 10 s step rate",
)

# the workcell suite's checklist items. Item 7 is not one of them: it verifies an
# implementation risk
# (the collision:False strip sim_manager.py applies to a render-only prop is proven in mock only,
# never on hardware, because omni and pxr are not importable in this environment), not a plan
# requirement, and is labelled as such in its own text.
WORKCELL_ITEMS: tuple[str, ...] = (
    "1. every component in the fragment renders at the pose its frame declares: its own group "
    "frame primitive against the fragment, and every box and capsule it draws against the prim "
    "the sim spawned for it",
    "2. the pick station and pallet stop a dropped box at the height the component's "
    "geometry implies, and every declared `frame.geometry` collider spawns at the pose, "
    "orientation and size it declares",
    "3. the arm cannot plan through the fences or the scan tunnel, shown by a plan that "
    "detours rather than intersecting",
    "4. the single-box pick and place runs end to end in the workcell",
    "5. whether `scan-tunnel` needs a derived collider the way `robot-pedestal` does, "
    "answered by driving the arm at it",
    "6. cost against the first-box suite's recorded run: cold and warm `ready` and the 10 s "
    "step rate, since this cell has far more geometry",
    "7. [implementation risk, not a plan requirement] render-only scenery has no collider "
    "proven on hardware: drop a box onto `caution-tape` and show it passes through to the floor",
)

PACK_ITEMS: tuple[str, ...] = (
    "1. eight boxes placed, the sequencer's target and the measured pose per box, as a table",
    "2. the order placed matches `get_pack_order`",
    "3. obstacles come from the world state store rather than this service's own hand-built "
    "WorldState, shown by running it both ways and reporting whether plans stay collision-free "
    "under the store",
    "4. `set_box_transform` moves the box in the viewer to where physics put it, with the "
    "largest plan-versus-actual difference recorded across the eight",
    "5. a box the arm fails to place is `skip_box`ed and the run continues",
    "6. cost against the workcell suite's recorded run: cold and warm `ready` and the 10 s "
    "step rate",
)

# the epick suite's checklist items: the Robotiq EPick vacuum gripper adopted as
# the cell's tool. Item 0 is cited rather than re-run (see its own check),
# and every other item drives the arm and gripper directly rather than
# through box-palletizer, the same reasoning the first-box suite's items 1, 2 and 4 use.
EPICK_ITEMS: tuple[str, ...] = (
    "0. smoke: one attachment point on the tool body grips a box prop, then four on the cup "
    "pattern",
    "1. the EPick renders on the wrist at the module's dimensions and get_kinematics returns "
    "the module's own epick_model.json, with the 26 mm approach gap free of colliders",
    "2. gating: a grab with the cups 5 mm over the box holds, one 40 mm over it does not, and "
    "the refusal arrives within grab_delay_ms plus the retry interval",
    "3. swing: during the level cross-cell carry the box's tilt relative to the tool is "
    "nonzero and bounded, the grip holds, max and residual tilt printed",
    "4. tear-off: a load past the cups' holding force drops on lift, one under it holds, "
    "and the 2 kg box holds",
    "5. the place descent in both arm configurations by the deterministic reproduction, place "
    "error per configuration",
    "6. release at contact: descend until the box stops the arm on the deck, open, landing "
    "error against the 25 mm drop",
    "7. cost against the workcell suite's recorded run: cold and warm ready and the 10 s step rate",
)

# item 1's tolerance between a fragment component's declared world-frame position and its
# own `get_pose` reply: same translation budget as REACH_TOLERANCE_MM, a prim-pose/viam-pose
# comparison
FRAME_POSE_TOLERANCE_MM = 1.0

# items 1 and 2's tolerance between a declared orientation and a spawned one. Both sides are
# authored from the same quaternion, so anything past float noise is a composition error, and
# the smallest one this cell can produce is a fence turned 90 degrees
FRAME_ANGLE_TOLERANCE_DEG = 0.5

# item 2's and item 7's tolerance between a dropped box's expected resting height (derived from
# the support's own geometry, never a constant) and its measured resting height. Carried from
# infeed_box's contact_offset in examples/configs/sim-palletizer-cell.json (5 mm), the same
# settle slack the box's own contact physics allows
RESTING_HEIGHT_TOLERANCE_MM = 5.0

# item 2's and item 7's clearance above a support's derived top face before the box free-falls,
# and how long to wait for it to settle. Shorter than item 3 (first-box suite)'s SETTLE_WINDOW_S
# since a freshly dropped box has nowhere to drift once it has landed
# item 4's grab-with-nothing-under-the-tool needs a spot with, in fact,
# nothing under the tool. How far from every prop that spot has to be, and the
# offsets along the pick station it tries in order.
#
# The test used to aim at the box's own pose, on the assumption that the
# service run before it had carried the box away. When that run fails the box
# is still there, the cup descends onto it and grabs it, and `grab()=True
# holding=True` is the right answer to the wrong question.
GRAB_NOTHING_CLEARANCE_MM = 300.0
GRAB_NOTHING_OFFSETS_MM = ((0.0, -400.0), (0.0, 400.0), (0.0, -700.0), (0.0, 700.0))

DROP_HOVER_MM = 200.0
DROP_SETTLE_S = 2.0

# how long a teleported box is given to settle before its pose is read back, and how far the
# settled pose may sit from the one asked for. `set_prop_pose` returns before physics has moved
# the prop, and item 4 derives its reach target from the box's LIVE pose, so a read taken too
# early aims the arm at wherever the box was last left rather than at the pick pose. After item
# 2's drops, where it was last left is inside the cell's own furniture.
RESET_SETTLE_S = 1.0

# the workcell suite's item 6 cites this rather than re-measuring, since the first-box
# suite's own cell was deleted before that run: `sim_time_ratio` 0.563 over a 10 s window,
# from the first-box suite's own recorded GPU run of 2026-09-15 on isaac-sim-devin-2. The
# first-box suite's own ready_time_s reads 0.0 both cold and warm, because the checklist
# connects after the finalizer has already signalled ready, which is a measurement gap
# rather than a result, so it carries no comparable baseline of its own. The workcell
# suite's own ready_time_s is printed as a first reading instead, to become the pack
# suite's baseline
FIRST_BOX_BASELINE_SIM_TIME_RATIO = 0.563

# item 1's tolerance between the arm's reported end position and the pick grasp pose:
# same translation budget gpu_checklist_arm.py uses for a prim-pose/viam-pose comparison
REACH_TOLERANCE_MM = 1.0

# item 2's expected vertical travel of a lift back to the grasp standoff, and how far
# off that (plus how far off dead-vertical) still counts as "rode with the tool" rather
# than "float32 pose noise" or "left behind". A box left behind reads ~0 mm vertical here.
LIFT_DISTANCE_MM = 100.0  # pickcell.poses.PRE_GRASP_STANDOFF_MM, the standoff the lift retraces
LIFT_VERTICAL_TOLERANCE_MM = 5.0
LIFT_HORIZONTAL_TOLERANCE_MM = 5.0

# item 3's tolerance between the released box's resting pose and the place target, in the
# horizontal plane only (the resting height depends on packing details this checklist does
# not own), and between two readings taken SETTLE_WINDOW_S apart (still there after 5 s)
PLACEMENT_TOLERANCE_MM = 10.0
DRIFT_TOLERANCE_MM = 5.0
SETTLE_WINDOW_S = 5.0

# the floor the planner must not swing a link through
FLOOR_Z_MM = 0.0

# item 4's bounded wait: a grab that never resolves is a failure, not a hang
GRAB_NOTHING_TIMEOUT_S = 15.0

# where the arm waits while a box is teleported onto the station or the
# pallet: pointing down, high, over the empty floor between the pedestal and
# the pallet, with no link of the arm above either surface from there. The
# previous test leaves the arm wherever it ended, and a drop test should not
# depend on that.
PARK_XYZ_MM = (-300.0, 200.0, 700.0)

# a prop the service's run moved by more than this is put back afterwards
RESTORE_TOLERANCE_MM = 1.0

# item 5, same shape as gpu_checklist_photoreal.py's cost item
DEFAULT_STEP_RATE_WINDOW_S = 10.0
DEFAULT_READY_POLL_S = 0.5
DEFAULT_READY_TIMEOUT_S = 600.0

# item 3's bounded wait on the box-palletizer service's own run: a start that never
# leaves "running" is a failure, not a hang. One pick and place is a handful of
# straight-line motion-service moves, so two minutes is generous
STATUS_POLL_S = 0.5
STATUS_TIMEOUT_S = 120.0

# the pack suite's bounded wait on a full eight-box pack: eight picks and places, so eight
# times the first-box suite's single-box budget
PACK_STATUS_TIMEOUT_S = STATUS_TIMEOUT_S * 8

# item 2's slack on top of the deterministic grab delay and the retry window: the grab
# loop's own scheduling jitter, not a second retry
GRAB_REFUSAL_SLACK_S = 1.0

# item 2's miss height: far enough over the top face that no cup reaches it, reproducing
# a grab over nothing rather than a marginal one. CUP_APPROACH_GAP_MM (5 mm) is the holding
# case
GRAB_GATING_MISS_MM = 40.0

# item 3's carry distance and sample rate. 500 mm is far enough to hold the level
# constraint through a real trajectory rather than one waypoint, short enough to stay
# inside the cell
SWING_CARRY_DISTANCE_MM = 500.0
TRAJECTORY_SAMPLE_HZ = 20.0
TRAJECTORY_SAMPLE_INTERVAL_S = 1.0 / TRAJECTORY_SAMPLE_HZ

# item 3's tilt bounds: the floor rules out a move so constrained it never actually
# tipped the box, the ceiling is the cups' own lateral limit
# (pickcell.movers.CARRY_ORIENTATION_TOLERANCE_DEG, the service's own level-carry bound).
# Past it the cups let go rather than ride it out.
MIN_SWING_TILT_DEG = 0.05
MAX_SWING_TILT_DEG = 15.0
# item 3's bound on how far the tilt may still be sitting after the move returns plus
# SWING_SETTLE_S: a carried box that never settles level again rode the carry wrong
SWING_RESIDUAL_TILT_TOLERANCE_DEG = 1.0
SWING_SETTLE_S = 1.0


def tear_off_mass_kg(cups: Sequence[str], coaxial_limit_n: float) -> float:
    """The over-limit tear-off box's mass in kg: enough that it outweighs
    what every cup can hold at its own rated coaxial break force, with 25%
    margin."""
    return round(len(cups) * coaxial_limit_n * 1.25 / STANDARD_GRAVITY_M_S2)


# item 4's light stand-in headroom: 40% under what four cups hold at their own
# rated coaxial break force, so a correct plugin holds it well clear of the
# threshold rather than at its edge. The module's 0.1 s mean carries a few
# newtons of a lift's onset, so 20% margin sat too close to the threshold.
TEAR_OFF_UNDER_LIMIT_MARGIN = 0.6


def under_limit_mass_kg(cups: Sequence[str], coaxial_limit_n: float) -> float:
    """The under-limit tear-off stand-in's mass in kg: below every cup's own
    rated coaxial hold by TEAR_OFF_UNDER_LIMIT_MARGIN, so this item proves
    the threshold from both sides rather than the over-limit case alone."""
    return round(TEAR_OFF_UNDER_LIMIT_MARGIN * len(cups) * coaxial_limit_n / STANDARD_GRAVITY_M_S2)


# item 4's overload and under-limit stand-in prop names. Their masses are computed at
# run time from the configured --coaxial-limit-n, not from a module-level constant: the
# arm's own lift capacity, not this file, decides which limit the item can actually prove.
TEAR_OFF_BOX_PROP = "epick_tearoff_box"
TEAR_OFF_UNDER_LIMIT_BOX_PROP = "epick_under_limit_box"
# item 4's own budget for "did not rise": a box still on the station reads lift noise
# of this size or less, the same magnitude RESTING_HEIGHT_TOLERANCE_MM allows for a
# settled read
TEAR_OFF_RISE_TOLERANCE_MM = 5.0
# item 4's cap on how much of a stall message's own text gets printed when it carries no
# "stuck joints" clause to key off of: enough to show which waypoint and how many
# consecutive stalls, short of dumping the whole exception
LIFT_STALL_MESSAGE_HEAD_CHARS = 160
# item 4's clearance for sidestepping the real box, along the station's own travel
# axis (y, the same axis GRAB_NOTHING_OFFSETS_MM uses), off its resting spot before
# either stand-in spawns there
TEAR_OFF_SIDESTEP_MM = -400.0
# item 4's stow spots for the two stand-ins once the item is done with them: far
# outside every other item's operating envelope, since this checklist has no remove
# verb, and 1 m apart along x so neither spawn lands on the other
TEAR_OFF_STOW_XYZ_MM = (-3000.0, -3000.0, 500.0)
TEAR_OFF_UNDER_LIMIT_STOW_XYZ_MM = (-2000.0, -3000.0, 500.0)

# item 4's stand-in physics, carried from the cell's own infeed box
# (examples/configs/sim-palletizer-cell.json's infeed_box_1) rather than left at Isaac's
# authored defaults. A stand-in spawned with `mass` alone takes the default contact
# offset of 0.1 m and floats on it: the 2026-09-22 run read it settled 39 mm above a
# resting box's own height, tilted 19 degrees.
TEAR_OFF_BOX_FRICTION = 0.7
TEAR_OFF_BOX_RESTITUTION = 0.0
TEAR_OFF_BOX_CONTACT_OFFSET_M = 0.005

# items 5 and 6's two arm configurations at the reproduction's waypoint 10 (the place
# descent target), in degrees, from forward kinematics over the two branches recorded on
# the GPU machine. Index 4 (wrist 2) is positive in A, negative in B - see branch_of.
PLACE_DESCENT_SEED_JOINTS_DEG: dict[str, tuple[float, ...]] = {
    "A": (-232.33, -62.61, 72.77, 79.83, 90.0, -142.33),
    "B": (243.57, -104.8, 111.8, -97.0, -90.0, 153.57),
}
# item 5's raised approach above place_release_pose, before the confirmed-branch descent
DESCENT_START_RAISE_MM = 80.0
# items 5 and 6's settle window before the landing pose is read, matching the first-box
# suite's own item 3 post-release settle reasoning at a shorter, single-box scale
PLACE_SETTLE_S = 2.0
# item 6's bound on the landed box's own tilt off upright, read the way the workcell
# suite's own item 1 orientation_delta_deg reads any other pose disagreement
RELEASE_TILT_TOLERANCE_DEG = 2.0


# pure helpers, unit-tested without a robot in tests/test_gpu_checklist_palletizer.py


def verdict(name: str, ok: bool, detail: str) -> str:
    """Format one checklist line: "[PASS|FAIL] name: detail"."""
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {name}: {detail}"


def pose_delta_mm(pose_a: Mapping[str, float], pose_b: Mapping[str, float]) -> float:
    """Euclidean distance in mm between two x/y/z pose mappings, the same
    shape gpu_checklist_world's prop_pose_mm returns."""
    return math.sqrt(sum((pose_a[axis] - pose_b[axis]) ** 2 for axis in ("x", "y", "z")))


def frame_orientation_quat(orientation: Mapping[str, Any] | None) -> Quat:
    """A fragment frame block's `orientation` as a (w, x, y, z) quaternion.

    Absent means unrotated. Only the orientation-vector forms are read, since
    they are the only ones the vendored fragment uses, and any other type
    raises rather than pass as unrotated: a declaration this checklist cannot
    read is a declaration it cannot check."""
    if not orientation:
        return (1.0, 0.0, 0.0, 0.0)
    kind = orientation.get("type")
    if kind not in ("ov_degrees", "ov_radians"):
        raise ValueError(f"frame orientation type {kind!r} is not one this checklist reads")
    value = orientation.get("value") or {}
    axis = (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))
    if axis == (0.0, 0.0, 0.0):
        return (1.0, 0.0, 0.0, 0.0)
    theta = float(value.get("th", 0.0))
    return ov_to_quat(*axis, math.radians(theta) if kind == "ov_degrees" else theta)


def pose_mm(position_mm: Vec3, orientation_wxyz: Quat) -> dict[str, float]:
    """A pose in the `pose_in_world_mm` shape `prop_geometries` reports: x, y, z
    in mm and an orientation vector whose theta is in degrees."""
    x, y, z = position_mm
    o_x, o_y, o_z, theta = quat_to_ov(orientation_wxyz)
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "o_x": o_x,
        "o_y": o_y,
        "o_z": o_z,
        "theta": math.degrees(theta),
    }


def pose_quat(pose: Mapping[str, float]) -> Quat:
    """The orientation of a `pose_in_world_mm` mapping. No orientation vector,
    or a zero one, is unrotated."""
    axis = (float(pose.get("o_x", 0.0)), float(pose.get("o_y", 0.0)), float(pose.get("o_z", 0.0)))
    if axis == (0.0, 0.0, 0.0):
        return (1.0, 0.0, 0.0, 0.0)
    return ov_to_quat(*axis, math.radians(float(pose.get("theta", 0.0))))


def orientation_delta_deg(pose_a: Mapping[str, float], pose_b: Mapping[str, float]) -> float:
    """The smallest rotation, in degrees, taking one pose's orientation to the
    other's."""
    relative = quat_mul(quat_conj(pose_quat(pose_a)), pose_quat(pose_b))
    return math.degrees(2.0 * math.acos(min(1.0, abs(relative[0]))))


def _axes_mm(value: Mapping[str, Any] | None) -> Vec3:
    """An x/y/z mapping as a millimetre triple, zeros where absent."""
    value = value or {}
    return (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))


def lift_delta_mm(
    before_mm: Mapping[str, float], after_mm: Mapping[str, float]
) -> tuple[float, float]:
    """(vertical, horizontal) delta in mm between two pose readings."""
    vertical = after_mm["z"] - before_mm["z"]
    horizontal = math.hypot(after_mm["x"] - before_mm["x"], after_mm["y"] - before_mm["y"])
    return vertical, horizontal


def rode_with_tool(
    before_mm: Mapping[str, float],
    after_mm: Mapping[str, float],
    expected_lift_mm: float = LIFT_DISTANCE_MM,
    vertical_tolerance_mm: float = LIFT_VERTICAL_TOLERANCE_MM,
    horizontal_tolerance_mm: float = LIFT_HORIZONTAL_TOLERANCE_MM,
) -> tuple[bool, float, float]:
    """True when a box's pose moved with the tool through a vertical lift:
    the vertical delta is within tolerance of the expected lift distance, and
    the horizontal delta stayed near zero (a lift is purely vertical). Also
    returns the two deltas so a caller can print them. A box left behind
    reads a ~0 mm vertical delta here, which is what this is checked against."""
    vertical_mm, horizontal_mm = lift_delta_mm(before_mm, after_mm)
    ok = (
        abs(vertical_mm - expected_lift_mm) <= vertical_tolerance_mm
        and horizontal_mm <= horizontal_tolerance_mm
    )
    return ok, vertical_mm, horizontal_mm


def records_all_placed(records: Any) -> bool:
    """Whether the service's own outcome records say the box was placed.

    An empty record list is not success: it means the run ended before it
    recorded anything."""
    entries = list(records or [])
    if not entries:
        return False
    return all(dict(entry).get("outcome") == OUTCOME_PLACED for entry in entries)


def placed_seqs_in_order(records: Sequence[Mapping[str, Any]]) -> list[int]:
    """The `seq` of every `OUTCOME_PLACED` record, in the order the run
    placed them. A seq that failed and was later skipped never appears
    here, only in the raw record list."""
    return [int(record["seq"]) for record in records if record.get("outcome") == OUTCOME_PLACED]


def order_matches_pack_order(
    records: Sequence[Mapping[str, Any]], expected_seqs: Sequence[int]
) -> tuple[bool, list[int]]:
    """Whether the seqs placed, in the order the run placed them, equal the
    sequencer's own ascending pack order (`get_pack_order`'s placements,
    sorted by seq - the order `next_box` always serves them in). Returns
    the observed order too, so a mismatch is printable."""
    observed = placed_seqs_in_order(records)
    return observed == list(expected_seqs), observed


def placement_delta_mm(record: Mapping[str, Any]) -> float | None:
    """The Euclidean distance in mm between one record's `target_pose_mm`
    and `measured_pose_mm`, or None when the box never registered a
    measured pose (it never reached `set_box_transform`)."""
    target = record.get("target_pose_mm")
    measured = record.get("measured_pose_mm")
    if target is None or measured is None:
        return None
    return pose_delta_mm(target, measured)


def placement_table_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per record, sorted by seq: the box, its target and measured
    xyz, and the delta between them. `measured_mm` and `delta_mm` are None
    for a box that never got placed."""
    rows = []
    for record in sorted(records, key=lambda entry: int(entry["seq"])):
        rows.append(
            {
                "seq": int(record["seq"]),
                "box_prop": record.get("box_prop"),
                "outcome": record.get("outcome"),
                "target_mm": record.get("target_pose_mm") or {},
                "measured_mm": record.get("measured_pose_mm"),
                "delta_mm": placement_delta_mm(record),
            }
        )
    return rows


def format_placement_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Item 1's headline table: one line per box with its seq, its target
    and its measured pose, and the delta between them."""
    header = (
        f"{'seq':>3} {'box':<16} {'target xyz mm':<30} {'measured xyz mm':<30} {'delta mm':>10}"
    )
    lines = [header]
    for row in rows:
        target = row["target_mm"]
        measured = row["measured_mm"]
        target_x, target_y, target_z = (
            target.get("x", 0.0),
            target.get("y", 0.0),
            target.get("z", 0.0),
        )
        target_str = f"({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
        measured_str = (
            f"({measured['x']:.1f}, {measured['y']:.1f}, {measured['z']:.1f})"
            if measured is not None
            else "-"
        )
        delta_str = f"{row['delta_mm']:.2f}" if row["delta_mm"] is not None else "-"
        lines.append(
            f"{row['seq']:>3} {row['box_prop']!s:<16} {target_str:<30} {measured_str:<30} "
            f"{delta_str:>10}"
        )
    return "\n".join(lines)


def max_placement_delta_mm(
    records: Sequence[Mapping[str, Any]],
) -> tuple[float | None, str | None]:
    """The largest target-versus-measured delta across every record that
    reached a measured pose, and which box it belongs to. `(None, None)`
    when no record ever measured one."""
    best_delta_mm: float | None = None
    best_box_prop: str | None = None
    for record in records:
        delta_mm = placement_delta_mm(record)
        if delta_mm is None:
            continue
        if best_delta_mm is None or delta_mm > best_delta_mm:
            best_delta_mm = delta_mm
            best_box_prop = record.get("box_prop")
    return best_delta_mm, best_box_prop


def skip_handled_correctly(skipped_seqs: Sequence[int], final_state: str) -> tuple[bool, str]:
    """Whether a run that skipped at least one seq still reached
    `complete` rather than hanging or aborting outright. `(True, ...)` when
    nothing was skipped this run - a natural absence of failure is not
    itself a failure of this check, since forcing one needs a human hand in
    the viewport."""
    if not skipped_seqs:
        return True, "no seq was skipped this run"
    ok = final_state == "complete"
    return ok, f"skipped {list(skipped_seqs)}, run ended in state {final_state!r}"


def placement_check(
    box_pose_mm: Mapping[str, float] | None,
    place_xy_mm: Mapping[str, float],
    tolerance_mm: float = PLACEMENT_TOLERANCE_MM,
) -> tuple[bool, float]:
    """(ok, error_mm) of a box's horizontal position against the configured
    place target's x/y. Never ok when the box has no pose (it never
    registered as placed). Horizontal only: the resting height depends on
    packing details (box height, clearance) this checklist does not own."""
    if box_pose_mm is None:
        return False, math.inf
    error_mm = math.hypot(box_pose_mm["x"] - place_xy_mm["x"], box_pose_mm["y"] - place_xy_mm["y"])
    return error_mm <= tolerance_mm, error_mm


def settle_check(
    first_mm: Mapping[str, float] | None,
    second_mm: Mapping[str, float] | None,
    tolerance_mm: float = DRIFT_TOLERANCE_MM,
) -> tuple[bool, float]:
    """(ok, drift_mm) between two readings of the same box's pose taken
    SETTLE_WINDOW_S apart. Never ok when either reading is missing."""
    if first_mm is None or second_mm is None:
        return False, math.inf
    drift_mm = pose_delta_mm(first_mm, second_mm)
    return drift_mm <= tolerance_mm, drift_mm


def ready_time_s(samples: Sequence[tuple[float, Mapping[str, Any]]]) -> float | None:
    """Seconds from the first sample to the first sample where
    ``status["ready"]`` is true, or None if no sample ever was."""
    if not samples:
        return None
    started_at = samples[0][0]
    for sampled_at, status in samples:
        if status.get("ready"):
            return sampled_at - started_at
    return None


def fragment_component_frames_mm(fragment: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Every fragment component's declared world pose, keyed by component
    name, in the `pose_in_world_mm` shape: mm plus an orientation vector.

    Only components whose frame parents directly to ``"world"`` are
    included, which is every scenery component in
    `fragments/isaac-sim-palletizing.json`: none of them parents to another
    component. A component with no ``frame`` (`pallet-empty`, `tray-dock`)
    is skipped rather than assumed to sit at the origin."""
    frames: dict[str, dict[str, float]] = {}
    for component in fragment.get("components", []):
        frame = component.get("frame")
        if not frame or frame.get("parent") != "world":
            continue
        frames[str(component["name"])] = pose_mm(
            _axes_mm(frame.get("translation")), frame_orientation_quat(frame.get("orientation"))
        )
    return frames


def empty_spot_mm(
    geometries: Sequence[Mapping[str, Any]],
    near_xyz_mm: tuple[float, float, float],
    offsets_mm: Sequence[tuple[float, float]] = GRAB_NOTHING_OFFSETS_MM,
    clearance_mm: float = GRAB_NOTHING_CLEARANCE_MM,
) -> tuple[float, float, float] | None:
    """The first offset from `near_xyz_mm` with no non-fixed prop within
    `clearance_mm` in x and y, or None when every offset has something under
    it.

    Fixed props are ignored on purpose. A cup cannot take hold of a table or a
    pallet deck, so standing over one is still standing over nothing as far as
    a grab is concerned, and the station's own deck is the only surface these
    offsets can reach."""
    x, y, z = near_xyz_mm
    for offset_x, offset_y in offsets_mm:
        spot = (x + offset_x, y + offset_y)
        if all(
            abs(float(geometry["pose_in_world_mm"]["x"]) - spot[0]) >= clearance_mm
            or abs(float(geometry["pose_in_world_mm"]["y"]) - spot[1]) >= clearance_mm
            for geometry in geometries
            if not geometry.get("fixed")
        ):
            return (spot[0], spot[1], z)
    return None


def props_to_restore(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    exclude: set[str],
    tolerance_mm: float = RESTORE_TOLERANCE_MM,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Every non-fixed prop, other than `exclude`, that moved more than
    `tolerance_mm` between two `prop_geometries` readings, with the position
    it had in the first. The service's run restages the next box onto the
    station before a stop can land, since the restage is a teleport rather
    than a motion, and a box left there is what every later reset of the test
    box teleports into. On 2026-09-22 three runs in a row read the test box
    80 to 150 mm above the deck, tilted, resting on the second box."""
    before_by_name = {
        str(geometry["name"]): geometry["pose_in_world_mm"]
        for geometry in before
        if not geometry.get("fixed")
    }
    moved: list[tuple[str, tuple[float, float, float]]] = []
    for geometry in after:
        name = str(geometry.get("name", ""))
        if name in exclude or geometry.get("fixed") or name not in before_by_name:
            continue
        original = before_by_name[name]
        if pose_delta_mm(original, geometry["pose_in_world_mm"]) <= tolerance_mm:
            continue
        moved.append((name, (float(original["x"]), float(original["y"]), float(original["z"]))))
    return moved


def support_geometry_mm(
    geometries: Sequence[Mapping[str, Any]], component: str
) -> Mapping[str, Any] | None:
    """The first `prop_geometries` entry that belongs to `component`.

    Two naming schemes reach the stage. A collider declared as
    `frame.geometry` is spawned as `frame_<component>`, and prim names cannot
    hold a hyphen, so `pick-station` becomes `frame_pick_station`. Render
    scenery from `workcell_scenery.scenery_props` is still
    `<component>-<label>`. Matching only the second reported "no collider
    found" for components whose collider was sitting on the stage correctly.
    """
    frame_name = f"frame_{component.replace('-', '_')}"
    render_prefix = f"{component}-"
    for geometry in geometries:
        if str(geometry.get("name", "")) == frame_name:
            return geometry
    for geometry in geometries:
        if str(geometry.get("name", "")).startswith(render_prefix):
            return geometry
    return None


def support_top_z_mm(support: Mapping[str, Any]) -> float:
    """The world-frame z, in mm, of a support geometry's top face: its pose
    plus half its own z dimension."""
    pose = support["pose_in_world_mm"]
    _dim_x, _dim_y, dim_z = support["box_dims_mm"]
    return float(pose["z"]) + float(dim_z) / 2.0


def expected_rest_z_mm(support_top_z_mm: float, box_height_mm: float) -> float:
    """The z, in mm, a box of `box_height_mm` should rest at once it stops
    on a support whose top face is at `support_top_z_mm`."""
    return support_top_z_mm + box_height_mm / 2.0


def resting_check(
    expected_z_mm: float,
    measured_z_mm: float | None,
    tolerance_mm: float = RESTING_HEIGHT_TOLERANCE_MM,
) -> tuple[bool, float]:
    """(ok, error_mm) of a box's measured resting height against the height
    its support's own geometry implies. Never ok when the box never
    registered a pose."""
    if measured_z_mm is None:
        return False, math.inf
    error_mm = abs(measured_z_mm - expected_z_mm)
    return error_mm <= tolerance_mm, error_mm


def box_rests_on_support(
    box_mm: Mapping[str, float] | None,
    box_height_mm: float,
    support: Mapping[str, Any] | None,
    tolerance_mm: float = RESTING_HEIGHT_TOLERANCE_MM,
) -> tuple[bool, str]:
    """(ok, detail) of whether a box has come to rest ON a support: inside its
    footprint in x and y, at the height its top face implies.

    Item 4 derives its reach target from the box's LIVE pose, so where the box
    is within the station does not matter, and a teleport that lands 24 mm
    from where it was sent is not a failure. What matters is that the box is
    on the station at all. The runs of 2026-09-16 put it on the pedestal
    beside the arm and on the floor under the caution tape, and both of those
    are targets no plan can reach."""
    if support is None:
        return False, "the support has no collider in prop_geometries"
    if box_mm is None:
        return False, "the box registered no pose"
    support_pose = support["pose_in_world_mm"]
    dim_x, dim_y, _dim_z = (float(dim) for dim in support["box_dims_mm"])
    inside = {
        axis: abs(float(box_mm[axis]) - float(support_pose[axis])) <= half
        for axis, half in (("x", dim_x / 2.0), ("y", dim_y / 2.0))
    }
    expected_z_mm = expected_rest_z_mm(support_top_z_mm(support), box_height_mm)
    height_ok, height_error_mm = resting_check(expected_z_mm, float(box_mm["z"]), tolerance_mm)
    ok = all(inside.values()) and height_ok
    return ok, (
        f"inside x={inside['x']} y={inside['y']}, expected rest {expected_z_mm:.1f} mm, "
        f"measured z {float(box_mm['z']):.1f} mm, height error {height_error_mm:.3f} mm"
    )


def fragment_component_colliders_mm(
    fragment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Every world-parented component's declared `frame.geometry`, as the
    world pose and box dimensions its spawned collider should carry.

    The geometry's pose is composed onto the frame's, orientation included: a
    geometry translation is an offset within the frame rather than a second
    world position, and a frame turned 90 degrees turns its box with it. The
    two fences that turn are the case that a plain sum of translations got
    wrong. Components that declare no box geometry are absent, which is a
    declaration in itself: `scan-tunnel` is an arch, and one box would seal
    its opening."""
    colliders: dict[str, dict[str, Any]] = {}
    for component in fragment.get("components", []):
        frame = component.get("frame") or {}
        if frame.get("parent") != "world":
            continue
        geometry = frame.get("geometry") or {}
        if geometry.get("type") != "box":
            continue
        position_mm, orientation = compose_pose(
            _axes_mm(frame.get("translation")),
            frame_orientation_quat(frame.get("orientation")),
            _axes_mm(geometry.get("translation")),
            frame_orientation_quat(geometry.get("orientation")),
        )
        colliders[str(component["name"])] = {
            "pose_in_world_mm": pose_mm(position_mm, orientation),
            "box_dims_mm": tuple(float(geometry.get(axis, 0.0)) for axis in ("x", "y", "z")),
        }
    return colliders


def collider_matches_declaration(
    collider: Mapping[str, Any] | None,
    declared: Mapping[str, Any],
    tolerance_mm: float = FRAME_POSE_TOLERANCE_MM,
    angle_tolerance_deg: float = FRAME_ANGLE_TOLERANCE_DEG,
) -> tuple[bool, str]:
    """(ok, detail) of a spawned collider against the `frame.geometry` it came
    from: same world position, same orientation, same box dimensions.

    What a drop test proves for a support with a clear top face, this proves
    for one without. `robot-pedestal` carries the arm on its top face, so a
    box dropped on it lands on the robot and tumbles off, which says nothing
    about the collider and leaves the box somewhere the next item has to
    plan around. A fence has no top face a box could rest on at all, and its
    orientation is the whole question."""
    if collider is None:
        return False, "no collider found in prop_geometries"
    pose_error_mm = pose_delta_mm(collider["pose_in_world_mm"], declared["pose_in_world_mm"])
    angle_error_deg = orientation_delta_deg(
        collider["pose_in_world_mm"], declared["pose_in_world_mm"]
    )
    dims_error_mm = max(
        abs(float(spawned) - float(wanted))
        for spawned, wanted in zip(collider["box_dims_mm"], declared["box_dims_mm"], strict=True)
    )
    ok = (
        pose_error_mm <= tolerance_mm
        and angle_error_deg <= angle_tolerance_deg
        and dims_error_mm <= tolerance_mm
    )
    return ok, (
        f"declared {declared['box_dims_mm']} at {_pose_summary(declared['pose_in_world_mm'])}, "
        f"spawned {tuple(collider['box_dims_mm'])} at "
        f"{_pose_summary(collider['pose_in_world_mm'])}, pose error {pose_error_mm:.3f} mm, "
        f"orientation error {angle_error_deg:.3f} deg, dimension error {dims_error_mm:.3f} mm"
    )


def _pose_summary(pose: Mapping[str, float]) -> str:
    """One pose on one line: position in mm and the yaw its orientation vector
    carries, which is the only rotation this cell declares."""
    o_x, o_y, o_z, theta = quat_to_ov(pose_quat(pose))
    return (
        f"({pose['x']:.1f}, {pose['y']:.1f}, {pose['z']:.1f}) "
        f"ov ({o_x:.2f}, {o_y:.2f}, {o_z:.2f}) theta {math.degrees(theta):.1f}"
    )


def expected_render_prims_mm(
    component: str, frame_pose_mm: Mapping[str, float], visuals_reply: Mapping[str, Any]
) -> list[tuple[str, dict[str, float]]]:
    """Every box and capsule `component` draws, as the stage prim path the sim
    spawns it at and the world pose it should carry there: the primitive's own
    pose composed onto the frame the fragment declares for the component.

    Built with the same parser and the same prim naming the module spawns
    with, so a disagreement with what `prim_pose` reads back is a disagreement
    about composition, never about names. Spheres and arrows are not spawned
    and are not listed."""
    frame_position_m = tuple(float(frame_pose_mm[axis]) / MM_PER_M for axis in ("x", "y", "z"))
    frame_quat = pose_quat(frame_pose_mm)
    prims: list[tuple[str, dict[str, float]]] = []
    for primitive in parse_visuals(visuals_reply):
        if primitive.kind == "mesh":
            continue
        position_m, orientation = compose_pose(
            (frame_position_m[0], frame_position_m[1], frame_position_m[2]),
            frame_quat,
            primitive.position_m,
            primitive.orientation_wxyz,
        )
        path = f"/World/{prim_name(f'{component}-{primitive.label}')}"
        position_mm = (
            position_m[0] * MM_PER_M,
            position_m[1] * MM_PER_M,
            position_m[2] * MM_PER_M,
        )
        prims.append((path, pose_mm(position_mm, orientation)))
    return prims


def prim_pose_reply_mm(reply: Mapping[str, Any]) -> dict[str, float]:
    """A world `prim_pose` reply as the same pose mapping `prop_geometries`
    reports, so one comparison serves a collider and a render prim alike."""
    x, y, z = (float(value) for value in reply["position_mm"])
    vector = reply["orientation_vector"]
    return {
        "x": x,
        "y": y,
        "z": z,
        "o_x": float(vector["o_x"]),
        "o_y": float(vector["o_y"]),
        "o_z": float(vector["o_z"]),
        "theta": float(vector["theta_deg"]),
    }


def prim_matches_expected(
    measured: Mapping[str, float],
    expected: Mapping[str, float],
    tolerance_mm: float = FRAME_POSE_TOLERANCE_MM,
    angle_tolerance_deg: float = FRAME_ANGLE_TOLERANCE_DEG,
) -> tuple[bool, float, float]:
    """(ok, position error mm, orientation error deg) of a spawned prim
    against where its component's frame says it should be."""
    position_error_mm = pose_delta_mm(measured, expected)
    angle_error_deg = orientation_delta_deg(measured, expected)
    return (
        position_error_mm <= tolerance_mm and angle_error_deg <= angle_tolerance_deg,
        position_error_mm,
        angle_error_deg,
    )


def collision_reach_z_mm(model: Mapping[str, Any]) -> float:
    """The closest a kinematics file's own collision geometry comes to the
    TCP (z=0), in mm: the largest `translation.z + z/2` over every link's box
    geometry. The cup boxes reach closer than the body or the plate, so this
    is always a cup's own number, independent of which cup."""
    return max(
        float(link["geometry"]["translation"]["z"]) + float(link["geometry"]["z"]) / 2.0
        for link in model["links"]
    )


def offset_along_tool_mm(tcp_pose: Mapping[str, float], prim_pose: Mapping[str, float]) -> float:
    """How far `prim_pose` sits behind `tcp_pose` along the tool's own +Z
    axis, in mm. Positive is behind (opposite the direction the tool
    points), which is where every EPick render solid is authored."""
    tool_axis = quat_rotate(pose_quat(tcp_pose), (0.0, 0.0, 1.0))
    displacement_mm = (
        prim_pose["x"] - tcp_pose["x"],
        prim_pose["y"] - tcp_pose["y"],
        prim_pose["z"] - tcp_pose["z"],
    )
    projection_mm = sum(
        delta * axis for delta, axis in zip(displacement_mm, tool_axis, strict=True)
    )
    return -projection_mm


def refusal_within(
    elapsed_s: float, grab_delay_ms: float, retry_interval_s: float, slack_s: float
) -> bool:
    """Whether a refused `grab()` returned inside its expected window: the
    gripper's own deterministic grab delay, the automatic-mode retry
    interval it waits out before giving up, and `slack_s` of scheduling
    jitter on top of both."""
    return elapsed_s <= grab_delay_ms / 1000.0 + retry_interval_s + slack_s


def tilt_deg(tool_quat_wxyz: Quat, box_quat_wxyz: Quat) -> float:
    """The angle, in degrees, between the tool's own +Z axis (the cup axis,
    pointing down at the box it holds) and the box's own -Z axis (down
    through the box from its top face): how far a carried box has tipped
    relative to the cup holding it, independent of either one's yaw about
    that axis.

    A box hanging flat under the cups reads 0 by this measure. Comparing
    both axes' -Z, the shape this replaced, reads 180 for that same box: the
    tool points down at the box's top face, so the two -Z axes point
    opposite each other by construction, not because anything tipped."""
    tool_axis = quat_rotate(tool_quat_wxyz, (0.0, 0.0, 1.0))
    box_axis = quat_rotate(box_quat_wxyz, (0.0, 0.0, -1.0))
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(tool_axis, box_axis, strict=True))))
    return math.degrees(math.acos(cosine))


def swing_verdict(
    max_tilt_deg: float, residual_tilt_deg: float, held_throughout: bool
) -> tuple[bool, str]:
    """(ok, detail) of item 3's swing: the carry tipped the box some
    measurable amount without exceeding the cups' own lateral limit, it
    settled back level once the move stopped, and the grip never let go
    along the way."""
    tilt_ok = MIN_SWING_TILT_DEG <= max_tilt_deg < MAX_SWING_TILT_DEG
    residual_ok = residual_tilt_deg < SWING_RESIDUAL_TILT_TOLERANCE_DEG
    ok = tilt_ok and residual_ok and held_throughout
    return ok, (
        f"max tilt {max_tilt_deg:.3f} deg (bounds [{MIN_SWING_TILT_DEG}, {MAX_SWING_TILT_DEG})), "
        f"residual {residual_tilt_deg:.3f} deg (< {SWING_RESIDUAL_TILT_TOLERANCE_DEG}), "
        f"held throughout={held_throughout}"
    )


@dataclass
class TearOffReading:
    """One stand-in's lift: whether the plugin still reported it held, how
    far it rose, and how far its top face sagged below the TCP while held.

    `sag_mm` is None when the grab itself returned False, since the lift
    never happened and there is nothing to read it after. `could_not_lift`
    is True when the arm stalled trying to raise this stand-in back to its
    standoff: `held` and `rise_mm` are still read off the world afterwards,
    but neither dropped nor held is the right word for a lift the arm never
    completed. `mass_kg` is the stand-in's own configured mass, carried onto
    the reading so a failing verdict can name it. `peak_load_n` and
    `released_load_n` are read off the same `is_holding_something` reply
    that set `held`, and are None when that reply's meta carries neither
    key."""

    held: bool
    rise_mm: float
    sag_mm: float | None = None
    could_not_lift: bool = False
    mass_kg: float = 0.0
    peak_load_n: float | None = None
    released_load_n: float | None = None


def lift_stall_detail(name: str, error_message: str) -> str:
    """Item 4's print line when a stand-in's lift stalls the arm rather than
    completing: `name` plus whichever part of the stall message names the
    stuck joints, since that is the diagnosis worth reading, or the first
    LIFT_STALL_MESSAGE_HEAD_CHARS of the raw message when it carries no
    "stuck joints" clause to key off of."""
    marker = "stuck joints"
    marker_index = error_message.find(marker)
    stall_text = (
        error_message[marker_index:]
        if marker_index != -1
        else error_message[:LIFT_STALL_MESSAGE_HEAD_CHARS]
    )
    return f"{name}: the arm could not lift it ({stall_text})"


def _peak_release_suffix(peak_load_n: float | None, released_load_n: float | None) -> str:
    """The `, peak <n> N per cup[, released at <n> N]` clause `tear_off_verdict`
    appends to each box's detail, or "" when the reading carries no coaxial
    load meta at all."""
    if peak_load_n is None:
        return ""
    suffix = f", peak {peak_load_n:.1f} N per cup"
    if released_load_n is not None:
        suffix += f", released at {released_load_n:.1f} N"
    return suffix


def tear_off_verdict(
    over: TearOffReading, under: TearOffReading, light: TearOffReading
) -> tuple[bool, str]:
    """(ok, detail) of item 4's tear-off: a load past the cups' own rated
    coaxial hold (`over`) drops on lift, one comfortably under it (`under`)
    holds and rides LIFT_DISTANCE_MM, and the 2 kg box (`light`) does too.

    This proves the coaxial check as a threshold rather than a cliff: the
    over-limit reading alone cannot tell a plugin that never sees any cup
    load from one that correctly refuses only past the rated force. A held
    reading past TEAR_OFF_RISE_TOLERANCE_MM is a FAIL whose own sag is the
    diagnosis, since `over.sag_mm` near zero means the mass never took and
    near the spring's own full deflection means the spring is not
    reported. A stand-in the arm could not lift at all settles neither
    reading: the configured coaxial limit is proving a load this arm cannot
    raise, not a fact about the plugin."""
    if over.could_not_lift:
        return False, (
            f"the arm cannot lift {over.mass_kg:.0f} kg: lower the configured coaxial limit so "
            "four cups hold less than the arm can raise"
        )
    dropped = not over.held and over.rise_mm < TEAR_OFF_RISE_TOLERANCE_MM
    lifted = over.held and over.rise_mm >= TEAR_OFF_RISE_TOLERANCE_MM
    under_ok = under.held and abs(under.rise_mm - LIFT_DISTANCE_MM) <= LIFT_VERTICAL_TOLERANCE_MM
    light_ok = light.held and abs(light.rise_mm - LIFT_DISTANCE_MM) <= LIFT_VERTICAL_TOLERANCE_MM

    if dropped:
        over_detail = (
            f"over-limit released at the grab or on lift: held=False rise {over.rise_mm:.1f} mm"
            f"{_peak_release_suffix(over.peak_load_n, over.released_load_n)}"
        )
    elif lifted:
        sag_str = f"{over.sag_mm:.1f} mm" if over.sag_mm is not None else "n/a"
        over_detail = (
            "the over-limit stand-in was lifted and held: the module's monitor did not release "
            f"it (sag {sag_str})"
        )
    else:
        over_detail = (
            f"over-limit reading disagreed with itself: held={over.held} rise {over.rise_mm:.1f} mm"
        )
    under_detail = (
        f"under-limit: held={under.held} rise {under.rise_mm:.1f} mm (pass if held and within "
        f"{LIFT_VERTICAL_TOLERANCE_MM:.0f} mm of {LIFT_DISTANCE_MM:.0f})"
        f"{_peak_release_suffix(under.peak_load_n, under.released_load_n)}"
    )
    light_detail = (
        f"light (2 kg): held={light.held} rise {light.rise_mm:.1f} mm (pass if held and within "
        f"{LIFT_VERTICAL_TOLERANCE_MM:.0f} mm of {LIFT_DISTANCE_MM:.0f})"
        f"{_peak_release_suffix(light.peak_load_n, light.released_load_n)}"
    )
    ok = dropped and under_ok and light_ok
    detail = f"{over_detail}; {under_detail}; {light_detail}"
    return ok, detail


def hold_load_detail(meta: Mapping[str, Any]) -> str:
    """The coaxial load reading from an `is_holding_something` reply's own
    `meta`, for the print lines in items 3 to 6: the windowed mean pull on
    the most loaded cup and the peak single-step pull since the last grab,
    plus the windowed pull at release once the module has opened the
    gripper. An older module with none of these keys reads as "load n/a"."""
    if "coaxial_load_n" not in meta or "peak_coaxial_load_n" not in meta:
        return "load n/a"
    detail = (
        f"load {meta['coaxial_load_n']:.1f} N mean, peak {meta['peak_coaxial_load_n']:.1f} N "
        "per cup"
    )
    released_load_n = meta.get("released_load_n")
    if released_load_n is not None:
        detail += f", released at {released_load_n:.1f} N"
    monitor = meta.get("coaxial_monitor")
    if monitor is not None:
        detail += f", monitor {monitor}"
    return detail


def hold_lost_detail(leg: str, box_pose: Mapping[str, float] | None) -> str:
    """The failure detail for item 3, 5 or 6 when `is_holding_something` reads
    False right after a leg that moved the arm with the box held: which leg
    it happened after, and where the box was left when it let go."""
    location = f"box at {dict(box_pose)}" if box_pose is not None else "box has no known pose"
    return f"hold lost after {leg}: {location}"


def branch_of(joints_deg: Sequence[float]) -> str:
    """Which of the two wrist-2 solutions a joint configuration sits in:
    "A" for wrist 2 turned positive, "B" for negative. Index 4 is
    wrist_2_joint in UR_JOINT_NAMES order."""
    return "A" if joints_deg[4] > 0 else "B"


def branch_report(seed_name: str, at_standoff: str, at_descent_start: str) -> str:
    """Item 5's per-seed branch trace: which branch the standoff and the
    descent start actually reached, named against the seed asked for rather
    than assumed to match it, so the detail line says which configuration
    was actually measured."""
    held = at_standoff == at_descent_start
    matched_seed = at_standoff == seed_name and at_descent_start == seed_name
    outcome = "held the seed" if matched_seed else "did not hold the seed"
    changed_note = "" if held else ", branch changed between standoff and descent start"
    return (
        f"seed {seed_name}: standoff branch {at_standoff}, descent start branch "
        f"{at_descent_start} ({outcome}{changed_note})"
    )


def wrist_fold_deg(samples: Sequence[Sequence[float]]) -> float:
    """The largest reversal of wrist 1 (index 3) against its net direction
    of travel across a sampled descent, in degrees: how far it backtracked
    from the furthest point it had already reached in that direction. Zero
    for a monotone descent, since the running extreme is always the current
    value."""
    wrist_1_deg = [float(sample[3]) for sample in samples]
    if len(wrist_1_deg) < 2:
        return 0.0
    direction = 1.0 if wrist_1_deg[-1] >= wrist_1_deg[0] else -1.0
    running_extreme_deg = wrist_1_deg[0]
    worst_fold_deg = 0.0
    for value in wrist_1_deg[1:]:
        if direction > 0:
            running_extreme_deg = max(running_extreme_deg, value)
            fold_deg = running_extreme_deg - value
        else:
            running_extreme_deg = min(running_extreme_deg, value)
            fold_deg = value - running_extreme_deg
        worst_fold_deg = max(worst_fold_deg, fold_deg)
    return worst_fold_deg


# async drivers, not unit-tested (need a live world/arm/gripper/motion)


def _pose_to_mm(pose: Any) -> dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "z": float(pose.z)}


def _full_pose_mm(pose: Any) -> dict[str, float]:
    """A viam Pose's position and orientation vector as the same mapping
    shape `pose_in_world_mm` uses, so `pose_quat` and `tilt_deg` read a live
    TCP reading the same way they read a `prop_geometries` entry."""
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "o_x": float(pose.o_x),
        "o_y": float(pose.o_y),
        "o_z": float(pose.o_z),
        "theta": float(pose.theta),
    }


def _mm(reply: Mapping[str, Any], axis: str) -> float:
    """One millimetre axis out of a DoCommand reply, 0.0 when absent.

    A reply value is typed wide enough to be a mapping or a list, so a bare
    float() over it does not typecheck even though the wire only ever carries
    a number here.
    """
    value = reply.get(axis, 0.0)
    if isinstance(value, (int, float, str)):
        return float(value)
    return 0.0


async def _box_pose_mm(world: WorldApi, box_prop: str) -> Mapping[str, float] | None:
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    return prop_pose_mm(geometries, box_prop)


async def _box_top_face_xyz_mm(world: WorldApi, box_prop: str) -> tuple[float, float, float] | None:
    """``box_prop``'s current (x, y, top-face z) read live from the world,
    rather than assumed. None when the prop is not (yet) registered."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    for geometry in geometries:
        if geometry.get("name") != box_prop:
            continue
        pose = geometry["pose_in_world_mm"]
        _dim_x, _dim_y, dim_z = geometry["box_dims_mm"]
        return (float(pose["x"]), float(pose["y"]), float(pose["z"]) + float(dim_z) / 2.0)
    return None


async def _cell_world_state(world: WorldApi, exclude: set[str], held: Any = None) -> Any:
    """The planner's obstacles, built the same way the palletizer service
    builds them. Without these the planner routes the arm straight through the
    tables it is standing on, and physics stops it where the plan did not.

    `held` carries a box already on the cup (`held_box_transform`), so a free
    move plans the tool AND its payload as one rigid body - the level-carry
    items need this, since the tool alone clears what the box would sweep
    through."""
    from pickcell.obstacles import obstacles_from_prop_geometries, support_obstacle, world_state

    response = await world.do_command({"command": "prop_geometries"})
    geometries = response.get("geometries", [])
    obstacles = obstacles_from_prop_geometries(geometries, exclude)
    transforms = (held,) if held is not None else ()
    return world_state(None, obstacles, support_obstacle(FLOOR_Z_MM), transforms)


async def _box_dims_mm(world: WorldApi, box_prop: str) -> tuple[float, float, float]:
    """`box_prop`'s own x/y/z dimensions read live from the world, for the
    held-box transform items 3, 5 and 6 attach to the planner."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    for geometry in geometries:
        if geometry.get("name") == box_prop:
            return cast(
                "tuple[float, float, float]", tuple(float(d) for d in geometry["box_dims_mm"])
            )
    raise ValueError(f"{box_prop!r} has no known geometry in the world")


async def _move(
    gripper: Any,
    motion: Any,
    pose: Any,
    state: Any,
    linear: bool = False,
    level: bool = False,
    leg: str = "move",
) -> None:
    """Drives the GRIPPER's frame through `pickcell.movers.RealMover`, the
    same mover the palletizer service uses.

    Three things that mover already gets right and a hand-rolled motion call
    does not. It plans the gripper's frame, so the cup lands on the target
    instead of the flange landing there and driving the tool a tool-length
    into whatever is below. `linear=True` carries a linear constraint, so
    a short descent or lift stays in the arm's current configuration instead
    of being replanned into a different inverse-kinematics branch, which turns
    a 100 mm lift into a full elbow flip through the table. And `level=True`
    bounds a free move's tool orientation the way a carried box needs, the
    same CARRY_ORIENTATION_TOLERANCE_DEG constraint box-palletizer applies
    while the cup holds something."""
    from pickcell.movers import RealMover

    # this cell never calls look_from, so the mover's camera frame is unused
    mover = RealMover(motion, gripper.name, gripper.name)
    try:
        await mover.move_to(pose, state, linear=linear, level=level)
    except Exception as error:
        # which leg refused matters more than the message. The 2026-09-16 run
        # reported 'motion planner failed to find path' with no leg named, and
        # the standoff had in fact succeeded: it was the linear descent that
        # could not be planned, which is a different problem with a different
        # fix.
        kind = "linear" if linear else "free"
        raise RuntimeError(
            f"{leg} ({kind}) to {_pose_to_mm(pose)} failed: {type(error).__name__}: {error}"
        ) from error


def _pick_grasp_poses(top_face_xyz_mm: tuple[float, float, float]) -> tuple[Any, Any]:
    """(standoff, grasp) TCP poses over the box, from its live top-face
    reading, so items 1, 2 and 4 test the grab mechanism against an
    independent reading of the box's actual pose rather than the service's
    own arithmetic."""
    from viam.proto.common import Pose

    x, y, top_face_z_mm = top_face_xyz_mm
    # the cup stops short of the face rather than on it, or the arm stalls
    # against a box it cannot push through
    grasp_z_mm = top_face_z_mm + CUP_APPROACH_GAP_MM
    grasp = Pose(x=x, y=y, z=grasp_z_mm, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    standoff = Pose(
        x=x, y=y, z=grasp_z_mm + LIFT_DISTANCE_MM, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0
    )
    return standoff, grasp


async def _check_reach(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    box_prop: str,
    state: Any,
    approach_state: Any,
) -> tuple[str, bool]:
    from viam.proto.common import PoseInFrame

    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(
            FIRST_BOX_ITEMS[0], False, f"{box_prop!r} has no known geometry in the world"
        )
        print(line)
        return line, False

    standoff, target = _pick_grasp_poses(top_face_xyz_mm)
    # printed BEFORE the move: a planner that cannot reach the target is the
    # case where knowing what the target was matters most, and a print after
    # the move never runs
    print(f"  standoff (mm): {_pose_to_mm(standoff)}")
    print(f"  target (mm): {_pose_to_mm(target)}")

    async def approach(pose: Any, linear: bool) -> bool:
        """The leg to the standoff, planned with the box in the obstacle set
        so the arm does not sweep through it on the way."""
        await _move(gripper, motion, pose, approach_state, linear=linear, leg="approach")
        # the arm CONFIGURATION the approach lands in, not just its cup pose.
        # One cup pose has more than one inverse-kinematics solution and the
        # descent is only feasible from some of them, so which one this
        # approach found is the reading that explains a refusal.
        joints = await arm.get_joint_positions()
        print(f"  approached, joints (deg): {[round(v, 2) for v in joints.values]}")
        return True

    async def descend(pose: Any, linear: bool) -> bool:
        """The descent onto the box, planned with the box out of the obstacle
        set, since reaching it is the point."""
        await _move(gripper, motion, pose, state, linear=linear, leg="grasp descent")
        # a free descent, the fallback when the straight line is refused, is
        # a replan and can land in another configuration entirely. The run of
        # 2026-09-22 arrived with the base turned 25 degrees and the elbow
        # flipped, and every leg after it inherited that.
        joints = await arm.get_joint_positions()
        kind = "linear" if linear else "free"
        print(f"  descended ({kind}), joints (deg): {[round(v, 2) for v in joints.values]}")
        return True

    # the palletizer service's own approach, so this item exercises the code
    # that ships rather than a second copy of it
    await approach_and_descend(approach, descend, standoff, target)
    # the cup's pose in WORLD. arm.get_end_position() answers in the arm's own
    # base frame, which sits ARM_BASE_Z_MM above the floor, so comparing it to
    # a world target is off by the pedestal height.
    arrived: Any = await motion.get_pose(
        component_name=gripper.name,
        destination_frame="world",
    )
    end_position = arrived.pose if isinstance(arrived, PoseInFrame) else arrived
    error_mm = pose_delta_mm(_pose_to_mm(end_position), _pose_to_mm(target))
    ok = error_mm <= REACH_TOLERANCE_MM
    print(f"  arrived (mm): {_pose_to_mm(end_position)}")
    line = verdict(
        FIRST_BOX_ITEMS[0],
        ok,
        f"reach error {error_mm:.3f} mm; human: confirm the vacuum tool renders on the wrist "
        "in the camera image",
    )
    print(line)
    return line, ok


async def _check_grab_carry_release(
    world: WorldApi, gripper: Any, arm: Any, motion: Any, box_prop: str, state: Any
) -> tuple[str, bool]:
    """A self-contained demo, independent of item 1's final pose: descend
    onto the known box, grab, lift straight up, then release. Proves attach,
    carry, and release without touching the place target, leaving the box
    back near its pick pose for item 3's service run to pick up for real."""
    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(
            FIRST_BOX_ITEMS[1], False, f"{box_prop!r} has no known geometry in the world"
        )
        print(line)
        return line, False
    standoff, _grasp = _pick_grasp_poses(top_face_xyz_mm)

    # Item 1 has already driven the cup onto the box, so this starts from
    # there rather than repositioning first. A free-space move between two
    # poses 100 mm apart is replanned into a different arm configuration, and
    # the arm winds itself into the table on the way, which says nothing about
    # whether the cup can pick anything up.
    grabbed = await gripper.grab()
    before_lift_mm = await _box_pose_mm(world, box_prop)
    # the lift is constrained: a free replan here would flip the arm's
    # configuration and the payload would not travel straight up
    await _move(gripper, motion, standoff, state, linear=True)
    after_lift_mm = await _box_pose_mm(world, box_prop)
    print(f"  before lift (mm): {before_lift_mm}")
    print(f"  after lift (mm): {after_lift_mm}")
    joints = await arm.get_joint_positions()
    print(f"  lifted, joints (deg): {[round(v, 2) for v in joints.values]}")

    if before_lift_mm is None or after_lift_mm is None:
        rode_ok, vertical_mm, horizontal_mm = False, 0.0, 0.0
    else:
        rode_ok, vertical_mm, horizontal_mm = rode_with_tool(before_lift_mm, after_lift_mm)

    await gripper.open()
    holding_status = await gripper.is_holding_something()
    released_ok = not holding_status.is_holding_something
    # where the box came to rest after the release, which the service's own
    # pick reads next. A box that did not fall back onto the station, or fell
    # somewhere else, is the difference between the service's standoff and
    # the cup's actual position.
    await asyncio.sleep(RESET_SETTLE_S)
    print(f"  after release (mm): {await _box_pose_mm(world, box_prop)}")

    ok = bool(grabbed) and rode_ok and released_ok
    detail = (
        f"grabbed={grabbed} lift delta {vertical_mm:.1f} mm vertical / {horizontal_mm:.1f} mm "
        f"horizontal, holding after release={holding_status.is_holding_something}"
    )
    line = verdict(FIRST_BOX_ITEMS[1], ok, detail)
    print(line)
    return line, ok


async def _await_first_record(palletizer: Any) -> Mapping[str, Any]:
    """Polls `status` until the run has recorded its first box or stopped
    running, bounded by STATUS_TIMEOUT_S. Raises on the timeout."""
    started_at = time.monotonic()
    status = await palletizer.do_command({"command": "status"})
    while status.get("state") == "running" and not list(status.get("records") or []):
        if time.monotonic() - started_at > STATUS_TIMEOUT_S:
            raise TimeoutError(
                f"box-palletizer still running with no record after {STATUS_TIMEOUT_S:.0f}s"
            )
        await asyncio.sleep(STATUS_POLL_S)
        status = await palletizer.do_command({"command": "status"})
    return status


async def _await_not_running(palletizer: Any) -> Mapping[str, Any]:
    """Polls `status` until the run leaves `running`, bounded by
    STATUS_TIMEOUT_S. Raises on the timeout."""
    started_at = time.monotonic()
    status = await palletizer.do_command({"command": "status"})
    while status.get("state") == "running":
        if time.monotonic() - started_at > STATUS_TIMEOUT_S:
            raise TimeoutError(
                f"box-palletizer still running {STATUS_TIMEOUT_S:.0f}s after being told to stop"
            )
        await asyncio.sleep(STATUS_POLL_S)
        status = await palletizer.do_command({"command": "status"})
    return status


async def _restore_moved_props(
    world: WorldApi, props_before: Sequence[Mapping[str, Any]], box_prop: str
) -> None:
    """Puts back every prop other than `box_prop` that the service's run
    moved, and says which. The test box itself is where the items after this
    one want to read it."""
    props_after = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    for name, position_mm in props_to_restore(props_before, props_after, {box_prop}):
        await world.do_command(
            {
                "command": "set_prop_pose",
                "name": name,
                "position": list(position_mm),
                "orientation_rpy_deg": [0.0, 0.0, 0.0],
            }
        )
        print(f"  restored {name} to {position_mm}")


async def _check_placement(
    world: WorldApi,
    palletizer: Any,
    sequencer: SequencerClient,
    arm: Any,
    gripper: Any,
    box_prop: str,
) -> tuple[str, bool]:
    """The first-box suite's own done-when: `{"command": "start"}` on the
    `box-palletizer` service, the first box's record, then the box's resting
    pose measured against the sequencer's own slot, immediately after release
    and again after SETTLE_WINDOW_S.

    One box, not the pack. The service runs the sequencer's whole order, so
    this stops it once the first record lands and judges that record alone.
    The sequencer's cursor is reset first: it keeps its progress across runs,
    and on 2026-09-22 the warm run started at seq 2, restaged the second box
    onto the station where this checklist had just put the first, and picked
    a box that had been teleported into another one. The slot the box is
    measured against is the record's own `target_pose_mm`, the sequencer's
    number rather than this file's `--place-*-mm` arguments, which name a
    place the sequencer never uses."""
    import json

    await sequencer.reset_cursor()
    props_before = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    start_result = await palletizer.do_command({"command": "start"})
    print(f"  start: {json.dumps(dict(start_result), default=str, sort_keys=True)}")

    status = await _await_first_record(palletizer)
    if status.get("state") == "running":
        await palletizer.do_command({"command": "stop"})
        status = await _await_not_running(palletizer)
    await _restore_moved_props(world, props_before, box_prop)

    records = list(status.get("records") or [])
    print(f"  status: {json.dumps(dict(status), default=str, sort_keys=True)}")
    first = dict(records[0]) if records else {}
    # The run's state and the record's outcome are different claims: a run
    # that finishes having failed its pick reports "complete" with a record
    # whose outcome is "failed". Checking only the state passes such a run
    # whenever a box happens to be sitting at the target already, which is
    # exactly what a rerun leaves behind. And a run stopped after its first
    # box carries a second, "stopped" record that says nothing about the
    # first.
    if first.get("outcome") != OUTCOME_PLACED:
        # the two readings a stall leaves behind: the configuration the arm is
        # held in, and where the box ended up. The 2026-09-22 run had the box
        # standing on end on the pallet by the next item with nothing in the
        # output to say when it left the station.
        joints = await arm.get_joint_positions()
        print(f"  arm after the run, joints (deg): {[round(v, 2) for v in joints.values]}")
        print(f"  box after the run (mm): {await _box_pose_mm(world, box_prop)}")
        # a run that failed between grab and release leaves the box on the
        # cup, and a welded box ignores every teleport the items after this
        # one send it. The 2026-09-22 runs read 250 mm of "error" in item 7
        # for exactly that. Say so, then let go so the rest of the run means
        # something.
        holding = (await gripper.is_holding_something()).is_holding_something
        print(f"  cup still holding after the run: {holding}")
        if holding:
            await gripper.open()
            await asyncio.sleep(RESET_SETTLE_S)
            print(f"  box after letting go (mm): {await _box_pose_mm(world, box_prop)}")
        line = verdict(
            FIRST_BOX_ITEMS[2],
            False,
            f"box-palletizer did not place the box: state={status.get('state')!r} {records}",
        )
        print(line)
        return line, False

    slot_mm = {
        "x": float(first["target_pose_mm"]["x"]),
        "y": float(first["target_pose_mm"]["y"]),
    }
    print(f"  sequencer slot (mm): {slot_mm}")
    placed_mm = await _box_pose_mm(world, box_prop)
    placed_ok, placed_error_mm = placement_check(placed_mm, slot_mm)
    print(f"  placed (mm): {placed_mm}, error {placed_error_mm:.3f} mm")

    await asyncio.sleep(SETTLE_WINDOW_S)
    settled_mm = await _box_pose_mm(world, box_prop)
    settled_ok, settled_error_mm = placement_check(settled_mm, slot_mm)
    drift_ok, drift_mm = settle_check(placed_mm, settled_mm)
    print(f"  after {SETTLE_WINDOW_S:.0f}s (mm): {settled_mm}, error {settled_error_mm:.3f} mm")

    ok = placed_ok and settled_ok and drift_ok
    detail = (
        f"first record={first}, place error {placed_error_mm:.3f} mm against the sequencer's "
        f"slot, settle error {settled_error_mm:.3f} mm, drift {drift_mm:.3f} mm over "
        f"{SETTLE_WINDOW_S:.0f}s"
    )
    line = verdict(FIRST_BOX_ITEMS[2], ok, detail)
    print(line)
    return line, ok


async def _check_grab_nothing(
    world: WorldApi,
    gripper: Any,
    arm: Any,
    motion: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    state: Any,
) -> tuple[str, bool]:
    """Moves over a spot with nothing under it and attempts a grab, bounded
    so a hang is reported as a failure.

    The spot is chosen against a live `prop_geometries` reading rather than
    assumed, and searched along the pick station from the configured pick
    pose. Searching from the box's own pose sent the cup over the arm's base
    on 2026-09-22, since the service had carried the box to the pallet, and
    "zero IK solutions" is not a statement about the gripper."""
    box_height_mm = await _box_height_mm(world, box_prop)
    station_top_face_xyz_mm = (
        float(pick_pose_mm["x"]),
        float(pick_pose_mm["y"]),
        float(pick_pose_mm["z"]) + box_height_mm / 2.0,
    )
    print(f"  box now (mm): {await _box_pose_mm(world, box_prop)}")
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    empty_xyz_mm = empty_spot_mm(geometries, station_top_face_xyz_mm)
    if empty_xyz_mm is None:
        line = verdict(
            FIRST_BOX_ITEMS[3],
            False,
            "no spot within reach has nothing under it, so a grab here would test the "
            "opposite of what this item claims",
        )
        print(line)
        return line, False
    print(f"  empty spot (mm): {empty_xyz_mm}")
    standoff, grasp = _pick_grasp_poses(empty_xyz_mm)
    await _move(gripper, motion, standoff, state)
    await _move(gripper, motion, grasp, state, linear=True)
    try:
        grabbed = await asyncio.wait_for(gripper.grab(), timeout=GRAB_NOTHING_TIMEOUT_S)
        holding_status = await asyncio.wait_for(
            gripper.is_holding_something(), timeout=GRAB_NOTHING_TIMEOUT_S
        )
    except TimeoutError:
        line = verdict(
            FIRST_BOX_ITEMS[3], False, f"timed out after {GRAB_NOTHING_TIMEOUT_S:.0f}s (hung)"
        )
        print(line)
        return line, False

    ok = not grabbed and not holding_status.is_holding_something
    line = verdict(
        FIRST_BOX_ITEMS[3],
        ok,
        f"grab()={grabbed} holding={holding_status.is_holding_something}",
    )
    print(line)
    return line, ok


async def sample_ready_time(
    world: WorldApi,
    poll_s: float = DEFAULT_READY_POLL_S,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
) -> dict[str, Any]:
    """Polls ``status`` through the world's ``do_command`` until ``ready``
    is true or ``timeout_s`` elapses, and reports the elapsed ready time."""
    samples: list[tuple[float, Mapping[str, Any]]] = []
    started_at = time.monotonic()
    while True:
        status = await world.do_command({"command": "status"})
        samples.append((time.monotonic(), status))
        if status.get("ready") or time.monotonic() - started_at > timeout_s:
            break
        await asyncio.sleep(poll_s)
    return {"ready_time_s": ready_time_s(samples), "sample_count": len(samples)}


async def _check_cost(
    world: WorldApi, run_label: str, item: str = FIRST_BOX_ITEMS[4]
) -> tuple[str, bool]:
    """The first-box suite's cost item, and (with `item` overridden) the
    epick suite's item 7: both read the same cold/warm ready time and step
    rate, so one reading function serves both labels."""
    import json

    ready = await sample_ready_time(world)
    print(f"  [{run_label}] ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  [{run_label}] step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")
    sim_time_ratio = step_rate.get("baseline", {}).get("sim_time_ratio")
    line = verdict(
        item,
        True,
        f"run_label={run_label} ready_time_s={ready['ready_time_s']} "
        f"sim_time_ratio={sim_time_ratio}",
    )
    print(line)
    return line, True


async def _park_arm(world: WorldApi, gripper: Any, motion: Any) -> None:
    """Moves the cup to PARK_XYZ_MM, clear of every surface a box is about to
    be teleported onto, with the whole cell as obstacles. Where the previous
    test left the arm is not this test's business."""
    from viam.proto.common import Pose

    park = Pose(
        x=PARK_XYZ_MM[0],
        y=PARK_XYZ_MM[1],
        z=PARK_XYZ_MM[2],
        o_x=0.0,
        o_y=0.0,
        o_z=-1.0,
        theta=0.0,
    )
    await _move(gripper, motion, park, await _cell_world_state(world, set()), leg="park")
    print(f"  arm parked over {PARK_XYZ_MM[:2]} at {PARK_XYZ_MM[2]:.0f} mm")


async def _reset_pick_box(
    world: WorldApi, box_prop: str, pick_pose_mm: Mapping[str, float], support: str
) -> tuple[bool, str]:
    """Put the box back at its known pick pose, then read back whether it came
    to rest on `support`. Returns (ok, detail).

    Every item here assumes the box starts at `pick_pose_mm`, and a rerun
    against a cell left as the previous run finished measures whatever is
    lying around instead. A world `reset` restores neither prop poses nor the
    arm, so the pose is written explicitly.

    The read-back is not ceremony. `set_prop_pose` returns before physics has
    moved the prop, and the reach target is derived from the box's live pose,
    so an unsettled read sends the arm to the box's PREVIOUS location, which
    item 7 leaves on the floor under the caution tape. The box settling a
    couple of centimetres from where it was sent is not a failure, since the
    target follows it, so what is checked is the support it landed on."""
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": box_prop,
            "position": [pick_pose_mm["x"], pick_pose_mm["y"], pick_pose_mm["z"]],
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    await asyncio.sleep(RESET_SETTLE_S)
    box_height_mm = await _box_height_mm(world, box_prop)
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    settled_mm = prop_pose_mm(geometries, box_prop)
    ok, detail = box_rests_on_support(
        settled_mm, box_height_mm, support_geometry_mm(geometries, support)
    )
    print(
        f"  reset {box_prop} to {dict(pick_pose_mm)}: settled at {settled_mm}, "
        f"on {support}: {detail} ({'ok' if ok else 'NOT ON THE SUPPORT'})"
    )
    return ok, detail


async def _box_height_mm(world: WorldApi, box_prop: str) -> float:
    """`box_prop`'s own z dimension read live from the world, so a drop
    test's expected resting height never carries an invented box size."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    for geometry in geometries:
        if geometry.get("name") == box_prop:
            return float(geometry["box_dims_mm"][2])
    raise ValueError(f"{box_prop!r} has no known geometry in the world")


def group_frame_pose_mm(
    visuals_reply: Mapping[str, Any], component: str
) -> dict[str, float] | None:
    """The world pose of a workcell component's own `<component>/group` frame
    primitive, which is the frame origin every other primitive it draws is
    posed against.

    `None` when the reply carries no such primitive, which is how a component
    that is not workcell scenery reports "nothing to compare" rather than a
    misleading zero.
    """
    primitives = visuals_reply.get("visuals")
    if not isinstance(primitives, list):
        primitives = next((v for v in visuals_reply.values() if isinstance(v, list)), [])
    for primitive in primitives:
        if not isinstance(primitive, Mapping):
            continue
        if primitive.get("label") != f"{component}/group":
            continue
        pose = primitive.get("pose")
        if not isinstance(pose, Mapping):
            return None
        return {axis: _mm(pose, axis) for axis in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
    return None


async def _check_spawned_prims(
    world: WorldApi,
    arm_name: str,
    component: str,
    frame_pose_mm: Mapping[str, float],
    visuals_reply: Mapping[str, Any],
) -> tuple[bool, int, str]:
    """(ok, prims compared, detail) of every box and capsule `component`
    draws, read back off the stage through the world's `prim_pose` verb and
    compared with where the fragment's frame puts it.

    This is the half of item 1 that reads the SIM rather than the component.
    The group frame check above it proves the component knows where it is.
    This proves the module put its primitives there. The runs before
    2026-09-22 had the first without the second, and the pick station drew
    585 mm from its own collider for a week with item 1 passing."""
    worst_position = (0.0, "")
    worst_angle = (0.0, "")
    compared = 0
    ok = True
    for prim_path, expected_mm in expected_render_prims_mm(component, frame_pose_mm, visuals_reply):
        try:
            reply = await world.do_command(
                {"command": "prim_pose", "name": arm_name, "prim_path": prim_path}
            )
        except Exception as exc:  # noqa: BLE001 - a missing prim is this check's finding, not its end
            return False, compared, f"{prim_path} not on the stage: {exc}"
        prim_ok, position_error_mm, angle_error_deg = prim_matches_expected(
            prim_pose_reply_mm(reply), expected_mm
        )
        compared += 1
        ok = ok and prim_ok
        if position_error_mm >= worst_position[0]:
            worst_position = (position_error_mm, prim_path)
        if angle_error_deg >= worst_angle[0]:
            worst_angle = (angle_error_deg, prim_path)
    if compared == 0:
        return True, 0, "draws no box or capsule"
    return (
        ok,
        compared,
        (
            f"{compared} prims on the stage, worst position error {worst_position[0]:.3f} mm "
            f"({worst_position[1]}), worst orientation error {worst_angle[0]:.3f} deg "
            f"({worst_angle[1]})"
        ),
    )


async def _check_component_frames(
    world: WorldApi,
    robot: Any,
    arm_name: str,
    fragment_frames_mm: Mapping[str, Mapping[str, float]],
) -> tuple[str, bool]:
    """Item 1: where each fragment component actually renders, against the
    world pose its fragment frame declares, read two ways.

    First from the component's own `<name>/group` visual primitive, whose
    pose is the frame origin it draws everything else relative to. NOT from
    `get_pose` or `get_attributes.pose`, which report the component's CORNER:
    pick-station declares a frame at (400, -650, 200) and reports a pose of
    (200, -1200, 220), so comparing the two comes out 585 mm apart for a
    component that is placed correctly. The pallet passes only because its
    corner and its frame origin happen to coincide.

    Then from the stage itself: every box and capsule the component draws,
    at the prim the module spawned for it, against the fragment frame
    composed with the primitive's own pose.
    """
    from viam.components.generic import Generic

    all_ok = True
    prims_compared = 0
    for name, expected_mm in fragment_frames_mm.items():
        try:
            client = Generic.from_robot(robot, name)
            visuals_reply = await client.do_command({"get_visuals": True})
        except Exception as exc:  # noqa: BLE001 - one component's failure should not stop the rest
            print(f"  {name}: get_visuals failed: {exc!r}")
            all_ok = False
            continue
        group_pose = group_frame_pose_mm(visuals_reply, name)
        if group_pose is None:
            print(f"  {name}: no {name}/group frame primitive in get_visuals")
            all_ok = False
            continue
        error_mm = pose_delta_mm(expected_mm, group_pose)
        angle_error_deg = orientation_delta_deg(expected_mm, group_pose)
        ok = error_mm <= FRAME_POSE_TOLERANCE_MM and angle_error_deg <= FRAME_ANGLE_TOLERANCE_DEG
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  {name}: declared {_pose_summary(expected_mm)}, group frame "
            f"{_pose_summary(group_pose)}, error {error_mm:.3f} mm / {angle_error_deg:.3f} deg "
            f"({status})"
        )
        prims_ok, compared, detail = await _check_spawned_prims(
            world, arm_name, name, expected_mm, visuals_reply
        )
        all_ok = all_ok and prims_ok
        prims_compared += compared
        print(f"    on the stage: {detail} ({'PASS' if prims_ok else 'FAIL'})")
    line = verdict(
        WORKCELL_ITEMS[0],
        all_ok,
        f"{len(fragment_frames_mm)} fragment components checked, {prims_compared} spawned "
        "prims compared with their frame",
    )
    print(line)
    return line, all_ok


async def _drop_box_on_support(
    world: WorldApi, box_prop: str, component: str, box_height_mm: float
) -> tuple[bool, float]:
    """Item 2's per-support drop: teleport `box_prop` to hover
    `DROP_HOVER_MM` above `component`'s own collider top face, let it settle,
    and check its resting z against the height that collider's geometry
    implies. Returns (ok, error_mm); an unresolved support geometry fails."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    support = support_geometry_mm(geometries, component)
    if support is None:
        print(f"  {component}: no collider found in prop_geometries")
        return False, math.inf

    top_z_mm = support_top_z_mm(support)
    expected_z_mm = expected_rest_z_mm(top_z_mm, box_height_mm)
    hover_mm = {
        "x": float(support["pose_in_world_mm"]["x"]),
        "y": float(support["pose_in_world_mm"]["y"]),
        "z": expected_z_mm + DROP_HOVER_MM,
    }
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": box_prop,
            "position": [hover_mm["x"], hover_mm["y"], hover_mm["z"]],
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    await asyncio.sleep(DROP_SETTLE_S)
    rested_mm = await _box_pose_mm(world, box_prop)
    ok, error_mm = resting_check(expected_z_mm, rested_mm["z"] if rested_mm else None)
    status = "PASS" if ok else "FAIL"
    print(
        f"  {component}: support top {top_z_mm:.1f} mm, expected rest {expected_z_mm:.1f} mm, "
        f"measured {rested_mm}, error {error_mm:.3f} mm ({status})"
    )
    return ok, error_mm


async def _check_support_drops(
    world: WorldApi, box_prop: str, declared_colliders_mm: Mapping[str, Mapping[str, Any]]
) -> tuple[str, bool]:
    """Item 2: the pick station and pallet each stop a dropped box at the
    height their own geometry implies, and every declared collider is checked
    against its declaration, position, orientation and size. The drop test
    cannot reach a pedestal the arm stands on or a fence with no top face,
    and it cannot see a panel standing along the wrong axis at all."""
    box_height_mm = await _box_height_mm(world, box_prop)
    all_ok = True
    for component in ("pick-station", "pallet"):
        ok, _error_mm = await _drop_box_on_support(world, box_prop, component, box_height_mm)
        all_ok = all_ok and ok

    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    for component, declared in declared_colliders_mm.items():
        ok, detail = collider_matches_declaration(
            support_geometry_mm(geometries, component), declared
        )
        all_ok = all_ok and ok
        print(f"  {component}: {detail} ({'PASS' if ok else 'FAIL'})")

    line = verdict(
        WORKCELL_ITEMS[1],
        all_ok,
        "drop test against pick-station and pallet, declaration check against every one of the "
        f"{len(declared_colliders_mm)} frame.geometry colliders, orientation included",
    )
    print(line)
    return line, all_ok


async def _check_fence_and_tunnel_obstacles(world: WorldApi) -> tuple[str, bool]:
    """Item 3's automatable half: `fence-*` and `scan-tunnel-*` colliders are
    registered in `prop_geometries`, so `pickcell.obstacles` picks them up as
    planner obstacles the way `_cell_world_state` builds them for every move
    in this file. Whether a plan actually detours around them, rather than
    merely never approaching them, is a Kit-viewport read a human still has
    to make."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    names = {str(geometry["name"]) for geometry in geometries}
    fence_present = any(name.startswith(("fence-", "frame_fence_")) for name in names)
    tunnel_present = any(name.startswith(("scan-tunnel-", "frame_scan_tunnel")) for name in names)
    # scan-tunnel deliberately declares no frame.geometry. It is an arch the
    # arm and boxes pass through, and a frame carries one shape, so a single
    # box would seal the opening. Whether it needs a collider at all is item
    # 5's question, so counting its absence as a failure here reports one
    # decision twice and hides whether the fences actually landed.
    ok = fence_present
    detail = (
        f"fence obstacle present={fence_present}, "
        f"scan-tunnel present={tunnel_present} (none declared, by design, item 5); "
        "human: confirm in the Kit viewport that item 4's pick-to-place plan swings clear of "
        "both rather than clipping through them"
    )
    line = verdict(WORKCELL_ITEMS[2], ok, detail)
    print(line)
    return line, ok


async def _run_pick_and_place_regression(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    palletizer: Any,
    sequencer: SequencerClient,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
) -> tuple[str, bool]:
    """Item 4: the first-box suite's own mechanism, reach, grab/carry/release,
    the service's placement, and a grab with nothing under the tool, run end
    to end against the workcell."""
    await _park_arm(world, gripper, motion)
    reset_ok, reset_detail = await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    if not reset_ok:
        line = verdict(
            WORKCELL_ITEMS[3],
            False,
            f"the box never came to rest on the pick station ({reset_detail}), so every target "
            "below would have been derived from wherever it was left instead",
        )
        print(line)
        return line, False
    state = await _cell_world_state(world, {box_prop})
    # the same cell with the box left IN, for the legs that have no business
    # passing through it
    approach_state = await _cell_world_state(world, set())

    print("  -- reach --")
    _reach_line, reach_ok = await _check_reach(
        world, arm, gripper, motion, box_prop, state, approach_state
    )
    print("  -- grab, carry, release --")
    _carry_line, carry_ok = await _check_grab_carry_release(
        world, gripper, arm, motion, box_prop, state
    )
    print("  -- placement (box-palletizer service) --")
    _place_line, place_ok = await _check_placement(
        world, palletizer, sequencer, arm, gripper, box_prop
    )
    print("  -- grab with nothing under the tool --")
    _grab_nothing_line, grab_nothing_ok = await _check_grab_nothing(
        world, gripper, arm, motion, box_prop, pick_pose_mm, state
    )

    ok = reach_ok and carry_ok and place_ok and grab_nothing_ok
    line = verdict(
        WORKCELL_ITEMS[3], ok, "the first-box suite's reach/carry/place/grab-nothing sequence"
    )
    print(line)
    return line, ok


async def _check_scan_tunnel_collider(world: WorldApi) -> tuple[str, bool]:
    """Item 5: whether `scan-tunnel` already has a collider registered, the
    way `robot-pedestal`'s derived one does. Always informational, since
    either answer is a valid finding to record, not a failure: this decides
    whether `workcell_scenery.DERIVED_COLLIDER_MODELS` needs a second entry,
    which is a source change this checklist does not make."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    has_collider = any(
        str(geometry["name"]).startswith(("scan-tunnel-", "frame_scan_tunnel"))
        for geometry in geometries
    )
    detail = (
        f"scan-tunnel collider registered in prop_geometries={has_collider}; human: drive the "
        "arm through the tunnel's span and confirm whether it stops at a real collider (needs "
        "workcell_scenery.DERIVED_COLLIDER_MODELS the way robot-pedestal does) or passes through"
    )
    line = verdict(WORKCELL_ITEMS[4], True, detail)
    print(line)
    return line, True


async def _check_cost_against_first_box(world: WorldApi, run_label: str) -> tuple[str, bool]:
    """Item 6: cost, with `sim_time_ratio` printed alongside the cited
    first-box suite baseline rather than re-measuring a cell this suite
    deleted. The first-box suite's `ready_time_s` is not comparable (see
    `FIRST_BOX_BASELINE_SIM_TIME_RATIO`'s comment), so this suite's own
    `ready_time_s` is printed as a first reading rather than a delta against
    that gap, for the pack suite to use as its baseline."""
    import json

    ready = await sample_ready_time(world)
    print(f"  [{run_label}] ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  [{run_label}] step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")
    sim_time_ratio = step_rate.get("baseline", {}).get("sim_time_ratio")
    print(
        f"  first-box suite baseline: sim_time_ratio={FIRST_BOX_BASELINE_SIM_TIME_RATIO} "
        "(ready_time_s is not a baseline: the first-box suite's own reading is a measurement "
        "gap, not a result)"
    )
    print(
        f"  [{run_label}] ready_time_s={ready['ready_time_s']} is a first reading, to carry "
        "forward as the pack suite's baseline, not a delta against the first-box suite"
    )
    line = verdict(
        WORKCELL_ITEMS[5],
        True,
        f"run_label={run_label} ready_time_s={ready['ready_time_s']} (first reading) "
        f"sim_time_ratio={sim_time_ratio}, first-box suite baseline sim_time_ratio="
        f"{FIRST_BOX_BASELINE_SIM_TIME_RATIO}",
    )
    print(line)
    return line, True


async def _check_render_only_has_no_collider(
    world: WorldApi,
    gripper: Any,
    box_prop: str,
    fragment_frames_mm: Mapping[str, Mapping[str, float]],
) -> tuple[str, bool]:
    """Item 7, an implementation risk rather than a plan requirement:
    `sim_manager.py` builds a `collision: False` cube as a cuboid and then
    strips `UsdPhysics.CollisionAPI` and `RigidBodyAPI`, since `omni` and
    `pxr` are not importable in this environment and the mesh path's bare-
    authoring route could not be tested here. Drops `box_prop` over
    `caution-tape` (a render-only floor decal) and checks it rests on the
    floor rather than on the tape."""
    # a box still on the cup from item 4 ignores the teleport below
    await gripper.open()
    box_height_mm = await _box_height_mm(world, box_prop)
    tape_mm = fragment_frames_mm["caution-tape"]
    hover_mm = {
        "x": tape_mm["x"],
        "y": tape_mm["y"],
        "z": FLOOR_Z_MM + box_height_mm / 2.0 + DROP_HOVER_MM,
    }
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": box_prop,
            "position": [hover_mm["x"], hover_mm["y"], hover_mm["z"]],
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    await asyncio.sleep(DROP_SETTLE_S)
    rested_mm = await _box_pose_mm(world, box_prop)
    expected_floor_rest_z_mm = FLOOR_Z_MM + box_height_mm / 2.0
    ok, error_mm = resting_check(expected_floor_rest_z_mm, rested_mm["z"] if rested_mm else None)
    line = verdict(
        WORKCELL_ITEMS[6],
        ok,
        f"expected floor rest {expected_floor_rest_z_mm:.1f} mm, measured {rested_mm}, error "
        f"{error_mm:.3f} mm",
    )
    print(line)
    return line, ok


async def _run_full_pack(palletizer: Any) -> dict[str, Any]:
    """Starts `box-palletizer`'s own run and polls `status` until it leaves
    `running`, bounded by `PACK_STATUS_TIMEOUT_S`. Every pack suite item
    reads the same run's final status rather than driving the arm itself,
    since the sequencer owns every target pose here."""
    import json

    start_result = await palletizer.do_command({"command": "start"})
    print(f"  start: {json.dumps(dict(start_result), default=str, sort_keys=True)}")

    started_at = time.monotonic()
    status = await palletizer.do_command({"command": "status"})
    while status.get("state") == "running":
        if time.monotonic() - started_at > PACK_STATUS_TIMEOUT_S:
            raise TimeoutError(
                f"box-palletizer timed out after {PACK_STATUS_TIMEOUT_S:.0f}s still running (hung)"
            )
        await asyncio.sleep(STATUS_POLL_S)
        status = await palletizer.do_command({"command": "status"})
    print(f"  status: {json.dumps(dict(status), default=str, sort_keys=True)}")
    return cast("dict[str, Any]", status)


def _check_placement_table(
    records: Sequence[Mapping[str, Any]], expected_count: int
) -> tuple[str, bool]:
    """Item 1: the headline table, plus whether every one of the
    sequencer's boxes actually reached `OUTCOME_PLACED`."""
    print(format_placement_table(placement_table_rows(records)))
    placed = placed_seqs_in_order(records)
    ok = len(placed) == expected_count and len(set(placed)) == expected_count
    line = verdict(PACK_ITEMS[0], ok, f"{len(placed)} of {expected_count} boxes placed")
    print(line)
    return line, ok


def _check_pack_order(
    records: Sequence[Mapping[str, Any]], expected_seqs: Sequence[int]
) -> tuple[str, bool]:
    """Item 2: the order boxes were placed in against the sequencer's own
    `get_pack_order`."""
    ok, observed = order_matches_pack_order(records, expected_seqs)
    line = verdict(
        PACK_ITEMS[1],
        ok,
        f"placed order {observed}, get_pack_order order {list(expected_seqs)}",
    )
    print(line)
    return line, ok


def _check_obstacle_source(
    records: Sequence[Mapping[str, Any]], obstacle_source_label: str
) -> tuple[str, bool]:
    """Item 3's automatable half: every box placed without an aborted plan,
    under whichever `obstacle_source` the overlay is currently deployed
    with. `obstacle_source_label` only tags this reading, since the config
    itself is a deploy-time attribute this script cannot flip - run the
    script once per value and compare, the same as `--run-label`."""
    ok = records_all_placed(records)
    line = verdict(
        PACK_ITEMS[2],
        ok,
        f"obstacle_source={obstacle_source_label}: every box placed with no plan aborting={ok}; "
        "human: confirm in the Kit viewport that no plan clipped through a fence, the scan "
        "tunnel or the pallet stack, then rerun with the overlay's obstacle_source set to the "
        "other value and compare",
    )
    print(line)
    return line, ok


def _check_measured_delta(records: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    """Item 4: the largest plan-versus-actual difference `set_box_transform`
    fed back, across the eight boxes."""
    max_delta_mm, box_prop = max_placement_delta_mm(records)
    ok = max_delta_mm is not None
    detail = (
        f"largest plan-versus-actual delta {max_delta_mm:.2f} mm ({box_prop})"
        if max_delta_mm is not None
        else "no record ever measured a placed pose"
    )
    line = verdict(PACK_ITEMS[3], ok, detail)
    print(line)
    return line, ok


async def _check_skip_handling(sequencer: SequencerClient, final_state: str) -> tuple[str, bool]:
    """Item 5's automatable half: when a seq was in fact skipped this run,
    the run still reached `complete` rather than hanging. Forcing a
    placement to fail needs a human hand in the Kit viewport, so this stays
    informational when nothing failed."""
    progress = await sequencer.progress()
    ok, detail = skip_handled_correctly(progress.skipped_seqs, final_state)
    line = verdict(
        PACK_ITEMS[4],
        ok,
        f"{detail}; human: to exercise this path, drag one box out of the arm's reach in the "
        "Kit viewport partway through a run and confirm it gets skip_box'ed rather than hanging "
        "the run",
    )
    print(line)
    return line, ok


async def _check_cost_against_workcell(world: WorldApi, run_label: str) -> tuple[str, bool]:
    """Item 6: cost, printed as a first reading against the workcell suite's.
    The workcell suite has no recorded GPU run yet, so no `sim_time_ratio`
    number exists there to cite the way `FIRST_BOX_BASELINE_SIM_TIME_RATIO`
    does. Printed for a human to diff by eye against the workcell suite's own
    run output instead of fabricating one here."""
    import json

    ready = await sample_ready_time(world)
    print(f"  [{run_label}] ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  [{run_label}] step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")
    sim_time_ratio = step_rate.get("baseline", {}).get("sim_time_ratio")
    line = verdict(
        PACK_ITEMS[5],
        True,
        f"run_label={run_label} ready_time_s={ready['ready_time_s']} "
        f"sim_time_ratio={sim_time_ratio}; compare by eye against the workcell suite's own "
        "recorded run",
    )
    print(line)
    return line, True


async def _run_pack(
    world: WorldApi,
    palletizer: Any,
    sequencer: SequencerClient,
    run_label: str,
    obstacle_source_label: str,
) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []

    pack_order = await sequencer.pack_order()
    expected_seqs = sorted(placement.seq for placement in pack_order.placements)

    print("\n-- full pack run --")
    status = await _run_full_pack(palletizer)
    records = list(status.get("records", []))

    print("\n-- placement table --")
    results.append(_check_placement_table(records, pack_order.quantity))

    print("\n-- placement order vs get_pack_order --")
    results.append(_check_pack_order(records, expected_seqs))

    print("\n-- obstacle source --")
    results.append(_check_obstacle_source(records, obstacle_source_label))

    print("\n-- measured vs target delta --")
    results.append(_check_measured_delta(records))

    print("\n-- skip on repeated failure --")
    results.append(await _check_skip_handling(sequencer, str(status.get("state", ""))))

    print("\n-- cost vs the workcell suite --")
    results.append(await _check_cost_against_workcell(world, run_label))

    print("\n== summary ==")
    for line, _ in results:
        print(line)
    return results


async def _guarded(item: str, check: Callable[[], Awaitable[tuple[str, bool]]]) -> tuple[str, bool]:
    """Run one checklist item, recording a failure that raises as a failed
    item rather than as the end of the run.

    Both GPU runs of 2026-09-16 lost items 5, 6 and 7 to one planner error
    inside item 4: the exception left the process, so a cell that takes
    minutes to boot answered four questions instead of seven. An item that
    cannot run has failed, and the items after it still have something to
    say."""
    try:
        return await check()
    # catching everything is the point: the next item is worth running whatever
    # this one raised, and the reason is recorded in the item's own verdict
    except Exception as error:  # noqa: BLE001
        line = verdict(item, False, f"raised {type(error).__name__}: {error}")
        print(line)
        return line, False


async def _run_workcell(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    palletizer: Any,
    sequencer: SequencerClient,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    run_label: str,
    robot: Any,
    fragment_path: Path,
) -> list[tuple[str, bool]]:
    import json

    fragment = json.loads(fragment_path.read_text())
    fragment_frames_mm = fragment_component_frames_mm(fragment)

    results: list[tuple[str, bool]] = []

    # a cup left engaged by a previous run re-attaches the box the moment the
    # tool touches it, and a prop welded to the gripper ignores set_prop_pose,
    # which turns every drop test and every reset below into a reading of the
    # arm's pose. The warm run of 2026-09-16 reported 244 mm of nonsense for
    # exactly this reason. The arm is parked for the same reason: item 2 drops
    # boxes onto the station and the pallet, and the previous run left the arm
    # wherever its last test ended.
    await gripper.open()
    try:
        await _park_arm(world, gripper, motion)
    except Exception as exc:  # noqa: BLE001 - a park that fails leaves the items to say what it cost
        print(f"  arm could not be parked before the run: {exc!r}")

    declared_colliders_mm = fragment_component_colliders_mm(fragment)

    print("\n-- component frames --")
    results.append(
        await _guarded(
            WORKCELL_ITEMS[0],
            lambda: _check_component_frames(world, robot, arm.name, fragment_frames_mm),
        )
    )

    print("\n-- support drop tests --")
    results.append(
        await _guarded(
            WORKCELL_ITEMS[1],
            lambda: _check_support_drops(world, box_prop, declared_colliders_mm),
        )
    )

    print("\n-- fence and scan-tunnel obstacles --")
    results.append(
        await _guarded(WORKCELL_ITEMS[2], lambda: _check_fence_and_tunnel_obstacles(world))
    )

    print("\n-- first-box suite pick and place regression --")
    results.append(
        await _guarded(
            WORKCELL_ITEMS[3],
            lambda: _run_pick_and_place_regression(
                world,
                arm,
                gripper,
                motion,
                palletizer,
                sequencer,
                box_prop,
                pick_pose_mm,
            ),
        )
    )

    print("\n-- scan-tunnel collider question --")
    results.append(await _guarded(WORKCELL_ITEMS[4], lambda: _check_scan_tunnel_collider(world)))

    print("\n-- cost vs the first-box suite --")
    results.append(
        await _guarded(WORKCELL_ITEMS[5], lambda: _check_cost_against_first_box(world, run_label))
    )

    print("\n-- render-only scenery risk (caution-tape) --")
    results.append(
        await _guarded(
            WORKCELL_ITEMS[6],
            lambda: _check_render_only_has_no_collider(
                world, gripper, box_prop, fragment_frames_mm
            ),
        )
    )

    print("\n== summary ==")
    for line, _ in results:
        print(line)
    return results


async def _check_epick_smoke_cited() -> tuple[str, bool]:
    """Item 0: not re-run. The rig that produced this result drives a bare
    gripper it authors over the arm's mount link itself, and the module now
    owns that link for the EPick body, so running it again would author a
    second gripper over it."""
    line = verdict(
        EPICK_ITEMS[0],
        True,
        "cited: passed 2026-09-22 20:02 on the GPU machine, three passes (one attachment "
        "point on the tool body grips a box prop, then four on the cup pattern); not re-run "
        "here, since the rig that produced this result would author a second gripper over "
        "the mount link the model now owns",
    )
    print(line)
    return line, True


async def _check_epick_render_and_kinematics(
    world: WorldApi, arm: Any, gripper: Any, motion: Any
) -> tuple[str, bool]:
    """Item 1: `get_kinematics` serves the vendored `epick_model.json`
    byte for byte, that file's own cup collision boxes reach no closer than
    the 26 mm approach gap, and the render body's prim sits where the
    module's own EPICK dimensions say it should, along the tool axis from a
    TCP reading independent of the render code."""
    import json

    from viam.proto.common import PoseInFrame

    vendored_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "isaac_module"
        / "kinematics_files"
        / "epick_model.json"
    )
    vendored_bytes = vendored_path.read_bytes()
    kinematics_reply = await gripper.get_kinematics()
    sva_bytes = bytes(kinematics_reply[1])
    bytes_match = sva_bytes == vendored_bytes
    model = json.loads(sva_bytes)
    reach_z_mm = collision_reach_z_mm(model)
    expected_reach_z_mm = float(EPICK["cups"]["tcp_clearance_z_mm"])
    reach_ok = reach_z_mm == expected_reach_z_mm

    arrived: Any = await motion.get_pose(component_name=gripper.name, destination_frame="world")
    tcp_mm = _full_pose_mm(arrived.pose if isinstance(arrived, PoseInFrame) else arrived)

    body_prim_path = f"/World/{prim_name(arm.name)}/wrist_3_link/{EPICK_PRIM}/render/body"
    prim_reply = await world.do_command(
        {"command": "prim_pose", "name": arm.name, "prim_path": body_prim_path}
    )
    prim_mm = prim_pose_reply_mm(prim_reply)
    offset_mm = offset_along_tool_mm(tcp_mm, prim_mm)
    expected_offset_mm = abs(float(EPICK["body"]["visual_center_z_mm"]))
    offset_ok = abs(offset_mm - expected_offset_mm) <= REACH_TOLERANCE_MM

    ok = bytes_match and reach_ok and offset_ok
    detail = (
        f"kinematics bytes match vendored file={bytes_match}, collision reach {reach_z_mm:.1f} "
        f"mm (expected {expected_reach_z_mm:.1f}), render body {offset_mm:.2f} mm behind the "
        f"TCP (expected {expected_offset_mm:.1f})"
    )
    line = verdict(EPICK_ITEMS[1], ok, detail)
    print(line)
    return line, ok


async def _check_gating(
    world: WorldApi,
    gripper: Any,
    motion: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    grab_delay_ms: float,
    retry_interval_s: float,
) -> tuple[str, bool]:
    """Item 2: a grab at the normal CUP_APPROACH_GAP_MM holds, one over
    GRAB_GATING_MISS_MM does not, and the refusal itself is bounded rather
    than a hang."""
    from viam.proto.common import Pose

    reset_ok, reset_detail = await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    if not reset_ok:
        line = verdict(
            EPICK_ITEMS[2], False, f"box did not settle on the pick station ({reset_detail})"
        )
        print(line)
        return line, False

    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(EPICK_ITEMS[2], False, f"{box_prop!r} has no known geometry in the world")
        print(line)
        return line, False
    state = await _cell_world_state(world, {box_prop})
    standoff = pick_grasp_standoff_pose(top_face_xyz_mm)
    grasp = pick_grasp_pose(top_face_xyz_mm)

    async def approach(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, state, linear=linear, leg="gating approach")
        return True

    async def descend(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, state, linear=linear, leg="gating descent")
        return True

    await approach_and_descend(approach, descend, standoff, grasp)
    holds_at_gap = await gripper.grab()
    print(f"  grab at {CUP_APPROACH_GAP_MM:.0f} mm over the top face: {holds_at_gap}")
    await gripper.open()
    await _move(gripper, motion, standoff, state, linear=True, leg="gating retreat")

    x, y, top_face_z_mm = top_face_xyz_mm
    miss_pose = Pose(
        x=x, y=y, z=top_face_z_mm + GRAB_GATING_MISS_MM, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0
    )
    await _move(gripper, motion, miss_pose, state, linear=True, leg="gating miss descent")
    started_at = time.monotonic()
    grabbed_at_miss = await gripper.grab()
    elapsed_s = time.monotonic() - started_at
    print(
        f"  grab at {GRAB_GATING_MISS_MM:.0f} mm over the top face: {grabbed_at_miss}, elapsed "
        f"{elapsed_s:.2f} s"
    )
    await gripper.open()
    timing_ok = refusal_within(elapsed_s, grab_delay_ms, retry_interval_s, GRAB_REFUSAL_SLACK_S)

    ok = holds_at_gap and not grabbed_at_miss and timing_ok
    bound_s = grab_delay_ms / 1000.0 + retry_interval_s + GRAB_REFUSAL_SLACK_S
    detail = (
        f"holds at {CUP_APPROACH_GAP_MM:.0f} mm={holds_at_gap}, refuses at "
        f"{GRAB_GATING_MISS_MM:.0f} mm={not grabbed_at_miss}, refusal took {elapsed_s:.2f} s "
        f"(bound {bound_s:.2f} s)={timing_ok}"
    )
    line = verdict(EPICK_ITEMS[2], ok, detail)
    print(line)
    return line, ok


async def _check_swing(
    world: WorldApi,
    gripper: Any,
    motion: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    place_pose_mm: Mapping[str, float],
) -> tuple[str, bool]:
    """Item 3: a level cross-cell carry, the same free-move-plus-orientation-
    constraint box-palletizer uses once the cup holds something, sampled at
    TRAJECTORY_SAMPLE_HZ so the tilt it actually rode is measured rather than
    assumed from the constraint's own bound."""
    from viam.proto.common import Pose, PoseInFrame

    reset_ok, reset_detail = await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    if not reset_ok:
        line = verdict(
            EPICK_ITEMS[3], False, f"box did not settle on the pick station ({reset_detail})"
        )
        print(line)
        return line, False

    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(EPICK_ITEMS[3], False, f"{box_prop!r} has no known geometry in the world")
        print(line)
        return line, False

    standoff = pick_grasp_standoff_pose(top_face_xyz_mm)
    grasp = pick_grasp_pose(top_face_xyz_mm)
    pick_state = await _cell_world_state(world, {box_prop})

    async def approach(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, pick_state, linear=linear, leg="swing approach")
        return True

    async def descend(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, pick_state, linear=linear, leg="swing descent")
        return True

    await approach_and_descend(approach, descend, standoff, grasp)
    if not await gripper.grab():
        line = verdict(EPICK_ITEMS[3], False, "grab() returned False at the pick pose")
        print(line)
        return line, False

    try:
        await _move(gripper, motion, standoff, pick_state, linear=True, leg="swing lift")
        swing_lift_status = await gripper.is_holding_something()
        held_after_lift = swing_lift_status.is_holding_something
        print(
            f"  held after swing lift: {held_after_lift}, "
            f"{hold_load_detail(swing_lift_status.meta)}"
        )
        if not held_after_lift:
            box_mm = await _box_pose_mm(world, box_prop)
            print(f"  box pose: {box_mm}")
            line = verdict(EPICK_ITEMS[3], False, hold_lost_detail("swing lift", box_mm))
            print(line)
            return line, False

        box_dims_mm = await _box_dims_mm(world, box_prop)
        held = held_box_transform(box_prop, box_dims_mm, gripper.name)
        carry_state = await _cell_world_state(world, {box_prop}, held)

        pick_x, pick_y, _pick_z = top_face_xyz_mm
        dx, dy = place_pose_mm["x"] - pick_x, place_pose_mm["y"] - pick_y
        norm = math.hypot(dx, dy) or 1.0
        carry_target = Pose(
            x=pick_x + dx / norm * SWING_CARRY_DISTANCE_MM,
            y=pick_y + dy / norm * SWING_CARRY_DISTANCE_MM,
            z=standoff.z,
            o_x=0.0,
            o_y=0.0,
            o_z=-1.0,
            theta=0.0,
        )

        async def _read_tool_mm() -> dict[str, float]:
            arrived: Any = await motion.get_pose(
                component_name=gripper.name, destination_frame="world"
            )
            return _full_pose_mm(arrived.pose if isinstance(arrived, PoseInFrame) else arrived)

        samples: list[tuple[dict[str, float], Mapping[str, float] | None, bool]] = []

        async def _sample_loop() -> None:
            while True:
                tool_mm = await _read_tool_mm()
                box_mm = await _box_pose_mm(world, box_prop)
                holding = (await gripper.is_holding_something()).is_holding_something
                samples.append((tool_mm, box_mm, holding))
                await asyncio.sleep(TRAJECTORY_SAMPLE_INTERVAL_S)

        carry_started_at = time.monotonic()
        sampler = asyncio.create_task(_sample_loop())
        try:
            await _move(
                gripper, motion, carry_target, carry_state, level=True, leg="cross-cell carry"
            )
        finally:
            sampler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler
        carry_duration_s = time.monotonic() - carry_started_at

        carry_status = await gripper.is_holding_something()
        held_after_carry = carry_status.is_holding_something
        print(
            f"  held after cross-cell carry: {held_after_carry}, "
            f"{hold_load_detail(carry_status.meta)}"
        )
        if not held_after_carry:
            box_mm = await _box_pose_mm(world, box_prop)
            print(f"  box pose: {box_mm}")
            line = verdict(EPICK_ITEMS[3], False, hold_lost_detail("cross-cell carry", box_mm))
            print(line)
            return line, False

        await asyncio.sleep(SWING_SETTLE_S)
        residual_tool_mm = await _read_tool_mm()
        residual_box_mm = await _box_pose_mm(world, box_prop)

        tilts_deg = [
            tilt_deg(pose_quat(tool_mm), pose_quat(box_mm))
            for tool_mm, box_mm, _holding in samples
            if box_mm is not None
        ]
        held_throughout = bool(samples) and all(holding for _t, _b, holding in samples)
        max_tilt_deg = max(tilts_deg) if tilts_deg else 0.0
        residual_tilt_deg = (
            tilt_deg(pose_quat(residual_tool_mm), pose_quat(residual_box_mm))
            if residual_box_mm is not None
            else math.inf
        )
        print(
            f"  carry duration {carry_duration_s:.2f} s, {len(samples)} samples, max tilt "
            f"{max_tilt_deg:.3f} deg, residual {residual_tilt_deg:.3f} deg, held "
            f"throughout={held_throughout}"
        )

        ok, detail = swing_verdict(max_tilt_deg, residual_tilt_deg, held_throughout)
        line = verdict(EPICK_ITEMS[3], ok, detail)
        print(line)
        return line, ok
    finally:
        try:
            await gripper.open()
        except Exception:  # noqa: BLE001 - the box restore below still matters if opening failed
            pass
        try:
            await asyncio.sleep(RESET_SETTLE_S)
            await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
        except Exception:  # noqa: BLE001 - report the item's own failure, not the cleanup's
            pass


async def _stow_tear_off_stand_in(
    world: WorldApi,
    prop: str,
    mass_kg: float,
    stow_xyz_mm: tuple[float, float, float],
    real_box_dims_mm: tuple[float, float, float],
) -> None:
    """Finds `prop` and teleports it to `stow_xyz_mm`, or spawns it there at
    `mass_kg` scaled to the real box's own measured `real_box_dims_mm` if it
    has never existed this run, then reads back where it settled."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    already_spawned = any(str(geometry.get("name")) == prop for geometry in geometries)
    if already_spawned:
        print(f"  reusing {prop} from an earlier run: its mass is whatever that run spawned")
        await world.do_command(
            {
                "command": "set_prop_pose",
                "name": prop,
                "position": list(stow_xyz_mm),
                "orientation_rpy_deg": [0.0, 0.0, 0.0],
            }
        )
    else:
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {
                    "name": prop,
                    "type": "cube",
                    # a unit-edge cube scaled to the real box's own measured dimensions: the
                    # live world reports box_dims_mm composed already, not the config's
                    # original size/scale split
                    "position": [v / MM_PER_M for v in stow_xyz_mm],
                    "size": 1.0,
                    "scale": [d / MM_PER_M for d in real_box_dims_mm],
                    "mass": mass_kg,
                    "friction": TEAR_OFF_BOX_FRICTION,
                    "restitution": TEAR_OFF_BOX_RESTITUTION,
                    "contact_offset": TEAR_OFF_BOX_CONTACT_OFFSET_M,
                },
            }
        )
    await asyncio.sleep(RESET_SETTLE_S)
    settled_geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    settled_mm = prop_pose_mm(settled_geometries, prop)
    print(f"  stand-in {prop} sent to stow {stow_xyz_mm}: settled at {settled_mm}")


async def _tear_off_lift(
    world: WorldApi,
    gripper: Any,
    motion: Any,
    arm: Any,
    prop: str,
    mass_kg: float,
    leg_label: str,
) -> TearOffReading | None:
    """Approaches, grabs and lifts `prop` back to its standoff, printing the
    reading the same way for the over-limit stand-in, the under-limit
    stand-in and the 2 kg box: grab's own return, the rise, whether the
    plugin still reports holding, and the sag (the box's top face below the
    TCP while held). None when `prop` has no known geometry to approach."""
    top_face_xyz_mm = await _box_top_face_xyz_mm(world, prop)
    if top_face_xyz_mm is None:
        return None
    standoff = pick_grasp_standoff_pose(top_face_xyz_mm)
    grasp = pick_grasp_pose(top_face_xyz_mm)
    state = await _cell_world_state(world, {prop})

    async def approach(pose: Any, linear: bool) -> bool:
        # the arm's own planner from the seeded configuration, the way item 5
        # reaches its standoff: the motion service's free move from the same
        # seed twice resolved to the shoulder folded into the pedestal and
        # stalled the base joint (GPU machine, 2026-09-23)
        await arm.move_to_position(pose)
        return True

    async def descend(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, state, linear=linear, leg=f"{leg_label} descent")
        return True

    # the approach starts from the same seeded configuration item 5 uses for
    # this pick, so the arm's planner resolves the standoff to the same branch
    from viam.proto.component.arm import JointPositions

    seed_joints_deg = PLACE_DESCENT_SEED_JOINTS_DEG["B"]
    await arm.move_to_joint_positions(JointPositions(values=list(seed_joints_deg)))
    print(f"  {leg_label}: seeded branch B before the approach")
    await approach_and_descend(approach, descend, standoff, grasp)
    grabbed = await gripper.grab()
    if not grabbed:
        print(f"  {prop} ({mass_kg:.0f} kg): grab=False rise 0.0 mm holding=False sag n/a")
        return TearOffReading(held=False, rise_mm=0.0, sag_mm=None, mass_kg=mass_kg)

    before_mm = await _box_pose_mm(world, prop)
    try:
        await _move(gripper, motion, standoff, state, linear=True, leg=f"{leg_label} lift")
    except Exception as error:
        if not is_execution_failure(error):
            raise
        print(f"  {lift_stall_detail(prop, str(error))}")
        after_mm = await _box_pose_mm(world, prop)
        stalled_status = await gripper.is_holding_something()
        holding = stalled_status.is_holding_something
        rise_mm = (
            lift_delta_mm(before_mm, after_mm)[0]
            if before_mm is not None and after_mm is not None
            else 0.0
        )
        print(
            f"  {prop} ({mass_kg:.0f} kg): holding={holding} rise {rise_mm:.1f} mm, "
            f"{hold_load_detail(stalled_status.meta)}"
        )
        return TearOffReading(
            held=holding,
            rise_mm=rise_mm,
            sag_mm=None,
            could_not_lift=True,
            mass_kg=mass_kg,
            peak_load_n=stalled_status.meta.get("peak_coaxial_load_n"),
            released_load_n=stalled_status.meta.get("released_load_n"),
        )

    after_mm = await _box_pose_mm(world, prop)
    lift_status = await gripper.is_holding_something()
    holding = lift_status.is_holding_something
    rise_mm = (
        lift_delta_mm(before_mm, after_mm)[0]
        if before_mm is not None and after_mm is not None
        else 0.0
    )

    from viam.proto.common import PoseInFrame

    tcp_reply: Any = await motion.get_pose(component_name=gripper.name, destination_frame="world")
    tcp_mm = _full_pose_mm(tcp_reply.pose if isinstance(tcp_reply, PoseInFrame) else tcp_reply)
    top_face_after_mm = await _box_top_face_xyz_mm(world, prop)
    sag_mm = tcp_mm["z"] - top_face_after_mm[2] if top_face_after_mm is not None else None
    sag_str = f"{sag_mm:.1f} mm" if sag_mm is not None else "n/a"
    print(
        f"  {prop} ({mass_kg:.0f} kg): grab=True rise {rise_mm:.1f} mm holding={holding} sag "
        f"{sag_str}, {hold_load_detail(lift_status.meta)}"
    )
    return TearOffReading(
        held=holding,
        rise_mm=rise_mm,
        sag_mm=sag_mm,
        mass_kg=mass_kg,
        peak_load_n=lift_status.meta.get("peak_coaxial_load_n"),
        released_load_n=lift_status.meta.get("released_load_n"),
    )


async def _move_tear_off_prop_onto_pick_spot(
    world: WorldApi, prop: str, target_mm: Mapping[str, float], box_height_mm: float
) -> bool:
    """Teleports `prop` onto the real box's own resting spot and reads back
    whether it settled on the pick station rather than beside it."""
    target_pick_mm = [target_mm["x"], target_mm["y"], target_mm["z"]]
    print(f"  moving stand-in {prop} onto the pick spot {target_pick_mm}")
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": prop,
            "position": target_pick_mm,
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    await asyncio.sleep(RESET_SETTLE_S)
    settled_geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    settled_mm = prop_pose_mm(settled_geometries, prop)
    rest_ok, rest_detail = box_rests_on_support(
        settled_mm, box_height_mm, support_geometry_mm(settled_geometries, "pick-station")
    )
    print(f"  stand-in settled at {settled_mm}, {rest_detail}")
    return rest_ok


async def _stow_tear_off_after_lift(
    world: WorldApi, gripper: Any, prop: str, stow_xyz_mm: tuple[float, float, float]
) -> None:
    """Opens the gripper, lets `prop` drop and settle, and teleports it to
    its stow spot so the next stand-in's lift is not fouled by it."""
    await gripper.open()
    await asyncio.sleep(DROP_SETTLE_S)
    dropped_geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    print(f"  {prop} after the drop: {prop_pose_mm(dropped_geometries, prop)}")
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": prop,
            "position": list(stow_xyz_mm),
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    stow_read_geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    print(f"  {prop} stowed at {prop_pose_mm(stow_read_geometries, prop)}")


async def _check_tear_off(
    world: WorldApi,
    gripper: Any,
    motion: Any,
    arm: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    coaxial_limit_n: float,
) -> tuple[str, bool]:
    """Item 4: three lifts prove the coaxial threshold rather than a cliff.
    The over-limit stand-in (past what the cups can hold at `coaxial_limit_n`,
    `gripper-1`'s own configured per-cup break force) drops on lift, the
    under-limit stand-in (well under it) holds, and the 2 kg box holds too.
    See tear_off_verdict for how the three readings settle a pass.

    Both stand-ins are found or spawned at their own stow spots BEFORE
    anything else moves. A spawn resets the world, which puts every prop
    back at its configured pose, so spawning either one after the 2 kg box
    has been sidestepped undoes the sidestep and lands props on top of
    each other."""
    cups = EPICK["cups"]["names"]
    over_mass_kg = tear_off_mass_kg(cups, coaxial_limit_n)
    under_mass_kg = under_limit_mass_kg(cups, coaxial_limit_n)
    four_cup_hold_kg = len(cups) * coaxial_limit_n / STANDARD_GRAVITY_M_S2
    print(
        f"  coaxial limit {coaxial_limit_n:.1f} N per cup, four cups hold "
        f"{four_cup_hold_kg:.1f} kg, over-limit stand-in {over_mass_kg:.0f} kg, under-limit "
        f"stand-in {under_mass_kg:.0f} kg, window {COAXIAL_LOAD_WINDOW_S:.1f} s"
    )

    # the previous item leaves the arm wherever its last leg ended, on the fourth GPU pass
    # hovering over the pick spot, and a stand-in spawned into the tool
    await _park_arm(world, gripper, motion)
    reset_ok, reset_detail = await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    if not reset_ok:
        line = verdict(
            EPICK_ITEMS[4], False, f"box did not settle on the pick station ({reset_detail})"
        )
        print(line)
        return line, False

    box_pose_mm = await _box_pose_mm(world, box_prop)
    box_dims_mm = await _box_dims_mm(world, box_prop)
    if box_pose_mm is None:
        line = verdict(EPICK_ITEMS[4], False, f"{box_prop!r} has no known geometry in the world")
        print(line)
        return line, False

    await _stow_tear_off_stand_in(
        world, TEAR_OFF_BOX_PROP, over_mass_kg, TEAR_OFF_STOW_XYZ_MM, box_dims_mm
    )
    await _stow_tear_off_stand_in(
        world,
        TEAR_OFF_UNDER_LIMIT_BOX_PROP,
        under_mass_kg,
        TEAR_OFF_UNDER_LIMIT_STOW_XYZ_MM,
        box_dims_mm,
    )

    # clear the real box off its own resting spot before either stand-in moves onto it,
    # along the station's own travel axis. The station runs from y -1200 to -100 mm and the
    # box rests near its +y end, so the sidestep goes toward -y to stay on the deck
    sidestep_target_mm = {
        "x": box_pose_mm["x"],
        "y": box_pose_mm["y"] + TEAR_OFF_SIDESTEP_MM,
        "z": box_pose_mm["z"],
    }
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": box_prop,
            "position": [
                sidestep_target_mm["x"],
                sidestep_target_mm["y"],
                sidestep_target_mm["z"],
            ],
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    await asyncio.sleep(RESET_SETTLE_S)
    sidestep_geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    sidestepped_mm = prop_pose_mm(sidestep_geometries, box_prop)
    sidestep_ok, sidestep_detail = box_rests_on_support(
        sidestepped_mm, box_dims_mm[2], support_geometry_mm(sidestep_geometries, "pick-station")
    )
    print(
        f"  sidestepped {box_prop} to {sidestep_target_mm}: settled at {sidestepped_mm}, "
        f"{sidestep_detail}"
    )
    if not sidestep_ok:
        line = verdict(
            EPICK_ITEMS[4],
            False,
            f"box did not settle off its spot after sidestep ({sidestep_detail})",
        )
        print(line)
        return line, False

    try:
        if not await _move_tear_off_prop_onto_pick_spot(
            world, TEAR_OFF_BOX_PROP, box_pose_mm, box_dims_mm[2]
        ):
            line = verdict(
                EPICK_ITEMS[4], False, "over-limit stand-in did not settle on the pick station"
            )
            print(line)
            return line, False
        over_reading = await _tear_off_lift(
            world, gripper, motion, arm, TEAR_OFF_BOX_PROP, over_mass_kg, "tear-off over-limit"
        )
        if over_reading is None:
            line = verdict(EPICK_ITEMS[4], False, f"{TEAR_OFF_BOX_PROP!r} never spawned")
            print(line)
            return line, False
        await _stow_tear_off_after_lift(world, gripper, TEAR_OFF_BOX_PROP, TEAR_OFF_STOW_XYZ_MM)

        if not await _move_tear_off_prop_onto_pick_spot(
            world, TEAR_OFF_UNDER_LIMIT_BOX_PROP, box_pose_mm, box_dims_mm[2]
        ):
            line = verdict(
                EPICK_ITEMS[4], False, "under-limit stand-in did not settle on the pick station"
            )
            print(line)
            return line, False
        under_reading = await _tear_off_lift(
            world,
            gripper,
            motion,
            arm,
            TEAR_OFF_UNDER_LIMIT_BOX_PROP,
            under_mass_kg,
            "tear-off under-limit",
        )
        if under_reading is None:
            line = verdict(
                EPICK_ITEMS[4], False, f"{TEAR_OFF_UNDER_LIMIT_BOX_PROP!r} never spawned"
            )
            print(line)
            return line, False
        await _stow_tear_off_after_lift(
            world, gripper, TEAR_OFF_UNDER_LIMIT_BOX_PROP, TEAR_OFF_UNDER_LIMIT_STOW_XYZ_MM
        )

        restore_ok, restore_detail = await _reset_pick_box(
            world, box_prop, pick_pose_mm, "pick-station"
        )
        if not restore_ok:
            line = verdict(
                EPICK_ITEMS[4], False, f"could not restore {box_prop!r} ({restore_detail})"
            )
            print(line)
            return line, False

        light_reading = await _tear_off_lift(
            world, gripper, motion, arm, box_prop, 2.0, "tear-off control"
        )
        if light_reading is None:
            line = verdict(
                EPICK_ITEMS[4], False, f"{box_prop!r} has no known geometry in the world"
            )
            print(line)
            return line, False

        ok, detail = tear_off_verdict(over_reading, under_reading, light_reading)
        line = verdict(EPICK_ITEMS[4], ok, detail)
        print(line)
        return line, ok
    finally:
        # each step guarded so one failure never hides the next. A second stow of an
        # already-stowed stand-in, or a second restore of an already-restored box, is
        # harmless, so this runs even when the try block's own cleanup already succeeded
        try:
            await gripper.open()
        except Exception:  # noqa: BLE001 - the stows and restore below still matter
            pass
        try:
            await world.do_command(
                {
                    "command": "set_prop_pose",
                    "name": TEAR_OFF_BOX_PROP,
                    "position": list(TEAR_OFF_STOW_XYZ_MM),
                    "orientation_rpy_deg": [0.0, 0.0, 0.0],
                }
            )
        except Exception:  # noqa: BLE001 - the other stow and the restore below still matter
            pass
        try:
            await world.do_command(
                {
                    "command": "set_prop_pose",
                    "name": TEAR_OFF_UNDER_LIMIT_BOX_PROP,
                    "position": list(TEAR_OFF_UNDER_LIMIT_STOW_XYZ_MM),
                    "orientation_rpy_deg": [0.0, 0.0, 0.0],
                }
            )
        except Exception:  # noqa: BLE001 - the box restore below still matters
            pass
        try:
            await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
        except Exception:  # noqa: BLE001 - report the item's own failure, not the cleanup's
            pass


async def _check_place_descent_branches(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    sequencer: SequencerClient,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
) -> tuple[str, bool]:
    """Item 5: the place descent reproduced from each of the two arm
    configurations the 2026-09-22 stall investigation recorded
    (`PLACE_DESCENT_SEED_JOINTS_DEG`).

    The seed joint move happens before the box is ever picked, with the
    gripper empty, so it only leans the planner toward a branch rather than
    swinging a held box through the 150 degrees of wrist travel the
    2026-09-22 run measured between the two seeds. Every later move is a
    planned Cartesian move, either `arm.move_to_position` or a linear
    descent through the motion service, never a second joint teleport. The
    branch actually reached at the standoff and at the descent start is read
    back and reported rather than assumed to match the seed."""
    from viam.proto.common import Pose
    from viam.proto.component.arm import JointPositions

    all_ok = True
    details: list[str] = []
    for branch_name, seed_joints_deg in PLACE_DESCENT_SEED_JOINTS_DEG.items():
        reset_ok, reset_detail = await _reset_pick_box(
            world, box_prop, pick_pose_mm, "pick-station"
        )
        if not reset_ok:
            all_ok = False
            details.append(
                f"{branch_name}: box did not settle on the pick station ({reset_detail})"
            )
            continue
        await gripper.open()

        try:
            print(f"  {branch_name}: seed joints (deg) {[round(v, 2) for v in seed_joints_deg]}")
            await arm.move_to_joint_positions(JointPositions(values=list(seed_joints_deg)))
            seeded_joints = await arm.get_joint_positions()
            print(
                f"  {branch_name}: seeded joints (deg): "
                f"{[round(v, 2) for v in seeded_joints.values]}"
            )

            top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
            if top_face_xyz_mm is None:
                all_ok = False
                details.append(f"{branch_name}: {box_prop!r} has no known geometry in the world")
                continue

            standoff, grasp = _pick_grasp_poses(top_face_xyz_mm)
            pick_state = await _cell_world_state(world, {box_prop})

            print(f"  {branch_name}: standoff (mm) {_pose_to_mm(standoff)}")
            await arm.move_to_position(standoff)
            at_standoff_joints = await arm.get_joint_positions()
            branch_at_standoff = branch_of(at_standoff_joints.values)
            print(
                f"  {branch_name}: at standoff, joints (deg): "
                f"{[round(v, 2) for v in at_standoff_joints.values]}, branch "
                f"{branch_at_standoff} "
                f"({'matches' if branch_at_standoff == branch_name else 'does not match'} "
                "the seed)"
            )

            print(f"  {branch_name}: grasp (mm) {_pose_to_mm(grasp)}")
            await _move(
                gripper, motion, grasp, pick_state, linear=True, leg=f"{branch_name} descent"
            )
            if not await gripper.grab():
                all_ok = False
                details.append(f"{branch_name}: grab() returned False at the pick pose")
                continue

            await _move(
                gripper, motion, standoff, pick_state, linear=True, leg=f"{branch_name} lift"
            )
            branch_lift_status = await gripper.is_holding_something()
            held_after_lift = branch_lift_status.is_holding_something
            print(
                f"  held after {branch_name} lift: {held_after_lift}, "
                f"{hold_load_detail(branch_lift_status.meta)}"
            )
            if not held_after_lift:
                box_mm = await _box_pose_mm(world, box_prop)
                print(f"  box pose: {box_mm}")
                all_ok = False
                details.append(hold_lost_detail(f"{branch_name} lift", box_mm))
                continue

            await sequencer.reset_cursor()
            next_box = await sequencer.next_box()
            if next_box.is_complete or next_box.place_end_in_world is None:
                all_ok = False
                details.append(f"{branch_name}: the sequencer has no next slot")
                continue
            place_end = next_box.place_end_in_world
            release_pose = place_release_pose(place_end)
            descent_start = Pose(
                x=release_pose.x,
                y=release_pose.y,
                z=release_pose.z + DESCENT_START_RAISE_MM,
                o_x=release_pose.o_x,
                o_y=release_pose.o_y,
                o_z=release_pose.o_z,
                theta=release_pose.theta,
            )
            print(f"  {branch_name}: descent start (mm) {_pose_to_mm(descent_start)}")
            await arm.move_to_position(descent_start)
            confirm_joints = await arm.get_joint_positions()
            branch_at_descent_start = branch_of(confirm_joints.values)
            print(
                f"  {branch_name}: at descent start, joints (deg): "
                f"{[round(v, 2) for v in confirm_joints.values]}, branch "
                f"{branch_at_descent_start}"
            )
            descent_start_status = await gripper.is_holding_something()
            held_after_descent_start = descent_start_status.is_holding_something
            print(
                f"  held after {branch_name} descent start move: {held_after_descent_start}, "
                f"{hold_load_detail(descent_start_status.meta)}"
            )
            if not held_after_descent_start:
                box_mm = await _box_pose_mm(world, box_prop)
                print(f"  box pose: {box_mm}")
                all_ok = False
                details.append(hold_lost_detail(f"{branch_name} descent start move", box_mm))
                continue

            box_dims_mm = await _box_dims_mm(world, box_prop)
            held = held_box_transform(box_prop, box_dims_mm, gripper.name)
            descent_state = await _cell_world_state(world, {box_prop}, held)

            samples: list[list[float]] = []

            async def _sample_joints(into: list[list[float]] = samples) -> None:
                while True:
                    current = await arm.get_joint_positions()
                    into.append(list(current.values))
                    await asyncio.sleep(TRAJECTORY_SAMPLE_INTERVAL_S)

            async def _place_descent_move(
                pose: Any, linear: bool, state: Any = descent_state, branch: str = branch_name
            ) -> bool:
                await _move(
                    gripper,
                    motion,
                    pose,
                    state,
                    linear=linear,
                    leg=f"{branch} place descent",
                )
                return True

            print(f"  {branch_name}: place release (mm) {_pose_to_mm(release_pose)}")
            sampler = asyncio.create_task(_sample_joints())
            try:
                await move_linear_or_free(
                    _place_descent_move, release_pose, f"{branch_name} place descent"
                )
            finally:
                sampler.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sampler

            branch_descent_status = await gripper.is_holding_something()
            held_after_descent = branch_descent_status.is_holding_something
            print(
                f"  held after {branch_name} descent: {held_after_descent}, "
                f"{hold_load_detail(branch_descent_status.meta)}"
            )
            if not held_after_descent:
                box_mm = await _box_pose_mm(world, box_prop)
                print(f"  box pose: {box_mm}")
                all_ok = False
                details.append(hold_lost_detail(f"{branch_name} descent", box_mm))
                continue

            await gripper.open()
            await asyncio.sleep(PLACE_SETTLE_S)

            landed_mm = await _box_pose_mm(world, box_prop)
            slot_xy_mm = {"x": place_end.x, "y": place_end.y}
            placed_ok, place_error_mm = placement_check(landed_mm, slot_xy_mm)
            fold_deg = wrist_fold_deg(samples) if samples else 0.0
            all_ok = all_ok and placed_ok
            details.append(
                f"{branch_report(branch_name, branch_at_standoff, branch_at_descent_start)}, "
                f"place error {place_error_mm:.2f} mm, wrist fold {fold_deg:.2f} deg"
            )
            print(f"  {details[-1]}")
        finally:
            try:
                await gripper.open()
            except Exception:  # noqa: BLE001 - the box restore below still matters if this failed
                pass
            try:
                await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
            except Exception:  # noqa: BLE001 - report the branch's own result, not the cleanup's
                pass

    line = verdict(EPICK_ITEMS[5], all_ok, "; ".join(details))
    print(line)
    return line, all_ok


async def _check_release_at_contact(
    world: WorldApi,
    gripper: Any,
    motion: Any,
    sequencer: SequencerClient,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
) -> tuple[str, bool]:
    """Item 6: descend to `place_end` itself, not the release pose, catching
    a stall against the deck the way `_descend_to_release` does, then read
    how far the box's bottom actually sat above the deck at that moment
    against the 25 mm the service releases at."""
    from viam.proto.common import PoseInFrame

    reset_ok, reset_detail = await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    if not reset_ok:
        line = verdict(
            EPICK_ITEMS[6], False, f"box did not settle on the pick station ({reset_detail})"
        )
        print(line)
        return line, False

    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(EPICK_ITEMS[6], False, f"{box_prop!r} has no known geometry in the world")
        print(line)
        return line, False
    standoff = pick_grasp_standoff_pose(top_face_xyz_mm)
    grasp = pick_grasp_pose(top_face_xyz_mm)
    pick_state = await _cell_world_state(world, {box_prop})

    async def approach(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, pick_state, linear=linear, leg="release approach")
        return True

    async def descend(pose: Any, linear: bool) -> bool:
        await _move(gripper, motion, pose, pick_state, linear=linear, leg="release descent")
        return True

    await approach_and_descend(approach, descend, standoff, grasp)
    if not await gripper.grab():
        line = verdict(EPICK_ITEMS[6], False, "grab() returned False at the pick pose")
        print(line)
        return line, False

    try:
        await _move(gripper, motion, standoff, pick_state, linear=True, leg="release lift")
        release_lift_status = await gripper.is_holding_something()
        held_after_lift = release_lift_status.is_holding_something
        print(
            f"  held after release lift: {held_after_lift}, "
            f"{hold_load_detail(release_lift_status.meta)}"
        )
        if not held_after_lift:
            box_mm = await _box_pose_mm(world, box_prop)
            print(f"  box pose: {box_mm}")
            line = verdict(EPICK_ITEMS[6], False, hold_lost_detail("release lift", box_mm))
            print(line)
            return line, False

        await sequencer.reset_cursor()
        next_box = await sequencer.next_box()
        if next_box.is_complete or next_box.place_end_in_world is None:
            line = verdict(EPICK_ITEMS[6], False, "the sequencer has no next slot")
            print(line)
            return line, False
        place_end = next_box.place_end_in_world

        box_dims_mm = await _box_dims_mm(world, box_prop)
        held = held_box_transform(box_prop, box_dims_mm, gripper.name)
        place_state = await _cell_world_state(world, {box_prop}, held)

        async def to_deck(pose: Any, linear: bool) -> bool:
            await _move(
                gripper, motion, pose, place_state, linear=linear, leg="release descent to deck"
            )
            return True

        try:
            await move_linear_or_free(to_deck, place_end, "release descent")
        except Exception as error:
            if not is_execution_failure(error):
                raise
            arrived: Any = await motion.get_pose(
                component_name=gripper.name, destination_frame="world"
            )
            cup_pose = arrived.pose if isinstance(arrived, PoseInFrame) else arrived
            if not touched_down(cup_pose, place_end):
                line = verdict(
                    EPICK_ITEMS[6], False, f"place descent stalled away from the deck: {error}"
                )
                print(line)
                return line, False
            print(
                f"  stalled at {_pose_to_mm(cup_pose)}, within {PLACE_STALL_TOLERANCE_MM:.0f} mm "
                "of the deck: box is down"
            )

        release_descent_status = await gripper.is_holding_something()
        held_after_descent = release_descent_status.is_holding_something
        print(
            f"  held after release descent to deck: {held_after_descent}, "
            f"{hold_load_detail(release_descent_status.meta)}"
        )

        box_pose_at_release_mm = await _box_pose_mm(world, box_prop)
        if held_after_descent:
            geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
            support = support_geometry_mm(geometries, "pallet")
            if support is not None and box_pose_at_release_mm is not None:
                clearance_mm = (
                    float(box_pose_at_release_mm["z"])
                    - box_dims_mm[2] / 2.0
                    - support_top_z_mm(support)
                )
                print(
                    f"  release clearance candidate: {clearance_mm:.2f} mm (service uses "
                    f"{PLACE_RELEASE_CLEARANCE_MM:.1f} mm)"
                )
        else:
            box_mm = box_pose_at_release_mm
            print(f"  box pose: {box_mm}")
            line = verdict(
                EPICK_ITEMS[6], False, hold_lost_detail("release descent to deck", box_mm)
            )
            print(line)
            return line, False

        await gripper.open()
        await asyncio.sleep(PLACE_SETTLE_S)

        landed_mm = await _box_pose_mm(world, box_prop)
        slot_xy_mm = {"x": place_end.x, "y": place_end.y}
        placed_ok, place_error_mm = placement_check(landed_mm, slot_xy_mm)
        if landed_mm is not None:
            box_tilt_deg = orientation_delta_deg({}, landed_mm)
            tilt_ok = box_tilt_deg < RELEASE_TILT_TOLERANCE_DEG
        else:
            box_tilt_deg = math.inf
            tilt_ok = False

        ok = placed_ok and tilt_ok
        detail = (
            f"place error {place_error_mm:.2f} mm (tolerance {PLACEMENT_TOLERANCE_MM:.0f}), tilt "
            f"{box_tilt_deg:.2f} deg (tolerance {RELEASE_TILT_TOLERANCE_DEG:.0f})"
        )
        line = verdict(EPICK_ITEMS[6], ok, detail)
        print(line)
        return line, ok
    finally:
        try:
            await gripper.open()
        except Exception:  # noqa: BLE001 - the box restore below still matters if opening failed
            pass
        try:
            await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
        except Exception:  # noqa: BLE001 - report the item's own failure, not the cleanup's
            pass


async def _guarded_if_listed(
    idx: int,
    item: str,
    items: frozenset[int] | None,
    check: Callable[[], Awaitable[tuple[str, bool]]],
) -> tuple[str, bool] | None:
    """Runs item `idx` through `_guarded`, or skips it with no verdict line
    printed at all when `items` is given and does not list `idx`."""
    if items is not None and idx not in items:
        return None
    return await _guarded(item, check)


async def _run_epick(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    sequencer: SequencerClient,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    place_pose_mm: Mapping[str, float],
    run_label: str,
    grab_delay_ms: float,
    retry_interval_s: float,
    coaxial_limit_n: float,
    items: frozenset[int] | None = None,
) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []

    await gripper.open()
    try:
        await _park_arm(world, gripper, motion)
    except Exception as exc:  # noqa: BLE001 - a park that fails leaves the items to say what it cost
        print(f"  arm could not be parked before the run: {exc!r}")

    print("\n-- smoke (cited) --")
    result = await _guarded_if_listed(0, EPICK_ITEMS[0], items, _check_epick_smoke_cited)
    if result is not None:
        results.append(result)

    print("\n-- EPick render, kinematics, collision reach --")
    result = await _guarded_if_listed(
        1,
        EPICK_ITEMS[1],
        items,
        lambda: _check_epick_render_and_kinematics(world, arm, gripper, motion),
    )
    if result is not None:
        results.append(result)

    print("\n-- gating --")
    result = await _guarded_if_listed(
        2,
        EPICK_ITEMS[2],
        items,
        lambda: _check_gating(
            world, gripper, motion, box_prop, pick_pose_mm, grab_delay_ms, retry_interval_s
        ),
    )
    if result is not None:
        results.append(result)

    print("\n-- swing --")
    result = await _guarded_if_listed(
        3,
        EPICK_ITEMS[3],
        items,
        lambda: _check_swing(world, gripper, motion, box_prop, pick_pose_mm, place_pose_mm),
    )
    if result is not None:
        results.append(result)

    print("\n-- tear-off --")
    result = await _guarded_if_listed(
        4,
        EPICK_ITEMS[4],
        items,
        lambda: _check_tear_off(
            world, gripper, motion, arm, box_prop, pick_pose_mm, coaxial_limit_n
        ),
    )
    if result is not None:
        results.append(result)

    print("\n-- place descent branches --")
    result = await _guarded_if_listed(
        5,
        EPICK_ITEMS[5],
        items,
        lambda: _check_place_descent_branches(
            world, arm, gripper, motion, sequencer, box_prop, pick_pose_mm
        ),
    )
    if result is not None:
        results.append(result)

    print("\n-- release at contact --")
    result = await _guarded_if_listed(
        6,
        EPICK_ITEMS[6],
        items,
        lambda: _check_release_at_contact(
            world, gripper, motion, sequencer, box_prop, pick_pose_mm
        ),
    )
    if result is not None:
        results.append(result)

    print("\n-- cost vs the workcell suite --")
    result = await _guarded_if_listed(
        7, EPICK_ITEMS[7], items, lambda: _check_cost(world, run_label, item=EPICK_ITEMS[7])
    )
    if result is not None:
        results.append(result)

    print("\n== summary ==")
    for line, _ in results:
        print(line)
    return results


async def _run_first_box(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    palletizer: Any,
    sequencer: SequencerClient,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    run_label: str,
) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    # the box is the payload, not an obstacle: the cup has to descend onto it
    state = await _cell_world_state(world, {box_prop})

    print("\n-- reach --")
    results.append(
        await _check_reach(
            world, arm, gripper, motion, box_prop, state, await _cell_world_state(world, set())
        )
    )

    print("\n-- grab, carry, release --")
    results.append(await _check_grab_carry_release(world, gripper, arm, motion, box_prop, state))

    print("\n-- placement (box-palletizer service) --")
    results.append(await _check_placement(world, palletizer, sequencer, arm, gripper, box_prop))

    print("\n-- grab with nothing under the tool --")
    results.append(
        await _check_grab_nothing(world, gripper, arm, motion, box_prop, pick_pose_mm, state)
    )

    print("\n-- cost --")
    results.append(await _check_cost(world, run_label))

    print("\n== summary ==")
    for line, _ in results:
        print(line)
    return results


@dataclass
class Args:
    address: str
    api_key: str | None
    api_key_id: str | None
    world: str
    arm: str
    gripper: str
    motion: str
    palletizer: str
    sequencer: str
    # None only under the pack suite, which reads the infeed pose and every
    # place target off box-palletizer's own status records instead.
    # _parse_args rejects a missing one under the other three suites.
    box_prop: str | None
    pick_x_mm: float | None
    pick_y_mm: float | None
    pick_z_mm: float | None
    place_x_mm: float | None
    place_y_mm: float | None
    suite: str
    run_label: str
    obstacle_source_label: str
    fragment: str
    grab_delay_ms: float
    retry_interval_s: float
    coaxial_limit_n: float
    items: frozenset[int] | None


def _parse_items(value: str) -> frozenset[int]:
    """Parses `--items`' comma-separated item numbers into the set `_run_epick`
    filters against. Raises on anything that is not an integer, so argparse
    reports it as a usage error rather than a silent empty selection."""
    try:
        return frozenset(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid --items value {value!r}: {error}") from error


def _parse_args(argv: Sequence[str] | None) -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--address", help="machine address (required)")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--world", default="isaac-world", help="the isaac-sim world component name")
    parser.add_argument("--arm", default="arm-1", help="the arm component name")
    parser.add_argument("--gripper", default="gripper-1", help="the vacuum gripper component name")
    parser.add_argument("--motion", default="builtin", help="the motion service name")
    parser.add_argument(
        "--palletizer", default="box-palletizer", help="the box-palletizer service name"
    )
    parser.add_argument(
        "--sequencer",
        default="pack-sequencer",
        help="the viam:pack-sequencer:sequencer service name (box-palletizer's own "
        "'sequencer' attribute), the pack suite only",
    )
    parser.add_argument(
        "--box-prop",
        help="prop name of the single box driven directly for the first-box, workcell and "
        "epick suites' grab-mechanism items. The pack suite drives all of box-palletizer's own "
        "'box_props' list instead, and reads them off the service's own status records rather "
        "than this flag",
    )
    parser.add_argument("--pick-x-mm", type=float, help="the box's known resting centre, x in mm")
    parser.add_argument("--pick-y-mm", type=float, help="the box's known resting centre, y in mm")
    parser.add_argument("--pick-z-mm", type=float, help="the box's known resting centre, z in mm")
    parser.add_argument(
        "--place-x-mm",
        type=float,
        help="the place target x in mm for the first-box, workcell and epick suites' own "
        "single-box place test, unrelated to box-palletizer's config (it has no "
        "place_pose_mm attribute; the sequencer owns every place target for the pack suite)",
    )
    parser.add_argument(
        "--place-y-mm",
        type=float,
        help="the place target y in mm for the first-box, workcell and epick suites' own "
        "single-box place test, unrelated to box-palletizer's config (it has no "
        "place_pose_mm attribute; the sequencer owns every place target for the pack suite)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=("first-box", "workcell", "epick", "pack"),
        default="first-box",
        help="which suite's checklist items to run (default: first-box)",
    )
    parser.add_argument(
        "--grab-delay-ms",
        type=float,
        default=250.0,
        help="the epick suite item 2's expected grab delay, the overlay's own value "
        "(default: 250.0)",
    )
    parser.add_argument(
        "--retry-interval-s",
        type=float,
        default=2.0,
        help="the epick suite item 2's automatic-mode retry window, the gripper's own value "
        "(default: 2.0)",
    )
    parser.add_argument(
        "--coaxial-limit-n",
        type=float,
        default=DEFAULT_COAXIAL_FORCE_LIMIT_N,
        help="the per-cup coaxial break force gripper-1 is configured with, so the tear-off "
        f"item's stand-ins are sized against it (default: {DEFAULT_COAXIAL_FORCE_LIMIT_N})",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        choices=("cold", "warm"),
        help="tags the cost item's ready/step-rate reading so a cold and a warm reading can't "
        "be conflated. Run the script twice, once per label: right after a module restart with "
        "cold, then again without restarting with warm",
    )
    parser.add_argument(
        "--obstacle-source-label",
        default="world_state_store",
        choices=("world_state_store", "prop_geometries"),
        help="tags the pack suite item 3's obstacle-source reading with whichever value the "
        "overlay's box-palletizer.obstacle_source is currently deployed with. Run the script "
        "once per value, redeploying the overlay between runs, and compare",
    )
    parser.add_argument(
        "--fragment",
        default=str(
            Path(__file__).resolve().parent.parent / "fragments" / "isaac-sim-palletizing.json"
        ),
        help="path to the vendored fragment JSON, for the workcell suite item 1's declared "
        "component frames",
    )
    parser.add_argument(
        "--items",
        type=_parse_items,
        default=None,
        help="comma-separated item numbers to run, the epick suite only (default: every item "
        "in the suite). An item not listed is skipped with no verdict line",
    )
    ns = parser.parse_args(argv)
    # The first-box, workcell and epick suites drive the box directly and need
    # its resting pose and a place target. The pack suite reads both off
    # box-palletizer's own status records, so requiring them there would mean
    # typing numbers the run ignores, which is how an invented value ends up
    # looking like a measurement.
    if ns.suite in ("first-box", "workcell", "epick"):
        missing = [
            name
            for name, value in (
                ("--box-prop", ns.box_prop),
                ("--pick-x-mm", ns.pick_x_mm),
                ("--pick-y-mm", ns.pick_y_mm),
                ("--pick-z-mm", ns.pick_z_mm),
                ("--place-x-mm", ns.place_x_mm),
                ("--place-y-mm", ns.place_y_mm),
            )
            if value is None
        ]
        if missing:
            parser.error(f"--suite {ns.suite} requires {', '.join(missing)}")
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        world=ns.world,
        arm=ns.arm,
        gripper=ns.gripper,
        motion=ns.motion,
        palletizer=ns.palletizer,
        sequencer=ns.sequencer,
        box_prop=ns.box_prop,
        pick_x_mm=ns.pick_x_mm,
        pick_y_mm=ns.pick_y_mm,
        pick_z_mm=ns.pick_z_mm,
        place_x_mm=ns.place_x_mm,
        place_y_mm=ns.place_y_mm,
        suite=ns.suite,
        run_label=ns.run_label,
        obstacle_source_label=ns.obstacle_source_label,
        fragment=ns.fragment,
        grab_delay_ms=ns.grab_delay_ms,
        retry_interval_s=ns.retry_interval_s,
        coaxial_limit_n=ns.coaxial_limit_n,
        items=ns.items,
    )


def _single_box_args(args: Args) -> tuple[str, dict[str, float], dict[str, float]]:
    """The first-box, workcell and epick suites' directly-driven box: its
    prop name, its known resting centre and the place target. ``_parse_args``
    rejects a missing one under those suites, so this narrows the types
    rather than re-checking a case a caller can reach."""
    if (
        args.box_prop is None
        or args.pick_x_mm is None
        or args.pick_y_mm is None
        or args.pick_z_mm is None
        or args.place_x_mm is None
        or args.place_y_mm is None
    ):
        raise ValueError(f"--suite {args.suite} needs the box prop and the pick and place poses")
    return (
        args.box_prop,
        {"x": args.pick_x_mm, "y": args.pick_y_mm, "z": args.pick_z_mm},
        {"x": args.place_x_mm, "y": args.place_y_mm},
    )


async def _run_real(args: Args) -> None:
    from viam.components.arm import Arm
    from viam.components.generic import Generic
    from viam.components.gripper import Gripper
    from viam.robot.client import RobotClient

    # the world is a generic COMPONENT and the palletizer a generic SERVICE;
    # they are different clients with the same name
    from viam.services.generic import Generic as GenericService
    from viam.services.motion import MotionClient
    from viam.services.worldstatestore import WorldStateStore

    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()

    robot = await RobotClient.at_address(args.address, opts)
    try:
        world = Generic.from_robot(robot, args.world)
        arm = Arm.from_robot(robot, args.arm)
        gripper = Gripper.from_robot(robot, args.gripper)
        motion = MotionClient.from_robot(robot, args.motion)
        palletizer = GenericService.from_robot(robot, args.palletizer)
        # the sequencer is an rdk:service:world_state_store, not a generic
        # service, so it is reached through that API's own client. Asking the
        # generic API for it answers ResourceNotFoundError, which is what ended
        # the 15:12 run before its first item and what the pack suite's own
        # path, never yet run, would have hit too.
        sequencer = SequencerClient(WorldStateStore.from_robot(robot, args.sequencer))
        if args.suite == "pack":
            await _run_pack(
                world,
                palletizer,
                sequencer,
                args.run_label,
                args.obstacle_source_label,
            )
            return
        # the place target the flags name is not used by the first-box or
        # workcell suites' own checks any more: the sequencer owns the slot,
        # and item 4 judges against its record. The epick suite's swing item
        # still wants a rough direction to carry the box across the cell, so
        # it reads the same flag.
        box_prop, pick_pose_mm, place_xy_mm = _single_box_args(args)
        if args.suite == "workcell":
            await _run_workcell(
                world,
                arm,
                gripper,
                motion,
                palletizer,
                sequencer,
                box_prop,
                pick_pose_mm,
                args.run_label,
                robot,
                Path(args.fragment),
            )
        elif args.suite == "epick":
            await _run_epick(
                world,
                arm,
                gripper,
                motion,
                sequencer,
                box_prop,
                pick_pose_mm,
                place_xy_mm,
                args.run_label,
                args.grab_delay_ms,
                args.retry_interval_s,
                args.coaxial_limit_n,
                args.items,
            )
        else:
            await _run_first_box(
                world,
                arm,
                gripper,
                motion,
                palletizer,
                sequencer,
                box_prop,
                pick_pose_mm,
                args.run_label,
            )
    finally:
        await robot.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.address:
        print("FAILED: --address is required")
        return 1
    try:
        asyncio.run(_run_real(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure as a clean exit code
        print(f"FAILED: {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
