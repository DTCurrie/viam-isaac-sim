"""GPU acceptance checklist for the palletizing cell (phase 1: vacuum gripper,
one box picked and placed by the box-palletizer service; phase 2: the
vendored `viam:workcell-components` workcell adopted under the same
mechanism). Pass `--phase 1` (default) or `--phase 2`.

Connects to a running Viam machine (the module running on the Isaac GPU box):
`arm-1`, `gripper-1`, `builtin` motion, and the `box-palletizer`
service. Item 3 is the phase's own done-when, so it drives the actual pick
and place through `box-palletizer`'s DoCommand rather than re-driving the
arm and gripper itself. Items 1, 2 and 4 drive the arm and gripper directly
to test the underlying grab mechanism, reading the box's pose from the world
at run time rather than the service's own arithmetic, so a green result there
can't just be agreement with the code under test.

The box's known resting pose before a run and the target place position are
facts about the deployed cell, the same facts `box-palletizer`'s own
`place_pose_mm` attribute is configured with, so they come in as CLI
arguments rather than an invented constant in this file.

Prints PASS/FAIL and the raw numbers for each item.

Phase-1 checklist items:

1. the vacuum tool renders on the wrist and the arm reaches the box
2. `grab` attaches, the box rides the tool through a carry, and release drops it
3. the box lands within tolerance of the place position and is still there after 5 s
4. a grab attempted with nothing under the tool reports not holding rather
   than hanging
5. cost: cold and warm `ready` and the 10 s step rate

Phase-2 checklist items, the cell having been adopted from Viam's own
palletizer workcell:

1. every component in the fragment renders at the pose its frame declares,
   compared against `get_pose` per component
2. the pick station, pallet and pedestal stop a dropped box, and the box
   rests on each at the height the component's geometry implies
3. the arm cannot plan through the fences or the scan tunnel, shown by a
   plan that detours rather than intersecting
4. phase 1's pick and place runs end to end in the new cell
5. whether `scan-tunnel` needs a derived collider the way `robot-pedestal`
   does, answered by driving the arm at it
6. cost: cold and warm `ready` and the 10 s step rate against phase 1's cell
7. implementation risk, not a plan requirement: render-only scenery has no
   collider proven on hardware

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
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# python examples/gpu_checklist_palletizer.py (standalone, no PYTHONPATH set) needs the
# repo's src/ on sys.path before isaac_module is importable. pytest already adds
# it (pyproject pythonpath = ["src"]), so this is a no-op there.
try:
    import isaac_module  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu_checklist_world import WorldApi, prop_pose_mm, run_step_rate_measurement

from isaac_module.models.palletizer import CUP_APPROACH_GAP_MM
from isaac_module.sort_plan import OUTCOME_PLACED

PHASE_1_ITEMS: tuple[str, ...] = (
    "1. the vacuum tool renders on the wrist and the arm reaches the box",
    "2. `grab` attaches, the box rides the tool through a carry, and release drops it",
    "3. the box lands within tolerance of the place position and is still there after 5 s",
    "4. a grab attempted with nothing under the tool reports not holding rather than hanging",
    "5. cost: cold and warm `ready` and the 10 s step rate",
)

# phase 2's checklist items. Item 7 is not one of them: it verifies an
# implementation risk
# (the collision:False strip sim_manager.py applies to a render-only prop is proven in mock only,
# never on hardware, because omni and pxr are not importable in this environment), not a plan
# requirement, and is labelled as such in its own text.
PHASE_2_ITEMS: tuple[str, ...] = (
    "1. every component in the fragment renders at the pose its frame declares, compared "
    "against `get_pose` per component",
    "2. the pick station, pallet and pedestal stop a dropped box, and the box rests on each "
    "at the height the component's geometry implies",
    "3. the arm cannot plan through the fences or the scan tunnel, shown by a plan that "
    "detours rather than intersecting",
    "4. phase 1's pick and place runs end to end in the new cell",
    "5. whether `scan-tunnel` needs a derived collider the way `robot-pedestal` does, "
    "answered by driving the arm at it",
    "6. cost: cold and warm `ready` and the 10 s step rate against phase 1's cell, since "
    "this one has far more geometry",
    "7. [implementation risk, not a plan requirement] render-only scenery has no collider "
    "proven on hardware: drop a box onto `caution-tape` and show it passes through to the floor",
)

# item 1's tolerance between a fragment component's declared world-frame position and its
# own `get_pose` reply: same translation budget as REACH_TOLERANCE_MM, a prim-pose/viam-pose
# comparison
FRAME_POSE_TOLERANCE_MM = 1.0

# item 2's and item 7's tolerance between a dropped box's expected resting height (derived from
# the support's own geometry, never a constant) and its measured resting height. Carried from
# infeed_box's contact_offset in examples/configs/sim-palletizer-cell.json (5 mm), the same
# settle slack the box's own contact physics allows
RESTING_HEIGHT_TOLERANCE_MM = 5.0

# item 2's and item 7's clearance above a support's derived top face before the box free-falls,
# and how long to wait for it to settle. Shorter than item 3 (phase 1)'s SETTLE_WINDOW_S since a
# freshly dropped box has nowhere to drift once it has landed
DROP_HOVER_MM = 200.0
DROP_SETTLE_S = 2.0

# item 6's phase-1 baseline, from that phase's recorded GPU run of
# 2026-09-15: `sim_time_ratio` 0.563 over a 10 s window,
# on isaac-sim-devin-2. Cited rather than re-measured, since phase 1's own cell was deleted this
# phase. Phase 1's ready_time_s reads 0.0 both cold and warm because the checklist connects after
# the finalizer has already signalled ready, so PLAN.md's "Carried in from phase 1" section names
# it a measurement gap rather than a result. It is not a baseline: this phase's own ready_time_s
# is printed as a first reading instead, to become phase 3's baseline
PHASE_1_BASELINE_SIM_TIME_RATIO = 0.563

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

# item 5, same shape as gpu_checklist_photoreal.py's cost item
DEFAULT_STEP_RATE_WINDOW_S = 10.0
DEFAULT_READY_POLL_S = 0.5
DEFAULT_READY_TIMEOUT_S = 600.0

# item 3's bounded wait on the box-palletizer service's own run: a start that never
# leaves "running" is a failure, not a hang. One pick and place is a handful of
# straight-line motion-service moves, so two minutes is generous
STATUS_POLL_S = 0.5
STATUS_TIMEOUT_S = 120.0


# pure helpers, unit-tested without a robot in tests/test_gpu_checklist_palletizer.py


def verdict(name: str, ok: bool, detail: str) -> str:
    """Format one checklist line: "[PASS|FAIL] name: detail"."""
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {name}: {detail}"


def pose_delta_mm(pose_a: Mapping[str, float], pose_b: Mapping[str, float]) -> float:
    """Euclidean distance in mm between two x/y/z pose mappings, the same
    shape gpu_checklist_world's prop_pose_mm returns."""
    return math.sqrt(sum((pose_a[axis] - pose_b[axis]) ** 2 for axis in ("x", "y", "z")))


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
    """Every fragment component's declared world-frame position in mm, keyed
    by component name.

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
        translation = frame.get("translation") or {}
        frames[str(component["name"])] = {
            "x": float(translation.get("x", 0.0)),
            "y": float(translation.get("y", 0.0)),
            "z": float(translation.get("z", 0.0)),
        }
    return frames


def support_geometry_mm(
    geometries: Sequence[Mapping[str, Any]], component: str
) -> Mapping[str, Any] | None:
    """The first `prop_geometries` entry that belongs to `component`, named
    ``{component}-{label}`` by `workcell_scenery.scenery_props`. `None` when
    the component spawned no collider or render geometry with that prefix."""
    prefix = f"{component}-"
    for geometry in geometries:
        if str(geometry.get("name", "")).startswith(prefix):
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


# async drivers, not unit-tested (need a live world/arm/gripper/motion)


def _pose_to_mm(pose: Any) -> dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "z": float(pose.z)}


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


async def _cell_world_state(world: WorldApi, exclude: set[str]) -> Any:
    """The planner's obstacles, built the same way the palletizer service
    builds them. Without these the planner routes the arm straight through the
    tables it is standing on, and physics stops it where the plan did not."""
    from pickcell.obstacles import obstacles_from_prop_geometries, support_obstacle, world_state

    response = await world.do_command({"command": "prop_geometries"})
    geometries = response.get("geometries", [])
    obstacles = obstacles_from_prop_geometries(geometries, exclude)
    return world_state(None, obstacles, support_obstacle(FLOOR_Z_MM))


async def _move(gripper: Any, motion: Any, pose: Any, state: Any, linear: bool = False) -> None:
    """Drives the GRIPPER's frame through `pickcell.movers.RealMover`, the
    same mover the palletizer service uses.

    Two things that mover already gets right and a hand-rolled motion call
    does not. It plans the gripper's frame, so the cup lands on the target
    instead of the flange landing there and driving the tool a tool-length
    into whatever is below. And `linear=True` carries a linear constraint, so
    a short descent or lift stays in the arm's current configuration instead
    of being replanned into a different inverse-kinematics branch, which turns
    a 100 mm lift into a full elbow flip through the table."""
    from pickcell.movers import RealMover

    # this cell never calls look_from, so the mover's camera frame is unused
    mover = RealMover(motion, gripper.name, gripper.name)
    await mover.move_to(pose, state, linear=linear)


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
    world: WorldApi, arm: Any, gripper: Any, motion: Any, box_prop: str, state: Any
) -> tuple[str, bool]:
    from viam.proto.common import PoseInFrame

    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(PHASE_1_ITEMS[0], False, f"{box_prop!r} has no known geometry in the world")
        print(line)
        return line, False

    standoff, target = _pick_grasp_poses(top_face_xyz_mm)
    await _move(gripper, motion, standoff, state)
    await _move(gripper, motion, target, state, linear=True)
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
    print(f"  target (mm): {_pose_to_mm(target)}")
    print(f"  arrived (mm): {_pose_to_mm(end_position)}")
    line = verdict(
        PHASE_1_ITEMS[0],
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
        line = verdict(PHASE_1_ITEMS[1], False, f"{box_prop!r} has no known geometry in the world")
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

    if before_lift_mm is None or after_lift_mm is None:
        rode_ok, vertical_mm, horizontal_mm = False, 0.0, 0.0
    else:
        rode_ok, vertical_mm, horizontal_mm = rode_with_tool(before_lift_mm, after_lift_mm)

    await gripper.open()
    holding_status = await gripper.is_holding_something()
    released_ok = not holding_status.is_holding_something

    ok = bool(grabbed) and rode_ok and released_ok
    detail = (
        f"grabbed={grabbed} lift delta {vertical_mm:.1f} mm vertical / {horizontal_mm:.1f} mm "
        f"horizontal, holding after release={holding_status.is_holding_something}"
    )
    line = verdict(PHASE_1_ITEMS[1], ok, detail)
    print(line)
    return line, ok


async def _check_placement(
    world: WorldApi, palletizer: Any, box_prop: str, place_xy_mm: Mapping[str, float]
) -> tuple[str, bool]:
    """The phase's own done-when: `{"command": "start"}` on the
    `box-palletizer` service, polled bounded by STATUS_TIMEOUT_S, then the
    box's resting pose measured against the place target independently of
    the service's own pose arithmetic, immediately after release and again
    after SETTLE_WINDOW_S."""
    import json

    start_result = await palletizer.do_command({"command": "start"})
    print(f"  start: {json.dumps(dict(start_result), default=str, sort_keys=True)}")

    started_at = time.monotonic()
    status = await palletizer.do_command({"command": "status"})
    while status.get("state") == "running":
        if time.monotonic() - started_at > STATUS_TIMEOUT_S:
            line = verdict(
                PHASE_1_ITEMS[2],
                False,
                f"box-palletizer timed out after {STATUS_TIMEOUT_S:.0f}s still running (hung)",
            )
            print(line)
            return line, False
        await asyncio.sleep(STATUS_POLL_S)
        status = await palletizer.do_command({"command": "status"})

    records = status.get("records", [])
    print(f"  status: {json.dumps(dict(status), default=str, sort_keys=True)}")
    # The run's state and the record's outcome are different claims: a run
    # that finishes having failed its pick reports "complete" with a record
    # whose outcome is "failed". Checking only the state passes such a run
    # whenever a box happens to be sitting at the target already, which is
    # exactly what a rerun leaves behind.
    if status.get("state") == "failed" or not records_all_placed(records):
        line = verdict(
            PHASE_1_ITEMS[2],
            False,
            f"box-palletizer did not place the box: state={status.get('state')!r} {records}",
        )
        print(line)
        return line, False

    placed_mm = await _box_pose_mm(world, box_prop)
    placed_ok, placed_error_mm = placement_check(placed_mm, place_xy_mm)
    print(f"  placed (mm): {placed_mm}, error {placed_error_mm:.3f} mm")

    await asyncio.sleep(SETTLE_WINDOW_S)
    settled_mm = await _box_pose_mm(world, box_prop)
    settled_ok, settled_error_mm = placement_check(settled_mm, place_xy_mm)
    drift_ok, drift_mm = settle_check(placed_mm, settled_mm)
    print(f"  after {SETTLE_WINDOW_S:.0f}s (mm): {settled_mm}, error {settled_error_mm:.3f} mm")

    ok = placed_ok and settled_ok and drift_ok
    detail = (
        f"records={records}, place error {placed_error_mm:.3f} mm, settle error "
        f"{settled_error_mm:.3f} mm, drift {drift_mm:.3f} mm over {SETTLE_WINDOW_S:.0f}s"
    )
    line = verdict(PHASE_1_ITEMS[2], ok, detail)
    print(line)
    return line, ok


async def _check_grab_nothing(
    world: WorldApi, gripper: Any, arm: Any, motion: Any, box_prop: str, state: Any
) -> tuple[str, bool]:
    """Moves over the now-empty pick pose (item 3's service run carried the
    box away) and attempts a grab, bounded so a hang is reported as a
    failure."""
    top_face_xyz_mm = await _box_top_face_xyz_mm(world, box_prop)
    if top_face_xyz_mm is None:
        line = verdict(PHASE_1_ITEMS[3], False, f"{box_prop!r} has no known geometry in the world")
        print(line)
        return line, False
    standoff, grasp = _pick_grasp_poses(top_face_xyz_mm)
    await _move(gripper, motion, standoff, state)
    await _move(gripper, motion, grasp, state, linear=True)
    try:
        grabbed = await asyncio.wait_for(gripper.grab(), timeout=GRAB_NOTHING_TIMEOUT_S)
        holding_status = await asyncio.wait_for(
            gripper.is_holding_something(), timeout=GRAB_NOTHING_TIMEOUT_S
        )
    except TimeoutError:
        line = verdict(
            PHASE_1_ITEMS[3], False, f"timed out after {GRAB_NOTHING_TIMEOUT_S:.0f}s (hung)"
        )
        print(line)
        return line, False

    ok = not grabbed and not holding_status.is_holding_something
    line = verdict(
        PHASE_1_ITEMS[3],
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


async def _check_cost(world: WorldApi, run_label: str) -> tuple[str, bool]:
    import json

    ready = await sample_ready_time(world)
    print(f"  [{run_label}] ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  [{run_label}] step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")
    sim_time_ratio = step_rate.get("baseline", {}).get("sim_time_ratio")
    line = verdict(
        PHASE_1_ITEMS[4],
        True,
        f"run_label={run_label} ready_time_s={ready['ready_time_s']} "
        f"sim_time_ratio={sim_time_ratio}",
    )
    print(line)
    return line, True


async def _reset_pick_box(
    world: WorldApi, box_prop: str, pick_pose_mm: Mapping[str, float]
) -> None:
    """Put the box back at its known pick pose before anything runs.

    Every item here assumes the box starts at `pick_pose_mm`, and a rerun
    against a cell left as the previous run finished measures whatever is
    lying around instead. A world `reset` restores neither prop poses nor the
    arm, so the pose is written explicitly."""
    await world.do_command(
        {
            "command": "set_prop_pose",
            "name": box_prop,
            "position": [pick_pose_mm["x"], pick_pose_mm["y"], pick_pose_mm["z"]],
            "orientation_rpy_deg": [0.0, 0.0, 0.0],
        }
    )
    print(f"  reset {box_prop} to the pick pose")


async def _box_height_mm(world: WorldApi, box_prop: str) -> float:
    """`box_prop`'s own z dimension read live from the world, so a drop
    test's expected resting height never carries an invented box size."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    for geometry in geometries:
        if geometry.get("name") == box_prop:
            return float(geometry["box_dims_mm"][2])
    raise ValueError(f"{box_prop!r} has no known geometry in the world")


async def _check_component_frames(
    world: WorldApi, robot: Any, fragment_frames_mm: Mapping[str, Mapping[str, float]]
) -> tuple[str, bool]:
    """Item 1: every fragment component's own `get_pose` reply against the
    world-frame position its fragment frame declares."""
    from viam.components.generic import Generic

    all_ok = True
    for name, expected_mm in fragment_frames_mm.items():
        try:
            client = Generic.from_robot(robot, name)
            pose_reply = await client.do_command({"get_pose": True})
        except Exception as exc:  # noqa: BLE001 - one component's failure should not stop the rest
            print(f"  {name}: get_pose failed: {exc!r}")
            all_ok = False
            continue
        measured_mm = {
            "x": _mm(pose_reply, "x"),
            "y": _mm(pose_reply, "y"),
            "z": _mm(pose_reply, "z"),
        }
        error_mm = pose_delta_mm(expected_mm, measured_mm)
        ok = error_mm <= FRAME_POSE_TOLERANCE_MM
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  {name}: declared {expected_mm}, get_pose {measured_mm}, error "
            f"{error_mm:.3f} mm ({status})"
        )
    line = verdict(
        PHASE_2_ITEMS[0], all_ok, f"{len(fragment_frames_mm)} fragment components checked"
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


async def _check_support_drops(world: WorldApi, box_prop: str) -> tuple[str, bool]:
    """Item 2: the pick station, pallet and pedestal each stop a dropped box
    at the height their own geometry implies."""
    box_height_mm = await _box_height_mm(world, box_prop)
    all_ok = True
    for component in ("pick-station", "pallet", "robot-pedestal"):
        ok, _error_mm = await _drop_box_on_support(world, box_prop, component, box_height_mm)
        all_ok = all_ok and ok
    line = verdict(
        PHASE_2_ITEMS[1], all_ok, "drop test against pick-station, pallet and robot-pedestal"
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
    fence_present = any(name.startswith("fence-") for name in names)
    tunnel_present = any(name.startswith("scan-tunnel-") for name in names)
    ok = fence_present and tunnel_present
    detail = (
        f"fence obstacle present={fence_present}, scan-tunnel obstacle present={tunnel_present}; "
        "human: confirm in the Kit viewport that item 4's pick-to-place plan swings clear of "
        "both rather than clipping through them"
    )
    line = verdict(PHASE_2_ITEMS[2], ok, detail)
    print(line)
    return line, ok


async def _run_pick_and_place_regression(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    palletizer: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    place_xy_mm: Mapping[str, float],
) -> tuple[str, bool]:
    """Item 4: phase 1's own mechanism, reach, grab/carry/release, the
    service's placement, and a grab with nothing under the tool, run end to
    end against the new cell."""
    await _reset_pick_box(world, box_prop, pick_pose_mm)
    state = await _cell_world_state(world, {box_prop})

    print("  -- reach --")
    _reach_line, reach_ok = await _check_reach(world, arm, gripper, motion, box_prop, state)
    print("  -- grab, carry, release --")
    _carry_line, carry_ok = await _check_grab_carry_release(
        world, gripper, arm, motion, box_prop, state
    )
    print("  -- placement (box-palletizer service) --")
    _place_line, place_ok = await _check_placement(world, palletizer, box_prop, place_xy_mm)
    print("  -- grab with nothing under the tool --")
    _grab_nothing_line, grab_nothing_ok = await _check_grab_nothing(
        world, gripper, arm, motion, box_prop, state
    )

    ok = reach_ok and carry_ok and place_ok and grab_nothing_ok
    line = verdict(PHASE_2_ITEMS[3], ok, "phase 1's reach/carry/place/grab-nothing sequence")
    print(line)
    return line, ok


async def _check_scan_tunnel_collider(world: WorldApi) -> tuple[str, bool]:
    """Item 5: whether `scan-tunnel` already has a collider registered, the
    way `robot-pedestal`'s derived one does. Always informational, since
    either answer is a valid finding to record, not a failure: this decides
    whether `workcell_scenery.DERIVED_COLLIDER_MODELS` needs a second entry,
    which is a source change this checklist does not make."""
    geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    has_collider = any(str(geometry["name"]).startswith("scan-tunnel-") for geometry in geometries)
    detail = (
        f"scan-tunnel collider registered in prop_geometries={has_collider}; human: drive the "
        "arm through the tunnel's span and confirm whether it stops at a real collider (needs "
        "workcell_scenery.DERIVED_COLLIDER_MODELS the way robot-pedestal does) or passes through"
    )
    line = verdict(PHASE_2_ITEMS[4], True, detail)
    print(line)
    return line, True


async def _check_cost_against_phase_1(world: WorldApi, run_label: str) -> tuple[str, bool]:
    """Item 6: cost, with `sim_time_ratio` printed alongside the cited
    phase-1 baseline rather than re-measuring a cell this phase deleted.
    Phase 1's `ready_time_s` is not comparable (see
    `PHASE_1_BASELINE_SIM_TIME_RATIO`'s comment), so this phase's own
    `ready_time_s` is printed as a first reading rather than a delta against
    that gap, for phase 3 to use as its baseline."""
    import json

    ready = await sample_ready_time(world)
    print(f"  [{run_label}] ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  [{run_label}] step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")
    sim_time_ratio = step_rate.get("baseline", {}).get("sim_time_ratio")
    print(
        f"  phase 1 baseline: sim_time_ratio={PHASE_1_BASELINE_SIM_TIME_RATIO} "
        "(ready_time_s is not a baseline: phase 1's reading is a measurement gap, not a result, "
        'see PLAN.md\'s "Carried in from phase 1")'
    )
    print(
        f"  [{run_label}] ready_time_s={ready['ready_time_s']} is a first reading, to carry "
        "forward as phase 3's baseline, not a delta against phase 1"
    )
    line = verdict(
        PHASE_2_ITEMS[5],
        True,
        f"run_label={run_label} ready_time_s={ready['ready_time_s']} (first reading) "
        f"sim_time_ratio={sim_time_ratio}, phase 1 baseline sim_time_ratio="
        f"{PHASE_1_BASELINE_SIM_TIME_RATIO}",
    )
    print(line)
    return line, True


async def _check_render_only_has_no_collider(
    world: WorldApi, box_prop: str, fragment_frames_mm: Mapping[str, Mapping[str, float]]
) -> tuple[str, bool]:
    """Item 7, an implementation risk rather than a plan requirement:
    `sim_manager.py` builds a `collision: False` cube as a cuboid and then
    strips `UsdPhysics.CollisionAPI` and `RigidBodyAPI`, since `omni` and
    `pxr` are not importable in this environment and the mesh path's bare-
    authoring route could not be tested here. Drops `box_prop` over
    `caution-tape` (a render-only floor decal) and checks it rests on the
    floor rather than on the tape."""
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
        PHASE_2_ITEMS[6],
        ok,
        f"expected floor rest {expected_floor_rest_z_mm:.1f} mm, measured {rested_mm}, error "
        f"{error_mm:.3f} mm",
    )
    print(line)
    return line, ok


async def _run_phase_2(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    palletizer: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    place_xy_mm: Mapping[str, float],
    run_label: str,
    robot: Any,
    fragment_path: Path,
) -> list[tuple[str, bool]]:
    import json

    fragment = json.loads(fragment_path.read_text())
    fragment_frames_mm = fragment_component_frames_mm(fragment)

    results: list[tuple[str, bool]] = []

    print("\n-- component frames --")
    results.append(await _check_component_frames(world, robot, fragment_frames_mm))

    print("\n-- support drop tests --")
    results.append(await _check_support_drops(world, box_prop))

    print("\n-- fence and scan-tunnel obstacles --")
    results.append(await _check_fence_and_tunnel_obstacles(world))

    print("\n-- phase 1 pick and place regression --")
    results.append(
        await _run_pick_and_place_regression(
            world, arm, gripper, motion, palletizer, box_prop, pick_pose_mm, place_xy_mm
        )
    )

    print("\n-- scan-tunnel collider question --")
    results.append(await _check_scan_tunnel_collider(world))

    print("\n-- cost vs phase 1 --")
    results.append(await _check_cost_against_phase_1(world, run_label))

    print("\n-- render-only scenery risk (caution-tape) --")
    results.append(await _check_render_only_has_no_collider(world, box_prop, fragment_frames_mm))

    print("\n== summary ==")
    for line, _ in results:
        print(line)
    return results


async def _run(
    world: WorldApi,
    arm: Any,
    gripper: Any,
    motion: Any,
    palletizer: Any,
    box_prop: str,
    pick_pose_mm: Mapping[str, float],
    place_xy_mm: Mapping[str, float],
    run_label: str,
) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    await _reset_pick_box(world, box_prop, pick_pose_mm)
    # the box is the payload, not an obstacle: the cup has to descend onto it
    state = await _cell_world_state(world, {box_prop})

    print("\n-- reach --")
    results.append(await _check_reach(world, arm, gripper, motion, box_prop, state))

    print("\n-- grab, carry, release --")
    results.append(await _check_grab_carry_release(world, gripper, arm, motion, box_prop, state))

    print("\n-- placement (box-palletizer service) --")
    results.append(await _check_placement(world, palletizer, box_prop, place_xy_mm))

    print("\n-- grab with nothing under the tool --")
    results.append(await _check_grab_nothing(world, gripper, arm, motion, box_prop, state))

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
    box_prop: str
    pick_x_mm: float
    pick_y_mm: float
    pick_z_mm: float
    place_x_mm: float
    place_y_mm: float
    phase: int
    run_label: str
    fragment: str


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
    parser.add_argument("--box-prop", required=True, help="prop name of the box to pick and place")
    parser.add_argument(
        "--pick-x-mm", type=float, required=True, help="the box's known resting centre, x in mm"
    )
    parser.add_argument(
        "--pick-y-mm", type=float, required=True, help="the box's known resting centre, y in mm"
    )
    parser.add_argument(
        "--pick-z-mm", type=float, required=True, help="the box's known resting centre, z in mm"
    )
    parser.add_argument(
        "--place-x-mm",
        type=float,
        required=True,
        help="the place target x in mm, matching box-palletizer's configured place_pose_mm",
    )
    parser.add_argument(
        "--place-y-mm",
        type=float,
        required=True,
        help="the place target y in mm, matching box-palletizer's configured place_pose_mm",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=1,
        help="which phase's checklist items to run (default: 1)",
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
        "--fragment",
        default=str(
            Path(__file__).resolve().parent.parent / "fragments" / "isaac-sim-palletizing.json"
        ),
        help="path to the vendored fragment JSON, for phase 2 item 1's declared component frames",
    )
    ns = parser.parse_args(argv)
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        world=ns.world,
        arm=ns.arm,
        gripper=ns.gripper,
        motion=ns.motion,
        palletizer=ns.palletizer,
        box_prop=ns.box_prop,
        pick_x_mm=ns.pick_x_mm,
        pick_y_mm=ns.pick_y_mm,
        pick_z_mm=ns.pick_z_mm,
        place_x_mm=ns.place_x_mm,
        place_y_mm=ns.place_y_mm,
        phase=ns.phase,
        run_label=ns.run_label,
        fragment=ns.fragment,
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
        pick_pose_mm = {"x": args.pick_x_mm, "y": args.pick_y_mm, "z": args.pick_z_mm}
        place_xy_mm = {"x": args.place_x_mm, "y": args.place_y_mm}
        if args.phase == 2:
            await _run_phase_2(
                world,
                arm,
                gripper,
                motion,
                palletizer,
                args.box_prop,
                pick_pose_mm,
                place_xy_mm,
                args.run_label,
                robot,
                Path(args.fragment),
            )
        else:
            await _run(
                world,
                arm,
                gripper,
                motion,
                palletizer,
                args.box_prop,
                pick_pose_mm,
                place_xy_mm,
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
