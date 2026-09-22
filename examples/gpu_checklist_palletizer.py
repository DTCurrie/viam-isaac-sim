"""GPU acceptance checklist for the palletizing cell (phase 1: vacuum gripper,
one box picked and placed by the box-palletizer service; phase 2: the
vendored `viam:workcell-components` workcell adopted under the same
mechanism; phase 3: the full eight-box pack, planned by
`viam:pack-sequencer:sequencer` and executed by `box-palletizer`). Pass
`--phase 1` (default), `--phase 2` or `--phase 3`.

Connects to a running Viam machine (the module running on the Isaac GPU box):
`arm-1`, `gripper-1`, `builtin` motion, the `box-palletizer` service, and
(phase 3 only) the `pack-sequencer` service. Item 3 is phase 1's own
done-when, so it drives the actual pick and place through
`box-palletizer`'s DoCommand rather than re-driving the arm and gripper
itself. Items 1, 2 and 4 drive the arm and gripper directly to test the
underlying grab mechanism, reading the box's pose from the world at run
time rather than the service's own arithmetic, so a green result there
can't just be agreement with the code under test. Phase 3 drives the whole
pack through `box-palletizer`'s own `start`/`status` surface throughout,
since the sequencer, not this script, owns every target pose.

For phases 1 and 2, the box's known resting pose before a run and the
target place position are facts about the deployed cell, so they come in
as CLI arguments (`--pick-*-mm`, `--place-*-mm`) rather than an invented
constant in this file. Phase 3 needs neither: `box-palletizer` sources the
infeed pose from the sim itself and the sequencer owns every place target,
so its checklist items read poses off the service's own status records
instead.

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
   compared against the component's own group frame primitive
2. the pick station and pallet stop a dropped box at the height their own
   geometry implies, and the pedestal's collider matches what it declares
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

from isaac_module.models.palletizer import CUP_APPROACH_GAP_MM, approach_and_descend
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
    quat_to_ov,
)
from isaac_module.workcell_scenery import parse_visuals

MM_PER_M = 1000.0

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
    "1. every component in the fragment renders at the pose its frame declares: its own group "
    "frame primitive against the fragment, and every box and capsule it draws against the prim "
    "the sim spawned for it",
    "2. the pick station and pallet stop a dropped box at the height the component's "
    "geometry implies, and every declared `frame.geometry` collider spawns at the pose, "
    "orientation and size it declares",
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

PHASE_3_ITEMS: tuple[str, ...] = (
    "1. eight boxes placed, the sequencer's target and the measured pose per box, as a table",
    "2. the order placed matches `get_pack_order`",
    "3. obstacles come from the world state store rather than this service's own hand-built "
    "WorldState, shown by running it both ways and reporting whether plans stay collision-free "
    "under the store",
    "4. `set_box_transform` moves the box in the viewer to where physics put it, with the "
    "largest plan-versus-actual difference recorded across the eight",
    "5. a box the arm fails to place is `skip_box`ed and the run continues",
    "6. cost against phase 2: cold and warm `ready` and the 10 s step rate",
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
# and how long to wait for it to settle. Shorter than item 3 (phase 1)'s SETTLE_WINDOW_S since a
# freshly dropped box has nowhere to drift once it has landed
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

# phase 3's bounded wait on a full eight-box pack: eight picks and places, so eight times
# phase 1's single-box budget
PHASE_3_STATUS_TIMEOUT_S = STATUS_TIMEOUT_S * 8


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


async def _move(
    gripper: Any,
    motion: Any,
    pose: Any,
    state: Any,
    linear: bool = False,
    leg: str = "move",
) -> None:
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
    try:
        await mover.move_to(pose, state, linear=linear)
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
        line = verdict(PHASE_1_ITEMS[0], False, f"{box_prop!r} has no known geometry in the world")
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
    line = verdict(PHASE_1_ITEMS[1], ok, detail)
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
    """The phase's own done-when: `{"command": "start"}` on the
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
            PHASE_1_ITEMS[2],
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
    line = verdict(PHASE_1_ITEMS[2], ok, detail)
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
            PHASE_1_ITEMS[3],
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
        PHASE_2_ITEMS[0],
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
        PHASE_2_ITEMS[1],
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
    line = verdict(PHASE_2_ITEMS[2], ok, detail)
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
    """Item 4: phase 1's own mechanism, reach, grab/carry/release, the
    service's placement, and a grab with nothing under the tool, run end to
    end against the new cell."""
    await _park_arm(world, gripper, motion)
    reset_ok, reset_detail = await _reset_pick_box(world, box_prop, pick_pose_mm, "pick-station")
    if not reset_ok:
        line = verdict(
            PHASE_2_ITEMS[3],
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
    has_collider = any(
        str(geometry["name"]).startswith(("scan-tunnel-", "frame_scan_tunnel"))
        for geometry in geometries
    )
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
        PHASE_2_ITEMS[6],
        ok,
        f"expected floor rest {expected_floor_rest_z_mm:.1f} mm, measured {rested_mm}, error "
        f"{error_mm:.3f} mm",
    )
    print(line)
    return line, ok


async def _run_full_pack(palletizer: Any) -> dict[str, Any]:
    """Starts `box-palletizer`'s own run and polls `status` until it leaves
    `running`, bounded by `PHASE_3_STATUS_TIMEOUT_S`. Every phase 3 item
    reads the same run's final status rather than driving the arm itself,
    since the sequencer owns every target pose here."""
    import json

    start_result = await palletizer.do_command({"command": "start"})
    print(f"  start: {json.dumps(dict(start_result), default=str, sort_keys=True)}")

    started_at = time.monotonic()
    status = await palletizer.do_command({"command": "status"})
    while status.get("state") == "running":
        if time.monotonic() - started_at > PHASE_3_STATUS_TIMEOUT_S:
            raise TimeoutError(
                f"box-palletizer timed out after {PHASE_3_STATUS_TIMEOUT_S:.0f}s still running "
                "(hung)"
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
    line = verdict(PHASE_3_ITEMS[0], ok, f"{len(placed)} of {expected_count} boxes placed")
    print(line)
    return line, ok


def _check_pack_order(
    records: Sequence[Mapping[str, Any]], expected_seqs: Sequence[int]
) -> tuple[str, bool]:
    """Item 2: the order boxes were placed in against the sequencer's own
    `get_pack_order`."""
    ok, observed = order_matches_pack_order(records, expected_seqs)
    line = verdict(
        PHASE_3_ITEMS[1],
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
        PHASE_3_ITEMS[2],
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
    line = verdict(PHASE_3_ITEMS[3], ok, detail)
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
        PHASE_3_ITEMS[4],
        ok,
        f"{detail}; human: to exercise this path, drag one box out of the arm's reach in the "
        "Kit viewport partway through a run and confirm it gets skip_box'ed rather than hanging "
        "the run",
    )
    print(line)
    return line, ok


async def _check_cost_against_phase_2(world: WorldApi, run_label: str) -> tuple[str, bool]:
    """Item 6: cost, printed as a first reading against phase 2's. Phase
    2's own GPU checklist run is recorded as PENDING in its own phase
    document, so no `sim_time_ratio` number exists there to cite the way
    `PHASE_1_BASELINE_SIM_TIME_RATIO` does. Printed for a human to diff by
    eye against phase 2's own run output instead of fabricating one here."""
    import json

    ready = await sample_ready_time(world)
    print(f"  [{run_label}] ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  [{run_label}] step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")
    sim_time_ratio = step_rate.get("baseline", {}).get("sim_time_ratio")
    line = verdict(
        PHASE_3_ITEMS[5],
        True,
        f"run_label={run_label} ready_time_s={ready['ready_time_s']} "
        f"sim_time_ratio={sim_time_ratio}; compare by eye against phase 2's own recorded run",
    )
    print(line)
    return line, True


async def _run_phase_3(
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

    print("\n-- cost vs phase 2 --")
    results.append(await _check_cost_against_phase_2(world, run_label))

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


async def _run_phase_2(
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
            PHASE_2_ITEMS[0],
            lambda: _check_component_frames(world, robot, arm.name, fragment_frames_mm),
        )
    )

    print("\n-- support drop tests --")
    results.append(
        await _guarded(
            PHASE_2_ITEMS[1],
            lambda: _check_support_drops(world, box_prop, declared_colliders_mm),
        )
    )

    print("\n-- fence and scan-tunnel obstacles --")
    results.append(
        await _guarded(PHASE_2_ITEMS[2], lambda: _check_fence_and_tunnel_obstacles(world))
    )

    print("\n-- phase 1 pick and place regression --")
    results.append(
        await _guarded(
            PHASE_2_ITEMS[3],
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
    results.append(await _guarded(PHASE_2_ITEMS[4], lambda: _check_scan_tunnel_collider(world)))

    print("\n-- cost vs phase 1 --")
    results.append(
        await _guarded(PHASE_2_ITEMS[5], lambda: _check_cost_against_phase_1(world, run_label))
    )

    print("\n-- render-only scenery risk (caution-tape) --")
    results.append(
        await _guarded(
            PHASE_2_ITEMS[6],
            lambda: _check_render_only_has_no_collider(
                world, gripper, box_prop, fragment_frames_mm
            ),
        )
    )

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
    # None only under --phase 3, which reads the infeed pose and every place
    # target off box-palletizer's own status records instead. _parse_args
    # rejects a missing one under phases 1 and 2.
    box_prop: str | None
    pick_x_mm: float | None
    pick_y_mm: float | None
    pick_z_mm: float | None
    place_x_mm: float | None
    place_y_mm: float | None
    phase: int
    run_label: str
    obstacle_source_label: str
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
    parser.add_argument(
        "--sequencer",
        default="pack-sequencer",
        help="the viam:pack-sequencer:sequencer service name (box-palletizer's own "
        "'sequencer' attribute), phase 3 only",
    )
    parser.add_argument(
        "--box-prop",
        help="prop name of the single box driven directly for phases 1 and 2's grab-mechanism "
        "items. Phase 3 drives all of box-palletizer's own 'box_props' list instead, and reads "
        "them off the service's own status records rather than this flag",
    )
    parser.add_argument("--pick-x-mm", type=float, help="the box's known resting centre, x in mm")
    parser.add_argument("--pick-y-mm", type=float, help="the box's known resting centre, y in mm")
    parser.add_argument("--pick-z-mm", type=float, help="the box's known resting centre, z in mm")
    parser.add_argument(
        "--place-x-mm",
        type=float,
        help="the place target x in mm for phases 1 and 2's own single-box place test, "
        "unrelated to box-palletizer's config (it has no place_pose_mm attribute; the "
        "sequencer owns every place target for phase 3)",
    )
    parser.add_argument(
        "--place-y-mm",
        type=float,
        help="the place target y in mm for phases 1 and 2's own single-box place test, "
        "unrelated to box-palletizer's config (it has no place_pose_mm attribute; the "
        "sequencer owns every place target for phase 3)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2, 3),
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
        "--obstacle-source-label",
        default="world_state_store",
        choices=("world_state_store", "prop_geometries"),
        help="tags phase 3 item 3's obstacle-source reading with whichever value the overlay's "
        "box-palletizer.obstacle_source is currently deployed with. Run the script once per "
        "value, redeploying the overlay between runs, and compare",
    )
    parser.add_argument(
        "--fragment",
        default=str(
            Path(__file__).resolve().parent.parent / "fragments" / "isaac-sim-palletizing.json"
        ),
        help="path to the vendored fragment JSON, for phase 2 item 1's declared component frames",
    )
    ns = parser.parse_args(argv)
    # Phases 1 and 2 drive the box directly and need its resting pose and a
    # place target. Phase 3 reads both off box-palletizer's own status
    # records, so requiring them there would mean typing numbers the run
    # ignores, which is how an invented value ends up looking like a
    # measurement.
    if ns.phase in (1, 2):
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
            parser.error(f"--phase {ns.phase} requires {', '.join(missing)}")
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
        phase=ns.phase,
        run_label=ns.run_label,
        obstacle_source_label=ns.obstacle_source_label,
        fragment=ns.fragment,
    )


def _single_box_args(args: Args) -> tuple[str, dict[str, float], dict[str, float]]:
    """Phases 1 and 2's directly-driven box: its prop name, its known resting
    centre and the place target. ``_parse_args`` rejects a missing one under
    those phases, so this narrows the types rather than re-checking a case a
    caller can reach."""
    if (
        args.box_prop is None
        or args.pick_x_mm is None
        or args.pick_y_mm is None
        or args.pick_z_mm is None
        or args.place_x_mm is None
        or args.place_y_mm is None
    ):
        raise ValueError(f"--phase {args.phase} needs the box prop and the pick and place poses")
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
        # the 15:12 run before its first item and what phase 3's own path,
        # never yet run, would have hit too.
        sequencer = SequencerClient(WorldStateStore.from_robot(robot, args.sequencer))
        if args.phase == 3:
            await _run_phase_3(
                world,
                palletizer,
                sequencer,
                args.run_label,
                args.obstacle_source_label,
            )
            return
        # the place target the flags name is not used by any check any more:
        # the sequencer owns the slot, and item 4 judges against its record
        box_prop, pick_pose_mm, _place_xy_mm = _single_box_args(args)
        if args.phase == 2:
            await _run_phase_2(
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
        else:
            await _run(
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
