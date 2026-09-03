"""The pick sequence, the seam protocols it depends on, and the JSON markers
each step prints.

W36's sequence: look -> detect -> open -> pre-grasp -> grasp -> grab -> lift ->
held-block Transform -> place on the pad (when the scene has one) -> open.
Orchestration only (DEC-4) - no IK, no planning, no sim ground truth in the
control path."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from viam.proto.common import Geometry, Pose, Transform, WorldState

from pickcell.measurement import (
    DETECT_Z_TOLERANCE_MM,
    MEASURED_SIZE_DEGENERATE_FRACTION,
    TallestEstimate,
    carry_clear_above_support_mm,
    keepout_height_mm,
    tallest_in_region_mm,
)
from pickcell.obstacles import (
    CARRY_CLEAR_ABOVE_SUPPORT_MM,
    HELD_BLOCK_PADDING_MM,
    KEEPOUT_HEIGHT_MM,
    held_block_transform,
    obstacles_from_prop_geometries,
    pad_top_centre_mm,
    pick_area_keepout,
    reachable_region_mm,
    support_obstacle,
    table_recipe_unless_served,
    world_state,
)
from pickcell.poses import (
    FINGERTIP_OVERHANG_MM,
    FOCUS_HEIGHT_ABOVE_SUPPORT_MM,
    PRE_GRASP_STANDOFF_MM,
    SCAN_ATTEMPTS,
    _pointing_down,
    _pose_to_dict,
    _poses_match_mm,
    corrected_pose,
    grasp_height_mm,
    grasp_pose,
    pre_grasp_pose,
    tallest_sweep_attempts,
    with_z,
)

HELD_BLOCK_TRANSFORM_MARKER = "HELD_BLOCK_TRANSFORM_JSON="
DETECTED_BLOCK_POSE_MARKER = "DETECTED_BLOCK_POSE_JSON="
GRAB_DIAGNOSTICS_MARKER = "GRAB_DIAGNOSTICS_JSON="
MOVE_DIAGNOSTICS_MARKER = "MOVE_DIAGNOSTICS_JSON="
HOLD_SAMPLES_MARKER = "HOLD_SAMPLES_JSON="
RESET_MID_HOLD_MARKER = "RESET_MID_HOLD_JSON="
MEASURED_BLOCK_MARKER = "MEASURED_BLOCK_JSON="
PLACED_BLOCK_MARKER = "PLACED_BLOCK_JSON="
MEASURED_TALLEST_MARKER = "MEASURED_TALLEST_JSON="

# checklist item 5 wants the block held 100 mm up for 5 s; item 6 resets the
# world mid-hold and expects the post-reset hooks (ARM-15/XC-5) to keep it held
HOLD_SAMPLE_S = 1.0

# refine the look when the detected block sits farther than this from the scan
# centre; closer than this, a second measurement gains nothing
FOCUS_LOOK_OFFSET_MM = 30.0
# how close (x, y, z, mm) detection's final pose must be to the block-airspace
# waypoint to skip the extra free move before the linear pre-grasp approach
# (checklist item 3 / GPU run 21: a free plan swung the gripper through an
# unprotected block on the way to a joint-space pre-grasp)
AIRSPACE_WAYPOINT_TOLERANCE_MM = 1.0

# 2F-85 opens 85 mm (W13/W15); 10 mm covers the closed fingers' own
# clearance, so a measured block wider than this cannot be grasped.
JAW_MAX_BLOCK_MM = 75.0

# placing: the held block hovers this gap above the pad top at release and
# drops it - the pad stays a planner obstacle, so the gripper never touches it
PLACE_CLEARANCE_MM = 15.0
# headroom above the tallest placed block for the pad-zone keep-out box, so a
# planner-legal carry can never graze a placed block within tracking error
PLACED_KEEPOUT_CLEARANCE_MM = 30.0
PLACE_XY_TOLERANCE_MM = 100.0  # verdict: block centre inside the 200 mm pad footprint
PLACE_Z_TOLERANCE_MM = 10.0
PLACE_SETTLE_S = 1.0  # wall-clock pause before reading the placed pose back


class JawLimitError(RuntimeError):
    """The measured block exceeds the gripper's jaw limit; the pick is refused
    with no arm motion, so a caller can classify it as a skip, not a failure."""


class Detector(Protocol):
    async def block_pose_world(self) -> Pose:
        """The target block's pose (mm), in the world frame, detected from a
        stationary pre-grasp pose (motion blur)."""
        ...


class Mover(Protocol):
    async def look_from(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        """Move the wrist CAMERA frame to ``pose`` (world, mm) so the block is
        in view before detection; the arm's zero pose looks away from it.
        ``linear`` asks for a straight-line path (the focus move, which enters
        the detected block's airspace)."""
        ...

    async def move_to(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        """Move the gripper's TCP frame to `pose` (mm, world frame). ``linear``
        asks for a straight-line approach (the grasp descent and the lift)."""
        ...


class GripperApi(Protocol):
    async def open(self) -> None: ...

    async def grab(self) -> bool: ...

    async def is_holding_something(self) -> Any: ...


class WorldApi(Protocol):
    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        """The world component's DoCommand (a Generic client in real mode,
        the world model itself in mock) - used here for ``prop_geometries``
        and ``randomize_props``."""
        ...


class TallestScanner(Protocol):
    async def scan_world_mm(self) -> np.ndarray:
        """World-frame mm points (N x 3) of the scanner's current view, fed
        to ``tallest_in_region_mm``."""
        ...


@dataclass
class PickPipeline:
    """W36's sequence: look -> detect -> open -> pre-grasp -> grasp -> grab -> lift ->
    held-block Transform -> place on the pad (when the scene has one) -> open.
    Orchestration only - no IK, no planning."""

    detector: Detector
    mover: Mover
    gripper: GripperApi
    block_name: str
    # None = measure from the focused segment's point cloud (the default);
    # a float overrides the measurement end to end (today's fixed-size path)
    block_size_mm: float | None
    gripper_name: str
    table: Geometry | None = None  # W4's table exists only in the P5 cell: opt in with --table
    other_blocks: Sequence[Geometry] = ()
    # the live sim is the default source of obstacle blocks (prop_geometries);
    # None keeps the pipeline to table/support/other_blocks only (mock's world
    # has no configured props, so this is a no-op there without --randomize-seed)
    world: WorldApi | None = None
    target_prop_name: str | None = None  # excluded from sim obstacles and from randomize_props
    # blocks randomised together with the target; empty = derive every
    # non-fixed prop with known dims from the live scene (distractors too)
    movable_prop_names: Sequence[str] = ()
    randomize_seed: int | None = None  # checklist item 1: re-randomise before this pick
    randomize_region_mm: tuple[Sequence[float], Sequence[float]] | None = None
    # adds size_range_mm to the randomize_props payload for every movable
    # name (None = today's payload, byte-identical)
    randomize_size_range_mm: tuple[float, float] | None = None
    # Measured over seeds 0-99, six 60 mm cubes in [450, 700] x [-250, 250] mm:
    # 200 mm succeeds 0/100, 140 mm succeeds 100/100. W26's gripper-clearance
    # intent still holds at 140 (worst-case face gap 80 mm vs 12.5 mm jaw
    # overhang; the 2F-85 opens 85 mm).
    randomize_min_separation_mm: float = 140.0
    # centre of the scatter region a randomize run used; the camera scans the
    # workspace from above it - the DETECTOR finds the block, never sim truth
    scan_centre_mm: tuple[float, float] | None = None
    # the scatter region itself; when known, the carry gets a keep-out box
    # over it instead of the slow constrained linear traverse
    pick_region_mm: tuple[Sequence[float], Sequence[float]] | None = None
    # the mock detector reports a camera-frame z (mock mode has no frame
    # system), so the resting-height gate only applies to real detections
    verify_detection_height: bool = True
    # the fixed prop to set the block down on; None = release at the lift pose
    place_prop_name: str | None = None
    place_clearance_mm: float = PLACE_CLEARANCE_MM
    # pad top-face centre (x, y, top z) in mm; set by _sim_obstacles when found
    place_pad_top_mm: tuple[float, float, float] | None = None
    # (x, y) mm added to the pad centre as the place target, so several blocks
    # spread across one pad; None places at the centre. The placement verdict
    # measures against the offset target.
    place_offset_mm: tuple[float, float] | None = None
    # the color named in operator-facing messages; "red" keeps the original
    # single-color script's wording, the conductor passes each block's color
    block_label: str = "red"
    # place-pad zone keep-out (conductor loop mode): the region over the pads
    # and the tallest block already placed there this run. When both are set,
    # the carry cannot cut through the pad zone below that height and the
    # pre-place hover is floored so the held block's bottom clears it.
    place_region_mm: tuple[Sequence[float], Sequence[float]] | None = None
    placed_tallest_mm: float | None = None
    # TCP-z floor for the pre-place hover; set with the place keep-out above
    place_clear_tcp_z_mm: float | None = None
    standoff_mm: float = PRE_GRASP_STANDOFF_MM
    look_pose: Pose | None = None  # None = detect from wherever the arm is
    support_z_mm: float = 0.0  # the surface the block rests on (floor in the current fragment)
    fingertip_overhang_mm: float = FINGERTIP_OVERHANG_MM
    # optional: gathers jaw angle, pad poses and the block's actual pose after a
    # failed grab, so the failure explains itself (missed vs closed-but-not-holding)
    diagnose: Callable[[], Awaitable[dict[str, Any]]] | None = None
    # optional: measures (believed - physical) TCP in mm at the current pose;
    # the result is added to the grasp/lift targets (None = no correction)
    tcp_correction: Callable[[], Awaitable[tuple[float, float, float]]] | None = None
    # checklist item 5: hold at the lift pose this many seconds, sampling
    # is_holding_something once per HOLD_SAMPLE_S (0 = release immediately)
    hold_s: float = 0.0
    # checklist item 6: called mid-hold to reset the world; must report
    # holding_before_reset/holding_after_reset (None = no reset probe)
    mid_hold_reset: Callable[[], Awaitable[dict[str, Any]]] | None = None
    # set by _detect_block once a detection is accepted: block_size_mm when
    # explicit, else the measured size - every downstream consumer reads
    # this, never block_size_mm directly
    resolved_block_size_mm: float | None = None
    # phase 4: primary/fallback tallest-object scanners, run only when
    # randomize_size_range_mm is set (dynamic keep-out/carry heights)
    side_scanner: TallestScanner | None = None
    wrist_scanner: TallestScanner | None = None
    # set by _sim_obstacles from the randomize response's sizes_mm (log-only
    # ground truth - the control path never consumes it): max drawn z-dim,
    # None without a sizes-bearing response
    drawn_tallest_mm: float | None = None
    # set by _measure_tallest: the trusted-or-fallback tallest estimate, its
    # source, and the wrist-sweep vantages tried - fed into the derived
    # keep-out/carry heights once the held size is known (_run_steps, after
    # the jaw check) and printed in MEASURED_TALLEST_JSON
    tallest_estimate: TallestEstimate | None = None
    tallest_source: str | None = None
    tallest_scan_poses_mm: list[dict[str, float]] = field(default_factory=list)
    # set alongside the marker: the derived heights that replace
    # KEEPOUT_HEIGHT_MM / CARRY_CLEAR_ABOVE_SUPPORT_MM for this pick
    measured_keepout_height_mm: float | None = None
    measured_carry_clear_above_support_mm: float | None = None

    async def _move_or_diagnose(
        self, pose: Pose, move_world_state: WorldState, linear: bool = False
    ) -> None:
        """A failed move prints the arm's joint state (measured vs drive
        target), the gripper pads and the block's pose before re-raising."""
        try:
            await self.mover.move_to(pose, move_world_state, linear)
        except Exception:
            if self.diagnose is not None:
                report = await self.diagnose()
                print(f"{MOVE_DIAGNOSTICS_MARKER}{json.dumps(report, default=str, sort_keys=True)}")
            raise

    async def _sim_obstacles(self) -> list[Geometry]:
        if self.world is None:
            return []
        if self.randomize_seed is not None:
            names = list(self.movable_prop_names)
            if not names:
                pre = await self.world.do_command({"command": "prop_geometries"})
                names = [
                    g["name"]
                    for g in pre.get("geometries", [])
                    if not g["fixed"] and any(d > 0 for d in g["box_dims_mm"])
                ]
            if not names:
                print("step: randomize skipped - no movable props in the scene")
            else:
                region = self.randomize_region_mm or reachable_region_mm(
                    face_z_mm=self.support_z_mm
                )
                print(
                    f"step: randomize props (checklist item 1, seed {self.randomize_seed}, "
                    f"names {names}, region {region})"
                )
                randomize_command: dict[str, Any] = {
                    "command": "randomize_props",
                    "names": names,
                    "region": [list(region[0]), list(region[1])],
                    "seed": self.randomize_seed,
                    "min_separation": self.randomize_min_separation_mm,
                }
                if self.randomize_size_range_mm is not None:
                    randomize_command["size_range_mm"] = list(self.randomize_size_range_mm)
                response = await self.world.do_command(randomize_command)
                (x0, y0, _z0), (x1, y1, _z1) = region
                self.scan_centre_mm = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                self.pick_region_mm = region
                print(f"  scattered: {sorted(response.get('positions') or {})}")
                # log-only ground truth: the pipeline never consumes these
                # dims (the camera measures the target), but the log showing
                # them is what proves a size range was actually applied
                sizes_mm = response.get("sizes_mm") or {}
                if sizes_mm:
                    rounded = {
                        name: [round(float(v), 1) for v in dims]
                        for name, dims in sorted(sizes_mm.items())
                    }
                    print(f"  sizes (mm, module-reported): {rounded}")
                    self.drawn_tallest_mm = max(float(dims[2]) for dims in sizes_mm.values())
        response = await self.world.do_command({"command": "prop_geometries"})
        geometries = response.get("geometries", [])
        if self.place_prop_name is not None:
            self.place_pad_top_mm = pad_top_centre_mm(geometries, self.place_prop_name)
        exclude = {self.target_prop_name} if self.target_prop_name else set()
        return obstacles_from_prop_geometries(geometries, exclude)

    async def _measure_tallest(self, move_world_state: WorldState) -> None:
        """Tallest-object measurement (phase 4): side scan first (occlusion-
        proof except for a nearer silhouette hiding a farther block), then
        the wrist-sweep ladder when the side scan is untrusted, then the
        size-range max as a conservative last resort. ``verify_detection_height``
        doubles as the mock/real switch (the existing convention): mock skips
        the region clip and reads the support at 0 (seam's mock mapping)."""
        assert self.randomize_size_range_mm is not None
        mock = not self.verify_detection_height
        region_mm = None if mock else self.pick_region_mm
        support_z_mm = 0.0 if mock else self.support_z_mm

        if self.side_scanner is not None:
            points = await self.side_scanner.scan_world_mm()
            estimate = tallest_in_region_mm(
                points, region_mm, support_z_mm, self.randomize_size_range_mm
            )
            print(f"  tallest side scan: {estimate}")
            if estimate.trusted:
                self.tallest_estimate = estimate
                self.tallest_source = "side"
                return

        if (
            self.wrist_scanner is not None
            and self.look_pose is not None
            and self.scan_centre_mm is not None
            and self.pick_region_mm is not None
        ):
            for offset_x, offset_y, wrist_theta_deg in tallest_sweep_attempts(self.pick_region_mm):
                vantage = _pointing_down(
                    self.scan_centre_mm[0] + offset_x,
                    self.scan_centre_mm[1] + offset_y,
                    self.look_pose.z,
                    theta_deg=wrist_theta_deg,
                )
                print(f"step: tallest wrist sweep vantage: {_pose_to_dict(vantage)}")
                await self.mover.look_from(vantage, move_world_state)
                self.tallest_scan_poses_mm.append(_pose_to_dict(vantage))
                points = await self.wrist_scanner.scan_world_mm()
                estimate = tallest_in_region_mm(
                    points, region_mm, support_z_mm, self.randomize_size_range_mm
                )
                print(f"  tallest wrist sweep attempt: {estimate}")
                if estimate.trusted:
                    self.tallest_estimate = estimate
                    self.tallest_source = "wrist_sweep"
                    return

        lo_mm, hi_mm = self.randomize_size_range_mm
        print(
            "  WARNING: tallest measurement untrusted from every vantage - falling back to "
            f"the size-range max {hi_mm:.1f} mm as a conservative keep-out ceiling"
        )
        self.tallest_estimate = TallestEstimate(
            tallest_mm=hi_mm, points=0, trusted=False, reasons=["no trusted scan"]
        )
        self.tallest_source = "fallback"

    async def _place(
        self,
        grasp: Pose,
        lift: Pose,
        held_world_state: WorldState,
        carry_world_state: WorldState | None,
        free_world_state: WorldState,
    ) -> bool:
        """Carry the held block over the pad, descend as far as the planner
        allows, release, retreat, then report whether the block rests on the
        pad. Returns True once the block has been released here (the caller
        then skips its release)."""
        assert self.place_pad_top_mm is not None
        pad_x, pad_y, pad_top_z = self.place_pad_top_mm
        if self.place_offset_mm is not None:
            pad_x += self.place_offset_mm[0]
            pad_y += self.place_offset_mm[1]
        # reproduce the grasp configuration over the pad: same TCP-to-block
        # offset, the support is now the pad top, plus the drop gap
        place_z = grasp.z + (pad_top_z - self.support_z_mm) + self.place_clearance_mm
        pre_place_z = max(lift.z, place_z + PRE_GRASP_STANDOFF_MM)
        if self.place_clear_tcp_z_mm is not None:
            pre_place_z = max(pre_place_z, self.place_clear_tcp_z_mm)
        pre_place = _pointing_down(pad_x, pad_y, pre_place_z)
        if carry_world_state is not None:
            # keep-out carry (GPU run 12): hop above the no-fly box, then let
            # the planner move freely - it cannot enter the block airspace
            clear_above_support_mm = (
                self.measured_carry_clear_above_support_mm
                if self.measured_carry_clear_above_support_mm is not None
                else CARRY_CLEAR_ABOVE_SUPPORT_MM
            )
            clear = _pointing_down(lift.x, lift.y, self.support_z_mm + clear_above_support_mm)
            print(f"step: raise above the pick-area keep-out: {_pose_to_dict(clear)}")
            await self._move_or_diagnose(clear, held_world_state, linear=True)
            print(f"step: carry to pre-place (free, keep-out boxed): {_pose_to_dict(pre_place)}")
            try:
                await self.mover.move_to(pre_place, carry_world_state, False)
            except Exception as error:
                print(f"  keep-out carry failed ({error}); falling back to the linear carry")
                await self._move_or_diagnose(pre_place, held_world_state, linear=True)
        else:
            # no known scatter region to box off: carry along a straight line
            # at constant height so the held cube cannot dip (GPU run 11)
            carry_start = _pointing_down(lift.x, lift.y, pre_place.z)
            print(f"step: raise to carry height: {_pose_to_dict(carry_start)}")
            await self._move_or_diagnose(carry_start, held_world_state, linear=True)
            print(f"step: carry to pre-place: {_pose_to_dict(pre_place)}")
            try:
                await self.mover.move_to(pre_place, held_world_state, True)
            except Exception as error:
                print(f"  linear carry failed ({error}); replanning the carry freely")
                await self._move_or_diagnose(pre_place, held_world_state)

        # the current fragment's pad sits 783 mm from the arm base - near the
        # UR5e's reach boundary, where a constraint-locked straight descent can
        # have no continuous IK solution (GPU run 6). Try increasingly
        # permissive descents; failing all, drop from the hover pose onto the
        # pad (restitution 0 keeps the bounce small; the verdict reports it).
        half_z = (place_z + pre_place.z) / 2.0
        descents = [
            ("linear", _pointing_down(pad_x, pad_y, place_z), True),
            ("planned", _pointing_down(pad_x, pad_y, place_z), False),
            ("half-height linear", _pointing_down(pad_x, pad_y, half_z), True),
        ]
        stage = "hover_drop"
        release_pose = pre_place
        for label, pose, linear in descents:
            print(f"step: move to place ({label}): {_pose_to_dict(pose)}")
            try:
                await self.mover.move_to(pose, held_world_state, linear)
            except Exception as error:
                print(f"  place descent ({label}) failed: {error}")
                continue
            stage, release_pose = label, pose
            break
        if stage == "hover_drop":
            print("  every descent failed - releasing from the hover pose")

        print(f"step: open (place, {stage})")
        await self.gripper.open()
        if release_pose is not pre_place:
            print("step: retreat after place")
            try:
                await self.mover.move_to(pre_place, free_world_state, True)
            except Exception:
                await self.mover.move_to(pre_place, free_world_state, False)
        await self._report_placement(pad_x, pad_y, pad_top_z, stage)
        return True

    async def _report_placement(
        self, pad_x: float, pad_y: float, pad_top_z: float, stage: str
    ) -> None:
        if self.world is None:
            return
        await asyncio.sleep(PLACE_SETTLE_S)
        response = await self.world.do_command({"command": "prop_geometries"})
        block_name = self.target_prop_name or self.block_name
        block = next((g for g in response.get("geometries", []) if g["name"] == block_name), None)
        if block is None:
            print(f"  placement: prop {block_name!r} not found in prop_geometries")
            return
        pose = block["pose_in_world_mm"]
        assert self.resolved_block_size_mm is not None
        expected_z = pad_top_z + self.resolved_block_size_mm / 2.0
        placed_on_pad = (
            abs(pose["x"] - pad_x) <= PLACE_XY_TOLERANCE_MM
            and abs(pose["y"] - pad_y) <= PLACE_XY_TOLERANCE_MM
            and abs(pose["z"] - expected_z) <= PLACE_Z_TOLERANCE_MM
        )
        report = {
            "block_pose_mm": pose,
            "expected_z_mm": expected_z,
            "placed_on_pad": placed_on_pad,
            "place_stage": stage,
        }
        print(f"{PLACED_BLOCK_MARKER}{json.dumps(report, default=str, sort_keys=True)}")

    async def _set_ignored(self, names: Sequence[str]) -> None:
        """DEC-21 route (c): with sim-world's live GetGeometries in the frame
        system, the target block must not obstruct its own pick - ignore it
        for the run, restore afterwards."""
        if self.world is None or self.target_prop_name is None:
            return
        await self.world.do_command({"command": "ignore_props", "names": list(names)})
        print(f"step: ignore_props {list(names)}")

    async def run(self) -> Transform:
        await self._set_ignored([self.target_prop_name] if self.target_prop_name else [])
        try:
            return await self._run_steps()
        finally:
            await self._set_ignored([])

    def _expected_block_z_mm(self, block_size_mm: float) -> float:
        return self.support_z_mm + block_size_mm / 2.0

    def _is_resting_height(self, pose: Pose, block_size_mm: float) -> bool:
        if not self.verify_detection_height:
            return True
        return abs(pose.z - self._expected_block_z_mm(block_size_mm)) <= DETECT_Z_TOLERANCE_MM

    def _current_measurement(self) -> dict[str, Any] | None:
        """The most recent measurement from the detector's last
        ``block_pose_world`` call, or None when block_size_mm is explicit
        (no measurement happens) or the detector offers none."""
        if self.block_size_mm is not None:
            return None
        last_measurement = getattr(self.detector, "last_measurement", None)
        return last_measurement() if last_measurement is not None else None

    def _resolve_block_size_mm(self, measurement: dict[str, Any] | None) -> float | None:
        """The size in effect for this detection: block_size_mm when
        explicit, else the measured size, or None when the view was
        degenerate (no measurement to grasp on)."""
        if self.block_size_mm is not None:
            return self.block_size_mm
        return measurement["size_mm"] if measurement is not None else None

    async def _airspace_move(
        self,
        description: str,
        move: Callable[[Pose, WorldState, bool], Awaitable[None]],
        pose: Pose,
        look_world_state: WorldState,
        move_world_state: WorldState,
    ) -> None:
        """A move that approaches the detected block's airspace. The block is
        excluded from the obstacles, so a free plan may swing the arm through
        it (GPU phase-1 runs 2-3: the ur20 batted the block away on two
        different free segments). With the pick-area keep-out available the
        move plans FREELY boxed out of that airspace (fast joint motion, the
        GPU run 12 carry pattern - constrained linear moves crawl); without
        it the move is linear. Either way the other mode is the fallback."""
        boxed = look_world_state is not move_world_state
        try:
            if boxed:
                await move(pose, look_world_state, False)
            else:
                await move(pose, move_world_state, True)
        except Exception as error:  # noqa: BLE001 - planner failure, not a defect
            if boxed:
                print(f"  keep-out {description} failed ({error}); retrying it linear")
                await move(pose, look_world_state, True)
            else:
                print(f"  linear {description} failed ({error}); replanning it freely")
                await move(pose, move_world_state, False)

    async def _focus_on_block(
        self,
        block_pose: Pose,
        wrist_theta_deg: float,
        look_world_state: WorldState,
        move_world_state: WorldState,
    ) -> tuple[Pose, dict[str, Any] | None, float | None, Pose]:
        """Re-aim directly above ``block_pose`` at FOCUS_HEIGHT_ABOVE_SUPPORT_MM
        and measure again: the top-face estimate degrades off-centre (GPU run
        5: ~14 mm) and at the wide initial scan height (checklist item 3).
        Returns (block_pose, measurement, block_size_mm, focus_pose)."""
        focus = _pointing_down(
            block_pose.x,
            block_pose.y,
            self.support_z_mm + FOCUS_HEIGHT_ABOVE_SUPPORT_MM,
            theta_deg=wrist_theta_deg,
        )
        print(f"step: focus above the detected block: {_pose_to_dict(focus)}")
        await self._airspace_move(
            "focus move", self.mover.look_from, focus, look_world_state, move_world_state
        )
        print("step: detect (focused)")
        focused_pose = await self.detector.block_pose_world()
        print(f"  block_pose_world (mm): {_pose_to_dict(focused_pose)}")
        measurement = self._current_measurement()
        block_size_mm = self._resolve_block_size_mm(measurement)
        return focused_pose, measurement, block_size_mm, focus

    async def _detect_block(
        self, look_world_state: WorldState, move_world_state: WorldState
    ) -> tuple[Pose, Pose | None]:
        """Scan, detect, focus, and sanity-check the red block's pose. A
        detection whose height cannot be a resting block means the gripper's
        shadow swallowed it (GPU run 10: z 115 for a 60 mm cube), so the scan
        walks SCAN_ATTEMPTS instead of grasping at a phantom. When
        block_size_mm is None, a degenerate size measurement (footprint and
        height disagreeing) tries one focused look above the detected pose -
        the pose estimate is usable even when the size is not (checklist item
        3) - before walking the same ladder on a still-bad number (phase 3
        seam decision). Each attempt focuses at most once. Returns the block's
        pose and the look/focus pose the pipeline ended detection at, so a
        caller can tell whether the arm is already above the block."""
        for offset_x, offset_y, wrist_theta_deg in SCAN_ATTEMPTS:
            look_pose = self.look_pose
            if look_pose is not None and self.scan_centre_mm is not None:
                look_pose = _pointing_down(
                    self.scan_centre_mm[0] + offset_x,
                    self.scan_centre_mm[1] + offset_y,
                    look_pose.z,
                    theta_deg=wrist_theta_deg,
                )
            if look_pose is not None:
                print(f"step: look (scan the workspace from {_pose_to_dict(look_pose)})")
                if look_world_state is move_world_state:
                    await self.mover.look_from(look_pose, move_world_state)
                else:
                    try:
                        await self.mover.look_from(look_pose, look_world_state)
                    except Exception as error:  # noqa: BLE001
                        # an aborted earlier run can leave the arm inside the
                        # keep-out; replan the scan without it to get back out
                        print(
                            f"  keep-out scan move failed ({error}); "
                            "replanning without the keep-out"
                        )
                        await self.mover.look_from(look_pose, move_world_state)

            print("step: detect (from the stationary look pose)")
            block_pose = await self.detector.block_pose_world()
            print(f"  block_pose_world (mm): {_pose_to_dict(block_pose)}")
            detect_pose = look_pose
            measurement = self._current_measurement()
            block_size_mm = self._resolve_block_size_mm(measurement)
            if block_size_mm is None:
                if look_pose is not None and self.scan_centre_mm is not None:
                    (
                        block_pose,
                        measurement,
                        block_size_mm,
                        detect_pose,
                    ) = await self._focus_on_block(
                        block_pose, wrist_theta_deg, look_world_state, move_world_state
                    )
                if block_size_mm is None:
                    print(
                        "  degenerate size measurement (footprint/height disagree by more "
                        f"than {MEASURED_SIZE_DEGENERATE_FRACTION:.0%} of the median) - "
                        "re-scanning"
                    )
                    if self.scan_centre_mm is None:
                        break
                    continue
            elif (
                look_pose is not None
                and self.scan_centre_mm is not None
                and self._is_resting_height(block_pose, block_size_mm)
                and math.hypot(block_pose.x - look_pose.x, block_pose.y - look_pose.y)
                > FOCUS_LOOK_OFFSET_MM
            ):
                block_pose, measurement, block_size_mm, detect_pose = await self._focus_on_block(
                    block_pose, wrist_theta_deg, look_world_state, move_world_state
                )
                if block_size_mm is None:
                    print(
                        "  degenerate size measurement (footprint/height disagree by "
                        f"more than {MEASURED_SIZE_DEGENERATE_FRACTION:.0%} of the "
                        "median) - re-scanning"
                    )
                    if self.scan_centre_mm is None:
                        break
                    continue
            if self._is_resting_height(block_pose, block_size_mm):
                print(f"{DETECTED_BLOCK_POSE_MARKER}{json.dumps(_pose_to_dict(block_pose))}")
                self.resolved_block_size_mm = block_size_mm
                if measurement is not None:
                    report = dict(measurement)
                    report["scan_pose_mm"] = _pose_to_dict(detect_pose) if detect_pose else None
                    print(
                        f"{MEASURED_BLOCK_MARKER}{json.dumps(report, default=str, sort_keys=True)}"
                    )
                return block_pose, detect_pose
            print(
                f"  implausible detection: z {block_pose.z:.1f} vs expected "
                f"{self._expected_block_z_mm(block_size_mm):.1f} mm for a resting block "
                "- the gripper likely shadows it; re-scanning from an offset pose"
            )
            if self.scan_centre_mm is None:
                break
        raise RuntimeError(
            f"no plausible {self.block_label}-block detection from any scan pose - is the "
            "block visible and resting on its support?"
        )

    async def _run_steps(self) -> Transform:
        sim_obstacles = await self._sim_obstacles()
        table = table_recipe_unless_served(self.table, sim_obstacles)
        if table is None and self.table is not None:
            print(f"  --table dropped: the live scene already serves a {self.table.label!r} box")
        self.table = table
        move_world_state = world_state(
            self.table, (*self.other_blocks, *sim_obstacles), support_obstacle(self.support_z_mm)
        )
        # looks and airspace approaches plan freely but boxed out of the
        # blocks' airspace by the carry's pick-area keep-out (GPU run 12
        # pattern); without a known region they fall back to linear (slow)
        look_world_state = move_world_state
        if self.pick_region_mm is not None:
            look_world_state = world_state(
                self.table,
                (*self.other_blocks, *sim_obstacles, pick_area_keepout(self.pick_region_mm)),
                support_obstacle(self.support_z_mm),
            )
        if self.randomize_size_range_mm is not None:
            await self._measure_tallest(move_world_state)
        block_pose, detect_pose = await self._detect_block(look_world_state, move_world_state)
        assert self.resolved_block_size_mm is not None
        block_size_mm = self.resolved_block_size_mm
        if self.randomize_size_range_mm is not None:
            lo_mm, hi_mm = self.randomize_size_range_mm
            if not (
                lo_mm - DETECT_Z_TOLERANCE_MM <= block_size_mm <= hi_mm + DETECT_Z_TOLERANCE_MM
            ):
                print(
                    f"  WARNING: measured size {block_size_mm:.1f} mm falls outside the "
                    f"--randomize-size-mm range [{lo_mm:.1f}, {hi_mm:.1f}] mm "
                    f"(+/- {DETECT_Z_TOLERANCE_MM:.0f} mm tolerance)"
                )
        print(f"step: jaw check ({block_size_mm:.1f} mm measured vs {JAW_MAX_BLOCK_MM:.0f} mm jaw)")
        if block_size_mm > JAW_MAX_BLOCK_MM:
            raise JawLimitError(
                f"target block measures {block_size_mm:.1f} mm, wider than the gripper's "
                f"{JAW_MAX_BLOCK_MM:.0f} mm jaw limit (2F-85 85 mm open - 10 mm finger "
                "clearance) - refusing the grasp, arm left parked"
            )

        if self.tallest_estimate is not None:
            self.measured_keepout_height_mm = keepout_height_mm(
                self.tallest_estimate.tallest_mm, block_size_mm
            )
            self.measured_carry_clear_above_support_mm = carry_clear_above_support_mm(
                self.tallest_estimate.tallest_mm, block_size_mm
            )
            drawn_delta_mm = (
                self.tallest_estimate.tallest_mm - self.drawn_tallest_mm
                if self.drawn_tallest_mm is not None
                else None
            )
            marker = {
                "tallest_mm": self.tallest_estimate.tallest_mm,
                "source": self.tallest_source,
                "trusted": self.tallest_estimate.trusted,
                "reasons": self.tallest_estimate.reasons,
                "points": self.tallest_estimate.points,
                "scan_poses_mm": self.tallest_scan_poses_mm,
                "keepout_height_mm": self.measured_keepout_height_mm,
                "carry_clear_above_support_mm": self.measured_carry_clear_above_support_mm,
                "drawn_tallest_mm": self.drawn_tallest_mm,
                "drawn_delta_mm": drawn_delta_mm,
            }
            print(f"{MEASURED_TALLEST_MARKER}{json.dumps(marker, default=str, sort_keys=True)}")

        grasp_z = grasp_height_mm(
            block_pose.z, block_size_mm, self.support_z_mm, self.fingertip_overhang_mm
        )
        held_centre_below_tcp_mm = grasp_z - block_pose.z
        if grasp_z != block_pose.z:
            print(
                f"  grasp height raised from {block_pose.z:.1f} to {grasp_z:.1f} mm "
                f"(block size {block_size_mm:.0f}, support z {self.support_z_mm:.0f}, "
                f"fingertip overhang {self.fingertip_overhang_mm:.0f})"
            )
            block_pose = with_z(block_pose, grasp_z)

        print("step: open")
        await self.gripper.open()

        # go directly above the block first when detection did not already end
        # there; below the waypoint the descent is linear (_airspace_move says
        # why free plans must stay boxed out of the block's airspace)
        waypoint = _pointing_down(
            block_pose.x, block_pose.y, self.support_z_mm + FOCUS_HEIGHT_ABOVE_SUPPORT_MM
        )
        if not _poses_match_mm(detect_pose, waypoint, AIRSPACE_WAYPOINT_TOLERANCE_MM):
            print(f"step: move to block-airspace waypoint: {_pose_to_dict(waypoint)}")
            await self._airspace_move(
                "waypoint move", self.mover.move_to, waypoint, look_world_state, move_world_state
            )

        pre_grasp = pre_grasp_pose(block_pose, self.standoff_mm)
        print(f"step: move to pre-grasp: {_pose_to_dict(pre_grasp)}")
        await self._move_or_diagnose(pre_grasp, move_world_state, linear=True)

        grasp = grasp_pose(block_pose)
        lift = pre_grasp
        if self.tcp_correction is not None:
            delta = await self.tcp_correction()
            print(
                f"  tcp correction at pre-grasp (believed - physical, mm): "
                f"({delta[0]:.1f}, {delta[1]:.1f}, {delta[2]:.1f})"
            )
            grasp = corrected_pose(grasp, delta)
            lift = corrected_pose(pre_grasp, delta)
        print(f"step: move to grasp: {_pose_to_dict(grasp)}")
        await self._move_or_diagnose(grasp, move_world_state, linear=True)

        print("step: grab")
        grabbed = await self.gripper.grab()
        print(f"  grab: {grabbed}")
        if not grabbed:
            if self.diagnose is not None:
                report = await self.diagnose()
                print(f"{GRAB_DIAGNOSTICS_MARKER}{json.dumps(report, default=str, sort_keys=True)}")
            raise RuntimeError(
                f"grab failed: gripper {self.gripper_name!r} reported no hold at "
                f"{_pose_to_dict(grasp)}"
            )

        transform = held_block_transform(
            self.block_name,
            block_size_mm,
            self.gripper_name,
            centre_below_tcp_mm=held_centre_below_tcp_mm,
        )
        padded_transform = held_block_transform(
            self.block_name,
            block_size_mm + HELD_BLOCK_PADDING_MM,
            self.gripper_name,
            centre_below_tcp_mm=held_centre_below_tcp_mm,
        )
        held_world_state = world_state(
            self.table,
            (*self.other_blocks, *sim_obstacles),
            support_obstacle(self.support_z_mm),
            transforms=(padded_transform,),
        )
        carry_keepouts: list[Geometry] = []
        if self.pick_region_mm is not None:
            pick_keepout_height_mm = (
                self.measured_keepout_height_mm
                if self.measured_keepout_height_mm is not None
                else KEEPOUT_HEIGHT_MM
            )
            carry_keepouts.append(pick_area_keepout(self.pick_region_mm, pick_keepout_height_mm))
        if self.place_region_mm is not None and self.placed_tallest_mm:
            place_keepout_height_mm = self.placed_tallest_mm + PLACED_KEEPOUT_CLEARANCE_MM
            carry_keepouts.append(
                pick_area_keepout(
                    self.place_region_mm,
                    place_keepout_height_mm,
                    label="place_area_keepout",
                )
            )
            # the pre-place hover must hold the held block's BOTTOM above the
            # keep-out box, or the hover pose itself sits inside the obstacle
            keepout_top_z_mm = float(self.place_region_mm[0][2]) + place_keepout_height_mm
            self.place_clear_tcp_z_mm = (
                keepout_top_z_mm + held_centre_below_tcp_mm + block_size_mm / 2.0
            )
        carry_world_state = None
        if carry_keepouts:
            carry_world_state = world_state(
                self.table,
                (*self.other_blocks, *sim_obstacles, *carry_keepouts),
                support_obstacle(self.support_z_mm),
                transforms=(padded_transform,),
            )
        print(f"step: move to lift: {_pose_to_dict(lift)}")
        await self._move_or_diagnose(lift, held_world_state, linear=True)

        if self.hold_s > 0:
            print(f"step: hold at the lift pose for {self.hold_s:.0f} s (checklist item 5)")
            samples: list[dict[str, Any]] = []
            for _ in range(max(1, round(self.hold_s / HOLD_SAMPLE_S))):
                await asyncio.sleep(HOLD_SAMPLE_S)
                status = await self.gripper.is_holding_something()
                meta = dict(getattr(status, "meta", None) or {})
                samples.append(
                    {"holding": bool(status.is_holding_something), "jaw_deg": meta.get("jaw_deg")}
                )
            print(f"{HOLD_SAMPLES_MARKER}{json.dumps(samples, default=str)}")
            if not all(sample["holding"] for sample in samples):
                if self.diagnose is not None:
                    report = await self.diagnose()
                    payload = json.dumps(report, default=str, sort_keys=True)
                    print(f"{GRAB_DIAGNOSTICS_MARKER}{payload}")
                raise RuntimeError(
                    f"hold failed: is_holding_something sampled {samples} at the lift pose"
                )

        if self.mid_hold_reset is not None:
            print("step: world reset mid-hold (checklist item 6)")
            reset_report = await self.mid_hold_reset()
            payload = json.dumps(reset_report, default=str, sort_keys=True)
            print(f"{RESET_MID_HOLD_MARKER}{payload}")
            if reset_report.get("holding_after_reset"):
                print("  reset mid-hold: the grip survived the reset")
            else:
                # GPU run 26: isaac's world.reset() returns every scene-registered
                # prim (props included) to its spawn state, so "not holding" after
                # a reset is the designed outcome, not a dropped grip. The JSON's
                # block_prim_pose says where the block went - a human judges it.
                print(
                    "  reset mid-hold: not holding after the reset (isaac's world.reset() "
                    "returns props to their spawn state; judge RESET_MID_HOLD_JSON)"
                )

        transform_json = json.dumps(
            MessageToDict(transform, preserving_proto_field_name=True), sort_keys=True
        )
        print("held-block transform (DEC-14):")
        print(f"{HELD_BLOCK_TRANSFORM_MARKER}{transform_json}")

        placed = False
        if self.place_pad_top_mm is not None and self.mid_hold_reset is None:
            placed = await self._place(
                grasp, lift, held_world_state, carry_world_state, move_world_state
            )
        elif self.place_prop_name is not None and self.place_pad_top_mm is None:
            print(f"step: place skipped - no prop {self.place_prop_name!r} in the scene")
        if not placed:
            print("step: open (release)")
            await self.gripper.open()
        print("done")
        return transform
