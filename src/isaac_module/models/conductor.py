"""viam:isaac-sim-devin:conductor - the generic service that sorts one scatter
end to end: scatter (optional), one source-zone scan to build the work list,
then nearest-first pick-verify-carry-place of every block onto its color's
pad, with no python script in the loop.

Attributes:
  world (string, required)       - name of the viam:isaac-sim-devin:world component
  arm (string, required)         - name of the arm component (boot ordering only;
                                    every motion goes through "motion", DEC-13)
  gripper (string, required)     - name of the gripper component
  camera (string, required)      - name of the wrist camera
  side_camera (string, required) - name of the fixed side camera
  motion (string, required)      - name of the motion service ("builtin" works)
  detectors (object, required)   - {color: segmenter vision-service name} for
                                    exactly cell_layout.BLOCK_COLORS (red maps
                                    to the existing "block-segmenter"; the five
                                    new colors use "<color>-segmenter")
  size_range_mm ([lo, hi])       - default [50, 80]; hi must be <=
                                    cell_layout.MAX_BLOCK_SIZE_MM (80)

DoCommand:
  {"command": "start", "seed"?: int, "counts"?: {color: int}, "loops"?: int,
    "continuous"?: bool} -> neither "loops" nor "continuous" is phase-4
    single-shot (with "seed", scatters via the world's scatter_cell first;
    without, sorts the standing scatter), recorded in telemetry as one loop.
    "loops": N >= 1 runs N loops; "loops": 0 or "continuous": true runs
    until "stop". Loop mode always resets each loop (including the first)
    with clear_cell then scatter_cell at a per-loop seed derived from
    "seed" (or a time-derived base when absent). {"ok": true,
    "state": "running"}, or {"ok": false, "state": "running"} unchanged
    when already running |
  {"command": "stop"} -> cancels between motions AND between passes (never
    mid-motion) and at loop boundaries, {"ok": true} |
  {"command": "status"} -> {"state": "idle|running|stopping|complete|failed",
    "remaining": [names], "current": name|null,
    "outcomes": {name: {"outcome": "placed|skipped_oversize|failed",
    "prim"?: str, "reason"?: str, "attempts"?: int}}, "seed": int|null,
    "pass": int, "run": {"loops_requested": int|0-for-continuous|null,
    "continuous": bool, "loop": int, "loops_completed": int,
    "loops_errored": int, "base_seed": int|null, "placed": int,
    "failed": int, "skipped_oversize": int}, "success_rate": float|null,
    "loop_records": [LoopRecord.to_dict(), ...]}. A loop killed by a
    transient failure (e.g. a dropped gRPC stream to viam-server) is
    recorded with an "error" field and skipped - the run continues with
    the next loop's seed, and only MAX_CONSECUTIVE_LOOP_ERRORS (3) such
    loops in a row fail the run. Single-shot keeps phase-4 semantics: any
    exception fails the run.

Sorting is multi-pass (phase-4e): a dense scatter can leave a crowded block
genuinely unpickable until a neighbour clears (GPU evidence - the gripper's
descent corridor clips a still-present neighbour at typical spacing), so one
pass = census -> resolve prims -> a clearance-ordered attempt loop (isolated
blocks first, per ``sort_plan.clearance_ordered``), skipping any prim already
placed or skipped-oversize in an earlier pass (oversize never shrinks, so it
is never retried). "pass" in status is the 1-indexed count of census passes
run so far. A pass ends the sort when every attempted item this pass placed
or was skipped (no failures), when the pass placed nothing at all (no
progress to justify another census), or after 5 passes (a hard cap against
runaway re-censusing). Outcomes are keyed by the resolved prim name once one
is known (the physical identity that persists across a re-census), so a
later pass's success overwrites that prim's earlier failure; an item that
never resolves to a prim (a phantom/duplicate segment) keeps its detection
name as the key, since it has no prim identity to key by.

Every run is bookended by a move to ``PARK_POSE_TCP_MM``: once before the
first census, and once (best-effort, its own failure never changes the run's
outcome) after the final pass, whether the run completed, stopped, or
failed.

DEC-4: vision alone decides what to pick and where (pose/size come only from
detections, never sim truth). The world's ``prop_geometries`` is consulted
twice, both times for bookkeeping only: building planner obstacles, and (after
the scan) resolving each detection to the real scattered prim nearest its
detected position - the name the pipeline needs for ``ignore_props`` and the
held-block transform, and the color that decides the destination pad when a
detector misclassifies (the prim wins, not the detector).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from typing_extensions import Self
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, WorldState
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.utils import ValueTypes, struct_to_dict

from pickcell.detector import RealDetector
from pickcell.measurement import (
    footprint_extents_mm,
    measured_block_size_mm,
    parse_pcd,
    top_face_centre_m,
)
from pickcell.movers import RealMover
from pickcell.obstacles import obstacles_from_prop_geometries, support_obstacle, world_state
from pickcell.pipeline import (
    JAW_MAX_BLOCK_MM,
    Detector,
    GripperApi,
    JawLimitError,
    Mover,
    PickPipeline,
    WorldApi,
)
from pickcell.poses import (
    FOCUS_HEIGHT_ABOVE_SUPPORT_MM,
    SCAN_HEIGHT_ABOVE_SUPPORT_MM,
    _pointing_down,
)

from .. import FAMILY, NAMESPACE, cell_layout
from ..run_log import (
    MAX_ATTEMPTS_PER_BLOCK,
    MAX_CONSECUTIVE_LOOP_ERRORS,
    LoopRecord,
    PickRecord,
    RollingLog,
    current_rss_mb,
)
from ..run_log import (
    loop_seed as compute_loop_seed,
)
from ..run_log import (
    success_rate as compute_success_rate,
)
from ..sort_plan import (
    OUTCOME_FAILED,
    OUTCOME_PLACED,
    OUTCOME_SKIPPED_OVERSIZE,
    SlotTracker,
    WorkItem,
    clearance_ordered,
)
from ..spatial import ov_to_quat, quat_rotate

LOGGER = getLogger(__name__)

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_COMPLETE = "complete"
STATE_FAILED = "failed"

DEFAULT_SIZE_RANGE_MM: tuple[float, float] = (50.0, 80.0)

# a hard cap on re-census passes, against a scatter so dense it never fully
# clears - after this many passes the sort finishes with whatever remains
# recorded as failed, rather than re-censusing forever
MAX_PASSES = 5

# a neutral bookend pose, over the arm table's scatter side (planar radius
# 500 mm from the arm base - comfortably mid-envelope): without it, the
# first move of a run was a long free plan from wherever the previous run
# ended (over the place pads, ~1.4 m away) to the first census pose, legal
# to swing through wild joint reconfigurations (GPU evidence). z is 550 mm
# above the table top (1300 - 750), clearing the pick-area keep-out ceiling
# (table top + 130) by 420 mm, and x=-500 sits 150 mm outside the scatter
# zone's own keep-out footprint (which starts at x=-650 with margin)
PARK_POSE_TCP_MM: tuple[float, float, float] = (-500.0, 0.0, 1300.0)

_DEPENDENCY_ATTRS = ("world", "arm", "gripper", "camera", "side_camera", "motion")

# a detection resolves to the nearest scattered pool-block prim within this
# planar radius; half a block's own footprint at the cell's size ceiling, so
# two distinct blocks can never both claim the same prim
POOL_PRIM_MATCH_TOLERANCE_MM = cell_layout.MAX_BLOCK_SIZE_MM / 2.0

_POOL_PRIM_COLORS: dict[str, str] = {
    cell_layout.pool_block_name(color, index): color
    for color in cell_layout.BLOCK_COLORS
    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
}

# GPU findings (phase-4 Notes): the arm covers up to 26% of a single census
# frame and can occlude whole blocks; the arm also renders blue-gray and can
# produce phantom segments at arm height. A kept census item's top face must
# sit just above the table (clear of glints) and well below the arm's resting
# height - a real block never measures taller than MAX_BLOCK_SIZE_MM.
_CENSUS_MIN_TOP_Z_MM = cell_layout.TABLE_TOP_Z_MM + 20.0
_CENSUS_MAX_TOP_Z_MM = cell_layout.TABLE_TOP_Z_MM + cell_layout.MAX_BLOCK_SIZE_MM + 15.0

# arm silhouette occlusion (Notes): three census look poses instead of one -
# the scatter-zone centre plus centre +/- this offset - so a block hidden
# from one vantage is visible from another (the arm parks differently per
# approach). The offset stays inside the safe-reach guard: planar radius from
# the arm base is hypot(1275, 150) =~ 1284 mm, under MAX_PLANAR_REACH_MM
# (1400 mm = cell_layout.REACH_SAFETY_FRACTION of UR20_REACH_MM).
_CENSUS_LOOK_OFFSET_MM: tuple[float, float] = (250.0, 150.0)

# items whose planar positions land within this radius are the same block,
# whether seen from different poses or (yellow/orange) different detectors
CENSUS_PROXIMITY_MERGE_TOLERANCE_MM = 40.0


def _place_zone_mm() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The place-pad envelope for the pipeline's place-zone keep-out, with z
    at the pad top (the surface placed blocks stand on)."""
    return (
        (
            cell_layout.PLACE_ZONE_X_MM[0],
            cell_layout.PLACE_ZONE_Y_MM[0],
            cell_layout.PAD_TOP_Z_MM,
        ),
        (
            cell_layout.PLACE_ZONE_X_MM[1],
            cell_layout.PLACE_ZONE_Y_MM[1],
            cell_layout.PAD_TOP_Z_MM,
        ),
    )


def _census_look_points_mm() -> tuple[tuple[float, float], ...]:
    """The three census look-pose (x, y) points: the scatter-zone centre and
    centre +/- ``_CENSUS_LOOK_OFFSET_MM``."""
    centre_x, centre_y, _z = cell_layout.SCATTER_CENTRE_MM
    dx, dy = _CENSUS_LOOK_OFFSET_MM
    return ((centre_x, centre_y), (centre_x + dx, centre_y + dy), (centre_x - dx, centre_y - dy))


def _within_census_z_band(top_z_mm: float) -> bool:
    """Whether a detected top-face z is in the band real blocks occupy - too
    low is a table glint, too high is the arm (vision-only, DEC-4)."""
    return _CENSUS_MIN_TOP_Z_MM <= top_z_mm <= _CENSUS_MAX_TOP_Z_MM


@dataclass(frozen=True)
class _CensusHit:
    """One census detection paired with how oblique its view was - the
    planar distance from the look pose that produced it to its own detected
    position, the tie-break proximity dedup uses to pick the least-oblique
    measurement among duplicates."""

    item: WorkItem
    view_distance_mm: float


def _merge_census_hits(hits: list[_CensusHit]) -> list[WorkItem]:
    """Merge hits whose planar positions lie within
    ``CENSUS_PROXIMITY_MERGE_TOLERANCE_MM`` of each other - across poses AND
    across detectors, since yellow/orange deliberately both fire on both
    families now (cell_layout.RENDERED_BLOCK_HUE_DEG). The merge only needs to
    be deterministic, never clever about color: downstream prim-color routing
    already decides the true destination. Each connected group keeps the hit
    with the least oblique view, ties broken by work-item name."""
    parent = list(range(len(hits)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            distance = math.hypot(
                hits[i].item.x_mm - hits[j].item.x_mm, hits[i].item.y_mm - hits[j].item.y_mm
            )
            if distance <= CENSUS_PROXIMITY_MERGE_TOLERANCE_MM:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(hits)):
        groups.setdefault(find(index), []).append(index)

    merged: list[WorkItem] = []
    for indices in groups.values():
        winner = min(
            indices, key=lambda index: (hits[index].view_distance_mm, hits[index].item.name)
        )
        merged.append(hits[winner].item)
    return merged


def _validate_size_range_mm(resource_name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f'{resource_name}: "size_range_mm" must be [lo, hi]')
    lo, hi = value
    if not isinstance(lo, (int, float)) or isinstance(lo, bool):
        raise ValueError(f'{resource_name}: "size_range_mm" entries must be numbers')
    if not isinstance(hi, (int, float)) or isinstance(hi, bool):
        raise ValueError(f'{resource_name}: "size_range_mm" entries must be numbers')
    lo_f, hi_f = float(lo), float(hi)
    if not (0.0 < lo_f <= hi_f):
        raise ValueError(
            f'{resource_name}: "size_range_mm" [{lo_f}, {hi_f}] must satisfy 0 < lo <= hi'
        )
    if hi_f > cell_layout.MAX_BLOCK_SIZE_MM:
        raise ValueError(
            f'{resource_name}: "size_range_mm" hi {hi_f} exceeds MAX_BLOCK_SIZE_MM '
            f"{cell_layout.MAX_BLOCK_SIZE_MM}"
        )
    return lo_f, hi_f


class _CameraFrameTransform:
    """Adapts the motion service's ``get_pose`` into the ``robot.transform_pose``
    shape ``pickcell.detector.RealDetector`` expects. DEC-13 (every motion
    through the motion service) plus the conductor never importing
    ``viam.robot.client`` rules out the usual self-connected RobotClient this
    library otherwise assumes; ``get_pose`` on the already-required motion
    dependency gives the same camera-to-world pose the frame system would."""

    def __init__(self, motion: MotionClient) -> None:
        self._motion = motion

    async def transform_pose(
        self, pose_in_frame: PoseInFrame, destination_frame: str
    ) -> PoseInFrame:
        frame_pose = (
            await self._motion.get_pose(pose_in_frame.reference_frame, destination_frame)
        ).pose
        quat = ov_to_quat(
            frame_pose.o_x, frame_pose.o_y, frame_pose.o_z, math.radians(frame_pose.theta)
        )
        local = pose_in_frame.pose
        dx, dy, dz = quat_rotate(quat, (local.x, local.y, local.z))
        return PoseInFrame(
            reference_frame=destination_frame,
            pose=Pose(
                x=frame_pose.x + dx,
                y=frame_pose.y + dy,
                z=frame_pose.z + dz,
                o_x=0.0,
                o_y=0.0,
                o_z=1.0,
                theta=0.0,
            ),
        )


class IsaacConductor(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the service, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "conductor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._state: str = STATE_IDLE
        self._task: asyncio.Task[None] | None = None
        self._cancel_requested = False
        self._current: str | None = None
        self._remaining: list[str] = []
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._pass: int = 0
        self._seed: int | None = None
        self._size_range_mm: tuple[float, float] = DEFAULT_SIZE_RANGE_MM
        # loop/telemetry state (phase 5): the rolling log and record-id
        # counter live for the module's lifetime and are never reset by
        # "start" (Notes ruling); the "_run_*" fields are run-cumulative and
        # reset on every "start"
        self._rolling_log = RollingLog()
        self._record_id_counter: int = 0
        self._run_loops_requested: int | None = None
        self._run_continuous: bool = False
        self._run_loop_number: int = 0
        self._run_loops_completed: int = 0
        self._run_loops_errored: int = 0
        self._run_base_seed: int | None = None
        self._run_placed: int = 0
        self._run_failed: int = 0
        self._run_skipped_oversize: int = 0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        conductor = cls(config.name)
        conductor.reconfigure(config, dependencies)
        return conductor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        dependencies: list[str] = []
        for key in _DEPENDENCY_ATTRS:
            value = attrs.get(key)
            if not value or not isinstance(value, str):
                raise ValueError(f'{config.name}: set the "{key}" attribute to a resource name')
            dependencies.append(value)

        detectors = attrs.get("detectors")
        if not isinstance(detectors, Mapping):
            raise ValueError(
                f'{config.name}: "detectors" must map each of '
                f"{list(cell_layout.BLOCK_COLORS)} to a segmenter vision-service name"
            )
        missing = [color for color in cell_layout.BLOCK_COLORS if color not in detectors]
        if missing:
            raise ValueError(f'{config.name}: "detectors" is missing colors: {missing}')
        unknown = sorted(set(detectors) - set(cell_layout.BLOCK_COLORS))
        if unknown:
            raise ValueError(f'{config.name}: "detectors" has unknown colors: {unknown}')
        for color in cell_layout.BLOCK_COLORS:
            name = detectors[color]
            if not name or not isinstance(name, str):
                raise ValueError(f"{config.name}: detectors[{color!r}] must be a non-empty string")
            dependencies.append(str(name))

        size_range = attrs.get("size_range_mm", list(DEFAULT_SIZE_RANGE_MM))
        _validate_size_range_mm(config.name, size_range)
        return dependencies, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
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
        self._gripper = cast("GripperApi", dep("gripper"))
        self._gripper_name = str(attrs["gripper"])
        self._camera_name = str(attrs["camera"])
        self._side_camera_name = str(attrs["side_camera"])
        self._motion = cast(MotionClient, dep("motion"))

        detectors_attr = cast("Mapping[str, str]", attrs.get("detectors", {}))
        self._vision_by_color: dict[str, VisionClient] = {
            color: cast(VisionClient, by_name[str(detectors_attr[color])])
            for color in cell_layout.BLOCK_COLORS
        }

        size_range = attrs.get("size_range_mm", list(DEFAULT_SIZE_RANGE_MM))
        self._size_range_mm = _validate_size_range_mm(config.name, size_range)
        self._camera_transform = _CameraFrameTransform(self._motion)

    # -- background-task lifecycle -----------------------------------------

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Mapping[str, ValueTypes]:
        cmd = str(command.get("command", ""))
        if cmd == "start":
            return await self._handle_start(command)
        if cmd == "stop":
            return self._handle_stop()
        if cmd == "status":
            return cast("Mapping[str, ValueTypes]", self._status_snapshot())
        raise ValueError(f"unknown command {cmd!r}; supported: start, stop, status")

    async def _handle_start(self, command: Mapping[str, ValueTypes]) -> dict[str, ValueTypes]:
        if self._state in (STATE_RUNNING, STATE_STOPPING):
            return {"ok": False, "state": self._state}

        seed_value = command.get("seed")
        seed = (
            int(cast(Any, seed_value))
            if isinstance(seed_value, (int, float)) and not isinstance(seed_value, bool)
            else None
        )
        counts = command.get("counts")
        counts_arg = (
            {str(color): int(cast(Any, n)) for color, n in counts.items()}
            if isinstance(counts, Mapping)
            else None
        )

        loops_value = command.get("loops")
        continuous_value = command.get("continuous")
        loop_mode = loops_value is not None or continuous_value is True
        if loop_mode:
            loops_int = (
                int(cast(Any, loops_value))
                if isinstance(loops_value, (int, float)) and not isinstance(loops_value, bool)
                else 0
            )
            continuous = continuous_value is True or loops_int == 0
            loops_requested: int | None = 0 if continuous else loops_int
            # scatter_cell requires a seed; unattended continuous/loop runs
            # should need no arguments (Notes ruling), so an absent seed
            # falls back to a time-derived base, reported in status.run
            base_seed: int | None = seed if seed is not None else int(time.time()) % 1_000_000
        else:
            continuous = False
            loops_requested = None
            base_seed = seed

        self._seed = seed
        self._state = STATE_RUNNING
        self._current = None
        self._remaining = []
        self._outcomes = {}
        self._pass = 0
        self._cancel_requested = False
        self._run_loops_requested = loops_requested
        self._run_continuous = continuous
        self._run_loop_number = 0
        self._run_loops_completed = 0
        self._run_loops_errored = 0
        self._run_base_seed = base_seed
        self._run_placed = 0
        self._run_failed = 0
        self._run_skipped_oversize = 0
        self._task = asyncio.create_task(
            self._run(seed, counts_arg, loops_requested, continuous, base_seed)
        )
        return {"ok": True, "state": "running"}

    def _handle_stop(self) -> dict[str, ValueTypes]:
        if self._state == STATE_RUNNING:
            self._state = STATE_STOPPING
        self._cancel_requested = True
        return {"ok": True}

    def _status_snapshot(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "remaining": list(self._remaining),
            "current": self._current,
            "outcomes": {name: dict(outcome) for name, outcome in self._outcomes.items()},
            "seed": self._seed,
            "pass": self._pass,
            "run": {
                "loops_requested": self._run_loops_requested,
                "continuous": self._run_continuous,
                "loop": self._run_loop_number,
                "loops_completed": self._run_loops_completed,
                "loops_errored": self._run_loops_errored,
                "base_seed": self._run_base_seed,
                "placed": self._run_placed,
                "failed": self._run_failed,
                "skipped_oversize": self._run_skipped_oversize,
            },
            "success_rate": compute_success_rate(self._run_placed, self._run_failed),
            "loop_records": [record.to_dict() for record in self._rolling_log.records()],
        }

    def _next_record_id(self) -> int:
        """Monotonic for the module's lifetime; never reset by ``start``."""
        self._record_id_counter += 1
        return self._record_id_counter

    async def wait_until_done(self) -> None:
        """Test-only join on the background sort task started by ``start``.
        Never awaited from ``do_command``: production callers poll ``status``."""
        if self._task is not None:
            await self._task

    # -- the sort run itself --------------------------------------------

    async def _run(
        self,
        seed: int | None,
        counts: dict[str, int] | None,
        loops_requested: int | None,
        continuous: bool,
        base_seed: int | None,
    ) -> None:
        # loops_requested is None for phase-4 single-shot semantics (exactly
        # one loop, no clear/scatter unless a seed was given); otherwise loop
        # mode always resets with clear_cell + scatter_cell, every loop
        # including the first (Notes ruling)
        loop_mode = loops_requested is not None
        try:
            await self._park()
            loop_index = 0
            consecutive_loop_errors = 0
            while True:
                if self._cancel_requested:
                    break
                self._run_loop_number = loop_index + 1

                loop_start = time.monotonic()
                try:
                    if loop_mode:
                        assert base_seed is not None  # scatter_cell requires a seed in loop mode
                        current_seed = compute_loop_seed(base_seed, loop_index)
                        await self._world.do_command({"command": "clear_cell"})
                        await self._scatter(current_seed, counts)
                        self._seed = current_seed
                    elif seed is not None:
                        await self._scatter(seed, counts)
                        self._seed = seed
                    else:
                        self._seed = None

                    self._current = None
                    self._remaining = []
                    self._outcomes = {}
                    self._pass = 0

                    picks = await self._run_one_loop()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # a transient failure (a dropped gRPC stream to viam-server,
                    # GPU run 7) costs one loop, never the run: record it,
                    # advance the seed, and try the next loop
                    consecutive_loop_errors += 1
                    self._record_loop(
                        LoopRecord(
                            record_id=self._next_record_id(),
                            loop=self._run_loop_number,
                            seed=self._seed,
                            duration_s=time.monotonic() - loop_start,
                            passes=self._pass,
                            picks=(),
                            error=f"{type(error).__name__}: {error}",
                            rss_mb=current_rss_mb(),
                        )
                    )
                    self._run_loops_errored += 1
                    if not loop_mode or consecutive_loop_errors >= MAX_CONSECUTIVE_LOOP_ERRORS:
                        raise
                    LOGGER.exception(
                        "loop %d failed (%d consecutive); continuing with the next loop",
                        self._run_loop_number,
                        consecutive_loop_errors,
                    )
                    if self._cancel_requested:
                        break
                    loop_index += 1
                    if not continuous and loop_index >= cast(int, loops_requested):
                        break
                    continue
                consecutive_loop_errors = 0
                loop_duration_s = time.monotonic() - loop_start

                loop_record = LoopRecord(
                    record_id=self._next_record_id(),
                    loop=self._run_loop_number,
                    seed=self._seed,
                    duration_s=loop_duration_s,
                    passes=self._pass,
                    picks=tuple(picks),
                    rss_mb=current_rss_mb(),
                )
                self._record_loop(loop_record)
                self._run_placed += loop_record.placed
                self._run_failed += loop_record.failed
                self._run_skipped_oversize += loop_record.skipped_oversize
                self._run_loops_completed += 1

                if self._cancel_requested:
                    break
                if not loop_mode:
                    break
                loop_index += 1
                if not continuous and loop_index >= cast(int, loops_requested):
                    break
            self._state = STATE_IDLE if self._cancel_requested else STATE_COMPLETE
        except Exception:
            self._state = STATE_FAILED
            LOGGER.exception("conductor sort run failed")
        finally:
            self._cancel_requested = False
            try:
                await self._park()
            except Exception:  # noqa: BLE001 - best-effort: never changes the run's outcome
                LOGGER.exception("conductor end-of-run park move failed")

    async def _scatter(self, seed: int, counts: dict[str, int] | None) -> None:
        """Issues one ``scatter_cell`` at ``seed`` with the config
        ``size_range_mm``, ``counts`` passed through when given."""
        scatter_command: dict[str, Any] = {
            "command": "scatter_cell",
            "seed": seed,
            "size_range_mm": list(self._size_range_mm),
        }
        if counts is not None:
            scatter_command["counts"] = counts
        await self._world.do_command(scatter_command)

    async def _run_one_loop(self) -> list[PickRecord]:
        """The phase-4 census -> clearance-ordered attempt loop, run once for
        the current loop's standing scatter. Failure policy (phase 5): a
        pipeline failure is retried on a later pass while its resolved
        prim's attempt count is under ``MAX_ATTEMPTS_PER_BLOCK``; at the cap
        it is terminal for this loop. Returns the ``PickRecord``s for every
        resolved (non-phantom) outcome, for the loop's telemetry record."""
        slot_tracker = SlotTracker()
        handled_prims: set[str] = set()
        # tallest block standing on the pads so far this loop (census-measured
        # size at place time), sizing the pipeline's place-zone keep-out
        placed_tallest_mm = 0.0
        attempts_by_prim: dict[str, int] = {}
        color_by_prim: dict[str, str] = {}
        first_attempt_at: dict[str, float] = {}
        # the timestamp right after each attempt's outcome lands - overwritten
        # on every attempt, so a block whose retry never runs (the pass loop
        # ended first) keeps its last attempt's end as its terminal time,
        # rather than the whole loop's end absorbing later passes into a
        # block that already went terminal in an earlier one
        last_attempt_at: dict[str, float] = {}
        pass_number = 0
        while True:
            pass_number += 1
            self._pass = pass_number
            work_items = await self._scan_work_list()
            matched, phantom_outcomes = await self._resolve_prims(work_items)
            self._outcomes.update(phantom_outcomes)

            pending = [
                (item, prim_name, prim_color)
                for item, prim_name, prim_color in matched
                if prim_name not in handled_prims
            ]
            prim_by_item = {
                item.name: (prim_name, prim_color) for item, prim_name, prim_color in pending
            }
            ordered = clearance_ordered(
                [item for item, _prim_name, _prim_color in pending],
                cell_layout.ARM_BASE_XY_MM,
            )
            self._remaining = [item.name for item in ordered]

            placed_this_pass = 0
            failed_this_pass = 0
            for item in ordered:
                if self._cancel_requested:
                    break
                self._remaining.pop(0)
                self._current = item.name
                prim_name, prim_color = prim_by_item[item.name]
                color_by_prim[prim_name] = prim_color
                first_attempt_at.setdefault(prim_name, time.monotonic())
                attempts_by_prim[prim_name] = attempts_by_prim.get(prim_name, 0) + 1
                attempts = attempts_by_prim[prim_name]

                outcome = dict(
                    await self._run_one(
                        item, prim_name, prim_color, slot_tracker, placed_tallest_mm or None
                    )
                )
                last_attempt_at[prim_name] = time.monotonic()
                if outcome["outcome"] == OUTCOME_PLACED:
                    outcome["attempts"] = attempts
                    handled_prims.add(prim_name)
                    placed_this_pass += 1
                    placed_tallest_mm = max(placed_tallest_mm, item.size_mm)
                elif outcome["outcome"] == OUTCOME_SKIPPED_OVERSIZE:
                    handled_prims.add(prim_name)
                elif outcome["outcome"] == OUTCOME_FAILED:
                    outcome["attempts"] = attempts
                    failed_this_pass += 1
                    if attempts >= MAX_ATTEMPTS_PER_BLOCK:
                        handled_prims.add(prim_name)
                self._outcomes[prim_name] = outcome
                self._current = None

            if self._cancel_requested:
                break
            if failed_this_pass == 0 or placed_this_pass == 0 or pass_number >= MAX_PASSES:
                break

        picks: list[PickRecord] = []
        for prim_name, outcome in self._outcomes.items():
            if "prim" not in outcome:
                continue  # phantom/duplicate segment - no prim identity
            started_at = first_attempt_at.get(prim_name)
            duration_s = 0.0
            if started_at is not None:
                duration_s = last_attempt_at.get(prim_name, started_at) - started_at
            picks.append(
                PickRecord(
                    name=prim_name,
                    color=color_by_prim.get(prim_name, ""),
                    outcome=outcome["outcome"],
                    attempts=attempts_by_prim.get(prim_name, 1),
                    duration_s=duration_s,
                    reason=outcome.get("reason"),
                )
            )
        return picks

    def _record_loop(self, record: LoopRecord) -> None:
        """Appends ``record`` to the rolling log and logs one summary line.

        Both the completed-loop and errored-loop sites call this so the
        appended record and the logged line can never drift apart.
        """
        self._rolling_log.append(record)
        rss_display = "n/a" if record.rss_mb is None else f"{record.rss_mb:.0f}"
        error_suffix = "" if record.error is None else f" error={record.error}"
        LOGGER.info(
            "loop %d record %d: placed=%d failed=%d oversize=%d duration_s=%.2f rss_mb=%s%s",
            record.loop,
            record.record_id,
            record.placed,
            record.failed,
            record.skipped_oversize,
            record.duration_s,
            rss_display,
            error_suffix,
        )

    async def _park(self) -> None:
        """Moves the TCP to ``PARK_POSE_TCP_MM``, bookending the run so the
        first move is always the same short, well-conditioned one (see
        ``PARK_POSE_TCP_MM``'s comment)."""
        mover = self._build_mover()
        state = await self._park_world_state()
        park_pose = _pointing_down(*PARK_POSE_TCP_MM)
        await mover.move_to(park_pose, state)

    async def _park_world_state(self) -> WorldState:
        """The world state for the park move: boxed per real prop, like the
        census look, when ``prop_geometries`` succeeds; just the plain
        support slab otherwise. The park move is best-effort at the end of a
        run and must never be blocked by a world query failure."""
        try:
            return await self._scan_world_state()
        except Exception:  # noqa: BLE001 - fall back rather than block the park move
            return world_state(None, (), support_obstacle(cell_layout.TABLE_TOP_Z_MM))

    async def _scan_work_list(self) -> list[WorkItem]:
        mover = self._build_mover()
        state = await self._scan_world_state()
        hits: list[_CensusHit] = []
        for look_x, look_y in _census_look_points_mm():
            scan_pose = _pointing_down(
                look_x, look_y, cell_layout.TABLE_TOP_Z_MM + SCAN_HEIGHT_ABOVE_SUPPORT_MM
            )
            await mover.look_from(scan_pose, state)
            for color in cell_layout.BLOCK_COLORS:
                for item in await self._detect_color(color):
                    view_distance_mm = math.hypot(item.x_mm - look_x, item.y_mm - look_y)
                    hits.append(_CensusHit(item, view_distance_mm))
        return _merge_census_hits(hits)

    async def _scan_world_state(self) -> WorldState:
        response = await self._world.do_command({"command": "prop_geometries"})
        geometries = response.get("geometries", [])
        obstacles = obstacles_from_prop_geometries(
            cast("Sequence[Mapping[str, Any]]", geometries), set()
        )
        return world_state(None, obstacles, support_obstacle(cell_layout.TABLE_TOP_Z_MM))

    async def _detect_color(self, color: str) -> list[WorkItem]:
        """One vision call per color, every returned segment kept (unlike
        ``RealDetector.block_pose_world``, which only reports the largest -
        a color's whole pool of up to 3 blocks has to be counted here)."""
        vision = self._vision_by_color[color]
        objects = await vision.get_object_point_clouds(self._camera_name)
        items: list[WorkItem] = []
        for index, obj in enumerate(objects):
            xyz, rgb = parse_pcd(obj.point_cloud)
            top = top_face_centre_m(xyz, rgb)
            if top is None:
                continue
            footprint_mm = footprint_extents_mm(xyz)
            if footprint_mm is None:
                continue
            camera_pif = PoseInFrame(
                reference_frame=self._camera_name,
                pose=Pose(
                    x=top[0] * 1000.0,
                    y=top[1] * 1000.0,
                    z=top[2] * 1000.0,
                    o_x=0.0,
                    o_y=0.0,
                    o_z=1.0,
                    theta=0.0,
                ),
            )
            top_world = (await self._camera_transform.transform_pose(camera_pif, "world")).pose
            if not _within_census_z_band(top_world.z):
                continue
            height_mm = top_world.z - cell_layout.TABLE_TOP_Z_MM
            measured = measured_block_size_mm([footprint_mm[0], footprint_mm[1], height_mm])
            if measured is None:
                continue
            size_mm, _estimates = measured
            items.append(
                WorkItem(
                    name=f"{color}-{index}",
                    color=color,
                    x_mm=top_world.x,
                    y_mm=top_world.y,
                    size_mm=size_mm,
                )
            )
        return items

    async def _resolve_prims(
        self, items: list[WorkItem]
    ) -> tuple[list[tuple[WorkItem, str, str]], dict[str, dict[str, Any]]]:
        """Match each detection to the real scattered pool-block prim nearest
        it (nearest-wins over every item/prim pair, so a duplicate segment on
        one block loses to the closer detection). An item with no prim within
        ``POOL_PRIM_MATCH_TOLERANCE_MM`` is dropped before any motion, recorded
        as ``failed`` (a phantom or duplicate segment, never a real block)."""
        response = await self._world.do_command({"command": "prop_geometries"})
        candidates: dict[str, tuple[float, float]] = {}
        for geometry in response.get("geometries", []):
            name = str(geometry["name"])
            if name not in _POOL_PRIM_COLORS:
                continue
            x, y = geometry["pose_in_world_mm"]["x"], geometry["pose_in_world_mm"]["y"]
            in_zone = min(cell_layout.SCATTER_ZONE_X_MM) <= x <= max(
                cell_layout.SCATTER_ZONE_X_MM
            ) and min(cell_layout.SCATTER_ZONE_Y_MM) <= y <= max(cell_layout.SCATTER_ZONE_Y_MM)
            if in_zone:
                candidates[name] = (x, y)

        pairs = sorted(
            (
                (math.hypot(item.x_mm - x, item.y_mm - y), item, name)
                for item in items
                for name, (x, y) in candidates.items()
            ),
            key=lambda pair: pair[0],
        )
        claimed_items: set[str] = set()
        claimed_prims: set[str] = set()
        matched: list[tuple[WorkItem, str, str]] = []
        for distance, item, name in pairs:
            if distance > POOL_PRIM_MATCH_TOLERANCE_MM:
                break
            if item.name in claimed_items or name in claimed_prims:
                continue
            claimed_items.add(item.name)
            claimed_prims.add(name)
            matched.append((item, name, _POOL_PRIM_COLORS[name]))

        phantom_outcomes = {
            item.name: {
                "outcome": OUTCOME_FAILED,
                "reason": (
                    f"no scattered prim within {POOL_PRIM_MATCH_TOLERANCE_MM:.0f} mm of the "
                    "detection - phantom or duplicate segment"
                ),
            }
            for item in items
            if item.name not in claimed_items
        }
        return matched, phantom_outcomes

    def _build_detector(self, color: str) -> Detector:
        return RealDetector(
            cast(Any, self._camera_transform),  # DEC-13: motion-service pose, not a RobotClient
            self._vision_by_color[color],
            self._camera_name,
            None,
            support_z_mm=cell_layout.TABLE_TOP_Z_MM,
            # the per-pick look is centred on the target; a bigger same-band
            # neighbor at the frame edge must never out-compete it (GPU seed-7)
            prefer_centred=True,
        )

    def _build_mover(self) -> Mover:
        return RealMover(self._motion, self._gripper_name, self._camera_name)

    async def _run_one(
        self,
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        slot_tracker: SlotTracker,
        placed_tallest_mm: float | None,
    ) -> dict[str, Any]:
        max_size_mm = min(self._size_range_mm[1], JAW_MAX_BLOCK_MM)
        if item.size_mm > max_size_mm:
            return {
                "outcome": OUTCOME_SKIPPED_OVERSIZE,
                "prim": prim_name,
                "reason": (
                    f"measured size {item.size_mm:.1f} mm exceeds the {max_size_mm:.1f} mm "
                    "limit (config size_range_mm hi / gripper jaw, whichever is smaller)"
                ),
            }
        try:
            offset = slot_tracker.next_slot(prim_color)
        except ValueError as error:
            return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": str(error)}
        try:
            look_pose = _pointing_down(
                item.x_mm, item.y_mm, cell_layout.TABLE_TOP_Z_MM + FOCUS_HEIGHT_ABOVE_SUPPORT_MM
            )
            scatter_zone_mm = (
                (
                    cell_layout.SCATTER_ZONE_X_MM[0],
                    cell_layout.SCATTER_ZONE_Y_MM[0],
                    cell_layout.TABLE_TOP_Z_MM,
                ),
                (
                    cell_layout.SCATTER_ZONE_X_MM[1],
                    cell_layout.SCATTER_ZONE_Y_MM[1],
                    cell_layout.TABLE_TOP_Z_MM,
                ),
            )
            pipeline = PickPipeline(
                detector=self._build_detector(item.color),
                mover=self._build_mover(),
                gripper=self._gripper,
                block_name=prim_name,
                block_size_mm=None,
                gripper_name=self._gripper_name,
                world=cast(WorldApi, self._world),
                movable_prop_names=(),
                randomize_seed=None,
                look_pose=look_pose,
                support_z_mm=cell_layout.TABLE_TOP_Z_MM,
                pick_region_mm=scatter_zone_mm,
                place_prop_name=cell_layout.pad_name(prim_color),
                place_offset_mm=offset,
                place_region_mm=_place_zone_mm(),
                placed_tallest_mm=placed_tallest_mm,
                block_label=item.color,
                target_prop_name=prim_name,
            )
            await pipeline.run()
            outcome: dict[str, Any] = {"outcome": OUTCOME_PLACED, "prim": prim_name}
            if prim_color != item.color:
                outcome["reason"] = (
                    f"detected {item.color}, prim {prim_name} - routed to {prim_color} pad"
                )
            return outcome
        except JawLimitError as error:
            slot_tracker.release(prim_color, offset)
            return {"outcome": OUTCOME_SKIPPED_OVERSIZE, "prim": prim_name, "reason": str(error)}
        except Exception as error:  # noqa: BLE001 - per-block failure, the loop continues
            slot_tracker.release(prim_color, offset)
            return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": str(error)}
