"""Conductor mock e2e (phase-4 mock gate): a real mock-backend world (built
the way tests/test_scatter_cell.py builds one - MockWorldHandle, no Isaac Sim,
no thread) behind ``start``'s real ``scatter_cell`` seed plumbing, with fake
vision/arm/gripper/motion supplying fabricated multi-color detections across
three census look poses. Proves the multi-pose union carries a block seen
only from one vantage into the work list, the census-level proximity dedup
absorbs a duplicate segment before it ever reaches prim resolution, a
detection with no real prim underneath still resolves as a phantom, a
misclassified detection routes to the PRIM's color pad, and a jaw-limit
refusal classifies as skipped_oversize rather than failed."""

from typing import Any

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.utils import dict_to_struct

from isaac_module import cell_layout
from isaac_module.models.conductor import IsaacConductor
from isaac_module.sim_manager import MockWorldHandle, SimManager
from isaac_module.sort_plan import (
    OUTCOME_FAILED,
    OUTCOME_PLACED,
    OUTCOME_SKIPPED_OVERSIZE,
    WorkItem,
)

MM_PER_M = 1000.0
SEED = 7
# one block per color for red/blue/green/purple (deterministic prims to match
# detections against), none for yellow/orange (never detected in this test)
COUNTS = {"red": 1, "green": 1, "blue": 1, "yellow": 0, "purple": 1, "orange": 0}

POOL_NAMES = [
    cell_layout.pool_block_name(color, index)
    for color in cell_layout.BLOCK_COLORS
    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
]


def _pool_block(name: str) -> dict[str, Any]:
    x, y = cell_layout.park_positions_m()[name]
    return {"type": "cube", "name": name, "size": 0.06, "position": [x, y, 0.03 + 0.0005]}


def _mock_world_handle() -> MockWorldHandle:
    manager = SimManager()
    manager.mock = True
    props = [_pool_block(name) for name in POOL_NAMES]
    return MockWorldHandle(manager, props)


def _expected_positions_mm(
    seed: int, counts: dict[str, int]
) -> dict[str, tuple[float, float, float]]:
    """The real scattered mm positions for ``seed``/``counts``, read off a
    throwaway handle before the conductor's own (separately-instanced, same
    seed) scatter reproduces them - scatter_cell is deterministic per seed,
    but only when called with the same size_range_m the conductor's default
    size_range_mm ([50, 80], DEFAULT_SIZE_RANGE_MM) converts to (the seeded
    draw order changes once a size range asks it to redraw sizes too)."""
    result = _mock_world_handle().scatter_cell(seed, size_range_m=(0.05, 0.08), counts=counts)
    return {name: tuple(v * MM_PER_M for v in pos) for name, pos in result.positions_m.items()}


class _MockWorldApi:
    """Translates the two DoCommands the conductor issues into
    ``MockWorldHandle`` calls, mirroring ``IsaacWorld.do_command``'s mm/m
    conversions without booting a real (or even mocked) Generic component."""

    def __init__(self, handle: MockWorldHandle) -> None:
        self._handle = handle

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        cmd = command["command"]
        if cmd == "scatter_cell":
            size_range_mm = command.get("size_range_mm")
            size_range_m = (
                tuple(v / MM_PER_M for v in size_range_mm) if size_range_mm is not None else None
            )
            result = self._handle.scatter_cell(
                command["seed"], size_range_m=size_range_m, counts=command.get("counts")
            )
            return {
                "seed": result.seed,
                "counts": result.counts,
                "positions": {
                    name: [v * MM_PER_M for v in pos] for name, pos in result.positions_m.items()
                },
                "sizes_mm": {
                    name: [v * MM_PER_M for v in dims] for name, dims in result.sizes_m.items()
                },
                "parked": result.parked,
            }
        if cmd == "prop_geometries":
            geometries = []
            for prop in self._handle.prop_geometries():
                geometries.append(
                    {
                        "name": prop.name,
                        "box_dims_mm": [d * MM_PER_M for d in prop.box_dims_m],
                        "pose_in_world_mm": {
                            "x": prop.position_m[0] * MM_PER_M,
                            "y": prop.position_m[1] * MM_PER_M,
                            "z": prop.position_m[2] * MM_PER_M,
                            "o_x": 0.0,
                            "o_y": 0.0,
                            "o_z": 1.0,
                            "theta": 0.0,
                        },
                        "color": list(prop.color) if prop.color is not None else None,
                        "fixed": prop.fixed,
                    }
                )
            return {"geometries": geometries}
        if cmd == "ignore_props":
            return {"ignored": command.get("names", [])}
        if cmd == "clear_cell":
            cleared = self._handle.clear_cell()
            return {"parked": cleared.parked}
        raise ValueError(f"unsupported mock world command: {cmd!r}")


class _FakeGripper:
    async def open(self) -> None:
        return None

    async def grab(self) -> bool:
        return True

    async def is_holding_something(self) -> Any:
        raise NotImplementedError


class _FakeMover:
    async def look_from(self, pose: Pose, world_state: Any, linear: bool = False) -> None:
        return None

    async def move_to(self, pose: Pose, world_state: Any, linear: bool = False) -> None:
        return None


class _FakeDetector:
    """A Detector fixed to one fabricated work item's pose - the fake vision
    this e2e stands in for the color segmenter. ``reported_size_mm`` lets a
    detection's re-measurement disagree with the size the scan used to gate
    ``_run_one``'s pre-motion oversize check (the purple jaw-limit case)."""

    def __init__(self, item: WorkItem, reported_size_mm: float | None = None) -> None:
        self._item = item
        self._reported_size_mm = item.size_mm if reported_size_mm is None else reported_size_mm

    async def block_pose_world(self) -> Pose:
        z = cell_layout.TABLE_TOP_Z_MM + self._reported_size_mm / 2.0
        return Pose(
            x=self._item.x_mm, y=self._item.y_mm, z=z, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0
        )

    def last_measurement(self) -> dict[str, Any]:
        return {
            "footprint_mm": [self._reported_size_mm, self._reported_size_mm],
            "height_mm": self._reported_size_mm,
            "size_mm": self._reported_size_mm,
        }


def _config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _make_conductor(world: Any) -> IsaacConductor:
    attrs = {
        "world": "world-1",
        "arm": "arm-1",
        "gripper": "gripper-1",
        "camera": "camera-1",
        "side_camera": "side-camera-1",
        "motion": "builtin",
        "detectors": {color: f"{color}-segmenter" for color in cell_layout.BLOCK_COLORS},
    }
    dependencies: dict[ResourceName, Any] = {
        ResourceName(name="world-1"): world,
        ResourceName(name="arm-1"): object(),
        ResourceName(name="gripper-1"): _FakeGripper(),
        ResourceName(name="camera-1"): object(),
        ResourceName(name="side-camera-1"): object(),
        ResourceName(name="builtin"): object(),
    }
    for color in cell_layout.BLOCK_COLORS:
        dependencies[ResourceName(name=f"{color}-segmenter")] = object()
    return IsaacConductor.new(_config("conductor-1", attrs), dependencies)


async def test_conductor_resolves_prims_routes_mismatches_and_classifies_jaw_limit():
    positions = _expected_positions_mm(SEED, COUNTS)
    red_prim = cell_layout.pool_block_name("red", 1)
    blue_prim = cell_layout.pool_block_name("blue", 1)
    green_prim = cell_layout.pool_block_name("green", 1)
    purple_prim = cell_layout.pool_block_name("purple", 1)
    red_x, red_y, _red_z = positions[red_prim]
    blue_x, blue_y, _blue_z = positions[blue_prim]
    green_x, green_y, _green_z = positions[green_prim]
    purple_x, purple_y, _purple_z = positions[purple_prim]

    # a detection with no real prim underneath at all (COUNTS has no orange
    # blocks) - the genuine phantom-or-duplicate-segment case, far from every
    # real prim and every other fabricated item so census dedup never touches it
    orange_x = cell_layout.SCATTER_ZONE_X_MM[1] - 50.0
    orange_y = cell_layout.SCATTER_ZONE_Y_MM[1] - 50.0

    fabricated = {
        "red-0": WorkItem("red-0", "red", red_x, red_y, 60.0),
        # a block that split into two segments 20 mm apart: the census-level
        # proximity dedup merges them into one work item before prim
        # resolution ever runs, so only one outcome key survives
        "blue-0": WorkItem("blue-0", "blue", blue_x, blue_y, 60.0),
        "blue-1": WorkItem("blue-1", "blue", blue_x + 20.0, blue_y, 60.0),
        # the yellow detector locked onto the green block: the prim (not the
        # detector) decides the destination pad
        "yellow-0": WorkItem("yellow-0", "yellow", green_x, green_y, 60.0),
        # passes the scan's pre-motion size gate (70 <= 75), but the
        # pipeline's own re-measurement (90 mm) trips the real jaw check
        "purple-0": WorkItem("purple-0", "purple", purple_x, purple_y, 70.0),
        "orange-0": WorkItem("orange-0", "orange", orange_x, orange_y, 60.0),
    }

    detect_calls = {color: 0 for color in cell_layout.BLOCK_COLORS}

    async def fake_detect_color(color: str) -> list[WorkItem]:
        call_index = detect_calls[color]
        detect_calls[color] += 1
        if color == "red":
            # arm occlusion: the source block is hidden from the first
            # census look pose and only appears from the second
            return [fabricated["red-0"]] if call_index == 1 else []
        if call_index > 0:
            return []
        return [item for item in fabricated.values() if item.color == color]

    detectors_by_color = {
        "red": _FakeDetector(fabricated["red-0"]),
        "blue": _FakeDetector(fabricated["blue-0"]),
        "yellow": _FakeDetector(fabricated["yellow-0"]),
        "purple": _FakeDetector(fabricated["purple-0"], reported_size_mm=90.0),
    }

    world = _MockWorldApi(_mock_world_handle())
    conductor = _make_conductor(world)
    conductor._detect_color = fake_detect_color  # type: ignore[method-assign]
    conductor._build_detector = lambda color: detectors_by_color[color]  # type: ignore[method-assign]
    conductor._build_mover = lambda: _FakeMover()  # type: ignore[method-assign]

    started = await conductor.do_command({"command": "start", "seed": SEED, "counts": COUNTS})
    assert started == {"ok": True, "state": "running"}

    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["remaining"] == []
    assert status["current"] is None

    # every color detector was consulted once per census look pose (three)
    assert detect_calls == {color: 3 for color in cell_layout.BLOCK_COLORS}

    # outcomes are keyed by resolved prim name (the physical identity that
    # persists across a re-census), not by the transient detection name
    red_outcome = status["outcomes"][red_prim]
    assert red_outcome["outcome"] == OUTCOME_PLACED
    assert red_outcome["prim"] == red_prim

    # the split blue segment collapsed to exactly one outcome before prim
    # resolution ever ran - the loser leaves no trace, placed or failed
    blue_outcome = status["outcomes"][blue_prim]
    assert blue_outcome["outcome"] == OUTCOME_PLACED
    assert blue_outcome["prim"] == blue_prim

    mismatch_outcome = status["outcomes"][green_prim]
    assert mismatch_outcome["outcome"] == OUTCOME_PLACED
    assert mismatch_outcome["prim"] == green_prim
    assert "detected yellow" in mismatch_outcome["reason"]
    assert "routed to green pad" in mismatch_outcome["reason"]

    jaw_limit_outcome = status["outcomes"][purple_prim]
    assert jaw_limit_outcome["outcome"] == OUTCOME_SKIPPED_OVERSIZE
    assert jaw_limit_outcome["prim"] == purple_prim
    assert "jaw" in jaw_limit_outcome["reason"].lower()

    # never resolved to a prim (phantom/duplicate segment) - no prim
    # identity to key by, so it keeps its detection name
    phantom_outcome = status["outcomes"]["orange-0"]
    assert phantom_outcome["outcome"] == OUTCOME_FAILED
    assert "phantom or duplicate segment" in phantom_outcome["reason"]

    # every remaining item this pass placed or was skipped (no failures),
    # so the sort finished after a single census pass
    assert status["pass"] == 1


async def test_two_loop_run_over_the_mock_backend_ends_complete_with_loop_records():
    """A 2-loop ``start`` against the real mock world backend (clear_cell +
    scatter_cell for real, per loop): one red block per loop, detected by
    querying the backend's own current prop position each census, so the
    detection always matches wherever that loop's scatter actually placed
    it (see the module docstring for why the fabricated-item approach used
    above needs a fixed position instead)."""
    counts = {"red": 1, "green": 0, "blue": 0, "yellow": 0, "purple": 0, "orange": 0}
    handle = _mock_world_handle()
    world = _MockWorldApi(handle)
    red_prim = cell_layout.pool_block_name("red", 1)

    def _red_position_mm() -> tuple[float, float]:
        for prop in handle.prop_geometries():
            if prop.name == red_prim:
                return prop.position_m[0] * MM_PER_M, prop.position_m[1] * MM_PER_M
        raise AssertionError("red prim missing from prop_geometries")

    async def fake_detect_color(color: str) -> list[WorkItem]:
        if color != "red":
            return []
        x, y = _red_position_mm()
        return [WorkItem("red-0", "red", x, y, 60.0)]

    def build_detector(color: str) -> Any:
        x, y = _red_position_mm()
        return _FakeDetector(WorkItem("red-0", "red", x, y, 60.0))

    conductor = _make_conductor(world)
    conductor._detect_color = fake_detect_color  # type: ignore[method-assign]
    conductor._build_detector = build_detector  # type: ignore[method-assign]
    conductor._build_mover = lambda: _FakeMover()  # type: ignore[method-assign]

    started = await conductor.do_command(
        {"command": "start", "seed": SEED, "counts": counts, "loops": 2}
    )
    assert started == {"ok": True, "state": "running"}

    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["run"]["loops_completed"] == 2
    assert status["run"]["loop"] == 2

    loop_records = status["loop_records"]
    assert len(loop_records) == 2
    seeds = [record["seed"] for record in loop_records]
    assert seeds == [SEED, SEED + 1]
    assert seeds[0] != seeds[1]
    for record in loop_records:
        assert record["placed"] == 1
        assert record["failed"] == 0
        assert record["skipped_oversize"] == 0
        assert record["duration_s"] >= 0

    assert status["run"]["placed"] == 2
    assert status["run"]["failed"] == 0
    assert status["success_rate"] == 1.0
