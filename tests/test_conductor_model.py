"""Conductor lifecycle and validation, over fake dependencies: no vision,
motion, gripper, or world client is real here - the sort loop's own
orchestration (start/stop/status, ordering, oversize skip, per-block failure)
is what's under test. See test_conductor_e2e.py for the mock world backend."""

import asyncio
from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.utils import dict_to_struct

from isaac_module import cell_layout
from isaac_module.models import conductor as conductor_module
from isaac_module.models.conductor import (
    PARK_POSE_TCP_MM,
    POOL_PRIM_MATCH_TOLERANCE_MM,
    IsaacConductor,
    _CensusHit,
    _merge_census_hits,
    _within_census_z_band,
)
from isaac_module.run_log import MAX_ATTEMPTS_PER_BLOCK, MAX_CONSECUTIVE_LOOP_ERRORS
from isaac_module.sort_plan import (
    OUTCOME_FAILED,
    OUTCOME_PLACED,
    OUTCOME_SKIPPED_OVERSIZE,
    SlotTracker,
    WorkItem,
)

WORLD_NAME = "world-1"
ARM_NAME = "arm-1"
GRIPPER_NAME = "gripper-1"
CAMERA_NAME = "camera-1"
SIDE_CAMERA_NAME = "side-camera-1"
MOTION_NAME = "builtin"


def _config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _valid_attrs(**overrides: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "world": WORLD_NAME,
        "arm": ARM_NAME,
        "gripper": GRIPPER_NAME,
        "camera": CAMERA_NAME,
        "side_camera": SIDE_CAMERA_NAME,
        "motion": MOTION_NAME,
        "detectors": {color: f"{color}-segmenter" for color in cell_layout.BLOCK_COLORS},
    }
    attrs.update(overrides)
    return attrs


class FakeWorld:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        self.commands.append(dict(command))
        if command["command"] == "scatter_cell":
            return {
                "seed": command["seed"],
                "counts": {},
                "positions": {},
                "sizes_mm": {},
                "parked": [],
            }
        if command["command"] == "prop_geometries":
            return {"geometries": []}
        return {"ok": True}


class FakeGripper:
    async def open(self) -> None:
        return None

    async def grab(self) -> bool:
        return True

    async def is_holding_something(self) -> Any:
        raise NotImplementedError


def _dependencies(world: Any, gripper: Any) -> dict[ResourceName, Any]:
    deps: dict[ResourceName, Any] = {
        ResourceName(name=WORLD_NAME): world,
        ResourceName(name=ARM_NAME): object(),
        ResourceName(name=GRIPPER_NAME): gripper,
        ResourceName(name=CAMERA_NAME): object(),
        ResourceName(name=SIDE_CAMERA_NAME): object(),
        ResourceName(name=MOTION_NAME): object(),
    }
    for color in cell_layout.BLOCK_COLORS:
        deps[ResourceName(name=f"{color}-segmenter")] = object()
    return deps


def _make_conductor(world: Any | None = None, gripper: Any | None = None) -> IsaacConductor:
    world = world if world is not None else FakeWorld()
    gripper = gripper if gripper is not None else FakeGripper()
    config = _config("conductor-1", _valid_attrs())
    return IsaacConductor.new(config, _dependencies(world, gripper))


# ----------------------------------------------------------------------
# validate_config
# ----------------------------------------------------------------------


def test_validate_config_returns_every_named_dependency():
    config = _config("conductor-1", _valid_attrs())
    dependencies, optional = IsaacConductor.validate_config(config)
    assert set(dependencies) == {
        WORLD_NAME,
        ARM_NAME,
        GRIPPER_NAME,
        CAMERA_NAME,
        SIDE_CAMERA_NAME,
        MOTION_NAME,
        *(f"{color}-segmenter" for color in cell_layout.BLOCK_COLORS),
    }
    assert optional == []


def test_validate_config_missing_detector_color_raises():
    attrs = _valid_attrs()
    detectors = dict(attrs["detectors"])
    del detectors["blue"]
    attrs["detectors"] = detectors
    with pytest.raises(ValueError, match="missing colors"):
        IsaacConductor.validate_config(_config("conductor-1", attrs))


def test_validate_config_unknown_color_raises():
    attrs = _valid_attrs()
    detectors = dict(attrs["detectors"])
    detectors["magenta"] = "magenta-segmenter"
    attrs["detectors"] = detectors
    with pytest.raises(ValueError, match="unknown colors"):
        IsaacConductor.validate_config(_config("conductor-1", attrs))


def test_validate_config_oversize_size_range_raises():
    attrs = _valid_attrs(size_range_mm=[50, 90])
    with pytest.raises(ValueError, match="MAX_BLOCK_SIZE_MM"):
        IsaacConductor.validate_config(_config("conductor-1", attrs))


# ----------------------------------------------------------------------
# start / stop / status lifecycle
# ----------------------------------------------------------------------


async def _passthrough_resolve_prims(
    items: list[WorkItem],
) -> tuple[list[tuple[WorkItem, str, str]], dict[str, Any]]:
    """Test-only stand-in: every item resolves to itself, so lifecycle tests
    can drive ``_run`` without a real ``prop_geometries`` scattered scene."""
    return [(item, item.name, item.color) for item in items], {}


async def test_second_start_while_running_is_a_noop():
    conductor = _make_conductor()
    gate = asyncio.Event()

    async def scan() -> list[WorkItem]:
        return [WorkItem("red-0", "red", 0.0, 0.0, 60.0)]

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        await gate.wait()
        return {"outcome": "placed"}

    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    first = await conductor.do_command({"command": "start"})
    assert first == {"ok": True, "state": "running"}

    second = await conductor.do_command({"command": "start"})
    assert second == {"ok": False, "state": "running"}

    gate.set()
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["outcomes"] == {"red-0": {"outcome": "placed", "attempts": 1}}


async def test_stop_between_motions_halts_before_the_next_block():
    conductor = _make_conductor()
    items = [WorkItem(f"red-{i}", "red", float(i), 0.0, 60.0) for i in range(3)]
    calls: list[str] = []
    observed_state_during_stop: list[str] = []

    async def scan() -> list[WorkItem]:
        return items

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        calls.append(item.name)
        if item.name == "red-0":
            await conductor.do_command({"command": "stop"})
            observed_state_during_stop.append(conductor._state)
        return {"outcome": "placed"}

    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert calls == ["red-0"]
    assert observed_state_during_stop == ["stopping"]
    assert status["state"] == "idle"
    assert len(status["outcomes"]) == 1
    assert status["remaining"] == ["red-1", "red-2"]


async def test_status_snapshot_is_isolated_from_the_next_call():
    conductor = _make_conductor()
    conductor._outcomes = {"red-0": {"outcome": "placed"}}
    conductor._remaining = ["red-1"]

    first = await conductor.do_command({"command": "status"})
    first["outcomes"]["red-0"]["outcome"] = "mutated"
    first["remaining"].append("hacked")

    second = await conductor.do_command({"command": "status"})
    assert second["outcomes"]["red-0"]["outcome"] == "placed"
    assert second["remaining"] == ["red-1"]


# ----------------------------------------------------------------------
# multi-pass sorting (phase-4e): re-census when a pass makes progress but
# still has failures, stop when it doesn't, never retry oversize
# ----------------------------------------------------------------------


async def test_multi_pass_re_censuses_and_places_survivors_after_a_crowded_pass():
    # pass 1: two crowded blocks (crowded-0, crowded-1) fail with a
    # neighbour-collision RuntimeError while their neighbour is still
    # present; a third block (clear-0) places cleanly. pass 2's census
    # shows only the (now-clear) survivors, and they place.
    pass1_items = [
        WorkItem("clear-0", "red", 0.0, 0.0, 60.0),
        WorkItem("crowded-0", "red", 100.0, 0.0, 60.0),
        WorkItem("crowded-1", "red", 105.0, 0.0, 60.0),
    ]
    pass2_items = [
        WorkItem("crowded-0", "red", 100.0, 0.0, 60.0),
        WorkItem("crowded-1", "red", 105.0, 0.0, 60.0),
    ]
    scans = [pass1_items, pass2_items]
    attempts: dict[str, int] = {}

    async def scan() -> list[WorkItem]:
        return scans.pop(0)

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        attempts[prim_name] = attempts.get(prim_name, 0) + 1
        if prim_name in ("crowded-0", "crowded-1") and attempts[prim_name] == 1:
            return {
                "outcome": OUTCOME_FAILED,
                "prim": prim_name,
                "reason": f"obstacle constraint: violation between {prim_name} and pick-grip",
            }
        return {"outcome": "placed", "prim": prim_name}

    conductor = _make_conductor()
    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["pass"] == 2
    assert status["outcomes"]["clear-0"]["outcome"] == "placed"
    assert status["outcomes"]["crowded-0"]["outcome"] == "placed"
    assert status["outcomes"]["crowded-1"]["outcome"] == "placed"


async def test_multi_pass_stops_when_a_pass_places_nothing():
    items = [WorkItem("stuck-0", "red", 0.0, 0.0, 60.0)]

    async def scan() -> list[WorkItem]:
        return items

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": "always fails"}

    conductor = _make_conductor()
    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["pass"] == 1
    assert status["outcomes"]["stuck-0"]["outcome"] == OUTCOME_FAILED


async def test_multi_pass_never_retries_an_oversize_skip():
    item = WorkItem("big-0", "red", 0.0, 0.0, 90.0)
    other = WorkItem("fails-0", "red", 50.0, 0.0, 60.0)
    scan_calls = 0

    async def scan() -> list[WorkItem]:
        nonlocal scan_calls
        scan_calls += 1
        return [item, other]

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        if prim_name == "big-0":
            return {"outcome": OUTCOME_SKIPPED_OVERSIZE, "prim": prim_name, "reason": "too big"}
        return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": "always fails"}

    conductor = _make_conductor()
    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    # "fails-0" never places, so every pass makes no progress and the sort
    # stops after pass 1 - "big-0" is only ever attempted once regardless
    assert status["state"] == "complete"
    assert status["pass"] == 1
    assert status["outcomes"]["big-0"]["outcome"] == OUTCOME_SKIPPED_OVERSIZE
    assert scan_calls == 1


# ----------------------------------------------------------------------
# park-pose bookends (revision: a start-of-run "freak out" - a long free
# plan straight from the previous run's end pose to the first census pose)
# ----------------------------------------------------------------------


async def test_park_move_bookends_the_run_before_census_and_after_the_final_pass():
    calls: list[tuple[str, Any]] = []

    class _RecordingMover:
        async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
            calls.append(("look", pose))

        async def move_to(self, pose: Any, world_state: Any, linear: bool = False) -> None:
            calls.append(("move", pose))

    async def no_detections(color: str) -> list[WorkItem]:
        return []

    conductor = _make_conductor()
    conductor._build_mover = lambda: _RecordingMover()  # type: ignore[method-assign]
    conductor._detect_color = no_detections  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    assert [kind for kind, _pose in calls] == ["move", "look", "look", "look", "move"]
    park_x, park_y, park_z = PARK_POSE_TCP_MM
    for kind, pose in (calls[0], calls[-1]):
        assert kind == "move"
        assert (pose.x, pose.y, pose.z) == (park_x, park_y, park_z)


async def test_park_failure_at_end_of_run_does_not_change_the_outcome():
    class _FailsOnSecondMoveMover:
        def __init__(self) -> None:
            self.move_calls = 0

        async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
            return None

        async def move_to(self, pose: Any, world_state: Any, linear: bool = False) -> None:
            self.move_calls += 1
            if self.move_calls > 1:
                raise RuntimeError("park move failed")

    mover = _FailsOnSecondMoveMover()

    async def no_detections(color: str) -> list[WorkItem]:
        return []

    conductor = _make_conductor()
    conductor._build_mover = lambda: mover  # type: ignore[method-assign]
    conductor._detect_color = no_detections  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    # both the start park (succeeded) and the end park (raised) were
    # attempted - the second attempt's failure never touched the state
    assert mover.move_calls == 2


# ----------------------------------------------------------------------
# _run_one: oversize skip and per-block failure
# ----------------------------------------------------------------------


async def test_run_one_skips_an_oversize_block_without_motion():
    conductor = _make_conductor()

    item = WorkItem("red-0", "red", 0.0, 0.0, 81.0)
    outcome = await conductor._run_one(item, "block_red_1", "red", SlotTracker(), None)
    assert outcome["outcome"] == OUTCOME_SKIPPED_OVERSIZE
    assert outcome["prim"] == "block_red_1"
    assert "reason" in outcome


class _RaisingDetector:
    async def block_pose_world(self) -> Any:
        raise RuntimeError("no detection")


class _NoopMover:
    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        return None

    async def move_to(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        return None


async def test_run_one_records_failed_on_pipeline_exception():
    conductor = _make_conductor()
    conductor._build_detector = lambda color: _RaisingDetector()  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    item = WorkItem("red-0", "red", 0.0, 0.0, 60.0)
    outcome = await conductor._run_one(item, "block_red_1", "red", SlotTracker(), None)
    assert outcome["outcome"] == OUTCOME_FAILED
    assert outcome["prim"] == "block_red_1"
    assert "no detection" in outcome["reason"]


# ----------------------------------------------------------------------
# _resolve_prims: nearest-wins claiming, tolerance edge
# ----------------------------------------------------------------------


def _prim_geometry(name: str, x_mm: float, y_mm: float) -> dict[str, Any]:
    return {
        "name": name,
        "box_dims_mm": [60.0, 60.0, 60.0],
        "pose_in_world_mm": {
            "x": x_mm,
            "y": y_mm,
            "z": cell_layout.TABLE_TOP_Z_MM + 30.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
        "color": None,
        "fixed": False,
    }


class FakeWorldWithPrims(FakeWorld):
    def __init__(self, geometries: list[dict[str, Any]]) -> None:
        super().__init__()
        self._geometries = geometries

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        if command["command"] == "prop_geometries":
            self.commands.append(dict(command))
            return {"geometries": self._geometries}
        return await super().do_command(command)


_PRIM_X_MM = cell_layout.SCATTER_ZONE_X_MM[0] + 50.0
_PRIM_Y_MM = 0.0
_RED_PRIM_NAME = cell_layout.pool_block_name("red", 1)


async def test_resolve_prims_nearest_item_claims_the_prim():
    conductor = _make_conductor(
        world=FakeWorldWithPrims([_prim_geometry(_RED_PRIM_NAME, _PRIM_X_MM, _PRIM_Y_MM)])
    )
    near = WorkItem("red-0", "red", _PRIM_X_MM, _PRIM_Y_MM, 60.0)
    far = WorkItem("red-1", "red", _PRIM_X_MM + 20.0, _PRIM_Y_MM, 60.0)

    matched, phantom = await conductor._resolve_prims([far, near])

    assert [(item.name, prim, color) for item, prim, color in matched] == [
        ("red-0", _RED_PRIM_NAME, "red")
    ]
    assert set(phantom) == {"red-1"}
    assert phantom["red-1"]["outcome"] == OUTCOME_FAILED
    assert "phantom or duplicate segment" in phantom["red-1"]["reason"]


async def test_resolve_prims_beyond_tolerance_matches_nothing():
    conductor = _make_conductor(
        world=FakeWorldWithPrims([_prim_geometry(_RED_PRIM_NAME, _PRIM_X_MM, _PRIM_Y_MM)])
    )
    too_far = WorkItem(
        "red-0", "red", _PRIM_X_MM + POOL_PRIM_MATCH_TOLERANCE_MM + 1.0, _PRIM_Y_MM, 60.0
    )

    matched, phantom = await conductor._resolve_prims([too_far])

    assert matched == []
    assert set(phantom) == {"red-0"}


# ----------------------------------------------------------------------
# pad-slot release on a non-placed attempt (regression: a slot leaked on
# every failed/skipped attempt, starving a color's pool even though
# nothing of that color had actually placed)
# ----------------------------------------------------------------------


class _MaybeFailingDetector:
    """Fails with a RuntimeError while ``pass_state`` reads below
    ``fails_before_pass`` (an unrelated collision, standing in for a still-
    crowded neighbour); reports a normal resting pose once it doesn't."""

    def __init__(
        self,
        x_mm: float,
        y_mm: float,
        size_mm: float,
        pass_state: dict[str, int],
        fails_before_pass: int = 0,
    ) -> None:
        self._x_mm = x_mm
        self._y_mm = y_mm
        self._size_mm = size_mm
        self._pass_state = pass_state
        self._fails_before_pass = fails_before_pass

    async def block_pose_world(self) -> Any:
        if self._pass_state["pass"] < self._fails_before_pass:
            raise RuntimeError("obstacle constraint: violation between neighbour and pick-grip")
        z = cell_layout.TABLE_TOP_Z_MM + self._size_mm / 2.0
        return Pose(x=self._x_mm, y=self._y_mm, z=z, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

    def last_measurement(self) -> dict[str, Any]:
        return {
            "footprint_mm": [self._size_mm, self._size_mm],
            "height_mm": self._size_mm,
            "size_mm": self._size_mm,
        }


_YELLOW_PRIM_NAMES = [cell_layout.pool_block_name("yellow", index) for index in range(1, 4)]


async def test_multi_pass_recovers_yellow_siblings_without_losing_pad_slots():
    # a clear red block places in pass 1 (giving the sort progress to
    # justify a pass 2), while all three yellow siblings fail pass 1 for an
    # unrelated reason - each one must draw and then release a pad slot,
    # since none of them placed. pass 2's census shows the same three
    # yellows again (nothing physically changed) and this time they place -
    # only possible if pass 1's failures actually returned their slots
    red_item = WorkItem("red-0", "red", _PRIM_X_MM, _PRIM_Y_MM, 60.0)
    yellow_positions = [(_PRIM_X_MM + 200.0 + 40.0 * i, _PRIM_Y_MM) for i in range(3)]
    yellow_items = [
        WorkItem(f"yellow-{i}", "yellow", x, y, 60.0) for i, (x, y) in enumerate(yellow_positions)
    ]
    geometries = [_prim_geometry(_RED_PRIM_NAME, _PRIM_X_MM, _PRIM_Y_MM)] + [
        _prim_geometry(name, x, y)
        for name, (x, y) in zip(_YELLOW_PRIM_NAMES, yellow_positions, strict=True)
    ]

    pass_state = {"pass": 1}
    scan_calls = 0

    async def scan() -> list[WorkItem]:
        nonlocal scan_calls
        scan_calls += 1
        if scan_calls > 1:
            pass_state["pass"] = 2
        return [red_item, *yellow_items]

    detectors_by_color = {
        "red": _MaybeFailingDetector(_PRIM_X_MM, _PRIM_Y_MM, 60.0, pass_state),
        "yellow": [
            _MaybeFailingDetector(x, y, 60.0, pass_state, fails_before_pass=2)
            for x, y in yellow_positions
        ],
    }
    yellow_call_index = {"count": 0}

    def build_detector(color: str) -> Any:
        if color == "red":
            return detectors_by_color["red"]
        detector = detectors_by_color["yellow"][yellow_call_index["count"] % 3]
        yellow_call_index["count"] += 1
        return detector

    conductor = _make_conductor(world=FakeWorldWithPrims(geometries))
    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._build_detector = build_detector  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["pass"] == 2
    assert status["outcomes"][_RED_PRIM_NAME]["outcome"] == OUTCOME_PLACED
    for prim_name in _YELLOW_PRIM_NAMES:
        assert status["outcomes"][prim_name]["outcome"] == OUTCOME_PLACED


async def test_run_one_releases_the_slot_on_an_oversize_skip():
    # the pre-motion size gate only rejects sizes over size_range_mm[1]/jaw;
    # a re-measurement inside the pipeline (JawLimitError) also skips, and
    # must also give its slot back - three same-color oversize skips in a
    # row must not exhaust the pool
    conductor = _make_conductor()
    tracker = SlotTracker()
    oversize_item = WorkItem("purple-0", "purple", 0.0, 0.0, 70.0)

    class _OversizeDetector:
        async def block_pose_world(self) -> Any:
            return Pose(
                x=0.0,
                y=0.0,
                z=cell_layout.TABLE_TOP_Z_MM + 45.0,
                o_x=0.0,
                o_y=0.0,
                o_z=-1.0,
                theta=0.0,
            )

        def last_measurement(self) -> dict[str, Any]:
            return {"footprint_mm": [90.0, 90.0], "height_mm": 90.0, "size_mm": 90.0}

    conductor._build_detector = lambda color: _OversizeDetector()  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    for _ in range(cell_layout.POOL_BLOCKS_PER_COLOR):
        outcome = await conductor._run_one(oversize_item, "block_purple_1", "purple", tracker, None)
        assert outcome["outcome"] == OUTCOME_SKIPPED_OVERSIZE

    # a fourth attempt still finds a free slot - none were lost to the
    # first three oversize skips
    outcome = await conductor._run_one(oversize_item, "block_purple_1", "purple", tracker, None)
    assert outcome["outcome"] == OUTCOME_SKIPPED_OVERSIZE


# ----------------------------------------------------------------------
# census hardening: z-band filter, multi-pose union, proximity dedup
# ----------------------------------------------------------------------


def test_within_census_z_band_keeps_a_top_face_just_above_the_table():
    assert _within_census_z_band(cell_layout.TABLE_TOP_Z_MM + 30.0) is True


def test_within_census_z_band_drops_a_table_glint():
    assert _within_census_z_band(cell_layout.TABLE_TOP_Z_MM + 10.0) is False


def test_within_census_z_band_drops_the_arm_at_scan_height():
    assert _within_census_z_band(cell_layout.TABLE_TOP_Z_MM + 250.0) is False


def test_merge_census_hits_collapses_duplicates_within_tolerance():
    same_block = WorkItem("red-0", "red", 0.0, 0.0, 60.0)
    duplicate_segment = WorkItem("red-1", "red", 10.0, 0.0, 60.0)
    hits = [
        _CensusHit(same_block, view_distance_mm=500.0),
        _CensusHit(duplicate_segment, view_distance_mm=100.0),
    ]

    merged = _merge_census_hits(hits)

    assert len(merged) == 1
    # the least-oblique view (smaller view_distance_mm) wins the merge
    assert merged[0].name == "red-1"


def test_merge_census_hits_leaves_distant_items_untouched():
    far_apart = [
        _CensusHit(WorkItem("red-0", "red", 0.0, 0.0, 60.0), view_distance_mm=100.0),
        _CensusHit(
            WorkItem("red-1", "red", cell_layout.MAX_BLOCK_SIZE_MM + 100.0, 0.0, 60.0),
            view_distance_mm=100.0,
        ),
    ]

    merged = _merge_census_hits(far_apart)

    assert {item.name for item in merged} == {"red-0", "red-1"}


def test_merge_census_hits_cross_detector_conflation_deduped_across_poses():
    # yellow and orange both fire on the same physical block from two
    # different census poses - same planar position, different color guess,
    # different view obliqueness
    yellow_guess = WorkItem("yellow-0", "yellow", 5.0, 5.0, 60.0)
    orange_guess = WorkItem("orange-0", "orange", 0.0, 0.0, 60.0)
    hits = [
        _CensusHit(yellow_guess, view_distance_mm=50.0),
        _CensusHit(orange_guess, view_distance_mm=20.0),
    ]

    merged = _merge_census_hits(hits)

    assert len(merged) == 1
    assert merged[0].name == "orange-0"


async def test_scan_work_list_unions_a_block_detected_only_from_one_pose():
    conductor = _make_conductor()
    look_poses_seen: list[Any] = []
    detect_calls: dict[str, int] = {color: 0 for color in cell_layout.BLOCK_COLORS}

    class _RecordingMover:
        async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
            look_poses_seen.append(pose)

        async def move_to(self, pose: Any, world_state: Any, linear: bool = False) -> None:
            return None

    async def fake_detect_color(color: str) -> list[WorkItem]:
        call_index = detect_calls[color]
        detect_calls[color] += 1
        if color == "green":
            # only visible from the second census look pose
            return [WorkItem("green-0", "green", 0.0, 0.0, 60.0)] if call_index == 1 else []
        return []

    conductor._build_mover = lambda: _RecordingMover()  # type: ignore[method-assign]
    conductor._detect_color = fake_detect_color  # type: ignore[method-assign]

    items = await conductor._scan_work_list()

    assert len(look_poses_seen) == 3
    assert detect_calls == {color: 3 for color in cell_layout.BLOCK_COLORS}
    assert [item.name for item in items] == ["green-0"]


# ----------------------------------------------------------------------
# phase 5: loop controller, failure policy, telemetry
# ----------------------------------------------------------------------


def _empty_resolve_prims_conductor() -> IsaacConductor:
    """A conductor whose census finds nothing, so ``_run_one_loop`` finishes
    in exactly one pass per loop - the loop-controller tests only care about
    the loop boundary machinery, not the per-block attempt loop."""

    async def no_detections(color: str) -> list[WorkItem]:
        return []

    conductor = _make_conductor()
    conductor._detect_color = no_detections  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    return conductor


def _scatter_seeds(world: FakeWorld) -> list[int]:
    return [cmd["seed"] for cmd in world.commands if cmd["command"] == "scatter_cell"]


async def test_loop_mode_advances_seed_deterministically_from_the_base():
    world = FakeWorld()
    conductor = _empty_resolve_prims_conductor()
    conductor._world = world  # type: ignore[attr-defined]

    await conductor.do_command({"command": "start", "seed": 100, "loops": 3})
    await conductor.wait_until_done()

    assert _scatter_seeds(world) == [100, 101, 102]
    clear_count = sum(1 for cmd in world.commands if cmd["command"] == "clear_cell")
    assert clear_count == 3


async def test_n_loop_run_ends_complete_with_n_loop_records(monkeypatch):
    monkeypatch.setattr("isaac_module.models.conductor.current_rss_mb", lambda: 123.5)
    conductor = _empty_resolve_prims_conductor()

    await conductor.do_command({"command": "start", "seed": 1, "loops": 3})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["run"]["loops_completed"] == 3
    assert status["run"]["loop"] == 3
    assert len(status["loop_records"]) == 3
    for record in status["loop_records"]:
        assert record["duration_s"] >= 0
        assert record["rss_mb"] == 123.5


async def test_continuous_mode_loops_until_stop_then_lands_idle():
    conductor = _empty_resolve_prims_conductor()
    loop_count = {"n": 0}

    async def no_detections_then_stop(color: str) -> list[WorkItem]:
        return []

    async def scan_and_maybe_stop() -> list[WorkItem]:
        loop_count["n"] += 1
        if loop_count["n"] >= 3:
            await conductor.do_command({"command": "stop"})
        return []

    conductor._scan_work_list = scan_and_maybe_stop  # type: ignore[method-assign]
    conductor._detect_color = no_detections_then_stop  # type: ignore[method-assign]

    await conductor.do_command({"command": "start", "seed": 1, "continuous": True})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "idle"
    assert status["run"]["continuous"] is True
    assert status["run"]["loops_requested"] == 0
    assert status["run"]["loops_completed"] == 3


async def test_loops_zero_is_equivalent_to_continuous():
    conductor = _empty_resolve_prims_conductor()

    async def scan_then_stop() -> list[WorkItem]:
        await conductor.do_command({"command": "stop"})
        return []

    conductor._scan_work_list = scan_then_stop  # type: ignore[method-assign]

    await conductor.do_command({"command": "start", "seed": 1, "loops": 0})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "idle"
    assert status["run"]["continuous"] is True
    assert status["run"]["loops_completed"] == 1


async def test_a_transiently_failed_loop_is_recorded_and_the_run_continues(monkeypatch):
    """GPU run 7: a dropped gRPC stream mid-census killed the whole 3-loop
    run. One failed loop costs that loop only - recorded with an error, the
    next loop's seed still runs."""
    monkeypatch.setattr("isaac_module.models.conductor.current_rss_mb", lambda: 123.5)
    conductor = _empty_resolve_prims_conductor()
    scan_calls = {"n": 0}

    async def scan_fails_on_loop_2() -> list[WorkItem]:
        scan_calls["n"] += 1
        if scan_calls["n"] == 2:
            raise RuntimeError("Connection lost")
        return []

    conductor._scan_work_list = scan_fails_on_loop_2  # type: ignore[method-assign]

    await conductor.do_command({"command": "start", "seed": 100, "loops": 3})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["run"]["loops_completed"] == 2
    assert status["run"]["loops_errored"] == 1
    assert len(status["loop_records"]) == 3
    errored = status["loop_records"][1]
    assert errored["loop"] == 2
    assert errored["seed"] == 101
    assert errored["error"] == "RuntimeError: Connection lost"
    assert errored["picks"] == []
    # the clean records carry no error key at all
    assert "error" not in status["loop_records"][0]
    assert "error" not in status["loop_records"][2]
    assert status["loop_records"][2]["seed"] == 102
    for record in status["loop_records"]:
        assert record["rss_mb"] == 123.5


async def test_consecutive_loop_errors_at_the_cap_fail_the_run():
    conductor = _empty_resolve_prims_conductor()

    async def scan_always_fails() -> list[WorkItem]:
        raise RuntimeError("Connection lost")

    conductor._scan_work_list = scan_always_fails  # type: ignore[method-assign]

    await conductor.do_command({"command": "start", "seed": 100, "loops": 10})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "failed"
    # exactly the cap, not one more: the third consecutive error re-raises
    assert status["run"]["loops_errored"] == MAX_CONSECUTIVE_LOOP_ERRORS
    assert len(status["loop_records"]) == MAX_CONSECUTIVE_LOOP_ERRORS


async def test_single_shot_exception_still_fails_the_run():
    conductor = _empty_resolve_prims_conductor()

    async def scan_fails() -> list[WorkItem]:
        raise RuntimeError("Connection lost")

    conductor._scan_work_list = scan_fails  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert status["run"]["loops_errored"] == 1


async def test_single_shot_start_reports_exactly_one_loop_and_null_loops_requested():
    conductor = _empty_resolve_prims_conductor()

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["run"]["loops_requested"] is None
    assert status["run"]["continuous"] is False
    assert status["run"]["loops_completed"] == 1
    assert len(status["loop_records"]) == 1


async def test_loop_mode_without_a_seed_derives_a_base_seed():
    conductor = _empty_resolve_prims_conductor()

    await conductor.do_command({"command": "start", "loops": 1})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    assert status["run"]["base_seed"] is not None
    assert isinstance(status["run"]["base_seed"], int)


async def _run_retry_scenario(conductor: IsaacConductor, always_fails: bool) -> dict[str, Any]:
    """Two blocks: "block-a" always places (giving each pass progress so a
    re-census is worth it), "block-b" fails according to ``always_fails``
    (always, or once-then-succeeds). Returns the final status."""
    items = [
        WorkItem("block-a", "red", 0.0, 0.0, 60.0),
        WorkItem("block-b", "red", 200.0, 0.0, 60.0),
    ]
    call_counts: dict[str, int] = {}

    async def scan() -> list[WorkItem]:
        return items

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        call_counts[prim_name] = call_counts.get(prim_name, 0) + 1
        if prim_name == "block-a":
            return {"outcome": OUTCOME_PLACED, "prim": prim_name}
        if always_fails:
            return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": "always fails"}
        if call_counts[prim_name] == 1:
            return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": "first attempt fails"}
        return {"outcome": OUTCOME_PLACED, "prim": prim_name}

    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    status["_call_counts"] = call_counts
    return status


async def test_retry_policy_terminal_failure_gets_exactly_two_attempts():
    conductor = _make_conductor()
    status = await _run_retry_scenario(conductor, always_fails=True)

    assert status["_call_counts"]["block-b"] == MAX_ATTEMPTS_PER_BLOCK
    assert status["_call_counts"]["block-b"] != 1
    assert status["_call_counts"]["block-b"] != MAX_ATTEMPTS_PER_BLOCK + 1
    outcome = status["outcomes"]["block-b"]
    assert outcome["outcome"] == OUTCOME_FAILED
    assert outcome["attempts"] == MAX_ATTEMPTS_PER_BLOCK


async def test_placed_tallest_ratchets_up_and_briefs_each_next_pick():
    """The place keep-out's height source: pick N is briefed with the tallest
    block already placed this loop (None before the first placement)."""
    conductor = _make_conductor()
    items = [
        WorkItem("block-a", "red", 0.0, 0.0, 60.0),
        WorkItem("block-b", "red", 200.0, 0.0, 75.0),
        WorkItem("block-c", "red", 400.0, 0.0, 55.0),
    ]
    briefed: list[float | None] = []
    sizes_called: list[float] = []

    async def scan() -> list[WorkItem]:
        return items

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        briefed.append(placed_tallest_mm)
        sizes_called.append(item.size_mm)
        return {"outcome": OUTCOME_PLACED, "prim": prim_name}

    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    assert len(briefed) == 3
    assert briefed[0] is None
    assert briefed[1] == sizes_called[0]
    assert briefed[2] == max(sizes_called[0], sizes_called[1])


async def test_retry_policy_recovers_on_second_attempt_and_reports_placed():
    conductor = _make_conductor()
    status = await _run_retry_scenario(conductor, always_fails=False)

    assert status["_call_counts"]["block-b"] == 2
    outcome = status["outcomes"]["block-b"]
    assert outcome["outcome"] == OUTCOME_PLACED
    assert outcome["attempts"] == 2


async def test_pick_duration_covers_only_up_to_its_own_terminal_attempt(
    monkeypatch: pytest.MonkeyPatch,
):
    # a fixed-step fake clock: every call to time.monotonic() advances by
    # exactly one step, so a pick's duration_s is exactly the count of
    # monotonic() calls made between its first and terminal attempt -
    # deterministic, no real-time sleeps
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(conductor_module.time, "monotonic", fake_monotonic)

    items = [
        WorkItem("block-a", "red", 0.0, 0.0, 60.0),
        WorkItem("block-b", "red", 200.0, 0.0, 60.0),
    ]
    call_counts: dict[str, int] = {}

    async def scan() -> list[WorkItem]:
        return items

    async def run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        call_counts[prim_name] = call_counts.get(prim_name, 0) + 1
        if prim_name == "block-a":
            # places on its only (pass-1) attempt
            return {"outcome": OUTCOME_PLACED, "prim": prim_name}
        # fails pass 1, places on its pass-2 retry
        if call_counts[prim_name] == 1:
            return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": "fails once"}
        return {"outcome": OUTCOME_PLACED, "prim": prim_name}

    conductor = _make_conductor()
    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()

    status = await conductor.do_command({"command": "status"})
    picks_by_name = {pick["name"]: pick for pick in status["loop_records"][-1]["picks"]}

    # block-a's terminal (and only) attempt is pass 1 - its duration is
    # exactly the one step between its first-attempt timestamp and its own
    # outcome landing, and must not have absorbed pass 2's extra work
    assert picks_by_name["block-a"]["duration_s"] == 1.0
    # block-b's terminal attempt is pass 2 - its duration spans the retry
    assert picks_by_name["block-b"]["duration_s"] > picks_by_name["block-a"]["duration_s"]


async def test_record_id_is_monotonic_across_a_second_start():
    conductor = _empty_resolve_prims_conductor()

    await conductor.do_command({"command": "start", "seed": 1, "loops": 2})
    await conductor.wait_until_done()
    first_run_ids = [
        record["record_id"]
        for record in (await conductor.do_command({"command": "status"}))["loop_records"]
    ]

    await conductor.do_command({"command": "start", "seed": 2, "loops": 2})
    await conductor.wait_until_done()
    all_records = (await conductor.do_command({"command": "status"}))["loop_records"]
    second_run_ids = [record["record_id"] for record in all_records[len(first_run_ids) :]]

    assert first_run_ids == [1, 2]
    assert second_run_ids == [3, 4]
    assert len(all_records) == 4


async def test_run_counters_reset_on_a_new_start():
    conductor = _make_conductor()

    items = [WorkItem("red-0", "red", 0.0, 0.0, 60.0)]

    async def scan() -> list[WorkItem]:
        return items

    async def failing_run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        return {"outcome": OUTCOME_FAILED, "prim": prim_name, "reason": "always fails"}

    conductor._scan_work_list = scan  # type: ignore[method-assign]
    conductor._resolve_prims = _passthrough_resolve_prims  # type: ignore[method-assign]
    conductor._run_one = failing_run_one  # type: ignore[method-assign]
    conductor._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()
    first_status = await conductor.do_command({"command": "status"})
    assert first_status["run"]["failed"] == 1

    async def placing_run_one(
        item: WorkItem,
        prim_name: str,
        prim_color: str,
        tracker: Any,
        placed_tallest_mm: float | None = None,
    ) -> dict[str, Any]:
        return {"outcome": OUTCOME_PLACED, "prim": prim_name}

    conductor._run_one = placing_run_one  # type: ignore[method-assign]

    await conductor.do_command({"command": "start"})
    await conductor.wait_until_done()
    second_status = await conductor.do_command({"command": "status"})
    assert second_status["run"]["failed"] == 0
    assert second_status["run"]["placed"] == 1
