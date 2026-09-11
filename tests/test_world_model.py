import asyncio
import itertools
import logging
import threading
import time

import pytest
from grpclib import Status
from viam.components.arm import JointPositions
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import cell_layout
from isaac_module.errors import PrimNotFoundError
from isaac_module.models.arm import IsaacArm
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import (
    DEFAULT_MIN_SEPARATION_M,
    UR_JOINT_NAMES,
    RandomizeResult,
    ScatterCellResult,
    SimManager,
)

MIN_SEPARATION_MM = DEFAULT_MIN_SEPARATION_M * 1000.0

POOL_NAMES = [
    cell_layout.pool_block_name(color, index)
    for color in cell_layout.BLOCK_COLORS
    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
]

# scatter_cell/clear_cell take their names, region and park grid from the
# DoCommand payload (the world component knows no cell): these are the
# same values cell_layout carries, in the shapes the payload expects.
POOL_NAMES_BY_COLOR = {
    color: [
        cell_layout.pool_block_name(color, index)
        for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
    ]
    for color in cell_layout.BLOCK_COLORS
}
_SCATTER_REGION_M = cell_layout.scatter_region_m()
SCATTER_REGION_MM = [
    [v * 1000.0 for v in _SCATTER_REGION_M[0]],
    [v * 1000.0 for v in _SCATTER_REGION_M[1]],
]
PARK_POSITIONS_M = cell_layout.park_positions_m()
PARK_POSITIONS_MM = {name: [x * 1000.0, y * 1000.0] for name, (x, y) in PARK_POSITIONS_M.items()}


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


class _RecordingWorldHandle:
    """Duck-types WorldHandle and records every call it receives, so the
    every-verb test can assert do_command reaches it and nothing else."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def status(self) -> dict:
        self.calls.append(("status",))
        return {"recorded": True}

    def play(self) -> None:
        self.calls.append(("play",))

    def pause(self) -> None:
        self.calls.append(("pause",))

    def reset(self, soft: bool = False) -> None:
        self.calls.append(("reset", soft))

    def add_usd(self, usd_path, prim_path, position_m, orientation_wxyz=None) -> None:
        self.calls.append(("add_usd", usd_path, prim_path, position_m, orientation_wxyz))

    def prop_geometries(self) -> list:
        self.calls.append(("prop_geometries",))
        return []

    def spawn_prop(self, prop) -> None:
        self.calls.append(("spawn_prop", prop))

    def set_prop_pose(self, name, position_m, orientation_wxyz=None) -> None:
        self.calls.append(("set_prop_pose", name, position_m, orientation_wxyz))

    def randomize_props(
        self,
        names,
        region,
        seed,
        min_separation_m=DEFAULT_MIN_SEPARATION_M,
        size_range_m=None,
    ):
        self.calls.append(("randomize_props", names, region, seed, min_separation_m, size_range_m))
        return RandomizeResult(
            positions_m={name: (0.0, 0.0, 0.0) for name in names},
            dims_m={name: (0.05, 0.05, 0.05) for name in names},
        )


def test_every_verb_routes_through_handle(world, monkeypatch):
    fake = _RecordingWorldHandle()
    monkeypatch.setattr(SimManager, "world_handle", lambda self: fake)

    def _boom(*args, **kwargs):
        raise AssertionError("do_command must not call SimManager scene methods directly")

    monkeypatch.setattr(SimManager, "play", _boom)
    monkeypatch.setattr(SimManager, "pause", _boom)
    monkeypatch.setattr(SimManager, "reset", _boom)
    monkeypatch.setattr(SimManager, "status", _boom)
    monkeypatch.setattr(SimManager, "add_usd_reference", _boom)

    async def scenario():
        await world.do_command({"command": "status"})
        await world.do_command({"command": "play"})
        await world.do_command({"command": "pause"})
        await world.do_command({"command": "reset"})
        await world.do_command({"command": "reset", "soft": True})
        await world.do_command(
            {
                "command": "add_usd",
                "usd_path": "a.usd",
                "prim_path": "/World/a",
                "position": [1.0, 2.0, 3.0],
            }
        )
        await world.do_command({"command": "prop_geometries"})
        await world.do_command({"command": "spawn_prop", "prop": {"name": "verb_test_prop"}})
        await world.do_command(
            {"command": "set_prop_pose", "name": "verb_test_prop", "position": [10.0, 20.0, 30.0]}
        )
        await world.do_command(
            {
                "command": "randomize_props",
                "names": ["verb_test_prop"],
                "region": [[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]],
                "seed": 1,
            }
        )

    asyncio.run(scenario())

    verbs = [call[0] for call in fake.calls]
    assert verbs == [
        "status",
        "play",
        "pause",
        "reset",
        "reset",
        "add_usd",
        "prop_geometries",
        "spawn_prop",
        "set_prop_pose",
        "randomize_props",
    ]
    assert fake.calls[3] == ("reset", False)
    assert fake.calls[4] == ("reset", True)


def test_do_command_runs_the_handler_off_the_event_loop(world, monkeypatch):
    """do_command must not block the module's event loop: the handler has
    to actually execute on a worker thread, not just get scheduled as if
    it would. A synchronous `handler(...)` call here (the bug this guards
    against) would record the event loop's own thread id instead."""
    main_thread_id = threading.get_ident()
    seen_thread_ids: list[int] = []

    class _ThreadRecordingHandle(_RecordingWorldHandle):
        def status(self) -> dict:
            seen_thread_ids.append(threading.get_ident())
            return super().status()

    monkeypatch.setattr(SimManager, "world_handle", lambda self: _ThreadRecordingHandle())

    asyncio.run(world.do_command({"command": "status"}))

    assert seen_thread_ids
    assert seen_thread_ids[0] != main_thread_id


def test_get_geometries_reads_prop_geometries_off_the_event_loop(world, monkeypatch):
    main_thread_id = threading.get_ident()
    seen_thread_ids: list[int] = []

    class _ThreadRecordingHandle(_RecordingWorldHandle):
        def prop_geometries(self) -> list:
            seen_thread_ids.append(threading.get_ident())
            return []

    monkeypatch.setattr(SimManager, "world_handle", lambda self: _ThreadRecordingHandle())

    asyncio.run(world.get_geometries())

    assert seen_thread_ids
    assert seen_thread_ids[0] != main_thread_id


def test_spawn_prop_and_prop_geometries_round_trip(world):
    prop = {
        "name": "rt_prop",
        "type": "cube",
        "position": [0.1, 0.2, 0.3],
        "size": 0.05,
        "scale": [1.0, 2.0, 3.0],
    }

    async def scenario():
        await world.do_command({"command": "spawn_prop", "prop": prop})
        return await world.do_command({"command": "prop_geometries"})

    result = asyncio.run(scenario())
    entry = next(g for g in result["geometries"] if g["name"] == "rt_prop")
    pose = entry["pose_in_world_mm"]
    assert pose["x"] == pytest.approx(100.0)
    assert pose["y"] == pytest.approx(200.0)
    assert pose["z"] == pytest.approx(300.0)
    assert pose["theta"] == pytest.approx(0.0, abs=1e-6)
    assert entry["box_dims_mm"][0] == pytest.approx(50.0)
    assert entry["box_dims_mm"][1] == pytest.approx(100.0)
    assert entry["box_dims_mm"][2] == pytest.approx(150.0)


def test_spawn_prop_with_orientation_reports_rotated_pose(world):
    async def scenario():
        await world.do_command(
            {"command": "spawn_prop", "prop": {"name": "rot_base", "position": [0.0, 0.0, 0.0]}}
        )
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {
                    "name": "rot_test",
                    "position": [0.0, 0.0, 0.0],
                    "orientation_rpy_deg": [0.0, 90.0, 0.0],
                },
            }
        )
        return await world.do_command({"command": "prop_geometries"})

    result = asyncio.run(scenario())
    geoms = {g["name"]: g["pose_in_world_mm"] for g in result["geometries"]}
    base, rotated = geoms["rot_base"], geoms["rot_test"]
    changed = (
        rotated["o_x"] != pytest.approx(base["o_x"])
        or rotated["o_y"] != pytest.approx(base["o_y"])
        or rotated["theta"] != pytest.approx(base["theta"])
    )
    assert changed


def test_set_prop_pose_then_reset_restores_configured_pose(world):
    async def scenario():
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {"name": "reset_prop", "position": [0.1, 0.1, 0.1]},
            }
        )
        await world.do_command(
            {
                "command": "set_prop_pose",
                "name": "reset_prop",
                "position": [500.0, 500.0, 500.0],
            }
        )
        moved = await world.do_command({"command": "prop_geometries"})
        await world.do_command({"command": "reset"})
        after = await world.do_command({"command": "prop_geometries"})
        return moved, after

    moved, after = asyncio.run(scenario())
    moved_entry = next(g for g in moved["geometries"] if g["name"] == "reset_prop")
    after_entry = next(g for g in after["geometries"] if g["name"] == "reset_prop")
    assert moved_entry["pose_in_world_mm"]["x"] == pytest.approx(500.0)
    assert after_entry["pose_in_world_mm"]["x"] == pytest.approx(100.0)


def test_randomize_props_deterministic_and_within_region(world):
    names = ["rand_a", "rand_b", "rand_c"]
    region = [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]]

    async def scenario():
        for name in names:
            await world.do_command(
                {"command": "spawn_prop", "prop": {"name": name, "position": [0.0, 0.0, 0.0]}}
            )
        first = await world.do_command(
            {"command": "randomize_props", "names": names, "region": region, "seed": 1}
        )
        second = await world.do_command(
            {"command": "randomize_props", "names": names, "region": region, "seed": 1}
        )
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second
    positions = first["positions"]
    for name in names:
        x, y, _z = positions[name]
        assert 0.0 <= x <= 1000.0
        assert 0.0 <= y <= 1000.0
    for name_a, name_b in itertools.combinations(names, 2):
        ax, ay, _ = positions[name_a]
        bx, by, _ = positions[name_b]
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        assert distance >= MIN_SEPARATION_MM - 1e-6

    sizes = first["sizes_mm"]
    for name in names:
        assert sizes[name] == pytest.approx([50.0, 50.0, 50.0])  # unranged: default cube size


def test_randomize_props_size_range_mm_list_form_reaches_the_handle_in_meters(world):
    names = ["sz_a", "sz_b"]
    region = [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]]

    async def scenario():
        for name in names:
            await world.do_command({"command": "spawn_prop", "prop": {"name": name}})
        return await world.do_command(
            {
                "command": "randomize_props",
                "names": names,
                "region": region,
                "seed": 1,
                "size_range_mm": [30.0, 90.0],
            }
        )

    result = asyncio.run(scenario())
    for name in names:
        x, y, z = result["sizes_mm"][name]
        assert 30.0 <= x <= 90.0
        assert x == y == z


def test_randomize_props_size_range_mm_map_form_reaches_the_handle_in_meters(world):
    names = ["sz_c", "sz_d"]
    region = [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]]

    async def scenario():
        for name in names:
            await world.do_command({"command": "spawn_prop", "prop": {"name": name}})
        return await world.do_command(
            {
                "command": "randomize_props",
                "names": names,
                "region": region,
                "seed": 1,
                "size_range_mm": {"sz_c": [30.0, 90.0]},
            }
        )

    result = asyncio.run(scenario())
    x, y, z = result["sizes_mm"]["sz_c"]
    assert 30.0 <= x <= 90.0
    assert x == y == z
    assert result["sizes_mm"]["sz_d"] == pytest.approx([50.0, 50.0, 50.0])


def test_randomize_props_size_range_mm_validation_errors(world):
    async def randomize(size_range_mm):
        return await world.do_command(
            {
                "command": "randomize_props",
                "names": ["sz_bad"],
                "region": [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]],
                "seed": 1,
                "size_range_mm": size_range_mm,
            }
        )

    asyncio.run(world.do_command({"command": "spawn_prop", "prop": {"name": "sz_bad"}}))

    with pytest.raises(ValueError):
        asyncio.run(randomize([0.0, 90.0]))  # lo not > 0
    with pytest.raises(ValueError):
        asyncio.run(randomize([90.0, 30.0]))  # lo > hi
    with pytest.raises(ValueError):
        asyncio.run(randomize([30.0]))  # wrong arity
    with pytest.raises(ValueError):
        asyncio.run(randomize(["a", 90.0]))  # non-number entry
    with pytest.raises(ValueError):
        asyncio.run(randomize({"not_sz_bad": [30.0, 90.0]}))  # key not in names


def test_ignore_props_and_get_geometries(world):
    async def scenario():
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {"name": "geo_a", "position": [0.1, 0.2, 0.3], "size": 0.05},
            }
        )
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {
                    "name": "geo_b",
                    "type": "usd",
                    "usd_path": "x.usd",
                    "position": [0.0, 0.0, 0.0],
                },
            }
        )
        await world.do_command({"command": "ignore_props", "names": ["geo_a"]})
        while_ignored = await world.get_geometries()
        await world.do_command({"command": "ignore_props", "names": []})
        after_clear = await world.get_geometries()
        return while_ignored, after_clear

    while_ignored, after_clear = asyncio.run(scenario())

    labels_while_ignored = {g.label for g in while_ignored}
    assert "geo_a" not in labels_while_ignored
    assert "geo_b" not in labels_while_ignored  # zero (unknown) dims stays excluded

    labels_after_clear = {g.label for g in after_clear}
    assert "geo_a" in labels_after_clear
    assert "geo_b" not in labels_after_clear

    entry = next(g for g in after_clear if g.label == "geo_a")
    assert entry.center.x == pytest.approx(100.0)
    assert entry.box.dims_mm.x == pytest.approx(50.0)


def test_spawn_prop_validation_errors(world):
    async def spawn(prop):
        return await world.do_command({"command": "spawn_prop", "prop": prop})

    with pytest.raises(ValueError, match="orientation_rpy_deg"):
        asyncio.run(spawn({"name": "bad_1", "orientation_rpy_deg": [1.0, 2.0]}))

    with pytest.raises(ValueError, match="only one of"):
        asyncio.run(
            spawn(
                {
                    "name": "bad_2",
                    "orientation_rpy_deg": [1.0, 2.0, 3.0],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            )
        )

    with pytest.raises(ValueError, match="box_dims"):
        asyncio.run(spawn({"name": "bad_3", "box_dims": [1.0, -1.0, 1.0]}))

    with pytest.raises(ValueError, match="prop"):
        asyncio.run(world.do_command({"command": "spawn_prop"}))


@pytest.fixture(scope="module")
def pool_world(world):
    """The session ``world`` fixture with all 18 pool blocks spawned once,
    so scatter_cell/clear_cell have prims to draw from and park."""

    async def spawn():
        for name in POOL_NAMES:
            x, y = cell_layout.park_positions_m()[name]
            await world.do_command(
                {
                    "command": "spawn_prop",
                    "prop": {
                        "type": "cube",
                        "name": name,
                        "size": 0.06,
                        "position": [x, y, 0.03 + 0.0005],
                    },
                }
            )

    asyncio.run(spawn())
    return world


def _scatter_command(**overrides: object) -> dict:
    command: dict = {
        "command": "scatter_cell",
        "names_by_color": POOL_NAMES_BY_COLOR,
        "region": SCATTER_REGION_MM,
        "park_positions_mm": PARK_POSITIONS_MM,
    }
    command.update(overrides)
    return command


def _clear_command() -> dict:
    return {
        "command": "clear_cell",
        "names_by_color": POOL_NAMES_BY_COLOR,
        "park_positions_mm": PARK_POSITIONS_MM,
    }


def test_scatter_cell_response_shape_and_mm_conversion(pool_world):
    handle = pool_world._handle()
    seed = 9001

    async def scenario():
        return await pool_world.do_command(_scatter_command(seed=seed, size_range_mm=[30.0, 90.0]))

    result = asyncio.run(scenario())
    direct = handle.scatter_cell(
        POOL_NAMES_BY_COLOR, _SCATTER_REGION_M, PARK_POSITIONS_M, seed, size_range_m=(0.03, 0.09)
    )

    assert result["seed"] == seed
    assert result["counts"] == direct.counts
    assert set(result["positions"]) == set(direct.positions_m)
    assert set(result["sizes_mm"]) == set(direct.sizes_m)
    assert sorted(result["parked"]) == sorted(direct.parked)
    assert set(result["positions"]) | set(result["parked"]) == set(POOL_NAMES)

    for name, position_m in direct.positions_m.items():
        assert result["positions"][name] == pytest.approx([v * 1000.0 for v in position_m])
    for name, dims_m in direct.sizes_m.items():
        assert result["sizes_mm"][name] == pytest.approx([v * 1000.0 for v in dims_m])


def test_scatter_cell_same_seed_is_deterministic(pool_world):
    async def scatter():
        return await pool_world.do_command(_scatter_command(seed=42))

    first = asyncio.run(scatter())
    second = asyncio.run(scatter())
    assert first == second
    assert set(first["positions"]) | set(first["parked"]) == set(POOL_NAMES)


def test_scatter_cell_missing_seed_raises_value_error(pool_world):
    with pytest.raises(ValueError):
        asyncio.run(pool_world.do_command(_scatter_command()))


def test_scatter_cell_bad_size_range_mm_raises_value_error(pool_world):
    with pytest.raises(ValueError):
        asyncio.run(pool_world.do_command(_scatter_command(seed=1, size_range_mm=[90.0, 30.0])))


def test_clear_cell_parks_all_eighteen_pool_blocks(pool_world):
    async def scenario():
        await pool_world.do_command(_scatter_command(seed=7))
        return await pool_world.do_command(_clear_command())

    result = asyncio.run(scenario())
    assert sorted(result["parked"]) == sorted(POOL_NAMES)


def test_scatter_cell_counts_override_zero_parks_the_whole_color(pool_world):
    async def scenario():
        return await pool_world.do_command(_scatter_command(seed=3, counts={"red": 0}))

    result = asyncio.run(scenario())
    red_names = [
        cell_layout.pool_block_name("red", index)
        for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
    ]
    assert result["counts"]["red"] == 0
    assert all(name not in result["positions"] for name in red_names)
    assert all(name in result["parked"] for name in red_names)


def test_scatter_cell_concurrent_calls_serialize(world, monkeypatch):
    """Off the event loop, two scatter_cell calls that arrive together both
    reach the handle; do_command's lock must still keep only one of them
    inside the handler at a time. Without the lock this would race up to
    max_active == 2."""
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class _SlowScatterHandle(_RecordingWorldHandle):
        def scatter_cell(
            self, names_by_color, region, park_positions_m, seed, size_range_m=None, counts=None
        ):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with active_lock:
                active -= 1
            return ScatterCellResult(seed=seed, counts={}, positions_m={}, sizes_m={}, parked=[])

    monkeypatch.setattr(SimManager, "world_handle", lambda self: _SlowScatterHandle())

    async def scenario():
        await asyncio.gather(
            world.do_command(_scatter_command(seed=1)),
            world.do_command(_scatter_command(seed=2)),
        )

    asyncio.run(scenario())
    assert max_active == 1


def test_usd_stage_without_lighting_warns(caplog):
    config = _config("isaac-world-warn", {"mock": True, "usd_stage": "foo.usd"})
    with caplog.at_level(logging.WARNING):
        IsaacWorld.new(config, {})
    assert any("stage must provide floor and lights" in record.message for record in caplog.records)


def test_do_command_dof_names(world):
    IsaacArm.new(
        _config("world-arm-dof-names", {"world": "isaac-world", "asset": "ur20", "mock_dof": 12}),
        {},
    )
    result = asyncio.run(world.do_command({"command": "dof_names", "name": "world-arm-dof-names"}))
    names = result["dof_names"]
    assert len(names) == 12
    assert list(names[:6]) == list(UR_JOINT_NAMES)


def test_do_command_all_dof_names(world):
    IsaacArm.new(
        _config(
            "world-arm-all-dof-names", {"world": "isaac-world", "asset": "ur20", "mock_dof": 12}
        ),
        {},
    )
    result = asyncio.run(
        world.do_command({"command": "dof_names", "name": "world-arm-all-dof-names", "all": True})
    )
    names = result["dof_names"]
    assert len(names) == 12
    assert list(names[:6]) == list(UR_JOINT_NAMES)


def test_do_command_prim_pose_default_prim(world):
    IsaacArm.new(
        _config("world-arm-prim-pose", {"world": "isaac-world", "asset": "ur5e", "mock_dof": 6}),
        {},
    )

    async def scenario():
        return await world.do_command({"command": "prim_pose", "name": "world-arm-prim-pose"})

    result = asyncio.run(scenario())
    # NOTE: the brief expected [-300, 0, 300] (root rotated by the ur5e
    # correction); MockArmHandle._ee_world_pose actually composes the
    # fixed local EE onto Viam's un-rotated base frame (it cancels the
    # correction out via viam_base_frame), so this is invariant to
    # base_frame_correction and always [300, 0, 300] here.
    assert result["position_mm"] == pytest.approx([300.0, 0.0, 300.0], abs=1e-3)
    assert len(result["quaternion_wxyz"]) == 4


def test_do_command_prim_pose_unknown_prim_raises(world):
    IsaacArm.new(
        _config(
            "world-arm-prim-pose-unknown",
            {"world": "isaac-world", "asset": "ur5e", "mock_dof": 6},
        ),
        {},
    )

    async def scenario():
        await world.do_command(
            {
                "command": "prim_pose",
                "name": "world-arm-prim-pose-unknown",
                "prim_path": "/World/nope",
            }
        )

    with pytest.raises(PrimNotFoundError) as excinfo:
        asyncio.run(scenario())
    assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT


def test_joint_state_do_command_reports_targets_next_to_positions(world):
    arm = IsaacArm.new(
        _config("world-arm-joint-state", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}),
        {},
    )

    async def scenario():
        await arm.move_to_joint_positions(JointPositions(values=[10, -20, 30, 0, 5, -5]))
        return await world.do_command({"command": "joint_state", "name": "world-arm-joint-state"})

    out = asyncio.run(scenario())
    joints = out["joints"]
    assert [j["name"] for j in joints] == list(UR_JOINT_NAMES)
    assert all(j["named"] for j in joints)
    assert [j["target_deg"] for j in joints] == pytest.approx([10, -20, 30, 0, 5, -5])
    assert [j["position_deg"] for j in joints] == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)


def test_tcp_pose_do_command_measures_the_configured_offset_in_mock(world):
    """GPU checklist item 4: the fingertip midpoint sits tcp_offset_m along the
    mount link's +Z, so measured == configured and delta is 0 in the mock."""
    arm = IsaacArm.new(_config("world-tcp-arm", {"world": "isaac-world", "asset": "ur5e"}), {})
    IsaacGripper.new(
        _config("world-tcp-grip", {"world": "isaac-world", "arm": arm.name, "tcp_offset_m": 0.115}),
        {},
    )
    out = asyncio.run(world.do_command({"command": "tcp_pose", "name": "world-tcp-grip"}))
    assert out["configured_tcp_offset_mm"] == pytest.approx(115.0)
    assert out["measured_tcp_offset_mm"] == pytest.approx(115.0)
    assert out["delta_mm"] == pytest.approx(0.0)
    assert out["pad_center_midpoint_mm"] == pytest.approx([0.0, 0.0, 115.0])
    assert out["jaw_gap_mm"] == pytest.approx(85.0)
    assert out["fingertip_reach_mm"] == pytest.approx(115.0 + 19.0)
    assert set(out) >= {"parent", "left_inner_finger", "right_inner_finger", "fingertips"}


def test_do_command_unknown_name_raises(world):
    with pytest.raises(ValueError, match="no sim component named"):
        asyncio.run(world.do_command({"command": "joint_state", "name": "does-not-exist"}))


def test_do_command_wrong_kind_name_raises(world):
    arm = IsaacArm.new(
        _config("world-wrong-kind-arm", {"world": "isaac-world", "asset": "ur5e", "mock_dof": 6}),
        {},
    )
    IsaacGripper.new(
        _config("world-wrong-kind-grip", {"world": "isaac-world", "arm": arm.name}), {}
    )
    with pytest.raises(ValueError, match="not an arm"):
        asyncio.run(world.do_command({"command": "joint_state", "name": "world-wrong-kind-grip"}))


def test_unknown_command_lists_verbs(world):
    with pytest.raises(ViamGRPCError) as exc_info:
        asyncio.run(world.do_command({"command": "bogus"}))
    assert exc_info.value.grpc_code == Status.INVALID_ARGUMENT
    message = exc_info.value.message
    for verb in (
        "status",
        "play",
        "pause",
        "reset",
        "add_usd",
        "prop_geometries",
        "spawn_prop",
        "set_prop_pose",
        "randomize_props",
        "ignore_props",
    ):
        assert verb in message


def test_get_geometries_serves_the_floor_when_the_module_owns_the_stage(world):
    async def scenario():
        await world.do_command({"command": "ignore_props", "names": []})
        return await world.get_geometries()

    geometries = asyncio.run(scenario())
    floor = next(g for g in geometries if g.label == "floor")
    assert floor.box.dims_mm.z == 200.0
    assert floor.center.z == -100.0  # top face exactly at z = 0

    async def hide_floor():
        await world.do_command({"command": "ignore_props", "names": ["floor"]})
        try:
            return await world.get_geometries()
        finally:
            await world.do_command({"command": "ignore_props", "names": []})

    assert all(g.label != "floor" for g in asyncio.run(hide_floor()))


def test_get_geometries_serves_no_floor_over_a_user_stage():
    from isaac_module.models.world import IsaacWorld

    stage_world = IsaacWorld.new(
        _config("stage-world", {"mock": True, "usd_stage": "user_stage.usd"}), {}
    )

    async def scenario():
        return await stage_world.get_geometries()

    assert all(g.label != "floor" for g in asyncio.run(scenario()))
