import asyncio
import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "gpu_checklist_world.py"
_spec = importlib.util.spec_from_file_location("gpu_checklist_world", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
checklist = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_world"] = checklist
_spec.loader.exec_module(checklist)

RANDOMIZE_PROP = "gpu_checklist_world_test_randomize_cube"
SPAWN_TELEPORT_PROP = "gpu_checklist_world_test_spawn_teleport_cube"
SOFT_RESET_PROP = "gpu_checklist_world_test_soft_reset_cube"

RANDOMIZE_REGION_MM = ((300.0, -300.0, 50.0), (900.0, 300.0, 50.0))
SPAWN_POSITION_MM = (500.0, 0.0, 100.0)
TELEPORT_POSITION_MM = (700.0, 200.0, 150.0)
TINY_WINDOW_S = 0.02


def test_pose_close_and_prop_pose_mm_helpers():
    geometries = [
        {"name": "a", "pose_in_world_mm": {"x": 1.0, "y": 2.0, "z": 3.0}},
        {"name": "b", "pose_in_world_mm": {"x": 4.0, "y": 5.0, "z": 6.0}},
    ]
    assert checklist.prop_pose_mm(geometries, "b") == {"x": 4.0, "y": 5.0, "z": 6.0}
    assert checklist.prop_pose_mm(geometries, "missing") is None
    assert checklist.pose_close({"x": 1.0, "y": 1.0, "z": 1.0}, {"x": 1.0, "y": 1.0, "z": 1.0})
    assert not checklist.pose_close({"x": 1.0, "y": 1.0, "z": 1.0}, {"x": 2.0, "y": 1.0, "z": 1.0})


async def _spawn(world, name: str, position_mm) -> None:
    await world.do_command(
        {
            "command": "spawn_prop",
            "prop": {
                "name": name,
                "type": "cube",
                "position": [v / 1000.0 for v in position_mm],
                "size": 0.05,
            },
        }
    )


def test_randomize_props_determinism_same_seed_deterministic_different_seed_reported(world):
    async def scenario():
        await _spawn(world, RANDOMIZE_PROP, SPAWN_POSITION_MM)
        return await checklist.run_randomize_props_determinism(
            world, names=[RANDOMIZE_PROP], region_mm=RANDOMIZE_REGION_MM, seed_a=11, seed_b=12
        )

    report = asyncio.run(scenario())

    assert report["deterministic_same_seed"] is True
    assert set(report["positions_seed_a"]) == {RANDOMIZE_PROP}
    assert set(report["positions_seed_b"]) == {RANDOMIZE_PROP}
    assert isinstance(report["prop_geometries_before"], list)
    assert isinstance(report["prop_geometries_after"], list)
    assert "seed=12" not in report["pick_command"]  # the seed is interpolated, not literal
    assert "--randomize-seed 12" in report["pick_command"]


def test_randomize_props_determinism_different_seeds_usually_differ(world):
    async def scenario():
        await checklist.run_randomize_props_determinism(
            world, names=[RANDOMIZE_PROP], region_mm=RANDOMIZE_REGION_MM, seed_a=21, seed_b=22
        )
        return await checklist.run_randomize_props_determinism(
            world, names=[RANDOMIZE_PROP], region_mm=RANDOMIZE_REGION_MM, seed_a=21, seed_b=99
        )

    report = asyncio.run(scenario())
    assert report["positions_seed_a"][RANDOMIZE_PROP] != report["positions_seed_b"][RANDOMIZE_PROP]


def test_step_rate_measurement_mock_has_no_sim_time_but_reports_wall_time(world):
    calls = []

    async def rgb_activity():
        calls.append("rgb")
        await asyncio.sleep(TINY_WINDOW_S)

    async def scenario():
        return await checklist.run_step_rate_measurement(
            world, window_s=TINY_WINDOW_S, camera_activity={"rgb": rgb_activity}
        )

    report = asyncio.run(scenario())

    assert report["baseline"]["sim_time_ratio"] is None
    assert report["baseline"]["note"] == "mock: no sim_time"
    assert isinstance(report["baseline"]["wall_s"], float)
    assert report["baseline"]["wall_s"] >= 0.0

    assert report["rgb"]["sim_time_ratio"] is None
    assert report["rgb"]["note"] == "mock: no sim_time"
    assert calls == ["rgb"]


def test_spawn_teleport_reset_round_trip(world):
    async def scenario():
        return await checklist.run_spawn_teleport_reset(
            world,
            prop_name=SPAWN_TELEPORT_PROP,
            spawn_position_mm=SPAWN_POSITION_MM,
            teleport_position_mm=TELEPORT_POSITION_MM,
        )

    report = asyncio.run(scenario())

    assert report["appeared_after_spawn"] is True
    assert report["moved_after_teleport"] is True
    assert report["restored_after_reset"] is True
    assert checklist.pose_close(report["spawn_pose_mm"], report["post_reset_pose_mm"])
    assert not checklist.pose_close(report["spawn_pose_mm"], report["teleported_pose_mm"])


def test_mesh_table_and_executor_checks_report_not_applicable():
    assert checklist.run_mesh_table_check() == checklist.MESH_TABLE_NOT_APPLICABLE_NOTE
    assert checklist.run_executor_check() == checklist.EXECUTOR_CHECK_DEFERRED_NOTE


def test_soft_reset_demo_restores_the_spawn_pose(world):
    async def scenario():
        await _spawn(world, SOFT_RESET_PROP, SPAWN_POSITION_MM)
        return await checklist.run_soft_reset_demo(
            world,
            prop_name=SOFT_RESET_PROP,
            teleport_position_mm=TELEPORT_POSITION_MM,
            spawn_position_mm=SPAWN_POSITION_MM,
        )

    report = asyncio.run(scenario())

    assert report["restored_matches_spawn"] is True
    assert report["teleported_at_commanded"] is True
    assert not checklist.pose_close(report["spawn_pose_mm"], report["teleported_pose_mm"])


def test_scene_randomization_defaults_derive_movable_names_and_face_height():
    geometries = [
        {
            "name": "block",
            "fixed": False,
            "box_dims_mm": [60.0, 60.0, 60.0],
            "pose_in_world_mm": {"x": 700.0, "y": 250.0, "z": 30.0},
        },
        {
            "name": "pad",
            "fixed": True,
            "box_dims_mm": [200.0, 200.0, 10.0],
            "pose_in_world_mm": {"x": 700.0, "y": -350.0, "z": 5.0},
        },
        {
            "name": "unknown_usd",
            "fixed": False,
            "box_dims_mm": [0.0, 0.0, 0.0],
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    ]
    names, (lo, hi) = checklist.scene_randomization_defaults(geometries)
    assert names == ["block"]
    # face = the block's bottom: 30 - 60/2 = 0 (the floor)
    assert (lo[2], hi[2]) == (0.0, 0.0)
    assert (lo[0], lo[1], hi[0], hi[1]) == (300.0, -300.0, 900.0, 300.0)


def test_scene_randomization_defaults_with_no_movable_props_returns_empty():
    names, region = checklist.scene_randomization_defaults(
        [
            {
                "name": "pad",
                "fixed": True,
                "box_dims_mm": [200.0, 200.0, 10.0],
                "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 5.0},
            }
        ]
    )
    assert names == []
    assert region == ([300.0, -300.0, 50.0], [900.0, 300.0, 50.0])


def test_camera_frame_activity_grabs_frames_from_the_mock_camera(world):
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from isaac_module.models.camera import IsaacCamera

    camera = IsaacCamera.new(
        ComponentConfig(
            name="gpu_checklist_world_test_cam",
            attributes=dict_to_struct(
                {"world": "isaac-world", "width": 64, "height": 48, "depth": True}
            ),
        ),
        {},
    )

    async def scenario():
        await checklist._camera_frame_activity(camera, depth=False)
        await checklist._camera_frame_activity(camera, depth=True)

    asyncio.run(scenario())


def test_sample_step_rate_keeps_the_activity_running_for_the_whole_window(world):
    calls = 0

    async def activity():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)

    async def scenario():
        return await checklist._sample_step_rate(world, window_s=0.05, activity=activity)

    result = asyncio.run(scenario())
    assert calls >= 2  # looped, not a single grab
    assert result["wall_s"] >= 0.05


def test_resting_at_commanded_allows_fall_but_not_drift_or_tumble():
    commanded = (500.0, 0.0, 100.0)
    at_spot = {"x": 500.0, "y": 0.0, "z": 100.0, "o_x": 0.0, "o_y": 0.0, "theta": 0.0}
    fallen = dict(at_spot, z=44.8)  # free fall below the commanded height is fine
    drifted = dict(at_spot, x=503.0)
    risen = dict(at_spot, z=101.0)
    tumbled = dict(fallen, o_x=0.013, theta=25.0)

    assert checklist.resting_at_commanded(at_spot, commanded)
    assert checklist.resting_at_commanded(fallen, commanded)
    assert not checklist.resting_at_commanded(drifted, commanded)
    assert not checklist.resting_at_commanded(risen, commanded)
    assert not checklist.resting_at_commanded(tumbled, commanded)
    assert not checklist.resting_at_commanded(None, commanded)
