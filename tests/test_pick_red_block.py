"""Unit + end-to-end tests for examples/pick_red_block.py (XC-8, W36, DEC-14, DEC-20).

The module lives under examples/ (not a package under src/) and depends only
on the stdlib, viam-sdk and numpy (isaac_module is imported lazily, only in
--mock code paths), so it is loaded here via importlib rather than adding
examples/ to pyproject's pythonpath - mirrors
tests/test_gpu_checklist_camera.py's idiom exactly.
"""

import argparse
import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from viam.proto.common import Pose

from isaac_module.encoding import xyz_rgb_to_pcd
from isaac_module.mock_camera import MockCameraHandle
from isaac_module.sim_manager import SimManager

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "pick_red_block.py"
_spec = importlib.util.spec_from_file_location("pick_red_block", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
pick_red_block = importlib.util.module_from_spec(_spec)
sys.modules["pick_red_block"] = pick_red_block
_spec.loader.exec_module(pick_red_block)

pre_grasp_pose = pick_red_block.pre_grasp_pose
grasp_pose = pick_red_block.grasp_pose
table_obstacle = pick_red_block.table_obstacle
held_block_transform = pick_red_block.held_block_transform
parse_pcd = pick_red_block.parse_pcd
red_centroid_m = pick_red_block.red_centroid_m
HELD_BLOCK_TRANSFORM_MARKER = pick_red_block.HELD_BLOCK_TRANSFORM_MARKER
DETECTED_BLOCK_POSE_MARKER = pick_red_block.DETECTED_BLOCK_POSE_MARKER
HOLD_SAMPLES_MARKER = pick_red_block.HOLD_SAMPLES_MARKER
GRAB_DIAGNOSTICS_MARKER = pick_red_block.GRAB_DIAGNOSTICS_MARKER
RESET_MID_HOLD_MARKER = pick_red_block.RESET_MID_HOLD_MARKER
MEASURED_BLOCK_MARKER = pick_red_block.MEASURED_BLOCK_MARKER
JAW_MAX_BLOCK_MM = pick_red_block.JAW_MAX_BLOCK_MM
footprint_extents_mm = pick_red_block.footprint_extents_mm
measured_block_size_mm = pick_red_block.measured_block_size_mm
TallestEstimate = pick_red_block.TallestEstimate
tallest_in_region_mm = pick_red_block.tallest_in_region_mm
keepout_height_mm = pick_red_block.keepout_height_mm
carry_clear_above_support_mm = pick_red_block.carry_clear_above_support_mm
tallest_sweep_attempts = pick_red_block.tallest_sweep_attempts
MEASURED_TALLEST_MARKER = pick_red_block.MEASURED_TALLEST_MARKER
main = pick_red_block.main


def _asymmetric_block_pose() -> Pose:
    # x, y, z all distinct so a wrong-axis bug in pre_grasp_pose/grasp_pose shows up.
    return Pose(x=111.0, y=222.0, z=333.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=45.0)


def test_pre_grasp_pose_adds_standoff_in_z_and_points_down():
    block_pose = _asymmetric_block_pose()
    pose = pre_grasp_pose(block_pose)
    assert pose.x == pytest.approx(111.0)
    assert pose.y == pytest.approx(222.0)
    assert pose.z == pytest.approx(333.0 + 100.0)
    assert pose.o_x == 0.0
    assert pose.o_y == 0.0
    assert pose.o_z == -1.0


def test_pre_grasp_pose_honours_custom_standoff():
    block_pose = _asymmetric_block_pose()
    pose = pre_grasp_pose(block_pose, standoff_mm=50.0)
    assert pose.z == pytest.approx(333.0 + 50.0)


def test_grasp_pose_keeps_block_z_and_points_down():
    block_pose = _asymmetric_block_pose()
    pose = grasp_pose(block_pose)
    assert pose.x == pytest.approx(111.0)
    assert pose.y == pytest.approx(222.0)
    assert pose.z == pytest.approx(333.0)
    assert pose.o_z == -1.0


def test_table_obstacle_matches_readme_recipe():
    table = table_obstacle()
    box = table.box
    assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (1200.0, 800.0, 750.0)
    center = table.center
    assert (center.x, center.y, center.z) == (-1200.0, 0.0, 375.0)


def test_held_block_transform_shape():
    transform = held_block_transform("pick_cube", 60.0, "pick-grip")
    assert transform.reference_frame == "pick_cube"
    assert transform.pose_in_observer_frame.reference_frame == "pick-grip"
    box = transform.physical_object.box
    assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (60.0, 60.0, 60.0)
    assert transform.physical_object.label == "pick_cube"


def test_parse_pcd_round_trips_xyz_rgb_to_pcd():
    xyz = np.array(
        [[0.1, 0.2, 0.3], [1.0, -1.0, 2.0], [0.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    rgb = np.array([[224, 32, 32], [10, 200, 10], [50, 50, 50]], dtype=np.uint8)
    data = xyz_rgb_to_pcd(xyz, rgb)

    parsed_xyz, parsed_rgb = parse_pcd(data)

    assert parsed_xyz == pytest.approx(xyz, abs=1e-6)
    assert parsed_rgb is not None
    assert parsed_rgb.tolist() == rgb.tolist()


def test_parse_pcd_uncoloured():
    xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    data = xyz_rgb_to_pcd(xyz, None)

    parsed_xyz, parsed_rgb = parse_pcd(data)

    assert parsed_xyz == pytest.approx(xyz, abs=1e-6)
    assert parsed_rgb is None


def test_red_centroid_m_ignores_grey_and_returns_red_mean():
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # grey - ignored
            [1.0, 0.0, 0.0],  # red
            [3.0, 0.0, 0.0],  # red
            [10.0, 10.0, 10.0],  # grey - ignored
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [50, 50, 50],
            [224, 32, 32],
            [200, 10, 10],
            [90, 90, 90],
        ],
        dtype=np.uint8,
    )
    centroid = red_centroid_m(points, colors)
    assert centroid == pytest.approx((2.0, 0.0, 0.0))


def test_red_centroid_m_raises_when_no_red_points():
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[50, 50, 50]], dtype=np.uint8)
    with pytest.raises(ValueError, match="no red points"):
        red_centroid_m(points, colors)


def test_main_mock_runs_the_full_pick_sequence(capsys):
    # --block-size-mm explicit: today's fixed-size path, no measurement (the
    # mock's default fixed pixel rectangle measures ~82 mm, over the jaw limit)
    exit_code = main(["--mock", "--hold-s", "0", "--block-size-mm", "60"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "grab: True" in out

    transform_line = next(
        line for line in out.splitlines() if line.startswith(HELD_BLOCK_TRANSFORM_MARKER)
    )
    payload = json.loads(transform_line.removeprefix(HELD_BLOCK_TRANSFORM_MARKER))
    assert payload["reference_frame"] == "block_red_1"
    assert payload["pose_in_observer_frame"]["reference_frame"] == "mock-pick-grip"

    camera_handle = SimManager.get()._handles["mock-wrist-cam"][1]
    assert isinstance(camera_handle, MockCameraHandle)
    expected_center_mm = tuple(v * 1000.0 for v in camera_handle.red_block_center_m)

    detected_line = next(
        line for line in out.splitlines() if line.startswith(DETECTED_BLOCK_POSE_MARKER)
    )
    detected = json.loads(detected_line.removeprefix(DETECTED_BLOCK_POSE_MARKER))
    delta_mm = max(
        abs(detected["x"] - expected_center_mm[0]),
        abs(detected["y"] - expected_center_mm[1]),
        abs(detected["z"] - expected_center_mm[2]),
    )
    assert delta_mm <= 20.0


def test_look_pose_points_the_camera_down_at_the_requested_point():
    pose = pick_red_block.look_pose_from("700,250,500")
    assert (pose.x, pose.y, pose.z) == (700.0, 250.0, 500.0)
    assert pose.o_z == -1.0 and pose.o_x == 0.0 and pose.o_y == 0.0


def test_main_mock_runs_the_look_step_before_detect(capsys):
    assert pick_red_block.main(["--mock", "--hold-s", "0", "--block-size-mm", "60"]) == 0
    out = capsys.readouterr().out
    assert out.index("step: look") < out.index("step: detect")


def test_main_mock_holds_and_survives_a_reset_mid_hold(capsys, monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.05)
    monkeypatch.setattr(pick_red_block, "HOLD_SAMPLE_S", 0.05)
    exit_code = main(["--mock", "--hold-s", "0.1", "--reset-mid-hold", "--block-size-mm", "60"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.index("step: move to lift") < out.index("step: hold at the lift pose")
    assert out.index("step: hold at the lift pose") < out.index("step: world reset mid-hold")
    assert out.index("step: world reset mid-hold") < out.index("step: open (release)")

    samples_line = next(line for line in out.splitlines() if line.startswith(HOLD_SAMPLES_MARKER))
    samples = json.loads(samples_line.removeprefix(HOLD_SAMPLES_MARKER))
    assert [sample["holding"] for sample in samples] == [True, True]
    assert all(sample["jaw_deg"] is not None for sample in samples)

    reset_line = next(line for line in out.splitlines() if line.startswith(RESET_MID_HOLD_MARKER))
    report = json.loads(reset_line.removeprefix(RESET_MID_HOLD_MARKER))
    assert report["holding_before_reset"] is True
    assert report["holding_after_reset"] is True


def test_reset_mid_hold_report_resets_the_world_between_holding_reads(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    commands: list[dict] = []

    class FakeWorld:
        async def do_command(self, command):
            commands.append(dict(command))
            return {"ok": True}

    class FakeStatus:
        def __init__(self, holding: bool) -> None:
            self.is_holding_something = holding
            self.meta = {"jaw_deg": 14.6}

    class FakeGripper:
        def __init__(self) -> None:
            self._answers = iter([True, False])

        async def open(self) -> None: ...

        async def grab(self) -> bool:
            return True

        async def is_holding_something(self):
            return FakeStatus(next(self._answers))

    report = asyncio.run(pick_red_block._reset_mid_hold_report(FakeWorld(), FakeGripper()))
    assert commands == [{"command": "reset"}]
    assert report["holding_before_reset"] is True
    assert report["holding_after_reset"] is False
    assert "diagnostics" not in report


def test_world_state_without_a_table_has_no_obstacle_geometries():
    state = pick_red_block.world_state(None)
    assert len(state.obstacles) == 1
    assert list(state.obstacles[0].geometries) == []


def test_world_state_with_the_table_carries_it():
    table = pick_red_block.table_obstacle()
    state = pick_red_block.world_state(table)
    assert list(state.obstacles[0].geometries) == [table]


def test_table_recipe_dropped_when_the_live_scene_serves_a_table():
    """P5 cell regression: the fragment's table prop arrives via
    prop_geometries, and the motion service rejects two WorldState geometries
    named 'table' - the --table recipe box must yield to the live one."""
    recipe = pick_red_block.table_obstacle()
    live_table = pick_red_block.table_obstacle()
    assert pick_red_block.table_recipe_unless_served(recipe, [live_table]) is None


def test_table_recipe_kept_when_the_live_scene_has_no_table():
    recipe = pick_red_block.table_obstacle()
    support = pick_red_block.support_obstacle(0.0)
    assert pick_red_block.table_recipe_unless_served(recipe, [support]) is recipe
    assert pick_red_block.table_recipe_unless_served(None, []) is None


def test_grasp_height_lifts_a_low_depth_centroid_off_the_support():
    # GPU run 13/19: detected z 17 mm for a 60 mm cube on the floor -> the
    # fingertip floor (19 mm overhang + 20 mm clearance) wins over the centre (30)
    assert pick_red_block.grasp_height_mm(17.0, 60.0, 0.0) == 39.0


def test_grasp_height_keeps_a_plausible_high_detection():
    assert pick_red_block.grasp_height_mm(45.0, 60.0, 0.0) == 45.0


def test_grasp_height_keeps_fingertips_off_the_support():
    # a thin block: centre floor 10, but the 19 mm overhang + 20 mm clearance wins
    assert pick_red_block.grasp_height_mm(10.0, 20.0, 0.0) == 39.0
    # on a 750 mm table top the same rules apply relative to the table
    assert pick_red_block.grasp_height_mm(760.0, 60.0, 750.0) == 789.0


def test_centre_depth_uses_only_points_near_the_optical_axis():
    import numpy as np

    points = np.array(
        [[0.0, 0.0, 0.35], [0.005, -0.004, 0.36], [0.2, 0.1, 0.5], [-0.3, 0.0, 0.9]],
        dtype=np.float32,
    )
    assert pick_red_block.centre_depth_mm(points) == pytest.approx(355.0)
    assert pick_red_block.centre_depth_mm(points[2:]) is None


def test_top_face_centre_ignores_floor_points_and_side_faces():
    """GPU runs 13-16: the segment box contains floor around the cube (deeper
    along the ray) and the cube's near sides; only the red nearest-depth band
    is the top face."""
    import numpy as np

    top = [[0.10 + dx, 0.05 + dy, 0.290] for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02)]
    sides = [[0.07, 0.05, 0.31], [0.07, 0.05, 0.33], [0.10, 0.02, 0.32]]  # red, deeper
    floor = [[0.14, 0.09, 0.350], [0.15, 0.10, 0.350], [0.16, 0.11, 0.350]]  # grey, deepest
    xyz = np.array(top + sides + floor, dtype=np.float32)
    rgb = np.array([[224, 32, 32]] * (len(top) + len(sides)) + [[120, 120, 120]] * len(floor))
    centre = pick_red_block.top_face_centre_m(xyz, rgb)
    assert centre == pytest.approx((0.10, 0.05, 0.290), abs=1e-6)
    # a plain centroid would have been dragged toward the floor points
    assert xyz.mean(axis=0)[2] > 0.30


def test_top_face_centre_falls_back_to_all_points_without_colour():
    import numpy as np

    xyz = np.array([[0.0, 0.0, 0.30], [0.01, 0.0, 0.301], [0.0, 0.0, 0.40]], dtype=np.float32)
    # the 2 mm band keeps the 1 mm-deeper point and drops the 100 mm-deeper one
    assert pick_red_block.top_face_centre_m(xyz, None) == pytest.approx(
        (0.005, 0.0, 0.3005), abs=1e-6
    )


def test_support_obstacle_top_is_at_the_support_height():
    slab = pick_red_block.support_obstacle(750.0)
    assert slab.center.z + slab.box.dims_mm.z / 2 == pytest.approx(750.0)
    assert slab.label == "support"


def test_world_state_puts_the_support_first_and_skips_none():
    slab = pick_red_block.support_obstacle(0.0)
    state = pick_red_block.world_state(None, support=slab)
    assert list(state.obstacles[0].geometries) == [slab]


def test_segment_stats_counts_red_and_band_points_and_red_extents():
    import numpy as np

    xyz = np.array(
        [[0.0, 0.0, 0.29], [0.06, 0.0, 0.29], [0.03, 0.0, 0.31], [0.1, 0.1, 0.35]],
        dtype=np.float32,
    )
    rgb = np.array([[224, 32, 32], [224, 32, 32], [224, 32, 32], [120, 120, 120]])
    stats = pick_red_block.segment_stats(xyz, rgb)
    assert (stats["points"], stats["red"], stats["band"]) == (4, 3, 2)
    assert stats["red_min_mm"] == [0.0, 0.0, 290.0]
    assert stats["red_max_mm"] == [60.0, 0.0, 310.0]
    assert stats["band_min_mm"] == [0.0, 0.0, 290.0]
    assert stats["band_max_mm"] == [60.0, 0.0, 290.0]


def test_top_face_is_the_nearest_surface_even_when_it_is_too_bright_to_read_as_red():
    """GPU run 21: the lit top face failed the red test and only a vertical face
    passed it, so a red-first search picked that face's upper edge."""
    import numpy as np

    top = [[0.10 + dx, 0.05 + dy, 0.290] for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02)]
    near_face = [[0.07, 0.05 + dy, 0.29 + dz] for dy in (-0.02, 0.0, 0.02) for dz in (0.0, 0.03)]
    floor = [[0.15, 0.10, 0.350], [0.16, 0.11, 0.350]]
    xyz = np.array(top + near_face + floor, dtype=np.float32)
    rgb = np.array(
        [[255, 170, 170]] * len(top) + [[224, 32, 32]] * len(near_face) + [[120, 120, 120]] * 2
    )
    centre = pick_red_block.top_face_centre_m(xyz, rgb)
    # 9 washed-out top points + 3 red near-face top-edge points in the band: red is
    # only 25% of the band, so the whole band (top face + strip) is used
    assert centre[0] == pytest.approx((9 * 0.10 + 3 * 0.07) / 12, abs=1e-6)
    assert centre[2] == pytest.approx(0.290, abs=1e-6)


def test_top_face_ignores_the_gripper_fingers_in_front_of_the_camera():
    import numpy as np

    fingers = [[0.0, 0.03, 0.09], [0.0, -0.03, 0.09]]
    top = [[0.10, 0.05, 0.29], [0.12, 0.05, 0.29]]
    xyz = np.array(fingers + top, dtype=np.float32)
    assert pick_red_block.top_face_centre_m(xyz, None) == pytest.approx((0.11, 0.05, 0.29))


def test_corrected_pose_shifts_by_the_measured_delta_and_keeps_orientation():
    pose = pick_red_block.Pose(x=700.0, y=250.0, z=39.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    out = pick_red_block.corrected_pose(pose, (1.0, -0.5, 17.0))
    assert (out.x, out.y, out.z) == (701.0, 249.5, 56.0)
    assert (out.o_z, out.theta) == (-1.0, 0.0)


def test_corrected_pose_caps_a_wild_reading():
    pose = pick_red_block.Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    out = pick_red_block.corrected_pose(pose, (500.0, -500.0, 41.0))
    assert (out.x, out.y, out.z) == (40.0, -40.0, 40.0)


def _prop_geometry(
    name: str,
    box_dims_mm: list[float],
    pose_in_world_mm: dict[str, float] | None = None,
) -> dict:
    return {
        "name": name,
        "box_dims_mm": box_dims_mm,
        "pose_in_world_mm": pose_in_world_mm
        or {"x": 100.0, "y": 200.0, "z": 30.0, "o_x": 0.0, "o_y": 0.0, "o_z": 1.0, "theta": 0.0},
        "color": [0.9, 0.1, 0.1],
        "fixed": False,
    }


def test_obstacles_from_prop_geometries_carries_dims_and_orientation():
    geometries = [
        _prop_geometry(
            "place_pad",
            [200.0, 200.0, 10.0],
            {"x": 700.0, "y": -350.0, "z": 5.0, "o_x": 0.1, "o_y": 0.2, "o_z": 0.9, "theta": 30.0},
        )
    ]
    obstacles = pick_red_block.obstacles_from_prop_geometries(geometries, exclude=set())
    assert len(obstacles) == 1
    obstacle = obstacles[0]
    assert obstacle.label == "place_pad"
    box = obstacle.box
    assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (200.0, 200.0, 10.0)
    center = obstacle.center
    assert (center.x, center.y, center.z) == (700.0, -350.0, 5.0)
    assert (center.o_x, center.o_y, center.o_z, center.theta) == (0.1, 0.2, 0.9, 30.0)


def test_obstacles_from_prop_geometries_excludes_named_props():
    geometries = [_prop_geometry("pick_cube", [60.0, 60.0, 60.0])]
    obstacles = pick_red_block.obstacles_from_prop_geometries(geometries, exclude={"pick_cube"})
    assert obstacles == []


def test_obstacles_from_prop_geometries_skips_zero_dim_entries():
    geometries = [_prop_geometry("unknown_usd_prop", [0.0, 0.0, 0.0])]
    obstacles = pick_red_block.obstacles_from_prop_geometries(geometries, exclude=set())
    assert obstacles == []


def test_randomize_region_mm_is_the_table_footprint_at_the_support_height():
    (x0, y0, z0), (x1, y1, z1) = pick_red_block.randomize_region_mm(margin_mm=50.0)
    assert (x0, y0, z0) == (-1750.0, -350.0, 0.0)
    assert (x1, y1, z1) == (-650.0, 350.0, 0.0)
    (_, _, top0), (_, _, top1) = pick_red_block.randomize_region_mm(margin_mm=50.0, face_z_mm=750.0)
    assert (top0, top1) == (750.0, 750.0)


def test_sim_pipeline_randomizes_then_fetches_geometries_in_order(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    commands: list[dict] = []

    class FakeWorld:
        async def do_command(self, command):
            commands.append(dict(command))
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [100.0, 200.0, 30.0]}}
            return {
                "geometries": [
                    _prop_geometry("pick_cube", [60.0, 60.0, 60.0]),
                    _prop_geometry("place_pad", [200.0, 200.0, 10.0]),
                ]
            }

    class FakeDetector:
        async def block_pose_world(self):
            return Pose(x=100.0, y=200.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class FakeMover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            return None

    class FakeGripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            raise AssertionError("not exercised in this test")

    pipeline = pick_red_block.PickPipeline(
        detector=FakeDetector(),
        mover=FakeMover(),
        gripper=FakeGripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=FakeWorld(),
        target_prop_name="pick_cube",
        movable_prop_names=["pick_cube"],
        randomize_seed=7,
    )
    asyncio.run(pipeline.run())

    assert [command["command"] for command in commands] == [
        "ignore_props",  # the pick run ignores its target for route-(c) planners
        "randomize_props",
        "prop_geometries",
        "ignore_props",  # the finally clears it
    ]
    randomize_command = commands[1]  # commands[0] is the route-(c) ignore_props
    assert randomize_command["names"] == ["pick_cube"]
    assert randomize_command["seed"] == 7
    # default region: the reach-safe rectangle at the pipeline's support_z_mm (floor)
    assert randomize_command["region"] == [[-1350.0, -300.0, 0.0], [-700.0, 300.0, 0.0]]


def test_sim_obstacles_excludes_the_target_and_includes_other_props(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)

    class FakeWorld:
        async def do_command(self, command):
            if command["command"] == "ignore_props":
                return {"ignored": command["names"]}
            assert command == {"command": "prop_geometries"}
            return {
                "geometries": [
                    _prop_geometry("pick_cube", [60.0, 60.0, 60.0]),
                    _prop_geometry("place_pad", [200.0, 200.0, 10.0]),
                ]
            }

    class FakeDetector:
        async def block_pose_world(self):
            return Pose(x=100.0, y=200.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    move_world_states = []

    class RecordingMover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            move_world_states.append(world_state)

    class FakeGripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            raise AssertionError("not exercised in this test")

    pipeline = pick_red_block.PickPipeline(
        detector=FakeDetector(),
        mover=RecordingMover(),
        gripper=FakeGripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=FakeWorld(),
        target_prop_name="pick_cube",
    )
    asyncio.run(pipeline.run())

    labels = {g.label for g in move_world_states[0].obstacles[0].geometries}
    assert "place_pad" in labels
    assert "pick_cube" not in labels


def test_reachable_region_mm_stays_inside_the_arm_envelope():
    lo, hi = pick_red_block.reachable_region_mm(face_z_mm=5.0)
    assert lo == [-1350.0, -300.0, 5.0]
    assert hi == [-700.0, 300.0, 5.0]
    # no corner beyond 80% of the ur20's 1750 mm reach (cell_layout's
    # MAX_PLANAR_REACH_MM), arm base at the origin
    max_planar_reach_mm = 1750.0 * 0.8
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            assert (x**2 + y**2) ** 0.5 <= max_planar_reach_mm + 1e-9


def test_scan_then_focus_uses_the_detector_not_sim_truth(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    look_poses = []

    class FakeWorld:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [123.0, -45.0, 30.0]}}
            return {"geometries": []}

    class FakeDetector:
        async def block_pose_world(self):
            return Pose(x=123.0, y=-45.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class FakeMover:
        async def look_from(self, pose, world_state, linear=False):
            look_poses.append(pose)

        async def move_to(self, pose, world_state, linear=False):
            return None

    class FakeGripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=FakeDetector(),
        mover=FakeMover(),
        gripper=FakeGripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=FakeWorld(),
        target_prop_name="pick_cube",
        movable_prop_names=("pick_cube",),
        randomize_seed=3,
        look_pose=pick_red_block._pointing_down(500.0, 150.0, 350.0),
    )
    asyncio.run(pipeline.run())

    # scan from the scatter-region centre, never from the sim's ground truth
    scan = look_poses[0]
    assert (scan.x, scan.y, scan.z) == (-1025.0, 0.0, 350.0)
    # then focus above where the DETECTOR said the block is
    focus = look_poses[1]
    assert (focus.x, focus.y, focus.z) == (123.0, -45.0, 350.0)
    assert len(look_poses) == 2


def test_pad_top_centre_mm_finds_the_pad_top():
    geometries = [
        _prop_geometry("pick_cube", [60.0, 60.0, 60.0]),
        {
            "name": "place_pad",
            "fixed": True,
            "box_dims_mm": [200.0, 200.0, 10.0],
            "pose_in_world_mm": {
                "x": 700.0,
                "y": -350.0,
                "z": 5.0,
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": 1.0,
                "theta": 0.0,
            },
        },
    ]
    assert pick_red_block.pad_top_centre_mm(geometries, "place_pad") == (700.0, -350.0, 10.0)
    assert pick_red_block.pad_top_centre_mm(geometries, "missing") is None


def _pad_geometry(x_mm=700.0, y_mm=-350.0, z_mm=5.0):
    return {
        "name": "place_pad",
        "fixed": True,
        "box_dims_mm": [200.0, 200.0, 10.0],
        "pose_in_world_mm": {
            "x": x_mm,
            "y": y_mm,
            "z": z_mm,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
    }


def _block_geometry(x_mm, y_mm, z_mm):
    geometry = _prop_geometry("pick_cube", [60.0, 60.0, 60.0])
    geometry["pose_in_world_mm"] = {
        "x": x_mm,
        "y": y_mm,
        "z": z_mm,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 1.0,
        "theta": 0.0,
    }
    return geometry


class _PlaceFakes:
    def __init__(self, with_pad: bool):
        self.commands: list[dict] = []
        self.moves: list[tuple] = []
        self.move_world_states: list = []
        self.opens = 0
        self.with_pad = with_pad
        outer = self

        class World:
            async def do_command(self, command):
                outer.commands.append(dict(command))
                geometries = [_block_geometry(600.0, 100.0, 40.0)]
                if outer.with_pad:
                    geometries.append(_pad_geometry())
                return {"geometries": geometries}

        class Detector:
            async def block_pose_world(self):
                return Pose(x=600.0, y=100.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

        class Mover:
            async def look_from(self, pose, world_state, linear=False):
                return None

            async def move_to(self, pose, world_state, linear=False):
                outer.moves.append((pose, linear))
                outer.move_world_states.append(world_state)

        class Gripper:
            async def open(self):
                outer.opens += 1

            async def grab(self):
                return True

            async def is_holding_something(self):
                return True

        self.pipeline = pick_red_block.PickPipeline(
            detector=Detector(),
            mover=Mover(),
            gripper=Gripper(),
            block_name="pick_cube",
            block_size_mm=60.0,
            gripper_name="pick-grip",
            world=World(),
            target_prop_name="pick_cube",
            place_prop_name="place_pad",
        )


def test_place_step_descends_over_the_pad_and_reports_placement(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    fakes = _PlaceFakes(with_pad=True)
    asyncio.run(fakes.pipeline.run())

    # waypoint, pre-grasp, grasp, lift, raise, carry, place, retreat
    assert len(fakes.moves) == 8
    grasp_pose_used = fakes.moves[2][0]
    raise_move, carry, place, retreat = (m[0] for m in fakes.moves[4:8])
    # the carry runs at constant height on a straight (linear) line
    assert (raise_move.x, raise_move.y) == (grasp_pose_used.x, grasp_pose_used.y)
    assert raise_move.z == carry.z
    assert fakes.moves[4][1] is True and fakes.moves[5][1] is True
    assert (carry.x, carry.y) == (700.0, -350.0)
    # place z reproduces the grasp offset over the pad top (10) plus the gap (15)
    assert place.z == pytest.approx(grasp_pose_used.z + 10.0 + 15.0)
    assert fakes.moves[6][1] is True  # the descent is linear
    assert retreat.z == carry.z
    # open at the start plus open over the pad - and no release at the lift pose
    assert fakes.opens == 2
    # the placement verdict re-reads prop_geometries at the end
    assert fakes.commands[-2]["command"] == "prop_geometries"
    assert fakes.commands[-1]["command"] == "ignore_props"  # finally: clear the pick ignore


def test_place_skipped_when_the_scene_has_no_pad(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    fakes = _PlaceFakes(with_pad=False)
    asyncio.run(fakes.pipeline.run())

    # waypoint, pre-grasp, grasp, lift only - and the release open still happens
    assert len(fakes.moves) == 4  # no carry moves without a pad
    assert fakes.opens == 2


class _FailingMoverFakes(_PlaceFakes):
    """Place fakes whose mover refuses configurable descents over the pad
    (y == -350 and z below a threshold), like a planner at its reach limit."""

    def __init__(self, fail_linear_below=None, fail_any_below=None):
        super().__init__(with_pad=True)
        outer = self

        class Mover:
            async def look_from(self, pose, world_state, linear=False):
                return None

            async def move_to(self, pose, world_state, linear=False):
                over_pad = pose.y == -350.0
                if over_pad and fail_any_below is not None and pose.z < fail_any_below:
                    raise RuntimeError("motion planner failed to find path")
                if (
                    over_pad
                    and linear
                    and fail_linear_below is not None
                    and pose.z < fail_linear_below
                ):
                    raise RuntimeError("motion planner failed to find path")
                outer.moves.append((pose, linear))

        self.pipeline.mover = Mover()


def test_place_falls_back_to_a_planned_descent(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    fakes = _FailingMoverFakes(fail_linear_below=100.0)
    asyncio.run(fakes.pipeline.run())

    # pre-grasp, grasp, lift, pre-place, planned place, retreat (linear retreat
    # is above the threshold, so it lands)
    place_moves = [(p, linear) for p, linear in fakes.moves if p.y == -350.0 and p.z < 100.0]
    assert len(place_moves) == 1
    assert place_moves[0][1] is False  # the planned (non-linear) descent released
    assert fakes.opens == 2
    assert fakes.commands[-2]["command"] == "prop_geometries"
    assert fakes.commands[-1]["command"] == "ignore_props"  # finally: clear the pick ignore


def test_place_releases_from_hover_when_no_descent_plans(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    fakes = _FailingMoverFakes(fail_any_below=150.0)
    asyncio.run(fakes.pipeline.run())

    # every descent refused: released from the hover pose, no retreat needed
    assert all(p.z >= 150.0 for p, _linear in fakes.moves if p.y == -350.0)
    assert fakes.opens == 2
    assert fakes.commands[-2]["command"] == "prop_geometries"
    assert fakes.commands[-1]["command"] == "ignore_props"  # finally: clear the pick ignore


def _scene_entry(name, fixed, dims_mm, x_mm=0.0, y_mm=0.0, z_mm=30.0):
    return {
        "name": name,
        "fixed": fixed,
        "box_dims_mm": list(dims_mm),
        "pose_in_world_mm": {
            "x": x_mm,
            "y": y_mm,
            "z": z_mm,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
    }


class _DeriveFakes:
    """Full pipeline fakes whose world reports movables, a fixed pad, and an
    unknown-size usd prop, for the derive-movables randomize path."""

    def __init__(self, geometries):
        self.commands: list[dict] = []
        outer = self

        class World:
            async def do_command(self, command):
                outer.commands.append(dict(command))
                if command["command"] == "randomize_props":
                    return {"positions": {name: [500.0, 0.0, 30.0] for name in command["names"]}}
                return {"geometries": geometries}

        class Detector:
            async def block_pose_world(self):
                return Pose(x=500.0, y=0.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

        class Mover:
            async def look_from(self, pose, world_state, linear=False):
                return None

            async def move_to(self, pose, world_state, linear=False):
                return None

        class Gripper:
            async def open(self):
                return None

            async def grab(self):
                return True

            async def is_holding_something(self):
                return True

        self.pipeline = pick_red_block.PickPipeline(
            detector=Detector(),
            mover=Mover(),
            gripper=Gripper(),
            block_name="pick_cube",
            block_size_mm=60.0,
            gripper_name="pick-grip",
            world=World(),
            target_prop_name="pick_cube",
            randomize_seed=5,
        )


def test_randomize_derives_movable_names_from_the_scene(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    fakes = _DeriveFakes(
        [
            _scene_entry("pick_cube", False, [60.0, 60.0, 60.0], 700.0, 250.0),
            _scene_entry("ignore_cube_green", False, [60.0, 60.0, 60.0], 550.0, -50.0),
            _scene_entry("place_pad", True, [200.0, 200.0, 10.0], 700.0, -350.0, 5.0),
            _scene_entry("mystery_usd", False, [0.0, 0.0, 0.0]),
        ]
    )
    asyncio.run(fakes.pipeline.run())

    randomize = next(c for c in fakes.commands if c["command"] == "randomize_props")
    assert randomize["names"] == ["pick_cube", "ignore_cube_green"]
    assert randomize["min_separation"] == 140.0  # six-block packing envelope


def test_randomize_skipped_when_the_scene_has_no_movables(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    fakes = _DeriveFakes([_scene_entry("place_pad", True, [200.0, 200.0, 10.0])])
    asyncio.run(fakes.pipeline.run())

    assert all(c["command"] != "randomize_props" for c in fakes.commands)


def test_held_block_rides_in_world_state_transforms_while_grasping(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    fakes = _PlaceFakes(with_pad=True)
    asyncio.run(fakes.pipeline.run())

    # waypoint, pre-grasp, grasp, lift, raise, carry, place, retreat
    states = fakes.move_world_states
    assert len(states) == 8
    assert list(states[0].transforms) == []
    assert list(states[1].transforms) == []
    assert list(states[2].transforms) == []
    for held in states[3:7]:
        (transform,) = held.transforms
        assert transform.reference_frame == "pick_cube"
        assert transform.pose_in_observer_frame.reference_frame == "pick-grip"
        # detected centre z 30, grasp TCP 39: the box hangs 9 mm below the TCP
        assert transform.pose_in_observer_frame.pose.z == pytest.approx(-9.0)
        # 60 mm block + 20 mm planning padding (ARM-10 execution error)
        assert transform.physical_object.box.dims_mm.x == 80.0
    assert list(states[7].transforms) == []  # released before the retreat


class _ShadowFakes:
    """Pipeline fakes whose detector serves poses from a queue, for the
    gripper-shadow re-scan path."""

    def __init__(self, detections):
        self.look_poses = []
        queue = list(detections)
        outer = self

        class World:
            async def do_command(self, command):
                if command["command"] == "randomize_props":
                    return {"positions": {"pick_cube": [123.0, -45.0, 30.0]}}
                return {"geometries": []}

        class Detector:
            async def block_pose_world(self):
                return queue.pop(0)

        class Mover:
            async def look_from(self, pose, world_state, linear=False):
                outer.look_poses.append(pose)

            async def move_to(self, pose, world_state, linear=False):
                return None

        class Gripper:
            async def open(self):
                return None

            async def grab(self):
                return True

            async def is_holding_something(self):
                return True

        self.pipeline = pick_red_block.PickPipeline(
            detector=Detector(),
            mover=Mover(),
            gripper=Gripper(),
            block_name="pick_cube",
            block_size_mm=60.0,
            gripper_name="pick-grip",
            world=World(),
            target_prop_name="pick_cube",
            movable_prop_names=("pick_cube",),
            randomize_seed=3,
            look_pose=pick_red_block._pointing_down(500.0, 150.0, 350.0),
        )


def _detection(x_mm, y_mm, z_mm):
    return Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)


def test_detection_rescans_when_the_gripper_shadows_the_block(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    fakes = _ShadowFakes(
        [
            _detection(504.0, 34.0, 115.0),  # shadowed: impossible height, no focus chase
            _detection(123.0, -45.0, 30.0),  # quarter-turn re-scan sees the block
            _detection(123.0, -45.0, 30.0),  # focused measurement
        ]
    )
    asyncio.run(fakes.pipeline.run())

    scan, rescan, focus = fakes.look_poses[:3]
    assert (scan.x, scan.y, scan.theta) == (-1025.0, 0.0, 0.0)
    # same spot, wrist turned a quarter: the shadow sweeps off the block
    assert (rescan.x, rescan.y, rescan.theta) == (-1025.0, 0.0, 90.0)
    assert (focus.x, focus.y, focus.theta) == (123.0, -45.0, 90.0)
    assert len(fakes.look_poses) == 3


def test_detection_raises_when_every_scan_pose_is_shadowed(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    fakes = _ShadowFakes([_detection(504.0, 34.0, 115.0)] * 6)
    with pytest.raises(RuntimeError, match="plausible"):
        asyncio.run(fakes.pipeline.run())
    # every rung of the ladder ran: four wrist angles, then two side-steps
    assert len(fakes.look_poses) == 6


def test_pick_area_keepout_boxes_the_region_with_margin():
    keepout = pick_red_block.pick_area_keepout(([-1350.0, -300.0, 0.0], [-700.0, 300.0, 0.0]))
    assert keepout.label == "pick_area_keepout"
    assert (keepout.center.x, keepout.center.y, keepout.center.z) == (-1025.0, 0.0, 65.0)
    dims = keepout.box.dims_mm
    assert (dims.x, dims.y, dims.z) == (750.0, 700.0, 130.0)


def test_pick_area_keepout_stands_on_the_region_support():
    """P5 cell regression (GPU run 13): the scatter region's z is the table
    top, and the no-fly box must cover the block airspace ABOVE it, not the
    floor's."""
    keepout = pick_red_block.pick_area_keepout(([-1350.0, -300.0, 750.0], [-700.0, 300.0, 750.0]))
    assert keepout.center.z == 750.0 + 65.0
    assert keepout.box.dims_mm.z == 130.0


def test_default_scan_pose_rides_the_support_height():
    """GPU run 13: an absolute 350 mm scan z sent the camera 400 mm below the
    P5 table top - the default scan height is above the support."""
    floor = pick_red_block.default_scan_pose(0.0)
    table = pick_red_block.default_scan_pose(750.0)
    assert (floor.x, floor.y, floor.z) == (-1025.0, 0.0, 650.0)
    assert (table.x, table.y, table.z) == (-1025.0, 0.0, 1400.0)
    assert table.o_z == pick_red_block.POINTING_DOWN_O_Z


def test_randomized_carry_plans_freely_over_the_keepout(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    moves = []

    class World:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [520.0, 20.0, 30.0]}}
            return {
                "geometries": [
                    _block_geometry(520.0, 20.0, 30.0),
                    _pad_geometry(),
                ]
            }

    class Detector:
        async def block_pose_world(self):
            return Pose(x=520.0, y=20.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            moves.append((pose, world_state, linear))

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=World(),
        target_prop_name="pick_cube",
        movable_prop_names=("pick_cube",),
        randomize_seed=4,
        place_prop_name="place_pad",
    )
    asyncio.run(pipeline.run())

    # waypoint, pre-grasp, grasp, lift, raise-above-keepout, free carry, place, retreat
    assert len(moves) == 8
    clear_pose, clear_state, clear_linear = moves[4]
    assert clear_pose.z == 200.0  # support 0 + CARRY_CLEAR_ABOVE_SUPPORT_MM
    assert clear_linear is True
    labels = {g.label for frame in clear_state.obstacles for g in frame.geometries}
    assert "pick_area_keepout" not in labels  # the hop starts inside the box

    carry_pose, carry_state, carry_linear = moves[5]
    assert (carry_pose.x, carry_pose.y) == (700.0, -350.0)
    assert carry_linear is False  # free, fast plan
    carry_labels = {g.label for frame in carry_state.obstacles for g in frame.geometries}
    assert "pick_area_keepout" in carry_labels
    (transform,) = carry_state.transforms  # the held block still rides along
    assert transform.reference_frame == "pick_cube"


def _footprint_points(
    x_mm: float, y_mm: float, n: int = 500, seed: int = 0, top_depth_m: float = 0.29
):
    """Synthetic segment (camera frame, metres) the band estimator sees: a
    top face of n points at ``top_depth_m`` spread over an x_mm by y_mm
    footprint, a floor patch 60 mm deeper and twice as wide (the band must
    exclude it), and two in-band outliers past the 2nd/98th percentile trim
    so the trim's effect is exercised."""
    rng = np.random.default_rng(seed)
    half_x_m, half_y_m = x_mm / 2000.0, y_mm / 2000.0
    xy = np.column_stack([rng.uniform(-half_x_m, half_x_m, n), rng.uniform(-half_y_m, half_y_m, n)])
    z = np.full((n, 1), top_depth_m, dtype=np.float32)
    top = np.column_stack([xy, z]).astype(np.float32)
    floor_xy = np.column_stack(
        [rng.uniform(-2 * half_x_m, 2 * half_x_m, n), rng.uniform(-2 * half_y_m, 2 * half_y_m, n)]
    )
    floor = np.column_stack([floor_xy, np.full((n, 1), top_depth_m + 0.06)]).astype(np.float32)
    outliers = np.array(
        [
            [half_x_m * 10, half_y_m * 10, top_depth_m],
            [-half_x_m * 10, -half_y_m * 10, top_depth_m],
        ],
        dtype=np.float32,
    )
    return np.vstack([top, outliers, floor])


def test_footprint_extents_mm_trims_outliers_at_40mm():
    footprint = footprint_extents_mm(_footprint_points(40.0, 40.0))
    assert footprint is not None
    assert footprint[0] == pytest.approx(40.0, rel=0.15)
    assert footprint[1] == pytest.approx(40.0, rel=0.15)


def test_footprint_extents_mm_trims_outliers_at_75mm():
    footprint = footprint_extents_mm(_footprint_points(75.0, 60.0))
    assert footprint is not None
    assert footprint[0] == pytest.approx(75.0, rel=0.15)
    assert footprint[1] == pytest.approx(60.0, rel=0.15)


def test_footprint_extents_mm_ignores_the_deeper_floor_band():
    # the floor patch is twice the block's width; measuring it would double
    # the extents, so a correct band selection is what keeps this ~40 mm
    footprint = footprint_extents_mm(_footprint_points(40.0, 40.0))
    assert footprint is not None
    assert footprint[0] < 60.0 and footprint[1] < 60.0


def test_footprint_extents_mm_none_when_only_gripper_depth_points():
    xyz = np.array([[0.0, 0.0, 0.05], [0.01, 0.0, 0.06]], dtype=np.float32)
    assert footprint_extents_mm(xyz) is None


def test_measured_block_size_mm_takes_the_median_at_40mm():
    result = measured_block_size_mm([39.0, 41.0, 40.5])
    assert result is not None
    size_mm, estimates = result
    assert size_mm == pytest.approx(40.5)
    assert estimates == [39.0, 41.0, 40.5]


def test_measured_block_size_mm_takes_the_median_at_75mm():
    result = measured_block_size_mm([74.0, 76.0, 75.5])
    assert result is not None
    size_mm, _ = result
    assert size_mm == pytest.approx(75.5)


def test_measured_block_size_mm_flags_a_degenerate_view():
    # height reads 60 against a ~40 mm footprint - a 50% spread, over the 25% cross-check
    assert measured_block_size_mm([40.0, 40.0, 60.0]) is None


def test_parse_randomize_size_mm_accepts_lo_hi():
    assert pick_red_block._parse_randomize_size_mm("30,90") == (30.0, 90.0)


def test_parse_randomize_size_mm_rejects_inverted_range():
    with pytest.raises(argparse.ArgumentTypeError):
        pick_red_block._parse_randomize_size_mm("90,30")


def test_parse_randomize_size_mm_rejects_non_positive():
    with pytest.raises(argparse.ArgumentTypeError):
        pick_red_block._parse_randomize_size_mm("0,90")


def test_randomize_props_payload_omits_size_range_by_default(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    commands: list[dict] = []

    class FakeWorld:
        async def do_command(self, command):
            commands.append(dict(command))
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [100.0, 200.0, 30.0]}}
            return {"geometries": [_prop_geometry("pick_cube", [60.0, 60.0, 60.0])]}

    class FakeDetector:
        async def block_pose_world(self):
            return Pose(x=100.0, y=200.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class FakeMover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            return None

    class FakeGripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=FakeDetector(),
        mover=FakeMover(),
        gripper=FakeGripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=FakeWorld(),
        target_prop_name="pick_cube",
        movable_prop_names=["pick_cube"],
        randomize_seed=7,
    )
    asyncio.run(pipeline.run())

    randomize_command = next(c for c in commands if c["command"] == "randomize_props")
    assert "size_range_mm" not in randomize_command


def test_randomize_props_payload_carries_size_range_mm_when_set(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    commands: list[dict] = []

    class FakeWorld:
        async def do_command(self, command):
            commands.append(dict(command))
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [100.0, 200.0, 30.0]}}
            return {"geometries": [_prop_geometry("pick_cube", [60.0, 60.0, 60.0])]}

    class FakeDetector:
        async def block_pose_world(self):
            return Pose(x=100.0, y=200.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class FakeMover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            return None

    class FakeGripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=FakeDetector(),
        mover=FakeMover(),
        gripper=FakeGripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=FakeWorld(),
        target_prop_name="pick_cube",
        movable_prop_names=["pick_cube"],
        randomize_seed=7,
        randomize_size_range_mm=(30.0, 90.0),
    )
    asyncio.run(pipeline.run())

    randomize_command = next(c for c in commands if c["command"] == "randomize_props")
    assert randomize_command["size_range_mm"] == [30.0, 90.0]


def test_jaw_refusal_skips_the_grasp_and_leaves_the_arm_parked(capsys):
    class FakeMeasuredDetector:
        async def block_pose_world(self):
            return Pose(x=100.0, y=200.0, z=40.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

        def last_measurement(self):
            return {"footprint_mm": [80.0, 80.0], "height_mm": 80.0, "size_mm": 80.0}

    class RefusingMover:
        async def look_from(self, pose, world_state, linear=False):
            raise AssertionError("the arm must stay parked on a jaw refusal")

        async def move_to(self, pose, world_state, linear=False):
            raise AssertionError("the arm must stay parked on a jaw refusal")

    class RefusingGripper:
        async def open(self):
            raise AssertionError("no grasp should be attempted on a jaw refusal")

        async def grab(self):
            raise AssertionError("no grasp should be attempted on a jaw refusal")

        async def is_holding_something(self):
            raise AssertionError("not exercised in this test")

    pipeline = pick_red_block.PickPipeline(
        detector=FakeMeasuredDetector(),
        mover=RefusingMover(),
        gripper=RefusingGripper(),
        block_name="pick_cube",
        block_size_mm=None,
        gripper_name="pick-grip",
        verify_detection_height=False,
    )

    with pytest.raises(RuntimeError, match="jaw"):
        asyncio.run(pipeline.run())
    assert f"{JAW_MAX_BLOCK_MM:.0f} mm jaw" in capsys.readouterr().out


class _MeasurementFakes:
    """Pipeline fakes whose detector serves (pose, measurement) pairs from a
    queue, for the degenerate-measurement re-scan path."""

    def __init__(self, readings):
        self.look_poses = []
        queue = list(readings)
        outer = self

        class World:
            async def do_command(self, command):
                if command["command"] == "randomize_props":
                    return {"positions": {"pick_cube": [-1020.0, 5.0, 20.0]}}
                return {"geometries": []}

        class Detector:
            def __init__(self):
                self._measurement = None

            async def block_pose_world(self):
                pose, measurement = queue.pop(0)
                self._measurement = measurement
                return pose

            def last_measurement(self):
                return self._measurement

        class Mover:
            async def look_from(self, pose, world_state, linear=False):
                outer.look_poses.append(pose)

            async def move_to(self, pose, world_state, linear=False):
                return None

        class Gripper:
            async def open(self):
                return None

            async def grab(self):
                return True

            async def is_holding_something(self):
                return True

        self.pipeline = pick_red_block.PickPipeline(
            detector=Detector(),
            mover=Mover(),
            gripper=Gripper(),
            block_name="pick_cube",
            block_size_mm=None,
            gripper_name="pick-grip",
            world=World(),
            target_prop_name="pick_cube",
            movable_prop_names=("pick_cube",),
            randomize_seed=3,
            look_pose=pick_red_block._pointing_down(500.0, 150.0, 350.0),
        )


def test_detection_retries_on_a_degenerate_size_measurement(monkeypatch, capsys):
    """A degenerate measurement that stays degenerate through the one-shot
    focus (checklist item 3) still falls back to the SCAN_ATTEMPTS ladder."""
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    degenerate_pose = Pose(x=-1020.0, y=5.0, z=999.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    accepted_pose = Pose(x=-1025.0, y=0.0, z=20.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    fakes = _MeasurementFakes(
        [
            (degenerate_pose, None),  # initial: footprint/height disagree - focus once
            (degenerate_pose, None),  # focused re-check: still degenerate - re-scan
            (
                accepted_pose,
                {"footprint_mm": [40.0, 40.0], "height_mm": 40.0, "size_mm": 40.0},
            ),
        ]
    )

    transform = asyncio.run(fakes.pipeline.run())

    # scan, focus (still degenerate), next SCAN_ATTEMPTS variant (accepted)
    assert len(fakes.look_poses) == 3
    assert transform.physical_object.box.dims_mm.x == 40.0
    out = capsys.readouterr().out
    assert "degenerate size measurement" in out
    measured_line = next(
        line for line in out.splitlines() if line.startswith(MEASURED_BLOCK_MARKER)
    )
    measured = json.loads(measured_line.removeprefix(MEASURED_BLOCK_MARKER))
    assert measured["size_mm"] == 40.0


def test_detection_focuses_once_on_a_degenerate_initial_measurement(monkeypatch, capsys):
    """checklist item 3 (GPU run 20): the wide initial scan degrades the size
    measurement but the pose is already usable, so the pipeline focuses
    directly above it instead of walking the SCAN_ATTEMPTS offset ladder."""
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    degenerate_pose = Pose(x=-1210.0, y=187.0, z=999.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    accepted_pose = Pose(x=-1210.0, y=187.0, z=20.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    fakes = _MeasurementFakes(
        [
            (degenerate_pose, None),  # initial: good pose, degenerate size
            (
                accepted_pose,
                {"footprint_mm": [40.0, 40.0], "height_mm": 40.0, "size_mm": 40.0},
            ),
        ]
    )

    transform = asyncio.run(fakes.pipeline.run())

    assert len(fakes.look_poses) == 2  # scan, then one focus - no offset-variant walk
    focus = fakes.look_poses[1]
    assert (focus.x, focus.y, focus.z) == (-1210.0, 187.0, 350.0)  # support (0) + FOCUS_HEIGHT
    assert transform.physical_object.box.dims_mm.x == 40.0
    out = capsys.readouterr().out
    assert "degenerate size measurement" not in out


def _contains_keepout(world_state) -> bool:
    return any(
        geometry.label == "pick_area_keepout"
        for frame_geometries in world_state.obstacles
        for geometry in frame_geometries.geometries
    )


def test_pre_grasp_move_is_linear_and_skips_the_waypoint_after_a_focus(monkeypatch):
    """checklist item 3: the block is excluded from the planner's obstacles,
    so looks plan freely only when the pick-area keep-out boxes the blocks'
    airspace off, and the descent below the waypoint is linear. When detection
    already ended at the waypoint (the common, focused-detection path), no
    extra move runs first."""
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    look_calls: list[tuple[Pose, bool, object]] = []
    moves: list[tuple[Pose, bool, object]] = []
    readings = iter(
        [
            (Pose(x=-1210.0, y=187.0, z=999.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0), None),
            (
                Pose(x=-1210.0, y=187.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
                {"footprint_mm": [60.0, 60.0], "height_mm": 60.0, "size_mm": 60.0},
            ),
        ]
    )

    class World:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [-1210.0, 187.0, 30.0]}}
            return {"geometries": []}

    class Detector:
        def __init__(self):
            self._measurement = None

        async def block_pose_world(self):
            pose, measurement = next(readings)
            self._measurement = measurement
            return pose

        def last_measurement(self):
            return self._measurement

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            look_calls.append((pose, linear, world_state))

        async def move_to(self, pose, world_state, linear=False):
            moves.append((pose, linear, world_state))

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=None,
        gripper_name="pick-grip",
        world=World(),
        target_prop_name="pick_cube",
        movable_prop_names=("pick_cube",),
        randomize_seed=1,
        look_pose=pick_red_block.default_scan_pose(0.0),
    )
    asyncio.run(pipeline.run())

    assert len(look_calls) == 2  # scan, then the one-shot focus
    scan_pose, scan_linear, scan_ws = look_calls[0]
    assert scan_linear is False
    assert _contains_keepout(scan_ws)  # boxed out of the blocks' airspace
    focus_pose, focus_linear, focus_ws = look_calls[1]
    assert (focus_pose.x, focus_pose.y, focus_pose.z) == (-1210.0, 187.0, 350.0)
    assert focus_linear is False  # free (fast) BECAUSE the keep-out is present
    assert _contains_keepout(focus_ws)
    pre_grasp_pose, pre_grasp_linear, pre_grasp_ws = moves[0]
    assert pre_grasp_linear is True
    assert not _contains_keepout(pre_grasp_ws)  # the descent must pierce the box
    assert (pre_grasp_pose.x, pre_grasp_pose.y) == (-1210.0, 187.0)


def test_pre_grasp_waypoint_move_runs_first_when_detection_ends_elsewhere(monkeypatch):
    """A detection that never focused (the block sits close enough to the
    scan centre) ends at the SCAN height, not the FOCUS height the waypoint
    uses - the waypoint move (free, keep-out boxed) must run before the
    linear pre-grasp."""
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    moves: list[tuple[Pose, bool, object]] = []

    class World:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [-1025.0, 5.0, 30.0]}}
            return {"geometries": []}

    class Detector:
        async def block_pose_world(self):
            return Pose(x=-1025.0, y=5.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            moves.append((pose, linear, world_state))

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=World(),
        target_prop_name="pick_cube",
        movable_prop_names=("pick_cube",),
        randomize_seed=1,
        look_pose=pick_red_block.default_scan_pose(0.0),
    )
    asyncio.run(pipeline.run())

    waypoint_pose, waypoint_linear, waypoint_ws = moves[0]
    assert (waypoint_pose.x, waypoint_pose.y, waypoint_pose.z) == (-1025.0, 5.0, 350.0)
    assert waypoint_linear is False  # free (fast) BECAUSE the keep-out is present
    assert _contains_keepout(waypoint_ws)

    pre_grasp_pose, pre_grasp_linear, pre_grasp_ws = moves[1]
    assert pre_grasp_linear is True
    assert not _contains_keepout(pre_grasp_ws)


def test_focus_falls_back_to_linear_when_the_keep_out_plan_fails(monkeypatch, capsys):
    """A keep-out-boxed free focus plan can fail (an aborted run can strand
    the arm inside the box); the pipeline retries the focus linear instead of
    aborting the pick."""
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    look_calls: list[tuple[Pose, bool]] = []
    readings = iter(
        [
            (Pose(x=-1210.0, y=187.0, z=999.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0), None),
            (
                Pose(x=-1210.0, y=187.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
                {"footprint_mm": [60.0, 60.0], "height_mm": 60.0, "size_mm": 60.0},
            ),
        ]
    )

    class World:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [-1210.0, 187.0, 30.0]}}
            return {"geometries": []}

    class Detector:
        def __init__(self):
            self._measurement = None

        async def block_pose_world(self):
            pose, measurement = next(readings)
            self._measurement = measurement
            return pose

        def last_measurement(self):
            return self._measurement

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            look_calls.append((pose, linear))
            if not linear and pose.z == 350.0:
                raise RuntimeError("motion move of 'wrist-cam' reported failure")

        async def move_to(self, pose, world_state, linear=False):
            return None

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=None,
        gripper_name="pick-grip",
        world=World(),
        target_prop_name="pick_cube",
        movable_prop_names=("pick_cube",),
        randomize_seed=1,
        look_pose=pick_red_block.default_scan_pose(0.0),
    )
    asyncio.run(pipeline.run())

    # scan (free, boxed), free focus (raises), linear focus retry, same pose
    assert [(pose.z, linear) for pose, linear in look_calls] == [
        (650.0, False),
        (350.0, False),
        (350.0, True),
    ]
    assert (look_calls[2][0].x, look_calls[2][0].y) == (-1210.0, 187.0)
    assert "keep-out focus move failed" in capsys.readouterr().out


def test_focus_is_linear_when_no_pick_region_boxes_the_airspace(monkeypatch):
    """Without a randomize region there is no keep-out to box the blocks'
    airspace, so the focus move must be linear (slow but incapable of
    swinging through the excluded target block)."""
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    look_calls: list[tuple[Pose, bool]] = []
    readings = iter(
        [
            (Pose(x=-1210.0, y=187.0, z=999.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0), None),
            (
                Pose(x=-1210.0, y=187.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
                {"footprint_mm": [60.0, 60.0], "height_mm": 60.0, "size_mm": 60.0},
            ),
        ]
    )

    class World:
        async def do_command(self, command):
            return {"geometries": []}

    class Detector:
        def __init__(self):
            self._measurement = None

        async def block_pose_world(self):
            pose, measurement = next(readings)
            self._measurement = measurement
            return pose

        def last_measurement(self):
            return self._measurement

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            look_calls.append((pose, linear))

        async def move_to(self, pose, world_state, linear=False):
            return None

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=None,
        gripper_name="pick-grip",
        world=World(),
        target_prop_name="pick_cube",
        look_pose=pick_red_block.default_scan_pose(0.0),
        scan_centre_mm=(-1025.0, 0.0),
    )
    asyncio.run(pipeline.run())

    assert [(pose.z, linear) for pose, linear in look_calls] == [
        (650.0, False),
        (350.0, True),
    ]


def test_hold_failure_prints_diagnostics_before_raising(monkeypatch, capsys):
    monkeypatch.setattr(pick_red_block, "HOLD_SAMPLE_S", 0.01)

    class Detector:
        async def block_pose_world(self):
            return Pose(x=100.0, y=200.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            return None

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return SimpleNamespace(is_holding_something=False, meta={"jaw_deg": 47.0})

    async def diagnose():
        return {"jaw_deg": 47.0, "block_pose_mm": {"x": 100.0, "y": 200.0, "z": 30.0}}

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        verify_detection_height=False,
        hold_s=1.0,
        diagnose=diagnose,
    )

    with pytest.raises(RuntimeError, match="hold failed"):
        asyncio.run(pipeline.run())

    out = capsys.readouterr().out
    diagnostics_line = next(
        line for line in out.splitlines() if line.startswith(GRAB_DIAGNOSTICS_MARKER)
    )
    report = json.loads(diagnostics_line.removeprefix(GRAB_DIAGNOSTICS_MARKER))
    assert report["jaw_deg"] == 47.0


def test_main_mock_measures_a_non_60mm_block_and_completes_the_pick(capsys):
    # the mock-wrist-cam handle is cached by name for the process (XC-4): a
    # spawn attribute (block_size_mm) changing from whatever an earlier test
    # left behind is rejected unless the cached handle is released first
    SimManager.get().release_handle("mock-wrist-cam")
    try:
        exit_code = main(["--mock", "--hold-s", "0", "--mock-block-size-mm", "50"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "grab: True" in out

        measured_line = next(
            line for line in out.splitlines() if line.startswith(MEASURED_BLOCK_MARKER)
        )
        measured = json.loads(measured_line.removeprefix(MEASURED_BLOCK_MARKER))
        assert measured["size_mm"] == pytest.approx(50.0, rel=0.1)
    finally:
        SimManager.get().release_handle("mock-wrist-cam")


_TALLEST_REGION_MM = ([500.0, -100.0, 0.0], [700.0, 100.0, 0.0])
_TALLEST_SIZE_RANGE_MM = (50.0, 70.0)


def _region_quadrant_points(region_mm, z_mm, count_per_quadrant=50, seed=0):
    """count_per_quadrant synthetic points (world mm) spread evenly across
    each of the region footprint's four quadrants, all at ``z_mm``."""
    rng = np.random.default_rng(seed)
    (x0, y0, _z0), (x1, y1, _z1) = region_mm
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    mid_x, mid_y = (lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0
    quadrants = []
    for x_lo, x_hi in ((lo_x, mid_x), (mid_x, hi_x)):
        for y_lo, y_hi in ((lo_y, mid_y), (mid_y, hi_y)):
            xs = rng.uniform(x_lo + 1.0, x_hi - 1.0, count_per_quadrant)
            ys = rng.uniform(y_lo + 1.0, y_hi - 1.0, count_per_quadrant)
            zs = np.full(count_per_quadrant, z_mm)
            quadrants.append(np.column_stack([xs, ys, zs]))
    return np.vstack(quadrants)


def test_tallest_in_region_clean_cloud_is_trusted_with_the_max_height():
    cloud = _region_quadrant_points(_TALLEST_REGION_MM, 60.0)
    estimate = tallest_in_region_mm(cloud, _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM)
    assert estimate.trusted
    assert estimate.reasons == []
    assert estimate.tallest_mm == pytest.approx(60.0)
    assert estimate.points == 200


def test_tallest_in_region_clips_to_the_region_footprint():
    cloud = _region_quadrant_points(_TALLEST_REGION_MM, 60.0)
    outside = np.array([[900.0, 900.0, 500.0]] * 10)
    estimate = tallest_in_region_mm(
        np.vstack([cloud, outside]), _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM
    )
    assert estimate.tallest_mm == pytest.approx(60.0)
    assert estimate.points == 200


def test_tallest_in_region_drops_points_at_or_below_the_support():
    cloud = _region_quadrant_points(_TALLEST_REGION_MM, 60.0)
    support_points = _region_quadrant_points(_TALLEST_REGION_MM, 0.0, seed=1)
    estimate = tallest_in_region_mm(
        np.vstack([cloud, support_points]), _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM
    )
    assert estimate.points == 200
    assert estimate.tallest_mm == pytest.approx(60.0)


def test_tallest_in_region_point_floor_reason_fires_alone():
    cloud = _region_quadrant_points(_TALLEST_REGION_MM, 60.0, count_per_quadrant=12)
    estimate = tallest_in_region_mm(cloud, _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM)
    assert not estimate.trusted
    assert any("point floor" in reason for reason in estimate.reasons)
    assert not any("size window" in reason for reason in estimate.reasons)
    assert not any("lone-point top" in reason for reason in estimate.reasons)
    assert not any("quadrant" in reason for reason in estimate.reasons)


def test_tallest_in_region_size_window_reason_fires_alone():
    cloud = _region_quadrant_points(_TALLEST_REGION_MM, 200.0)
    estimate = tallest_in_region_mm(cloud, _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM)
    assert not estimate.trusted
    assert any("size window" in reason for reason in estimate.reasons)
    assert not any("point floor" in reason for reason in estimate.reasons)
    assert not any("lone-point top" in reason for reason in estimate.reasons)
    assert not any("quadrant" in reason for reason in estimate.reasons)


def test_tallest_in_region_lone_point_top_reason_fires_alone():
    bulk = _region_quadrant_points(_TALLEST_REGION_MM, 60.0)
    outliers = np.array([[550.0, -50.0, 75.0], [550.0, -40.0, 75.0], [550.0, -30.0, 75.0]])
    estimate = tallest_in_region_mm(
        np.vstack([bulk, outliers]), _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM
    )
    assert not estimate.trusted
    assert estimate.tallest_mm == pytest.approx(75.0)
    assert any("lone-point top" in reason for reason in estimate.reasons)
    assert not any("point floor" in reason for reason in estimate.reasons)
    assert not any("size window" in reason for reason in estimate.reasons)
    assert not any("quadrant" in reason for reason in estimate.reasons)


def test_tallest_in_region_quadrant_coverage_reason_fires_alone():
    rng = np.random.default_rng(2)
    xs = rng.uniform(501.0, 599.0, 200)
    ys = rng.uniform(-99.0, -1.0, 200)
    cloud = np.column_stack([xs, ys, np.full(200, 60.0)])
    estimate = tallest_in_region_mm(cloud, _TALLEST_REGION_MM, 0.0, _TALLEST_SIZE_RANGE_MM)
    assert not estimate.trusted
    assert any("quadrant" in reason for reason in estimate.reasons)
    assert not any("point floor" in reason for reason in estimate.reasons)
    assert not any("size window" in reason for reason in estimate.reasons)
    assert not any("lone-point top" in reason for reason in estimate.reasons)


def test_tallest_in_region_skips_the_quadrant_check_without_a_region():
    rng = np.random.default_rng(3)
    xs = rng.uniform(-5000.0, 5000.0, 150)
    ys = rng.uniform(-5000.0, 5000.0, 150)
    cloud = np.column_stack([xs, ys, np.full(150, 60.0)])
    estimate = tallest_in_region_mm(cloud, None, 0.0, _TALLEST_SIZE_RANGE_MM)
    assert estimate.trusted
    assert estimate.reasons == []
    assert estimate.tallest_mm == pytest.approx(60.0)
    assert estimate.points == 150


def test_keepout_height_mm_matches_the_gpu_validated_anchor():
    assert keepout_height_mm(60.0, 60.0) == 130.0


def test_carry_clear_above_support_mm_matches_the_gpu_validated_anchor():
    assert carry_clear_above_support_mm(60.0, 60.0) == 200.0


def test_keepout_height_and_carry_clear_strictly_increase_with_tallest():
    tallest_values = [30.0, 60.0, 90.0, 120.0]
    keepouts = [keepout_height_mm(t, 60.0) for t in tallest_values]
    carries = [carry_clear_above_support_mm(t, 60.0) for t in tallest_values]
    assert keepouts == sorted(keepouts)
    assert len(set(keepouts)) == len(keepouts)
    assert carries == sorted(carries)
    assert len(set(carries)) == len(carries)


def test_keepout_height_mm_stays_within_tallest_plus_150_over_a_grid():
    for tallest_mm in (30.0, 60.0, 90.0, 120.0):
        for held_mm in (30.0, 50.0, 75.0):
            height = keepout_height_mm(tallest_mm, held_mm)
            assert tallest_mm < height < tallest_mm + 150.0


def test_tallest_sweep_attempts_leads_with_four_region_corners_then_scan_attempts():
    """Corners first: a region-centre vantage hangs the gripper inside the
    footprint where it reads as a ~274 mm object (GPU phase-4 run 1)."""
    attempts = tallest_sweep_attempts(_TALLEST_REGION_MM)
    scan_attempts = pick_red_block.SCAN_ATTEMPTS
    assert attempts[4:] == scan_attempts
    assert len(attempts) == len(scan_attempts) + 4


def test_tallest_sweep_attempts_corner_offsets_stay_inside_the_region():
    (x0, y0, _z0), (x1, y1, _z1) = _TALLEST_REGION_MM
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    centre_x, centre_y = (lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0
    corners = tallest_sweep_attempts(_TALLEST_REGION_MM)[:4]
    assert len(corners) == 4
    for x_offset, y_offset, theta in corners:
        assert lo_x <= centre_x + x_offset <= hi_x
        assert lo_y <= centre_y + y_offset <= hi_y
        assert theta == 0.0


def test_pick_area_keepout_defaults_to_the_legacy_height():
    keepout = pick_red_block.pick_area_keepout(([450.0, -250.0, 0.0], [700.0, 250.0, 0.0]))
    assert keepout.box.dims_mm.z == pick_red_block.KEEPOUT_HEIGHT_MM


def test_pick_area_keepout_honours_a_custom_height_mm():
    keepout = pick_red_block.pick_area_keepout(
        ([450.0, -250.0, 0.0], [700.0, 250.0, 0.0]), height_mm=180.0
    )
    assert keepout.box.dims_mm.z == 180.0
    assert keepout.center.z == 90.0


def test_measured_tallest_marker_is_a_distinct_marker_string():
    assert MEASURED_TALLEST_MARKER == "MEASURED_TALLEST_JSON="


def test_main_mock_measures_tallest_from_the_side_camera(capsys):
    SimManager.get().release_handle("mock-wrist-cam")
    SimManager.get().release_handle("mock-side-cam")
    try:
        exit_code = main(
            ["--mock", "--hold-s", "0", "--block-size-mm", "60", "--randomize-size-mm", "30,90"]
        )
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "grab: True" in out

        tallest_line = next(
            line for line in out.splitlines() if line.startswith(MEASURED_TALLEST_MARKER)
        )
        marker = json.loads(tallest_line.removeprefix(MEASURED_TALLEST_MARKER))
        assert marker["source"] == "side"
        assert marker["trusted"] is True
        assert marker["tallest_mm"] == pytest.approx(90.0, abs=10.0)
        assert marker["keepout_height_mm"] == pytest.approx(
            keepout_height_mm(marker["tallest_mm"], 60.0)
        )
        assert marker["carry_clear_above_support_mm"] == pytest.approx(
            carry_clear_above_support_mm(marker["tallest_mm"], 60.0)
        )
    finally:
        SimManager.get().release_handle("mock-wrist-cam")
        SimManager.get().release_handle("mock-side-cam")


class _FakeMover:
    def __init__(self):
        self.looked = []

    async def look_from(self, pose, world_state, linear=False):
        self.looked.append(pose)

    async def move_to(self, pose, world_state, linear=False):
        return None


class _FakeScanner:
    def __init__(self, clouds):
        self._clouds = list(clouds)

    async def scan_world_mm(self):
        return self._clouds.pop(0)


def test_measure_tallest_falls_through_side_then_wrist_sweep_to_first_trusted(monkeypatch):
    estimates = [
        pick_red_block.TallestEstimate(tallest_mm=10.0, points=1, trusted=False, reasons=["r0"]),
        pick_red_block.TallestEstimate(tallest_mm=20.0, points=1, trusted=False, reasons=["r1"]),
        pick_red_block.TallestEstimate(tallest_mm=90.0, points=500, trusted=True, reasons=[]),
    ]

    def fake_tallest(points, region_mm, support_z_mm, size_range_mm):
        return estimates.pop(0)

    monkeypatch.setattr(pick_red_block, "tallest_in_region_mm", fake_tallest)
    monkeypatch.setattr(
        pick_red_block,
        "tallest_sweep_attempts",
        lambda region: ((0.0, 0.0, 0.0), (10.0, 0.0, 90.0)),
    )

    mover = _FakeMover()
    side_scanner = _FakeScanner([np.zeros((1, 3))])
    wrist_scanner = _FakeScanner([np.zeros((1, 3)), np.zeros((1, 3))])
    pipeline = pick_red_block.PickPipeline(
        detector=None,
        mover=mover,
        gripper=None,
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        randomize_size_range_mm=(30.0, 90.0),
        pick_region_mm=([500.0, -100.0, 0.0], [700.0, 100.0, 0.0]),
        scan_centre_mm=(600.0, 0.0),
        look_pose=pick_red_block._pointing_down(500.0, 150.0, 350.0),
        side_scanner=side_scanner,
        wrist_scanner=wrist_scanner,
        verify_detection_height=True,
        support_z_mm=0.0,
    )

    asyncio.run(pipeline._measure_tallest(pick_red_block.world_state(None)))

    assert pipeline.tallest_source == "wrist_sweep"
    assert pipeline.tallest_estimate.trusted is True
    assert pipeline.tallest_estimate.tallest_mm == pytest.approx(90.0)
    assert len(mover.looked) == 2  # walked both wrist-sweep attempts before the second won
    assert len(pipeline.tallest_scan_poses_mm) == 2


def test_measure_tallest_falls_back_to_size_range_max_when_everything_is_untrusted(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        pick_red_block,
        "tallest_in_region_mm",
        lambda *a, **k: pick_red_block.TallestEstimate(
            tallest_mm=5.0, points=1, trusted=False, reasons=["nope"]
        ),
    )
    monkeypatch.setattr(pick_red_block, "tallest_sweep_attempts", lambda region: ((0.0, 0.0, 0.0),))

    mover = _FakeMover()
    pipeline = pick_red_block.PickPipeline(
        detector=None,
        mover=mover,
        gripper=None,
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        randomize_size_range_mm=(30.0, 90.0),
        pick_region_mm=([500.0, -100.0, 0.0], [700.0, 100.0, 0.0]),
        scan_centre_mm=(600.0, 0.0),
        look_pose=pick_red_block._pointing_down(500.0, 150.0, 350.0),
        side_scanner=_FakeScanner([np.zeros((1, 3))]),
        wrist_scanner=_FakeScanner([np.zeros((1, 3))]),
        verify_detection_height=True,
    )

    asyncio.run(pipeline._measure_tallest(pick_red_block.world_state(None)))

    assert pipeline.tallest_source == "fallback"
    assert pipeline.tallest_estimate.trusted is False
    assert pipeline.tallest_estimate.tallest_mm == 90.0
    assert "WARNING" in capsys.readouterr().out


def test_measure_tallest_runs_the_wrist_sweep_without_a_side_scanner(monkeypatch):
    """GPU phase-4 run 3: --tallest-camera "" must still sweep the wrist
    camera, not jump straight to the size-range ceiling."""
    monkeypatch.setattr(
        pick_red_block,
        "tallest_in_region_mm",
        lambda *a, **k: pick_red_block.TallestEstimate(
            tallest_mm=81.0, points=500, trusted=True, reasons=[]
        ),
    )
    monkeypatch.setattr(pick_red_block, "tallest_sweep_attempts", lambda region: ((0.0, 0.0, 0.0),))

    pipeline = pick_red_block.PickPipeline(
        detector=None,
        mover=_FakeMover(),
        gripper=None,
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        randomize_size_range_mm=(30.0, 90.0),
        pick_region_mm=([500.0, -100.0, 0.0], [700.0, 100.0, 0.0]),
        scan_centre_mm=(600.0, 0.0),
        look_pose=pick_red_block._pointing_down(500.0, 150.0, 350.0),
        side_scanner=None,
        wrist_scanner=_FakeScanner([np.zeros((1, 3))]),
        verify_detection_height=True,
    )

    asyncio.run(pipeline._measure_tallest(pick_red_block.world_state(None)))

    assert pipeline.tallest_source == "wrist_sweep"
    assert pipeline.tallest_estimate.trusted is True
    assert pipeline.tallest_estimate.tallest_mm == 81.0


def test_camera_world_transform_mm_recovers_a_known_rotation_and_translation():
    theta = math.radians(37.0)
    rotation_true = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_true = np.array([500.0, -200.0, 900.0])

    class FakeRobot:
        async def transform_pose(self, pose_in_frame, dest_frame):
            p = pose_in_frame.pose
            cam_point = np.array([p.x, p.y, p.z])
            world_point = rotation_true @ cam_point + translation_true
            return SimpleNamespace(
                pose=Pose(
                    x=float(world_point[0]),
                    y=float(world_point[1]),
                    z=float(world_point[2]),
                    o_x=0.0,
                    o_y=0.0,
                    o_z=1.0,
                    theta=0.0,
                )
            )

    rotation, translation = asyncio.run(
        pick_red_block.camera_world_transform_mm(FakeRobot(), "side-cam")
    )
    assert rotation == pytest.approx(rotation_true, abs=1e-6)
    assert translation == pytest.approx(translation_true, abs=1e-6)


def test_no_size_range_keeps_legacy_keepout_height_and_prints_no_tallest_marker(
    monkeypatch, capsys
):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(pick_red_block, "PLACE_SETTLE_S", 0.0)
    moves = []

    class World:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                return {"positions": {"pick_cube": [520.0, 20.0, 30.0]}}
            return {"geometries": [_block_geometry(520.0, 20.0, 30.0), _pad_geometry()]}

    class Detector:
        async def block_pose_world(self):
            return Pose(x=520.0, y=20.0, z=30.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class Mover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            moves.append((pose, world_state, linear))

    class Gripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    pipeline = pick_red_block.PickPipeline(
        detector=Detector(),
        mover=Mover(),
        gripper=Gripper(),
        block_name="pick_cube",
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=World(),
        target_prop_name="pick_cube",
        movable_prop_names=("pick_cube",),
        randomize_seed=4,
        place_prop_name="place_pad",
    )
    asyncio.run(pipeline.run())

    out = capsys.readouterr().out
    assert not any(line.startswith(MEASURED_TALLEST_MARKER) for line in out.splitlines())

    _, carry_state, _ = moves[5]  # waypoint, pre-grasp, grasp, lift, raise, carry
    keepout = next(
        g
        for frame in carry_state.obstacles
        for g in frame.geometries
        if g.label == "pick_area_keepout"
    )
    assert keepout.box.dims_mm.z == pick_red_block.KEEPOUT_HEIGHT_MM


def test_client_defaults_match_the_cell_layout_seam():
    """The client keeps literal geometry constants (it never imports
    isaac_module outside --mock), so this cross-checks them against the
    fragment's own source of truth instead of trusting a copy-paste."""
    from isaac_module import cell_layout

    assert pick_red_block.DEFAULT_LOOK_XY_MM == cell_layout.SCATTER_CENTRE_MM[:2]
    assert pick_red_block.REACHABLE_REGION_X_MM == cell_layout.SCATTER_ZONE_X_MM
    assert pick_red_block.REACHABLE_REGION_Y_MM == cell_layout.SCATTER_ZONE_Y_MM

    args = pick_red_block._parse_args([])
    assert args.support_z_mm == cell_layout.TABLE_TOP_Z_MM
    assert args.block == cell_layout.pool_block_name("red", 1)
    assert args.place_pad == cell_layout.pad_name("red")

    assert pick_red_block.TABLE_CENTER_MM == cell_layout.TABLE_CENTRES_MM["table_source"] + (
        cell_layout.TABLE_TOP_Z_MM / 2.0,
    )
    assert pick_red_block.TABLE_DIMS_MM == cell_layout.TABLE_DIMS_MM


def test_parse_randomize_region_accepts_corners():
    assert pick_red_block._parse_randomize_region_mm("-1350,-300,-700,300") == (
        -1350.0,
        -300.0,
        -700.0,
        300.0,
    )


def test_parse_randomize_region_rejects_wrong_count():
    with pytest.raises(argparse.ArgumentTypeError):
        pick_red_block._parse_randomize_region_mm("-1350,-300,-700")


def test_parse_randomize_region_rejects_inverted_corners():
    with pytest.raises(argparse.ArgumentTypeError):
        pick_red_block._parse_randomize_region_mm("-700,300,-1350,-300")


def test_randomize_names_default_keeps_every_movable_behaviour():
    args = pick_red_block._parse_args([])
    assert pick_red_block._movable_prop_names_from_args(args) == ()


def test_randomize_names_parses_comma_separated_props():
    args = pick_red_block._parse_args(["--randomize-names", "block_red_1,block_red_2"])
    assert pick_red_block._movable_prop_names_from_args(args) == ("block_red_1", "block_red_2")


def test_randomize_region_default_keeps_the_reachable_region_fallback():
    args = pick_red_block._parse_args([])
    assert pick_red_block._randomize_region_mm_from_args(args) is None


def test_randomize_region_wires_the_flag_at_the_support_height():
    args = pick_red_block._parse_args(
        ["--support-z-mm", "750", "--randomize-region=-1350,-300,-700,300"]
    )
    assert pick_red_block._randomize_region_mm_from_args(args) == (
        [-1350.0, -300.0, 750.0],
        [-700.0, 300.0, 750.0],
    )


def test_randomize_names_and_region_land_on_the_pipeline_fields(monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.0)
    captured: dict = {}

    class FakeWorld:
        async def do_command(self, command):
            if command["command"] == "randomize_props":
                captured["names"] = command["names"]
                captured["region"] = command["region"]
                return {"positions": {"block_red_1": [-1020.0, 5.0, 780.0]}}
            return {"geometries": []}

    class FakeDetector:
        async def block_pose_world(self):
            return Pose(x=-1020.0, y=5.0, z=780.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    class FakeMover:
        async def look_from(self, pose, world_state, linear=False):
            return None

        async def move_to(self, pose, world_state, linear=False):
            return None

    class FakeGripper:
        async def open(self):
            return None

        async def grab(self):
            return True

        async def is_holding_something(self):
            return True

    args = pick_red_block._parse_args(
        [
            "--support-z-mm",
            "750",
            "--randomize-names",
            "block_red_1",
            "--randomize-region=-1350,-300,-700,300",
        ]
    )
    pipeline = pick_red_block.PickPipeline(
        detector=FakeDetector(),
        mover=FakeMover(),
        gripper=FakeGripper(),
        block_name=args.block,
        block_size_mm=60.0,
        gripper_name="pick-grip",
        world=FakeWorld(),
        target_prop_name="block_red_1",
        movable_prop_names=pick_red_block._movable_prop_names_from_args(args),
        randomize_seed=9,
        randomize_region_mm=pick_red_block._randomize_region_mm_from_args(args),
        support_z_mm=args.support_z_mm,
    )
    asyncio.run(pipeline.run())

    assert captured["names"] == ["block_red_1"]
    assert captured["region"] == [[-1350.0, -300.0, 750.0], [-700.0, 300.0, 750.0]]
