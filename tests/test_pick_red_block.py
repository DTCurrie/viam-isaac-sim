"""Unit + end-to-end tests for examples/pick_red_block.py (XC-8, W36, DEC-14, DEC-20).

The module lives under examples/ (not a package under src/) and depends only
on the stdlib, viam-sdk and numpy (isaac_module is imported lazily, only in
--mock code paths), so it is loaded here via importlib rather than adding
examples/ to pyproject's pythonpath - mirrors
tests/test_gpu_checklist_phase2.py's idiom exactly.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

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
RESET_MID_HOLD_MARKER = pick_red_block.RESET_MID_HOLD_MARKER
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
    assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (1200.0, 800.0, 740.0)
    center = table.center
    assert (center.x, center.y, center.z) == (600.0, 0.0, 370.0)


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
    exit_code = main(["--mock", "--hold-s", "0"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "grab: True" in out

    transform_line = next(
        line for line in out.splitlines() if line.startswith(HELD_BLOCK_TRANSFORM_MARKER)
    )
    payload = json.loads(transform_line.removeprefix(HELD_BLOCK_TRANSFORM_MARKER))
    assert payload["reference_frame"] == "pick_cube"
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
    assert pick_red_block.main(["--mock", "--hold-s", "0"]) == 0
    out = capsys.readouterr().out
    assert out.index("step: look") < out.index("step: detect")


def test_main_mock_holds_and_survives_a_reset_mid_hold(capsys, monkeypatch):
    monkeypatch.setattr(pick_red_block, "RESET_SETTLE_S", 0.05)
    monkeypatch.setattr(pick_red_block, "HOLD_SAMPLE_S", 0.05)
    exit_code = main(["--mock", "--hold-s", "0.1", "--reset-mid-hold"])

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
    assert (x0, y0, z0) == (50.0, -350.0, 0.0)
    assert (x1, y1, z1) == (1150.0, 350.0, 0.0)
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
        async def look_from(self, pose, world_state):
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
    assert randomize_command["region"] == [[450.0, -250.0, 0.0], [700.0, 250.0, 0.0]]


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
        async def look_from(self, pose, world_state):
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
    assert lo == [450.0, -250.0, 5.0]
    assert hi == [700.0, 250.0, 5.0]
    # no corner beyond the phase-3 verified pick radius: the (700, 250) pick
    verified_radius = (700.0**2 + 250.0**2) ** 0.5
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            assert (x**2 + y**2) ** 0.5 <= verified_radius + 1e-9


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
        async def look_from(self, pose, world_state):
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
    assert (scan.x, scan.y, scan.z) == (575.0, 0.0, 350.0)
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
            async def look_from(self, pose, world_state):
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

    # pre-grasp, grasp, lift, raise, carry, place, retreat
    assert len(fakes.moves) == 7
    grasp_pose_used = fakes.moves[1][0]
    raise_move, carry, place, retreat = (m[0] for m in fakes.moves[3:7])
    # the carry runs at constant height on a straight (linear) line
    assert (raise_move.x, raise_move.y) == (grasp_pose_used.x, grasp_pose_used.y)
    assert raise_move.z == carry.z
    assert fakes.moves[3][1] is True and fakes.moves[4][1] is True
    assert (carry.x, carry.y) == (700.0, -350.0)
    # place z reproduces the grasp offset over the pad top (10) plus the gap (15)
    assert place.z == pytest.approx(grasp_pose_used.z + 10.0 + 15.0)
    assert fakes.moves[5][1] is True  # the descent is linear
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

    # pre-grasp, grasp, lift only - and the release open still happens
    assert len(fakes.moves) == 3  # no carry moves without a pad
    assert fakes.opens == 2


class _FailingMoverFakes(_PlaceFakes):
    """Place fakes whose mover refuses configurable descents over the pad
    (y == -350 and z below a threshold), like a planner at its reach limit."""

    def __init__(self, fail_linear_below=None, fail_any_below=None):
        super().__init__(with_pad=True)
        outer = self

        class Mover:
            async def look_from(self, pose, world_state):
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
            async def look_from(self, pose, world_state):
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
    assert randomize["min_separation"] == 200.0  # FINDINGS W26


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

    # pre-grasp, grasp, lift, raise, carry, place, retreat
    states = fakes.move_world_states
    assert len(states) == 7
    assert list(states[0].transforms) == []
    assert list(states[1].transforms) == []
    for held in states[2:6]:
        (transform,) = held.transforms
        assert transform.reference_frame == "pick_cube"
        assert transform.pose_in_observer_frame.reference_frame == "pick-grip"
        # detected centre z 30, grasp TCP 39: the box hangs 9 mm below the TCP
        assert transform.pose_in_observer_frame.pose.z == pytest.approx(-9.0)
        # 60 mm block + 20 mm planning padding (ARM-10 execution error)
        assert transform.physical_object.box.dims_mm.x == 80.0
    assert list(states[6].transforms) == []  # released before the retreat


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
            async def look_from(self, pose, world_state):
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
    assert (scan.x, scan.y, scan.theta) == (575.0, 0.0, 0.0)
    # same spot, wrist turned a quarter: the shadow sweeps off the block
    assert (rescan.x, rescan.y, rescan.theta) == (575.0, 0.0, 90.0)
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
    keepout = pick_red_block.pick_area_keepout(([450.0, -250.0, 0.0], [700.0, 250.0, 0.0]))
    assert keepout.label == "pick_area_keepout"
    assert (keepout.center.x, keepout.center.y, keepout.center.z) == (575.0, 0.0, 65.0)
    dims = keepout.box.dims_mm
    assert (dims.x, dims.y, dims.z) == (350.0, 600.0, 130.0)


def test_pick_area_keepout_stands_on_the_region_support():
    """P5 cell regression (GPU run 13): the scatter region's z is the table
    top, and the no-fly box must cover the block airspace ABOVE it, not the
    floor's."""
    keepout = pick_red_block.pick_area_keepout(([450.0, -250.0, 750.0], [700.0, 250.0, 750.0]))
    assert keepout.center.z == 750.0 + 65.0
    assert keepout.box.dims_mm.z == 130.0


def test_default_scan_pose_rides_the_support_height():
    """GPU run 13: an absolute 350 mm scan z sent the camera 400 mm below the
    P5 table top - the default scan height is above the support."""
    floor = pick_red_block.default_scan_pose(0.0)
    table = pick_red_block.default_scan_pose(750.0)
    assert (floor.x, floor.y, floor.z) == (500.0, 150.0, 350.0)
    assert (table.x, table.y, table.z) == (500.0, 150.0, 1100.0)
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
        async def look_from(self, pose, world_state):
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

    # pre-grasp, grasp, lift, raise-above-keepout, free carry, place, retreat
    assert len(moves) == 7
    clear_pose, clear_state, clear_linear = moves[3]
    assert clear_pose.z == 200.0  # support 0 + CARRY_CLEAR_ABOVE_SUPPORT_MM
    assert clear_linear is True
    labels = {g.label for frame in clear_state.obstacles for g in frame.geometries}
    assert "pick_area_keepout" not in labels  # the hop starts inside the box

    carry_pose, carry_state, carry_linear = moves[4]
    assert (carry_pose.x, carry_pose.y) == (700.0, -350.0)
    assert carry_linear is False  # free, fast plan
    carry_labels = {g.label for frame in carry_state.obstacles for g in frame.geometries}
    assert "pick_area_keepout" in carry_labels
    (transform,) = carry_state.transforms  # the held block still rides along
    assert transform.reference_frame == "pick_cube"
