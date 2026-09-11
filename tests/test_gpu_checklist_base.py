import importlib.util
import math
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "gpu_checklist_base.py"
_spec = importlib.util.spec_from_file_location("gpu_checklist_base", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
gpu_checklist_base = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_base"] = gpu_checklist_base
_spec.loader.exec_module(gpu_checklist_base)

pose_from_prim_result = gpu_checklist_base.pose_from_prim_result
forward_axis = gpu_checklist_base.forward_axis
forward_distance_mm = gpu_checklist_base.forward_distance_mm
planar_distance_mm = gpu_checklist_base.planar_distance_mm
heading_delta_deg = gpu_checklist_base.heading_delta_deg
within_relative_tolerance = gpu_checklist_base.within_relative_tolerance
verdict = gpu_checklist_base.verdict
parse_args = gpu_checklist_base._parse_args


def test_parse_args_applies_defaults():
    args = parse_args(["--address", "10.0.0.5"])
    assert args.address == "10.0.0.5"
    assert args.api_key is None
    assert args.api_key_id is None
    assert args.base == "jetbot-base"
    assert args.world == "isaac-world"


def test_parse_args_overrides_every_default():
    args = parse_args(
        [
            "--address",
            "10.0.0.5",
            "--api-key",
            "k",
            "--api-key-id",
            "kid",
            "--base",
            "my-base",
            "--world",
            "my-world",
        ]
    )
    assert args.api_key == "k"
    assert args.api_key_id == "kid"
    assert args.base == "my-base"
    assert args.world == "my-world"


def test_pose_from_prim_result_reads_position_and_orientation_vector():
    result = {
        "prim_path": "/World/jetbot_base",
        "position_mm": [500.0, 0.0, 30.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "orientation_vector": {"o_x": 0.0, "o_y": 0.0, "o_z": 1.0, "theta_deg": 45.0},
    }
    assert pose_from_prim_result(result) == (500.0, 0.0, 30.0, 0.0, 0.0, 1.0, 45.0)


def test_forward_axis_faces_plus_x_at_identity_orientation():
    identity_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert forward_axis(identity_pose) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_forward_axis_faces_plus_y_after_a_90_degree_heading():
    pose = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0)
    assert forward_axis(pose) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_forward_distance_mm_is_full_displacement_when_moving_straight_ahead():
    before = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    after = (300.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert forward_distance_mm(before, after) == pytest.approx(300.0, abs=1e-9)


def test_forward_distance_mm_is_zero_for_purely_sideways_drift():
    before = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    after = (0.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert forward_distance_mm(before, after) == pytest.approx(0.0, abs=1e-9)


def test_forward_distance_mm_projects_onto_the_starting_heading():
    # heading 90 deg means +x locally points along world +y
    before = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0)
    after = (0.0, 120.0, 0.0, 0.0, 0.0, 1.0, 90.0)
    assert forward_distance_mm(before, after) == pytest.approx(120.0, abs=1e-9)


def test_planar_distance_mm_identical_poses_is_zero():
    pose = (100.0, 200.0, 0.0, 0.0, 0.0, 1.0, 30.0)
    assert planar_distance_mm(pose, pose) == pytest.approx(0.0, abs=1e-9)


def test_planar_distance_mm_ignores_heading():
    a = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    b = (3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 179.0)
    assert planar_distance_mm(a, b) == pytest.approx(5.0, abs=1e-9)


def test_heading_delta_deg_simple_turn():
    a = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0)
    b = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0)
    assert heading_delta_deg(a, b) == pytest.approx(90.0, abs=1e-9)


def test_heading_delta_deg_wraps_across_the_180_boundary():
    a = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 170.0)
    b = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -170.0)
    assert heading_delta_deg(a, b) == pytest.approx(20.0, abs=1e-9)


def test_within_relative_tolerance_accepts_the_boundary():
    assert within_relative_tolerance(130.0, 100.0, 0.30)
    assert not within_relative_tolerance(131.0, 100.0, 0.30)


def test_within_relative_tolerance_uses_the_expected_values_magnitude():
    assert within_relative_tolerance(-9.0, -10.0, 0.10)
    assert not within_relative_tolerance(-8.9, -10.0, 0.10)


def test_verdict_formats_pass_and_fail():
    assert verdict("thing", True, "all good") == "[PASS] thing: all good"
    assert verdict("thing", False, "off by 5mm") == "[FAIL] thing: off by 5mm"


def test_forward_axis_matches_math_expectation_for_a_45_degree_heading():
    pose = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 45.0)
    expected = (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.0)
    assert forward_axis(pose) == pytest.approx(expected, abs=1e-9)
