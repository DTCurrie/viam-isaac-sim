import importlib.util
import math
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

_MODULE_PATH = _EXAMPLES_DIR / "gpu_checklist_palletizer.py"
_spec = importlib.util.spec_from_file_location("gpu_checklist_palletizer", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
checklist = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_palletizer"] = checklist
_spec.loader.exec_module(checklist)


def test_verdict_formats_pass_and_fail():
    assert checklist.verdict("thing", True, "all good") == "[PASS] thing: all good"
    assert checklist.verdict("thing", False, "off by 5mm") == "[FAIL] thing: off by 5mm"


def test_pose_delta_mm_identical_poses_is_zero():
    pose = {"x": 100.0, "y": 200.0, "z": 300.0}
    assert checklist.pose_delta_mm(pose, pose) == pytest.approx(0.0, abs=1e-9)


def test_pose_delta_mm_is_the_euclidean_distance():
    pose_a = {"x": 0.0, "y": 0.0, "z": 0.0}
    pose_b = {"x": 3.0, "y": 4.0, "z": 0.0}
    assert checklist.pose_delta_mm(pose_a, pose_b) == pytest.approx(5.0, abs=1e-9)


def test_lift_delta_mm_reports_pure_vertical_travel():
    before = {"x": 10.0, "y": 20.0, "z": 30.0}
    after = {"x": 10.0, "y": 20.0, "z": 130.0}
    vertical_mm, horizontal_mm = checklist.lift_delta_mm(before, after)
    assert vertical_mm == pytest.approx(100.0, abs=1e-9)
    assert horizontal_mm == pytest.approx(0.0, abs=1e-9)


def test_lift_delta_mm_reports_horizontal_travel_too():
    before = {"x": 0.0, "y": 0.0, "z": 0.0}
    after = {"x": 3.0, "y": 4.0, "z": 0.0}
    vertical_mm, horizontal_mm = checklist.lift_delta_mm(before, after)
    assert vertical_mm == pytest.approx(0.0, abs=1e-9)
    assert horizontal_mm == pytest.approx(5.0, abs=1e-9)


def test_rode_with_tool_true_when_the_box_lifted_with_the_tool():
    before = {"x": 500.0, "y": 0.0, "z": 780.0}
    after = {"x": 500.0, "y": 0.0, "z": 880.0}
    ok, vertical_mm, horizontal_mm = checklist.rode_with_tool(before, after)
    assert ok is True
    assert vertical_mm == pytest.approx(100.0, abs=1e-9)
    assert horizontal_mm == pytest.approx(0.0, abs=1e-9)


def test_rode_with_tool_false_when_the_box_was_left_behind():
    # a box left behind reads ~0 mm vertical delta, not the expected lift distance
    before = {"x": 500.0, "y": 0.0, "z": 780.0}
    after = {"x": 500.0, "y": 0.0, "z": 780.2}
    ok, vertical_mm, _horizontal_mm = checklist.rode_with_tool(before, after)
    assert ok is False
    assert vertical_mm == pytest.approx(0.2, abs=1e-9)


def test_rode_with_tool_false_when_horizontal_drift_exceeds_tolerance():
    before = {"x": 500.0, "y": 0.0, "z": 780.0}
    after = {"x": 550.0, "y": 0.0, "z": 880.0}
    ok, _vertical_mm, horizontal_mm = checklist.rode_with_tool(before, after)
    assert ok is False
    assert horizontal_mm == pytest.approx(50.0, abs=1e-9)


def test_placement_check_ok_within_tolerance_of_the_place_target():
    target = {"x": 700.0, "y": 0.0}
    box_pose = {"x": target["x"] + 2.0, "y": target["y"], "z": 354.0}
    ok, error_mm = checklist.placement_check(box_pose, target)
    assert ok is True
    assert error_mm == pytest.approx(2.0, abs=1e-9)


def test_placement_check_fails_past_the_tolerance():
    target = {"x": 700.0, "y": 0.0}
    box_pose = {"x": target["x"] + 50.0, "y": target["y"], "z": 354.0}
    ok, error_mm = checklist.placement_check(box_pose, target)
    assert ok is False
    assert error_mm == pytest.approx(50.0, abs=1e-9)


def test_placement_check_fails_when_the_box_never_registered():
    target = {"x": 700.0, "y": 0.0}
    ok, error_mm = checklist.placement_check(None, target)
    assert ok is False
    assert error_mm == math.inf


def test_settle_check_ok_when_the_box_did_not_drift():
    pose = {"x": 650.0, "y": -200.0, "z": 244.0}
    ok, drift_mm = checklist.settle_check(pose, pose)
    assert ok is True
    assert drift_mm == pytest.approx(0.0, abs=1e-9)


def test_settle_check_fails_past_the_drift_tolerance():
    first = {"x": 650.0, "y": -200.0, "z": 244.0}
    second = {"x": 650.0, "y": -200.0, "z": 244.0 + checklist.DRIFT_TOLERANCE_MM + 1.0}
    ok, drift_mm = checklist.settle_check(first, second)
    assert ok is False
    assert drift_mm == pytest.approx(checklist.DRIFT_TOLERANCE_MM + 1.0, abs=1e-9)


def test_settle_check_fails_when_a_reading_is_missing():
    ok, drift_mm = checklist.settle_check(None, {"x": 0.0, "y": 0.0, "z": 0.0})
    assert ok is False
    assert drift_mm == math.inf


def test_ready_time_s_returns_elapsed_time_to_the_first_ready_sample():
    samples = [
        (100.0, {"ready": False}),
        (100.5, {"ready": False}),
        (101.0, {"ready": True}),
        (101.5, {"ready": True}),
    ]
    assert checklist.ready_time_s(samples) == 1.0


def test_ready_time_s_returns_none_when_never_ready():
    samples = [(100.0, {"ready": False}), (100.5, {"ready": False})]
    assert checklist.ready_time_s(samples) is None


def test_ready_time_s_returns_none_for_an_empty_sequence():
    assert checklist.ready_time_s([]) is None


_REQUIRED_POSE_ARGS = [
    "--pick-x-mm",
    "0.0",
    "--pick-y-mm",
    "0.0",
    "--pick-z-mm",
    "0.0",
    "--place-x-mm",
    "700.0",
    "--place-y-mm",
    "0.0",
]


def test_parse_args_requires_box_prop():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            ["--address", "10.0.0.1", "--run-label", "cold", *_REQUIRED_POSE_ARGS]
        )


def test_parse_args_requires_run_label():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            ["--address", "10.0.0.1", "--box-prop", "infeed_box", *_REQUIRED_POSE_ARGS]
        )


def test_parse_args_rejects_an_unlabelled_run_label():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            [
                "--address",
                "10.0.0.1",
                "--box-prop",
                "infeed_box",
                "--run-label",
                "lukewarm",
                *_REQUIRED_POSE_ARGS,
            ]
        )


def test_parse_args_requires_pick_and_place_poses():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            ["--address", "10.0.0.1", "--box-prop", "infeed_box", "--run-label", "cold"]
        )


def test_parse_args_defaults():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.world == "isaac-world"
    assert args.arm == "arm-1"
    assert args.gripper == "gripper-1"
    assert args.motion == "builtin"
    assert args.palletizer == "box-palletizer"
    assert args.box_prop == "infeed_box"
    assert (args.pick_x_mm, args.pick_y_mm, args.pick_z_mm) == (0.0, 0.0, 0.0)
    assert (args.place_x_mm, args.place_y_mm) == (700.0, 0.0)
    assert args.phase == 1
    assert args.run_label == "cold"


def test_parse_args_accepts_a_warm_label():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "warm",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.run_label == "warm"


def test_main_reports_failure_and_exits_nonzero_when_address_is_missing(capsys):
    exit_code = checklist.main(
        ["--box-prop", "infeed_box", "--run-label", "cold", *_REQUIRED_POSE_ARGS]
    )
    assert exit_code == 1
    assert "--address is required" in capsys.readouterr().out


def test_fragment_component_frames_mm_reads_world_parented_translations():
    fragment = {
        "components": [
            {"name": "pallet", "frame": {"parent": "world", "translation": {"x": 200, "y": 500}}},
            {"name": "robot-pedestal", "frame": {"parent": "world", "translation": {}}},
            # parented to another component: not extracted, since composing its world pose
            # would need that component's own resolved pose, which this checklist never assumes
            {"name": "gripper-1", "frame": {"parent": "arm-1", "translation": {"z": 196}}},
            # no frame at all
            {"name": "tray-dock"},
        ]
    }
    frames = checklist.fragment_component_frames_mm(fragment)
    assert frames == {
        "pallet": {"x": 200.0, "y": 500.0, "z": 0.0},
        "robot-pedestal": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def test_support_geometry_mm_finds_the_named_components_prefix():
    geometries = [
        {"name": "pallet-slab", "box_dims_mm": [500, 350, 100]},
        {"name": "pick-station-conveyor", "box_dims_mm": [1100, 400, 200]},
    ]
    found = checklist.support_geometry_mm(geometries, "pallet")
    assert found is not None
    assert found["name"] == "pallet-slab"


def test_support_geometry_mm_returns_none_when_nothing_matches():
    geometries = [{"name": "pallet-slab", "box_dims_mm": [500, 350, 100]}]
    assert checklist.support_geometry_mm(geometries, "robot-pedestal") is None


def test_support_top_z_mm_is_the_supports_pose_plus_half_its_height():
    support = {
        "pose_in_world_mm": {"x": 200.0, "y": 500.0, "z": 200.0},
        "box_dims_mm": [500, 350, 100],
    }
    assert checklist.support_top_z_mm(support) == pytest.approx(250.0, abs=1e-9)


def test_expected_rest_z_mm_stacks_half_the_box_height_on_the_support_top():
    assert checklist.expected_rest_z_mm(250.0, 100.0) == pytest.approx(300.0, abs=1e-9)


def test_resting_check_ok_within_tolerance_of_the_expected_height():
    ok, error_mm = checklist.resting_check(300.0, 302.0)
    assert ok is True
    assert error_mm == pytest.approx(2.0, abs=1e-9)


def test_resting_check_fails_past_the_tolerance():
    ok, error_mm = checklist.resting_check(
        300.0, 300.0 + checklist.RESTING_HEIGHT_TOLERANCE_MM + 1.0
    )
    assert ok is False
    assert error_mm == pytest.approx(checklist.RESTING_HEIGHT_TOLERANCE_MM + 1.0, abs=1e-9)


def test_resting_check_fails_when_the_box_never_registered():
    ok, error_mm = checklist.resting_check(300.0, None)
    assert ok is False
    assert error_mm == math.inf


def test_fragment_component_frames_mm_against_the_real_vendored_fragment():
    """The real fragment file, not a hand-built fixture: every scenery
    component there parents straight to world, so item 1 has something to
    check against on a real machine."""
    import json

    fragment_path = _EXAMPLES_DIR.parent / "fragments" / "isaac-sim-palletizing.json"
    fragment = json.loads(fragment_path.read_text())
    frames = checklist.fragment_component_frames_mm(fragment)
    assert frames["pallet"] == {"x": 200.0, "y": 500.0, "z": 200.0}
    assert frames["caution-tape"] == {"x": 50.0, "y": 1250.0, "z": 0.0}
    # pallet-empty and tray-dock carry no frame at all in the vendored fragment
    assert "pallet-empty" not in frames
    assert "tray-dock" not in frames


def test_parse_args_accepts_phase_2():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--phase",
            "2",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.phase == 2


def test_parse_args_rejects_a_phase_outside_1_and_2():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            [
                "--address",
                "10.0.0.1",
                "--box-prop",
                "infeed_box",
                "--run-label",
                "cold",
                "--phase",
                "3",
                *_REQUIRED_POSE_ARGS,
            ]
        )


def test_parse_args_default_fragment_points_at_the_vendored_fragment():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.fragment.endswith("fragments/isaac-sim-palletizing.json")


def test_a_run_that_recorded_a_failed_pick_is_not_a_placement():
    """The run's state and the record's outcome are different claims. A run
    that finishes having failed its pick reports "complete" with a record whose
    outcome is "failed", and checking only the state passes such a run whenever
    a box is already sitting at the target, which is what a rerun leaves."""
    from isaac_module.sort_plan import OUTCOME_FAILED, OUTCOME_PLACED

    placed = {"box_prop": "infeed_box", "place_pose_mm": {"x": 700.0}, "outcome": OUTCOME_PLACED}
    failed = {"box_prop": "infeed_box", "place_pose_mm": {"x": 700.0}, "outcome": OUTCOME_FAILED}

    assert checklist.records_all_placed([placed]) is True
    assert checklist.records_all_placed([failed]) is False
    assert checklist.records_all_placed([placed, failed]) is False
    # a run that ended before recording anything never placed a box either
    assert checklist.records_all_placed([]) is False
    assert checklist.records_all_placed(None) is False
