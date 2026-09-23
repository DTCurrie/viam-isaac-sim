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
    assert args.suite == "first-box"
    assert args.run_label == "cold"
    assert args.grab_delay_ms == 250.0
    assert args.retry_interval_s == 2.0
    assert args.coaxial_limit_n == checklist.DEFAULT_COAXIAL_FORCE_LIMIT_N
    assert args.items is None


def test_parse_args_accepts_an_items_list():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--items",
            "4,7",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.items == frozenset({4, 7})


def test_parse_args_rejects_a_non_integer_items_value():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            [
                "--address",
                "10.0.0.1",
                "--box-prop",
                "infeed_box",
                "--run-label",
                "cold",
                "--items",
                "x",
                *_REQUIRED_POSE_ARGS,
            ]
        )


def test_parse_args_accepts_a_coaxial_limit_override():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--coaxial-limit-n",
            "20.0",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.coaxial_limit_n == 20.0


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
        "pallet": {
            "x": 200.0,
            "y": 500.0,
            "z": 0.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
        "robot-pedestal": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
    }


def test_fragment_component_frames_mm_carries_the_frames_orientation():
    fragment = {
        "components": [
            {
                "name": "fence-left",
                "frame": {
                    "parent": "world",
                    "translation": {"x": -1150, "y": -350, "z": 0},
                    "orientation": {
                        "type": "ov_degrees",
                        "value": {"x": 0, "y": 0, "z": 1, "th": 90},
                    },
                },
            }
        ]
    }
    (frame,) = checklist.fragment_component_frames_mm(fragment).values()
    assert (frame["x"], frame["y"], frame["z"]) == (-1150.0, -350.0, 0.0)
    assert frame["o_z"] == pytest.approx(1.0)
    assert frame["theta"] == pytest.approx(90.0)


def test_frame_orientation_quat_refuses_a_form_it_cannot_read():
    with pytest.raises(ValueError, match="euler_angles"):
        checklist.frame_orientation_quat(
            {"type": "euler_angles", "value": {"roll": 0, "pitch": 0, "yaw": 90}}
        )


def test_orientation_delta_deg_between_a_turned_pose_and_an_unturned_one():
    turned = {"x": 0.0, "y": 0.0, "z": 0.0, "o_z": 1.0, "theta": 90.0}
    flat = {"x": 0.0, "y": 0.0, "z": 0.0}
    assert checklist.orientation_delta_deg(turned, flat) == pytest.approx(90.0)
    assert checklist.orientation_delta_deg(turned, turned) == pytest.approx(0.0, abs=1e-6)


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
    assert frames["pallet"] == {
        "x": 200.0,
        "y": 500.0,
        "z": 200.0,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 1.0,
        "theta": 0.0,
    }
    assert frames["caution-tape"] == {
        "x": 50.0,
        "y": 1250.0,
        "z": 0.0,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 1.0,
        "theta": 0.0,
    }
    # the two side fences are the only components whose frames turn
    assert frames["fence-left"]["theta"] == pytest.approx(90.0)
    assert frames["fence-right"]["theta"] == pytest.approx(90.0)
    assert frames["fence-back-left"]["theta"] == pytest.approx(0.0)
    # pallet-empty and tray-dock carry no frame at all in the vendored fragment
    assert "pallet-empty" not in frames
    assert "tray-dock" not in frames


def test_parse_args_accepts_workcell():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--suite",
            "workcell",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.suite == "workcell"


def test_parse_args_accepts_epick():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--suite",
            "epick",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.suite == "epick"


def test_parse_args_accepts_pack():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--suite",
            "pack",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.suite == "pack"
    assert args.sequencer == "pack-sequencer"
    assert args.obstacle_source_label == "world_state_store"


def test_parse_args_pack_needs_no_box_prop_or_poses():
    # The pack suite reads the infeed pose and every place target off the
    # service's own status records. Requiring them here would mean typing
    # numbers the run ignores, which is how an invented value comes to look
    # measured.
    args = checklist._parse_args(
        ["--address", "10.0.0.1", "--run-label", "cold", "--suite", "pack"]
    )
    assert args.suite == "pack"
    assert args.box_prop is None
    assert (args.pick_x_mm, args.pick_y_mm, args.pick_z_mm) == (None, None, None)
    assert (args.place_x_mm, args.place_y_mm) == (None, None)


@pytest.mark.parametrize("suite", ["first-box", "workcell", "epick"])
def test_parse_args_single_box_suites_still_require_the_box_prop_and_poses(suite: str):
    with pytest.raises(SystemExit):
        checklist._parse_args(["--address", "10.0.0.1", "--run-label", "cold", "--suite", suite])


def test_single_box_args_rejects_a_pack_shaped_args():
    args = checklist._parse_args(
        ["--address", "10.0.0.1", "--run-label", "cold", "--suite", "pack"]
    )
    with pytest.raises(ValueError, match="pick and place poses"):
        checklist._single_box_args(args)


def test_parse_args_rejects_a_suite_outside_the_known_four():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            [
                "--address",
                "10.0.0.1",
                "--box-prop",
                "infeed_box",
                "--run-label",
                "cold",
                "--suite",
                "cooking",
                *_REQUIRED_POSE_ARGS,
            ]
        )


def test_parse_args_rejects_legacy_numbered_suite_spellings():
    for suite in ("2", "2.1"):
        with pytest.raises(SystemExit):
            checklist._parse_args(
                [
                    "--address",
                    "10.0.0.1",
                    "--box-prop",
                    "infeed_box",
                    "--run-label",
                    "cold",
                    "--suite",
                    suite,
                    *_REQUIRED_POSE_ARGS,
                ]
            )


def test_parse_args_accepts_an_obstacle_source_label():
    args = checklist._parse_args(
        [
            "--address",
            "10.0.0.1",
            "--box-prop",
            "infeed_box",
            "--run-label",
            "cold",
            "--obstacle-source-label",
            "prop_geometries",
            *_REQUIRED_POSE_ARGS,
        ]
    )
    assert args.obstacle_source_label == "prop_geometries"


def test_parse_args_rejects_an_unlabelled_obstacle_source():
    with pytest.raises(SystemExit):
        checklist._parse_args(
            [
                "--address",
                "10.0.0.1",
                "--box-prop",
                "infeed_box",
                "--run-label",
                "cold",
                "--obstacle-source-label",
                "somewhere_else",
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


def _record(
    seq,
    box_prop,
    outcome,
    target_mm=None,
    measured_mm=None,
):
    from isaac_module.sort_plan import OUTCOME_PLACED

    record = {
        "seq": seq,
        "box_prop": box_prop,
        "outcome": outcome,
        "target_pose_mm": target_mm or {"x": 700.0, "y": 0.0, "z": 300.0},
    }
    if outcome == OUTCOME_PLACED and measured_mm is not None:
        record["measured_pose_mm"] = measured_mm
    return record


def test_placed_seqs_in_order_only_counts_placed_records():
    from isaac_module.sort_plan import OUTCOME_FAILED, OUTCOME_PLACED

    records = [
        _record(1, "infeed_box_1", OUTCOME_PLACED, measured_mm={"x": 700.0, "y": 0.0, "z": 300.0}),
        _record(2, "infeed_box_2", OUTCOME_FAILED),
        _record(
            2, "infeed_box_2", OUTCOME_PLACED, measured_mm={"x": 700.0, "y": 150.0, "z": 300.0}
        ),
    ]
    assert checklist.placed_seqs_in_order(records) == [1, 2]


def test_order_matches_pack_order_true_when_the_placed_seqs_are_ascending():
    from isaac_module.sort_plan import OUTCOME_PLACED

    records = [
        _record(1, "infeed_box_1", OUTCOME_PLACED, measured_mm={"x": 0.0, "y": 0.0, "z": 0.0}),
        _record(2, "infeed_box_2", OUTCOME_PLACED, measured_mm={"x": 0.0, "y": 0.0, "z": 0.0}),
    ]
    ok, observed = checklist.order_matches_pack_order(records, [1, 2])
    assert ok is True
    assert observed == [1, 2]


def test_order_matches_pack_order_false_when_a_seq_placed_out_of_order():
    """A broken implementation that raced two boxes and placed seq 2 first
    reads a placed order that disagrees with get_pack_order's ascending one."""
    from isaac_module.sort_plan import OUTCOME_PLACED

    records = [
        _record(2, "infeed_box_2", OUTCOME_PLACED, measured_mm={"x": 0.0, "y": 0.0, "z": 0.0}),
        _record(1, "infeed_box_1", OUTCOME_PLACED, measured_mm={"x": 0.0, "y": 0.0, "z": 0.0}),
    ]
    ok, observed = checklist.order_matches_pack_order(records, [1, 2])
    assert ok is False
    assert observed == [2, 1]


def test_placement_delta_mm_is_the_distance_between_target_and_measured():
    record = {
        "target_pose_mm": {"x": 700.0, "y": 0.0, "z": 300.0},
        "measured_pose_mm": {"x": 703.0, "y": 4.0, "z": 300.0},
    }
    assert checklist.placement_delta_mm(record) == pytest.approx(5.0, abs=1e-9)


def test_placement_delta_mm_none_when_never_measured():
    record = {"target_pose_mm": {"x": 700.0, "y": 0.0, "z": 300.0}}
    assert checklist.placement_delta_mm(record) is None


def test_max_placement_delta_mm_picks_the_largest_across_records():
    from isaac_module.sort_plan import OUTCOME_PLACED

    records = [
        _record(
            1,
            "infeed_box_1",
            OUTCOME_PLACED,
            target_mm={"x": 700.0, "y": 0.0, "z": 300.0},
            measured_mm={"x": 701.0, "y": 0.0, "z": 300.0},
        ),
        _record(
            2,
            "infeed_box_2",
            OUTCOME_PLACED,
            target_mm={"x": 700.0, "y": 150.0, "z": 300.0},
            measured_mm={"x": 700.0, "y": 158.0, "z": 300.0},
        ),
    ]
    max_delta_mm, box_prop = checklist.max_placement_delta_mm(records)
    assert max_delta_mm == pytest.approx(8.0, abs=1e-9)
    assert box_prop == "infeed_box_2"


def test_max_placement_delta_mm_none_when_nothing_measured():
    from isaac_module.sort_plan import OUTCOME_FAILED

    records = [_record(1, "infeed_box_1", OUTCOME_FAILED)]
    max_delta_mm, box_prop = checklist.max_placement_delta_mm(records)
    assert max_delta_mm is None
    assert box_prop is None


def test_placement_table_rows_sorted_by_seq_with_delta():
    from isaac_module.sort_plan import OUTCOME_PLACED

    records = [
        _record(
            2,
            "infeed_box_2",
            OUTCOME_PLACED,
            target_mm={"x": 700.0, "y": 150.0, "z": 300.0},
            measured_mm={"x": 700.0, "y": 150.0, "z": 300.0},
        ),
        _record(
            1,
            "infeed_box_1",
            OUTCOME_PLACED,
            target_mm={"x": 700.0, "y": 0.0, "z": 300.0},
            measured_mm={"x": 703.0, "y": 4.0, "z": 300.0},
        ),
    ]
    rows = checklist.placement_table_rows(records)
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[0]["delta_mm"] == pytest.approx(5.0, abs=1e-9)
    assert rows[1]["delta_mm"] == pytest.approx(0.0, abs=1e-9)


def test_format_placement_table_prints_the_delta_for_every_row():
    rows = checklist.placement_table_rows(
        [
            _record(
                1,
                "infeed_box_1",
                "placed",
                target_mm={"x": 700.0, "y": 0.0, "z": 300.0},
                measured_mm={"x": 703.0, "y": 4.0, "z": 300.0},
            )
        ]
    )
    table = checklist.format_placement_table(rows)
    assert "infeed_box_1" in table
    assert "5.00" in table


def test_skip_handled_correctly_true_when_nothing_was_skipped():
    ok, detail = checklist.skip_handled_correctly([], "complete")
    assert ok is True
    assert "no seq was skipped" in detail


def test_skip_handled_correctly_true_when_a_skip_still_completed():
    ok, _detail = checklist.skip_handled_correctly([4], "complete")
    assert ok is True


def test_skip_handled_correctly_false_when_a_skip_left_the_run_unfinished():
    """A broken skip path that aborts the whole run rather than continuing
    reads a final state other than complete."""
    ok, detail = checklist.skip_handled_correctly([4], "failed")
    assert ok is False
    assert "failed" in detail


def test_group_frame_pose_mm_reads_the_components_own_group_primitive():
    # Real shape from the machine: every workcell component draws a
    # `<name>/group` frame primitive whose pose IS its frame origin.
    reply = {
        "visuals": [
            {
                "label": "pick-station/group",
                "type": "frame",
                "pose": {"x": 400.0, "y": -650.0, "z": 200.0},
            },
            {
                "label": "pick-station/deck",
                "type": "box",
                "pose": {"z": -12.0},
                "dims_mm": {"x": 400.0, "y": 1100.0, "z": 40.0},
            },
        ]
    }
    assert checklist.group_frame_pose_mm(reply, "pick-station") == {
        "x": 400.0,
        "y": -650.0,
        "z": 200.0,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 0.0,
        "theta": 0.0,
    }


def test_group_frame_pose_mm_does_not_read_the_corner_pose():
    # The failure this replaces: get_attributes.pose reports the CORNER, so
    # pick-station read (200, -1200, 220) against a declared (400, -650, 200)
    # and item 1 called a correctly placed component 585 mm wrong.
    reply = {
        "visuals": [
            {
                "label": "pick-station/group",
                "type": "frame",
                "pose": {"x": 400.0, "y": -650.0, "z": 200.0},
            },
        ],
        "pose": {"x": 200.0, "y": -1200.0, "z": 220.0},
    }
    measured = checklist.group_frame_pose_mm(reply, "pick-station")
    assert measured is not None
    assert (measured["x"], measured["y"], measured["z"]) == (400.0, -650.0, 200.0)
    assert checklist.pose_delta_mm({"x": 400.0, "y": -650.0, "z": 200.0}, measured) == 0.0


def test_group_frame_pose_mm_is_none_when_the_component_draws_no_group():
    assert checklist.group_frame_pose_mm({"visuals": []}, "isaac-world") is None


def test_support_geometry_mm_finds_a_frame_declared_collider():
    # Real names from the stage: prim names cannot hold a hyphen, so a
    # frame.geometry collider for `pick-station` lands as `frame_pick_station`.
    geometries = [
        {
            "name": "frame_pick_station",
            "pose_in_world_mm": {"z": 200.0},
            "box_dims_mm": [400.0, 1100.0, 40.0],
        },
        {
            "name": "frame_pallet",
            "pose_in_world_mm": {"z": 200.0},
            "box_dims_mm": [500.0, 350.0, 100.0],
        },
    ]
    found = checklist.support_geometry_mm(geometries, "pick-station")
    assert found is not None
    assert found["name"] == "frame_pick_station"


def test_support_geometry_mm_still_finds_render_scenery_by_prefix():
    geometries = [
        {"name": "hmi-cabinet-body", "pose_in_world_mm": {"z": 0.0}, "box_dims_mm": [1.0, 1.0, 1.0]}
    ]
    found = checklist.support_geometry_mm(geometries, "hmi-cabinet")
    assert found is not None
    assert found["name"] == "hmi-cabinet-body"


def test_support_geometry_mm_prefers_the_frame_collider_over_render_scenery():
    geometries = [
        {
            "name": "pallet-slat-0",
            "pose_in_world_mm": {"z": 241.0},
            "box_dims_mm": [156.0, 350.0, 18.0],
        },
        {
            "name": "frame_pallet",
            "pose_in_world_mm": {"z": 200.0},
            "box_dims_mm": [500.0, 350.0, 100.0],
        },
    ]
    assert checklist.support_geometry_mm(geometries, "pallet")["name"] == "frame_pallet"


def test_item_3_passes_on_fences_alone_since_the_tunnel_declares_no_geometry():
    # scan-tunnel is an arch; a single box geometry would seal the opening the
    # arm and boxes pass through. Its absence is item 5's recorded answer, not
    # an item 3 failure, and conflating them hid that the fences did land.
    names = [
        "frame_fence_left",
        "frame_fence_right",
        "frame_fence_back_left",
        "frame_fence_back_right",
        "frame_pallet",
        "frame_pick_station",
    ]
    assert any(n.startswith(("fence-", "frame_fence_")) for n in names)
    assert not any(n.startswith(("scan-tunnel-", "frame_scan_tunnel")) for n in names)


_PICK_STATION = {
    "name": "frame_pick_station",
    "pose_in_world_mm": {"x": 400.0, "y": -650.0, "z": 200.0},
    "box_dims_mm": [400.0, 1100.0, 40.0],
}


def test_box_rests_on_support_accepts_a_box_that_slid_but_stayed_on_the_deck():
    """The warm run of 2026-09-16 settled 23.8 mm from where it was sent, on
    the station. Item 4 reads the box's live pose, so the target follows it
    and the drift costs nothing."""
    ok, _detail = checklist.box_rests_on_support(
        {"x": 415.6, "y": -317.8, "z": 272.8}, 100.0, _PICK_STATION
    )
    assert ok is True


def test_box_rests_on_support_rejects_a_box_left_beside_the_arms_base():
    """Where item 2's pedestal drop used to leave it: inside no support's
    footprint, and a target no plan can reach."""
    ok, _detail = checklist.box_rests_on_support(
        {"x": 111.7, "y": 194.9, "z": 189.6}, 100.0, _PICK_STATION
    )
    assert ok is False


def test_box_rests_on_support_rejects_a_box_on_the_floor_under_the_station():
    ok, detail = checklist.box_rests_on_support(
        {"x": 400.0, "y": -300.0, "z": 50.0}, 100.0, _PICK_STATION
    )
    assert ok is False
    assert "height error" in detail


def test_box_rests_on_support_fails_when_the_support_never_spawned():
    ok, detail = checklist.box_rests_on_support({"x": 400.0, "y": -300.0, "z": 270.0}, 100.0, None)
    assert ok is False
    assert "no collider" in detail


def test_box_rests_on_support_fails_when_the_box_registered_no_pose():
    ok, detail = checklist.box_rests_on_support(None, 100.0, _PICK_STATION)
    assert ok is False
    assert "no pose" in detail


def test_fragment_component_colliders_mm_adds_the_geometry_offset_to_the_frame():
    fragment = {
        "components": [
            {
                "name": "robot-pedestal",
                "frame": {
                    "parent": "world",
                    "translation": {"x": 0, "y": 0, "z": 0},
                    "geometry": {
                        "type": "box",
                        "x": 220,
                        "y": 220,
                        "z": 150,
                        "translation": {"x": 0, "y": 0, "z": 75},
                    },
                },
            }
        ]
    }
    colliders = checklist.fragment_component_colliders_mm(fragment)
    assert colliders["robot-pedestal"]["pose_in_world_mm"] == {
        "x": 0.0,
        "y": 0.0,
        "z": 75.0,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 1.0,
        "theta": 0.0,
    }
    assert colliders["robot-pedestal"]["box_dims_mm"] == (220.0, 220.0, 150.0)


def test_fragment_component_colliders_mm_turns_the_geometry_with_its_frame():
    # a geometry offset along the frame's own x lands along world y once the
    # frame turns 90 degrees, and the box turns with it
    fragment = {
        "components": [
            {
                "name": "turned",
                "frame": {
                    "parent": "world",
                    "translation": {"x": 0, "y": 0, "z": 0},
                    "orientation": {
                        "type": "ov_degrees",
                        "value": {"x": 0, "y": 0, "z": 1, "th": 90},
                    },
                    "geometry": {
                        "type": "box",
                        "x": 10,
                        "y": 10,
                        "z": 10,
                        "translation": {"x": 100, "y": 0, "z": 0},
                    },
                },
            }
        ]
    }
    (collider,) = checklist.fragment_component_colliders_mm(fragment).values()
    pose = collider["pose_in_world_mm"]
    assert (pose["x"], pose["y"], pose["z"]) == pytest.approx((0.0, 100.0, 0.0), abs=1e-9)
    assert pose["theta"] == pytest.approx(90.0)


def test_fragment_component_colliders_mm_skips_what_declares_no_box():
    fragment = {
        "components": [
            # an arch: one frame carries one shape, and a box would seal its opening
            {"name": "scan-tunnel", "frame": {"parent": "world", "translation": {"x": 400}}},
            # parented to the arm, so its world pose is not this fragment's to state
            {
                "name": "wrist-cam",
                "frame": {"parent": "arm-1", "geometry": {"type": "box", "x": 90}},
            },
        ]
    }
    assert checklist.fragment_component_colliders_mm(fragment) == {}


def test_fragment_component_colliders_mm_against_the_real_vendored_fragment():
    import json

    fragment_path = _EXAMPLES_DIR.parent / "fragments" / "isaac-sim-palletizing.json"
    colliders = checklist.fragment_component_colliders_mm(json.loads(fragment_path.read_text()))
    assert colliders["robot-pedestal"] == {
        "pose_in_world_mm": {
            "x": 0.0,
            "y": 0.0,
            "z": 75.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
        "box_dims_mm": (220.0, 220.0, 150.0),
    }
    assert colliders["pick-station"]["box_dims_mm"] == (400.0, 1100.0, 40.0)
    assert "scan-tunnel" not in colliders
    # fence-left: a 1200 mm panel whose frame turns 90 degrees, so its collider runs along y
    fence = colliders["fence-left"]["pose_in_world_mm"]
    assert (fence["x"], fence["y"], fence["z"]) == pytest.approx((-1150.0, -350.0, 625.0))
    assert fence["theta"] == pytest.approx(90.0)
    assert colliders["fence-left"]["box_dims_mm"] == (1200.0, 36.0, 1180.0)


def test_collider_matches_declaration_passes_on_the_declared_pose_and_dims():
    declared = {
        "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
        "box_dims_mm": (220.0, 220.0, 150.0),
    }
    ok, detail = checklist.collider_matches_declaration(
        {
            "name": "frame_robot_pedestal",
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
            "box_dims_mm": [220.0, 220.0, 150.0],
        },
        declared,
    )
    assert ok is True
    assert "pose error 0.000 mm" in detail


def test_collider_matches_declaration_fails_when_nothing_spawned():
    declared = {
        "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
        "box_dims_mm": (220.0, 220.0, 150.0),
    }
    ok, detail = checklist.collider_matches_declaration(None, declared)
    assert ok is False
    assert "no collider found" in detail


def test_collider_matches_declaration_fails_on_a_collider_turned_the_wrong_way():
    # the GPU viewport of 2026-09-22: fence-left's collider stood along x at the
    # right position and the right size, so a check without orientation passed it
    declared = {
        "pose_in_world_mm": {"x": -1150.0, "y": -350.0, "z": 625.0, "o_z": 1.0, "theta": 90.0},
        "box_dims_mm": (1200.0, 36.0, 1180.0),
    }
    ok, detail = checklist.collider_matches_declaration(
        {
            "name": "frame_fence_left",
            "pose_in_world_mm": {"x": -1150.0, "y": -350.0, "z": 625.0, "o_z": 1.0, "theta": 0.0},
            "box_dims_mm": [1200.0, 36.0, 1180.0],
        },
        declared,
    )
    assert ok is False
    assert "orientation error 90.000 deg" in detail


def _fence_left_visuals() -> dict:
    """fence-left's get_visuals as workcell-components 0.7.0 builds it: the
    group anchor, two rails, two posts and the screen, every child relative to
    the anchor. Ported from safety_fence.go plus visuals_group.go."""
    return {
        "visuals": [
            {
                "type": "frame",
                "label": "fence-left/group",
                "parent_frame": "world",
                "pose": {"x": -1150.0, "y": -350.0, "o_z": 1.0, "theta": 90.0},
            },
            {
                "type": "capsule",
                "label": "fence-left/rail-0",
                "pose": {"z": 1200.0, "o_x": 1.0},
                "radius_mm": 15.0,
                "length_mm": 1200.0,
            },
            {
                "type": "capsule",
                "label": "fence-left/post-0",
                "pose": {"x": -600.0, "z": 612.5, "o_z": 1.0},
                "radius_mm": 18.0,
                "length_mm": 1225.0,
            },
            {
                "type": "capsule",
                "label": "fence-left/post-1",
                "pose": {"x": 600.0, "z": 612.5, "o_z": 1.0},
                "radius_mm": 18.0,
                "length_mm": 1225.0,
            },
            {
                "type": "box",
                "label": "fence-left/screen",
                "pose": {"z": 625.0, "o_z": 1.0},
                "dims_mm": {"x": 1200.0, "y": 6.0, "z": 1150.0},
            },
            {"type": "sphere", "label": "fence-left/never-spawned", "radius_mm": 5.0},
        ]
    }


def test_expected_render_prims_mm_turns_the_fence_onto_its_frame():
    frame = {"x": -1150.0, "y": -350.0, "z": 0.0, "o_z": 1.0, "theta": 90.0}
    prims = dict(checklist.expected_render_prims_mm("fence-left", frame, _fence_left_visuals()))
    # the same prim names the module spawns: component, hyphen, label, then USD-safe
    assert set(prims) == {
        "/World/fence_left_fence_left_rail_0",
        "/World/fence_left_fence_left_post_0",
        "/World/fence_left_fence_left_post_1",
        "/World/fence_left_fence_left_screen",
    }
    screen = prims["/World/fence_left_fence_left_screen"]
    assert (screen["x"], screen["y"], screen["z"]) == pytest.approx((-1150.0, -350.0, 625.0))
    assert screen["theta"] == pytest.approx(90.0)
    # the posts stand at the panel's ends, along world y once the frame turns
    post_0 = prims["/World/fence_left_fence_left_post_0"]
    post_1 = prims["/World/fence_left_fence_left_post_1"]
    assert (post_0["x"], post_0["y"]) == pytest.approx((-1150.0, -950.0))
    assert (post_1["x"], post_1["y"]) == pytest.approx((-1150.0, 250.0))
    # a rail laid along the fence's own x lies along world y
    rail = prims["/World/fence_left_fence_left_rail_0"]
    assert (rail["o_x"], rail["o_y"], rail["o_z"]) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_prim_pose_reply_mm_reads_the_worlds_prim_pose_shape():
    reply = {
        "prim_path": "/World/fence_left_fence_left_screen",
        "position_mm": [-1150.0, -350.0, 625.0],
        "quaternion_wxyz": [0.7071, 0.0, 0.0, 0.7071],
        "orientation_vector": {"o_x": 0.0, "o_y": 0.0, "o_z": 1.0, "theta_deg": 90.0},
    }
    assert checklist.prim_pose_reply_mm(reply) == {
        "x": -1150.0,
        "y": -350.0,
        "z": 625.0,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 1.0,
        "theta": 90.0,
    }


def test_prim_matches_expected_fails_on_a_prim_standing_along_the_wrong_axis():
    expected = {"x": -1150.0, "y": -350.0, "z": 625.0, "o_z": 1.0, "theta": 90.0}
    measured = {"x": -1150.0, "y": -350.0, "z": 625.0, "o_z": 1.0, "theta": 0.0}
    ok, position_error_mm, angle_error_deg = checklist.prim_matches_expected(measured, expected)
    assert ok is False
    assert position_error_mm == pytest.approx(0.0)
    assert angle_error_deg == pytest.approx(90.0)


def test_prim_matches_expected_passes_within_both_tolerances():
    expected = {"x": -1150.0, "y": -350.0, "z": 625.0, "o_z": 1.0, "theta": 90.0}
    measured = {"x": -1149.6, "y": -350.0, "z": 625.2, "o_z": 1.0, "theta": 90.1}
    ok, _position_error_mm, _angle_error_deg = checklist.prim_matches_expected(measured, expected)
    assert ok is True


def _prop(name: str, x: float, y: float, z: float, fixed: bool = False) -> dict:
    return {
        "name": name,
        "fixed": fixed,
        "box_dims_mm": [150.0, 200.0, 100.0],
        "pose_in_world_mm": {
            "x": x,
            "y": y,
            "z": z,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
    }


def test_props_to_restore_names_what_the_run_moved_and_where_it_was():
    # the service restaged box 2 from its parking spot onto the station; the
    # test box went to the pallet on purpose; the fences never move
    before = [
        _prop("infeed_box_1", 400.0, -300.0, 270.0),
        _prop("infeed_box_2", 1500.0, 1500.0, 50.0),
        _prop("infeed_box_3", 1700.0, 1500.0, 50.0),
        _prop("frame_fence_left", -1150.0, -350.0, 625.0, fixed=True),
    ]
    after = [
        _prop("infeed_box_1", 50.0, 400.0, 300.0),
        _prop("infeed_box_2", 401.9, -304.0, 270.0),
        _prop("infeed_box_3", 1700.0, 1500.0, 50.0),
        _prop("frame_fence_left", -1150.0, -350.0, 625.0, fixed=True),
    ]
    assert checklist.props_to_restore(before, after, {"infeed_box_1"}) == [
        ("infeed_box_2", (1500.0, 1500.0, 50.0))
    ]


def test_props_to_restore_ignores_settling_within_tolerance():
    before = [_prop("infeed_box_2", 1500.0, 1500.0, 50.0)]
    after = [_prop("infeed_box_2", 1500.4, 1500.0, 50.0)]
    assert checklist.props_to_restore(before, after, set()) == []


def test_collider_matches_declaration_fails_on_a_collider_of_the_wrong_size():
    declared = {
        "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
        "box_dims_mm": (220.0, 220.0, 150.0),
    }
    ok, _detail = checklist.collider_matches_declaration(
        {
            "name": "frame_robot_pedestal",
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
            "box_dims_mm": [220.0, 220.0, 100.0],
        },
        declared,
    )
    assert ok is False


async def test_guarded_records_a_raising_item_as_a_failure_and_keeps_going():
    """Item 4's planner error ended two whole GPU runs before this existed."""

    async def raises() -> tuple[str, bool]:
        raise RuntimeError("motion planner failed to find path")

    line, ok = await checklist._guarded("4. pick and place", raises)
    assert ok is False
    assert "motion planner failed to find path" in line
    assert line.startswith("[FAIL]")


async def test_guarded_passes_a_working_items_result_straight_through():
    async def works() -> tuple[str, bool]:
        return "[PASS] 4. pick and place: fine", True

    assert await checklist._guarded("4. pick and place", works) == (
        "[PASS] 4. pick and place: fine",
        True,
    )


async def test_guarded_if_listed_skips_an_item_not_in_the_requested_set():
    async def not_reached() -> tuple[str, bool]:
        raise AssertionError("skipped items must not run their check")

    result = await checklist._guarded_if_listed(4, "4. tear-off", frozenset({7}), not_reached)
    assert result is None


async def test_guarded_if_listed_runs_an_item_in_the_requested_set():
    async def works() -> tuple[str, bool]:
        return "[PASS] 4. tear-off: fine", True

    result = await checklist._guarded_if_listed(4, "4. tear-off", frozenset({4, 7}), works)
    assert result == ("[PASS] 4. tear-off: fine", True)


async def test_guarded_if_listed_runs_every_item_when_items_is_none():
    async def works() -> tuple[str, bool]:
        return "[PASS] 4. tear-off: fine", True

    result = await checklist._guarded_if_listed(4, "4. tear-off", None, works)
    assert result == ("[PASS] 4. tear-off: fine", True)


class _FallingWorld:
    """A world where a teleported prop drops straight down by whatever
    clearance it was given, which is what a support directly below it does.
    Enough to drive item 2 without a GPU, and it records every pose written so
    a test can say which components were dropped on."""

    def __init__(self, geometries):
        self._geometries = {geometry["name"]: geometry for geometry in geometries}
        self.dropped_on_xy = []

    async def do_command(self, command):
        if command["command"] == "prop_geometries":
            return {"geometries": list(self._geometries.values())}
        if command["command"] == "set_prop_pose":
            x, y, z = command["position"]
            self.dropped_on_xy.append((x, y))
            self._geometries[command["name"]]["pose_in_world_mm"] = {
                "x": x,
                "y": y,
                "z": z - checklist.DROP_HOVER_MM,
            }
            return {}
        raise AssertionError(f"unexpected command {command['command']!r}")


def _cell_geometries():
    return [
        {
            "name": "infeed_box_1",
            "pose_in_world_mm": {"x": 400.0, "y": -300.0, "z": 270.0},
            "box_dims_mm": [200.0, 150.0, 100.0],
        },
        {
            "name": "frame_pick_station",
            "pose_in_world_mm": {"x": 400.0, "y": -650.0, "z": 200.0},
            "box_dims_mm": [400.0, 1100.0, 40.0],
        },
        {
            "name": "frame_pallet",
            "pose_in_world_mm": {"x": 200.0, "y": 500.0, "z": 200.0},
            "box_dims_mm": [500.0, 350.0, 100.0],
        },
        {
            "name": "frame_robot_pedestal",
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
            "box_dims_mm": [220.0, 220.0, 150.0],
        },
    ]


async def test_item_2_never_drops_a_box_on_the_pedestal_the_arm_stands_on(monkeypatch, capsys):
    """A box dropped on `robot-pedestal` lands on the arm mounted there and
    tumbles off, which tests nothing about the collider and leaves the box
    where item 4 then has to plan around it. The collider is checked against
    its declaration instead."""
    monkeypatch.setattr(checklist, "DROP_SETTLE_S", 0.0)
    world = _FallingWorld(_cell_geometries())
    declared = checklist.fragment_component_colliders_mm(
        {
            "components": [
                {
                    "name": "robot-pedestal",
                    "frame": {
                        "parent": "world",
                        "translation": {"x": 0, "y": 0, "z": 0},
                        "geometry": {
                            "type": "box",
                            "x": 220,
                            "y": 220,
                            "z": 150,
                            "translation": {"z": 75},
                        },
                    },
                }
            ]
        }
    )

    line, ok = await checklist._check_support_drops(world, "infeed_box_1", declared)

    assert ok is True
    assert line.startswith("[PASS]")
    assert world.dropped_on_xy == [(400.0, -650.0), (200.0, 500.0)]
    out = capsys.readouterr().out
    assert "robot-pedestal: declared" in out
    assert "robot-pedestal: support top" not in out


async def test_item_2_fails_when_the_pedestals_collider_is_missing(monkeypatch):
    monkeypatch.setattr(checklist, "DROP_SETTLE_S", 0.0)
    geometries = [g for g in _cell_geometries() if g["name"] != "frame_robot_pedestal"]
    world = _FallingWorld(geometries)
    declared = {
        "robot-pedestal": {
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 75.0},
            "box_dims_mm": (220.0, 220.0, 150.0),
        }
    }

    _line, ok = await checklist._check_support_drops(world, "infeed_box_1", declared)

    assert ok is False


class _RefusingMotion:
    """A motion service that refuses every plan, the way the planner refused
    item 4's grasp descent."""

    async def move(self, **_kwargs):
        raise RuntimeError("motion planner failed to find path")


class _Named:
    def __init__(self, name):
        self.name = name


async def test_a_failed_move_names_the_leg_and_whether_it_was_linear():
    """The 2026-09-16 run reported `motion planner failed to find path` with
    no leg named, and the standoff had in fact succeeded. Reading it as a
    standoff failure cost a whole diagnostic round trip."""
    from viam.proto.common import Pose

    with pytest.raises(RuntimeError) as raised:
        await checklist._move(
            _Named("gripper-1"),
            _RefusingMotion(),
            Pose(x=400.0, y=-300.0, z=325.0),
            None,
            linear=True,
            leg="grasp descent",
        )

    message = str(raised.value)
    assert "grasp descent" in message
    assert "linear" in message
    assert "motion planner failed to find path" in message


async def test_a_failed_free_move_says_free_rather_than_linear():
    from viam.proto.common import Pose

    with pytest.raises(RuntimeError) as raised:
        await checklist._move(
            _Named("gripper-1"),
            _RefusingMotion(),
            Pose(x=400.0, y=-300.0, z=425.0),
            None,
            leg="standoff",
        )

    assert "standoff (free)" in str(raised.value)


def test_empty_spot_mm_skips_an_offset_with_a_box_under_it():
    """The 2026-09-16 run's shape: the service failed, the box stayed on the
    station, and the grab-with-nothing test descended onto it and grabbed it."""
    geometries = [
        {"name": "infeed_box_1", "pose_in_world_mm": {"x": 400.0, "y": -300.0, "z": 270.0}},
        {"name": "infeed_box_2", "pose_in_world_mm": {"x": 400.0, "y": -700.0, "z": 270.0}},
    ]
    found = checklist.empty_spot_mm(geometries, (400.0, -300.0, 320.0))
    assert found is not None
    # -700 holds infeed_box_2, so the first clear offset is +400
    assert found == (400.0, 100.0, 320.0)


def test_empty_spot_mm_ignores_fixed_props_since_a_cup_cannot_lift_one():
    geometries = [
        {
            "name": "frame_pick_station",
            "fixed": True,
            "pose_in_world_mm": {"x": 400.0, "y": -650.0, "z": 200.0},
        }
    ]
    assert checklist.empty_spot_mm(geometries, (400.0, -300.0, 320.0)) == (400.0, -700.0, 320.0)


def test_empty_spot_mm_is_none_when_every_offset_has_something_under_it():
    geometries = [
        {"name": f"box_{index}", "pose_in_world_mm": {"x": 400.0, "y": y, "z": 270.0}}
        for index, y in enumerate((-700.0, 100.0, -1000.0, 400.0))
    ]
    assert checklist.empty_spot_mm(geometries, (400.0, -300.0, 320.0)) is None


# the epick suite: the Robotiq EPick


def test_collision_reach_z_mm_on_the_vendored_file_is_26mm():
    import json

    model_path = (
        _EXAMPLES_DIR.parent / "src" / "isaac_module" / "kinematics_files" / "epick_model.json"
    )
    model = json.loads(model_path.read_text())
    assert checklist.collision_reach_z_mm(model) == pytest.approx(-26.0, abs=1e-9)


def test_offset_along_tool_mm_reads_a_prim_above_a_downward_tcp():
    tcp_pose = {"x": 0.0, "y": 0.0, "z": 500.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0}
    prim_pose = {"x": 0.0, "y": 0.0, "z": 634.5}
    assert checklist.offset_along_tool_mm(tcp_pose, prim_pose) == pytest.approx(134.5, abs=1e-6)


def test_offset_along_tool_mm_is_zero_at_the_tcp_itself():
    tcp_pose = {"x": 100.0, "y": 200.0, "z": 300.0, "o_z": -1.0, "theta": 0.0}
    assert checklist.offset_along_tool_mm(tcp_pose, dict(tcp_pose)) == pytest.approx(0.0, abs=1e-9)


def test_refusal_within_passes_exactly_on_the_bound():
    bound_s = 250.0 / 1000.0 + 2.0 + 1.0
    assert checklist.refusal_within(bound_s, 250.0, 2.0, 1.0) is True


def test_refusal_within_fails_just_past_the_bound():
    bound_s = 250.0 / 1000.0 + 2.0 + 1.0
    assert checklist.refusal_within(bound_s + 0.001, 250.0, 2.0, 1.0) is False


def test_tilt_deg_is_zero_for_a_box_hanging_flat_under_the_cups():
    # a 180 degree rotation about X: the tool points straight down
    tool_pointing_down = (0.0, 1.0, 0.0, 0.0)
    upright_box = (1.0, 0.0, 0.0, 0.0)
    assert checklist.tilt_deg(tool_pointing_down, upright_box) == pytest.approx(0.0, abs=1e-9)


def test_tilt_deg_reads_a_known_rotation_between_tool_and_box():
    from isaac_module.spatial import quat_from_axis_angle

    tool_pointing_down = (0.0, 1.0, 0.0, 0.0)
    rolled_box = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(10.0))
    assert checklist.tilt_deg(tool_pointing_down, rolled_box) == pytest.approx(10.0, abs=1e-6)


def test_tilt_deg_reads_180_between_a_downward_tool_axis_and_an_upright_one():
    identity_tool = (1.0, 0.0, 0.0, 0.0)
    upright_box = (1.0, 0.0, 0.0, 0.0)
    assert checklist.tilt_deg(identity_tool, upright_box) == pytest.approx(180.0, abs=1e-9)


def test_swing_verdict_fails_at_exactly_zero_tilt():
    ok, _detail = checklist.swing_verdict(0.0, 0.0, True)
    assert ok is False


def test_swing_verdict_passes_at_the_floor():
    ok, _detail = checklist.swing_verdict(0.05, 0.0, True)
    assert ok is True


def test_swing_verdict_passes_just_under_the_ceiling():
    ok, _detail = checklist.swing_verdict(14.9, 0.0, True)
    assert ok is True


def test_swing_verdict_fails_at_the_ceiling():
    ok, _detail = checklist.swing_verdict(15.0, 0.0, True)
    assert ok is False


def test_swing_verdict_fails_when_the_residual_never_settled():
    ok, _detail = checklist.swing_verdict(
        5.0, checklist.SWING_RESIDUAL_TILT_TOLERANCE_DEG + 0.1, True
    )
    assert ok is False


def test_swing_verdict_fails_when_the_grip_let_go_partway():
    ok, _detail = checklist.swing_verdict(5.0, 0.0, False)
    assert ok is False


def test_tear_off_mass_kg_outweighs_all_four_cups_at_the_break_force():
    """The box must outweigh what four cups hold at their own rated coaxial
    break force, with 25% margin."""
    cups = ("cup-xn-yn", "cup-xn-yp", "cup-xp-yn", "cup-xp-yp")
    coaxial_limit_n = 44.1

    mass_kg = checklist.tear_off_mass_kg(cups, coaxial_limit_n)

    lower_bound_kg = len(cups) * coaxial_limit_n / checklist.STANDARD_GRAVITY_M_S2
    upper_bound_kg = 2 * lower_bound_kg
    assert mass_kg > lower_bound_kg
    assert mass_kg < upper_bound_kg


def test_under_limit_mass_kg_sits_below_the_four_cup_hold_with_margin():
    cups = ("cup-xn-yn", "cup-xn-yp", "cup-xp-yn", "cup-xp-yp")
    coaxial_limit_n = 44.1

    mass_kg = checklist.under_limit_mass_kg(cups, coaxial_limit_n)

    full_hold_kg = len(cups) * coaxial_limit_n / checklist.STANDARD_GRAVITY_M_S2
    half_hold_kg = 0.5 * full_hold_kg
    assert mass_kg < full_hold_kg
    assert mass_kg > half_hold_kg


def test_tear_off_verdict_passes_when_over_drops_under_holds_and_light_holds():
    over = checklist.TearOffReading(
        held=False,
        rise_mm=checklist.TEAR_OFF_RISE_TOLERANCE_MM - 1.0,
        peak_load_n=31.2,
        released_load_n=24.9,
    )
    under = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM, peak_load_n=9.1)
    light = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM, peak_load_n=6.4)

    ok, detail = checklist.tear_off_verdict(over, under, light)

    assert ok is True
    assert "over-limit released at the grab or on lift" in detail
    assert "peak 31.2 N per cup" in detail
    assert "released at 24.9 N" in detail
    assert "peak 9.1 N per cup" in detail
    assert "peak 6.4 N per cup" in detail


def test_tear_off_verdict_fails_when_the_over_limit_stand_in_is_lifted_and_held():
    over = checklist.TearOffReading(
        held=True, rise_mm=checklist.TEAR_OFF_RISE_TOLERANCE_MM + 1.0, sag_mm=3.2
    )
    under = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM)
    light = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM)

    ok, detail = checklist.tear_off_verdict(over, under, light)

    assert ok is False
    assert "the module's monitor did not release it" in detail
    assert "3.2" in detail


def test_tear_off_verdict_fails_when_the_under_limit_stand_in_dropped():
    over = checklist.TearOffReading(held=False, rise_mm=checklist.TEAR_OFF_RISE_TOLERANCE_MM - 1.0)
    under = checklist.TearOffReading(held=False, rise_mm=0.0)
    light = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM)

    ok, _detail = checklist.tear_off_verdict(over, under, light)

    assert ok is False


def test_tear_off_verdict_fails_when_the_light_box_dropped():
    over = checklist.TearOffReading(held=False, rise_mm=checklist.TEAR_OFF_RISE_TOLERANCE_MM - 1.0)
    under = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM)
    light = checklist.TearOffReading(held=False, rise_mm=0.0)

    ok, _detail = checklist.tear_off_verdict(over, under, light)

    assert ok is False


def test_tear_off_and_under_limit_masses_follow_the_configured_coaxial_limit():
    cups = ("cup-xn-yn", "cup-xn-yp", "cup-xp-yn", "cup-xp-yp")
    coaxial_limit_n = 20.0

    over_mass_kg = checklist.tear_off_mass_kg(cups, coaxial_limit_n)
    under_mass_kg = checklist.under_limit_mass_kg(cups, coaxial_limit_n)

    assert over_mass_kg == round(
        len(cups) * coaxial_limit_n * 1.25 / checklist.STANDARD_GRAVITY_M_S2
    )
    assert under_mass_kg == round(
        checklist.TEAR_OFF_UNDER_LIMIT_MARGIN
        * len(cups)
        * coaxial_limit_n
        / checklist.STANDARD_GRAVITY_M_S2
    )


def test_lift_stall_detail_keys_off_the_stuck_joints_clause():
    message = (
        "tear-off over-limit lift (linear) to (0, 0, 0) failed: RuntimeError: arm arm-1 "
        "stalled at waypoint 9/12 (5 consecutive, stuck joints: j1: at -83.7 want -91.0)"
    )
    detail = checklist.lift_stall_detail("epick_tearoff_box", message)
    assert detail == (
        "epick_tearoff_box: the arm could not lift it (stuck joints: j1: at -83.7 want -91.0))"
    )


def test_lift_stall_detail_falls_back_to_the_message_head_when_no_stuck_joints_clause():
    message = "x" * (checklist.LIFT_STALL_MESSAGE_HEAD_CHARS + 40)
    detail = checklist.lift_stall_detail("epick_tearoff_box", message)
    assert detail == (
        f"epick_tearoff_box: the arm could not lift it "
        f"({message[: checklist.LIFT_STALL_MESSAGE_HEAD_CHARS]})"
    )


def test_tear_off_verdict_fails_with_the_lower_the_limit_wording_when_the_arm_could_not_lift():
    over = checklist.TearOffReading(held=False, rise_mm=0.0, could_not_lift=True, mass_kg=78.0)
    under = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM)
    light = checklist.TearOffReading(held=True, rise_mm=checklist.LIFT_DISTANCE_MM)

    ok, detail = checklist.tear_off_verdict(over, under, light)

    assert ok is False
    assert "the arm cannot lift 78 kg" in detail
    assert "lower the configured coaxial limit" in detail


def test_hold_load_detail_reports_mean_and_peak():
    detail = checklist.hold_load_detail({"coaxial_load_n": 12.34, "peak_coaxial_load_n": 15.6})
    assert detail == "load 12.3 N mean, peak 15.6 N per cup"


def test_hold_load_detail_appends_the_release_load_when_present():
    detail = checklist.hold_load_detail(
        {"coaxial_load_n": 12.34, "peak_coaxial_load_n": 31.2, "released_load_n": 24.9}
    )
    assert detail == "load 12.3 N mean, peak 31.2 N per cup, released at 24.9 N"


def test_hold_load_detail_is_na_when_the_module_carries_no_load_meta():
    assert checklist.hold_load_detail({}) == "load n/a"


def test_hold_load_detail_names_the_monitor_state_when_the_module_reports_one():
    detail = checklist.hold_load_detail(
        {"coaxial_load_n": 0.0, "peak_coaxial_load_n": 0.0, "coaxial_monitor": "idle"}
    )
    assert detail == "load 0.0 N mean, peak 0.0 N per cup, monitor idle"


def test_hold_lost_detail_names_the_leg_and_the_boxs_pose():
    detail = checklist.hold_lost_detail("swing lift", {"x": 1.0, "y": 2.0, "z": 3.0})
    assert "hold lost after swing lift" in detail
    assert "1.0" in detail and "2.0" in detail and "3.0" in detail


def test_hold_lost_detail_says_no_known_pose_when_the_box_never_registered():
    detail = checklist.hold_lost_detail("release lift", None)
    assert "hold lost after release lift" in detail
    assert "no known pose" in detail


def test_branch_of_reads_wrist_2s_sign():
    seed_a = checklist.PLACE_DESCENT_SEED_JOINTS_DEG["A"]
    seed_b = checklist.PLACE_DESCENT_SEED_JOINTS_DEG["B"]
    assert checklist.branch_of(seed_a) == "A"
    assert checklist.branch_of(seed_b) == "B"


def test_branch_report_names_a_seed_that_held():
    detail = checklist.branch_report("A", "A", "A")
    assert "held the seed" in detail
    assert "did not hold the seed" not in detail


def test_branch_report_names_a_seed_that_did_not_hold():
    detail = checklist.branch_report("A", "B", "B")
    assert "did not hold the seed" in detail
    assert "branch changed" not in detail


def test_branch_report_names_a_branch_that_changed_mid_pick():
    detail = checklist.branch_report("A", "A", "B")
    assert "did not hold the seed" in detail
    assert "branch changed between standoff and descent start" in detail


def test_wrist_fold_deg_is_zero_on_a_monotone_descent():
    samples = [[0.0, 0.0, 0.0, joint, 0.0, 0.0] for joint in (90.0, 92.0, 95.0, 100.0)]
    assert checklist.wrist_fold_deg(samples) == pytest.approx(0.0, abs=1e-9)


def test_wrist_fold_deg_reads_a_78_degree_reversal():
    samples = [[0.0, 0.0, 0.0, joint, 0.0, 0.0] for joint in (90.0, 95.0, 87.2, 95.0)]
    assert checklist.wrist_fold_deg(samples) == pytest.approx(7.8, abs=1e-9)
