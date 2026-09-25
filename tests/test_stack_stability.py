import math

import pytest

from isaac_module.stack_stability import (
    BoxPlacement,
    SettlePolicy,
    StackTolerances,
    format_stack_table,
    hold_excursion_mm,
    pose_tilt_deg,
    settle_time_s,
    stack_verdict,
    trailing_window_settled,
)


def _pose(
    x: float, y: float, z: float, o_x: float = 0.0, o_y: float = 0.0, o_z: float = 1.0
) -> dict:
    return {"x": x, "y": y, "z": z, "o_x": o_x, "o_y": o_y, "o_z": o_z, "theta": 0.0}


def _stack(offset_layer_2: bool = False, tip_layer_2: bool = False) -> list[BoxPlacement]:
    placements = []
    seq = 0
    for layer in (1, 2):
        for row in range(2):
            for col in range(2):
                seq += 1
                x = col * 200.0
                y = row * 150.0
                z = layer * 100.0
                target = _pose(x, y, z)
                if layer == 2 and offset_layer_2:
                    z -= 100.0
                if layer == 2 and tip_layer_2:
                    at_settle = _pose(x, y, z, o_x=1.0, o_z=0.0)
                else:
                    at_settle = _pose(x, y, z)
                placements.append(
                    BoxPlacement(
                        name=f"box-{seq}",
                        seq=seq,
                        layer=layer,
                        target_mm=target,
                        at_settle_mm=at_settle,
                        after_hold_mm=at_settle,
                    )
                )
    return placements


def test_stack_held_worst_values_match_largest_inputs():
    placements = _stack()
    verdict = stack_verdict(placements, hold=True)
    assert verdict.held is True
    assert verdict.worst_error_mm == 0.0
    assert verdict.worst_drift_mm == 0.0
    assert verdict.worst_tilt_deg == 0.0
    assert verdict.failures == ()


def test_one_box_creeps_during_hold_fails_only_that_box():
    placements = _stack()
    creeping = placements[0]
    moved_pose = dict(creeping.at_settle_mm)
    moved_pose["x"] += 5.0
    placements[0] = BoxPlacement(
        name=creeping.name,
        seq=creeping.seq,
        layer=creeping.layer,
        target_mm=creeping.target_mm,
        at_settle_mm=creeping.at_settle_mm,
        after_hold_mm=moved_pose,
    )
    verdict = stack_verdict(placements, hold=True)
    assert verdict.held is False
    assert len(verdict.failures) == 1
    assert creeping.name in verdict.failures[0]
    assert "drift" in verdict.failures[0]


def test_collapsed_layer_fails_only_that_layer_on_error_and_tilt():
    placements = _stack(offset_layer_2=True, tip_layer_2=True)
    layer_1_verdict = stack_verdict(placements, hold=True, layer=1)
    assert layer_1_verdict.held is True

    layer_2_verdict = stack_verdict(placements, hold=True, layer=2)
    assert layer_2_verdict.held is False
    for box in layer_2_verdict.boxes:
        assert not box.ok
        joined = "; ".join(box.reasons)
        assert "error" in joined
        assert "tilt" in joined

    whole_stack = stack_verdict(placements, hold=True)
    layer_2_names = {p.name for p in placements if p.layer == 2}
    failure_names = {failure.split(":")[0] for failure in whole_stack.failures}
    assert failure_names == layer_2_names
    assert len(whole_stack.failures) == 4


def test_never_placed_reports_never_placed_and_no_drift_reason():
    target = _pose(0, 0, 0)
    placement = BoxPlacement(
        name="ghost", seq=1, layer=1, target_mm=target, at_settle_mm=None, after_hold_mm=None
    )
    verdict = stack_verdict([placement], hold=True)
    box = verdict.boxes[0]
    assert math.isinf(box.error_mm)
    assert "never placed" in "; ".join(box.reasons)
    assert box.reasons.count("gone after the hold") == 0


def test_gone_after_the_hold_when_after_hold_missing():
    target = _pose(0, 0, 0)
    placement = BoxPlacement(
        name="vanished", seq=1, layer=1, target_mm=target, at_settle_mm=target, after_hold_mm=None
    )
    verdict = stack_verdict([placement], hold=True)
    box = verdict.boxes[0]
    assert math.isinf(box.drift_mm)
    assert "gone after the hold" in box.reasons


def test_hold_false_never_judges_drift_even_when_after_hold_is_far_off():
    target = _pose(0, 0, 0)
    far_pose = _pose(1000, 1000, 1000)
    placement = BoxPlacement(
        name="stable", seq=1, layer=1, target_mm=target, at_settle_mm=target, after_hold_mm=far_pose
    )
    verdict = stack_verdict([placement], hold=False)
    box = verdict.boxes[0]
    assert box.drift_mm is None
    assert not any("drift" in reason for reason in box.reasons)
    assert verdict.worst_drift_mm is None


def test_empty_placements_is_value_error():
    with pytest.raises(ValueError):
        stack_verdict([], hold=True)


def test_unknown_layer_is_value_error():
    placements = _stack()
    with pytest.raises(ValueError):
        stack_verdict(placements, hold=True, layer=99)


def test_error_exactly_at_bound_is_ok_and_just_over_is_not():
    target = _pose(0, 0, 0)
    tolerances = StackTolerances(placement_mm=3.0, drift_mm=1.0, tilt_deg=2.0)

    at_bound = _pose(3.0, 0, 0)
    placement_ok = BoxPlacement(
        name="edge", seq=1, layer=1, target_mm=target, at_settle_mm=at_bound, after_hold_mm=at_bound
    )
    verdict_ok = stack_verdict([placement_ok], tolerances, hold=True)
    assert verdict_ok.boxes[0].ok is True

    over_bound = _pose(3.01, 0, 0)
    placement_bad = BoxPlacement(
        name="over",
        seq=1,
        layer=1,
        target_mm=target,
        at_settle_mm=over_bound,
        after_hold_mm=over_bound,
    )
    verdict_bad = stack_verdict([placement_bad], tolerances, hold=True)
    assert verdict_bad.boxes[0].ok is False


def test_format_stack_table_has_header_and_one_row_per_box():
    target = _pose(0, 0, 0)
    never_placed = BoxPlacement(
        name="ghost", seq=1, layer=1, target_mm=target, at_settle_mm=None, after_hold_mm=None
    )
    hold_false_box = BoxPlacement(
        name="stable", seq=2, layer=1, target_mm=target, at_settle_mm=target, after_hold_mm=target
    )
    verdict = stack_verdict([never_placed, hold_false_box], hold=False)
    table = format_stack_table(verdict)
    lines = table.splitlines()
    assert len(lines) == 3
    assert "inf" in lines[1]
    assert "-" in lines[1]
    assert "-" in lines[2]
    assert not table.endswith("\n")


def test_settle_time_s_returns_the_opener_time_not_the_close_time():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.15, poll_s=0.1, timeout_s=1.0)
    still = _pose(0, 0, 0)
    samples = [
        (0.0, _pose(50, 0, 0)),
        (0.1, still),
        (0.2, still),
        (0.3, still),
    ]
    result = settle_time_s(samples, policy)
    assert result == pytest.approx(0.1)


def test_settle_time_s_none_when_body_never_stops():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.2, poll_s=0.1, timeout_s=1.0)
    samples = [
        (0.0, _pose(0, 0, 0)),
        (0.1, _pose(10, 0, 0)),
        (0.2, _pose(20, 0, 0)),
        (0.3, _pose(30, 0, 0)),
    ]
    assert settle_time_s(samples, policy) is None


def test_settle_time_s_none_when_none_pose_sits_inside_every_window():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.2, poll_s=0.1, timeout_s=1.0)
    still = _pose(0, 0, 0)
    samples = [
        (0.0, still),
        (0.1, None),
        (0.2, still),
    ]
    assert settle_time_s(samples, policy) is None


def test_settle_time_s_none_when_samples_do_not_span_a_window():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.5, poll_s=0.1, timeout_s=1.0)
    still = _pose(0, 0, 0)
    samples = [(0.0, still), (0.1, still)]
    assert settle_time_s(samples, policy) is None


def test_trailing_window_settled_true_for_still_body():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.2, poll_s=0.1, timeout_s=1.0)
    still = _pose(0, 0, 0)
    samples = [
        (0.0, still),
        (0.1, still),
        (0.2, still),
    ]
    assert trailing_window_settled(samples, policy) is True


def test_trailing_window_settled_false_when_newest_moved():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.2, poll_s=0.1, timeout_s=1.0)
    still = _pose(0, 0, 0)
    samples = [
        (0.0, still),
        (0.1, still),
        (0.2, _pose(50, 0, 0)),
    ]
    assert trailing_window_settled(samples, policy) is False


def test_trailing_window_settled_false_when_samples_span_less_than_a_window():
    policy = SettlePolicy(tolerance_mm=1.0, window_s=0.5, poll_s=0.1, timeout_s=1.0)
    still = _pose(0, 0, 0)
    samples = [(0.0, still), (0.1, still)]
    assert trailing_window_settled(samples, policy) is False


def test_hold_excursion_mm_reports_wobble_and_return():
    samples = [
        (0.0, _pose(0, 0, 0)),
        (0.1, _pose(20, 0, 0)),
        (0.2, _pose(0, 0, 0)),
    ]
    assert hold_excursion_mm(samples) == pytest.approx(20.0)


def test_hold_excursion_mm_none_sample_gives_inf():
    samples = [(0.0, _pose(0, 0, 0)), (0.1, None)]
    assert math.isinf(hold_excursion_mm(samples))


def test_hold_excursion_mm_no_samples_raises():
    with pytest.raises(ValueError):
        hold_excursion_mm([])


def test_pose_tilt_deg_zero_for_equal_vectors():
    a = _pose(0, 0, 0, o_x=0, o_y=0, o_z=1)
    b = _pose(1, 1, 1, o_x=0, o_y=0, o_z=1)
    assert pose_tilt_deg(a, b) == pytest.approx(0.0)


def test_pose_tilt_deg_ninety_for_perpendicular():
    a = _pose(0, 0, 0, o_x=0, o_y=0, o_z=1)
    b = _pose(0, 0, 0, o_x=1, o_y=0, o_z=0)
    assert pose_tilt_deg(a, b) == pytest.approx(90.0)


def test_pose_tilt_deg_zero_vector_raises():
    a = _pose(0, 0, 0, o_x=0, o_y=0, o_z=0)
    b = _pose(0, 0, 0, o_x=0, o_y=0, o_z=1)
    with pytest.raises(ValueError):
        pose_tilt_deg(a, b)


def test_settle_policy_rejects_non_positive_values():
    with pytest.raises(ValueError):
        SettlePolicy(tolerance_mm=0.0)
    with pytest.raises(ValueError):
        SettlePolicy(poll_s=0.0)


def test_settle_policy_rejects_window_shorter_than_poll():
    with pytest.raises(ValueError):
        SettlePolicy(window_s=0.05, poll_s=0.1)


def test_settle_policy_rejects_timeout_shorter_than_window():
    with pytest.raises(ValueError):
        SettlePolicy(timeout_s=0.1, window_s=0.5)


def test_stack_tolerances_rejects_non_positive_values():
    with pytest.raises(ValueError):
        StackTolerances(placement_mm=0.0)
    with pytest.raises(ValueError):
        StackTolerances(drift_mm=-1.0)
    with pytest.raises(ValueError):
        StackTolerances(tilt_deg=0.0)
