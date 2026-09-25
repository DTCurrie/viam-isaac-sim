import asyncio
import itertools
import json
import math
import time

import pytest
from grpclib import Status
from viam.components.arm import JointPositions
from viam.errors import MethodNotImplementedError
from viam.proto.app.robot import ComponentConfig
from viam.proto.component.arm import MoveOptions
from viam.utils import dict_to_struct

from isaac_module.models.arm import (
    WAYPOINT_MAX_STEP_RAD,
    ArmMoveStalledError,
    ArmMoveTimeoutError,
    IsaacArm,
    JointTargetOutOfLimitsError,
    densify_waypoints,
)
from isaac_module.sim_manager import SettleOutcome


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _arm(world, name: str = "test-arm") -> IsaacArm:
    return IsaacArm.new(_config(name, {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {})


def test_move_to_joint_positions_length_mismatch_raises(world):
    arm = _arm(world, "arm-mismatch")

    async def scenario():
        target = JointPositions(values=[10, -20, 30, 0, 5])
        with pytest.raises(ValueError, match=r"6.*5"):
            await arm.move_to_joint_positions(target)

    asyncio.run(scenario())


def test_move_to_joint_positions_matching_length_moves(world):
    arm = _arm(world, "arm-match")

    async def scenario():
        target = JointPositions(values=[10, -20, 30, 0, 5, -5])
        await arm.move_to_joint_positions(target)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

    asyncio.run(scenario())


def test_get_geometries_returns_empty_list(world):
    arm = _arm(world, "arm-geometries")

    async def scenario():
        assert await arm.get_geometries() == []

    asyncio.run(scenario())


def test_get_3d_models_returns_empty_dict(world):
    arm = _arm(world, "arm-3d-models")

    async def scenario():
        assert await arm.get_3d_models() == {}

    asyncio.run(scenario())


def test_do_command_raises_method_not_implemented(world):
    arm = _arm(world, "arm-unknown-command")

    async def scenario():
        with pytest.raises(MethodNotImplementedError) as excinfo:
            await arm.do_command({"command": "not-a-real-command"})
        assert excinfo.value.grpc_code == Status.UNIMPLEMENTED

    asyncio.run(scenario())


def test_move_to_joint_positions_stall_raises_quickly(world):
    arm = IsaacArm.new(
        _config(
            "arm-stall",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "mock_stall_fraction": 0.5},
        ),
        {},
    )

    async def scenario():
        target = JointPositions(values=[10, -20, 30, 0, 5, -5])
        start = time.monotonic()
        with pytest.raises(ArmMoveStalledError) as excinfo:
            await arm.move_to_joint_positions(target)
        elapsed = time.monotonic() - start
        assert excinfo.value.grpc_code == Status.ABORTED
        # a wall-clock deadline would have taken the full 30s move_timeout_sec
        assert elapsed < 2.0

    asyncio.run(scenario())


def test_move_through_joint_positions_timeout_raises(world):
    arm = IsaacArm.new(
        _config(
            "arm-timeout",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "move_timeout_sec": 0.2},
        ),
        {},
    )

    async def scenario():
        options = MoveOptions(max_vel_degs_per_sec=5.0)
        target = JointPositions(values=[90, 0, 0, 0, 0, 0])  # ~1.57 rad away
        with pytest.raises(ArmMoveTimeoutError) as excinfo:
            await arm.move_through_joint_positions([target], options)
        assert excinfo.value.grpc_code == Status.DEADLINE_EXCEEDED

    asyncio.run(scenario())


def test_move_to_joint_positions_out_of_limits_raises(world, tmp_path):
    sva = {
        "name": "limit-test",
        "kinematic_param_type": "SVA",
        "links": [],
        "joints": [
            {
                "id": f"j{i}",
                "type": "revolute",
                "parent": "base_link",
                "axis": {"x": 0, "y": 0, "z": 1},
                "min": -90,
                "max": 90,
            }
            for i in range(6)
        ],
    }
    path = tmp_path / "limit-test.json"
    path.write_text(json.dumps(sva))

    arm = IsaacArm.new(
        _config(
            "arm-limits",
            {
                "world": "isaac-world",
                "asset": "ur20",
                "mock_dof": 6,
                "kinematics_url": path.as_uri(),
            },
        ),
        {},
    )

    async def scenario():
        out_of_range = JointPositions(values=[120, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError) as excinfo:
            await arm.move_to_joint_positions(out_of_range)
        assert isinstance(excinfo.value, JointTargetOutOfLimitsError)
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

        in_range = JointPositions(values=[10, -20, 30, 0, 5, -5])
        await arm.move_to_joint_positions(in_range)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

    asyncio.run(scenario())


def test_boundary_joint_target_clamps_instead_of_raising(world, tmp_path):
    """GPU wedge: physics settle drifted wrist_2 to -360.00003 deg,
    the motion service echoed that reported state back as a plan waypoint,
    and the strict limit check rejected every subsequent plan. A target
    within the tolerance of a limit must clamp onto the limit and execute;
    only a target beyond the tolerance raises."""
    sva = {
        "name": "limit-test",
        "kinematic_param_type": "SVA",
        "links": [],
        "joints": [
            {
                "id": f"j{i}",
                "type": "revolute",
                "parent": "base_link",
                "axis": {"x": 0, "y": 0, "z": 1},
                "min": -90,
                "max": 90,
            }
            for i in range(6)
        ],
    }
    path = tmp_path / "limit-test.json"
    path.write_text(json.dumps(sva))

    arm = IsaacArm.new(
        _config(
            "arm-boundary-limits",
            {
                "world": "isaac-world",
                "asset": "ur20",
                "mock_dof": 6,
                "kinematics_url": path.as_uri(),
            },
        ),
        {},
    )

    async def scenario():
        just_past = JointPositions(values=[-90.00003, 0, 0, 0, 0, 0])
        await arm.move_to_joint_positions(just_past)
        end = await arm.get_joint_positions()
        assert end.values[0] == pytest.approx(-90.0, abs=0.5)

        beyond_tolerance = JointPositions(values=[-90.02, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError) as excinfo:
            await arm.move_to_joint_positions(beyond_tolerance)
        assert isinstance(excinfo.value, JointTargetOutOfLimitsError)

    asyncio.run(scenario())


def test_get_joint_positions_clamps_a_micro_drift_past_a_limit(world, tmp_path):
    """The reported state is the motion planner's start state, so a reading
    micro-degrees past a limit must be reported AS the limit (a reading beyond
    the tolerance stays visible - truth over cosmetics)."""
    sva = {
        "name": "limit-test",
        "kinematic_param_type": "SVA",
        "links": [],
        "joints": [
            {
                "id": f"j{i}",
                "type": "revolute",
                "parent": "base_link",
                "axis": {"x": 0, "y": 0, "z": 1},
                "min": -90,
                "max": 90,
            }
            for i in range(6)
        ],
    }
    path = tmp_path / "limit-test.json"
    path.write_text(json.dumps(sva))

    arm = IsaacArm.new(
        _config(
            "arm-report-limits",
            {
                "world": "isaac-world",
                "asset": "ur20",
                "mock_dof": 6,
                "kinematics_url": path.as_uri(),
            },
        ),
        {},
    )

    async def scenario():
        import math as math_module

        handle = arm._h()
        drifted = [math_module.radians(v) for v in (-90.00003, 0, 0, 0, 90.005, -91.0)]
        original = handle.get_joint_positions
        handle.get_joint_positions = lambda: drifted  # type: ignore[method-assign]
        try:
            reported = await arm.get_joint_positions()
        finally:
            handle.get_joint_positions = original  # type: ignore[method-assign]
        assert reported.values[0] == -90.0  # micro-drift below min: clamped
        assert reported.values[4] == 90.0  # within tolerance above max: clamped
        assert reported.values[5] == pytest.approx(-91.0)  # beyond tolerance: reported as-is

    asyncio.run(scenario())


def test_move_to_joint_positions_no_kinematics_skips_limit_check(world):
    arm = IsaacArm.new(
        _config("arm-no-kinematics", {"world": "isaac-world", "asset": "franka", "mock_dof": 6}), {}
    )

    async def scenario():
        # franka has no known kinematics url and none is configured: the
        # limit check must be skipped (and must not touch the network).
        target = JointPositions(values=[500, 0, 0, 0, 0, 0])
        await arm.move_to_joint_positions(target)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([500, 0, 0, 0, 0, 0], abs=0.5)

    asyncio.run(scenario())


def test_move_through_joint_positions_max_vel_option_is_slower(world):
    arm_default = IsaacArm.new(
        _config("arm-through-default", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )
    arm_slow = IsaacArm.new(
        _config("arm-through-slow", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        target = JointPositions(values=[45, 0, 0, 0, 0, 0])

        start = time.monotonic()
        await arm_default.move_through_joint_positions([target], None)
        default_elapsed = time.monotonic() - start

        start = time.monotonic()
        await arm_slow.move_through_joint_positions([target], MoveOptions(max_vel_degs_per_sec=5.0))
        slow_elapsed = time.monotonic() - start

        assert slow_elapsed > default_elapsed

    asyncio.run(scenario())


def test_move_through_joint_positions_per_joint_max_vel_wins_over_scalar(world):
    """viam.md: when max_vel_degs_per_sec_joints is set it is the ONLY
    velocity limit honored, and max_vel_degs_per_sec is ignored - not the
    other way around. Set a fast scalar (faster than the mock's own top
    speed, so it would be indistinguishable from "no limit" if it won) next
    to a slow per-joint limit; only honoring the per-joint value produces a
    move slower than the unlimited case."""
    arm_default = IsaacArm.new(
        _config("arm-priority-default", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}),
        {},
    )
    arm_both_set = IsaacArm.new(
        _config("arm-priority-both", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        target = JointPositions(values=[45, 0, 0, 0, 0, 0])
        options = MoveOptions(
            max_vel_degs_per_sec=1000.0,  # far above the mock's SPEED; a no-op if honored
            max_vel_degs_per_sec_joints=[5.0] * 6,  # should be the only limit applied
        )

        start = time.monotonic()
        await arm_default.move_through_joint_positions([target], None)
        default_elapsed = time.monotonic() - start

        start = time.monotonic()
        await arm_both_set.move_through_joint_positions([target], options)
        both_set_elapsed = time.monotonic() - start

        assert both_set_elapsed > default_elapsed

    asyncio.run(scenario())


def test_is_moving_false_after_move_true_during_move(world):
    arm = IsaacArm.new(
        _config("arm-is-moving", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        assert not await arm.is_moving()
        target = JointPositions(values=[90, 0, 0, 0, 0, 0])

        move_task = asyncio.ensure_future(arm.move_to_joint_positions(target))
        await asyncio.sleep(0.02)
        assert await arm.is_moving()
        await move_task
        assert not await arm.is_moving()

    asyncio.run(scenario())


def test_move_to_joint_positions_cancelled_holds_position(world):
    """A dropped RPC (task cancellation) must stop the drive where it is,
    not leave it pushing toward a target nothing is waiting on."""
    arm = IsaacArm.new(
        _config("arm-cancel-hold", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        target = JointPositions(values=[90, 0, 0, 0, 0, 0])
        move_task = asyncio.ensure_future(arm.move_to_joint_positions(target))
        await asyncio.sleep(0.02)
        move_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await move_task
        assert await arm.is_moving() is False
        held = await arm.get_joint_positions()
        assert 0.0 < held.values[0] < 90.0  # stopped partway, not at the target

    asyncio.run(scenario())


def test_move_through_joint_positions_cancelled_holds_position(world):
    arm = IsaacArm.new(
        _config(
            "arm-through-cancel-hold", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}
        ),
        {},
    )

    async def scenario():
        waypoints = [
            JointPositions(values=[45, 0, 0, 0, 0, 0]),
            JointPositions(values=[90, 0, 0, 0, 0, 0]),
        ]
        move_task = asyncio.ensure_future(arm.move_through_joint_positions(waypoints))
        await asyncio.sleep(0.02)
        move_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await move_task
        assert await arm.is_moving() is False

    asyncio.run(scenario())


def test_failed_move_holds_position_instead_of_pushing(world):
    """After a stall the drive target must be the current pose, so
    the arm stops pushing into whatever blocked it."""
    arm = IsaacArm.new(
        _config(
            "arm-hold-after-stall",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "mock_stall_fraction": 0.5},
        ),
        {},
    )

    async def scenario():
        with pytest.raises(ArmMoveStalledError):
            await arm.move_to_joint_positions(JointPositions(values=[40, 0, 0, 0, 0, 0]))
        assert await arm.is_moving() is False  # target re-pointed at where it stopped
        held = await arm.get_joint_positions()
        assert held.values[0] == pytest.approx(20.0, abs=1.0)  # halfway, and staying there

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# max_vel_degs_per_sec: the config-level default velocity cap (carries a
# real UR's speed_degs_per_sec) applied to any move that carries no
# MoveOptions cap of its own.
# ----------------------------------------------------------------------


def test_validate_config_rejects_non_positive_max_vel_degs_per_sec(world):
    config = _config(
        "arm-bad-max-vel", {"world": "isaac-world", "asset": "ur20", "max_vel_degs_per_sec": 0}
    )
    with pytest.raises(ValueError, match="max_vel_degs_per_sec"):
        IsaacArm.validate_config(config)


def test_move_to_joint_positions_honors_configured_max_vel_default(world):
    arm_default = IsaacArm.new(
        _config("arm-mtjp-default", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )
    arm_slow = IsaacArm.new(
        _config(
            "arm-mtjp-slow",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "max_vel_degs_per_sec": 5.0},
        ),
        {},
    )

    async def scenario():
        target = JointPositions(values=[45, 0, 0, 0, 0, 0])

        start = time.monotonic()
        await arm_default.move_to_joint_positions(target)
        default_elapsed = time.monotonic() - start

        start = time.monotonic()
        await arm_slow.move_to_joint_positions(target)
        slow_elapsed = time.monotonic() - start

        assert slow_elapsed > default_elapsed

    asyncio.run(scenario())


def test_move_through_joint_positions_falls_back_to_configured_max_vel(world):
    """MoveOptions() with no velocity fields set carries no cap of its own,
    so the configured max_vel_degs_per_sec attribute applies."""
    arm_default = IsaacArm.new(
        _config(
            "arm-through-fallback-default", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}
        ),
        {},
    )
    arm_configured = IsaacArm.new(
        _config(
            "arm-through-fallback-configured",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "max_vel_degs_per_sec": 5.0},
        ),
        {},
    )

    async def scenario():
        target = JointPositions(values=[45, 0, 0, 0, 0, 0])

        start = time.monotonic()
        await arm_default.move_through_joint_positions([target], MoveOptions())
        default_elapsed = time.monotonic() - start

        start = time.monotonic()
        await arm_configured.move_through_joint_positions([target], MoveOptions())
        configured_elapsed = time.monotonic() - start

        assert configured_elapsed > default_elapsed

    asyncio.run(scenario())


def test_move_through_joint_positions_explicit_option_overrides_configured_default(world):
    """A MoveOptions cap still wins over the configured default, even a
    faster one - the configured attribute is only a fallback."""
    arm = IsaacArm.new(
        _config(
            "arm-through-explicit-overrides",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "max_vel_degs_per_sec": 1.0},
        ),
        {},
    )

    async def scenario():
        target = JointPositions(values=[45, 0, 0, 0, 0, 0])
        start = time.monotonic()
        await arm.move_through_joint_positions([target], MoveOptions(max_vel_degs_per_sec=1000.0))
        elapsed = time.monotonic() - start
        # far faster than the 1.0 deg/s configured default would allow (45s)
        assert elapsed < 5.0

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# Known-asset kinematics ship inside the module and are prefetched on
# reconfigure, not the first RPC.
# ----------------------------------------------------------------------


def test_known_asset_kinematics_url_is_a_packaged_file(world):
    arm = IsaacArm.new(
        _config("arm-packaged-kinematics", {"world": "isaac-world", "asset": "ur20"}), {}
    )
    url = arm._kinematics_url()
    assert url is not None
    assert url.startswith("file://")
    assert url.endswith("kinematics_files/ur20.json")


def test_reconfigure_prefetches_kinematics_in_the_background(world):
    arm = IsaacArm.new(
        _config("arm-prefetch-kinematics", {"world": "isaac-world", "asset": "ur20"}), {}
    )
    deadline = time.monotonic() + 2.0
    while arm._kinematics is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert arm._kinematics is not None, "reconfigure did not prefetch kinematics in the background"


@pytest.mark.parametrize(
    ("measured_deg", "min_deg", "max_deg", "expected_deg"),
    [
        # the value a sorting run actually measured on a joint limited to 360
        (4135.62, -360.0, 360.0, 175.62),
        (-400.0, -360.0, 360.0, -40.0),
        (725.0, -180.0, 180.0, 5.0),
        # already legal, including exactly at a limit: nothing moves
        (360.0, -360.0, 360.0, 360.0),
        (0.0, -360.0, 360.0, 0.0),
        # no whole turn lands inside, so the caller sees the real value
        (200.0, -10.0, 10.0, 200.0),
    ],
)
def test_wound_joint_angles_report_the_same_pose_inside_the_declared_range(
    measured_deg: float, min_deg: float, max_deg: float, expected_deg: float
) -> None:
    """PhysX accumulates a revolute joint's angle, so an arm that keeps turning
    the same way reads past the range its kinematics declares and the motion
    service refuses to plan from it. A whole turn is the identity for a
    revolute joint, so the reported angle moves by whole turns only."""
    from isaac_module.models.arm import _wrapped_into_range

    wrapped = _wrapped_into_range(measured_deg, min_deg, max_deg)
    assert wrapped == pytest.approx(expected_deg)
    # whatever it reports has to be the same physical pose
    assert (wrapped - measured_deg) % 360.0 == pytest.approx(0.0, abs=1e-9)


def test_settle_drift_past_a_limit_reports_the_limit_rather_than_a_whole_turn_away():
    """A joint resting a hair past its limit is at that limit. Winding it by a
    turn would describe the same pose but move the reported number across the
    whole range, which is not what a drift of 3e-5 degrees means."""
    from isaac_module.models.arm import _JOINT_LIMIT_TOLERANCE_DEG, _wrapped_into_range

    drifted = -360.00003
    assert abs(drifted - (-360.0)) < _JOINT_LIMIT_TOLERANCE_DEG
    # the wrap on its own would move it a full turn, which is why
    # get_joint_positions checks the drift tolerance first
    assert _wrapped_into_range(drifted, -360.0, 360.0) == pytest.approx(-0.00003)


def test_an_isolated_waypoint_stall_does_not_abandon_the_trajectory(world):
    """A constrained path arrives as dozens of waypoints a couple of
    millimetres apart, and the arm sits still on a small residual at each one,
    which reads exactly like a blocked arm at that waypoint. Before this, the
    first such waypoint aborted the whole move."""
    arm = IsaacArm.new(
        _config(
            "arm-waypoint-flows-on",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "mock_stall_fraction": 0.5},
        ),
        {},
    )

    async def scenario():
        # this mock closes half the remaining gap per waypoint, so the early
        # ones stall short and the later ones converge, which is the shape of
        # a real dense path
        target = JointPositions(values=[10, -20, 30, 0, 5, -5])
        await arm.move_through_joint_positions([target] * 8)
        reached = (await arm.get_joint_positions()).values
        assert reached == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

    asyncio.run(scenario())


def test_a_run_of_stalled_waypoints_fails_rather_than_grinding_through_the_path(world):
    """The other half of the rule: a genuinely blocked arm must still fail
    fast, rather than walking every remaining waypoint first."""
    from isaac_module.models.arm import _MAX_CONSECUTIVE_WAYPOINT_STALLS

    assert _MAX_CONSECUTIVE_WAYPOINT_STALLS > 1

    arm = IsaacArm.new(
        _config(
            "arm-waypoint-run-of-stalls",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "mock_stall_fraction": 0.5},
        ),
        {},
    )

    async def scenario():
        # each waypoint steps further than the last, so closing half the gap
        # never catches up and every waypoint stalls, which is what a blocked
        # arm looks like
        waypoints = [
            JointPositions(values=[10.0 * k, -20.0 * k, 30.0 * k, 0.0, 0.0, 0.0])
            for k in range(1, _MAX_CONSECUTIVE_WAYPOINT_STALLS + 6)
        ]
        with pytest.raises(ArmMoveStalledError) as excinfo:
            await arm.move_through_joint_positions(waypoints)
        message = str(excinfo.value)
        assert f"{_MAX_CONSECUTIVE_WAYPOINT_STALLS} consecutive" in message
        # it gave up at the cap instead of walking the rest of the path; the
        # count it reports is of the densified path it was driving
        driven = len(
            densify_waypoints(
                [[math.radians(v) for v in wp.values] for wp in waypoints], WAYPOINT_MAX_STEP_RAD
            )
        )
        assert driven > len(waypoints)
        assert f"waypoint {_MAX_CONSECUTIVE_WAYPOINT_STALLS}/{driven}" in message

    asyncio.run(scenario())


def test_home_joints_place_the_arm_without_driving_it_there(world):
    """A UR asset's default pose is every joint at zero, which is the arm fully
    extended horizontally. In a cell with a tall object in front of it that is
    the arm resting on that object, and the first commanded move shoves it
    aside. The home pose has to be a placement, so the arm is already there
    when the first client connects rather than sweeping toward it."""
    home = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]
    arm = IsaacArm.new(
        _config(
            "arm-homed",
            {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "home_joints_deg": home},
        ),
        {},
    )

    async def scenario():
        # no settle wait: placed, not driven, so it reads home immediately
        assert (await arm.get_joint_positions()).values == pytest.approx(home, abs=1e-6)
        assert await arm.is_moving() is False

    asyncio.run(scenario())


def test_an_arm_without_a_home_pose_keeps_the_assets_default(world):
    arm = IsaacArm.new(
        _config("arm-unhomed", {"world": "isaac-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        assert (await arm.get_joint_positions()).values == pytest.approx([0.0] * 6, abs=1e-6)

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["notalist", [], [1.0, "two"], [True, 2.0]])
def test_a_malformed_home_pose_is_refused_at_build(world, bad):
    """A bad home pose would otherwise spawn the arm somewhere nobody asked
    for, and the failure would read as a physics problem rather than a config
    one."""
    with pytest.raises(ValueError, match="home_joints_deg"):
        IsaacArm.new(
            _config(
                f"arm-bad-home-{abs(hash(str(bad)))}",
                {"world": "isaac-world", "asset": "ur20", "mock_dof": 6, "home_joints_deg": bad},
            ),
            {},
        )


@pytest.mark.parametrize(
    "drive_gains",
    [
        {"wrist_3_joint": {"kp": "high"}},
        {"wrist_3_joint": {"stiffness": 1.0}},
        {"wrist_3_joint": {}},
        {},
        [1.0, 2.0],
    ],
)
def test_validate_config_rejects_malformed_drive_gains(world, drive_gains):
    config = _config(
        "arm-bad-gains", {"world": "isaac-world", "asset": "ur20", "drive_gains": drive_gains}
    )
    with pytest.raises(ValueError, match="drive_gains"):
        IsaacArm.validate_config(config)


def test_validate_config_accepts_per_joint_drive_gains(world):
    config = _config(
        "arm-good-gains",
        {
            "world": "isaac-world",
            "asset": "ur20",
            "drive_gains": {
                "wrist_3_joint": {"kp": 225781.4, "kd": 90.4},
                "wrist_2_joint": {"kd": 3.5},
            },
        },
    )
    IsaacArm.validate_config(config)


def test_densify_waypoints_splits_a_gap_into_equal_steps_with_exact_endpoints():
    start = [0.0, 0.1, -0.2]
    end = [math.radians(30.0), 0.1, -0.2]

    dense = densify_waypoints([start, end], math.radians(5.0))

    assert len(dense) == 7
    assert dense[0].positions == start
    assert dense[-1].positions == end
    # only the points between the two given waypoints are marked inserted
    assert [point.inserted for point in dense] == [False] + [True] * 5 + [False]
    steps = [dense[k + 1].positions[0] - dense[k].positions[0] for k in range(6)]
    assert steps == pytest.approx([math.radians(5.0)] * 6)
    # the joints that have nothing to do stay where they are
    assert all(point.positions[1:] == [0.1, -0.2] for point in dense)


def test_densify_waypoints_leaves_a_path_already_under_the_step_unchanged():
    path = [[0.0, 0.0], [0.02, -0.03], [0.05, -0.05]]

    dense = densify_waypoints(path, math.radians(5.0))
    assert [point.positions for point in dense] == path
    assert not any(point.inserted for point in dense)
    assert [p.positions for p in densify_waypoints([[1.0, 2.0]], math.radians(5.0))] == [[1.0, 2.0]]


def test_densify_waypoints_inserts_nothing_between_identical_points():
    path = [[0.3, -0.4], [0.3, -0.4], [0.3, -0.4]]

    dense = densify_waypoints(path, math.radians(5.0))
    assert [point.positions for point in dense] == path
    assert not any(point.inserted for point in dense)


def test_densify_waypoints_sizes_the_split_by_the_joint_with_the_most_to_do():
    # 12 degrees on the second joint against 1 degree on the first: 3 pieces
    dense = densify_waypoints(
        [[0.0, 0.0], [math.radians(1.0), math.radians(12.0)]], math.radians(5.0)
    )

    assert len(dense) == 4
    assert [point.inserted for point in dense] == [False, True, True, False]
    for a, b in itertools.pairwise(dense):
        assert (
            max(abs(y - x) for x, y in zip(a.positions, b.positions, strict=False))
            <= math.radians(5.0) + 1e-12
        )


def test_move_through_joint_positions_drives_dense_targets_between_sparse_waypoints(world):
    arm = _arm(world, "arm-through-dense")
    handle = arm._h()
    driven: list[list[float]] = []
    set_joint_targets = handle.set_joint_targets

    def record(positions, max_vel_rad_s=None):
        driven.append(list(positions))
        set_joint_targets(positions, max_vel_rad_s)

    handle.set_joint_targets = record

    async def scenario():
        await arm.move_through_joint_positions(
            [
                JointPositions(values=[0, 0, 0, 0, 0, 0]),
                JointPositions(values=[90, 0, 0, 0, 0, 0]),
            ]
        )

    asyncio.run(scenario())

    assert len(driven) >= 18
    base = [target[0] for target in driven]
    assert base == sorted(base)
    for a, b in itertools.pairwise(driven):
        assert max(abs(y - x) for x, y in zip(a, b, strict=False)) <= WAYPOINT_MAX_STEP_RAD + 1e-9
    assert driven[-1] == pytest.approx([math.radians(90), 0, 0, 0, 0, 0])


def test_move_through_joint_positions_passes_through_inserted_points_and_settles_on_planned_ones(
    world,
):
    arm = _arm(world, "arm-through-spy")
    handle = arm._h()
    calls: list[str] = []
    followed: list[list[list[float]]] = []
    reads: list[int] = []
    in_follow = False
    # what the handle's own follow_path does between points is its business, so
    # only the calls the model itself makes are recorded
    for name in ("wait_until_within", "wait_for_settle", "set_joint_targets"):
        original = getattr(handle, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            if not in_follow:
                calls.append(_name)
            return _original(*args, **kwargs)

        setattr(handle, name, spy)
    follow_path = handle.follow_path

    def spy_follow(points, *args, **kwargs):
        nonlocal in_follow
        calls.append("follow_path")
        followed.append([list(point) for point in points])
        in_follow = True
        try:
            return follow_path(points, *args, **kwargs)
        finally:
            in_follow = False

    handle.follow_path = spy_follow
    get_joint_positions = handle.get_joint_positions

    def counted_get(*args, **kwargs):
        reads.append(1)
        return get_joint_positions(*args, **kwargs)

    handle.get_joint_positions = counted_get

    async def scenario():
        await arm.move_through_joint_positions(
            [
                JointPositions(values=[0, 0, 0, 0, 0, 0]),
                JointPositions(values=[90, 0, 0, 0, 0, 0]),
            ]
        )

    asyncio.run(scenario())

    dense = densify_waypoints([[0.0] * 6, [math.radians(90)] + [0.0] * 5], WAYPOINT_MAX_STEP_RAD)
    inserted = [point.positions for point in dense if point.inserted]
    assert len(inserted) >= 17
    # the first planned waypoint, the one run of inserted points, the last planned waypoint
    assert calls == [
        "set_joint_targets",
        "wait_for_settle",
        "follow_path",
        "set_joint_targets",
        "wait_for_settle",
    ]
    assert followed == [inserted]
    # the trajectory reads the joints once, and nothing here stalls so it never reads again
    assert len(reads) == 1


def test_move_through_joint_positions_makes_one_follow_path_call_per_run_of_inserted_points(world):
    from isaac_module.models.arm import _WAYPOINT_DEADLINE_S, _WAYPOINT_TOLERANCE_RAD

    arm = _arm(world, "arm-through-runs")
    handle = arm._h()
    calls: list[tuple[int, float | None, float, float]] = []
    follow_path = handle.follow_path

    def spy_follow(points, max_vel_rad_s, tolerance_rad, point_timeout_s):
        calls.append((len(points), max_vel_rad_s, tolerance_rad, point_timeout_s))
        return follow_path(points, max_vel_rad_s, tolerance_rad, point_timeout_s)

    handle.follow_path = spy_follow

    async def scenario():
        await arm.move_through_joint_positions(
            [
                JointPositions(values=[0, 0, 0, 0, 0, 0]),
                JointPositions(values=[30, 0, 0, 0, 0, 0]),
                JointPositions(values=[30, 40, 0, 0, 0, 0]),
            ],
            MoveOptions(max_vel_degs_per_sec=90.0),
        )

    asyncio.run(scenario())

    # 30 and 40 degrees at 5 degrees a step insert 5 and 7 points
    assert calls == [
        (5, math.radians(90.0), _WAYPOINT_TOLERANCE_RAD, _WAYPOINT_DEADLINE_S),
        (7, math.radians(90.0), _WAYPOINT_TOLERANCE_RAD, _WAYPOINT_DEADLINE_S),
    ]


def _arm_with_scripted_follow(world, name: str, script):
    """An arm whose planned waypoints settle at once and whose follow_path is
    ``script``, so a test decides how a run of inserted points ends. Returns
    the arm, the points each follow_path call was handed, and the handle."""
    arm = _arm(world, name)
    handle = arm._h()
    handed: list[list[list[float]]] = []

    def scripted(points, max_vel_rad_s, tolerance_rad, point_timeout_s):
        handed.append([list(point) for point in points])
        return script(len(handed), points)

    handle.follow_path = scripted
    handle.set_joint_targets = lambda positions, max_vel_rad_s=None: None
    handle.wait_for_settle = lambda timeout_s, tolerance_rad=None: SettleOutcome.REACHED
    return arm, handed, handle


# 149 degrees at 5 degrees a step is 30 pieces, so 29 points are inserted
# between the two waypoints
_LONG_RUN_WAYPOINTS = [
    JointPositions(values=[0, 0, 0, 0, 0, 0]),
    JointPositions(values=[149, 0, 0, 0, 0, 0]),
]
_LONG_RUN_INSERTED = 29


@pytest.mark.parametrize(
    ("failure", "passed_first"),
    [(SettleOutcome.STALLED, 1), (SettleOutcome.TIMED_OUT, 0)],
)
def test_a_follow_that_returns_early_resumes_from_the_next_point_and_resets_the_count(
    world, failure, passed_first
):
    # more early returns than the consecutive-stall cap allows, none of them
    # consecutive: a stall after reached points starts the count again, and a
    # timeout clears it
    from isaac_module.models.arm import _MAX_CONSECUTIVE_WAYPOINT_STALLS

    early_returns = _MAX_CONSECUTIVE_WAYPOINT_STALLS + 2

    def script(call_number, points):
        if call_number <= early_returns:
            return failure, passed_first
        return SettleOutcome.REACHED, len(points)

    arm, handed, _handle = _arm_with_scripted_follow(world, f"arm-resume-{failure.value}", script)

    asyncio.run(arm.move_through_joint_positions(_LONG_RUN_WAYPOINTS))

    assert len(handed) == early_returns + 1
    assert len(handed[0]) == _LONG_RUN_INSERTED
    # each resumed call starts at the point after the one that failed
    for before, after in itertools.pairwise(handed[: early_returns + 1]):
        assert after == before[passed_first + 1 :]


def test_a_run_of_follows_stalled_at_their_first_point_fails_at_the_dense_index(world):
    from isaac_module.models.arm import _MAX_CONSECUTIVE_WAYPOINT_STALLS

    def script(call_number, points):
        return SettleOutcome.STALLED, 0

    arm, handed, handle = _arm_with_scripted_follow(world, "arm-follow-stall-run", script)
    stops: list[int] = []
    handle.stop = lambda: stops.append(1)

    async def scenario():
        with pytest.raises(ArmMoveStalledError) as excinfo:
            await arm.move_through_joint_positions(_LONG_RUN_WAYPOINTS)
        return str(excinfo.value)

    message = asyncio.run(scenario())

    # dense point 0 is the first planned waypoint and settles, so the run is
    # dense points 1 to 29 and the fifth stall falls on dense point 5, which
    # the message counts from one as waypoint 6 of 31
    assert len(handed) == _MAX_CONSECUTIVE_WAYPOINT_STALLS
    assert [len(points) for points in handed] == [
        _LONG_RUN_INSERTED - k for k in range(_MAX_CONSECUTIVE_WAYPOINT_STALLS)
    ]
    assert f"waypoint {_MAX_CONSECUTIVE_WAYPOINT_STALLS + 1}/{_LONG_RUN_INSERTED + 2}" in message
    assert f"{_MAX_CONSECUTIVE_WAYPOINT_STALLS} consecutive" in message
    assert stops == [1]


def test_move_through_joint_positions_rejects_a_wrong_length_waypoint_before_driving(world):
    arm = _arm(world, "arm-through-length")
    handle = arm._h()
    driven: list[list[float]] = []
    handle.set_joint_targets = lambda positions, max_vel_rad_s=None: driven.append(positions)

    async def scenario():
        with pytest.raises(ValueError, match=r"6.*5"):
            await arm.move_through_joint_positions(
                [
                    JointPositions(values=[0, 0, 0, 0, 0, 0]),
                    JointPositions(values=[10, 0, 0, 0, 0]),
                    JointPositions(values=[20, 0, 0, 0, 0, 0]),
                ]
            )

    asyncio.run(scenario())

    assert driven == []
