"""Unit tests for IsaacArm's Viam-facing contract in mock mode."""

import asyncio
import json
import time

import pytest
from grpclib import Status
from viam.components.arm import JointPositions, Pose
from viam.errors import MethodNotImplementedError
from viam.proto.app.robot import ComponentConfig
from viam.proto.component.arm import MoveOptions
from viam.utils import dict_to_struct

from isaac_module.errors import PrimNotFoundError
from isaac_module.models.arm import (
    ArmMoveStalledError,
    ArmMoveTimeoutError,
    IsaacArm,
    JointTargetOutOfLimitsError,
)
from isaac_module.sim_manager import UR_JOINT_NAMES


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _arm(world, name: str = "test-arm") -> IsaacArm:
    return IsaacArm.new(_config(name, {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {})


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


def test_move_to_position_raises_method_not_implemented(world):
    arm = _arm(world, "arm-move-to-position")

    async def scenario():
        with pytest.raises(MethodNotImplementedError):
            await arm.move_to_position(Pose())

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


def test_do_command_dof_names(world):
    arm = IsaacArm.new(
        _config("arm-dof-names", {"world": "sim-world", "asset": "ur20", "mock_dof": 12}), {}
    )

    async def scenario():
        result = await arm.do_command({"command": "dof_names"})
        names = result["dof_names"]
        assert len(names) == 12
        assert list(names[:6]) == list(UR_JOINT_NAMES)

    asyncio.run(scenario())


def test_do_command_prim_world_pose_default_prim(world):
    arm = IsaacArm.new(
        _config("arm-prim-pose", {"world": "sim-world", "asset": "ur5e", "mock_dof": 6}), {}
    )

    async def scenario():
        result = await arm.do_command({"command": "prim_world_pose"})
        # NOTE: the brief expected [-300, 0, 300] (root rotated by the ur5e
        # correction); MockArmHandle._ee_world_pose actually composes the
        # fixed local EE onto Viam's un-rotated base frame (it cancels the
        # correction out via viam_base_frame), so this is invariant to
        # base_frame_correction and always [300, 0, 300] here - see
        # Deviations in the slice report.
        assert result["position_mm"] == pytest.approx([300.0, 0.0, 300.0], abs=1e-3)
        assert len(result["quaternion_wxyz"]) == 4

    asyncio.run(scenario())


def test_do_command_prim_world_pose_unknown_prim_raises(world):
    arm = IsaacArm.new(
        _config("arm-prim-pose-unknown", {"world": "sim-world", "asset": "ur5e", "mock_dof": 6}),
        {},
    )

    async def scenario():
        with pytest.raises(PrimNotFoundError):
            await arm.do_command({"command": "prim_world_pose", "prim_path": "/World/nope"})

    asyncio.run(scenario())


def test_do_command_unknown_command_raises(world):
    arm = _arm(world, "arm-unknown-command")

    async def scenario():
        with pytest.raises(ValueError, match="unknown command"):
            await arm.do_command({"command": "not-a-real-command"})

    asyncio.run(scenario())


def test_do_command_all_dof_names(world):
    arm = IsaacArm.new(
        _config("arm-all-dof-names", {"world": "sim-world", "asset": "ur20", "mock_dof": 12}), {}
    )

    async def scenario():
        result = await arm.do_command({"command": "all_dof_names"})
        names = result["dof_names"]
        assert len(names) == 12
        assert list(names[:6]) == list(UR_JOINT_NAMES)

    asyncio.run(scenario())


def test_move_to_joint_positions_stall_raises_quickly(world):
    arm = IsaacArm.new(
        _config(
            "arm-stall",
            {"world": "sim-world", "asset": "ur20", "mock_dof": 6, "mock_stall_fraction": 0.5},
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
            {"world": "sim-world", "asset": "ur20", "mock_dof": 6, "move_timeout_sec": 0.2},
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
                "world": "sim-world",
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


def test_move_to_joint_positions_no_kinematics_skips_limit_check(world):
    arm = IsaacArm.new(
        _config("arm-no-kinematics", {"world": "sim-world", "asset": "franka", "mock_dof": 6}), {}
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
        _config("arm-through-default", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
    )
    arm_slow = IsaacArm.new(
        _config("arm-through-slow", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
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
    velocity limit honoured, and max_vel_degs_per_sec is ignored - not the
    other way around. Set a fast scalar (faster than the mock's own top
    speed, so it would be indistinguishable from "no limit" if it won) next
    to a slow per-joint limit; only honouring the per-joint value produces a
    move slower than the unlimited case."""
    arm_default = IsaacArm.new(
        _config("arm-priority-default", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
    )
    arm_both_set = IsaacArm.new(
        _config("arm-priority-both", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        target = JointPositions(values=[45, 0, 0, 0, 0, 0])
        options = MoveOptions(
            max_vel_degs_per_sec=1000.0,  # far above the mock's SPEED; a no-op if honoured
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
        _config("arm-is-moving", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
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


def test_failed_move_holds_position_instead_of_pushing(world):
    """GPU run 15: after a stall the drive target must be the current pose, so
    the arm stops pushing into whatever blocked it."""
    arm = IsaacArm.new(
        _config(
            "arm-hold-after-stall",
            {"world": "sim-world", "asset": "ur20", "mock_dof": 6, "mock_stall_fraction": 0.5},
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


def test_joint_state_do_command_reports_targets_next_to_positions(world):
    arm = _arm(world, "arm-joint-state")

    async def scenario():
        await arm.move_to_joint_positions(JointPositions(values=[10, -20, 30, 0, 5, -5]))
        out = await arm.do_command({"command": "joint_state"})
        joints = out["joints"]
        assert [j["name"] for j in joints] == list(UR_JOINT_NAMES)
        assert all(j["named"] for j in joints)
        assert [j["target_deg"] for j in joints] == pytest.approx([10, -20, 30, 0, 5, -5])
        assert [j["position_deg"] for j in joints] == pytest.approx(
            [10, -20, 30, 0, 5, -5], abs=0.5
        )

    asyncio.run(scenario())
