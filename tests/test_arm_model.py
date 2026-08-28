"""Unit tests for IsaacArm's Viam-facing contract in mock mode."""

import asyncio

import pytest
from viam.components.arm import JointPositions, Pose
from viam.errors import MethodNotImplementedError
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.errors import PrimNotFoundError
from isaac_module.models.arm import IsaacArm
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
