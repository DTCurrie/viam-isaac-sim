"""DOF-index-aware joint I/O on the arm handles (FINDINGS ARM-1; R-3):
ArmHandle.dof_names() exposes the full articulation, while
get_joint_positions/set_joint_targets/is_moving/stop operate on exactly the
arm's named joints, selected by index rather than position - so attaching a
gripper later cannot shift or truncate arm joints."""

import asyncio

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import UR_JOINT_NAMES, resolve_joint_indices

SETTLE_POLLS = 200
SETTLE_POLL_S = 0.01


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def test_mock_ur_arm_dof_names_and_padding(world):
    arm = IsaacArm.new(
        _config("padded-arm", {"world": "sim-world", "asset": "ur5e", "mock_dof": 12}), {}
    )
    handle = arm._handle
    names = handle.dof_names()
    assert len(names) == 12
    assert tuple(names[:6]) == UR_JOINT_NAMES

    async def scenario():
        start = await arm.get_joint_positions()
        assert len(start.values) == 6

    asyncio.run(scenario())


def test_mock_ur_arm_moves_selected_joints_only_padding_stays_zero(world):
    arm = IsaacArm.new(
        _config("padded-move-arm", {"world": "sim-world", "asset": "ur5e", "mock_dof": 12}),
        {},
    )
    handle = arm._handle

    async def scenario():
        await asyncio.to_thread(handle.set_joint_targets, [0.1] * 6)
        for _ in range(SETTLE_POLLS):
            if not await asyncio.to_thread(handle.is_moving):
                break
            await asyncio.sleep(SETTLE_POLL_S)
        assert not await asyncio.to_thread(handle.is_moving)

        end = await asyncio.to_thread(handle.get_joint_positions)
        assert end == pytest.approx([0.1] * 6, abs=1e-6)

        all_positions = handle.get_all_joint_positions()
        assert all_positions[:6] == pytest.approx([0.1] * 6, abs=1e-6)
        assert all_positions[6:] == pytest.approx([0.0] * 6, abs=1e-9)

    asyncio.run(scenario())


def test_mock_franka_arm_has_no_declared_joint_names(world):
    arm = IsaacArm.new(
        _config("franka-arm", {"world": "sim-world", "asset": "franka", "mock_dof": 7}), {}
    )
    handle = arm._handle
    assert len(handle.dof_names()) == 7

    async def scenario():
        positions = await arm.get_joint_positions()
        assert len(positions.values) == 7

    asyncio.run(scenario())


def test_mock_ur_arm_rejects_length_mismatch(world):
    arm = IsaacArm.new(
        _config("mismatch-arm", {"world": "sim-world", "asset": "ur5e", "mock_dof": 12}), {}
    )
    handle = arm._handle
    with pytest.raises(ValueError, match="6") as excinfo:
        handle.set_joint_targets([0.1] * 5)
    assert "5" in str(excinfo.value)


def test_resolve_joint_indices_maps_by_name_not_position():
    dof_names = [
        "wrist_2_joint",
        "shoulder_pan_joint",
        "wrist_3_joint",
        "shoulder_lift_joint",
        "wrist_1_joint",
        "elbow_joint",
    ]
    indices = resolve_joint_indices(dof_names, UR_JOINT_NAMES)
    assert indices == [1, 3, 5, 4, 0, 2]


def test_resolve_joint_indices_none_when_no_names_declared():
    assert resolve_joint_indices(["a", "b"], None) is None


def test_resolve_joint_indices_raises_naming_missing_joint():
    dof_names = ["shoulder_pan_joint", "shoulder_lift_joint"]
    with pytest.raises(ValueError, match="elbow_joint") as excinfo:
        resolve_joint_indices(dof_names, UR_JOINT_NAMES)
    assert "shoulder_pan_joint" in str(excinfo.value)
