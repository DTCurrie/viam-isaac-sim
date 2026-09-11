import asyncio

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.errors import PrimNotFoundError
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.gripper import IsaacGripper


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def test_do_command_prim_pose_default_prim_is_the_bases_own_root(world):
    IsaacBase.new(
        _config(
            "base_prim_pose",
            {
                "world": "isaac-world",
                "asset": "jetbot",
                "position": [0.3, -0.2, 0.1],
                "orientation_wxyz": [0.0, 0.0, 0.0, 1.0],
            },
        ),
        {},
    )

    async def scenario():
        return await world.do_command({"command": "prim_pose", "name": "base_prim_pose"})

    result = asyncio.run(scenario())
    assert result["prim_path"] == "/World/base_prim_pose"
    assert result["position_mm"] == pytest.approx([300.0, -200.0, 100.0], abs=1e-3)
    assert result["quaternion_wxyz"] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)


def test_do_command_prim_pose_explicit_prim_matches_the_default(world):
    IsaacBase.new(
        _config("base_prim_pose_explicit", {"world": "isaac-world", "asset": "jetbot"}),
        {},
    )

    async def scenario():
        return await world.do_command(
            {
                "command": "prim_pose",
                "name": "base_prim_pose_explicit",
                "prim_path": "/World/base_prim_pose_explicit",
            }
        )

    result = asyncio.run(scenario())
    assert result["position_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert result["quaternion_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_do_command_prim_pose_unknown_prim_raises(world):
    IsaacBase.new(
        _config("base_prim_pose_unknown", {"world": "isaac-world", "asset": "jetbot"}),
        {},
    )

    async def scenario():
        await world.do_command(
            {
                "command": "prim_pose",
                "name": "base_prim_pose_unknown",
                "prim_path": "/World/nope",
            }
        )

    with pytest.raises(PrimNotFoundError):
        asyncio.run(scenario())


def test_do_command_prim_pose_wrong_kind_names_both_arm_and_base(world):
    arm = IsaacArm.new(
        _config("base_prim_pose_wrong_kind_arm", {"world": "isaac-world", "asset": "ur5e"}),
        {},
    )
    IsaacGripper.new(
        _config("base_prim_pose_wrong_kind_grip", {"world": "isaac-world", "arm": arm.name}), {}
    )

    async def scenario():
        await world.do_command({"command": "prim_pose", "name": "base_prim_pose_wrong_kind_grip"})

    with pytest.raises(ValueError, match="not an arm or base"):
        asyncio.run(scenario())
