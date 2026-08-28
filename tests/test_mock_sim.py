"""End-to-end test of the module in mock mode: boots the SimManager on a
background thread (standing in for the process main thread) and exercises
the viam component models against it."""

import asyncio
import math

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Vector3
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera

SETTLE_POLLS = 50
SETTLE_POLL_S = 0.01


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def test_world_boots_and_status(world):
    status = asyncio.run(world.do_command({"command": "status"}))
    assert status["booted"] is True
    assert status["mock"] is True
    assert status["isaac_version"] is None  # no Isaac here; a string like "5.0.0" on the VM


def test_validate_requires_world():
    with pytest.raises(ValueError, match="world"):
        IsaacArm.validate_config(_config("a", {"asset": "ur20"}))


def test_validate_requires_source():
    with pytest.raises(ValueError, match="asset"):
        IsaacArm.validate_config(_config("a", {"world": "sim-world"}))


def test_validate_ok_returns_dependency():
    deps, _ = IsaacArm.validate_config(_config("a", {"world": "sim-world", "asset": "ur20"}))
    assert list(deps) == ["sim-world"]


def test_arm_moves(world):
    arm = IsaacArm.new(
        _config("my-arm", {"world": "sim-world", "asset": "ur20", "mock_dof": 6}), {}
    )

    async def scenario():
        start = await arm.get_joint_positions()
        assert start.values == pytest.approx([0.0] * 6)

        from viam.components.arm import JointPositions

        target = JointPositions(values=[10, -20, 30, 0, 5, -5])
        await arm.move_to_joint_positions(target)
        # move_to_joint_positions returns once within 0.5 deg of the target while the
        # mock's is_moving() uses a 1e-9 tolerance, so the last few ms of travel can
        # still read as "moving" (FINDINGS R-7; unified in ARM-12, phase 3). Assert
        # that the arm settles, bounded, rather than that it is instantly still.
        for _ in range(SETTLE_POLLS):
            if not await arm.is_moving():
                break
            await asyncio.sleep(SETTLE_POLL_S)
        assert not await arm.is_moving()
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([10, -20, 30, 0, 5, -5], abs=0.5)

        # the mock's fixed end pose is defined in the arm base frame
        # (FINDINGS ARM-10), not world - see test_arm_frames.py
        pose = await arm.get_end_position()
        assert pose.x == pytest.approx(300.0)
        assert pose.o_z == pytest.approx(1.0)

        # trajectory execution (what the motion service calls)
        waypoints = [
            JointPositions(values=[5, -10, 15, 0, 2, -2]),
            JointPositions(values=[0, 0, 0, 0, 0, 0]),
        ]
        await arm.move_through_joint_positions(waypoints)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx([0] * 6, abs=0.5)

    asyncio.run(scenario())


def test_camera_returns_image(world):
    cam = IsaacCamera.new(
        _config("my-cam", {"world": "sim-world", "width": 320, "height": 240}), {}
    )

    async def scenario():
        images, _meta = await cam.get_images()
        assert len(images) == 1
        assert images[0].name == "color"
        assert images[0].mime_type == "image/png"

        from io import BytesIO

        from PIL import Image

        decoded = Image.open(BytesIO(images[0].data))
        assert decoded.size == (320, 240)

        props = await cam.get_properties()
        assert props.supports_pcd is False

    asyncio.run(scenario())


def test_base_drives(world):
    base = IsaacBase.new(_config("my-base", {"world": "sim-world", "asset": "jetbot"}), {})

    async def scenario():
        assert not await base.is_moving()
        await base.set_velocity(Vector3(x=0, y=200, z=0), Vector3(x=0, y=0, z=45))
        assert await base.is_moving()
        await base.stop()
        assert not await base.is_moving()

        # short timed move
        await base.move_straight(distance=10, velocity=100)
        assert not await base.is_moving()

        props = await base.get_properties()
        assert props.wheel_circumference_meters == pytest.approx(2 * math.pi * 0.05)

    asyncio.run(scenario())
