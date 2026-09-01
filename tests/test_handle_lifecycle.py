"""XC-4 policy: close() releases a component's cached handle (and any
post-reset hooks it registered) while leaving the prim in the stage; a
config change to a SPAWN attribute after that is rejected until the module
restarts, while RUNTIME_KEYS re-apply silently and the same handle is kept."""

import asyncio

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.sim_manager import SimManager


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _config_with_frame_translation(name: str, attrs: dict, z_mm: float) -> ComponentConfig:
    cfg = _config(name, attrs)
    cfg.frame.parent = "world"
    cfg.frame.translation.z = z_mm
    return cfg


# -- arm ----------------------------------------------------------------


def test_arm_close_forgets_handle_and_next_create_makes_a_new_one(world):
    arm = IsaacArm.new(_config("close-arm-1", {"world": "sim-world", "asset": "ur20"}), {})
    first_handle = arm._handle
    assert "close-arm-1" in SimManager.get()._handles

    asyncio.run(arm.close())
    assert "close-arm-1" not in SimManager.get()._handles
    assert arm._handle is None

    arm2 = IsaacArm.new(_config("close-arm-1", {"world": "sim-world", "asset": "ur20"}), {})
    assert arm2._handle is not first_handle


def test_arm_close_is_idempotent(world):
    arm = IsaacArm.new(_config("close-arm-2", {"world": "sim-world", "asset": "ur20"}), {})
    asyncio.run(arm.close())
    asyncio.run(arm.close())  # must not raise
    assert "close-arm-2" not in SimManager.get()._handles


def test_arm_reconfigure_rejects_changed_spawn_attribute(world):
    arm = IsaacArm.new(
        _config("reconf-arm-1", {"world": "sim-world", "asset": "ur20", "position": [0, 0, 0]}),
        {},
    )
    changed = _config(
        "reconf-arm-1", {"world": "sim-world", "asset": "ur20", "position": [1, 0, 0]}
    )
    with pytest.raises(ValueError, match="restart the module"):
        arm.reconfigure(changed, {})


def test_arm_reconfigure_allows_changed_runtime_attribute_and_keeps_handle(world):
    arm = IsaacArm.new(
        _config(
            "reconf-arm-2",
            {"world": "sim-world", "asset": "ur20", "move_timeout_sec": 10},
        ),
        {},
    )
    original_handle = arm._handle
    changed = _config(
        "reconf-arm-2",
        {"world": "sim-world", "asset": "ur20", "move_timeout_sec": 99},
    )
    arm.reconfigure(changed, {})  # must not raise
    assert arm._handle is original_handle


def test_arm_reconfigure_rejects_changed_frame_translation(world):
    arm = IsaacArm.new(
        _config_with_frame_translation(
            "reconf-arm-3", {"world": "sim-world", "asset": "ur20"}, 0.0
        ),
        {},
    )
    changed = _config_with_frame_translation(
        "reconf-arm-3", {"world": "sim-world", "asset": "ur20"}, 60.0
    )
    with pytest.raises(ValueError, match="restart the module"):
        arm.reconfigure(changed, {})


def test_cached_handle_error_lists_runtime_keys(world):
    arm = IsaacArm.new(
        _config("reconf-arm-4", {"world": "sim-world", "asset": "ur20", "position": [0, 0, 0]}),
        {},
    )
    changed = _config(
        "reconf-arm-4", {"world": "sim-world", "asset": "ur20", "position": [1, 0, 0]}
    )
    with pytest.raises(ValueError, match="move_timeout_sec"):
        arm.reconfigure(changed, {})


# -- camera ---------------------------------------------------------------


def test_camera_close_forgets_handle_and_drops_post_reset_hook(world):
    sim = SimManager.get()
    camera = IsaacCamera.new(_config("close-cam-1", {"world": "sim-world"}), {})
    assert "close-cam-1" in sim._handles

    call_count = 0

    def _count_post_reset():
        nonlocal call_count
        call_count += 1

    camera._handle.post_reset = _count_post_reset

    sim.reset()
    assert call_count == 1

    asyncio.run(camera.close())
    assert "close-cam-1" not in sim._handles

    sim.reset()
    assert call_count == 1  # the hook was dropped, not fired again


# -- base -------------------------------------------------------------


def test_base_close_forgets_handle(world):
    base = IsaacBase.new(
        _config(
            "close-base-1",
            {"world": "sim-world", "asset": "jetbot"},
        ),
        {},
    )
    assert "close-base-1" in SimManager.get()._handles

    asyncio.run(base.close())
    assert "close-base-1" not in SimManager.get()._handles
    assert base._handle is None


# -- gripper (implemented by a sibling; close() is already in the seam) ---


def test_gripper_close_forgets_handle(world):
    arm = IsaacArm.new(_config("gripper-host-arm", {"world": "sim-world", "asset": "ur20"}), {})
    gripper = IsaacGripper.new(
        _config("close-gripper-1", {"world": "sim-world", "arm": arm.name}), {}
    )

    assert "close-gripper-1" in SimManager.get()._handles
    asyncio.run(gripper.close())
    assert "close-gripper-1" not in SimManager.get()._handles
