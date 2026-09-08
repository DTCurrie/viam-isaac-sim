"""The `world` attribute defaults to DEFAULT_WORLD_NAME across every model
that needs the sim, so swapping a real driver for its sim model requires no
extra attribute beyond the asset (see docs/SIMULATION.md)."""

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import DEFAULT_WORLD_NAME
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.conductor import IsaacConductor
from isaac_module.models.gripper import IsaacGripper


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def test_arm_defaults_world():
    deps, _ = IsaacArm.validate_config(_config("a", {"asset": "ur20"}))
    assert list(deps) == [DEFAULT_WORLD_NAME]


def test_arm_explicit_world_wins():
    deps, _ = IsaacArm.validate_config(_config("a", {"world": "other-world", "asset": "ur20"}))
    assert list(deps) == ["other-world"]


def test_arm_non_string_world_raises():
    with pytest.raises(ValueError, match="world"):
        IsaacArm.validate_config(_config("a", {"world": 5, "asset": "ur20"}))


def test_arm_empty_world_raises():
    with pytest.raises(ValueError, match="world"):
        IsaacArm.validate_config(_config("a", {"world": "", "asset": "ur20"}))


def test_camera_defaults_world():
    deps, _ = IsaacCamera.validate_config(_config("c", {}))
    assert list(deps) == [DEFAULT_WORLD_NAME]


def test_base_defaults_world():
    deps, _ = IsaacBase.validate_config(_config("b", {"asset": "jetbot"}))
    assert list(deps) == [DEFAULT_WORLD_NAME]


def test_gripper_defaults_world():
    deps, _ = IsaacGripper.validate_config(_config("g", {"arm": "pick-arm"}))
    assert list(deps) == [DEFAULT_WORLD_NAME, "pick-arm"]


def test_conductor_defaults_world():
    deps, _ = IsaacConductor.validate_config(
        _config(
            "block-sorter",
            {
                "arm": "pick-arm",
                "gripper": "pick-grip",
                "camera": "wrist-cam",
                "side_camera": "side-cam",
                "motion": "builtin",
                "detectors": {
                    "red": "red-segmenter",
                    "green": "green-segmenter",
                    "blue": "blue-segmenter",
                    "yellow": "yellow-segmenter",
                    "purple": "purple-segmenter",
                    "orange": "orange-segmenter",
                },
            },
        )
    )
    assert list(deps)[0] == DEFAULT_WORLD_NAME
