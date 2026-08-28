import math

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.utils import apply_frame_to_attrs, frame_pose
from isaac_module.spatial import quat_rotate


def _config(name: str, attrs: dict, parent: str | None = None) -> ComponentConfig:
    cfg = ComponentConfig(name=name, attributes=dict_to_struct(attrs))
    if parent is not None:
        cfg.frame.parent = parent
    return cfg


def test_no_frame():
    assert frame_pose(ComponentConfig(name="a")) == (None, None)


def test_translation_mm_to_meters():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 1000
    cfg.frame.translation.y = 2000
    cfg.frame.translation.z = 500
    pos, quat = frame_pose(cfg)
    assert pos == pytest.approx((1.0, 2.0, 0.5))
    assert quat is None  # no orientation set


def test_euler_yaw():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.orientation.euler_angles.yaw = math.pi / 2
    _, quat = frame_pose(cfg)
    assert quat_rotate(quat, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-9)


def test_quaternion_passthrough():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.orientation.quaternion.w = 1
    _, quat = frame_pose(cfg)
    assert quat == pytest.approx((1, 0, 0, 0))


def test_orientation_vector_degrees():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    ov = cfg.frame.orientation.vector_degrees
    ov.x, ov.y, ov.z, ov.theta = 0, 0, 1, 90
    _, quat = frame_pose(cfg)
    # OV (0,0,1) is identity direction; theta spins about it
    z = quat_rotate(quat, (0, 0, 1))
    assert z == pytest.approx((0, 0, 1), abs=1e-9)


def test_frame_wins_over_position_attr():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 3000
    attrs = apply_frame_to_attrs(cfg, {"position": [9, 9, 9]})
    assert attrs["position"] == [3.0, 0.0, 0.0]


def test_validate_frame_parent_world_ok():
    cfg = _config("a", {"world": "sim-world", "asset": "ur20"}, parent="world")
    deps, _ = IsaacArm.validate_config(cfg)
    assert list(deps) == ["sim-world"]


def test_validate_frame_parent_unset_ok():
    cfg = _config("a", {"world": "sim-world", "asset": "ur20"})
    deps, _ = IsaacArm.validate_config(cfg)
    assert list(deps) == ["sim-world"]


def test_validate_frame_parent_other_rejected():
    cfg = _config("a", {"world": "sim-world", "asset": "ur20"}, parent="table")
    with pytest.raises(ValueError, match="frame.parent"):
        IsaacArm.validate_config(cfg)


def test_validate_frame_parent_other_with_parent_prim_ok():
    cfg = _config(
        "a",
        {"world": "sim-world", "parent_prim": "/World/pick_arm"},
        parent="pick-arm",
    )
    deps, _ = IsaacCamera.validate_config(cfg)
    assert list(deps) == ["sim-world"]


def test_apply_frame_with_parent_prim_writes_local_position():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "pick-arm"
    cfg.frame.translation.z = 60
    attrs = apply_frame_to_attrs(cfg, {"parent_prim": "/World/pick_arm"})
    assert attrs["local_position"] == pytest.approx([0.0, 0.0, 0.06])
    assert attrs["local_orientation_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert "position" not in attrs
    assert "orientation_wxyz" not in attrs


def test_apply_frame_with_parent_prim_and_orientation():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "pick-arm"
    ov = cfg.frame.orientation.vector_degrees
    ov.x, ov.y, ov.z, ov.theta = 0, 0, 1, 90
    attrs = apply_frame_to_attrs(cfg, {"parent_prim": "/World/pick_arm"})
    _, expected_quat = frame_pose(cfg)
    assert attrs["local_orientation_wxyz"] == pytest.approx(list(expected_quat))
    assert "position" not in attrs
    assert "orientation_wxyz" not in attrs


def test_apply_frame_without_parent_prim_still_writes_position():
    cfg = ComponentConfig(name="a")
    cfg.frame.parent = "world"
    cfg.frame.translation.z = 60
    attrs = apply_frame_to_attrs(cfg, {})
    assert attrs["position"] == pytest.approx([0.0, 0.0, 0.06])
    assert "local_position" not in attrs


def test_validate_parent_prim_with_component_frame_parent_ok():
    cfg = _config(
        "a",
        {"world": "sim-world", "parent_prim": "/World/pick_arm/wrist_3_link"},
        parent="pick-arm",
    )
    deps, _ = IsaacCamera.validate_config(cfg)
    assert list(deps) == ["sim-world"]


def test_validate_parent_prim_with_link_frame_parent_ok():
    cfg = _config(
        "a",
        {"world": "sim-world", "parent_prim": "/World/pick_arm/wrist_3_link"},
        parent="pick-arm:ee_link",
    )
    deps, _ = IsaacCamera.validate_config(cfg)
    assert list(deps) == ["sim-world"]


def test_validate_parent_prim_wrong_owner_rejected():
    cfg = _config(
        "a",
        {"world": "sim-world", "parent_prim": "/World/pick_arm/wrist_3_link"},
        parent="other-arm",
    )
    with pytest.raises(ValueError, match="other-arm") as exc:
        IsaacCamera.validate_config(cfg)
    assert "pick_arm" in str(exc.value)


def test_validate_parent_prim_with_world_parent_rejected():
    cfg = _config(
        "a",
        {"world": "sim-world", "parent_prim": "/World/pick_arm"},
        parent="world",
    )
    with pytest.raises(ValueError):
        IsaacCamera.validate_config(cfg)


def test_validate_parent_prim_without_frame_rejected():
    cfg = _config("a", {"world": "sim-world", "parent_prim": "/World/pick_arm"})
    with pytest.raises(ValueError, match="frame.parent"):
        IsaacCamera.validate_config(cfg)


def test_validate_parent_prim_not_under_world_ok():
    cfg = _config(
        "a",
        {"world": "sim-world", "parent_prim": "/pick_arm/link"},
        parent="pick-arm",
    )
    deps, _ = IsaacCamera.validate_config(cfg)
    assert list(deps) == ["sim-world"]
