import math

import pytest
from viam.proto.app.robot import ComponentConfig

from isaac_module.models.utils import apply_frame_to_attrs, frame_pose
from isaac_module.spatial import quat_rotate


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
