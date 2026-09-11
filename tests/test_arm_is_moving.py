"""Pins VEL_EPS_RAD_S itself: is_moving()'s velocity threshold must stay a
constant a test imports, not a literal that could drift unnoticed."""

from isaac_module.handles.arm import VEL_EPS_RAD_S, IsaacArmHandle
from test_arm_handle import FakeArticulation, FakeSim


def _make_handle(dof_names=("j0", "j1")) -> tuple[IsaacArmHandle, FakeArticulation]:
    art = FakeArticulation("fake-arm", list(dof_names))
    sim = FakeSim()
    handle = IsaacArmHandle(sim, art, None, joint_names=None)
    return handle, art


def test_is_moving_false_when_every_joint_velocity_is_below_threshold():
    handle, art = _make_handle()
    # VEL_EPS_RAD_S sits above PhysX's residual velocity noise on a resting
    # arm, so a value just under it must still read as "not moving".
    art.velocities = [VEL_EPS_RAD_S * 0.9, -VEL_EPS_RAD_S * 0.9]
    art.positions = [0.0, 0.0]
    handle._targets = [0.0, 0.0]

    assert handle.is_moving() is False


def test_is_moving_true_when_one_joint_velocity_exceeds_threshold():
    handle, art = _make_handle()
    art.velocities = [VEL_EPS_RAD_S * 0.9, VEL_EPS_RAD_S * 1.1]
    art.positions = [0.0, 0.0]
    handle._targets = [0.0, 0.0]

    assert handle.is_moving() is True
