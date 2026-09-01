"""ArmHandle/GripperHandle contract, parametrised over mock and isaac
backends (FINDINGS ARM-18; OQ-5), plus a mock pick rehearsal at the model
level and a GPU-only DOF-count smoke.

Handle-level tests speak radians (the handle contract); the model-level
rehearsal speaks degrees (viam.components.arm.JointPositions) - never mixed.
"""

import itertools
import math
import threading
import time

import pytest
from viam.components.arm import JointPositions
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import compat
from isaac_module.models.arm import IsaacArm
from isaac_module.models.gripper import IsaacGripper
from isaac_module.sim_manager import (
    UR_JOINT_NAMES,
    MockArmHandle,
    SettleOutcome,
    SimConfig,
    SimManager,
)

SIM_THREAD_JOIN_TIMEOUT_S = 5
ARM_SETTLE_TIMEOUT_S = 10.0
ARM_TOLERANCE_ABS = {"mock": 1e-3, "isaac": 0.02}

_names = itertools.count()


def _unique(prefix: str) -> str:
    return f"{prefix}-{next(_names)}"


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _poll_until_not_moving(is_moving, timeout_s: float, poll_s: float) -> bool:
    """Bounded poll (MockArmHandle.STEP_S-sized on mock) - never a bare
    wall-clock sleep."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_moving():
            return True
        time.sleep(poll_s)
    return not is_moving()


@pytest.fixture(scope="module")
def isaac_sim():
    """A fresh, real (non-mock) SimManager - only ever reached under
    ``-m gpu`` on the GPU machine, per the ``backend`` fixture below."""
    manager = SimManager.get()
    sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
    sim_thread.start()
    manager.ensure_booted(SimConfig(mock=False, headless=True, livestream=False))
    yield manager
    manager.request_stop()
    sim_thread.join(timeout=SIM_THREAD_JOIN_TIMEOUT_S)


@pytest.fixture(
    params=[
        pytest.param("mock", id="mock"),
        pytest.param("isaac", marks=pytest.mark.gpu, id="isaac"),
    ]
)
def backend(request):
    if request.param == "mock":
        return request.param, request.getfixturevalue("sim")
    if compat.isaac_version() is None:
        pytest.skip("not running inside Isaac Sim's python")
    return request.param, request.getfixturevalue("isaac_sim")


@pytest.fixture
def arm_and_gripper(backend):
    backend_name, sim = backend
    arm_name = _unique(f"contract-arm-{backend_name}")
    gripper_name = _unique(f"contract-gripper-{backend_name}")
    arm = sim.create_arm(arm_name, {"world": "sim-world", "asset": "ur5e", "position": [0, 0, 0]})
    gripper = sim.create_gripper(
        gripper_name,
        {"world": "sim-world", "arm": arm_name, "mock_object_width_m": 0.05},
    )
    return backend_name, arm, gripper


# ----------------------------------------------------------------------
# ArmHandle contract
# ----------------------------------------------------------------------


def test_arm_dof_names_are_the_six_ur_joints(arm_and_gripper):
    _backend, arm, _gripper = arm_and_gripper
    assert arm.dof_names() == list(UR_JOINT_NAMES)


def test_arm_all_dof_names_is_superset_of_dof_names(arm_and_gripper):
    _backend, arm, _gripper = arm_and_gripper
    all_names = arm.all_dof_names()
    assert set(UR_JOINT_NAMES).issubset(set(all_names))


def test_arm_set_joint_targets_reaches_and_settles(arm_and_gripper):
    backend_name, arm, _gripper = arm_and_gripper
    # nonzero, small enough to be safe on a real UR5e, large enough that
    # "no movement" would fail the approx below.
    targets = [math.radians(5.0)] * len(UR_JOINT_NAMES)

    arm.set_joint_targets(targets)
    outcome = arm.wait_for_settle(ARM_SETTLE_TIMEOUT_S)
    assert outcome == SettleOutcome.REACHED

    measured = arm.get_joint_positions()
    tolerance = ARM_TOLERANCE_ABS[backend_name]
    assert measured == pytest.approx(targets, abs=tolerance)
    assert arm.is_moving() is False


def test_arm_stop_then_not_moving(arm_and_gripper):
    _backend, arm, _gripper = arm_and_gripper
    arm.stop()
    assert arm.is_moving() is False


def test_arm_release_twice_does_not_raise(arm_and_gripper):
    _backend, arm, _gripper = arm_and_gripper
    arm.release()
    arm.release()


# ----------------------------------------------------------------------
# GripperHandle contract
# ----------------------------------------------------------------------


def test_gripper_jaw_limits_open_before_closed(arm_and_gripper):
    _backend, _arm, gripper = arm_and_gripper
    open_rad, closed_rad = gripper.jaw_limits()
    assert open_rad < closed_rad


def test_gripper_open_then_close_moves_the_jaw(arm_and_gripper):
    _backend, _arm, gripper = arm_and_gripper
    open_rad, _closed_rad = gripper.jaw_limits()

    gripper.open()
    assert _poll_until_not_moving(gripper.is_moving, ARM_SETTLE_TIMEOUT_S, MockArmHandle.STEP_S)
    assert gripper.get_jaw() == pytest.approx(open_rad, abs=1e-3)

    gripper.close()
    assert _poll_until_not_moving(gripper.is_moving, ARM_SETTLE_TIMEOUT_S, MockArmHandle.STEP_S)
    assert gripper.get_jaw() > open_rad


def test_gripper_is_holding_returns_a_bool(arm_and_gripper):
    _backend, _arm, gripper = arm_and_gripper
    assert isinstance(gripper.is_holding(), bool)


def test_gripper_dof_names_nonempty_finger_joint_first(arm_and_gripper):
    _backend, _arm, gripper = arm_and_gripper
    names = gripper.dof_names()
    assert names
    assert names[0] == "finger_joint"


def test_gripper_release_twice_does_not_raise(arm_and_gripper):
    _backend, _arm, gripper = arm_and_gripper
    gripper.release()
    gripper.release()


# ----------------------------------------------------------------------
# Mock pick rehearsal at the model level (mock only) - degrees, JointPositions
# ----------------------------------------------------------------------


def test_mock_pick_rehearsal(world):
    arm_name = _unique("rehearsal-arm")
    gripper_name = _unique("rehearsal-gripper")

    arm = IsaacArm.new(
        _config(arm_name, {"world": "sim-world", "asset": "ur5e", "position": [0, 0, 0]}), {}
    )
    gripper = IsaacGripper.new(
        _config(
            gripper_name,
            {"world": "sim-world", "arm": arm_name, "mock_object_width_m": 0.05},
        ),
        {},
    )

    pre_grasp = JointPositions(values=[10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
    lift = JointPositions(values=[20.0, -20.0, 20.0, -20.0, 20.0, -20.0])

    async def scenario():
        await gripper.open()

        await arm.move_to_joint_positions(pre_grasp)
        before_grasp = await arm.get_joint_positions()

        assert await gripper.grab() is True
        status = await gripper.is_holding_something()
        assert status.is_holding_something is True

        await arm.move_to_joint_positions(lift)
        after_lift = await arm.get_joint_positions()
        assert after_lift.values != pytest.approx(before_grasp.values)

        status = await gripper.is_holding_something()
        assert status.is_holding_something is True

        await gripper.open()
        for _ in range(500):
            if not await gripper.is_moving():
                break
            time.sleep(MockArmHandle.STEP_S)
        status = await gripper.is_holding_something()
        assert status.is_holding_something is False

    import asyncio

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# GPU smoke (OQ-5 / R-3): the gripper's DOFs join the arm's articulation
# ----------------------------------------------------------------------


@pytest.mark.gpu
def test_arm_dof_count_with_gripper_attached(isaac_sim):
    if compat.isaac_version() is None:
        pytest.skip("not running inside Isaac Sim's python")

    arm_name = _unique("gpu-smoke-arm")
    gripper_name = _unique("gpu-smoke-gripper")
    arm = isaac_sim.create_arm(
        arm_name, {"world": "sim-world", "asset": "ur5e", "position": [0, 0, 0]}
    )
    isaac_sim.create_gripper(
        gripper_name, {"world": "sim-world", "arm": arm_name, "mock_object_width_m": 0.05}
    )

    all_names = arm.all_dof_names()
    print(f"gpu smoke: arm {arm_name!r} all_dof_names() = {all_names}")

    assert len(all_names) == 6 + compat.caps().gripper_dof_count
    assert len(arm.get_joint_positions()) == 6
