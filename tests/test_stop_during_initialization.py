"""Pins the cold-start contract: every stop() succeeds while the world is
still initializing, and every motion-commanding verb it sits beside still
raises SimInitializingError until the gate opens."""

import threading

import pytest
from grpclib import Status

from isaac_module.errors import SimInitializingError
from isaac_module.handles.arm import IsaacArmHandle
from isaac_module.handles.base import IsaacBaseHandle
from isaac_module.handles.gripper import IsaacGripperHandle
from isaac_module.sim_manager import SimManager


class _InlineTaskQueue:
    """Stands in for the sim thread: every queued task runs at once, so the
    test drives run()'s real off-thread path (the gate applies) without a
    sim thread."""

    def put(self, item) -> None:
        task, fut = item
        if fut.set_running_or_notify_cancel():
            try:
                fut.set_result(task())
            except BaseException as error:  # noqa: BLE001 - the future carries any failure to the caller
                fut.set_exception(error)

    def empty(self) -> bool:
        return True


def _initializing_manager() -> SimManager:
    manager = SimManager()
    manager._ready.clear()
    manager._sim_thread_id = threading.get_ident() + 1  # the caller is not the sim thread
    manager._tasks = _InlineTaskQueue()  # type: ignore[assignment]
    return manager


class FakeArticulationAction:
    def __init__(self, joint_positions=None, joint_indices=None) -> None:
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class FakeIsaacNamespace:
    ArticulationAction = FakeArticulationAction


class FakeArticulationController:
    def __init__(self, kps, kds) -> None:
        self._gains = (kps, kds)

    def get_gains(self):
        return self._gains

    def set_gains(self, kps, kds) -> None:
        self._gains = (kps, kds)


class FakeArticulation:
    def __init__(self, name: str, dof_names: list[str]) -> None:
        self.name = name
        self.dof_names = list(dof_names)
        self.positions = [0.0] * len(dof_names)
        self._controller = FakeArticulationController(
            [1.0] * len(dof_names), [1.0] * len(dof_names)
        )

    def get_articulation_controller(self):
        return self._controller

    def get_joint_positions(self, joint_indices=None):
        indices = range(len(self.positions)) if joint_indices is None else joint_indices
        return [self.positions[i] for i in indices]

    def apply_action(self, action: FakeArticulationAction) -> None:
        indices = (
            range(len(self.positions)) if action.joint_indices is None else action.joint_indices
        )
        for i, p in zip(indices, action.joint_positions, strict=True):
            self.positions[i] = float(p)


def test_base_set_velocity_gated_but_stop_and_is_moving_succeed():
    manager = _initializing_manager()
    base = IsaacBaseHandle(manager, robot=None, controller=None, wheel_radius=0.1, wheel_base=0.2)

    with pytest.raises(SimInitializingError) as excinfo:
        base.set_velocity(1.0, 0.5)
    assert excinfo.value.grpc_code == Status.UNAVAILABLE

    base._cmd = (1.0, 0.5)
    base.stop()

    assert base.is_moving() is False


def test_arm_stop_succeeds_while_set_joint_targets_is_gated():
    manager = _initializing_manager()
    manager._isaac = FakeIsaacNamespace()
    art = FakeArticulation("fake-arm", ["j0", "j1"])
    art.positions = [0.2, 0.4]
    arm = IsaacArmHandle(manager, art, None, joint_names=None)

    with pytest.raises(SimInitializingError) as excinfo:
        arm.set_joint_targets([0.9, 0.9])
    assert excinfo.value.grpc_code == Status.UNAVAILABLE

    arm.stop()

    assert art.positions == pytest.approx([0.2, 0.4])


def test_gripper_stop_succeeds_while_set_jaw_is_gated():
    manager = _initializing_manager()
    manager._isaac = FakeIsaacNamespace()
    art = FakeArticulation("fake-gripper", ["finger_joint"])
    art.positions = [0.3]
    gripper = IsaacGripperHandle(
        manager,
        art,
        drive_joint="finger_joint",
        open_rad=0.0,
        closed_rad=0.82,
        holding_tolerance_rad=0.01,
        prim_path="/World/fake_gripper",
    )

    with pytest.raises(SimInitializingError) as excinfo:
        gripper.set_jaw(0.5)
    assert excinfo.value.grpc_code == Status.UNAVAILABLE

    gripper.stop()

    assert art.positions == pytest.approx([0.3])


def test_gripper_stop_clamps_a_measured_overshoot_past_the_closed_limit():
    manager = _initializing_manager()
    manager._isaac = FakeIsaacNamespace()
    art = FakeArticulation("fake-gripper", ["finger_joint"])
    closed_rad = 0.82
    art.positions = [closed_rad + 0.05]  # a physics overshoot past the limit
    gripper = IsaacGripperHandle(
        manager,
        art,
        drive_joint="finger_joint",
        open_rad=0.0,
        closed_rad=closed_rad,
        holding_tolerance_rad=0.01,
        prim_path="/World/fake_gripper",
    )

    gripper.stop()

    assert gripper._target == closed_rad
