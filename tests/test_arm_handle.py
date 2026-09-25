import asyncio
import threading
import time

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.handles.arm import (
    SETTLE_WINDOW_STEPS,
    STALL_NO_PROGRESS_STEPS,
    STALL_STILL_STEPS,
    IsaacArmHandle,
    SettleOutcome,
    resolve_joint_indices,
)
from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import UR_JOINT_NAMES

SETTLE_POLLS = 200
SETTLE_POLL_S = 0.01


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def test_mock_ur_arm_dof_names_and_padding(world):
    arm = IsaacArm.new(
        _config("padded-arm", {"world": "isaac-world", "asset": "ur5e", "mock_dof": 12}), {}
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
        _config("padded-move-arm", {"world": "isaac-world", "asset": "ur5e", "mock_dof": 12}),
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
        _config("franka-arm", {"world": "isaac-world", "asset": "franka", "mock_dof": 7}), {}
    )
    handle = arm._handle
    assert len(handle.dof_names()) == 7

    async def scenario():
        positions = await arm.get_joint_positions()
        assert len(positions.values) == 7

    asyncio.run(scenario())


def test_mock_ur_arm_rejects_length_mismatch(world):
    arm = IsaacArm.new(
        _config("mismatch-arm", {"world": "isaac-world", "asset": "ur5e", "mock_dof": 12}), {}
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


# ----------------------------------------------------------------------
# IsaacArmHandle against a fake articulation/world
# (no real Isaac Sim). FakeSim.run(fn) calls fn() inline - the tests drive
# the registered physics callback directly from a helper thread to simulate
# the sim thread stepping while wait_for_settle blocks the caller thread.
# ----------------------------------------------------------------------

CALLBACK_WAIT_POLLS = 200
CALLBACK_WAIT_POLL_S = 0.005
FAKE_PHYSICS_STEP_S = 0.01  # step_size passed to the settle callback each fake step


class FakeArticulationAction:
    def __init__(self, joint_positions=None, joint_indices=None) -> None:
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class FakeArticulationController:
    def __init__(self, kps, kds) -> None:
        self._gains = (kps, kds)
        self.set_gains_calls: list[tuple] = []

    def get_gains(self):
        return self._gains

    def set_gains(self, kps, kds) -> None:
        self._gains = (kps, kds)
        self.set_gains_calls.append((kps, kds))


class FakeArticulationView:
    """Stands in for SingleArticulation._articulation_view, the batched view
    5.0 actually defines set_max_joint_velocities / get_joint_max_velocities
    on (SingleArticulation itself has neither)."""

    def __init__(self, dof_count: int, authored_max: float = 5.0) -> None:
        self.max_velocities = [authored_max] * dof_count
        self.set_calls: list[tuple[list[float], list[int] | None]] = []
        self.get_calls: list[list[int] | None] = []
        self.armatures = [0.0] * dof_count

    def get_armatures(self, joint_indices=None):
        indices = range(len(self.armatures)) if joint_indices is None else joint_indices
        return [self.armatures[i] for i in indices]

    def set_armatures(self, values, joint_indices=None) -> None:
        indices = range(len(self.armatures)) if joint_indices is None else joint_indices
        for i, v in zip(indices, values, strict=True):
            self.armatures[i] = float(v)

    def set_max_joint_velocities(self, velocities, joint_indices=None) -> None:
        indices = range(len(self.max_velocities)) if joint_indices is None else joint_indices
        for i, v in zip(indices, velocities, strict=True):
            self.max_velocities[i] = float(v)
        self.set_calls.append(([float(v) for v in velocities], joint_indices))

    def get_joint_max_velocities(self, joint_indices=None):
        indices = range(len(self.max_velocities)) if joint_indices is None else joint_indices
        self.get_calls.append(joint_indices)
        return [self.max_velocities[i] for i in indices]


class FakeArticulation:
    def __init__(self, name: str, dof_names: list[str]) -> None:
        self.name = name
        self.dof_names = list(dof_names)
        dof_count = len(dof_names)
        self.positions = [0.0] * dof_count
        self.velocities = [0.0] * dof_count
        self._controller = FakeArticulationController([1.0] * dof_count, [1.0] * dof_count)
        self.solver_iteration_calls: list[int] = []
        self.applied_actions: list[FakeArticulationAction] = []
        self.state_write_log: list[tuple[str, list[float]]] = []
        self._articulation_view = FakeArticulationView(dof_count)

    def get_articulation_controller(self):
        return self._controller

    def set_solver_position_iteration_count(self, count: int) -> None:
        self.solver_iteration_calls.append(count)

    def get_joint_positions(self, joint_indices=None):
        indices = range(len(self.positions)) if joint_indices is None else joint_indices
        return [self.positions[i] for i in indices]

    def get_joint_velocities(self, joint_indices=None):
        indices = range(len(self.velocities)) if joint_indices is None else joint_indices
        return [self.velocities[i] for i in indices]

    def set_joint_positions(self, positions, joint_indices=None) -> None:
        indices = range(len(self.positions)) if joint_indices is None else joint_indices
        for i, p in zip(indices, positions, strict=True):
            self.positions[i] = float(p)
        self.state_write_log.append(("positions", [float(p) for p in positions]))

    def set_joint_velocities(self, velocities, joint_indices=None) -> None:
        indices = range(len(self.velocities)) if joint_indices is None else joint_indices
        for i, v in zip(indices, velocities, strict=True):
            self.velocities[i] = float(v)
        self.state_write_log.append(("velocities", [float(v) for v in velocities]))

    def apply_action(self, action: FakeArticulationAction) -> None:
        self.applied_actions.append(action)
        self.state_write_log.append(("action", [float(p) for p in action.joint_positions]))
        indices = (
            range(len(self.positions)) if action.joint_indices is None else action.joint_indices
        )
        for i, p in zip(indices, action.joint_positions, strict=True):
            self.positions[i] = p


class FakeScene:
    def __init__(self) -> None:
        self.objects: dict[str, object] = {}
        self.removed: list[tuple[str, bool]] = []

    def get_object(self, name: str):
        return self.objects.get(name)

    def remove_object(self, name: str, registry_only: bool = False) -> None:
        self.removed.append((name, registry_only))
        self.objects.pop(name, None)


class FakeWorld:
    def __init__(self) -> None:
        self._callbacks: dict[str, object] = {}
        self.scene = FakeScene()

    def add_physics_callback(self, name: str, fn) -> None:
        self._callbacks[name] = fn

    def remove_physics_callback(self, name: str) -> None:
        self._callbacks.pop(name, None)

    def physics_callback_exists(self, name: str) -> bool:
        return name in self._callbacks


class FakeIsaacNamespace:
    ArticulationAction = FakeArticulationAction


class FakeSim:
    def __init__(self) -> None:
        self.world = FakeWorld()
        self._isaac = FakeIsaacNamespace()
        self.run_calls: list[bool] = []

    def run(self, fn, timeout: float = 30.0, *, allow_during_initialization: bool = False):
        self.run_calls.append(allow_during_initialization)
        return fn()


def _make_handle(dof_names=("j0", "j1")) -> tuple[IsaacArmHandle, FakeArticulation, FakeSim]:
    art = FakeArticulation("fake-arm", list(dof_names))
    sim = FakeSim()
    sim.world.scene.objects[art.name] = art
    handle = IsaacArmHandle(sim, art, None, joint_names=None)
    return handle, art, sim


def _wait_until(predicate, polls: int = CALLBACK_WAIT_POLLS, poll_s: float = CALLBACK_WAIT_POLL_S):
    for _ in range(polls):
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def test_isaac_velocity_cap_is_set_and_restored_via_the_articulation_view():
    handle, art, _sim = _make_handle()
    view = art._articulation_view

    handle.set_joint_targets([0.2, 0.3], max_vel_rad_s=1.0)

    assert view.set_calls[-1] == ([1.0, 1.0], None)
    assert view.get_calls  # the pre-cap authored limits were read to snapshot for restore

    handle.set_joint_targets([0.4, 0.5])  # max_vel_rad_s=None -> restore

    assert view.set_calls[-1] == ([5.0, 5.0], None)


def test_isaac_stop_is_one_run_call_allowed_during_initialization():
    handle, art, sim = _make_handle()
    art.positions = [0.3, 0.7]
    sim.run_calls.clear()

    handle.stop()

    assert sim.run_calls == [True]
    assert art.positions == pytest.approx([0.3, 0.7])
    assert handle._targets == pytest.approx([0.3, 0.7])


def test_isaac_settle_reaches_after_window_not_before():
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.0, 0.0]
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=5.0)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # a broken (no-window) implementation would report REACHED on step 1
    for _ in range(SETTLE_WINDOW_STEPS - 1):
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.REACHED
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_wait_until_within_reaches_on_the_first_step_inside_tolerance():
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.01, -0.01]
    art.velocities = [0.0, 0.0]
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_until_within(5.0, 0.05)),
        daemon=True,
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    sim.world._callbacks[callback_name](FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    # wait_for_settle on this same input needs SETTLE_WINDOW_STEPS steps
    assert result["outcome"] == SettleOutcome.REACHED
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_wait_until_within_stalls_when_still_and_off_target():
    handle, art, sim = _make_handle()
    handle._targets = [1.0, 1.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.0, 0.0]
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_until_within(5.0, 0.05)),
        daemon=True,
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]
    for _ in range(STALL_STILL_STEPS - 1):
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.STALLED


def test_mock_wait_until_within_reaches_while_the_arm_is_still_moving():
    from isaac_module.sim_manager import MockArmHandle, SettleOutcome

    handle = MockArmHandle("within-moving-arm", {"asset": "ur5e"})
    handle.set_joint_targets([1.0] + [0.0] * (len(handle.get_joint_positions()) - 1))
    # far from the target and moving: a tolerance wider than the whole move is met at once
    assert handle.wait_until_within(5.0, 2.0) == SettleOutcome.REACHED
    assert handle.is_moving() is True


def test_isaac_settle_stalls_when_still_but_off_target():
    handle, art, sim = _make_handle()
    handle._targets = [1.0, 1.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.0, 0.0]  # still, but never converges
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=5.0)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # a joint reversing direction reads still for a few steps; only a run of
    # STALL_STILL_STEPS is a pinned arm
    for _ in range(STALL_STILL_STEPS - 1):
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.STALLED
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_settle_times_out_while_still_converging():
    handle, art, sim = _make_handle()
    handle._targets = [1.0, 1.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.5, 0.5]  # still moving - never "still", never stalled
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=0.05)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # step_size accumulation crosses timeout_s (0.05s) well before the
    # window's worth of steps would (it never reaches the window anyway).
    for _ in range(10):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.TIMED_OUT
    assert not sim.world.physics_callback_exists(callback_name)


def _hold_commands_without_moving(art: FakeArticulation) -> None:
    """Make apply_action record the command and leave the joints wherever the
    test put them, so a test decides when a point is reached."""
    art.apply_action = art.applied_actions.append


def _start_follow(
    handle: IsaacArmHandle,
    sim: FakeSim,
    points: list[list[float]],
    max_vel_rad_s: float | None = None,
    tolerance_rad: float = 0.05,
    point_timeout_s: float = 5.0,
):
    """Run follow_path on a helper thread, since it blocks until the path
    ends. Returns the thread, a dict its (outcome, index) result lands in, and
    the physics callback it registered."""
    result: dict[str, tuple[SettleOutcome, int]] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__(
            "result", handle.follow_path(points, max_vel_rad_s, tolerance_rad, point_timeout_s)
        ),
        daemon=True,  # a follow thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    callback_name = handle._settle_callback_name()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    return thread, result, sim.world._callbacks[callback_name]


def test_isaac_follow_path_advances_on_the_first_in_tolerance_step_and_reaches_the_end():
    handle, art, sim = _make_handle()
    _hold_commands_without_moving(art)
    art.positions = [0.0, 0.0]
    art.velocities = [0.5, 0.5]  # moving, so nothing here can read as a stall
    points = [[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]
    callback_name = handle._settle_callback_name()

    thread, result, callback = _start_follow(handle, sim, points, max_vel_rad_s=1.0)

    # the one task that registered the callback also capped the velocity and set the first target
    assert art._articulation_view.set_calls == [([1.0, 1.0], None)]
    assert handle._targets == points[0]

    callback(FAKE_PHYSICS_STEP_S)  # still far from the first point
    assert handle._targets == points[0]
    art.positions = [0.49, 0.52]  # inside the 0.05 tolerance, with no window of steps held
    callback(FAKE_PHYSICS_STEP_S)
    assert handle._targets == points[1]
    art.positions = [1.0, 1.0]
    callback(FAKE_PHYSICS_STEP_S)
    assert handle._targets == points[2]
    assert "result" not in result
    art.positions = [1.5, 1.5]
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["result"] == (SettleOutcome.REACHED, 3)
    assert [action.joint_positions.tolist() for action in art.applied_actions] == points
    # the cap was applied once for the path, not once per point
    assert art._articulation_view.set_calls == [([1.0, 1.0], None)]
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_follow_path_stalls_at_a_stuck_point_and_leaves_the_target_there():
    handle, art, sim = _make_handle()
    _hold_commands_without_moving(art)
    art.positions = [0.5, 0.5]  # at the first point and never gets any further
    art.velocities = [0.0, 0.0]
    points = [[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]
    callback_name = handle._settle_callback_name()

    thread, result, callback = _start_follow(handle, sim, points)
    callback(FAKE_PHYSICS_STEP_S)  # passes the first point
    assert handle._targets == points[1]
    for _ in range(STALL_STILL_STEPS - 1):
        callback(FAKE_PHYSICS_STEP_S)
        assert "result" not in result
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["result"] == (SettleOutcome.STALLED, 1)
    assert handle._targets == points[1]
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_follow_path_gives_every_point_its_own_sim_time_budget():
    handle, art, sim = _make_handle()
    _hold_commands_without_moving(art)
    art.positions = [0.0, 0.0]
    art.velocities = [0.5, 0.5]
    points = [[0.5, 0.5], [1.0, 1.0]]

    thread, result, callback = _start_follow(handle, sim, points, point_timeout_s=0.05)
    for _ in range(3):  # 0.03 s of the first point's 0.05 s
        callback(FAKE_PHYSICS_STEP_S)
    art.positions = [0.5, 0.5]
    callback(FAKE_PHYSICS_STEP_S)  # passes the first point
    assert handle._targets == points[1]
    # a budget carried over from the first point would be spent after two more steps
    art.positions = [0.5, 0.5]
    for _ in range(3):
        callback(FAKE_PHYSICS_STEP_S)
        assert "result" not in result
    for _ in range(4):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["result"] == (SettleOutcome.TIMED_OUT, 1)
    assert handle._targets == points[1]


def test_isaac_stop_during_a_follow_ends_it_in_the_task_that_holds_and_freezes_the_target():
    handle, art, sim = _make_handle()
    _hold_commands_without_moving(art)
    art.positions = [0.5, 0.5]
    art.velocities = [0.5, 0.5]
    points = [[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]

    thread, result, callback = _start_follow(handle, sim, points)
    callback(FAKE_PHYSICS_STEP_S)  # passes the first point, so the second is in flight
    assert handle._targets == points[1]

    follow_ended = handle._active_settle["event"]
    ended_after_each_task: list[bool] = []
    run = sim.run

    def observe(fn, timeout: float = 30.0, *, allow_during_initialization: bool = False):
        value = run(fn, timeout, allow_during_initialization=allow_during_initialization)
        if allow_during_initialization:
            ended_after_each_task.append(follow_ended.is_set())
        return value

    sim.run = observe
    handle.stop()
    thread.join(timeout=2.0)

    # the follow was over by the time the one task that wrote the hold returned
    assert ended_after_each_task == [True]
    assert result["result"] == (SettleOutcome.REACHED, 1)
    held = list(handle._targets)
    assert held == pytest.approx([0.5, 0.5])
    actions_after_hold = len(art.applied_actions)

    # the arm now sits inside the second point's tolerance, which would have
    # advanced the target to the third had the follow gone on
    art.positions = [1.0, 1.0]
    for _ in range(SETTLE_WINDOW_STEPS + 2):
        callback(FAKE_PHYSICS_STEP_S)
    assert handle._targets == held
    assert len(art.applied_actions) == actions_after_hold


def test_isaac_follow_path_over_no_points_reaches_without_touching_the_sim():
    handle, _art, sim = _make_handle()

    assert handle.follow_path([], 1.0, 0.05, 5.0) == (SettleOutcome.REACHED, 0)

    assert sim.run_calls == []
    assert handle._targets is None


def test_isaac_release_during_a_follow_removes_the_callback():
    handle, art, sim = _make_handle()
    _hold_commands_without_moving(art)
    callback_name = handle._settle_callback_name()

    thread, _result, _callback = _start_follow(handle, sim, [[0.5, 0.5], [1.0, 1.0]])
    handle.release()

    assert not sim.world.physics_callback_exists(callback_name)
    handle.stop()  # ends the follow so its thread does not wait out the wall-clock guard
    thread.join(timeout=2.0)


def test_mock_follow_path_returns_the_first_unreached_index():
    from isaac_module.sim_manager import MockArmHandle, SettleOutcome

    # a stalled mock closes half of every move and stops
    handle = MockArmHandle("follow-stalls-arm", {"asset": "ur5e", "mock_stall_fraction": 0.5})
    rest = [0.0] * (len(handle.get_joint_positions()) - 1)

    # the first point is reached to within 0.06, the second stalls 0.075 short
    outcome = handle.follow_path([[0.1, *rest], [0.2, *rest], [0.3, *rest]], None, 0.06, 5.0)

    assert outcome == (SettleOutcome.STALLED, 1)
    # it stopped halfway from wherever the first point left it, which is
    # within 0.06 of 0.1, to the second target
    assert handle.get_joint_positions()[0] == pytest.approx(0.125, abs=0.006)

    # a first point out of reach is index 0
    far = MockArmHandle("follow-stalls-at-once-arm", {"asset": "ur5e", "mock_stall_fraction": 0.5})
    assert far.follow_path([[1.0, *rest], [2.0, *rest]], None, 0.06, 5.0) == (
        SettleOutcome.STALLED,
        0,
    )


def test_mock_follow_path_ends_when_stopped_partway():
    from isaac_module.sim_manager import MockArmHandle, SettleOutcome

    handle = MockArmHandle("follow-stopped-arm", {"asset": "ur5e"})
    rest = [0.0] * (len(handle.get_joint_positions()) - 1)
    result: dict[str, tuple[SettleOutcome, int]] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__(
            "result", handle.follow_path([[0.5, *rest], [1.0, *rest]], 0.2, 0.01, 10.0)
        ),
        daemon=True,
    )
    thread.start()
    assert _wait_until(lambda: handle.get_joint_positions()[0] > 0.0)

    handle.stop()
    thread.join(timeout=2.0)

    assert result["result"] == (SettleOutcome.REACHED, 0)
    assert handle.is_moving() is False
    assert handle.get_joint_positions()[0] < 0.5


def test_isaac_arm_post_reset_reapplies_solver_gains_and_targets():
    handle, art, _sim = _make_handle()
    handle._solver_iterations = 64
    handle._gains = ([2.0, 2.0], [3.0, 3.0])
    handle._targets = [0.4, 0.5]

    handle.post_reset()

    assert art.solver_iteration_calls == [64]
    assert art._controller.set_gains_calls == [([2.0, 2.0], [3.0, 3.0])]
    assert art.applied_actions
    last = art.applied_actions[-1]
    assert list(last.joint_positions) == [0.4, 0.5]


def test_isaac_arm_post_reset_teleports_joints_to_targets_before_the_hold_action():
    """A world.reset() leaves the arm at its default pose; re-commanding the
    last targets from there is an unplanned full-speed sweep through the
    scene (the scatter_cell rescale-reset "freak out"). post_reset must
    restore the joint STATE to the targets, zero the velocities, and only
    then apply the hold action."""
    handle, art, _sim = _make_handle()
    handle._targets = [0.4, 0.5]
    # the reset's damage: default pose, residual velocity
    art.positions = [0.0, 0.0]
    art.velocities = [0.7, -0.7]

    handle.post_reset()

    assert art.state_write_log == [
        ("positions", [0.4, 0.5]),
        ("velocities", [0.0, 0.0]),
        ("action", [0.4, 0.5]),
    ]
    assert art.positions == [0.4, 0.5]
    assert art.velocities == [0.0, 0.0]


def test_isaac_arm_release_removes_settle_callback_and_registry_entry():
    handle, art, sim = _make_handle()
    callback_name = handle._settle_callback_name()
    sim.world.add_physics_callback(callback_name, lambda step_size: None)

    handle.release()

    assert not sim.world.physics_callback_exists(callback_name)
    assert sim.world.scene.removed == [(art.name, True)]
    assert art.name not in sim.world.scene.objects


def test_mock_is_moving_true_while_stalled_and_false_after_stop():
    """IsMoving unifies the two "arrived" notions. A stalled arm
    (velocity ~0 but commanded != measured) still reports moving; stop()
    re-targets the current position so both terms agree and it reports still."""
    from isaac_module.sim_manager import MockArmHandle, SettleOutcome

    handle = MockArmHandle("stalled-is-moving-arm", {"asset": "ur5e", "mock_stall_fraction": 0.5})
    handle.set_joint_targets([0.2] * 6)
    assert handle.wait_for_settle(5.0) is SettleOutcome.STALLED
    assert handle.is_moving() is True  # position term: 0.1 rad short of every target
    handle.stop()
    assert handle.is_moving() is False


def test_replace_articulation_retakes_gains_and_dofs_from_the_new_topology():
    """A gripper joining the arm changes the DOF count; the fresh wrapper's
    gains snapshot comes from the new topology, and the named joints resolve
    against the new dof list (ARM-2 / ARM-15)."""
    handle, _old_art, _sim = _make_handle(("j0", "j1"))
    handle._solver_iterations = 64
    handle._gains = ("stale", "stale")
    new_art = FakeArticulation("fake-arm", ["j0", "j1", "finger_joint"])
    new_art.get_articulation_controller().set_gains([2.0] * 3, [3.0] * 3)

    handle.replace_articulation(new_art)
    assert handle._gains is None

    handle.refresh_dofs()
    assert handle.all_dof_names() == ["j0", "j1", "finger_joint"]
    assert handle._gains == ([2.0] * 3, [3.0] * 3)  # retaken from the new articulation


def test_isaac_settle_reaches_despite_residual_velocity_jitter():
    """A 12-DOF PhysX articulation idles with ~1e-3 rad/s of jitter
    and never reads exactly still. Holding within tolerance for the window IS
    settled - a velocity-gated REACHED would time out forever."""
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.004, -0.003]  # above the old 1e-3 gate, below any real motion
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=5.0)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]
    for _ in range(SETTLE_WINDOW_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.REACHED


def test_isaac_is_moving_ignores_jitter_but_not_motion():
    handle, art, _sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.004, -0.003]
    assert handle.is_moving() is False
    art.velocities = [0.05, 0.0]  # 2.9 deg/s: real motion
    assert handle.is_moving() is True


def test_isaac_settle_stalls_on_no_progress_even_while_vibrating():
    """Pinned against a block the joints vibrate above VEL_EPS and
    never read still, so the old stall rule let the arm push for the whole
    deadline. A worst-error that stops improving is a stall too."""
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.4, 0.0]  # pinned 0.4 rad short of target
    art.velocities = [0.05, -0.05]  # vibrating, well above VEL_EPS
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=30.0)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]
    # step 1 records the first best error; the window counts steps AFTER it
    for _ in range(STALL_NO_PROGRESS_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result  # progress window not yet exhausted
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.STALLED


def test_isaac_settle_tolerates_a_joint_that_loses_ground_and_recovers():
    """The shape of every carry stall on 2026-09-24: the base arrives, wrist 3
    is back-driven forty degrees the wrong way by the swing, then recovers at
    speed. Measured against the best error ever seen, the recovery reads as no
    progress for two seconds and the arm is stopped mid-recovery. Measured over
    a window, a joint closing on its target is making progress."""
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 1.2]
    art.velocities = [0.0, -0.3]  # moving, never still
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=60.0)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # 60 steps closing 1.2 -> 0.6, 60 steps losing ground 0.6 -> 1.0, then
    # closing 0.02 rad a step: the old best of 0.6 is not beaten again until
    # 80 steps into the recovery, 140 steps after it was set
    profile = (
        [1.2 - 0.01 * k for k in range(1, 61)]
        + [0.6 + (0.4 / 60.0) * k for k in range(1, 61)]
        + [max(0.0, 1.0 - 0.02 * k) for k in range(1, 61)]
    )
    for error in profile:
        art.positions = [0.0, error]
        art.velocities = [0.0, -0.3 if error > 0.0 else 0.0]
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result, f"stopped the arm at error {error:.2f}"
    # at the target and still: the settle window closes it as REACHED
    art.velocities = [0.0, 0.0]
    for _ in range(SETTLE_WINDOW_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.REACHED


def test_isaac_settle_does_not_call_a_joint_closing_at_speed_stalled():
    """The 14:02 stall: the window ending at the bottom of a back-drive reads
    worse than its start while the joint is already recovering at 20 deg/s.
    Closing on the target above STALL_CONVERGING_SPEED_RAD_S is progress."""
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 1.0]
    art.velocities = [0.0, -0.35]  # 20 deg/s toward the target
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=60.0)),
        daemon=True,  # a settle thread that outlives its join must not hold the interpreter open
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # the window fills while the error gets WORSE (a back-drive), then the
    # joint closes at 20 deg/s: the window still reads no net progress for a
    # while, and the speed is what keeps the arm moving
    for k in range(STALL_NO_PROGRESS_STEPS + 1):
        art.positions = [0.0, 1.0 + 0.002 * k]
        callback(FAKE_PHYSICS_STEP_S)
    worst = art.positions[1]
    for k in range(1, 40):
        art.positions = [0.0, worst - 0.006 * k]
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result, f"stopped a joint closing at speed, step {k}"
    thread_alive = thread.is_alive()
    # finish the move so the callback is torn down
    art.positions = [0.0, 0.0]
    art.velocities = [0.0, 0.0]
    for _ in range(SETTLE_WINDOW_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)
    assert thread_alive
    assert result["outcome"] == SettleOutcome.REACHED


def test_isaac_set_drive_gains_writes_the_named_joint_and_snapshots_it():
    """The UR5e asset's wrist_3 drive is a quarter as stiff and almost
    undamped next to wrist_1's, and was back-driven forty degrees by a base
    swing with a box on the cups (2026-09-24). An override has to reach the
    controller AND the snapshot post_reset restores, or a world reset puts
    the asset's tuning back."""
    handle, art, _sim = _make_handle(("j0", "j1"))
    controller = art.get_articulation_controller()
    kps_before, _kds_before = controller.get_gains()

    table = handle.set_drive_gains({"j1": {"kp": 4321.0, "kd": 12.5}})

    kps, kds = controller.get_gains()
    assert list(kps)[1] == 4321.0
    assert list(kds)[1] == 12.5
    assert float(kps[0]) == float(kps_before[0])  # the other joint keeps its gain
    assert table["dof_names"] == ["j0", "j1"]
    assert table["kp"][1] == 4321.0
    assert table["kd"][1] == 12.5
    assert list(handle._gains[0])[1] == 4321.0
    assert list(handle._gains[1])[1] == 12.5
    with pytest.raises(ValueError, match="no joint"):
        handle.set_drive_gains({"nope": {"kp": 1.0}})


def test_isaac_drive_gain_overrides_from_config_are_applied_when_dofs_refresh():
    art = FakeArticulation("fake-arm", ["j0", "j1"])
    sim = FakeSim()
    sim.world.scene.objects[art.name] = art
    handle = IsaacArmHandle(sim, art, None, joint_names=None, drive_gains={"j1": {"kd": 77.0}})

    handle.refresh_dofs()

    _kps, kds = art.get_articulation_controller().get_gains()
    assert list(kds)[1] == 77.0
    assert list(handle._gains[1])[1] == 77.0


def test_isaac_set_drive_gains_sets_joint_armature_and_reports_it():
    """The fix for the carry: a PhysX articulation drive goes soft under a
    body jointed on through maximal-coordinate joints, and armature is what
    restores it. Measured 2026-09-24 on the UR5e with a 2 kg box on the
    cups: wrist 3 accelerated at about one radian per second squared at the
    asset's zero armature, and ran at its 180 deg/s limit at 0.1."""
    handle, art, _sim = _make_handle(("j0", "j1"))

    table = handle.set_drive_gains({"j1": {"armature": 0.1}})

    assert art._articulation_view.armatures == [0.0, 0.1]
    assert table["armature"] == [0.0, 0.1]
    assert handle.drive_gains()["armature"] == [0.0, 0.1]
    # a reset rebuilds the view from the asset's zeros; the override comes back
    art._articulation_view.armatures = [0.0, 0.0]
    handle.post_reset()
    assert art._articulation_view.armatures == [0.0, 0.1]
