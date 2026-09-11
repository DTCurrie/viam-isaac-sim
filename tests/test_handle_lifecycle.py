import asyncio
import signal
import threading

from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

import main as main_module
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld
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
    arm = IsaacArm.new(_config("close-arm-1", {"world": "isaac-world", "asset": "ur20"}), {})
    first_handle = arm._handle
    assert "close-arm-1" in SimManager.get()._handles

    asyncio.run(arm.close())
    assert "close-arm-1" not in SimManager.get()._handles
    assert arm._handle is None

    arm2 = IsaacArm.new(_config("close-arm-1", {"world": "isaac-world", "asset": "ur20"}), {})
    assert arm2._handle is not first_handle


def test_arm_close_is_idempotent(world):
    arm = IsaacArm.new(_config("close-arm-2", {"world": "isaac-world", "asset": "ur20"}), {})
    asyncio.run(arm.close())
    asyncio.run(arm.close())  # must not raise
    assert "close-arm-2" not in SimManager.get()._handles


def test_arm_reconfigure_allows_changed_spawn_attribute_while_still_attached(world):
    # D14: Module.reconfigure_resource always removes the resource (close())
    # before building a fresh one, so a still-attached component's own
    # reconfigure() only ever sees a name already in SimManager._handles when
    # nothing closed it in between - the cached handle is returned as-is,
    # spawn attribute or not, instead of raising.
    arm = IsaacArm.new(
        _config("reconf-arm-1", {"world": "isaac-world", "asset": "ur20", "position": [0, 0, 0]}),
        {},
    )
    original_handle = arm._handle
    changed = _config(
        "reconf-arm-1", {"world": "isaac-world", "asset": "ur20", "position": [1, 0, 0]}
    )
    arm.reconfigure(changed, {})  # must not raise
    assert arm._handle is original_handle


def test_arm_reconfigure_allows_changed_runtime_attribute_and_keeps_handle(world):
    arm = IsaacArm.new(
        _config(
            "reconf-arm-2",
            {"world": "isaac-world", "asset": "ur20", "move_timeout_sec": 10},
        ),
        {},
    )
    original_handle = arm._handle
    changed = _config(
        "reconf-arm-2",
        {"world": "isaac-world", "asset": "ur20", "move_timeout_sec": 99},
    )
    arm.reconfigure(changed, {})  # must not raise
    assert arm._handle is original_handle


def test_arm_reconfigure_allows_changed_frame_translation_while_still_attached(world):
    arm = IsaacArm.new(
        _config_with_frame_translation(
            "reconf-arm-3", {"world": "isaac-world", "asset": "ur20"}, 0.0
        ),
        {},
    )
    original_handle = arm._handle
    changed = _config_with_frame_translation(
        "reconf-arm-3", {"world": "isaac-world", "asset": "ur20"}, 60.0
    )
    arm.reconfigure(changed, {})  # must not raise
    assert arm._handle is original_handle


def test_arm_reattach_after_close_rebuilds_handle_without_duplicate_scene_add_or_reset():
    """The D14 default: release_handle (close()) forgets the cached handle
    but deliberately leaves the prim in the stage, so the next create_arm
    for that name must rebuild a handle without re-adding the prim to the
    scene or resetting the whole world - both would either duplicate scene
    state or undo every other component's state Isaac-side."""
    manager = SimManager()
    manager.mock = False
    manager.world = _FakeArmWorld()
    manager._isaac = _FakeArmIsaacNamespace()
    manager._booted.set()
    manager._sim_thread_id = threading.get_ident()

    attrs = {"world": "isaac-world", "position": [0, 0, 0]}
    first = manager.create_arm("reattach-arm", attrs)
    assert manager.world.scene_add_count == 1
    assert manager.world.reset_count == 1

    manager.release_handle("reattach-arm")

    changed_attrs = {"world": "isaac-world", "position": [1, 0, 0]}
    second = manager.create_arm("reattach-arm", changed_attrs)

    assert second is not first  # rebuilds the handle once
    assert manager.world.scene_add_count == 1  # no duplicate scene.add
    assert manager.world.reset_count == 1  # no full world reset on re-attach


class _FakeScene:
    def __init__(self, world: "_FakeArmWorld") -> None:
        self._world = world

    def add(self, _obj) -> None:
        self._world.scene_add_count += 1

    def get_object(self, _name: str) -> None:
        return None  # release()'s remove_object guard: nothing registered


class _FakeArmWorld:
    def __init__(self) -> None:
        self.scene_add_count = 0
        self.reset_count = 0
        self.scene = _FakeScene(self)

    def reset(self) -> None:
        self.reset_count += 1

    def physics_callback_exists(self, _name: str) -> bool:
        return False  # release()'s remove_physics_callback guard


class _FakeControllerHandle:
    def get_gains(self) -> None:
        return None


class _FakeArt:
    def __init__(self, prim_path: str, name: str) -> None:
        self.prim_path = prim_path
        self.name = name
        self.dof_names: list[str] = []
        self.initialize_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1

    def set_solver_position_iteration_count(self, _count: int) -> None:
        pass

    def get_articulation_controller(self) -> _FakeControllerHandle:
        return _FakeControllerHandle()


class _FakeArmIsaacNamespace:
    def SingleArticulation(self, prim_path: str, name: str) -> _FakeArt:
        return _FakeArt(prim_path, name)


# -- camera ---------------------------------------------------------------


def test_camera_close_forgets_handle_and_drops_post_reset_hook(world):
    sim = SimManager.get()
    camera = IsaacCamera.new(_config("close-cam-1", {"world": "isaac-world"}), {})
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
            {"world": "isaac-world", "asset": "jetbot"},
        ),
        {},
    )
    assert "close-base-1" in SimManager.get()._handles

    asyncio.run(base.close())
    assert "close-base-1" not in SimManager.get()._handles
    assert base._handle is None


# -- gripper (implemented by a sibling; close() is already in the seam) ---


def test_gripper_close_forgets_handle(world):
    arm = IsaacArm.new(_config("gripper-host-arm", {"world": "isaac-world", "asset": "ur20"}), {})
    gripper = IsaacGripper.new(
        _config("close-gripper-1", {"world": "isaac-world", "arm": arm.name}), {}
    )

    assert "close-gripper-1" in SimManager.get()._handles
    asyncio.run(gripper.close())
    assert "close-gripper-1" not in SimManager.get()._handles


# -- world (D3: close() drops the world's own hooks, does not stop Kit) --


def test_world_close_drops_its_own_post_reset_hook_and_leaves_the_sim_booted(world):
    sim = SimManager.get()
    w = IsaacWorld.new(_config("close-world-1", {"mock": True}), {})

    call_count = 0

    def _count_post_reset():
        nonlocal call_count
        call_count += 1

    sim.register_post_reset(_count_post_reset, owner=w.name)

    sim.reset()
    assert call_count == 1

    asyncio.run(w.close())

    sim.reset()
    assert call_count == 1  # the hook was dropped, not fired again
    assert sim._booted.is_set()  # close() does not stop Kit ...
    assert not sim._stop.is_set()  # ... nor does it request a stop


# -- main.py module thread shutdown (VM-22, D3) --------------------------


def test_shutdown_schedules_module_stop_joins_then_stops_the_sim(monkeypatch):
    # VM-22: request_stop() alone used to leave Module.stop() - and any
    # RemoveResource in flight - never run before the process exited.
    calls: list = []

    def _fake_run_coroutine_threadsafe(coro, loop):
        calls.append("schedule_module_stop")
        coro.close()  # never actually run here; avoids an "unawaited" warning
        return None

    monkeypatch.setattr(
        main_module.asyncio, "run_coroutine_threadsafe", _fake_run_coroutine_threadsafe
    )

    class _FakeModule:
        async def stop(self) -> None:
            pass

    class _FakeModuleThread:
        def join(self, timeout=None) -> None:
            calls.append(("join", timeout))

    class _RecordingSim:
        def request_stop(self) -> None:
            calls.append("request_stop")

    state = main_module._ModuleThreadState()
    state.loop = object()  # never touched: run_coroutine_threadsafe is faked above
    state.module = _FakeModule()

    main_module._shutdown(_RecordingSim(), _FakeModuleThread(), state, signal.SIGTERM, None)

    assert calls == [
        "schedule_module_stop",
        ("join", main_module.MODULE_STOP_JOIN_TIMEOUT_S),
        "request_stop",
    ]


def test_shutdown_still_stops_the_sim_when_the_module_thread_never_set_up_state():
    # A signal arriving before _run_module has built its loop/Module (a
    # narrow startup race) must not crash the handler or skip request_stop.
    calls: list = []

    class _FakeModuleThread:
        def join(self, timeout=None) -> None:
            calls.append(("join", timeout))

    class _RecordingSim:
        def request_stop(self) -> None:
            calls.append("request_stop")

    state = main_module._ModuleThreadState()  # loop=None, module=None

    main_module._shutdown(_RecordingSim(), _FakeModuleThread(), state, signal.SIGTERM, None)

    assert calls == [("join", main_module.MODULE_STOP_JOIN_TIMEOUT_S), "request_stop"]
