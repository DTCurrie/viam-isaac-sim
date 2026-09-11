import logging
import threading
import types
from concurrent.futures import Future

import pytest
from grpclib import Status

from isaac_module.errors import SimInitializingError
from isaac_module.sim_manager import POST_FINALIZER_WARMUP_STEPS, SimConfig, SimManager


def _non_mock_manager() -> SimManager:
    """A directly constructed, non-mock manager wired the way main_loop's
    step loop expects: an app that reports running, ready for
    manager._boot to swap in a fake world."""
    manager = SimManager()
    manager._sim_app = types.SimpleNamespace(is_running=lambda: True, close=lambda: None)
    return manager


def test_queued_task_runs_before_finalize_scene_releases_stepping():
    setup_completed = threading.Event()
    rendered = threading.Event()

    class World:
        def step(self, *, render):
            assert render is True
            rendered.set()

    manager = _non_mock_manager()
    manager.cfg = SimConfig(wait_for_finalizer=True)
    manager._scene_finalized.clear()
    manager._ready.clear()
    manager._boot = lambda: setattr(manager, "world", World())
    manager._boot_requested.set()
    manager._tasks.put((setup_completed.set, Future()))

    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    try:
        assert setup_completed.wait(timeout=1)
        assert not rendered.is_set()

        manager.finalize_scene()
        assert rendered.wait(timeout=1)
    finally:
        manager.request_stop()
        thread.join(timeout=1)
        assert not thread.is_alive()


def test_ready_flips_only_after_the_warmup_step_count():
    permits = threading.Semaphore(0)
    reached_last_step = threading.Event()

    class World:
        def __init__(self):
            self.steps = 0

        def step(self, *, render):
            assert render is True
            assert permits.acquire(timeout=1)
            self.steps += 1
            if self.steps == POST_FINALIZER_WARMUP_STEPS - 1:
                reached_last_step.set()

    manager = _non_mock_manager()
    world = World()
    manager.cfg = SimConfig(wait_for_finalizer=True)
    manager._scene_finalized.clear()
    manager._ready.clear()
    manager._boot = lambda: setattr(manager, "world", world)
    manager._boot_requested.set()

    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    try:
        manager.finalize_scene()
        for _ in range(POST_FINALIZER_WARMUP_STEPS - 1):
            permits.release()
        assert reached_last_step.wait(timeout=1)
        assert not manager._ready.is_set()

        permits.release()
        assert manager._ready.wait(timeout=1)
    finally:
        manager.request_stop()
        permits.release()
        thread.join(timeout=1)
        assert not thread.is_alive()


def test_run_rejects_calls_until_ready_and_allows_the_opt_out():
    manager = SimManager()
    manager._sim_thread_id = threading.get_ident() + 1  # the caller is not the sim thread
    manager._ready.clear()

    with pytest.raises(SimInitializingError) as excinfo:
        manager.run(lambda: "operation")

    assert excinfo.value.grpc_code is Status.UNAVAILABLE
    assert str(excinfo.value) == "Isaac Sim is initializing; retry shortly"
    assert manager._tasks.empty()

    # the opt-out queues the task, so drain it the way the sim thread would
    result: list[str] = []
    caller = threading.Thread(
        target=lambda: result.append(manager.run(lambda: "setup", allow_during_initialization=True))
    )
    caller.start()
    while caller.is_alive():
        manager._drain_tasks()
        caller.join(timeout=0.01)
    assert result == ["setup"]


def test_run_on_the_sim_thread_ignores_the_gate():
    # Post-reset hooks and nested factory calls run on the sim thread while the
    # scene is still initializing. They never wait behind a slow step, so the
    # gate must not refuse them (observed 2026-09-14: every post-reset hook
    # failed on a gated boot).
    manager = SimManager()
    manager._sim_thread_id = threading.get_ident()
    manager._ready.clear()

    assert manager.run(lambda: "hook") == "hook"


def test_status_reports_ready_flag_and_never_trips_the_gate_when_closed():
    class _MustNotBeTouchedWorld:
        def is_playing(self):
            raise AssertionError("status() must not call run() while the gate is closed")

    manager = SimManager()
    manager._sim_thread_id = threading.get_ident()
    manager.mock = False
    manager._booted.set()
    manager._ready.clear()
    manager.world = _MustNotBeTouchedWorld()

    status = manager.status()
    assert status["ready"] is False
    assert "playing" not in status

    class _WorkingWorld:
        def is_playing(self):
            return True

        current_time = 0.0

    manager._ready.set()
    manager.world = _WorkingWorld()
    assert manager.status()["ready"] is True


def test_slow_step_logs_a_warning_with_the_elapsed_time(monkeypatch, caplog):
    monkeypatch.setattr("isaac_module.sim_manager.SLOW_STEP_WARN_S", 0.0)

    class World:
        def step(self, *, render):
            assert render is True

    manager = SimManager()
    manager.mock = False
    manager.world = World()
    # is_running() reports True for the first check (letting one step run)
    # and False afterward, so the loop steps exactly once then exits.
    stepped = {"count": 0}

    def _is_running() -> bool:
        return stepped["count"] == 0

    real_step = World.step

    def _counting_step(self, *, render):
        stepped["count"] += 1
        real_step(self, render=render)

    World.step = _counting_step
    manager._sim_app = types.SimpleNamespace(is_running=_is_running, close=lambda: None)

    with caplog.at_level(logging.WARNING, logger="viam-isaac-sim"):
        manager._step_loop()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert "world.step" in warnings[0].message
