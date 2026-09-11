import signal
import sys
import threading
import time
import types

import pytest

from isaac_module.errors import SimNotBootedError, SimTimeoutError
from isaac_module.sim_manager import SimConfig, SimManager

SHORT_WAIT_S = 0.05  # short enough to keep the suite fast, long enough to be a real wait


def _not_current_thread_id() -> int:
    """A thread id guaranteed not to equal the calling thread's, so
    SimManager.run takes the queue path instead of the same-thread fast path."""
    return threading.get_ident() + 1


def test_run_timeout_raises_sim_timeout_error_and_prevents_late_execution():
    manager = SimManager()
    manager._sim_thread_id = _not_current_thread_id()

    invoked = {"called": False}

    def fn():
        invoked["called"] = True
        return "should never run"

    with pytest.raises(SimTimeoutError) as exc_info:
        manager.run(fn, timeout=SHORT_WAIT_S)

    # SimTimeoutError subclasses TimeoutError so existing except clauses work.
    assert isinstance(exc_info.value, TimeoutError)

    # The task is still queued; simulate the sim thread finally getting to it.
    manager._drain_tasks()

    assert invoked["called"] is False


def test_run_returns_result_once_drained():
    manager = SimManager()
    manager._sim_thread_id = _not_current_thread_id()

    def drain_after_delay():
        time.sleep(SHORT_WAIT_S)
        manager._drain_tasks()

    drainer = threading.Thread(target=drain_after_delay)
    drainer.start()
    try:
        result = manager.run(lambda: 41 + 1, timeout=5.0)
    finally:
        drainer.join()

    assert result == 42


def test_run_propagates_exception_from_callable():
    manager = SimManager()
    manager._sim_thread_id = _not_current_thread_id()

    def raise_value_error():
        raise ValueError("boom")

    def drain_after_delay():
        time.sleep(SHORT_WAIT_S)
        manager._drain_tasks()

    drainer = threading.Thread(target=drain_after_delay)
    drainer.start()
    try:
        with pytest.raises(ValueError, match="boom"):
            manager.run(raise_value_error, timeout=5.0)
    finally:
        drainer.join()


def test_require_booted_raises_sim_not_booted_error():
    manager = SimManager()

    with pytest.raises(SimNotBootedError) as exc_info:
        manager._require_booted()

    assert isinstance(exc_info.value, RuntimeError)


def test_main_loop_sets_stopped_flag_on_every_exit_path():
    # PY-28: _stopped must be set even when main_loop returns before ever
    # booting (a stop requested before any world component boots the sim).
    manager = SimManager()
    manager._stop.set()

    manager.main_loop()

    assert manager._stopped is True


def test_require_booted_raises_immediately_once_stopped():
    # PY-28: after main_loop() exits, _booted stays set forever, so without
    # this check a call reaching _require_booted would pass straight
    # through and only fail once run() drains it (or times out).
    manager = SimManager()
    manager._booted.set()
    manager._stopped = True

    with pytest.raises(SimNotBootedError, match="stopped"):
        manager._require_booted()


def test_run_raises_immediately_once_stopped_instead_of_waiting_the_timeout():
    # PY-28: previously a call arriving after shutdown waited run()'s full
    # timeout (default 30s) before raising SimTimeoutError; this proves it
    # now fails on the spot with the right error instead.
    manager = SimManager()
    manager._sim_thread_id = _not_current_thread_id()
    manager._stopped = True

    started = time.monotonic()
    with pytest.raises(SimNotBootedError, match="stopped"):
        manager.run(lambda: "should never run", timeout=30.0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0  # nowhere near the 30s default; proves no wait happened


def test_boot_reinstalls_sigint_handler_after_simulation_app_replaces_it(monkeypatch):
    # IS-13: SimulationApp.__init__ unconditionally replaces SIGINT with a
    # handler that calls sys.exit(0) directly, bypassing main.py's whole
    # shutdown path, until _boot() puts the caller's handler back right
    # after construction.
    def _shutdown_handler(signum, frame):
        pass

    def _kit_sigint_handler(signum, frame):
        pass

    class _FakeSimulationApp:
        def __init__(self, config: dict) -> None:
            signal.signal(signal.SIGINT, _kit_sigint_handler)

    class _StopAfterConstruction(Exception):
        """Raised once _boot() moves on to isaac imports this test doesn't
        need to fake, so the test only has to assert about the handler
        installed between SimulationApp() and that point."""

    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = _FakeSimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)

    def _raise_stop(*args, **kwargs):
        raise _StopAfterConstruction

    monkeypatch.setattr("isaac_module.sim_manager.import_isaac", _raise_stop)

    manager = SimManager()
    manager.cfg = SimConfig(mock=False, livestream=False)
    original_handler = signal.signal(signal.SIGINT, _shutdown_handler)
    try:
        with pytest.raises(_StopAfterConstruction):
            manager._boot()

        assert signal.getsignal(signal.SIGINT) is _shutdown_handler
    finally:
        signal.signal(signal.SIGINT, original_handler)


def test_open_stage_and_wait_raises_when_open_stage_returns_false():
    # IS-8: open_stage()'s bool return used to be discarded, so a bad path
    # silently left an empty stage (no ground plane) instead of failing boot.
    class _FakeIsaacNamespace:
        @staticmethod
        def open_stage(path):
            return False

    manager = SimManager()
    manager._isaac = _FakeIsaacNamespace()

    with pytest.raises(RuntimeError, match=r"bad/stage\.usd"):
        manager._open_stage_and_wait("bad/stage.usd")


def test_open_stage_and_wait_drains_loading_then_resets_render_settings(monkeypatch):
    # IS-8: the documented sequence drains is_stage_loading() and calls
    # reset_render_settings() after a successful open_stage(), neither of
    # which the module used to do.
    remaining = {"count": 2}

    def is_stage_loading():
        if remaining["count"] > 0:
            remaining["count"] -= 1
            return True
        return False

    fake_stage_module = types.ModuleType("isaacsim.core.utils.stage")
    fake_stage_module.is_stage_loading = is_stage_loading
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage", fake_stage_module)

    class _FakeIsaacNamespace:
        @staticmethod
        def open_stage(path):
            return True

    class _FakeSimApp:
        def __init__(self) -> None:
            self.update_calls = 0
            self.reset_called = False

        def update(self) -> None:
            self.update_calls += 1

        def reset_render_settings(self) -> None:
            self.reset_called = True

    manager = SimManager()
    manager._isaac = _FakeIsaacNamespace()
    fake_app = _FakeSimApp()
    manager._sim_app = fake_app

    manager._open_stage_and_wait("good/stage.usd")

    assert fake_app.update_calls == 2
    assert fake_app.reset_called is True


def test_step_loop_records_boot_error_when_sim_app_stops_running():
    # IS-11: without checking is_running(), the step loop used to spin on a
    # dead app instead of recording a failure the world's status verb reports.
    manager = SimManager()
    manager.mock = False

    class _FakeSimApp:
        def is_running(self) -> bool:
            return False

    manager._sim_app = _FakeSimApp()

    def _must_not_step(render: bool) -> None:
        raise AssertionError("the step loop must not step a dead app")

    manager.world = type("_FakeWorld", (), {"step": staticmethod(_must_not_step)})()

    manager._step_loop()

    assert manager._boot_error is not None
    assert "running" in manager.status()["error"]


def test_resolve_usd_wraps_runtime_error_from_get_assets_root_path():
    # IS-17: get_assets_root_path() raises RuntimeError in 5.0 rather than
    # returning None, so the module's friendlier message never reached a user.
    class _RaisingIsaacNamespace:
        client = None

        @staticmethod
        def get_assets_root_path():
            raise RuntimeError("Could not find assets root folder")

    manager = SimManager()
    manager.mock = False
    manager._isaac = _RaisingIsaacNamespace()

    with pytest.raises(RuntimeError, match="could not reach the Isaac Sim assets server"):
        manager._resolve_usd({"asset": "jetbot"})
