"""Regression tests for SimManager.run's timeout handling (FINDINGS XC-3, R-8).

A call marshalled to the sim thread that times out must raise SimTimeoutError
and cancel its underlying future so the callable never executes later, even
if a drain loop eventually picks it up.
"""

import threading
import time

import pytest

from isaac_module.errors import SimNotBootedError, SimTimeoutError
from isaac_module.sim_manager import SimManager

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
