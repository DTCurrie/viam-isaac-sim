"""XC-5: a post-reset hook registry fires at every world-reset chokepoint."""

import inspect
import threading

from isaac_module import sim_manager
from isaac_module.sim_manager import SimConfig, SimManager


class FakeWorld:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def reset(self) -> None:
        self._log.append("world.reset")


def _fresh_manager() -> SimManager:
    return SimManager()


def test_hook_fires_once_per_reset(sim):
    calls: list[None] = []
    sim.register_post_reset(lambda: calls.append(None))
    sim.reset()
    assert len(calls) == 1
    sim.reset()
    assert len(calls) == 2


def test_hooks_fire_in_registration_order(sim):
    order: list[str] = []
    sim.register_post_reset(lambda: order.append("first"))
    sim.register_post_reset(lambda: order.append("second"))
    before = len(order)
    sim.reset()
    assert order[before:] == ["first", "second"]


def test_raising_hook_does_not_stop_others_or_raise(sim):
    order: list[str] = []

    def bad_hook() -> None:
        raise RuntimeError("boom")

    sim.register_post_reset(bad_hook)
    sim.register_post_reset(lambda: order.append("after-bad"))
    before = len(order)
    sim.reset()  # must not raise
    assert order[before:] == ["after-bad"]


def test_fake_world_reset_runs_before_hook():
    mgr = _fresh_manager()
    log: list[str] = []
    mgr.mock = False
    mgr.world = FakeWorld(log)
    mgr._booted.set()
    mgr._sim_thread_id = threading.get_ident()

    mgr.register_post_reset(lambda: log.append("hook"))
    mgr.reset()

    assert log == ["world.reset", "hook"]
    # exactly one world.reset() call
    assert log.count("world.reset") == 1


def test_single_reset_chokepoint_and_call_sites():
    source = inspect.getsource(sim_manager)
    assert source.count("self.world.reset()") == 1

    for method_name in (
        "_boot",
        "_create_arm_isaac",
        "_create_base_isaac",
        "reset",
    ):
        method = getattr(SimManager, method_name)
        assert "_reset_world(" in inspect.getsource(method), method_name


def test_boot_stores_lighting_and_fires_pre_registered_hook():
    mgr = _fresh_manager()
    lighting_cfg = {"dome": {"intensity": 1000, "color": [1, 1, 1]}, "sphere_intensity": 30000}
    mgr.cfg = SimConfig(mock=True, lighting=lighting_cfg)

    fired: list[None] = []
    mgr.register_post_reset(lambda: fired.append(None))

    mgr._boot()

    assert mgr.lighting == lighting_cfg
    assert len(fired) == 1
