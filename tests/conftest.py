import threading

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimConfig, SimManager

SIM_THREAD_JOIN_TIMEOUT_S = 5


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


@pytest.fixture(scope="session")
def sim():
    manager = SimManager.get()
    sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
    sim_thread.start()
    manager.ensure_booted(SimConfig(mock=True))
    yield manager
    manager.request_stop()
    sim_thread.join(timeout=SIM_THREAD_JOIN_TIMEOUT_S)


@pytest.fixture(scope="session")
def world(sim):
    return IsaacWorld.new(_config("sim-world", {"mock": True}), {})
