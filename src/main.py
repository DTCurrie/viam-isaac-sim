"""Module entrypoint.

Isaac Sim (Omniverse Kit) must own the thread it runs on, so the layout is:
  - main thread: SimManager.main_loop() - boots and steps the simulator
  - side thread: the Viam module gRPC server

When viam-server shuts the module down, the signal handler schedules
Module.stop() on the module thread's loop, gives the thread a short window
to react, then asks the sim loop to stop, which closes the SimulationApp.
"""

import asyncio
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from viam.logging import getLogger
from viam.module.module import Module
from viam.resource.registry import Registry

import isaac_module.models  # noqa: F401 - registers all models
from isaac_module import frame_system, sdk_patches
from isaac_module.sim_manager import SimManager

sdk_patches.apply()

LOGGER = getLogger("viam-isaac-sim.main")

# How long main()'s signal handler waits for the module thread to react to
# a scheduled Module.stop() before giving up and stopping the sim anyway
# (VM-22). request_stop() unconditionally follows, so this only bounds how
# long shutdown can take, never whether it happens.
MODULE_STOP_JOIN_TIMEOUT_S = 3.0

# The module thread's asyncio.to_thread default executor. Each component
# model (arm, gripper, base, camera) holds at most one blocking sim-thread
# call in flight at a time, so a small fixed pool bounds how many abandoned
# to_thread workers (asyncio.to_thread isn't cancellable) can pile up under
# concurrent RPC traffic, instead of leaving that pool size to the interpreter
# default.
MODULE_EXECUTOR_MAX_WORKERS = 8


class _ModuleThreadState:
    """Populated by _run_module once its event loop and Module instance
    exist, so the signal handler on the main thread can reach them to
    schedule a clean stop."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.module: Module | None = None


def _run_module(sim: SimManager, state: _ModuleThreadState) -> None:
    loop = asyncio.new_event_loop()
    # grpclib wants to install signal handlers, which is only possible on the
    # main thread - and the main thread belongs to isaac sim. Shutdown is
    # handled by the signal handlers installed in main() instead.
    loop.add_signal_handler = lambda *args, **kwargs: None  # type: ignore[method-assign]  # deliberate: signals are handled on the main thread
    asyncio.set_event_loop(loop)
    loop.set_default_executor(ThreadPoolExecutor(max_workers=MODULE_EXECUTOR_MAX_WORKERS))
    state.loop = loop
    try:
        # Module.run_from_registry()'s own body, inlined: it keeps the
        # Module instance it builds local, and _shutdown needs a reference
        # to schedule Module.stop() on this thread's loop.
        module = Module.from_args()
        state.module = module
        for key in Registry.REGISTERED_RESOURCE_CREATORS().keys():
            module.add_model_from_registry(*key.split("/"))  # type: ignore[arg-type]  # SDK: key is "api/model", matching run_from_registry's own (unchecked) body
        # scene-finalizer reads the frame system through this, which is a
        # machine-level fact no resource is otherwise handed. The SDK dials
        # the client lazily while resolving the first dependency, so the
        # module is published here and its parent read at call time.
        frame_system.set_module(module)
        loop.run_until_complete(module.start())
    except Exception:
        LOGGER.exception("module server exited with error")
    finally:
        sim.request_stop()


def _shutdown(
    sim: SimManager,
    module_thread: threading.Thread,
    state: _ModuleThreadState,
    signum: int,
    frame: object,
) -> None:
    """Signal handler body, called for both SIGTERM and SIGINT. Schedules
    Module.stop() on the module thread's loop, joins that thread with a
    short timeout, then stops the sim - so an in-flight RemoveResource and
    each resource's close() get a chance to run before the process exits
    (VM-22)."""
    if state.loop is not None and state.module is not None:
        asyncio.run_coroutine_threadsafe(state.module.stop(), state.loop)
    module_thread.join(timeout=MODULE_STOP_JOIN_TIMEOUT_S)
    sim.request_stop()


def main() -> None:
    sim = SimManager.get()
    state = _ModuleThreadState()
    t = threading.Thread(target=_run_module, args=(sim, state), name="viam-module", daemon=True)

    def _handle_signal(signum, frame):
        _shutdown(sim, t, state, signum, frame)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    t.start()
    sim.main_loop()
    sys.exit(0)


if __name__ == "__main__":
    main()
