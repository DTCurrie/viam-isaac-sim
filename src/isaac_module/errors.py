"""Module-level exception hierarchy.

Every error the sim layer raises derives from SimError so models can map it to
one gRPC status. The concrete classes ALSO subclass ``viam.errors.ViamGRPCError``
with the status below, and the builtin the code used to raise, so existing
``except`` clauses and tests keep working:

  SimNotBootedError  (RuntimeError)  -> FAILED_PRECONDITION
  PrimNotFoundError  (ValueError)    -> INVALID_ARGUMENT
  SimTimeoutError    (TimeoutError)  -> DEADLINE_EXCEEDED
  SimInitializingError (RuntimeError) -> UNAVAILABLE

UNIMPLEMENTED is ``viam.errors.MethodNotImplementedError``, raised directly by
the model method. This module stays import-light on purpose.
"""

from grpclib import Status
from viam.errors import ViamGRPCError


class SimError(Exception):
    """Base class for every error raised by the sim layer."""


class SimNotBootedError(ViamGRPCError, SimError, RuntimeError):
    """The Isaac world is not running (no world component, or boot failed)."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.FAILED_PRECONDITION)
        Exception.__init__(self, message)


class PrimNotFoundError(ViamGRPCError, SimError, ValueError):
    """A configured prim path does not exist in the stage."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.INVALID_ARGUMENT)
        Exception.__init__(self, message)


class CameraInitError(SimError, RuntimeError):
    """A camera's render product never became readable.

    ``Camera.initialize()`` looks up the render product's SDG pipeline node,
    which a freshly booted renderer only materializes after render ticks; this
    is raised once the bounded retry in ``SimManager._initialize_camera`` is
    exhausted.
    """


SIM_INITIALIZING_MESSAGE = "Isaac Sim is initializing; retry shortly"


class SimInitializingError(ViamGRPCError, SimError, RuntimeError):
    """The world is booted but not yet ready for operational calls.

    Raised by ``SimManager.run`` (and ``SimManager.require_ready``) while a
    world configured with ``wait_for_finalizer`` is still draining the scene
    finalizer's first warm-up steps, so a cold shader compile reads as
    "initializing" rather than as a timed-out call. UNAVAILABLE is the status
    clients and the app treat as transient. The message is fixed so a client
    can match it.
    """

    def __init__(self, message: str = SIM_INITIALIZING_MESSAGE) -> None:
        ViamGRPCError.__init__(self, message, Status.UNAVAILABLE)
        Exception.__init__(self, message)


class SimTimeoutError(ViamGRPCError, SimError, TimeoutError):
    """A call marshalled to the sim thread did not complete in time.

    Raised by ``SimManager.run`` after the underlying future has been
    cancelled, so the timed-out callable never executes later.
    """

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.DEADLINE_EXCEEDED)
        Exception.__init__(self, message)
