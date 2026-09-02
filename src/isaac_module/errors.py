"""Module-level exception hierarchy (FINDINGS XC-3).

Every error the sim layer raises derives from SimError so models can map it to
one gRPC status. The concrete classes ALSO subclass the builtin the code used
to raise, so existing ``except`` clauses and tests keep working:

  SimNotBootedError  (RuntimeError)  -> FAILED_PRECONDITION
  PrimNotFoundError  (ValueError)    -> INVALID_ARGUMENT
  SimTimeoutError    (TimeoutError)  -> DEADLINE_EXCEEDED

UNIMPLEMENTED is ``viam.errors.MethodNotImplementedError``, raised directly by
the model method. The status-mapping helper lands with its first consumer
(phase 2, CAM-18); this module stays import-light on purpose.
"""


class SimError(Exception):
    """Base class for every error raised by the sim layer."""


class SimNotBootedError(SimError, RuntimeError):
    """The Isaac world is not running (no world component, or boot failed)."""


class PrimNotFoundError(SimError, ValueError):
    """A configured prim path does not exist in the stage."""


class CameraInitError(SimError, RuntimeError):
    """A camera's render product never became readable.

    ``Camera.initialize()`` looks up the render product's SDG pipeline node,
    which a freshly booted renderer only materializes after render ticks; this
    is raised once the bounded retry in ``SimManager._initialize_camera`` is
    exhausted.
    """


class SimTimeoutError(SimError, TimeoutError):
    """A call marshalled to the sim thread did not complete in time.

    Raised by ``SimManager.run`` after the underlying future has been
    cancelled, so the timed-out callable never executes later.
    """
