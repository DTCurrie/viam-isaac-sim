"""The machine's own frame system, read back as static colliders for the sim.

A component that declares ``frame.geometry`` has told the whole machine what
shape it is. The motion service already plans around it, and this module makes
the simulator agree, so one declaration serves the planner and the physics
instead of each keeping its own copy.

That replaces asking every component for its shape over DoCommand.
``GetGeometries`` is not served over the generic API at all, it answers
UNIMPLEMENTED on the wire, and ``get_visuals`` describes what a component
DRAWS rather than what it IS. The pallet draws a 3.18 m outfeed conveyor it
does not physically own, so a collider built from its visuals would make that
conveyor solid.

Props are not this module's business. A box is spawned by the world and lives
in the sim, and Viam resources reach it through Viam APIs.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator, Sequence
from logging import Logger
from typing import Any, Protocol

from viam.proto.common import Pose
from viam.proto.robot import FrameSystemConfig

from .spatial import Quat, Vec3, compose_pose, ov_to_quat


class FrameSystemSource(Protocol):
    """The parent-robot call this module needs. ``viam.module.module.Module``
    dials a ``RobotClient`` to its own machine over a local socket and keeps
    it on ``parent``, which is what supplies this."""

    async def get_frame_system_config(self) -> Sequence[FrameSystemConfig]: ...


MM_PER_M = 1000.0

# Reading the frame system is one local call. Anything past this is the
# machine not answering, and the world must still boot.
FRAME_SYSTEM_TIMEOUT_S = 20.0

# The running Module, published by main.py. A resource is handed its
# dependencies but never a way back to the machine, and the frame system is a
# machine-level fact, so its client is the only route to it from inside a model.
_MODULE: Any = None


def set_module(module: Any) -> None:
    """Publish the running ``viam.module.module.Module``."""
    global _MODULE
    _MODULE = module


def parent() -> FrameSystemSource | None:
    """The module's client to its own machine, or None before the SDK has
    dialled it.

    The SDK dials lazily while resolving the first dependency, so this reads
    the client at call time rather than capturing it at startup. A caller that
    gets None materialises no frame-system colliders rather than failing the
    boot.
    """
    return getattr(_MODULE, "parent", None)


def collider_props(parts: Sequence[FrameSystemConfig], *, logger: Logger) -> list[dict[str, Any]]:
    """A ``_spawn_prop`` dict per frame-system part that declares a box.

    Poses come back resolved against the part's parent, and every static part
    in this cell is parented to ``world``, so they are world poses. A part
    with no geometry contributes nothing, which is how a camera or a light
    stays render-only without needing to be listed anywhere.
    """
    props: list[dict[str, Any]] = []
    for part in parts:
        prop = _collider_prop(part)
        if prop is None:
            continue
        props.append(prop)
    logger.info(
        "frame system: %d of %d parts declare a collider (%s)",
        len(props),
        len(parts),
        ", ".join(prop["name"] for prop in props) or "none",
    )
    return props


def _position_m(pose: Pose) -> Vec3:
    return (pose.x / MM_PER_M, pose.y / MM_PER_M, pose.z / MM_PER_M)


def _orientation(pose: Pose) -> Quat:
    """The ``(w, x, y, z)`` quaternion of a proto pose's orientation vector,
    whose ``theta`` is in degrees. An all-zero vector is the proto's default
    for a pose that was never given an orientation, so it means unrotated
    rather than pointing nowhere."""
    if not (pose.o_x or pose.o_y or pose.o_z):
        return (1.0, 0.0, 0.0, 0.0)
    return ov_to_quat(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta))


def _collider_prop(part: FrameSystemConfig) -> dict[str, Any] | None:
    """The collider is the geometry's box composed onto the part's frame:
    both poses carry an orientation, and both count.

    Adding the two translations and stopping there is what this did first,
    and the two fences whose frames turn 90 degrees about z showed the cost
    on 2026-09-22: their 1200 mm colliders stood across the cell along x, in
    front of the back fences, instead of running along y where the panels
    are drawn and where the planner already had them.
    """
    frame = part.frame
    geometry = frame.physical_object
    if not geometry.HasField("box"):
        return None
    dims = geometry.box.dims_mm
    if not (dims.x and dims.y and dims.z):
        return None
    origin = frame.pose_in_observer_frame.pose
    centre = geometry.center
    position, orientation = compose_pose(
        _position_m(origin), _orientation(origin), _position_m(centre), _orientation(centre)
    )
    return {
        "name": f"frame-{frame.reference_frame}",
        "type": "cube",
        "position": position,
        "orientation_wxyz": orientation,
        "size": 1.0,
        "scale": (dims.x / MM_PER_M, dims.y / MM_PER_M, dims.z / MM_PER_M),
        "fixed": True,
    }


def world_parts(parts: Sequence[FrameSystemConfig]) -> Iterator[FrameSystemConfig]:
    """The parts posed directly against ``world``. A part parented to another
    component carries a pose in THAT frame, which this module does not
    compose, so treating it as a world pose would put a collider in the wrong
    place."""
    for part in parts:
        if part.frame.pose_in_observer_frame.reference_frame == "world":
            yield part


def frame_system_colliders(
    *, logger: Logger, loop: asyncio.AbstractEventLoop | None = None
) -> list[dict[str, Any]]:
    """Every world-parented frame-system part's declared box, as a prop dict.

    Returns an empty list when the module has no client to its own machine
    yet, or when the call fails: a cell with no declared geometry is a cell
    whose planner has none either, which is a config fact rather than a reason
    to stop the world booting.
    """
    source = parent()
    if source is None:
        logger.warning("no client to this machine yet; no frame-system colliders")
        return []
    try:
        if loop is not None:
            future = asyncio.run_coroutine_threadsafe(source.get_frame_system_config(), loop)
            parts = future.result(FRAME_SYSTEM_TIMEOUT_S)
        else:
            parts = asyncio.run(source.get_frame_system_config())
    # the frame system is advisory for the sim, and the planner reads it
    # independently, so a failure here costs colliders and not the boot
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read the frame system (%s: %s)", type(exc).__name__, exc)
        return []
    return collider_props(list(world_parts(parts)), logger=logger)
