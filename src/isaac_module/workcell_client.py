"""Fetches each configured workcell component's geometry and visuals over
the Viam API, and turns the replies into ``ComponentScenery`` for
``SimManager.materialise_components``.

Every fact this module needs about a component (its model, the frame its
primitives are anchored to, its attributes, its render primitives, and its
collider box where one exists) is fetched live from the component itself. None of it is
kept in a second config, because a pose or a model kept in two places is a
pose or a model that can drift out of sync, silently, the moment one of the
two is edited and the other is not.

The verbs and their reply shapes below come from
``viam-labs/workcell-components`` at ``0.7.0``, read two ways: transcribed
from ``apps/workcell/index.html``, the vendored reference app that already
drives every one of these calls, and reconstructed by convention where the
app itself never calls the verb. Each function says which below.

``new``/``reconfigure`` on a Viam resource are synchronous in the installed
SDK (``module.py``'s ``add_resource`` never awaits the creator it calls), so
``materialise_workcell`` runs the async gathering on a dedicated thread with
its own event loop and blocks until it finishes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from collections.abc import Mapping
from logging import Logger
from typing import Any, cast

from viam.components.component_base import ComponentBase
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase

from .sim_manager import ComponentScenery
from .spatial import Quat, Vec3, ov_to_quat
from .workcell_scenery import SHAPED_MODELS

MM_PER_M = 1000.0

# A gather that never returns parks the world at its scene gate forever, which
# looks like a hung simulator rather than a failed probe. Bound it, and let the
# caller finalize without the scenery instead.
GATHER_TIMEOUT_S = 90.0

# One unresponsive dependency must not consume the whole gather's budget. A
# workcell component answers a probe in tens of milliseconds, so anything past
# this is a component that will not answer at all.
PROBE_TIMEOUT_S = 10.0


# Transcribed from apps/workcell/index.html: every DoCommand call there is
# `client.doCommand(toStruct({ <verb>: true }))`, e.g. `{ get_visuals: true }`
# (line 586) and `{ get_attributes: true }` (line 584), not the
# `{"command": <verb>}` shape this repo's own DoCommand verbs use.
def _verb(name: str) -> dict[str, Any]:
    return {name: True}


def bare_model_name(model: str) -> str:
    """The bare model name ``workcell_scenery.SHAPED_MODELS`` keys against,
    from a component's full model triple.

    ``"viam:workcell-components:pallet"`` becomes ``"pallet"``, matching
    ``shortModel()`` in ``apps/workcell/index.html`` (line 1133), which
    takes the third colon-separated segment for display. A string with no
    colon is returned unchanged, so a bare name passed in by mistake still
    matches rather than silently losing its shape.
    """
    return model.rsplit(":", 1)[-1]


def _is_workcell_component(schema_reply: Mapping[str, Any]) -> bool:
    """Whether a ``get_schema`` reply marks its resource as a
    ``viam:workcell-components`` component.

    Transcribed from index.html (lines 571-578): every generic dependency
    is probed with ``get_schema``, and one that errors or answers with no
    non-empty ``schema`` list is not a workcell component and is skipped,
    "duck-typed discovery" in the app's own words. This lets
    ``materialise_workcell`` probe every generic dependency scene-finalizer
    is given, world/arm/gripper/camera siblings included, with no need for
    a second list of names to tell them apart.
    """
    schema = schema_reply.get("schema")
    return isinstance(schema, list) and len(schema) > 0


IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


def _parse_pose(pose: Mapping[str, Any]) -> tuple[Vec3, Quat]:
    """A wire pose's position (metres) and ``(w, x, y, z)`` orientation:
    millimetres plus an ``o_x``/``o_y``/``o_z``/``theta`` orientation vector
    with theta in degrees, the one pose shape every workcell verb uses.

    An absent or zero orientation vector is unrotated. That differs from a
    render primitive's ``_pose_orientation``, whose zero vector means
    "standing upright" for a capsule, because here the pose is a frame and
    "no orientation reported" has to mean "not rotated".
    """
    position: Vec3 = (
        float(pose.get("x", 0.0)) / MM_PER_M,
        float(pose.get("y", 0.0)) / MM_PER_M,
        float(pose.get("z", 0.0)) / MM_PER_M,
    )
    ox = float(pose.get("o_x", 0.0))
    oy = float(pose.get("o_y", 0.0))
    oz = float(pose.get("o_z", 0.0))
    if (ox, oy, oz) == (0.0, 0.0, 0.0):
        return position, IDENTITY
    theta_deg = float(pose.get("theta", 0.0))
    return position, ov_to_quat(ox, oy, oz, math.radians(theta_deg))


def group_anchor(name: str, visuals: Mapping[str, Any]) -> tuple[Vec3, Quat]:
    """The pose every other primitive in a ``get_visuals`` reply is relative
    to: its ``<name>/group`` frame primitive, in metres and as a quaternion.

    Read from ``visuals_group.go`` in ``viam-labs/workcell-components``
    0.7.0: every component builds its primitives in the world frame, then
    ``groupUnderFrame`` re-expresses each one relative to an anchor frame
    labelled ``<component>/group`` that sits at the component's frame pose.
    So the anchor is the one pose the children are guaranteed to be relative
    to, by construction, and it travels in the same reply.

    ``get_attributes.pose`` is not that pose for every model, which is why
    it is not read here. ``pick-station`` reports its bottom-left-top corner
    there, centre plus ``(-width/2, -length/2, +thickness/2)``, while its
    primitives are anchored on the centre. Composing them onto the corner put
    the cell's whole pick station (200, -1200, 220) where its frame, and its
    collider, said (400, -650, 200). The pallet and every decoration report
    the same pose both ways, which is how the difference stayed hidden.

    A reply with no group frame carries primitives already in the world
    frame, so its anchor is the identity.
    """
    for primitive in visuals.get("visuals") or []:
        if not isinstance(primitive, Mapping) or primitive.get("type") != "frame":
            continue
        if primitive.get("label") != f"{name}/group":
            continue
        pose = primitive.get("pose")
        return _parse_pose(pose if isinstance(pose, Mapping) else {})
    return (0.0, 0.0, 0.0), IDENTITY


def _geometries_to_mappings(geometries: Any) -> list[Mapping[str, Any]]:
    """One ``GetGeometries`` reply's boxes as ``workcell_scenery.parse_geometries``
    expects: a label and a millimetre-axis ``box_dims_mm``."""
    return [
        {
            "label": geometry.label,
            "box_dims_mm": {
                "x": geometry.box.dims_mm.x,
                "y": geometry.box.dims_mm.y,
                "z": geometry.box.dims_mm.z,
            },
        }
        for geometry in geometries
    ]


async def gather_component_scenery(
    name: str, resource: ResourceBase, *, logger: Logger
) -> ComponentScenery | None:
    """One dependency's ``ComponentScenery``, or ``None`` when it is not a
    workcell component or fails to answer as one.

    ``get_schema`` decides whether ``resource`` is a
    ``viam:workcell-components`` component at all (transcribed
    discovery rule, see ``_is_workcell_component``); a dependency that is
    not one, such as this cell's own world, arm, gripper or camera, is
    skipped with no log line, since that is the expected shape of most
    dependencies scene-finalizer is given. Once a component passes that
    check, ``get_status`` (for its model, transcribed from index.html line
    582's ``st.model``), ``get_attributes`` (for its attributes, index.html
    line 584) and ``get_visuals`` (index.html line 586, and the frame its
    primitives are anchored to, see ``group_anchor``) are called, plus
    ``GetGeometries`` when the
    model's bare name is one of ``workcell_scenery.SHAPED_MODELS``. Any
    exception past the schema check is logged with the component's name and
    skipped, so one workcell component that stops answering never stops the
    rest of the cell from materialising.
    """
    try:
        schema_reply = await resource.do_command(_verb("get_schema"))
    # a third-party component may raise anything, and a probe is not allowed to stop a boot
    except Exception as exc:  # noqa: BLE001
        # Logged, not silent. A workcell component that answers this probe
        # fine from outside the module but raises here is the difference
        # between "correctly skipped a non-scenery sibling" and "lost half
        # the cell", and the two are indistinguishable without the reason.
        logger.warning("%r did not answer get_schema (%s: %s)", name, type(exc).__name__, exc)
        return None
    if not _is_workcell_component(schema_reply):
        return None

    try:
        status = await resource.do_command(_verb("get_status"))
        attrs = dict(await resource.do_command(_verb("get_attributes")))
        visuals = await resource.do_command(_verb("get_visuals"))
        model = str(status.get("model", name))
    # same rule as the probe: one component that stops answering never stops the cell
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "workcell component %r did not answer get_status/get_attributes/get_visuals "
            "(%s: %s); skipping its scenery",
            name,
            type(exc).__name__,
            exc,
        )
        return None

    geometries: list[Mapping[str, Any]] = []
    if bare_model_name(model) in SHAPED_MODELS:
        try:
            shapes = await cast(ComponentBase, resource).get_geometries()
            geometries = _geometries_to_mappings(shapes)
        # GetGeometries is served by the Go type but NOT over the generic API:
        # every workcell component answers UNIMPLEMENTED for it on the wire.
        # Losing a collider is not a reason to lose the component's visuals
        # too, which is what folding this in with the DoCommands above did.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%r has no GetGeometries on the wire (%s: %s); its collider must be "
                "derived from attributes instead",
                name,
                type(exc).__name__,
                exc,
            )

    # the reported pose is not the anchor the primitives are relative to (see
    # group_anchor), and it must not reach derived_collider as an attribute
    attrs.pop("pose", None)
    position, orientation = group_anchor(name, visuals)
    return ComponentScenery(
        model=model,
        visuals=visuals,
        geometries=geometries,
        attrs=attrs,
        frame_position_m=position,
        frame_orientation_wxyz=orientation,
    )


async def gather_workcell_scenery(
    resources: Mapping[str, ResourceBase], *, logger: Logger
) -> dict[str, ComponentScenery]:
    """Every dependency in ``resources`` that answers as a workcell
    component, in iteration order, as its ``ComponentScenery``."""
    scenery: dict[str, ComponentScenery] = {}
    logger.info(
        "gathering workcell scenery from %d dependencies: %s", len(resources), list(resources)
    )
    for name, resource in resources.items():
        started = time.monotonic()
        try:
            one = await asyncio.wait_for(
                gather_component_scenery(name, resource, logger=logger), PROBE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning(
                "%r did not answer within %.0fs; skipping its scenery", name, PROBE_TIMEOUT_S
            )
            continue
        logger.info(
            "probed %r in %.2fs: %s",
            name,
            time.monotonic() - started,
            "scenery" if one is not None else "not a workcell component",
        )
        if one is not None:
            scenery[name] = one
    logger.info("gathered scenery for %d components: %s", len(scenery), list(scenery))
    return scenery


def generic_dependencies(
    dependencies: Mapping[ResourceName, ResourceBase],
) -> dict[str, ResourceBase]:
    """The ``rdk:component:generic`` dependencies in ``dependencies``, keyed
    by name.

    ``viam:workcell-components`` scenery components are all
    ``rdk:component:generic``, the same as this module's own world and
    scene-finalizer, so this is a cheap first filter before the
    ``get_schema`` probe decides which of them are actually scenery.
    """
    return {
        resource_name.name: resource
        for resource_name, resource in dependencies.items()
        if resource_name.type == "component" and resource_name.subtype == "generic"
    }


def materialise_workcell(
    resources: Mapping[str, ResourceBase],
    *,
    logger: Logger,
    loop: asyncio.AbstractEventLoop | None = None,
) -> dict[str, ComponentScenery]:
    """The synchronous face of ``gather_workcell_scenery``, for a caller
    (``scene_finalizer``) whose own ``new``/``reconfigure`` cannot await.

    ``loop`` must be the module's own running event loop, and the gathering
    is scheduled onto it. A dependency handed to a module is a client bound
    to that loop: awaiting it from a private loop on another thread never
    completes, because the loop that owns its channel is not the one driving
    the await. That hangs on the first probe and never times out on its own,
    which parks the world at its scene gate forever and looks like a hung
    simulator rather than a failed call.

    With no loop (mock and tests, where the resources are plain fakes with no
    channel behind them) the gathering runs on a private loop instead.
    """
    if loop is not None:
        logger.info("gathering on the module's own event loop")
        future = asyncio.run_coroutine_threadsafe(
            gather_workcell_scenery(resources, logger=logger), loop
        )
        try:
            return future.result(GATHER_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"gathering workcell scenery did not finish within {GATHER_TIMEOUT_S}s; "
                "the scene is finalized without it so the world can still boot"
            ) from None

    logger.info("gathering on a private event loop, no module loop was captured")
    result: dict[str, dict[str, ComponentScenery]] = {}
    error: list[BaseException] = []

    def run() -> None:
        try:
            result["scenery"] = asyncio.run(gather_workcell_scenery(resources, logger=logger))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True, name="workcell-gather")
    thread.start()
    thread.join(GATHER_TIMEOUT_S)
    if thread.is_alive():
        raise TimeoutError(
            f"gathering workcell scenery did not finish within {GATHER_TIMEOUT_S}s; "
            "the scene is finalized without it so the world can still boot"
        )
    if error:
        raise error[0]
    return result["scenery"]
