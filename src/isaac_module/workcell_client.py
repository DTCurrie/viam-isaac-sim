"""Fetches each configured workcell component's geometry and visuals over
the Viam API, and turns the replies into ``ComponentScenery`` for
``SimManager.materialise_components``.

Every fact this module needs about a component (its model, its resolved
world pose, its attributes, its render primitives, and its collider box
where one exists) is fetched live from the component itself. None of it is
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
import math
import threading
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


def _parse_pose(pose: Mapping[str, Any]) -> tuple[Vec3, Quat]:
    """A resolved world pose's position (metres) and ``(w, x, y, z)``
    orientation.

    Reconstructed by convention, not transcribed: index.html never calls
    ``get_pose``, so its reply shape is not directly observed. Every other
    pose on this wire (``get_visuals``'s per-primitive ``pose``, and
    ``get_attributes``'s ``pose`` field, both millimetres plus an
    ``o_x``/``o_y``/``o_z``/``theta`` orientation vector in degrees) uses
    this same shape, so ``get_pose`` is assumed to match it. Unlike a
    render primitive's pose, a component's own resolved orientation defaults
    to identity when absent rather than pointing up, since "no orientation
    reported" should mean "not rotated", not the render-only convenience
    default ``_pose_orientation`` uses for a primitive standing upright.
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
        return position, (1.0, 0.0, 0.0, 0.0)
    theta_deg = float(pose.get("theta", 0.0))
    return position, ov_to_quat(ox, oy, oz, math.radians(theta_deg))


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
    582's ``st.model``), ``get_attributes`` (for its attributes and its
    resolved ``pose``, index.html lines 584 and 678) and ``get_visuals``
    (index.html line 586) are called, plus ``GetGeometries`` when the
    model's bare name is one of ``workcell_scenery.SHAPED_MODELS``. Any
    exception past the schema check is logged with the component's name and
    skipped, so one workcell component that stops answering never stops the
    rest of the cell from materialising.
    """
    try:
        schema_reply = await resource.do_command(_verb("get_schema"))
    # a third-party component may raise anything, and a probe is not allowed to stop a boot
    except Exception:  # noqa: BLE001
        return None
    if not _is_workcell_component(schema_reply):
        return None

    try:
        status = await resource.do_command(_verb("get_status"))
        attrs = dict(await resource.do_command(_verb("get_attributes")))
        visuals = await resource.do_command(_verb("get_visuals"))
        model = str(status.get("model", name))
        geometries: list[Mapping[str, Any]] = []
        if bare_model_name(model) in SHAPED_MODELS:
            shapes = await cast(ComponentBase, resource).get_geometries()
            geometries = _geometries_to_mappings(shapes)
    # same rule as the probe: one component that stops answering never stops the cell
    except Exception:  # noqa: BLE001
        logger.warning(
            "workcell component %r did not answer get_status/get_attributes/"
            "get_visuals/GetGeometries; skipping its scenery",
            name,
        )
        return None

    pose = attrs.pop("pose", None)
    position, orientation = _parse_pose(pose if isinstance(pose, Mapping) else {})
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
    for name, resource in resources.items():
        one = await gather_component_scenery(name, resource, logger=logger)
        if one is not None:
            scenery[name] = one
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
    resources: Mapping[str, ResourceBase], *, logger: Logger
) -> dict[str, ComponentScenery]:
    """The synchronous face of ``gather_workcell_scenery``, for a caller
    (``scene_finalizer``) whose own ``new``/``reconfigure`` cannot await.

    Runs the gathering on a dedicated thread with its own event loop and
    blocks until it finishes, rather than trying to drive it on whatever
    event loop the calling thread may or may not already have running.
    """
    result: dict[str, dict[str, ComponentScenery]] = {}
    error: list[BaseException] = []

    def run() -> None:
        try:
            result["scenery"] = asyncio.run(gather_workcell_scenery(resources, logger=logger))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result["scenery"]
