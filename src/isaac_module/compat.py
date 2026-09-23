"""Isaac Sim compatibility layer.

Everything the module needs from Isaac Sim is imported HERE, as data on a
namespace object, so a future Isaac Sim release only needs one place to
change. Version- and capability-dependent behavior is keyed on
:func:`isaac_version` / :func:`caps`, never on scattered ``try/except
ImportError`` in feature code.

Mock mode never imports Isaac: :func:`isaac_version` returns ``None`` and
:func:`caps` returns the 5.0 row.
"""

from __future__ import annotations

import functools
import importlib.metadata
import re
from types import SimpleNamespace
from typing import Any, NamedTuple, Protocol, cast

from viam.logging import getLogger

IsaacVersion = tuple[int, int, int]

_LOGGER = getLogger(__name__)

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class IsaacAPI(Protocol):
    """The attributes :func:`import_isaac` sets on its return value.

    Every attribute names an Isaac Sim (or ``pxr``) type, none of which is
    importable outside Isaac Sim, so each is typed ``Any`` rather than
    pulling in an import that can't be type-checked.
    """

    World: Any
    add_reference_to_stage: Any
    open_stage: Any
    get_assets_root_path: Any
    SingleArticulation: Any
    SingleXFormPrim: Any
    ArticulationAction: Any
    client: Any
    DynamicCuboid: Any
    FixedCuboid: Any
    get_prim_at_path: Any
    Camera: Any
    WheeledRobot: Any
    DifferentialController: Any
    PhysicsMaterial: Any
    OmniPBR: Any
    PreviewSurface: Any
    PhysxSchema: Any
    UsdPhysics: Any


def import_isaac() -> IsaacAPI:
    """Import everything the module needs from Isaac Sim's ``isaacsim.*``
    namespace."""

    ns = SimpleNamespace()

    from isaacsim.core.api import World

    ns.World = World

    from isaacsim.core.utils.stage import add_reference_to_stage, open_stage

    ns.add_reference_to_stage = add_reference_to_stage
    ns.open_stage = open_stage

    from isaacsim.storage.native import get_assets_root_path

    ns.get_assets_root_path = get_assets_root_path

    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim

    ns.SingleArticulation = SingleArticulation
    ns.SingleXFormPrim = SingleXFormPrim

    from isaacsim.core.utils.types import ArticulationAction

    ns.ArticulationAction = ArticulationAction

    try:
        import omni.client

        ns.client = omni.client
    except ImportError:
        ns.client = None

    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

    ns.DynamicCuboid = DynamicCuboid
    ns.FixedCuboid = FixedCuboid

    try:
        from isaacsim.core.utils.prims import get_prim_at_path
    except ImportError:
        get_prim_at_path = None
    ns.get_prim_at_path = get_prim_at_path

    from isaacsim.sensors.camera import Camera

    ns.Camera = Camera

    from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
        DifferentialController,
    )
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot

    ns.WheeledRobot = WheeledRobot
    ns.DifferentialController = DifferentialController

    # Physics materials and the USD physics schemas, exposed on the
    # namespace (None when absent) so physics.py can be driven with fakes on
    # a machine without Kit.
    try:
        from isaacsim.core.api.materials import PhysicsMaterial

        ns.PhysicsMaterial = PhysicsMaterial
    except ImportError:
        ns.PhysicsMaterial = None
    # visual materials (materials.py builds OmniPBR; PreviewSurface is the
    # flat-colour shader the cuboids carry by default), None when absent
    try:
        from isaacsim.core.api.materials import OmniPBR, PreviewSurface

        ns.OmniPBR = OmniPBR
        ns.PreviewSurface = PreviewSurface
    except ImportError:
        ns.OmniPBR = None
        ns.PreviewSurface = None
    try:
        from pxr import PhysxSchema, UsdPhysics

        ns.PhysxSchema = PhysxSchema
        ns.UsdPhysics = UsdPhysics
    except ImportError:
        ns.PhysxSchema = None
        ns.UsdPhysics = None

    return cast(IsaacAPI, ns)


@functools.lru_cache(maxsize=1)
def import_surface_gripper() -> dict[str, Any]:
    """Isaac's surface gripper extension and the USD modules that author it,
    imported once per process. Enables ``isaacsim.robot.surface_gripper``,
    then returns a mapping with the keys ``report`` (what was found, for a
    boot log line), ``surface_gripper`` (the ``_surface_gripper`` module,
    whose ``acquire_surface_gripper_interface()`` drives a gripper by path),
    ``robot_schema``, ``Gf``, ``Sdf``, ``UsdGeom``, ``UsdPhysics`` and
    ``PhysxSchema`` (None where pxr has none). Isaac only."""
    report: dict[str, Any] = {}
    from isaacsim.core.utils.extensions import enable_extension

    report["extension_enabled"] = bool(enable_extension("isaacsim.robot.surface_gripper"))
    from isaacsim.robot.surface_gripper import _surface_gripper

    try:
        from usd.schema.isaac import robot_schema
    except ImportError as first:
        try:
            from isaacsim.robot.schema import robot_schema
        except ImportError as second:
            raise ImportError(
                f"no robot_schema module: usd.schema.isaac ({first}); "
                f"isaacsim.robot.schema ({second})"
            ) from second
        report["robot_schema_module"] = "isaacsim.robot.schema"
    else:
        report["robot_schema_module"] = "usd.schema.isaac"
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:
        # a module name, kept as pxr spells it
        PhysxSchema = None
    report["physx_schema"] = PhysxSchema is not None

    return {
        "report": report,
        "surface_gripper": _surface_gripper,
        "robot_schema": robot_schema,
        "Gf": Gf,
        "Sdf": Sdf,
        "UsdGeom": UsdGeom,
        "UsdPhysics": UsdPhysics,
        "PhysxSchema": PhysxSchema,
    }


def _parse_version(value: Any) -> IsaacVersion | None:
    """Tolerantly extract (major, minor, patch) from an unknown-shape value.

    ``get_version()``'s return shape is undocumented: a ``str`` is regexed
    directly; a sequence is first regexed after joining its stringified
    elements, then falls back to its first three int-like elements.
    """
    if isinstance(value, str):
        match = _VERSION_RE.search(value)
        return (int(match[1]), int(match[2]), int(match[3])) if match else None

    if isinstance(value, (list, tuple)):
        joined = " ".join(str(item) for item in value)
        match = _VERSION_RE.search(joined)
        if match:
            return (int(match[1]), int(match[2]), int(match[3]))

        parts: list[int] = []
        for item in value:
            try:
                parts.append(int(item))
            except (TypeError, ValueError):
                continue
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        return None

    return None


def _probe_isaacsim_core_version() -> Any:
    from isaacsim.core.version import get_version

    return get_version()


def _probe_importlib_metadata() -> Any:
    return importlib.metadata.version("isaacsim")


_PROBES: tuple[Any, ...] = (
    _probe_isaacsim_core_version,
    _probe_importlib_metadata,
)


def isaac_version() -> IsaacVersion | None:
    """Best-effort (major, minor, patch) of the installed Isaac Sim.

    ``None`` when Isaac is not importable (mock mode, unit tests) or when no
    known probe answers. Try the candidates in order and stop at the first
    that works; never raise.
    """
    for probe in _PROBES:
        try:
            raw = probe()
        except Exception:  # noqa: BLE001 - any probe failure just tries the next
            continue
        parsed = _parse_version(raw)
        if parsed is not None:
            _LOGGER.info(
                "isaac_version: %s answered raw=%r parsed=%r",
                probe.__name__,
                raw,
                parsed,
            )
            return parsed
    return None


class Caps(NamedTuple):
    """Capability flags the feature code branches on.

    Keep the field order append-only.
    """

    # Robotiq 2F-85 asset differences per release. Kept here so
    # models/gripper.py never branches on the version.
    gripper_closed_deg: float  # finger_joint angle at full closure (open is 0)
    gripper_dof_count: int  # DOFs the gripper adds to the arm articulation
    # finger_joint angle at the OPEN limit: 0 on both releases (a 7.76 deg rest
    # seen on the GPU before the articulation fixes was an artifact; run 19
    # reaches 0.4 deg at an open target of 0)
    gripper_open_deg: float
    # Camera(annotator_device=...)/get_*(device=...) GPU-resident data path
    # exists only on 5.0 (CHANGELOG 0.4.0); 4.5 always lands in host numpy
    camera_supports_annotator_device: bool


CAPS_BY_RELEASE: dict[tuple[int, int], Caps] = {
    (5, 0): Caps(
        gripper_closed_deg=47.0,
        gripper_dof_count=6,
        gripper_open_deg=0.0,
        camera_supports_annotator_device=True,
    ),
}


def caps(version: IsaacVersion | None = None) -> Caps:
    """Flags for ``version`` (default: :func:`isaac_version`).

    ``CAPS_BY_RELEASE`` carries a single row now that Isaac Sim 4.5 support
    is dropped, so every version, known or not, resolves to it. ``version``
    stays a parameter so existing call sites are unchanged.
    """
    return CAPS_BY_RELEASE[(5, 0)]
