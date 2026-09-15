"""Viam's workcell components as Isaac props.

``viam:workcell-components`` describes a cell two ways, and between them
everything in it is covered.

``GetGeometries`` is the ``resource.Shaped`` path. ``pallet``, ``pick-station``
and ``safety-fence`` implement it and nothing else does. Each returns a coarse
typed box, which is what the frame system plans against and what PhysX wants
for a collider.

``get_visuals`` is a DoCommand verb every component serves. It decomposes a
component into primitives: a pedestal is a box base, a capsule column and a box
flange. That is render detail, not collision detail.

So the rule is: collider from ``GetGeometries`` where the component offers one,
render from ``get_visuals`` for everything, and a derived collider for the
components that have no ``GetGeometries`` but that the arm or a box still
touches. ``robot-pedestal`` is the known one, from its ``height_mm`` and
``diameter_mm``. Whether ``scan-tunnel`` needs the same is a GPU question, not
a source one.

Everything here is pure. A primitive is parsed, converted and unit-tested with
no stage, and ``SimManager`` turns the prop dicts this returns into prims the
same way it spawns any other configured prop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .spatial import ov_to_quat
from .visual_props import FIT_TRUE, VISUAL_PROP_KIND

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

PrimitiveKind = Literal["box", "capsule", "mesh"]

# the models that implement resource.Shaped, so their collider is read rather
# than derived. Read from viam:workcell-components 0.7.0.
SHAPED_MODELS: frozenset[str] = frozenset({"pallet", "pick-station", "safety-fence"})

# models with no GetGeometries that something in the cell still touches, so
# their collider comes from their own attributes. The value is the attribute
# set the derivation reads.
DERIVED_COLLIDER_MODELS: Mapping[str, tuple[str, ...]] = {
    "robot-pedestal": ("height_mm", "diameter_mm"),
}

MM_PER_M = 1000.0


@dataclass(frozen=True)
class SceneryPrimitive:
    """One primitive out of a component, in metres and world-agnostic.

    The pose is in the component's own frame. Placing it in the cell is the
    caller's job, because only the caller knows the component's frame.

    ``dims_m`` is set for a box, ``radius_m`` and ``length_m`` for a capsule,
    ``mesh_path`` for a mesh, and exactly one of those three groups is set.
    """

    kind: PrimitiveKind
    label: str
    position_m: Vec3 = (0.0, 0.0, 0.0)
    orientation_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    dims_m: Vec3 | None = None
    radius_m: float | None = None
    length_m: float | None = None
    mesh_path: str | None = None
    color: Vec3 | None = None
    opacity: float = 1.0


def _mm_axes_m(value: Mapping[str, Any] | None) -> Vec3:
    """An axis-keyed ``{"x":.., "y":.., "z":..}`` millimetre object as a
    metres triple, ``(0, 0, 0)`` when absent."""
    if value is None:
        return (0.0, 0.0, 0.0)
    return (
        float(value.get("x", 0.0)) / MM_PER_M,
        float(value.get("y", 0.0)) / MM_PER_M,
        float(value.get("z", 0.0)) / MM_PER_M,
    )


def _pose_position_m(pose: Mapping[str, Any]) -> Vec3:
    """A ``pose``'s ``x``/``y``/``z`` millimetres as a metres triple."""
    return _mm_axes_m(pose)


def _pose_orientation(pose: Mapping[str, Any]) -> Quat:
    """The ``(w, x, y, z)`` quaternion for one ``get_visuals`` primitive's
    ``pose``.

    Its ``o_x``/``o_y``/``o_z`` default to straight up when absent or zero
    length, matching the reference viewer's own fallback, and ``theta`` is
    authored in degrees.
    """
    ox = float(pose.get("o_x", 0.0))
    oy = float(pose.get("o_y", 0.0))
    oz = float(pose.get("o_z", 0.0))
    if (ox, oy, oz) == (0.0, 0.0, 0.0):
        ox, oy, oz = (0.0, 0.0, 1.0)
    theta_deg = float(pose.get("theta", 0.0))
    return ov_to_quat(ox, oy, oz, math.radians(theta_deg))


def _visual_color(item: Mapping[str, Any]) -> tuple[Vec3 | None, float]:
    """The primitive's colour as a 0-1 ``(r, g, b)`` triple and its opacity,
    from a nested ``color: {r, g, b, opacity}``. ``(None, 1.0)`` when the
    wire carries no ``color``."""
    color = item.get("color")
    if not isinstance(color, Mapping):
        return None, 1.0
    rgb = (
        float(color.get("r", 0.0)) / 255.0,
        float(color.get("g", 0.0)) / 255.0,
        float(color.get("b", 0.0)) / 255.0,
    )
    return rgb, float(color.get("opacity", 1.0))


def parse_visuals(payload: Mapping[str, Any]) -> list[SceneryPrimitive]:
    """The primitives in one component's ``get_visuals`` reply.

    Millimetres on the wire become metres here, and the ``pose``'s
    ``o_x``/``o_y``/``o_z`` plus ``theta`` becomes a ``(w, x, y, z)``
    quaternion via ``spatial.ov_to_quat``. A primitive whose type is not
    box, capsule or mesh is skipped rather than raising, so an upstream
    component gaining a shape does not stop a cell booting.
    """
    primitives: list[SceneryPrimitive] = []
    for item in payload.get("visuals") or []:
        kind = item.get("type")
        if kind not in ("box", "capsule", "mesh"):
            continue
        pose = item.get("pose") or {}
        color, opacity = _visual_color(item)
        primitives.append(
            SceneryPrimitive(
                kind=kind,
                label=str(item.get("label", "")),
                position_m=_pose_position_m(pose),
                orientation_wxyz=_pose_orientation(pose),
                dims_m=_mm_axes_m(item["dims_mm"]) if kind == "box" else None,
                radius_m=float(item["radius_mm"]) / MM_PER_M if kind == "capsule" else None,
                length_m=float(item["length_mm"]) / MM_PER_M if kind == "capsule" else None,
                mesh_path=str(item["mesh_path"]) if kind == "mesh" else None,
                color=color,
                opacity=opacity,
            )
        )
    return primitives


def parse_geometries(geometries: Sequence[Mapping[str, Any]]) -> list[SceneryPrimitive]:
    """The primitives in one component's ``GetGeometries`` reply.

    Every one is a box at a zero pose with a label, which is the only shape
    workcell-components returns here.
    """
    return [
        SceneryPrimitive(
            kind="box",
            label=str(geometry.get("label", "")),
            dims_m=_mm_axes_m(geometry["box_dims_mm"]),
        )
        for geometry in geometries
    ]


def derived_collider(model: str, attrs: Mapping[str, Any]) -> SceneryPrimitive | None:
    """The collider for a component that serves no ``GetGeometries``.

    ``None`` for a model that is not in ``DERIVED_COLLIDER_MODELS``, which is
    how scenery stays render-only by default. ``robot-pedestal`` becomes one
    box spanning its ``diameter_mm`` and standing its ``height_mm``, because a
    box under an arm base is enough and PhysX prefers it to a capsule.
    """
    if model not in DERIVED_COLLIDER_MODELS:
        return None
    height_m = float(attrs["height_mm"]) / MM_PER_M
    diameter_m = float(attrs["diameter_mm"]) / MM_PER_M
    return SceneryPrimitive(
        kind="box",
        label=f"{model}-collider",
        position_m=(0.0, 0.0, height_m / 2.0),
        dims_m=(diameter_m, diameter_m, height_m),
    )


def _bounding_box_dims_m(primitive: SceneryPrimitive) -> Vec3:
    """The primitive's own box dims, or its capsule's bounding box. Raises
    for a mesh, whose extent this module has no way to measure."""
    if primitive.dims_m is not None:
        return primitive.dims_m
    if primitive.radius_m is not None and primitive.length_m is not None:
        diameter = 2.0 * primitive.radius_m
        return (diameter, diameter, primitive.length_m)
    raise ValueError(
        f"cannot approximate a bounding box for a {primitive.kind!r} primitive "
        f"({primitive.label!r}) with no known extent"
    )


def collider_prop(primitive: SceneryPrimitive, *, name: str) -> dict[str, Any]:
    """A ``_spawn_prop`` cube dict for a primitive used as a collider.

    Fixed, uncoloured, and sized by ``size`` x ``scale`` the way every other
    fixed prop in a cell is. A non-box primitive is approximated by its
    bounding box, since a collider is allowed to be coarser than the render.
    """
    return {
        "name": name,
        "type": "cube",
        "position": primitive.position_m,
        "orientation_wxyz": primitive.orientation_wxyz,
        "size": 1.0,
        "scale": _bounding_box_dims_m(primitive),
        "fixed": True,
    }


def render_prop(primitive: SceneryPrimitive, *, name: str) -> dict[str, Any]:
    """A ``_spawn_prop`` dict for a primitive used as render only.

    A mesh primitive becomes a ``visual`` prop (``visual_props``), and a box
    or capsule becomes a cube carrying the primitive's colour. Every dict
    carries ``"collision": False``, since a render prop must never become
    physical geometry even before ``_spawn_prop`` learns to honour that key.
    """
    if primitive.kind == "mesh":
        return {
            "name": name,
            "type": VISUAL_PROP_KIND,
            "usd_path": primitive.mesh_path,
            "position": primitive.position_m,
            "orientation_wxyz": primitive.orientation_wxyz,
            "fit": FIT_TRUE,
            "collision": False,
        }
    prop: dict[str, Any] = {
        "name": name,
        "type": "cube",
        "position": primitive.position_m,
        "orientation_wxyz": primitive.orientation_wxyz,
        "size": 1.0,
        "scale": _bounding_box_dims_m(primitive),
        "fixed": True,
        "collision": False,
    }
    if primitive.color is not None:
        prop["color"] = primitive.color
    return prop


def scenery_props(
    component: str,
    model: str,
    *,
    visuals: Mapping[str, Any],
    geometries: Sequence[Mapping[str, Any]],
    attrs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Every prop dict one configured component contributes to the stage.

    Colliders first, then render props, because ``_spawn_prop`` resolves a
    visual prop's ``fit.collider`` against props already spawned. Prop names
    are derived from ``component`` and the primitive's label, so two
    components with the same primitive labels do not collide.
    """
    collider_primitives: list[SceneryPrimitive] = []
    if model in SHAPED_MODELS:
        collider_primitives.extend(parse_geometries(geometries))
    else:
        derived = derived_collider(model, attrs)
        if derived is not None:
            collider_primitives.append(derived)

    props: list[dict[str, Any]] = [
        collider_prop(primitive, name=f"{component}-{primitive.label}")
        for primitive in collider_primitives
    ]
    props.extend(
        render_prop(primitive, name=f"{component}-{primitive.label}")
        for primitive in parse_visuals(visuals)
    )
    return props
    raise NotImplementedError
