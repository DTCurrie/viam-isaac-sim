"""Visual-only props: a referenced USD with a pose and a scale, and nothing
else.

A ``type: "visual"`` prop dresses a collider the cell already verified. It is
referenced into the stage, posed, scaled, and then left alone: no rigid body,
no collider, no ``world.scene`` registration, no entry in the prop registry
every scene verb reads (``SimManager._prop_specs`` and the mock's registry).
``prop_geometries``, ``get_geometries``, ``randomize_props``, ``set_prop_pose``,
``scatter_cell``, ``clear_cell`` and the live rescale never see one, and each
rejects a visual prop's name with ``visual_prop_error``.

The scale comes from one of three places, at most one of them configured:

* ``scale: [sx, sy, sz]`` is used as given.
* ``fit: "true"`` (or neither key) keeps the asset's authored size, ``(1, 1, 1)``.
* ``fit: {"collider": "<cube prop name>"}`` scales the asset's authored bounds
  onto the named cube's ``size x scale`` box, per axis (``fit_scale``). The
  converter (``tools/convert_mesh.py``) puts the asset origin at the centre of
  its top face, so ``position`` is the collider's top-centre.

Everything here is pure so the mock and the Isaac path share one record shape
(``visual_prop_record``) and the arithmetic is unit-tested without a stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .physics import PROP_PHYSICS_KEYS

VISUAL_PROP_KIND = "visual"
PROP_KINDS = ("cube", "usd", VISUAL_PROP_KIND)
# the fit opt-out: keep the asset's authored size
FIT_TRUE = "true"
# a visual prop has no collider and no rigid body, so nothing that describes
# one may appear on it
VISUAL_REJECTED_KEYS = frozenset({"size", "color", "fixed", "box_dims"}) | frozenset(
    PROP_PHYSICS_KEYS
)
UNIT_SCALE = (1.0, 1.0, 1.0)

Vec3 = tuple[float, float, float]


def is_visual_prop(prop: Mapping[str, Any]) -> bool:
    return str(prop.get("type", "cube")) == VISUAL_PROP_KIND


def fit_collider_name(prop: Mapping[str, Any]) -> str | None:
    """The cube prop a ``fit: {"collider": ...}`` names, unsanitised, or
    ``None`` for ``fit: "true"``, a configured ``scale``, or neither."""
    fit = prop.get("fit")
    if isinstance(fit, Mapping):
        collider = fit.get("collider")
        return str(collider) if collider else None
    return None


def fit_scale(collider_dims_m: Sequence[float], mesh_dims_m: Sequence[float]) -> Vec3:
    """Per-axis ``collider / mesh``: the scale that maps the asset's authored
    bounds onto the collider's box. ``ValueError`` when a mesh dimension is
    not positive (a flat or empty asset cannot be fitted)."""
    scale: list[float] = []
    for axis, (collider, mesh) in enumerate(zip(collider_dims_m, mesh_dims_m, strict=True)):
        if mesh <= 0:
            raise ValueError(
                f"cannot fit a visual prop whose mesh has no extent on axis {axis} "
                f"(mesh dims {tuple(mesh_dims_m)})"
            )
        scale.append(float(collider) / float(mesh))
    return (scale[0], scale[1], scale[2])


def visual_scale(
    prop: Mapping[str, Any],
    collider_dims_m: Sequence[float] | None,
    mesh_dims_m: Sequence[float] | None,
) -> Vec3 | None:
    """The scale triple a visual prop is authored with. ``scale`` as
    configured; ``fit: "true"`` or neither key -> ``UNIT_SCALE``;
    ``fit.collider`` -> ``fit_scale`` when the mesh bounds are known, else
    ``None`` (the mock has no stage to measure)."""
    if prop.get("scale") is not None:
        sx, sy, sz = (float(v) for v in prop["scale"])
        return (sx, sy, sz)
    if fit_collider_name(prop) is None:
        return UNIT_SCALE
    if collider_dims_m is None or mesh_dims_m is None:
        return None
    return fit_scale(collider_dims_m, mesh_dims_m)


def visual_prop_record(
    prop: Mapping[str, Any],
    *,
    name: str,
    resolved_path: str,
    position: Vec3,
    orientation: tuple[float, float, float, float],
    collider_dims_m: Vec3 | None,
    mesh_dims_m: Vec3 | None,
    bounds_m: Mapping[str, Sequence[float]] | None,
) -> dict[str, Any]:
    """The one record shape for ``SimManager._visual_props[name]`` and the
    rows ``status()["visual_props"]`` lists (through ``status_row``).

    ``name`` is the sanitised prim name (the prim is ``/World/<name>``);
    ``resolved_path`` is ``usd_path`` after ``assets.resolve_asset``;
    ``collider_dims_m`` is the named cube's ``prop_box_dims`` or ``None``
    without ``fit.collider``; ``mesh_dims_m`` is the asset's authored bounds
    at unit scale (``None`` in the mock); ``bounds_m`` is the world-space
    axis-aligned box after pose and scale as ``{"min": [x, y, z], "max":
    [x, y, z]}`` (``None`` in the mock). ``scale`` is ``visual_scale`` and
    is ``None`` only for an unmeasured ``fit.collider``. ``collider_hidden``
    is true when the prop fits a collider: the Isaac path sets that cube's
    visibility to invisible so only the mesh renders, its physics untouched."""
    return {
        "name": name,
        "usd_path": str(prop["usd_path"]),
        "resolved_path": resolved_path,
        "position": tuple(float(v) for v in position),
        "spawn_orientation": tuple(float(v) for v in orientation),
        "fit": prop.get("fit"),
        "collider": fit_collider_name(prop),
        "collider_dims_m": collider_dims_m,
        "mesh_dims_m": mesh_dims_m,
        "scale": visual_scale(prop, collider_dims_m, mesh_dims_m),
        "bounds_m": dict(bounds_m) if bounds_m is not None else None,
        "collider_hidden": fit_collider_name(prop) is not None,
    }


def status_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """``visual_prop_record`` with every tuple as a list, so the row survives
    the SDK's struct conversion unchanged."""
    row: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, tuple):
            row[key] = list(value)
        elif isinstance(value, Mapping):
            row[key] = {k: list(v) if isinstance(v, tuple) else v for k, v in value.items()}
        else:
            row[key] = value
    return row


def visual_prop_error(verb: str, name: str) -> str:
    """The message every prop verb raises (as ``ValueError``) when handed a
    visual prop's name."""
    return (
        f'{verb}: {name!r} is a visual prop (type "visual"): it has no collider and '
        "cannot be posed, scattered, rescaled or listed in geometries"
    )
