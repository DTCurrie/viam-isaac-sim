"""PBR materials for cube props and the ground plane.

A prop's ``material`` is either a named bundled set (``"material":
"painted_wood"``) or an explicit object with any of ``albedo``, ``normal``,
``roughness`` and ``metallic`` (texture paths, the schemes ``assets.py``
accepts), ``tint`` (``[r, g, b]`` in [0, 1]) and ``texture_scale`` (``[u, v]``,
each > 0). ``ground.material`` takes the same shape.

Bundled sets live under ``assets/materials/<name>/`` beside one
``assets/materials/manifest.json`` naming each set's maps, its default
``texture_scale``, its source URL and its licence. Only CC0 sets at 1k live
there (each file at most ``MAX_MATERIAL_FILE_BYTES``).

Everything above ``build_material`` is pure so the mock, the validators and
the Isaac path share one record shape (``material_spec``) and one input
mapping (``material_inputs``), unit-tested without a stage.
``build_material`` is the one Isaac call site.

The look the sorting cell ships: anything the wrist camera classifies (the
blocks and the pads) carries no albedo texture. Its configured ``color`` is
the material's ``diffuse_color_constant`` under roughness and normal maps
only, so the six hue detectors see one flat hue per face. The floor may take
a full set with an albedo map.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from viam.logging import getLogger

from .assets import ASSETS_DIR_NAME, MODULE_ROOT, MODULE_SCHEME, REMOTE_ASSET_SCHEMES, resolve_asset

LOGGER = getLogger(__name__)

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

# the four texture maps a material may carry, in the order the docs list them
MATERIAL_MAP_KEYS: tuple[str, ...] = ("albedo", "normal", "roughness", "metallic")
# every key the explicit form of `material` accepts
MATERIAL_KEYS = frozenset(MATERIAL_MAP_KEYS) | frozenset({"tint", "texture_scale"})
MATERIALS_DIR_NAME = "materials"
MANIFEST_NAME = "manifest.json"
MANIFEST_PATH = MODULE_ROOT / ASSETS_DIR_NAME / MATERIALS_DIR_NAME / MANIFEST_NAME
# the sets the module ships; the manifest names exactly these, and the README's
# named-set table lists exactly these
BUNDLED_MATERIAL_NAMES: tuple[str, ...] = ("painted_wood", "painted_mat", "concrete_floor")
MAX_MATERIAL_FILE_BYTES = 3 * 1024 * 1024
# the key both prop registries record the normalised material under: the Isaac
# path in `SimManager._prop_specs[name]`, the mock in its registry entry
MATERIAL_SPEC_KEY = "material_spec"
# material prims live beside the props, one per prop
MATERIAL_PRIM_ROOT = "/World/Looks"

# OmniPBR.mdl input names on Isaac Sim 5.0.0. `material_inputs` produces
# exactly these keys. A texture map only takes effect when its influence
# input is 1, and a colour goes to `diffuse_color_constant` without an albedo
# map (the constant is the albedo then) and to `diffuse_tint` with one (the
# tint multiplies the map).
OMNIPBR_MAP_INPUTS: dict[str, str] = {
    "albedo": "diffuse_texture",
    "normal": "normalmap_texture",
    "roughness": "reflectionroughness_texture",
    "metallic": "metallic_texture",
}
OMNIPBR_INFLUENCE_INPUTS: dict[str, str] = {
    "roughness": "reflection_roughness_texture_influence",
    "metallic": "metallic_texture_influence",
}
# the Sdf.ValueTypeNames attribute each input is authored with
OMNIPBR_INPUT_SDF_TYPES: dict[str, str] = {
    "diffuse_texture": "Asset",
    "normalmap_texture": "Asset",
    "reflectionroughness_texture": "Asset",
    "metallic_texture": "Asset",
    "reflection_roughness_texture_influence": "Float",
    "metallic_texture_influence": "Float",
    "diffuse_color_constant": "Color3f",
    "diffuse_tint": "Color3f",
    "texture_scale": "Float2",
}


@dataclass(frozen=True)
class MaterialSet:
    """One bundled set as the manifest describes it. ``maps`` holds
    ``module://materials/<name>/<file>`` scheme strings, unresolved, so the
    record is the same on every machine."""

    name: str
    maps: Mapping[str, str]
    texture_scale: Vec2 | None
    source: str
    license: str


def _as_vec2(label: str, value: object) -> Vec2:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in value)
    ):
        raise ValueError(f"{label}: texture_scale must be two numbers > 0 (got {value!r})")
    return (float(value[0]), float(value[1]))


def _as_vec3(label: str, key: str, value: object) -> Vec3:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 3
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
        or any(not (0 <= v <= 1) for v in value)
    ):
        raise ValueError(f"{label}: {key} must be three numbers in [0, 1] (got {value!r})")
    return (float(value[0]), float(value[1]), float(value[2]))


_MANIFEST_SET_KEYS = frozenset({"maps", "texture_scale", "source", "license"})


def _manifest_map_path(label: str, set_name: str, map_key: object, file_name: object) -> str:
    if map_key not in MATERIAL_MAP_KEYS:
        raise ValueError(f"{label}: unknown map {map_key!r}, expected {MATERIAL_MAP_KEYS}")
    if (
        not isinstance(file_name, str)
        or not file_name
        or "/" in file_name
        or "\\" in file_name
        or file_name in (".", "..")
    ):
        raise ValueError(f"{label}: map {map_key!r} must be a bare file name")
    return f"{MODULE_SCHEME}{MATERIALS_DIR_NAME}/{set_name}/{file_name}"


def _manifest_set(path: Path, name: object, entry: object) -> MaterialSet:
    label = f"materials manifest set {name!r}"
    if not name or not isinstance(name, str):
        raise ValueError(f"{path}: set names must be non-empty strings (got {name!r})")
    if not isinstance(entry, Mapping):
        raise ValueError(f"{label}: must be an object")
    maps = entry.get("maps")
    if not isinstance(maps, Mapping) or not maps:
        raise ValueError(f"{label}: 'maps' must be a non-empty object")
    for key in entry:
        if key not in _MANIFEST_SET_KEYS:
            raise ValueError(f"{label}: unknown key {key!r}")
    texture_scale = entry.get("texture_scale")
    return MaterialSet(
        name=name,
        maps={
            map_key: _manifest_map_path(label, name, map_key, file_name)
            for map_key, file_name in maps.items()
        },
        texture_scale=_as_vec2(label, texture_scale) if texture_scale is not None else None,
        source=str(entry.get("source", "")),
        license=str(entry.get("license", "")),
    )


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, MaterialSet]:
    """Parse a materials manifest. Shape: ``{"<set name>": {"maps": {"<map
    key>": "<file name>", ...}, "texture_scale": [u, v], "source": "<url>",
    "license": "<licence>"}, ...}``. ``maps`` needs at least one entry, every
    map key is one of ``MATERIAL_MAP_KEYS``, every file name is a bare name
    (no separator, no ``..``); ``texture_scale`` is optional, ``source`` and
    ``license`` default to ``""``. ``ValueError`` names the offending set."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: manifest must be an object of named sets")
    return {name: _manifest_set(path, name, entry) for name, entry in raw.items()}


@lru_cache(maxsize=1)
def bundled_manifest() -> dict[str, MaterialSet]:
    """``load_manifest`` over the module's own ``MANIFEST_PATH``, read once."""
    return load_manifest(MANIFEST_PATH)


def named_material_names(manifest: Mapping[str, MaterialSet] | None = None) -> tuple[str, ...]:
    """The set names a ``material: "<name>"`` may use, sorted."""
    sets = bundled_manifest() if manifest is None else manifest
    return tuple(sorted(sets))


def _empty_material_spec() -> dict[str, Any]:
    return {
        "name": None,
        **{key: None for key in MATERIAL_MAP_KEYS},
        "tint": None,
        "texture_scale": None,
    }


def _named_material_spec(
    set_name: str,
    color: Sequence[float] | None,
    manifest: Mapping[str, MaterialSet] | None,
    resolve: Callable[[str], str],
) -> dict[str, Any]:
    sets = bundled_manifest() if manifest is None else manifest
    material_set = sets.get(set_name)
    if material_set is None:
        raise ValueError(
            f"unknown material {set_name!r}, bundled sets: {list(named_material_names(sets))}"
        )
    spec = _empty_material_spec()
    spec["name"] = set_name
    for map_key, scheme_path in material_set.maps.items():
        spec[map_key] = resolve(scheme_path)
    spec["texture_scale"] = material_set.texture_scale
    if color is not None:
        spec["tint"] = _as_vec3("material", "color", color)
    return spec


def _explicit_material_spec(
    material: Mapping[str, Any],
    color: Sequence[float] | None,
    resolve: Callable[[str], str],
) -> dict[str, Any]:
    for key in material:
        if key not in MATERIAL_KEYS:
            raise ValueError(f"material: unknown key {key!r}, expected {sorted(MATERIAL_KEYS)}")
    spec = _empty_material_spec()
    for map_key in MATERIAL_MAP_KEYS:
        value = material.get(map_key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"material: {map_key} must be a non-empty path (got {value!r})")
        spec[map_key] = resolve(value)
    tint = material.get("tint")
    if tint is not None and color is not None:
        raise ValueError("material.tint and color cannot both be set")
    if tint is not None:
        spec["tint"] = _as_vec3("material", "tint", tint)
    elif color is not None:
        spec["tint"] = _as_vec3("material", "color", color)
    if material.get("texture_scale") is not None:
        spec["texture_scale"] = _as_vec2("material", material["texture_scale"])
    if all(spec[map_key] is None for map_key in MATERIAL_MAP_KEYS) and spec["tint"] is None:
        raise ValueError(
            f"material needs at least one of {MATERIAL_MAP_KEYS} or a tint (or the prop's color)"
        )
    return spec


def material_spec(
    material: str | Mapping[str, Any],
    *,
    color: Sequence[float] | None,
    manifest: Mapping[str, MaterialSet] | None = None,
    resolve: Callable[[str], str] = resolve_asset,
) -> dict[str, Any]:
    """Pure given ``manifest`` and ``resolve``. One explicit record for either
    form of ``material``::

        {"name": "<set>" | None, "albedo": path | None, "normal": path | None,
         "roughness": path | None, "metallic": path | None,
         "tint": (r, g, b) | None, "texture_scale": (u, v) | None}

    Paths are ``resolve``d (``assets.resolve_asset`` by default). ``color``
    is the prop's own ``color``: with a named set it becomes ``tint``; with
    the explicit form it is the tint only when the object sets none, and
    ``color`` beside ``material.tint`` is a ``ValueError``. An unknown set
    name, an unknown key, and an explicit object with neither a map nor a
    ``tint`` are ``ValueError``s too."""
    if isinstance(material, str):
        return _named_material_spec(material, color, manifest, resolve)
    if not isinstance(material, Mapping):
        raise ValueError(f"material must be a set name or an object (got {material!r})")
    return _explicit_material_spec(material, color, resolve)


def material_inputs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Pure. ``material_spec`` -> ``{OmniPBR.mdl input name: value}``, only the
    inputs the spec sets. Paths stay strings, colours and scales stay
    tuples, influences are ``1.0``. Without an albedo the tint is
    ``diffuse_color_constant``; with one it is ``diffuse_tint``."""
    inputs: dict[str, Any] = {}
    for map_key, input_name in OMNIPBR_MAP_INPUTS.items():
        path = spec.get(map_key)
        if path is None:
            continue
        inputs[input_name] = str(path)
        influence = OMNIPBR_INFLUENCE_INPUTS.get(map_key)
        if influence is not None:
            inputs[influence] = 1.0
    tint = spec.get("tint")
    if tint is not None:
        r, g, b = (float(v) for v in tint)
        inputs["diffuse_tint" if spec.get("albedo") is not None else "diffuse_color_constant"] = (
            r,
            g,
            b,
        )
    texture_scale = spec.get("texture_scale")
    if texture_scale is not None:
        u, v = (float(x) for x in texture_scale)
        inputs["texture_scale"] = (u, v)
    return inputs


def prop_display_color(prop: Mapping[str, Any]) -> Vec3 | None:
    """The flat colour a prop reads as: its ``color``, else the explicit
    ``material.tint``, else ``None``. ``PropGeometry.color`` and the flat
    colour fallback when a material fails to build both come from here."""
    color = prop.get("color")
    if color is not None:
        return (float(color[0]), float(color[1]), float(color[2]))
    material = prop.get("material")
    if isinstance(material, Mapping) and material.get("tint") is not None:
        tint = material["tint"]
        return (float(tint[0]), float(tint[1]), float(tint[2]))
    return None


def material_prim_path(name: str) -> str:
    """Where a prop's material prim is authored: ``/World/Looks/<name>_material``."""
    return f"{MATERIAL_PRIM_ROOT}/{name}_material"


def missing_map_paths(
    spec: Mapping[str, Any], exists: Callable[[str], bool] = os.path.exists
) -> list[str]:
    """Pure given ``exists``. The local map paths in ``spec`` that are not on
    disk. Remote paths (``REMOTE_ASSET_SCHEMES``) are Isaac's resolver's to
    check and never listed."""
    missing: list[str] = []
    for map_key in MATERIAL_MAP_KEYS:
        path = spec.get(map_key)
        if path is None or str(path).startswith(REMOTE_ASSET_SCHEMES):
            continue
        if not exists(str(path)):
            missing.append(str(path))
    return missing


def build_material(
    isaac: Any,
    *,
    name: str,
    spec: Mapping[str, Any],
    exists: Callable[[str], bool] = os.path.exists,
) -> Any | None:
    """Author one OmniPBR material for the prop ``name`` (sim thread only) and
    return the ``OmniPBR`` object, or ``None`` when the prop should fall back
    to its flat colour. Best-effort: nothing here raises. Contract:

    1. ``missing_map_paths(spec, exists)`` non-empty -> ``LOGGER.warning``
       naming the prop and every missing path, return ``None`` (a bogus
       texture path renders the flat colour, never a magenta or black prim).
    2. ``isaac.OmniPBR`` is ``None`` (no ``isaacsim.core.api.materials``) ->
       one warning, return ``None``.
    3. ``material = isaac.OmniPBR(prim_path=material_prim_path(name),
       name=f"{name}_material")``, with no ``texture_path`` and no ``color``
       kwarg, since every input is authored in step 4.
    4. ``shader = material.shaders_list[0]``; for each ``(input_name, value)``
       of ``material_inputs(spec)``: ``shader.CreateInput(input_name,
       getattr(Sdf.ValueTypeNames, OMNIPBR_INPUT_SDF_TYPES[input_name])).Set(
       converted)`` where ``Asset`` inputs take the path string, ``Color3f``
       takes ``Gf.Vec3f(*value)``, ``Float2`` takes ``Gf.Vec2f(*value)`` and
       ``Float`` takes the float. ``pxr`` is imported lazily inside this
       function (it only exists inside Kit).
    5. Any exception -> ``LOGGER.exception`` naming the prop, return ``None``.

    The caller passes the object as ``visual_material=`` to the cuboid, or
    ``apply_visual_material`` on the ground plane object. Isaac Sim 5.0.0
    API, pinned once the GPU checklist confirms it:
    ``isaacsim.core.api.materials.OmniPBR(prim_path, name, shader=None,
    texture_path=None, texture_scale=None, texture_translate=None,
    color=None)`` with ``.shaders_list -> list[UsdShade.Shader]``."""
    missing = missing_map_paths(spec, exists)
    if missing:
        LOGGER.warning("prop %s: material maps missing on disk: %s", name, missing)
        return None
    if isaac.OmniPBR is None:
        LOGGER.warning("prop %s: isaacsim.core.api.materials.OmniPBR unavailable", name)
        return None
    try:
        from pxr import Gf, Sdf

        material = isaac.OmniPBR(prim_path=material_prim_path(name), name=f"{name}_material")
        shader = material.shaders_list[0]
        for input_name, value in material_inputs(spec).items():
            type_name = OMNIPBR_INPUT_SDF_TYPES[input_name]
            sdf_type = getattr(Sdf.ValueTypeNames, type_name)
            converted: Any
            if type_name == "Asset":
                converted = value
            elif type_name == "Color3f":
                converted = Gf.Vec3f(*value)
            elif type_name == "Float2":
                converted = Gf.Vec2f(*value)
            else:
                converted = float(value)
            shader.CreateInput(input_name, sdf_type).Set(converted)
        return material
    except Exception:
        LOGGER.exception("prop %s: failed to build material", name)
        return None
