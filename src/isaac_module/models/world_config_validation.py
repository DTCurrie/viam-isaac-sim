"""Config-attribute validators for the world component."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

from viam.proto.app.robot import ComponentConfig
from viam.utils import ValueTypes

from ..materials import material_spec
from ..physics import PROP_PHYSICS_KEYS
from ..sim_manager import prim_name
from ..visual_props import FIT_TRUE, PROP_KINDS, VISUAL_PROP_KIND, VISUAL_REJECTED_KEYS
from .component_frame_pose import frame_pose

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def validate_world_frame(config: ComponentConfig) -> None:
    """The world component's own get_geometries reports prop and floor poses
    in world coordinates, not relative to the world's own frame, so a
    non-identity frame.translation or frame.orientation would silently shift
    every reported geometry. Reject it instead."""
    if not config.HasField("frame"):
        return
    position, quat = frame_pose(config)
    translation_at_origin = position is None or all(abs(v) < 1e-9 for v in position)
    orientation_is_identity = quat is None or all(
        abs(a - b) < 1e-9 for a, b in zip(quat, _IDENTITY_QUAT, strict=True)
    )
    if not (translation_at_origin and orientation_is_identity):
        raise ValueError(
            f"{config.name}: frame.translation and frame.orientation must stay at the "
            "origin (zero translation, identity orientation); the world component's "
            "geometries (get_geometries) are reported in world coordinates, not relative "
            "to the world's own frame"
        )


# the ground plane the module adds when no usd_stage is configured, served by
# get_geometries so motion plans keep the arm out of the floor. It
# is thick, so discrete collision checks cannot step through it.
FLOOR_LABEL = "floor"
FLOOR_SIDE_MM = 10000.0
FLOOR_THICKNESS_MM = 200.0


def _require_name(command: Mapping[str, ValueTypes]) -> str:
    name = command.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("this command requires a 'name'")
    return name


def _prop_label(prop: object, index: int) -> str:
    if isinstance(prop, Mapping) and prop.get("name"):
        return str(prop["name"])
    return f"props[{index}]"


def _validate_number_triple(prop_label: str, key: str, value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise ValueError(f"prop {prop_label}: {key!r} must be a list of 3 numbers")
    for v in value:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"prop {prop_label}: {key!r} must be a list of 3 numbers")


def _validate_orientation(label: str, prop: Mapping[str, object]) -> None:
    has_rpy = "orientation_rpy_deg" in prop
    has_wxyz = "orientation_wxyz" in prop
    if has_rpy and has_wxyz:
        raise ValueError(
            f"prop {label}: only one of 'orientation_rpy_deg' or 'orientation_wxyz' may be set"
        )
    if has_rpy:
        _validate_number_triple(label, "orientation_rpy_deg", prop["orientation_rpy_deg"])
    if has_wxyz:
        value = prop["orientation_wxyz"]
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 4:
            raise ValueError(f"prop {label}: 'orientation_wxyz' must be a list of 4 numbers")
        for v in value:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"prop {label}: 'orientation_wxyz' must be a list of 4 numbers")
        if all(float(v) == 0.0 for v in value):
            raise ValueError(f"prop {label}: 'orientation_wxyz' must not be all zero")


def _validate_size_range_pair(range_value: object) -> tuple[float, float]:
    if (
        not isinstance(range_value, Sequence)
        or isinstance(range_value, str)
        or len(range_value) != 2
    ):
        raise ValueError("randomize_props: size_range_mm entries must be [lo, hi]")
    lo, hi = range_value
    if not isinstance(lo, (int, float)) or isinstance(lo, bool):
        raise ValueError("randomize_props: size_range_mm entries must be [lo, hi] numbers")
    if not isinstance(hi, (int, float)) or isinstance(hi, bool):
        raise ValueError("randomize_props: size_range_mm entries must be [lo, hi] numbers")
    lo_f, hi_f = float(lo), float(hi)
    if not (0.0 < lo_f <= hi_f):
        raise ValueError(
            f"randomize_props: size_range_mm [{lo_f}, {hi_f}] must satisfy 0 < lo <= hi"
        )
    return lo_f, hi_f


def _validate_size_range_mm(names: list[str], value: object) -> dict[str, tuple[float, float]]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(names)
        if unknown:
            raise ValueError(
                f"randomize_props: size_range_mm names not in names: {sorted(unknown)}"
            )
        return {
            str(name): _validate_size_range_pair(range_value) for name, range_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        pair = _validate_size_range_pair(value)
        return dict.fromkeys(names, pair)
    raise ValueError("randomize_props: size_range_mm must be [lo, hi] or {name: [lo, hi]}")


def _validate_box_dims(label: str, prop: Mapping[str, object]) -> None:
    if "box_dims" not in prop:
        return
    dims = cast("Sequence[float]", prop["box_dims"])
    _validate_number_triple(label, "box_dims", dims)
    for v in dims:
        is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
        if is_number and v <= 0:
            raise ValueError(f"prop {label}: 'box_dims' values must be positive")


def _cube_prop_names(props: Sequence[object]) -> set[str]:
    """The sanitised prim names of every ``type: "cube"`` (or default) prop,
    the only kind ``fit.collider`` may name."""
    names: set[str] = set()
    for prop in props:
        if not isinstance(prop, Mapping) or prop.get("type", "cube") != "cube":
            continue
        name = prop.get("name")
        if isinstance(name, str) and name:
            names.add(prim_name(name))
    return names


def _validate_fit(label: str, fit: object, cube_names: set[str]) -> None:
    if fit == FIT_TRUE:
        return
    if not isinstance(fit, Mapping) or set(fit) != {"collider"}:
        raise ValueError(
            f"prop {label}: 'fit' must be the string \"true\" or a mapping of the form "
            '{"collider": "<cube prop name>"}'
        )
    collider = fit["collider"]
    if not (isinstance(collider, str) and collider):
        raise ValueError(
            f"prop {label}: 'fit' must be the string \"true\" or a mapping of the form "
            '{"collider": "<cube prop name>"}'
        )
    if prim_name(collider) in cube_names:
        return
    raise ValueError(
        f"prop {label}: fit.collider {collider!r} is not a cube prop in this props "
        f"list (cube props: {sorted(cube_names)})"
    )


def validate_props(props: object) -> None:
    if not isinstance(props, Sequence) or isinstance(props, str):
        raise ValueError("props must be a list")
    seen_prim_names: set[str] = set()
    cube_names = _cube_prop_names(props)
    for index, prop in enumerate(props):
        label = _prop_label(prop, index)
        if not isinstance(prop, Mapping):
            raise ValueError(f"prop {label}: must be an object")
        name = prop.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"prop {label}: 'name' must be a non-empty string")
        sanitized_name = prim_name(name)
        if sanitized_name in seen_prim_names:
            raise ValueError(
                f"prop {label}: 'name' collides with another prop after sanitizing "
                f"to a USD prim name ({sanitized_name!r})"
            )
        seen_prim_names.add(sanitized_name)

        kind = prop.get("type", "cube")
        if kind not in PROP_KINDS:
            raise ValueError(
                f'prop {label}: \'type\' must be "cube", "usd", or "visual" (got {kind!r})'
            )
        if kind == "usd" and not prop.get("usd_path"):
            raise ValueError(f"prop {label}: 'usd_path' is required when 'type' is \"usd\"")
        if kind == VISUAL_PROP_KIND:
            usd_path = prop.get("usd_path")
            if usd_path is None:
                raise ValueError(f"prop {label}: 'usd_path' is required when 'type' is \"visual\"")
            if not isinstance(usd_path, str):
                raise ValueError(
                    f"prop {label}: 'usd_path' must be a string on a \"visual\" prop "
                    "(an empty string skips the prop)"
                )
            for key in prop:
                if key in VISUAL_REJECTED_KEYS:
                    raise ValueError(
                        f'prop {label}: {key!r} is not allowed on a "visual" prop (it has no '
                        "collider or rigid body)"
                    )

        if "fit" in prop and kind != VISUAL_PROP_KIND:
            raise ValueError(f"prop {label}: 'fit' is only valid on a \"visual\" prop")
        if kind == VISUAL_PROP_KIND:
            if "scale" in prop and "fit" in prop:
                raise ValueError(f"prop {label}: 'scale' and 'fit' cannot both be set")
            if "fit" in prop:
                _validate_fit(label, prop["fit"], cube_names)

        if "position" in prop:
            _validate_number_triple(label, "position", prop["position"])
        if "scale" in prop:
            _validate_number_triple(label, "scale", prop["scale"])
        if "color" in prop:
            color = prop["color"]
            _validate_number_triple(label, "color", color)
            if isinstance(color, Sequence) and not isinstance(color, str):
                for v in color:
                    is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
                    if is_number and not (0 <= v <= 1):
                        raise ValueError(f"prop {label}: 'color' values must be in [0, 1]")
        if "size" in prop:
            size = prop["size"]
            if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"prop {label}: 'size' must be a positive number")
        if "fixed" in prop and not isinstance(prop["fixed"], bool):
            raise ValueError(f"prop {label}: 'fixed' must be a bool")
        if "material" in prop:
            if kind != "cube":
                raise ValueError(f"prop {label}: 'material' is only valid on a \"cube\" prop")
            _validate_material(f"prop {label}", prop["material"], prop.get("color"))

        _validate_orientation(label, prop)
        _validate_box_dims(label, prop)
        _validate_prop_physics(label, prop)


def _validate_material(prefix: str, material: object, color: object) -> None:
    """``material_spec`` as a validator: its ``ValueError`` re-raised with the
    prop or ground label in front, so the message says where the key sits."""
    if not isinstance(material, (str, Mapping)):
        raise ValueError(f"{prefix}: material must be a set name or an object (got {material!r})")
    try:
        material_spec(material, color=color)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ValueError(f"{prefix}: {exc}") from exc


def _validate_number(label: str, key: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"prop {label}: {key!r} must be a number")
    return float(value)


def _validate_prop_physics(label: str, prop: Mapping[str, object]) -> None:
    values: dict[str, float] = {}
    for key in PROP_PHYSICS_KEYS:
        if key in prop:
            values[key] = _validate_number(label, key, prop[key])

    if "mass" in values and values["mass"] <= 0:
        raise ValueError(f"prop {label}: 'mass' must be positive")
    if "friction" in values and values["friction"] < 0:
        raise ValueError(f"prop {label}: 'friction' must be >= 0")
    if "restitution" in values and not (0 <= values["restitution"] <= 1):
        raise ValueError(f"prop {label}: 'restitution' must be in [0, 1]")
    if "contact_offset" in values and values["contact_offset"] < 0:
        raise ValueError(f"prop {label}: 'contact_offset' must be >= 0")
    if "rest_offset" in values and values["rest_offset"] < 0:
        raise ValueError(f"prop {label}: 'rest_offset' must be >= 0")
    if "rest_offset" in values and "contact_offset" in values:
        if values["rest_offset"] > values["contact_offset"]:
            raise ValueError(
                f"prop {label}: 'rest_offset' must not be greater than 'contact_offset'"
            )


_LIGHTING_KEYS = {"dome", "sphere_intensity"}
# lighting.dome: intensity and color as before; texture is a path or URL the
# resolver accepts or a module:// or data:// scheme (isaac_module/assets.py);
# texture_format is a UsdLux dome format, default "latlong"; rotation_deg is
# the yaw about Z. Kit orients the dome's pole to the Z-up stage on its own.
_DOME_KEYS = {"intensity", "color", "texture", "texture_format", "rotation_deg"}
DOME_TEXTURE_FORMATS = {"automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"}
DEFAULT_DOME_TEXTURE_FORMAT = "latlong"

# ground: the floor the module adds when it owns the stage (no usd_stage).
# "grid" is today's default environment, "plane" a plain ground plane with the
# plane-only keys below, "none" no floor at all.
# material: a PBR material on the plane, `materials.material_spec`'s shape,
# plane-only like the other look keys.
GROUND_KINDS = ("grid", "plane", "none")
_GROUND_KEYS = {"kind", "color", "size", "friction", "restitution", "matte", "material"}
# matte: the plane is invisible to the camera but catches shadows, so the
# dome texture's own floor shows through (RTX "Matte Object" post-process).
_GROUND_PLANE_ONLY_KEYS = {"color", "size", "friction", "restitution", "matte", "material"}
GROUND_DEFAULTS: dict[str, object] = {
    "kind": "grid",
    "color": [0.5, 0.5, 0.5],
    "size": 100.0,
    "friction": 0.5,
    "restitution": 0.0,
    "matte": False,
}


def validate_ground(value: object) -> None:
    """Object with keys in _GROUND_KEYS; kind in GROUND_KINDS; color a triple in
    [0, 1]; size > 0; friction >= 0; restitution in [0, 1]; a plane-only key
    with kind != "plane" is an error."""
    if not isinstance(value, Mapping):
        raise ValueError("ground must be an object")
    for key in value:
        if key not in _GROUND_KEYS:
            raise ValueError(f"ground: unknown key {key!r}")

    kind = value.get("kind")
    if kind is not None and kind not in GROUND_KINDS:
        raise ValueError(f"ground.kind must be one of {GROUND_KINDS} (got {kind!r})")

    if "color" in value:
        _validate_number_triple("ground", "color", value["color"])
        for v in value["color"]:
            is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
            if is_number and not (0 <= v <= 1):
                raise ValueError("ground.color values must be in [0, 1]")

    if "size" in value:
        size = value["size"]
        if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
            raise ValueError("ground.size must be a positive number")

    if "friction" in value:
        friction = value["friction"]
        if not isinstance(friction, (int, float)) or isinstance(friction, bool) or friction < 0:
            raise ValueError("ground.friction must be a number >= 0")

    if "restitution" in value:
        restitution = value["restitution"]
        is_number = isinstance(restitution, (int, float)) and not isinstance(restitution, bool)
        if not is_number or not (0 <= restitution <= 1):
            raise ValueError("ground.restitution must be a number in [0, 1]")

    if "matte" in value:
        if not isinstance(value["matte"], bool):
            raise ValueError("ground.matte must be a bool")

    if "material" in value:
        _validate_material("ground", value["material"], value.get("color"))

    effective_kind = kind if kind is not None else GROUND_DEFAULTS["kind"]
    if effective_kind != "plane":
        for key in _GROUND_PLANE_ONLY_KEYS:
            if key in value:
                raise ValueError(
                    f'ground.{key} is only valid when ground.kind is "plane" '
                    f"(got kind {effective_kind!r})"
                )


def validate_lighting(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("lighting must be an object")
    for key in value:
        if key not in _LIGHTING_KEYS:
            raise ValueError(f"lighting: unknown key {key!r}")

    dome = value.get("dome")
    if dome is not None:
        if not isinstance(dome, Mapping):
            raise ValueError("lighting.dome must be an object")
        for key in dome:
            if key not in _DOME_KEYS:
                raise ValueError(f"lighting.dome: unknown key {key!r}")
        if "intensity" in dome:
            intensity = dome["intensity"]
            is_number = isinstance(intensity, (int, float)) and not isinstance(intensity, bool)
            if not is_number or intensity <= 0:
                raise ValueError("lighting.dome.intensity must be a positive number")
        if "color" in dome:
            _validate_number_triple("lighting.dome", "color", dome["color"])
            for v in dome["color"]:
                is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
                if is_number and not (0 <= v <= 1):
                    raise ValueError("lighting.dome.color values must be in [0, 1]")
        if "texture" in dome:
            texture = dome["texture"]
            if not isinstance(texture, str) or not texture:
                raise ValueError("lighting.dome.texture must be a non-empty string")
        if "texture_format" in dome:
            texture_format = dome["texture_format"]
            if not isinstance(texture_format, str) or texture_format not in DOME_TEXTURE_FORMATS:
                raise ValueError(
                    "lighting.dome.texture_format must be one of "
                    f"{sorted(DOME_TEXTURE_FORMATS)} (got {texture_format!r})"
                )
        if "rotation_deg" in dome:
            rotation_deg = dome["rotation_deg"]
            is_number = isinstance(rotation_deg, (int, float)) and not isinstance(
                rotation_deg, bool
            )
            if not is_number or not math.isfinite(rotation_deg):
                raise ValueError("lighting.dome.rotation_deg must be a finite number")

    sphere_intensity = value.get("sphere_intensity")
    if sphere_intensity is not None:
        is_number = isinstance(sphere_intensity, (int, float)) and not isinstance(
            sphere_intensity, bool
        )
        if not is_number or sphere_intensity < 0:
            raise ValueError("lighting.sphere_intensity must be a non-negative number")


# the values sim_manager.py's kit_log_level.capitalize() forwards to Kit's
# "/log/outputStreamLevel" via extra_args (see _boot_extra_args)
_KIT_LOG_LEVELS = {"verbose", "info", "warning", "error"}


def validate_kit_log_level(value: object) -> None:
    if not isinstance(value, str) or value.lower() not in _KIT_LOG_LEVELS:
        raise ValueError(f"kit_log_level must be one of {sorted(_KIT_LOG_LEVELS)} (got {value!r})")


# viewport_grid is the first post-launch lever: written through carb.settings
# after Kit is up (best-effort), unlike the two launcher-time keys.
_RENDER_KEYS = {"motion_bvh", "disable_viewport_updates", "viewport_grid"}


def validate_render(value: object, livestream: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("render must be an object")
    for key in value:
        if key not in _RENDER_KEYS:
            raise ValueError(f"render: unknown key {key!r}")
        if not isinstance(value[key], bool):
            raise ValueError(f"render.{key} must be a bool")

    if value.get("disable_viewport_updates") and livestream:
        raise ValueError(
            "render.disable_viewport_updates cannot be true while livestream is true "
            "(the livestream needs viewport updates)"
        )
