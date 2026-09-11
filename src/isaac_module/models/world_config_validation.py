"""Config-attribute validators for the world component."""

from collections.abc import Mapping, Sequence
from typing import cast

from viam.proto.app.robot import ComponentConfig
from viam.utils import ValueTypes

from ..physics import PROP_PHYSICS_KEYS
from ..sim_manager import prim_name
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


def validate_props(props: object) -> None:
    if not isinstance(props, Sequence) or isinstance(props, str):
        raise ValueError("props must be a list")
    seen_prim_names: set[str] = set()
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
        if kind not in ("cube", "usd"):
            raise ValueError(f'prop {label}: \'type\' must be "cube" or "usd" (got {kind!r})')
        if kind == "usd" and not prop.get("usd_path"):
            raise ValueError(f"prop {label}: 'usd_path' is required when 'type' is \"usd\"")

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

        _validate_orientation(label, prop)
        _validate_box_dims(label, prop)
        _validate_prop_physics(label, prop)


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


_RENDER_KEYS = {"motion_bvh", "disable_viewport_updates"}


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
