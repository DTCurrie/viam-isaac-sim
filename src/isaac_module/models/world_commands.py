"""do_command verb implementations for the world component."""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from viam.utils import ValueTypes

from .. import component_diagnostics
from ..length_units import MM_PER_M
from ..prim_paths import default_base_prim_path, default_ee_prim_path
from ..sim_manager import ArmHandle, BaseHandle, GripperHandle, SimManager, WorldHandle
from ..spatial import Quat, quat_from_euler_deg, quat_to_ov, to_vec3
from .world_config_validation import (
    _require_name,
    _validate_size_range_mm,
    _validate_size_range_pair,
    validate_props,
)

if TYPE_CHECKING:
    from .world import IsaacWorld

DEFAULT_MIN_SEPARATION_MM = 150.0


def _orientation_wxyz_from_rpy_deg(rpy: Sequence[float] | None) -> Quat | None:
    if rpy is None:
        return None
    roll, pitch, yaw = (float(v) for v in rpy)
    return quat_from_euler_deg(roll, pitch, yaw)


def _pose_mm_from_m(position_m: Sequence[float], orientation_wxyz: Quat) -> dict[str, float]:
    x, y, z = position_m
    ox, oy, oz, theta = quat_to_ov(orientation_wxyz)
    return {
        "x": x * MM_PER_M,
        "y": y * MM_PER_M,
        "z": z * MM_PER_M,
        "o_x": ox,
        "o_y": oy,
        "o_z": oz,
        "theta": math.degrees(theta),
    }


def _cmd_status(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    return handle.status()


def _cmd_play(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    handle.play()
    return {"ok": True}


def _cmd_pause(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    handle.pause()
    return {"ok": True}


def _cmd_reset(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    handle.reset(soft=bool(command.get("soft", False)))
    return {"ok": True}


def _cmd_add_usd(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    usd_path = str(command.get("usd_path", ""))
    prim_path = str(command.get("prim_path", ""))
    if not usd_path or not prim_path:
        raise ValueError("add_usd requires usd_path and prim_path")
    position = cast("Sequence[float]", command.get("position") or [0.0, 0.0, 0.0])
    orientation_wxyz = _orientation_wxyz_from_rpy_deg(
        cast("Sequence[float] | None", command.get("orientation_rpy_deg"))
    )
    handle.add_usd(usd_path, prim_path, to_vec3(position), orientation_wxyz=orientation_wxyz)
    return {"ok": True}


def _cmd_prop_geometries(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    geometries: list[ValueTypes] = []
    for prop in handle.prop_geometries():
        pose_mm = _pose_mm_from_m(prop.position_m, prop.orientation_wxyz)
        geometries.append(
            {
                "name": prop.name,
                "box_dims_mm": [d * MM_PER_M for d in prop.box_dims_m],
                "pose_in_world_mm": pose_mm,
                "color": list(prop.color) if prop.color is not None else None,
                "fixed": prop.fixed,
            }
        )
    return {"geometries": geometries}


def _cmd_spawn_prop(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    prop_config = command.get("prop")
    if not isinstance(prop_config, Mapping):
        raise ValueError("spawn_prop requires a 'prop' object")
    prop_attrs = dict(prop_config)
    validate_props([prop_attrs])
    handle.spawn_prop(prop_attrs)
    return {"ok": True}


def _cmd_set_prop_pose(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    name = str(command.get("name", ""))
    if not name:
        raise ValueError("set_prop_pose requires 'name'")
    position_mm = cast("Sequence[float]", command.get("position"))
    if position_mm is None:
        raise ValueError("set_prop_pose requires 'position'")
    position_m = tuple(float(v) / MM_PER_M for v in position_mm)
    orientation_wxyz = _orientation_wxyz_from_rpy_deg(
        cast("Sequence[float] | None", command.get("orientation_rpy_deg"))
    )
    handle.set_prop_pose(name, cast("Any", position_m), orientation_wxyz=orientation_wxyz)
    return {"ok": True}


def _cmd_randomize_props(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    names = [str(n) for n in cast("Sequence[str]", command.get("names") or [])]
    region_mm = cast("Sequence[Sequence[float]]", command.get("region"))
    if not names or region_mm is None:
        raise ValueError("randomize_props requires 'names' and 'region'")
    (x0, y0, z0), (x1, y1, z1) = region_mm
    region_m = (
        (x0 / MM_PER_M, y0 / MM_PER_M, z0 / MM_PER_M),
        (x1 / MM_PER_M, y1 / MM_PER_M, z1 / MM_PER_M),
    )
    seed = int(cast("Any", command.get("seed", 0)))
    min_separation_mm = float(cast("Any", command.get("min_separation", DEFAULT_MIN_SEPARATION_MM)))
    size_range_mm = command.get("size_range_mm")
    size_range_m: dict[str, tuple[float, float]] | None = None
    if size_range_mm is not None:
        ranges_mm = _validate_size_range_mm(names, size_range_mm)
        size_range_m = {
            name: (lo / MM_PER_M, hi / MM_PER_M) for name, (lo, hi) in ranges_mm.items()
        }
    result = handle.randomize_props(
        names,
        cast("Any", region_m),
        seed,
        min_separation_m=min_separation_mm / MM_PER_M,
        size_range_m=size_range_m,
    )
    positions_mm: dict[str, ValueTypes] = {
        name: [v * MM_PER_M for v in position] for name, position in result.positions_m.items()
    }
    sizes_mm: dict[str, ValueTypes] = {
        name: [v * MM_PER_M for v in dims] for name, dims in result.dims_m.items()
    }
    return {"positions": positions_mm, "sizes_mm": sizes_mm}


def _cmd_ignore_props(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    names = [str(n) for n in cast("Sequence[str]", command.get("names") or [])]
    world._ignored_props = set(names)
    ignored_names = cast("list[ValueTypes]", sorted(world._ignored_props))
    return {"ignored": ignored_names}


def _parse_names_by_color(command: Mapping[str, ValueTypes], verb: str) -> dict[str, list[str]]:
    raw = command.get("names_by_color")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{verb} requires 'names_by_color'")
    names_by_color: dict[str, list[str]] = {}
    for color, names in raw.items():
        if not isinstance(names, Sequence) or isinstance(names, str) or not names:
            raise ValueError(f"{verb}: names_by_color[{color!r}] must be a non-empty list")
        names_by_color[str(color)] = [str(n) for n in names]
    return names_by_color


def _parse_park_positions_mm(
    command: Mapping[str, ValueTypes], verb: str
) -> dict[str, tuple[float, float]]:
    raw = command.get("park_positions_mm")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{verb} requires 'park_positions_mm'")
    park_positions_mm: dict[str, tuple[float, float]] = {}
    for name, xy in raw.items():
        if not isinstance(xy, Sequence) or isinstance(xy, str) or len(xy) != 2:
            raise ValueError(f"{verb}: park_positions_mm[{name!r}] must be [x, y]")
        x, y = xy
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise ValueError(f"{verb}: park_positions_mm[{name!r}] must be [x, y] numbers")
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            raise ValueError(f"{verb}: park_positions_mm[{name!r}] must be [x, y] numbers")
        park_positions_mm[str(name)] = (float(x), float(y))
    return park_positions_mm


def _cmd_scatter_cell(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    seed_value = command.get("seed")
    if not isinstance(seed_value, (int, float)) or isinstance(seed_value, bool):
        raise ValueError("scatter_cell requires 'seed'")
    seed = int(seed_value)
    names_by_color = _parse_names_by_color(command, "scatter_cell")
    region_mm = cast("Sequence[Sequence[float]]", command.get("region"))
    if region_mm is None:
        raise ValueError("scatter_cell requires 'region'")
    (x0, y0, z0), (x1, y1, z1) = region_mm
    region_m = (
        (x0 / MM_PER_M, y0 / MM_PER_M, z0 / MM_PER_M),
        (x1 / MM_PER_M, y1 / MM_PER_M, z1 / MM_PER_M),
    )
    park_positions_mm = _parse_park_positions_mm(command, "scatter_cell")
    park_positions_m = {
        name: (x / MM_PER_M, y / MM_PER_M) for name, (x, y) in park_positions_mm.items()
    }
    scatter_range_mm = command.get("size_range_mm")
    scatter_range_m: tuple[float, float] | None = None
    if scatter_range_mm is not None:
        lo_mm, hi_mm = _validate_size_range_pair(scatter_range_mm)
        scatter_range_m = (lo_mm / MM_PER_M, hi_mm / MM_PER_M)
    counts = command.get("counts")
    counts_arg: dict[str, int] | None = None
    if counts is not None:
        if not isinstance(counts, Mapping):
            raise ValueError("scatter_cell: counts must be an object")
        counts_arg = {str(color): int(cast("Any", value)) for color, value in counts.items()}
    scatter = handle.scatter_cell(
        names_by_color,
        cast("Any", region_m),
        park_positions_m,
        seed,
        size_range_m=scatter_range_m,
        counts=counts_arg,
    )
    scatter_positions_mm: dict[str, ValueTypes] = {
        name: [v * MM_PER_M for v in position] for name, position in scatter.positions_m.items()
    }
    scatter_sizes_mm: dict[str, ValueTypes] = {
        name: [v * MM_PER_M for v in dims] for name, dims in scatter.sizes_m.items()
    }
    return {
        "seed": scatter.seed,
        "counts": cast("Any", scatter.counts),
        "positions": scatter_positions_mm,
        "sizes_mm": scatter_sizes_mm,
        "parked": cast("Any", scatter.parked),
    }


def _cmd_clear_cell(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    names_by_color = _parse_names_by_color(command, "clear_cell")
    park_positions_mm = _parse_park_positions_mm(command, "clear_cell")
    park_positions_m = {
        name: (x / MM_PER_M, y / MM_PER_M) for name, (x, y) in park_positions_mm.items()
    }
    cleared = handle.clear_cell(names_by_color, park_positions_m)
    return {"parked": cast("Any", cleared.parked)}


def _cmd_joint_state(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    name = _require_name(command)
    attrs, entry_handle = SimManager.get().handle_entry(name)
    if not isinstance(entry_handle, ArmHandle):
        raise ValueError(f"{name!r} is a {type(entry_handle).__name__}, not an arm")
    return cast("Any", component_diagnostics.joint_state(attrs, entry_handle))


def _cmd_dof_names(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    name = _require_name(command)
    attrs, entry_handle = SimManager.get().handle_entry(name)
    all_dofs = bool(command.get("all", False))
    if all_dofs and not isinstance(entry_handle, ArmHandle):
        raise ValueError(f"{name!r} is a {type(entry_handle).__name__}, not an arm")
    if not isinstance(entry_handle, (ArmHandle, GripperHandle)):
        raise ValueError(f"{name!r} is a {type(entry_handle).__name__}, not an arm or gripper")
    return cast("Any", component_diagnostics.dof_names(attrs, entry_handle, all_dofs=all_dofs))


def _cmd_prim_pose(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    name = _require_name(command)
    attrs, entry_handle = SimManager.get().handle_entry(name)
    if not isinstance(entry_handle, (ArmHandle, BaseHandle)):
        raise ValueError(f"{name!r} is a {type(entry_handle).__name__}, not an arm or base")
    prim_path_value = command.get("prim_path")
    if prim_path_value is None:
        if isinstance(entry_handle, ArmHandle):
            prim_path_value = default_ee_prim_path(attrs, name)
        else:
            prim_path_value = default_base_prim_path(attrs, name)
    if not isinstance(prim_path_value, str):
        raise ValueError(f"prim_path must be a string, got {prim_path_value!r}")
    return cast(
        "Any", component_diagnostics.prim_pose(attrs, entry_handle, prim_path_value.strip())
    )


def _cmd_tcp_pose(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    name = _require_name(command)
    attrs, entry_handle = SimManager.get().handle_entry(name)
    if not isinstance(entry_handle, GripperHandle):
        raise ValueError(f"{name!r} is a {type(entry_handle).__name__}, not a gripper")
    return cast("Any", component_diagnostics.tcp_pose(attrs, entry_handle))


def _cmd_jaw_deg(
    world: "IsaacWorld", handle: WorldHandle, command: Mapping[str, ValueTypes]
) -> dict[str, ValueTypes]:
    name = _require_name(command)
    attrs, entry_handle = SimManager.get().handle_entry(name)
    if not isinstance(entry_handle, GripperHandle):
        raise ValueError(f"{name!r} is a {type(entry_handle).__name__}, not a gripper")
    return cast("Any", component_diagnostics.jaw_deg(attrs, entry_handle))


COMMAND_HANDLERS: dict[
    str, Callable[["IsaacWorld", WorldHandle, Mapping[str, ValueTypes]], dict[str, ValueTypes]]
] = {
    "status": _cmd_status,
    "play": _cmd_play,
    "pause": _cmd_pause,
    "reset": _cmd_reset,
    "add_usd": _cmd_add_usd,
    "prop_geometries": _cmd_prop_geometries,
    "spawn_prop": _cmd_spawn_prop,
    "set_prop_pose": _cmd_set_prop_pose,
    "randomize_props": _cmd_randomize_props,
    "ignore_props": _cmd_ignore_props,
    "scatter_cell": _cmd_scatter_cell,
    "clear_cell": _cmd_clear_cell,
    "joint_state": _cmd_joint_state,
    "dof_names": _cmd_dof_names,
    "prim_pose": _cmd_prim_pose,
    "tcp_pose": _cmd_tcp_pose,
    "jaw_deg": _cmd_jaw_deg,
}
