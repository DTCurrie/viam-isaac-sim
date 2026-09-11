"""Resolve a real machine's config into the config of its sim machine.

The real config is never edited. Every hardware component whose model has a
row in `simulates.json` is rewritten to the row's sim model, with `world`
set to the table's default world, the row's template applied, its `$`
references resolved against the real entry, and its `carry` map copied. The
sim module entry and the world component come from the public world
fragment when the config lacks them. `name`, `api`, `frame`, `depends_on`
and every service are left byte for byte.

Three rules live here rather than in the table, because they read more than
one entry. A component already on one of this module's models, or already
on `rdk:builtin:fake`, is passed through untouched, which makes resolving
twice a no-op. A swapped camera
whose frame parent is a swapped arm gets `parent_prim` set to that arm's
default end-effector prim, since the sim camera needs the prim to ride and
the real entry carries only the frame. A swapped arm, camera, base or
gripper whose frame parent is neither `world`, the world component, nor
(for a camera or gripper) a swapped arm, fails here instead of resolving to
a config that only fails once it reaches the GPU host, since this module
does not build the parent frame chain.

This is the reference behavior for an app button or a viam-server flag that
would do the same thing at save or construction time.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from . import FAMILY, NAMESPACE
from .prim_paths import default_ee_prim_path

# Component APIs that drive hardware and so must be simulated or explicitly
# passed through. `rdk:component:generic` is not listed: the world itself is
# generic and a generic component has no hardware to stand in for. Subtype
# names checked against the installed SDK's package layout, not the docs
# page, for audio: .venv/lib/python*/site-packages/viam/components/ lists
# `audio_in` and `audio_out`, not `audio_input`.
# Source: https://docs.viam.com/dev/reference/apis/, read 2026-09-09, audio
# subtypes cross-checked against the installed viam-sdk the same day.
HARDWARE_APIS: frozenset[str] = frozenset(
    f"rdk:component:{name}"
    for name in (
        "arm",
        "audio_in",
        "audio_out",
        "base",
        "board",
        "button",
        "camera",
        "encoder",
        "gantry",
        "gripper",
        "input_controller",
        "motor",
        "movement_sensor",
        "pose_tracker",
        "power_sensor",
        "sensor",
        "servo",
        "switch",
    )
)

WILDCARD_REAL_MODEL = "*"
CATCH_ALL_API = "*"
PLACEHOLDER_API = "rdk:component:generic"
FAKE_MODEL = "rdk:builtin:fake"
FRAME_PARENT_REFERENCE = "$frame.parent"

# The table's conventional filename, used only in the unmatched-hardware
# error since `resolve` receives a parsed table, not its path.
TABLE_PATH_HINT = "simulates.json"


class UnmatchedHardwareError(ValueError):
    """Hardware components with no row in the table, listed by name.

    Raised by `resolve` unless `allow_unmatched` is set, so a sim machine
    never silently keeps a driver that will fail to find its device.
    """

    def __init__(self, components: Sequence[str], table_path_hint: str) -> None:
        self.components = tuple(components)
        names = ", ".join(self.components)
        super().__init__(
            f"no simulates.json row for hardware components: {names} "
            f"(table: {table_path_hint}); pass allow_unmatched to keep them as they are"
        )


@dataclass(frozen=True)
class Resolution:
    """What `resolve` produced and which components it touched.

    `pruned_modules` lists the `module_id` of every entry dropped because no
    remaining component or service model needs it. `unresolved_variables`
    lists the `name` of every fragment `$variable` reference left in place
    because it carried no `default_value`.
    """

    config: dict[str, Any]
    swapped: tuple[str, ...]
    passed_through: tuple[str, ...]
    placeholders: tuple[str, ...] = ()
    pruned_modules: tuple[str, ...] = ()
    unresolved_variables: tuple[str, ...] = ()


def resolve(
    config: dict[str, Any],
    table: dict[str, Any],
    world_fragment: dict[str, Any],
    *,
    allow_unmatched: bool = False,
    only: Collection[str] | None = None,
) -> Resolution:
    """The sim machine's config for a real machine's config.

    `table` is the parsed `simulates.json`. `world_fragment` is the parsed
    `fragments/isaac-sim-world.json`, the source of the module entry and the
    world component inserted when absent. `only` limits the swap to the
    named components and passes every other one through, the per-resource
    door, and does not by itself make the output a valid remote setup. A
    hardware component with no row becomes a placeholder generic component
    through the table's catch-all row, or raises `UnmatchedHardwareError`
    when the table has no catch-all. `allow_unmatched` keeps such a
    component's real entry instead, and also skips the check that a swapped
    arm, camera, base or gripper's frame parent resolves to a sim prim.
    Never mutates `config`.
    """
    resolved = copy.deepcopy(config)
    rows: Sequence[dict[str, Any]] = table.get("rows", [])
    default_world = table["default_world"]
    world_attribute = table["world_attribute"]
    module_prefix = f"{NAMESPACE}:{FAMILY}:"

    original_components: list[dict[str, Any]] = resolved.get("components", [])
    swapped: list[str] = []
    passed_through: list[str] = []
    placeholders: list[str] = []
    unmatched: list[str] = []
    catch_all = _find_row(rows, CATCH_ALL_API, WILDCARD_REAL_MODEL)
    swapped_names: set[str] = set()
    new_components: list[dict[str, Any]] = []

    for component in original_components:
        name = component["name"]
        api = component.get("api")
        model = component.get("model", "")
        if model.startswith(module_prefix) or model == FAKE_MODEL:
            passed_through.append(name)
            new_components.append(component)
            continue
        if only is not None and name not in only:
            passed_through.append(name)
            new_components.append(component)
            continue
        row = _find_row(rows, api, model) or _find_row(rows, api, WILDCARD_REAL_MODEL)
        if row is None and api in HARDWARE_APIS and not allow_unmatched:
            if catch_all is None:
                unmatched.append(name)
            else:
                # The catch-all keeps the component's own api and points it at the
                # builtin fake model, which is registered for every hardware api but
                # pose_tracker. pose_tracker falls back to generic, the one api the
                # placeholder still has to change, so the config still loads.
                placeholder = dict(component)
                placeholder["api"] = PLACEHOLDER_API if api == "rdk:component:pose_tracker" else api
                placeholder["model"] = catch_all["sim_model"]
                placeholder["attributes"] = {}
                placeholders.append(name)
                new_components.append(placeholder)
                continue
        if row is None:
            passed_through.append(name)
            new_components.append(component)
            continue
        new_component = dict(component)
        new_component["model"] = row["sim_model"]
        new_component["attributes"] = _resolve_attributes(
            component, row, world_attribute, default_world
        )
        swapped.append(name)
        swapped_names.add(name)
        new_components.append(new_component)

    if unmatched and not allow_unmatched:
        raise UnmatchedHardwareError(unmatched, TABLE_PATH_HINT)

    by_name = {c["name"]: c for c in new_components}
    original_by_name = {c["name"]: c for c in original_components}
    _apply_camera_rules(new_components, by_name, original_by_name, swapped_names)
    if not allow_unmatched:
        _validate_frame_parents(new_components, by_name, swapped_names, default_world)

    resolved["components"], unresolved_variables = _with_world_component(
        new_components, world_fragment
    )
    resolved = _with_module_entry(resolved, world_fragment)
    resolved, pruned_modules = _prune_unused_modules(resolved)

    return Resolution(
        config=resolved,
        swapped=tuple(swapped),
        passed_through=tuple(passed_through),
        placeholders=tuple(placeholders),
        pruned_modules=tuple(pruned_modules),
        unresolved_variables=tuple(unresolved_variables),
    )


def _find_row(
    rows: Sequence[dict[str, Any]], api: str | None, real_model: str
) -> dict[str, Any] | None:
    for row in rows:
        if row["api"] == api and row["real_model"] == real_model:
            return row
    return None


def _resolve_attributes(
    component: dict[str, Any], row: dict[str, Any], world_attribute: str, default_world: str
) -> dict[str, Any]:
    real_attrs: dict[str, Any] = component.get("attributes") or {}
    new_attrs: dict[str, Any] = {world_attribute: default_world}
    for key, value in row.get("template", {}).items():
        if value == FRAME_PARENT_REFERENCE:
            frame = component.get("frame") or {}
            parent = frame.get("parent")
            if not parent:
                raise ValueError(
                    f"{component['name']}: template needs {FRAME_PARENT_REFERENCE} but the "
                    "component has no frame parent"
                )
            new_attrs[key] = parent.split(":")[0]
        else:
            new_attrs[key] = value
    for real_key, sim_key in row.get("carry", {}).items():
        if real_key in real_attrs:
            new_attrs[sim_key] = real_attrs[real_key]
    return new_attrs


def _apply_camera_rules(
    new_components: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    original_by_name: dict[str, dict[str, Any]],
    swapped_names: set[str],
) -> None:
    for component in new_components:
        name = component["name"]
        if component.get("api") != "rdk:component:camera" or name not in swapped_names:
            continue
        frame = component.get("frame") or {}
        parent_name = (frame.get("parent") or "").split(":")[0]
        parent = by_name.get(parent_name)
        if (
            parent is not None
            and parent.get("api") == "rdk:component:arm"
            and parent_name in swapped_names
        ):
            ee_prim_path = default_ee_prim_path(parent.get("attributes") or {}, parent_name)
            if ee_prim_path is not None:
                component["attributes"]["parent_prim"] = ee_prim_path
        real_attrs = original_by_name[name].get("attributes") or {}
        if "sensors" in real_attrs:
            sensors = real_attrs["sensors"]
            wants_depth = isinstance(sensors, list) and "depth" in sensors
        else:
            # The real RealSense driver defaults sensors to color plus depth when the
            # config omits the key, so an absent key resolves the same way.
            wants_depth = True
        if wants_depth:
            component["attributes"]["depth"] = True


# frame.parent values allowed to ride an arm without a spawned sim prim of their own,
# since the sim gripper and camera each get a mount point on the arm's own prim (the
# gripper through its "arm" attribute, the camera through _apply_camera_rules).
_RIDES_ARM_APIS = ("rdk:component:camera", "rdk:component:gripper")


def _validate_frame_parents(
    new_components: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    swapped_names: set[str],
    default_world: str,
) -> None:
    """A swapped arm, camera, base or gripper whose frame rides another
    component has no sim prim to land on: this module carries `frame`
    unchanged rather than building the parent frame chain (a plan of its
    own), so the swap would resolve to valid-looking JSON that only fails
    once it reaches the GPU host. Fail here instead, naming the remedy. A
    camera or gripper riding a swapped arm is the one exception, since each
    gets a mount point on that arm's own prim.
    """
    checked_apis = {
        "rdk:component:arm",
        "rdk:component:camera",
        "rdk:component:base",
        *_RIDES_ARM_APIS,
    }
    for component in new_components:
        name = component["name"]
        if name not in swapped_names or component.get("api") not in checked_apis:
            continue
        frame = component.get("frame") or {}
        parent = frame.get("parent") or ""
        if parent in ("", "world", default_world):
            continue
        parent_name = parent.split(":")[0]
        if component.get("api") in _RIDES_ARM_APIS:
            parent_component = by_name.get(parent_name)
            if (
                parent_component is not None
                and parent_component.get("api") == "rdk:component:arm"
                and parent_name in swapped_names
            ):
                continue
        raise ValueError(
            f'{name}: frame.parent {parent!r} names a component, not "world" or the '
            f"world component ({default_world!r}). This module carries frame "
            "unchanged rather than building the parent frame chain, so the resolved "
            f'config would fail to construct on the GPU host. Mount {name} on "world" '
            "for the sim, or pass allow_unmatched to resolve() to keep its real frame "
            "entry as it is"
        )


def _resolve_fragment_variables(node: Any, unresolved: list[str]) -> Any:
    """Replace each `{"$variable": {"name", "default_value"}}` object with its default.

    The app substitutes fragment variables when it renders a machine's config. A resolved
    config is a machine config, not a fragment, so the defaults are baked in here, and a
    caller that wants a different value sets the attribute on the emitted component. A
    variable with no `default_value` is left as its `$variable` object and its `name` is
    appended to `unresolved`, rather than raising, since the docs describe `default_value`
    narratively and do not guarantee every fragment sets it.
    """
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            if "default_value" not in variable:
                unresolved.append(variable.get("name", "<unnamed>"))
                return node
            return _resolve_fragment_variables(variable["default_value"], unresolved)
        return {key: _resolve_fragment_variables(value, unresolved) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_fragment_variables(item, unresolved) for item in node]
    return node


def _with_world_component(
    components: list[dict[str, Any]], world_fragment: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    existing_names = {c["name"] for c in components}
    unresolved: list[str] = []
    missing = [
        _resolve_fragment_variables(copy.deepcopy(c), unresolved)
        for c in world_fragment.get("components", [])
        if c["name"] not in existing_names
    ]
    return missing + components, unresolved


def _with_module_entry(config: dict[str, Any], world_fragment: dict[str, Any]) -> dict[str, Any]:
    fragment_modules: list[dict[str, Any]] = world_fragment.get("modules", [])
    existing_modules = config.get("modules")
    existing_ids = {m.get("module_id") for m in existing_modules} if existing_modules else set()
    missing = [copy.deepcopy(m) for m in fragment_modules if m.get("module_id") not in existing_ids]
    if not missing:
        return config
    if existing_modules is not None:
        config["modules"] = list(existing_modules) + missing
        return config
    new_config = {"modules": missing}
    new_config.update(config)
    return new_config


def _prune_unused_modules(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop `modules` entries whose `module_id` prefixes no remaining model.

    A swap replaces a component's model with one served by a different module, and the
    real driver's module entry then installs and runs on the sim host for nothing. A local
    module, one with no `module_id`, is left alone: nothing here can tell what model it
    serves.
    """
    modules: list[dict[str, Any]] = config.get("modules") or []
    if not modules:
        return config, []
    remaining_models = {
        resource.get("model")
        for resource in (*config.get("components", []), *config.get("services", []))
        if resource.get("model")
    }
    kept: list[dict[str, Any]] = []
    pruned: list[str] = []
    for module in modules:
        module_id = module.get("module_id")
        if not module_id:
            kept.append(module)
            continue
        if any(
            model == module_id or model.startswith(f"{module_id}:") for model in remaining_models
        ):
            kept.append(module)
        else:
            pruned.append(module_id)
    if not pruned:
        return config, []
    config["modules"] = kept
    return config, pruned
