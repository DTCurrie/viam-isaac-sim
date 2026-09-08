"""Resolve a real machine's config into the config of its sim machine.

The real config is never edited. Every hardware component whose model has a
row in `simulates.json` is rewritten to the row's sim model, with `world`
set to the table's default world, the row's template applied, its `$`
references resolved against the real entry, and its `carry` map copied. The
sim module entry and the world component come from the public world
fragment when the config lacks them. `name`, `api`, `frame`, `depends_on`
and every service are left byte for byte.

Two rules live here rather than in the table, because they read more than
one entry. A component already on one of this module's models is passed
through untouched, which makes resolving twice a no-op. A swapped camera
whose frame parent is a swapped arm gets `parent_prim` set to that arm's
default end-effector prim, since the sim camera needs the prim to ride and
the real entry carries only the frame.

This is the reference behavior for an app button or a viam-server flag that
would do the same thing at save or construction time.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import FAMILY, NAMESPACE
from .component_diagnostics import default_ee_prim_path

# Component APIs that drive hardware and so must be simulated or explicitly
# passed through. `rdk:component:generic` is not listed: the world itself is
# generic and a generic component has no hardware to stand in for. Source:
# https://docs.viam.com/dev/reference/apis/ component list (verify when
# implementing, and cite the page read).
HARDWARE_APIS: frozenset[str] = frozenset(
    f"rdk:component:{name}"
    for name in (
        "arm",
        "audio_input",
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
    """What `resolve` produced and which components it touched."""

    config: dict[str, Any]
    swapped: tuple[str, ...]
    passed_through: tuple[str, ...]
    placeholders: tuple[str, ...] = ()


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
    component's real entry instead. Never mutates `config`.
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
        if model.startswith(module_prefix):
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
                # The catch-all is the one row where the api changes: a placeholder generic
                # component keeps the name, frame and depends_on so the config loads, and
                # carries no attributes because nothing stands behind it.
                placeholder = dict(component)
                placeholder["api"] = PLACEHOLDER_API
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

    resolved["components"] = _with_world_component(new_components, world_fragment)
    resolved = _with_module_entry(resolved, world_fragment)

    return Resolution(
        config=resolved,
        swapped=tuple(swapped),
        passed_through=tuple(passed_through),
        placeholders=tuple(placeholders),
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
            component["attributes"]["parent_prim"] = default_ee_prim_path(
                parent.get("attributes") or {}, parent_name
            )
        sensors = (original_by_name[name].get("attributes") or {}).get("sensors")
        if isinstance(sensors, list) and "depth" in sensors:
            component["attributes"]["depth"] = True


def _resolve_fragment_variables(node: Any) -> Any:
    """Replace each `{"$variable": {"name", "default_value"}}` object with its default.

    The app substitutes fragment variables when it renders a machine's config. A resolved
    config is a machine config, not a fragment, so the defaults are baked in here and a
    caller that wants another value (the create-sim-machine flow filling the livestream
    IP) sets the attribute on the emitted component.
    """
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            return _resolve_fragment_variables(variable["default_value"])
        return {key: _resolve_fragment_variables(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_fragment_variables(item) for item in node]
    return node


def _with_world_component(
    components: list[dict[str, Any]], world_fragment: dict[str, Any]
) -> list[dict[str, Any]]:
    existing_names = {c["name"] for c in components}
    missing = [
        _resolve_fragment_variables(copy.deepcopy(c))
        for c in world_fragment.get("components", [])
        if c["name"] not in existing_names
    ]
    return missing + components


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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: `simulate_config.py <real-config.json> [--table PATH]
    [--world-fragment PATH] [--allow-unmatched] [--only a,b] [--out PATH]`.

    Writes the resolved config as two-space-indented JSON with a trailing
    newline to stdout or `--out`. Exit 0 on success, 2 for unmatched
    hardware, 1 for any other error, each with a one-line message on stderr.
    """
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Resolve a real machine config into the config of its sim machine"
    )
    parser.add_argument("config", help="path to the real machine's config JSON")
    parser.add_argument("--table", default=str(repo_root / "simulates.json"))
    parser.add_argument(
        "--world-fragment", default=str(repo_root / "fragments" / "isaac-sim-world.json")
    )
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated component names")
    parser.add_argument("--out", default=None, help="write to this path instead of stdout")
    args = parser.parse_args(argv)

    try:
        config = json.loads(Path(args.config).read_text())
        table = json.loads(Path(args.table).read_text())
        world_fragment = json.loads(Path(args.world_fragment).read_text())
        only = set(args.only.split(",")) if args.only else None
        resolution = resolve(
            config, table, world_fragment, allow_unmatched=args.allow_unmatched, only=only
        )
    except UnmatchedHardwareError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary, reported as a one-line error
        print(str(exc), file=sys.stderr)
        return 1

    output = json.dumps(resolution.config, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(output)
    else:
        sys.stdout.write(output)
    return 0
