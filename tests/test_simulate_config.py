"""The tool's output for the real UR5e cell example is the committed sim
twin, byte for byte, and the resolver's contract holds on synthetic
configs: idempotence, unmatched-hardware failure, `--only`, and mock
validation of every component it produces."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from isaac_module import DEFAULT_WORLD_NAME
from isaac_module.component_diagnostics import default_ee_prim_path
from isaac_module.config_resolver import (
    UnmatchedHardwareError,
    main,
    resolve,
)
from test_fragment import MODELS, _component_config

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "configs"
REAL_CONFIG_PATH = EXAMPLES_DIR / "real-ur5e-cell.json"
SIM_CONFIG_PATH = EXAMPLES_DIR / "sim-ur5e-cell.json"
TABLE_PATH = REPO_ROOT / "simulates.json"
WORLD_FRAGMENT_PATH = REPO_ROOT / "fragments" / "isaac-sim-world.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _table() -> dict[str, Any]:
    return _load(TABLE_PATH)


def _world_fragment() -> dict[str, Any]:
    return _load(WORLD_FRAGMENT_PATH)


def _by_name(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in components}


def test_tool_output_matches_committed_sim_example_byte_for_byte() -> None:
    real_text = REAL_CONFIG_PATH.read_text()
    sim_text = SIM_CONFIG_PATH.read_text()

    resolution = resolve(json.loads(real_text), _table(), _world_fragment())
    produced = json.dumps(resolution.config, indent=2) + "\n"

    assert produced == sim_text


def test_swapped_components_keep_everything_but_model_and_attributes() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())

    real_by_name = _by_name(real_config["components"])
    resolved_by_name = _by_name(resolution.config["components"])

    for name in ("pick-arm", "pick-grip", "wrist-cam"):
        assert name in resolution.swapped
        real_component = real_by_name[name]
        resolved_component = resolved_by_name[name]
        assert resolved_component["name"] == real_component["name"]
        assert resolved_component["api"] == real_component["api"]
        assert resolved_component["frame"] == real_component["frame"]
        assert resolved_component.get("depends_on") == real_component.get("depends_on")
        assert resolved_component["model"] != real_component["model"]
        assert resolved_component["attributes"] != real_component["attributes"]


def test_services_are_byte_identical() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())

    assert resolution.config["services"] == real_config["services"]


def test_wrist_cam_carries_resolution_depth_and_parent_prim() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    wrist_cam = resolved_by_name["wrist-cam"]
    assert wrist_cam["attributes"]["width"] == 640
    assert wrist_cam["attributes"]["height"] == 480
    assert wrist_cam["attributes"]["depth"] is True

    arm = resolved_by_name["pick-arm"]
    expected_prim = default_ee_prim_path(arm["attributes"], "pick-arm")
    assert wrist_cam["attributes"]["parent_prim"] == expected_prim


def test_pick_grip_carries_the_arm_it_rides() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    assert resolved_by_name["pick-grip"]["attributes"]["arm"] == "pick-arm"


def test_resolving_a_resolved_config_is_a_no_op() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    once = resolve(real_config, _table(), _world_fragment())
    twice = resolve(once.config, _table(), _world_fragment())

    assert twice.config == once.config


def test_never_mutates_the_input_config() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    original = copy.deepcopy(real_config)

    resolve(real_config, _table(), _world_fragment())

    assert real_config == original


def _motor() -> dict[str, Any]:
    return {
        "name": "stray-motor",
        "api": "rdk:component:motor",
        "model": "acme:motors:real",
        "attributes": {"pin": 7},
        "depends_on": ["pick-arm"],
    }


def test_unmatched_hardware_becomes_a_placeholder_generic_component() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["components"].append(_motor())

    resolution = resolve(real_config, _table(), _world_fragment())
    placeholder = _by_name(resolution.config["components"])["stray-motor"]

    assert placeholder["api"] == "rdk:component:generic"
    assert placeholder["model"] == "rdk:builtin:fake"
    assert placeholder["attributes"] == {}
    assert placeholder["depends_on"] == ["pick-arm"]
    assert resolution.placeholders == ("stray-motor",)
    assert "stray-motor" not in resolution.swapped


def test_unmatched_hardware_raises_when_the_table_has_no_catch_all() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["components"].append(_motor())
    table = _table()
    table["rows"] = [row for row in table["rows"] if row["real_model"] != "*"]

    with pytest.raises(UnmatchedHardwareError) as exc_info:
        resolve(real_config, table, _world_fragment())

    assert "stray-motor" in str(exc_info.value)


def test_allow_unmatched_passes_the_hardware_through_unchanged() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    motor_component = _motor()
    real_config["components"].append(dict(motor_component))

    resolution = resolve(real_config, _table(), _world_fragment(), allow_unmatched=True)
    resolved_by_name = _by_name(resolution.config["components"])

    assert resolved_by_name["stray-motor"] == motor_component
    assert "stray-motor" in resolution.passed_through


def test_only_limits_the_swap_to_the_named_component() -> None:
    real_config = _load(REAL_CONFIG_PATH)

    resolution = resolve(real_config, _table(), _world_fragment(), only={"pick-arm"})
    resolved_by_name = _by_name(resolution.config["components"])
    real_by_name = _by_name(real_config["components"])

    assert resolution.swapped == ("pick-arm",)
    assert resolved_by_name["pick-grip"] == real_by_name["pick-grip"]
    assert resolved_by_name["wrist-cam"] == real_by_name["wrist-cam"]


def test_module_entry_and_world_component_inserted_once() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    once = resolve(real_config, _table(), _world_fragment())

    world_fragment = _world_fragment()
    world_name = world_fragment["components"][0]["name"]
    module_id = world_fragment["modules"][0]["module_id"]

    world_components = [c for c in once.config["components"] if c["name"] == world_name]
    modules = [m for m in once.config.get("modules", []) if m.get("module_id") == module_id]
    assert len(world_components) == 1
    assert len(modules) == 1

    twice = resolve(once.config, _table(), _world_fragment())
    world_components_twice = [c for c in twice.config["components"] if c["name"] == world_name]
    modules_twice = [m for m in twice.config.get("modules", []) if m.get("module_id") == module_id]
    assert len(world_components_twice) == 1
    assert len(modules_twice) == 1


def test_swapped_components_validate_in_mock_and_depend_on_the_world() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    for name in ("pick-arm", "pick-grip", "wrist-cam"):
        component = resolved_by_name[name]
        short_name = component["model"].split(":")[-1]
        deps, _opt_deps = MODELS[short_name].validate_config(_component_config(component))
        assert DEFAULT_WORLD_NAME in deps


def test_main_writes_matching_bytes_and_exits_zero(tmp_path: Path) -> None:
    out_path = tmp_path / "sim.json"
    exit_code = main(
        [
            str(REAL_CONFIG_PATH),
            "--table",
            str(TABLE_PATH),
            "--world-fragment",
            str(WORLD_FRAGMENT_PATH),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    assert out_path.read_bytes() == SIM_CONFIG_PATH.read_bytes()


def test_main_exits_two_on_unmatched_hardware(tmp_path: Path) -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["components"].append(_motor())
    bad_config_path = tmp_path / "bad.json"
    bad_config_path.write_text(json.dumps(real_config))
    table = _table()
    table["rows"] = [row for row in table["rows"] if row["real_model"] != "*"]
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(table))

    exit_code = main(
        [
            str(bad_config_path),
            "--table",
            str(table_path),
            "--world-fragment",
            str(WORLD_FRAGMENT_PATH),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert exit_code == 2


def test_shim_runs_as_a_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "simulate_config.py"), str(REAL_CONFIG_PATH)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == json.loads(SIM_CONFIG_PATH.read_text())


def test_inserted_world_component_has_fragment_variables_resolved_to_defaults() -> None:
    resolution = resolve(_load(REAL_CONFIG_PATH), _table(), _world_fragment())
    world = next(c for c in resolution.config["components"] if c["name"] == DEFAULT_WORLD_NAME)
    assert world["attributes"]["livestream_public_ip"] == ""
    assert "$variable" not in json.dumps(world)
