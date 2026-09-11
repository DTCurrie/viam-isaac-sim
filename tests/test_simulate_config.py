import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from isaac_module import DEFAULT_WORLD_NAME
from isaac_module.config_resolver import UnmatchedHardwareError, resolve
from isaac_module.prim_paths import default_ee_prim_path
from test_fragment import MODELS, _component_config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import simulate_config as sim_cli  # noqa: E402

EXAMPLES_DIR = REPO_ROOT / "examples" / "configs"
REAL_CONFIG_PATH = EXAMPLES_DIR / "real-ur5e-cell.json"
SIM_CONFIG_PATH = EXAMPLES_DIR / "sim-ur5e-cell.json"
JETBOT_FAKE_MOTORS_CONFIG_PATH = EXAMPLES_DIR / "real-jetbot-fake-motors.json"
ROVER_MOTORS_BOARD_CONFIG_PATH = EXAMPLES_DIR / "real-rover-motors-board.json"
ARM_ON_GANTRY_CONFIG_PATH = EXAMPLES_DIR / "real-arm-on-gantry.json"
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


def test_unmatched_hardware_becomes_a_placeholder_on_its_own_api() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["components"].append(_motor())

    resolution = resolve(real_config, _table(), _world_fragment())
    placeholder = _by_name(resolution.config["components"])["stray-motor"]

    assert placeholder["api"] == "rdk:component:motor"
    assert placeholder["model"] == "rdk:builtin:fake"
    assert placeholder["attributes"] == {}
    assert placeholder["depends_on"] == ["pick-arm"]
    assert resolution.placeholders == ("stray-motor",)
    assert "stray-motor" not in resolution.swapped


def test_unmatched_pose_tracker_placeholder_falls_back_to_generic_api() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["components"].append(
        {
            "name": "stray-pose-tracker",
            "api": "rdk:component:pose_tracker",
            "model": "acme:tracking:real",
            "attributes": {"resolution": "1080p"},
        }
    )

    resolution = resolve(real_config, _table(), _world_fragment())
    placeholder = _by_name(resolution.config["components"])["stray-pose-tracker"]

    assert placeholder["api"] == "rdk:component:generic"
    assert placeholder["model"] == "rdk:builtin:fake"
    assert placeholder["attributes"] == {}
    assert resolution.placeholders == ("stray-pose-tracker",)


def test_component_already_on_the_fake_model_passes_through_untouched() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    fake_motor = {
        "name": "placeholder-motor",
        "api": "rdk:component:motor",
        "model": "rdk:builtin:fake",
        "attributes": {"max_rpm": 200},
    }
    real_config["components"].append(dict(fake_motor))

    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    assert resolved_by_name["placeholder-motor"] == fake_motor
    assert "placeholder-motor" in resolution.passed_through
    assert "placeholder-motor" not in resolution.swapped
    assert "placeholder-motor" not in resolution.placeholders


def test_audio_in_and_audio_out_are_recognized_as_hardware_apis() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["components"].append(
        {
            "name": "stray-mic",
            "api": "rdk:component:audio_in",
            "model": "acme:audio:real",
            "attributes": {},
        }
    )
    real_config["components"].append(
        {
            "name": "stray-speaker",
            "api": "rdk:component:audio_out",
            "model": "acme:audio:real",
            "attributes": {},
        }
    )

    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    assert set(resolution.placeholders) >= {"stray-mic", "stray-speaker"}
    assert resolved_by_name["stray-mic"]["api"] == "rdk:component:audio_in"
    assert resolved_by_name["stray-speaker"]["api"] == "rdk:component:audio_out"


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
    exit_code = sim_cli.main(
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

    exit_code = sim_cli.main(
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


def test_fragment_variable_without_default_value_is_left_unresolved_and_reported() -> None:
    world_fragment = {
        "components": [
            {
                "name": "isaac-world",
                "api": "rdk:component:generic",
                "model": "viam:isaac-sim-devin:world",
                "attributes": {"world": "isaac-world"},
            },
            {
                "name": "extra-sensor",
                "api": "rdk:component:sensor",
                "model": "viam:isaac-sim-devin:sensor",
                "attributes": {"reading_rate_hz": {"$variable": {"name": "reading-rate"}}},
            },
        ]
    }

    resolution = resolve({"components": []}, _table(), world_fragment)
    extra_sensor = _by_name(resolution.config["components"])["extra-sensor"]

    assert resolution.unresolved_variables == ("reading-rate",)
    assert extra_sensor["attributes"]["reading_rate_hz"] == {"$variable": {"name": "reading-rate"}}
    assert "$variable" in json.dumps(extra_sensor)


def test_realsense_without_sensors_key_defaults_to_depth_true() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    wrist_cam = _by_name(real_config["components"])["wrist-cam"]
    del wrist_cam["attributes"]["sensors"]

    resolution = resolve(real_config, _table(), _world_fragment())
    wrist_cam_resolved = _by_name(resolution.config["components"])["wrist-cam"]

    assert wrist_cam_resolved["attributes"]["depth"] is True


def test_realsense_sensors_without_depth_does_not_turn_depth_on() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    wrist_cam = _by_name(real_config["components"])["wrist-cam"]
    wrist_cam["attributes"]["sensors"] = ["color"]

    resolution = resolve(real_config, _table(), _world_fragment())
    wrist_cam_resolved = _by_name(resolution.config["components"])["wrist-cam"]

    assert "depth" not in wrist_cam_resolved["attributes"]


def test_pruned_modules_drops_entries_no_remaining_model_uses() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    real_config["modules"] = [
        {
            "type": "registry",
            "name": "universal_robots",
            "module_id": "viam:universal-robots",
            "version": "1.2.3",
        },
        {"type": "registry", "name": "orphan", "module_id": "acme:unrelated", "version": "0.0.1"},
        {"type": "local", "name": "custom", "executable_path": "/bin/custom"},
    ]

    resolution = resolve(real_config, _table(), _world_fragment())
    kept_ids = {m.get("module_id") for m in resolution.config["modules"]}

    assert set(resolution.pruned_modules) == {"viam:universal-robots", "acme:unrelated"}
    assert "viam:universal-robots" not in kept_ids
    assert "acme:unrelated" not in kept_ids
    assert "viam:isaac-sim-devin" in kept_ids
    assert any(m.get("name") == "custom" for m in resolution.config["modules"])


def test_jetbot_fake_motors_swaps_the_base_and_passes_the_motors_through() -> None:
    real_config = _load(JETBOT_FAKE_MOTORS_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    assert "rover-base" in resolution.swapped
    base = resolved_by_name["rover-base"]
    assert base["model"] == "viam:isaac-sim-devin:base"
    assert base["attributes"]["asset"] == "jetbot"

    for motor_name in ("left-motor", "right-motor"):
        assert motor_name in resolution.passed_through
        assert resolved_by_name[motor_name] == _by_name(real_config["components"])[motor_name]

    short_name = base["model"].split(":")[-1]
    deps, _opt_deps = MODELS[short_name].validate_config(_component_config(base))
    assert DEFAULT_WORLD_NAME in deps


def test_rover_motors_and_board_with_no_rows_become_placeholders() -> None:
    real_config = _load(ROVER_MOTORS_BOARD_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])
    real_by_name = _by_name(real_config["components"])

    expected_placeholders = {"rover-base", "left-motor", "right-motor", "power-board"}
    assert set(resolution.placeholders) == expected_placeholders
    for name in resolution.placeholders:
        placeholder = resolved_by_name[name]
        assert placeholder["api"] == real_by_name[name]["api"]
        assert placeholder["model"] == "rdk:builtin:fake"
        assert placeholder["attributes"] == {}
        assert placeholder["name"] == name


def test_arm_on_a_non_world_frame_parent_fails_at_resolve_time() -> None:
    # An arm whose frame.parent names another component now fails here, at resolve
    # time, naming the remedy, rather than resolving to a config that only fails once
    # it reaches the GPU host, after create_sim_machine.py has already made a machine.
    real_config = _load(ARM_ON_GANTRY_CONFIG_PATH)

    with pytest.raises(ValueError, match="pick-arm") as exc_info:
        resolve(real_config, _table(), _world_fragment())

    message = str(exc_info.value)
    assert "lift-gantry" in message
    assert "allow_unmatched" in message


def test_resolved_config_gets_a_finalizer_depending_on_the_world_and_every_swapped_component() -> (
    None
):
    real_config = _load(REAL_CONFIG_PATH)
    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    finalizers = [c for c in resolution.config["components"] if c["name"] == "scene-finalizer"]
    assert len(finalizers) == 1
    assert finalizers[0]["depends_on"] == [DEFAULT_WORLD_NAME, "pick-arm", "pick-grip", "wrist-cam"]
    assert resolved_by_name[DEFAULT_WORLD_NAME]["attributes"]["wait_for_finalizer"] is True


def test_resolved_config_with_no_sim_component_finalizer_depends_only_on_the_world() -> None:
    config = {"components": []}
    resolution = resolve(config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    finalizer = resolved_by_name["scene-finalizer"]
    assert finalizer["depends_on"] == [DEFAULT_WORLD_NAME]
    assert resolved_by_name[DEFAULT_WORLD_NAME]["attributes"]["wait_for_finalizer"] is False


def test_user_supplied_finalizer_is_passed_through_untouched() -> None:
    real_config = _load(REAL_CONFIG_PATH)
    user_finalizer = {
        "name": "scene-finalizer",
        "api": "rdk:component:generic",
        "model": "viam:isaac-sim-devin:scene-finalizer",
        "depends_on": ["pick-arm"],
    }
    real_config["components"].append(dict(user_finalizer))

    resolution = resolve(real_config, _table(), _world_fragment())
    resolved_by_name = _by_name(resolution.config["components"])

    assert resolved_by_name["scene-finalizer"] == user_finalizer
    world_attrs = resolved_by_name[DEFAULT_WORLD_NAME]["attributes"]
    assert "wait_for_finalizer" not in world_attrs or world_attrs["wait_for_finalizer"] is False


def test_arm_on_a_non_world_frame_parent_passes_through_with_allow_unmatched() -> None:
    real_config = _load(ARM_ON_GANTRY_CONFIG_PATH)
    real_by_name = _by_name(real_config["components"])

    resolution = resolve(real_config, _table(), _world_fragment(), allow_unmatched=True)
    resolved_by_name = _by_name(resolution.config["components"])

    assert "pick-arm" in resolution.swapped
    arm = resolved_by_name["pick-arm"]
    assert arm["frame"] == real_by_name["pick-arm"]["frame"]
    assert arm["frame"]["parent"] == "lift-gantry"

    assert "lift-gantry" in resolution.passed_through
    assert resolved_by_name["lift-gantry"] == real_by_name["lift-gantry"]
