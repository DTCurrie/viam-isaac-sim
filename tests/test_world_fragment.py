"""The public world fragment is the one thing a fresh machine must add: exactly
one module entry and the `isaac-world` component, nothing cell-specific."""

import json
import re
from pathlib import Path

from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import DEFAULT_WORLD_NAME, FAMILY, NAMESPACE
from isaac_module.models.world import IsaacWorld

FRAGMENT_PATH = Path(__file__).resolve().parent.parent / "fragments" / "isaac-sim-world.json"
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
API_PATTERN = re.compile(r"^rdk:component:[a-z_]+$")


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text())


def _resolve_variables(node):
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            return _resolve_variables(variable["default_value"])
        return {key: _resolve_variables(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_variables(item) for item in node]
    return node


def _component_config(component: dict) -> ComponentConfig:
    config = ComponentConfig(
        name=component["name"],
        attributes=dict_to_struct(_resolve_variables(component["attributes"])),
    )
    frame = component.get("frame")
    if frame is None:
        return config
    config.frame.parent = frame.get("parent", "world")
    translation = frame.get("translation", {})
    config.frame.translation.x = translation.get("x", 0)
    config.frame.translation.y = translation.get("y", 0)
    config.frame.translation.z = translation.get("z", 0)
    return config


def test_top_level_keys_are_known() -> None:
    fragment = _fragment()
    assert set(fragment.keys()) <= {"modules", "components", "services"}


def test_no_services() -> None:
    fragment = _fragment()
    assert fragment.get("services", []) == []


def test_exactly_one_module_entry() -> None:
    fragment = _fragment()
    modules = fragment["modules"]
    assert len(modules) == 1
    module = modules[0]
    assert module["module_id"] == f"{NAMESPACE}:{FAMILY}"
    assert VERSION_PATTERN.match(module["version"])


def test_exactly_one_component() -> None:
    fragment = _fragment()
    components = fragment["components"]
    assert len(components) == 1
    component = components[0]
    assert component["name"] == DEFAULT_WORLD_NAME
    assert component["model"] == str(IsaacWorld.MODEL)
    assert API_PATTERN.match(component["api"])
    assert component["api"] == "rdk:component:generic"


def test_no_props_in_attributes() -> None:
    fragment = _fragment()
    component = fragment["components"][0]
    assert "props" not in component["attributes"]


def test_frame_is_world_with_no_translation() -> None:
    fragment = _fragment()
    component = fragment["components"][0]
    frame = component["frame"]
    assert frame["parent"] == "world"
    translation = frame.get("translation")
    if translation is not None:
        assert translation.get("x", 0) == 0
        assert translation.get("y", 0) == 0
        assert translation.get("z", 0) == 0


def test_component_validates_in_mock() -> None:
    fragment = _fragment()
    component = fragment["components"][0]
    config = _component_config(component)
    IsaacWorld.validate_config(config)


def test_livestream_public_ip_is_a_fragment_variable_defaulting_to_auto_detect() -> None:
    component = _fragment()["components"][0]
    variable = component["attributes"]["livestream_public_ip"]["$variable"]
    assert variable["name"] == "livestream-public-ip"
    assert variable["default_value"] == ""
