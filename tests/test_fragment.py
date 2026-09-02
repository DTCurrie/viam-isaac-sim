"""The shipped fragment must validate against the current models — it is the
config a fresh machine is built from, so a validator change that breaks it
has to fail here, not on the machine."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import physics
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld

FRAGMENT_PATH = Path(__file__).resolve().parent.parent / "fragments" / "pick-and-place.json"
MODELS = {
    "world": IsaacWorld,
    "arm": IsaacArm,
    "camera": IsaacCamera,
    "base": IsaacBase,
    "gripper": IsaacGripper,
}
API_PATTERN = re.compile(r"^rdk:component:[a-z_]+$")

# Seam — P5 canonical cell: the five `$variable`s the fragment ships, keyed by
# name, with the default_value a fresh machine that sets nothing must boot.
EXPECTED_VARIABLE_DEFAULTS: dict[str, Any] = {
    "table-height-m": 0.75,
    "pick-block-color": [0.9, 0.1, 0.1],
    "distractor-color-green": [0.05, 0.65, 0.1],
    "distractor-color-blue": [0.05, 0.1, 0.9],
    "detect-color": "#EA8D8D",
}


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text())


def _resolve_variables(node: Any) -> Any:
    """Mimic the app-side `$variable` substitution
    (`fragment_variable_substitution.go`): replace every `{"$variable":
    {"name", "default_value"}}` object with its `default_value`, recursing
    into arrays so a variable inside a `color`/`scale` array resolves too."""
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            return _resolve_variables(variable["default_value"])
        return {key: _resolve_variables(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_variables(item) for item in node]
    return node


def _resolved_fragment() -> dict:
    return _resolve_variables(_fragment())


def _collect_variables(node: Any, found: dict[str, Any]) -> None:
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            found[variable["name"]] = variable.get("default_value")
            return
        for value in node.values():
            _collect_variables(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_variables(item, found)


def _component_config(component: dict) -> ComponentConfig:
    """Build the proto viam-server hands the module from the fragment's JSON
    (only the frame shapes the fragment actually uses are translated)."""
    config = ComponentConfig(
        name=component["name"], attributes=dict_to_struct(component["attributes"])
    )
    frame = component.get("frame")
    if frame is None:
        return config
    config.frame.parent = frame.get("parent", "world")
    translation = frame.get("translation", {})
    config.frame.translation.x = translation.get("x", 0)
    config.frame.translation.y = translation.get("y", 0)
    config.frame.translation.z = translation.get("z", 0)
    orientation = frame.get("orientation")
    if orientation is not None:
        if orientation["type"] != "ov_degrees":
            raise AssertionError(
                f"extend _component_config for orientation type {orientation['type']}"
            )
        value = orientation["value"]
        vector = config.frame.orientation.vector_degrees
        vector.x, vector.y, vector.z, vector.theta = value["x"], value["y"], value["z"], value["th"]
    return config


def test_fragment_is_valid_json_with_the_expected_components():
    names = [c["name"] for c in _fragment()["components"]]
    assert names == ["sim-world", "pick-arm", "pick-grip", "scene-cam", "wrist-cam"]


def test_every_component_uses_the_api_form_not_the_legacy_namespace_type_pair():
    for component in _fragment()["components"]:
        assert "namespace" not in component
        assert "type" not in component
        assert API_PATTERN.match(component["api"])


def test_every_non_world_component_names_sim_world_in_its_attributes():
    for component in _fragment()["components"]:
        if component["name"] == "sim-world":
            continue
        assert component["attributes"]["world"] == "sim-world"


@pytest.mark.parametrize("component", _resolved_fragment()["components"], ids=lambda c: c["name"])
def test_every_fragment_component_validates_against_its_model(component):
    model = MODELS[component["model"].rsplit(":", 1)[1]]
    dependencies, _ = model.validate_config(_component_config(component))
    # components riding the arm (the gripper, the wrist camera's parent_prim)
    # must depend on it so viam-server builds the arm's prim first
    if component["name"] in ("pick-grip", "wrist-cam"):
        assert list(dependencies) == ["sim-world", "pick-arm"]
    elif component["name"] != "sim-world":
        assert list(dependencies) == ["sim-world"]


def test_gripper_frame_matches_its_tcp_offset():
    gripper = next(c for c in _fragment()["components"] if c["name"] == "pick-grip")
    assert gripper["frame"]["parent"] == "pick-arm"
    z = gripper["frame"]["translation"]["z"]
    assert z == 134
    default_tcp_offset_m = 0.134
    assert default_tcp_offset_m * 1000 == z


def test_arm_frame_matches_w6():
    arm = next(c for c in _fragment()["components"] if c["name"] == "pick-arm")
    assert arm["frame"] == {
        "parent": "world",
        "translation": {"x": 150, "y": -250, "z": 750},
    }


def test_pick_cube_physics_matches_the_named_pick_cell_constant():
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    pick_cube = next(p for p in world["attributes"]["props"] if p["name"] == "pick_cube")
    physics_keys = {k: pick_cube[k] for k in physics.PICK_CELL_BLOCK_PHYSICS}
    assert physics_keys == physics.PICK_CELL_BLOCK_PHYSICS


def test_world_step_rates_match_the_pick_cell_constants():
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
    attrs = world["attributes"]
    assert attrs["physics_dt"] == pytest.approx(physics.PICK_CELL_PHYSICS_DT)
    assert attrs["rendering_dt"] == pytest.approx(physics.PICK_CELL_RENDERING_DT)


def test_wrist_camera_matches_the_phase_2_contract():
    wrist = next(c for c in _fragment()["components"] if c["name"] == "wrist-cam")
    assert wrist["attributes"]["depth"] is True
    assert wrist["frame"]["parent"] == "pick-arm"
    assert wrist["attributes"]["parent_prim"] == "/World/pick_arm/wrist_3_link"
    assert "local_position" not in wrist["attributes"]  # the frame is the mount's source of truth


def test_sim_world_is_in_the_frame_system():
    """DEC-21 route (c): the motion service only pulls sim-world's live
    GetGeometries (props + floor) into planning when the component has a
    frame (GPU run 7: without it, app-side moves plan with no obstacles)."""
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
    assert world.get("frame", {}).get("parent") == "world"


def test_sim_world_frame_has_no_translation():
    """Live GetGeometries are expressed in sim-world's own frame (DEC-21 route
    (c)); a frame translation would offset every served geometry."""
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
    assert "translation" not in world["frame"]


def test_sim_world_geometry_matches_the_table_prop_minus_ten_millimetres():
    """W4: the planner box rides inside sim-world's frame.geometry, sized to
    the table prop's x/y footprint with its top 10 mm below the real
    surface (R-24), centred so the box top sits at that height."""
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    table = next(p for p in world["attributes"]["props"] if p["name"] == "table")
    table_x_mm = table["size"] * table["scale"][0] * 1000
    table_y_mm = table["size"] * table["scale"][1] * 1000
    table_top_mm = (table["position"][2] + table["size"] * table["scale"][2] / 2) * 1000

    geometry = world["frame"]["geometry"]
    assert geometry["type"] == "box"
    assert geometry["x"] == pytest.approx(table_x_mm)
    assert geometry["y"] == pytest.approx(table_y_mm)
    assert geometry["z"] == pytest.approx(table_top_mm - 10)

    translation = geometry["translation"]
    assert translation["z"] + geometry["z"] / 2 == pytest.approx(table_top_mm - 10)


def test_three_blocks_follow_the_layout_rules():
    """W23-W26 via the DEC-20 naming: one red target, two colour-distinct
    distractors, all movable, spawned >= 0.20 m apart (W26) and inside the
    verified 743 mm pick radius measured from the arm base (150, -250 mm)."""
    import itertools
    import math

    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    props = {p["name"]: p for p in world["attributes"]["props"]}
    blocks = ["pick_cube", "ignore_cube_green", "ignore_cube_blue"]
    assert all(name in props for name in blocks)

    dominant_channels = [max(range(3), key=lambda i: props[b]["color"][i]) for b in blocks]
    assert dominant_channels == [0, 1, 2]  # red target, green and blue distractors

    arm_base_x, arm_base_y = 0.150, -0.250
    for name in blocks:
        assert not props[name].get("fixed", False)
        x, y, _z = props[name]["position"]
        assert math.hypot(x - arm_base_x, y - arm_base_y) <= 0.743

    for a, b in itertools.combinations(blocks, 2):
        ax, ay, _az = props[a]["position"]
        bx, by, _bz = props[b]["position"]
        assert math.hypot(ax - bx, ay - by) >= 0.20


def test_the_pick_cell_roster_is_present():
    fragment = _fragment()
    world = next(c for c in fragment["components"] if c["name"] == "sim-world")
    props = {p["name"] for p in world["attributes"]["props"]}
    assert props == {"table", "pick_cube", "ignore_cube_green", "ignore_cube_blue", "place_pad"}

    component_names = {c["name"] for c in fragment["components"]}
    assert component_names == {"sim-world", "pick-arm", "pick-grip", "wrist-cam", "scene-cam"}

    service_names = {(s["name"], s["api"]) for s in fragment["services"]}
    # RDK serves the builtin motion service implicitly, so the fragment may
    # carry the entry or omit it - either way the pick client's "builtin"
    # motion resource resolves.
    service_names.discard(("builtin", "rdk:service:motion"))
    assert service_names == {
        ("red-detector", "rdk:service:vision"),
        ("block-segmenter", "rdk:service:vision"),
    }


def test_the_five_variables_ship_with_the_seam_default_values():
    found: dict[str, Any] = {}
    _collect_variables(_fragment(), found)
    assert found == EXPECTED_VARIABLE_DEFAULTS
