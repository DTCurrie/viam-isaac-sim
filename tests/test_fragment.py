"""The shipped fragment must validate against the current models — it is the
config a fresh machine is built from, so a validator change that breaks it
has to fail here, not on the machine."""

import json
from pathlib import Path

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


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text())


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


@pytest.mark.parametrize("component", _fragment()["components"], ids=lambda c: c["name"])
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


def test_pick_cube_physics_matches_the_named_pick_cell_constant():
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
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
