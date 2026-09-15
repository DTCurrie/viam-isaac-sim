"""The vendored workcell fragment must keep every component and service the
upstream fragment declared, diverging only where the demo machine's
fragment_mod and the deliberate module re-pins say it should. The sim
overlay must line up with the demo machine config on the two poses that
matter: the arm's pedestal height and the gripper's TCP offset."""

import json
from pathlib import Path
from typing import Any

from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.palletizer import IsaacPalletizer
from isaac_module.models.scene_finalizer import IsaacSceneFinalizer
from isaac_module.models.vacuum import IsaacVacuum
from isaac_module.models.world import IsaacWorld

FRAGMENT_PATH = Path(__file__).resolve().parent.parent / "fragments" / "isaac-sim-palletizing.json"
OVERLAY_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "configs" / "sim-palletizer-cell.json"
)
# Byte for byte copy of the fragment the user pasted (id
# e42007f2-5a18-4dd1-aeb1-9e2d7bfd0df9), so the vendored copy is checked
# against the real upstream document rather than against a second
# transcription of it.
UPSTREAM_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "upstream-workcell-fragment.json"
)

# The two changes this phase deliberately makes to the vendored fragment,
# keyed by the JSON path a diff against the upstream fixture surfaces.
EXPECTED_DIVERGENCES = {
    ("components", "stack-light", "frame", "translation", "x"),
    ("modules", "viam:workcell-components", "version"),
    ("modules", "viam:pack-sequencer", "version"),
}


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text())


def _overlay() -> dict:
    return json.loads(OVERLAY_PATH.read_text())


def _upstream_fixture() -> dict:
    return json.loads(UPSTREAM_FIXTURE_PATH.read_text())


def _named(items: list[dict], key: str) -> dict[str, dict]:
    return {item[key]: item for item in items}


def _module_by_id(modules: list[dict]) -> dict[str, dict]:
    return {module["module_id"]: module for module in modules}


def _diff_paths(prefix: tuple[str, ...], upstream: Any, vendored: Any, found: set[tuple]) -> None:
    """Walk `upstream` and `vendored` together, recording the path of every
    leaf value where the two disagree. A dict compares key by key, and every
    other type compares by equality, so a divergence anywhere in a nested
    structure surfaces as one path rather than failing the whole traversal."""
    if isinstance(upstream, dict) and isinstance(vendored, dict):
        for key in upstream:
            _diff_paths((*prefix, key), upstream[key], vendored.get(key), found)
        return
    if upstream != vendored:
        found.add(prefix)


def test_no_upstream_component_or_service_was_dropped():
    fragment = _fragment()
    upstream = _upstream_fixture()
    assert [c["name"] for c in fragment["components"]] == [
        c["name"] for c in upstream["components"]
    ]
    assert [s["name"] for s in fragment["services"]] == [s["name"] for s in upstream["services"]]


def test_the_vendored_fragment_diverges_from_upstream_only_where_expected():
    upstream = _upstream_fixture()
    vendored = _fragment()

    upstream_view = {
        "components": _named(upstream["components"], "name"),
        "services": _named(upstream["services"], "name"),
        "modules": _module_by_id(upstream["modules"]),
    }
    vendored_view = {
        "components": _named(vendored["components"], "name"),
        "services": _named(vendored["services"], "name"),
        "modules": _module_by_id(vendored["modules"]),
    }

    found: set[tuple] = set()
    _diff_paths((), upstream_view, vendored_view, found)
    assert found == EXPECTED_DIVERGENCES


def test_the_stack_light_mod_landed():
    stack_light = next(c for c in _fragment()["components"] if c["name"] == "stack-light")
    assert stack_light["frame"]["translation"] == {"x": -1250, "y": 1050, "z": 0}


def test_the_modules_are_pinned_to_the_newest_published_versions():
    modules = {m["module_id"]: m["version"] for m in _fragment()["modules"]}
    assert modules == {
        "viam:workcell-components": "0.7.0",
        "viam:pack-sequencer": "0.4.0-rc3",
    }


def test_the_provenance_header_names_the_upstream_fragment():
    upstream = _fragment()["_upstream"]
    assert upstream["fragment_id"] == "e42007f2-5a18-4dd1-aeb1-9e2d7bfd0df9"
    assert upstream["pinned_versions"] == {
        "viam:workcell-components": "0.7.0",
        "viam:pack-sequencer": "0.4.0-rc3",
    }
    assert upstream["vendored_date"] == "2026-09-15"


def test_the_overlay_arm_sits_on_the_150_mm_pedestal_the_machine_config_used():
    arm = next(c for c in _overlay()["components"] if c["api"] == "rdk:component:arm")
    assert arm["model"] == "viam:isaac-sim-devin:arm"
    assert arm["attributes"]["asset"] == "ur5e"
    assert arm["frame"]["parent"] == "world"
    assert arm["frame"]["translation"]["z"] == 150
    # phase 1 proved this pose. A UR5e at all zeros is fully extended across
    # a cell with a pick station in front of it.
    assert arm["attributes"]["home_joints_deg"] == [0, -90, 0, -90, 0, 0]


def test_the_overlay_gripper_matches_the_machine_configs_epick_offset_and_delay():
    gripper = next(c for c in _overlay()["components"] if c["api"] == "rdk:component:gripper")
    assert gripper["model"] == "viam:isaac-sim-devin:vacuum"
    assert gripper["frame"]["parent"] != "world"
    assert gripper["frame"]["translation"]["z"] == 196
    assert gripper["attributes"]["grab_delay_ms"] == 250


def test_the_overlay_world_declares_only_the_box_prop():
    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    props = {p["name"]: p for p in world["attributes"]["props"]}
    assert set(props) == {"infeed_box"}

    box = props["infeed_box"]
    box_dims_mm = tuple(box["size"] * 1000 * scale for scale in box["scale"])
    # box_length_mm, box_width_mm, box_height_mm from the vendored
    # pack-sequencer service, in that order.
    assert box_dims_mm == (150, 200, 100)


def test_the_box_sits_at_the_pick_stations_frame_plus_its_box_origin_offset():
    pick_station = next(c for c in _fragment()["components"] if c["name"] == "pick-station")
    station_translation = pick_station["frame"]["translation"]
    offset = pick_station["attributes"]["box_origin_offset_mm"]

    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    box = next(p for p in world["attributes"]["props"] if p["name"] == "infeed_box")

    expected_x_m = (station_translation["x"] + offset["x"]) / 1000.0
    expected_y_m = (station_translation["y"] + offset["y"]) / 1000.0
    assert box["position"][0] == expected_x_m
    assert box["position"][1] == expected_y_m


def test_the_overlay_carries_no_floor_prop():
    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    prop_names = {p["name"] for p in world["attributes"]["props"]}
    assert "floor" not in prop_names


def test_the_overlay_names_our_module_and_the_palletizer_service():
    overlay = _overlay()
    module_ids = {m["module_id"] for m in overlay["modules"]}
    assert module_ids == {"viam:isaac-sim-devin"}

    service = next(s for s in overlay["services"] if s["api"] == "rdk:service:generic")
    assert service["model"] == "viam:isaac-sim-devin:palletizer"
    arm_name = next(c["name"] for c in overlay["components"] if c["api"] == "rdk:component:arm")
    gripper_name = next(
        c["name"] for c in overlay["components"] if c["api"] == "rdk:component:gripper"
    )
    assert service["attributes"]["arm"] == arm_name
    assert service["attributes"]["gripper"] == gripper_name


def test_the_palletizer_service_carries_a_place_pose_the_gripper_can_release_at():
    service = next(
        s for s in _overlay()["services"] if s["model"] == "viam:isaac-sim-devin:palletizer"
    )
    place_pose_mm = service["attributes"]["place_pose_mm"]

    # x, y from the vendored fragment's pallet frame, whose origin is the
    # bounding-box centroid per the upstream meta.json. z is the pallet's
    # top face (frame z 200 mm plus half its 100 mm thickness) plus the box
    # height (100 mm, from pack-sequencer) plus CUP_APPROACH_GAP_MM (the cup
    # stops short of the top face it is releasing onto, the same offset
    # pick_grasp_pose uses), all from isaac_module.models.palletizer.
    fragment = _fragment()
    pallet = next(c for c in fragment["components"] if c["name"] == "pallet")
    pallet_translation = pallet["frame"]["translation"]
    pallet_top_face_z_mm = pallet_translation["z"] + pallet["attributes"]["thickness_mm"] / 2

    pack_sequencer = next(s for s in fragment["services"] if s["name"] == "pack-sequencer")
    box_height_mm = pack_sequencer["attributes"]["box_height_mm"]
    cup_approach_gap_mm = 5.0  # isaac_module.models.palletizer.CUP_APPROACH_GAP_MM

    assert place_pose_mm["x"] == pallet_translation["x"]
    assert place_pose_mm["y"] == pallet_translation["y"]
    assert place_pose_mm["z"] == pallet_top_face_z_mm + box_height_mm + cup_approach_gap_mm


# Every model this module owns, keyed by the fully qualified model string a
# fragment or overlay component declares. viam:workcell-components models are
# not ours to import a validator for, so a component riding one is skipped.
OWNED_MODELS: dict[str, Any] = {
    "viam:isaac-sim-devin:world": IsaacWorld,
    "viam:isaac-sim-devin:arm": IsaacArm,
    "viam:isaac-sim-devin:camera": IsaacCamera,
    "viam:isaac-sim-devin:vacuum": IsaacVacuum,
    "viam:isaac-sim-devin:scene-finalizer": IsaacSceneFinalizer,
    "viam:isaac-sim-devin:palletizer": IsaacPalletizer,
}


def _resolve_variables(node: Any) -> Any:
    """Mimic the app-side `$variable` substitution: replace every
    `{"$variable": {"name", "default_value"}}` object with its
    `default_value`, the same as viam-server does before handing a config to
    a model's `validate_config`."""
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            return _resolve_variables(variable["default_value"])
        return {key: _resolve_variables(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_variables(item) for item in node]
    return node


def _overlay_component_config(resource: dict) -> ComponentConfig:
    """Build the proto viam-server hands the module from the overlay's JSON,
    translating only the frame shapes the overlay actually uses."""
    config = ComponentConfig(
        name=resource["name"],
        attributes=dict_to_struct(_resolve_variables(resource.get("attributes", {}))),
    )
    config.depends_on.extend(resource.get("depends_on", []))
    frame = resource.get("frame")
    if frame is None:
        return config
    config.frame.parent = frame.get("parent", "world")
    translation = frame.get("translation", {})
    config.frame.translation.x = translation.get("x", 0)
    config.frame.translation.y = translation.get("y", 0)
    config.frame.translation.z = translation.get("z", 0)
    orientation = frame.get("orientation")
    if orientation is not None:
        value = orientation["value"]
        vector = config.frame.orientation.vector_degrees
        vector.x, vector.y, vector.z, vector.theta = value["x"], value["y"], value["z"], value["th"]
    return config


def test_every_overlay_resource_this_module_owns_validates_against_its_model():
    overlay = _overlay()
    resources = [*overlay["components"], *overlay["services"]]
    checked = 0
    for resource in resources:
        model = OWNED_MODELS.get(resource["model"])
        if model is None:
            continue
        model.validate_config(_overlay_component_config(resource))
        checked += 1
    # every overlay resource in this file carries one of our models, so a
    # model rename or an accidental workcell-components entry that silently
    # skipped validation would still be caught by this count.
    assert checked == len(resources)
