"""The vendored workcell fragment must keep every component and service the
upstream fragment declared, diverging only where the demo machine's
fragment_mod and the deliberate module re-pins say it should. The sim
overlay must line up with the demo machine config on the two poses that
matter: the arm's pedestal height and the gripper's TCP offset."""

import json
from pathlib import Path
from typing import Any

import pytest
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

# The changes deliberately made to the vendored fragment, keyed
# by the JSON path a diff against the upstream fixture surfaces. Re-pinning
# viam:pack-sequencer to 0.3.0 lands it on the same version the upstream
# fixture itself pins, so that module no longer diverges.
EXPECTED_DIVERGENCES = {
    ("components", "stack-light", "frame", "translation", "x"),
    ("modules", "viam:workcell-components", "version"),
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


# Upstream's fragment does not carry these, but something in it names them, so
# a machine built from the fragment alone has a dependency that cannot resolve.
# Each entry is a name we add back, with the reference that needs it.
_ADDED_TO_CLOSE_A_DANGLING_REFERENCE = {
    # pick-station's `infeed_box_detect` names this sensor. Upstream keeps it in
    # the demo MACHINE config rather than the fragment, so vendoring the fragment
    # on its own left pick-station unable to build: "dependency box-detect is not
    # ready yet; reason=resource rdk:component:sensor/box-detect not available".
    "box-detect",
}


def test_no_upstream_component_or_service_was_dropped():
    fragment = _fragment()
    upstream = _upstream_fixture()
    vendored_names = [c["name"] for c in fragment["components"]]
    assert set(vendored_names) >= {c["name"] for c in upstream["components"]}
    assert set(vendored_names) - {c["name"] for c in upstream["components"]} == (
        _ADDED_TO_CLOSE_A_DANGLING_REFERENCE
    )
    assert [s["name"] for s in fragment["services"]] == [s["name"] for s in upstream["services"]]


def test_every_named_dependency_in_the_fragment_resolves():
    # The defect this catches cost a GPU run: a component naming a sibling that
    # the vendoring dropped fails to build, and takes everything downstream of
    # it with it. Attribute keys that hold a resource NAME, not a value.
    fragment = _fragment()
    overlay = _overlay()
    known = {c["name"] for c in fragment["components"]} | {c["name"] for c in overlay["components"]}
    known |= {s["name"] for s in fragment["services"]} | {
        s["name"] for s in overlay.get("services", [])
    }
    name_valued_attrs = ("infeed_box_detect", "tray_dock", "pallet")
    dangling = [
        (component["name"], attr, component["attributes"][attr])
        for component in fragment["components"] + fragment["services"]
        for attr in name_valued_attrs
        if isinstance(component.get("attributes", {}).get(attr), str)
        and component["attributes"][attr] not in known
    ]
    assert dangling == []


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


def test_the_modules_are_pinned_to_the_versions_this_cell_needs():
    # viam:pack-sequencer is deliberately NOT the newest published version.
    # 0.3.0 and 0.4.0-rc3 serve different verb sets, and this cell is built
    # against 0.3.0's.
    modules = {m["module_id"]: m["version"] for m in _fragment()["modules"]}
    assert modules == {
        "viam:workcell-components": "0.7.0",
        "viam:pack-sequencer": "0.3.0",
    }


def _pack_sequencer_attributes() -> dict:
    return next(s for s in _fragment()["services"] if s["name"] == "pack-sequencer")["attributes"]


def test_the_pack_sequencer_attributes_fill_a_full_2x2x2_pack_of_8():
    # packColumn's own formulas (viam-labs/pack-sequencer, 0.3.0), applied to
    # the vendored pallet (500 x 350 mm) and box (200 x 150 x 100 mm): a
    # change to either dimension that breaks the 2 x 2 x 2 pack of 8 should
    # fail here, on the fragment's own numbers, rather than on the GPU.
    pallet = next(c for c in _fragment()["components"] if c["name"] == "pallet")
    pack_sequencer = next(s for s in _fragment()["services"] if s["name"] == "pack-sequencer")
    attrs = pack_sequencer["attributes"]

    pallet_width_mm = pallet["attributes"]["width_mm"]
    pallet_length_mm = pallet["attributes"]["length_mm"]
    box_width_mm = attrs["box_width_mm"]
    box_length_mm = attrs["box_length_mm"]
    box_height_mm = attrs["box_height_mm"]

    cols = int((pallet_width_mm - box_width_mm) / box_width_mm) + 1
    rows = int((pallet_length_mm - box_length_mm) / box_length_mm) + 1
    layers = int(attrs["pallet_area_height_mm"] / box_height_mm)
    capacity = cols * rows * layers

    assert (cols, rows, layers) == (2, 2, 2)
    assert capacity == 8
    assert attrs["quantity"] == 8
    assert attrs["pallet"] == "pallet"


def test_the_provenance_header_names_the_upstream_fragment():
    upstream = _fragment()["_upstream"]
    assert upstream["fragment_id"] == "e42007f2-5a18-4dd1-aeb1-9e2d7bfd0df9"
    assert upstream["pinned_versions"] == {
        "viam:workcell-components": "0.7.0",
        "viam:pack-sequencer": "0.3.0",
    }
    assert upstream["vendored_date"] == "2026-09-15"


def test_the_overlay_arm_sits_on_the_150_mm_pedestal_the_machine_config_used():
    arm = next(c for c in _overlay()["components"] if c["api"] == "rdk:component:arm")
    assert arm["model"] == "viam:isaac-sim-devin:arm"
    assert arm["attributes"]["asset"] == "ur5e"
    assert arm["frame"]["parent"] == "world"
    assert arm["frame"]["translation"]["z"] == 150
    # This pose was proved out on the GPU. A UR5e at all zeros is fully extended across
    # a cell with a pick station in front of it.
    assert arm["attributes"]["home_joints_deg"] == [0, -90, 0, -90, 0, 0]


def test_the_overlay_gripper_matches_the_machine_configs_epick_offset_and_delay():
    gripper = next(c for c in _overlay()["components"] if c["api"] == "rdk:component:gripper")
    assert gripper["model"] == "viam:isaac-sim-devin:vacuum"
    assert gripper["frame"]["parent"] != "world"
    assert gripper["frame"]["translation"]["z"] == 196
    assert gripper["attributes"]["grab_delay_ms"] == 250


_BOX_PROP_NAMES = [f"infeed_box_{i}" for i in range(1, 9)]


def test_the_overlay_world_declares_exactly_the_eight_box_props():
    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    props = {p["name"]: p for p in world["attributes"]["props"]}
    assert set(props) == set(_BOX_PROP_NAMES)

    # A prop's footprint has to sit on the axes pack-sequencer's own slots
    # assume, or every box arrives at its slot turned ninety degrees. Its
    # packColumn puts box_width_mm along the pallet's x and box_length_mm
    # along its y, and the pallet's frame carries no yaw, so pallet x is
    # world x. A 200 mm box laid along y instead would need 400 mm of a
    # 350 mm pallet for one row of two.
    attrs = _pack_sequencer_attributes()
    expected_dims_mm = (attrs["box_width_mm"], attrs["box_length_mm"], attrs["box_height_mm"])
    for box in props.values():
        box_dims_mm = tuple(box["size"] * 1000 * scale for scale in box["scale"])
        assert box_dims_mm == pytest.approx(expected_dims_mm)


def test_the_infeed_box_sits_inside_the_pick_stations_declared_footprint():
    # The check that matters, and the one an earlier version lacked. It read
    # box_origin_offset_mm against the station's FRAME and got (600, 250),
    # which is 350 mm past the near edge of a station spanning y -1200..-100,
    # so the box fell straight through to the floor on the first GPU run.
    # The station's own summary reports `corner at (200, -1200, 220)`, and the
    # offset is measured from that corner.
    pick_station = next(c for c in _fragment()["components"] if c["name"] == "pick-station")
    geometry = pick_station["frame"]["geometry"]
    centre = pick_station["frame"]["translation"]

    half_x = geometry["x"] / 2.0
    half_y = geometry["y"] / 2.0
    top_z_mm = centre["z"] + geometry["z"] / 2.0

    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    props_by_name = {p["name"]: p for p in world["attributes"]["props"]}
    box = props_by_name["infeed_box_1"]
    box_x_mm, box_y_mm, box_z_mm = (value * 1000.0 for value in box["position"])
    box_height_mm = box["size"] * box["scale"][2] * 1000.0

    assert centre["x"] - half_x <= box_x_mm <= centre["x"] + half_x
    assert centre["y"] - half_y <= box_y_mm <= centre["y"] + half_y
    # resting ON the deck, not floating above it or sunk into it
    assert box_z_mm == pytest.approx(top_z_mm + box_height_mm / 2.0)


def test_only_the_infeed_box_is_on_the_pick_station():
    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    props_by_name = {p["name"]: p for p in world["attributes"]["props"]}
    pick_station = next(c for c in _fragment()["components"] if c["name"] == "pick-station")
    geometry = pick_station["frame"]["geometry"]
    centre = pick_station["frame"]["translation"]

    on_station = [
        name
        for name in _BOX_PROP_NAMES
        if abs(props_by_name[name]["position"][0] * 1000.0 - centre["x"]) <= geometry["x"] / 2.0
        and abs(props_by_name[name]["position"][1] * 1000.0 - centre["y"]) <= geometry["y"] / 2.0
    ]
    assert on_station == ["infeed_box_1"]


def test_the_seven_unpicked_boxes_are_parked_clear_of_the_infeed_box():
    world = next(c for c in _overlay()["components"] if c["name"] == "isaac-world")
    props_by_name = {p["name"]: p for p in world["attributes"]["props"]}

    infeed_xy = tuple(props_by_name["infeed_box_1"]["position"][:2])
    parked_names = _BOX_PROP_NAMES[1:]
    parked_xy = {tuple(props_by_name[name]["position"][:2]) for name in parked_names}

    # each parked box has its own spot, and none of them doubles as the
    # infeed pose.
    assert len(parked_xy) == len(parked_names)
    assert infeed_xy not in parked_xy


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


def test_the_palletizer_service_names_the_sequencer_and_its_boxes_in_pick_order():
    service = next(
        s for s in _overlay()["services"] if s["model"] == "viam:isaac-sim-devin:palletizer"
    )
    attrs = service["attributes"]
    assert attrs["sequencer"] == "pack-sequencer"
    assert attrs["box_props"] == _BOX_PROP_NAMES
    # obstacle_source is left unset so IsaacPalletizer's own default,
    # "world_state_store", applies.
    assert "obstacle_source" not in attrs
    assert "place_pose_mm" not in attrs
    assert "box_prop" not in attrs


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


def test_every_box_prop_carries_a_physical_density():
    """A prop's mass is a physics input, not decoration. The GPU run of
    2026-09-16 carried 5 kg on a 0.003 cubic metre box, which is 1667 kg per
    cubic metre, denser than packed sand, and heavier than the UR5e's whole
    rating once the 196 mm tool's moment is counted. The wrist swung visibly
    and stalled short of every place waypoint."""
    overlay = json.loads(OVERLAY_PATH.read_text())
    props = overlay["components"][0]["attributes"]["props"]
    assert props

    for prop in props:
        scale = prop["scale"]
        size_m = prop["size"]
        volume_m3 = (size_m * scale[0]) * (size_m * scale[1]) * (size_m * scale[2])
        density = prop["mass"] / volume_m3
        # cardboard and its contents: lighter than water, heavier than foam
        assert 100.0 <= density <= 1000.0, (
            f"{prop['name']}: {prop['mass']} kg over {volume_m3:.4f} m3 is {density:.0f} kg/m3"
        )
