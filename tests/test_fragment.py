"""The shipped fragment must validate against the current models — it is the
config a fresh machine is built from, so a validator change that breaks it
has to fail here, not on the machine. Every geometry literal in the fragment
must agree with `isaac_module.cell_layout`, the one source of truth for the
sorting cell's layout."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import cell_layout, physics
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.sorter_sensor import SorterSensor
from isaac_module.models.world import IsaacWorld

FRAGMENT_PATH = Path(__file__).resolve().parent.parent / "fragments" / "pick-and-place.json"
MODELS = {
    "world": IsaacWorld,
    "arm": IsaacArm,
    "camera": IsaacCamera,
    "base": IsaacBase,
    "gripper": IsaacGripper,
    "sorter-sensor": SorterSensor,
}
API_PATTERN = re.compile(r"^rdk:component:[a-z_]+$")

BLOCK_COLOR_DEFAULTS: dict[str, list[float]] = {
    "red": [0.9, 0.1, 0.1],
    "green": [0.05, 0.65, 0.1],
    "blue": [0.05, 0.1, 0.9],
    "yellow": [0.9, 0.75, 0.05],
    "purple": [0.55, 0.1, 0.75],
    "orange": [1.0, 0.4, 0.05],
}

# Seam — the nineteen `$variable`s the fragment ships, keyed by name, with the
# default_value a fresh machine that sets nothing must boot.
DETECT_COLOR_DEFAULTS: dict[str, str] = {
    "red": "#EA8D8D",
    "green": "#6AE28B",
    "blue": "#869EEE",
    "yellow": "#EEDE64",
    "purple": "#E399EB",
    "orange": "#F0D76B",
}
HUE_TOLERANCE_PCT_DEFAULTS: dict[str, float] = {
    "red": 0.05,
    "green": 0.05,
    "blue": 0.05,
    "yellow": 0.05,
    "purple": 0.05,
    "orange": 0.05,
}
EXPECTED_VARIABLE_DEFAULTS: dict[str, Any] = {
    "table-height-m": 0.75,
    **{f"block-color-{color}": default for color, default in BLOCK_COLOR_DEFAULTS.items()},
    "detect-color": DETECT_COLOR_DEFAULTS["red"],
    "hue-tolerance-pct": HUE_TOLERANCE_PCT_DEFAULTS["red"],
    **{
        f"detect-color-{color}": default
        for color, default in DETECT_COLOR_DEFAULTS.items()
        if color != "red"
    },
    **{
        f"hue-tolerance-pct-{color}": default
        for color, default in HUE_TOLERANCE_PCT_DEFAULTS.items()
        if color != "red"
    },
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


def _world_props() -> dict[str, dict]:
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    return {p["name"]: p for p in world["attributes"]["props"]}


def _expected_prop_names() -> set[str]:
    tables = {"table_source", "table_arm", "table_place"}
    pads = {cell_layout.pad_name(color) for color in cell_layout.BLOCK_COLORS}
    blocks = {
        cell_layout.pool_block_name(color, index)
        for color in cell_layout.BLOCK_COLORS
        for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
    }
    return tables | pads | blocks | {"side_cam_body", "side_cam_mount"}


def test_fragment_is_valid_json_with_the_expected_components():
    names = [c["name"] for c in _fragment()["components"]]
    assert names == [
        "sim-world",
        "pick-arm",
        "pick-grip",
        "scene-cam",
        "side-cam",
        "wrist-cam",
        "block-sorter-sensor",
    ]


def test_every_component_uses_the_api_form_not_the_legacy_namespace_type_pair():
    for component in _fragment()["components"]:
        assert "namespace" not in component
        assert "type" not in component
        assert API_PATTERN.match(component["api"])


def test_every_non_world_component_names_sim_world_in_its_attributes():
    # block-sorter-sensor polls the conductor, not the sim, so it carries no
    # `world` attribute of its own.
    for component in _fragment()["components"]:
        if component["name"] in ("sim-world", "block-sorter-sensor"):
            continue
        assert component["attributes"]["world"] == "sim-world"


@pytest.mark.parametrize("component", _resolved_fragment()["components"], ids=lambda c: c["name"])
def test_every_fragment_component_validates_against_its_model(component):
    # this test only ever iterated `components`, not `services`, so the new
    # `block-sorter` conductor entry (a service) is out of scope here already;
    # its model lands in slice 4c and gets its own validation test there.
    short_name = component["model"].rsplit(":", 1)[1]
    model = MODELS[short_name]
    dependencies, _ = model.validate_config(_component_config(component))
    # components riding the arm (the gripper, the wrist camera's parent_prim)
    # must depend on it so viam-server builds the arm's prim first
    if component["name"] in ("pick-grip", "wrist-cam"):
        assert list(dependencies) == ["sim-world", "pick-arm"]
    # the sorter sensor never touches the sim: it polls the conductor service
    elif component["name"] == "block-sorter-sensor":
        assert list(dependencies) == ["block-sorter"]
    elif component["name"] != "sim-world":
        assert list(dependencies) == ["sim-world"]


def test_gripper_frame_matches_its_tcp_offset():
    gripper = next(c for c in _fragment()["components"] if c["name"] == "pick-grip")
    assert gripper["frame"]["parent"] == "pick-arm"
    z = gripper["frame"]["translation"]["z"]
    assert z == 134
    default_tcp_offset_m = 0.134
    assert default_tcp_offset_m * 1000 == z


def test_arm_frame_matches_the_seam_base_pose():
    arm = next(c for c in _fragment()["components"] if c["name"] == "pick-arm")
    assert arm["frame"] == {
        "parent": "world",
        "translation": {
            "x": cell_layout.ARM_BASE_XY_MM[0],
            "y": cell_layout.ARM_BASE_XY_MM[1],
            "z": cell_layout.ARM_BASE_Z_MM,
        },
    }
    assert arm["attributes"]["asset"] == "ur20"


def test_block_physics_matches_the_named_pick_cell_constant_for_every_block():
    props = _world_props()
    for color in cell_layout.BLOCK_COLORS:
        for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1):
            block = props[cell_layout.pool_block_name(color, index)]
            physics_keys = {k: block[k] for k in physics.PICK_CELL_BLOCK_PHYSICS}
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


def test_wrist_and_side_cameras_carry_a_collision_geometry_for_the_planner():
    """Both cameras ride within reach of the planner's swept volume, so each
    carries its RealSense body (90x25x25 mm, centred on the frame origin) as
    frame geometry with no translation. A box, not the
    tools/generate_realsense_mesh.py mesh: the app's fragment validation
    rejected mesh geometries on 2026-09-02 (phase-4 §Deferred). The wrist
    camera's long side runs along the frame's x (its boresight); the side
    camera's long side runs along y, across its boresight (its frame's x),
    matching the physical `side_cam_body` prop's orientation."""
    expected_dims = {"wrist-cam": (90, 25, 25), "side-cam": (25, 90, 25)}
    for name, (expected_x, expected_y, expected_z) in expected_dims.items():
        component = next(c for c in _fragment()["components"] if c["name"] == name)
        geometry = component["frame"]["geometry"]
        assert geometry["type"] == "box"
        assert geometry["x"] == expected_x
        assert geometry["y"] == expected_y
        assert geometry["z"] == expected_z
        assert "translation" not in geometry


def test_side_cam_body_is_a_fixed_prop_facing_away_from_its_own_camera():
    """The rendered RealSense housing must be fixed (it's bolted, not
    scattered) and its near face, along the lens's own view axis (the
    body's shallow x dimension; the long side runs across the boresight),
    must sit at or behind the lens so the camera never sees the inside of
    its own body, without drifting so far behind that it intrudes into the
    scatter zone it looks at."""
    body = _world_props()["side_cam_body"]
    assert body["fixed"] is True
    lens_x_m = cell_layout.SIDE_CAMERA_POSITION_MM[0] / 1000.0
    scatter_far_edge_x_m = cell_layout.SCATTER_ZONE_X_MM[0] / 1000.0
    x, y, z = body["position"]
    assert y == pytest.approx(cell_layout.SIDE_CAMERA_POSITION_MM[1] / 1000.0)
    assert z == pytest.approx(cell_layout.SIDE_CAMERA_POSITION_MM[2] / 1000.0)
    x_half_extent = body["size"] * body["scale"][0] / 2
    near_face_x = x + x_half_extent
    assert near_face_x <= lens_x_m
    assert near_face_x <= scatter_far_edge_x_m


def test_side_cam_mount_stands_on_the_table_and_meets_the_camera_body():
    """The mount is the fixed post the camera body sits on: it must rise
    from the source table's own top face, and its top face must meet the
    body's bottom face flush, all computed from each prop's size, scale,
    and position rather than restated magic sums."""
    props = _world_props()
    mount = props["side_cam_mount"]
    body = props["side_cam_body"]
    table = props["table_source"]
    assert mount["fixed"] is True

    table_top_z_m = cell_layout.TABLE_TOP_Z_MM / 1000.0
    table_half_x = table["size"] * table["scale"][0] / 2
    table_min_x = table["position"][0] - table_half_x
    table_max_x = table["position"][0] + table_half_x

    mount_half_z = mount["size"] * mount["scale"][2] / 2
    mount_bottom_z = mount["position"][2] - mount_half_z
    mount_top_z = mount["position"][2] + mount_half_z
    assert mount_bottom_z == pytest.approx(table_top_z_m)
    assert table_min_x <= mount["position"][0] <= table_max_x

    body_half_z = body["size"] * body["scale"][2] / 2
    body_bottom_z = body["position"][2] - body_half_z
    assert mount_top_z == pytest.approx(body_bottom_z)


def test_side_camera_sits_outside_the_scatter_region_and_aims_at_its_centre():
    """Phase 4 seam: `side-cam` is planted just past the scatter region's far
    -x edge at lens height above the table top, looking back toward the arm.
    It aims via a FRAME orientation, not `target`: the frame is what
    transform_pose reports, so prim aim and frame claim must be one
    quaternion (GPU phase-4 run 1: `target` aimed the prim while the frame
    claimed identity, and side scans measured the backdrop at 7994 mm). The
    orientation vector must point from the lens to the scatter-region centre
    on the table top."""
    side = next(c for c in _fragment()["components"] if c["name"] == "side-cam")
    assert side["frame"]["parent"] == "world"
    translation = side["frame"]["translation"]
    assert translation["x"] == cell_layout.SIDE_CAMERA_POSITION_MM[0]
    assert translation["y"] == cell_layout.SIDE_CAMERA_POSITION_MM[1]
    assert translation["z"] == cell_layout.SIDE_CAMERA_POSITION_MM[2]
    assert translation["x"] <= cell_layout.SCATTER_ZONE_X_MM[0]
    assert side["attributes"]["depth"] is True
    assert "target" not in side["attributes"]

    orientation = side["frame"]["orientation"]
    assert orientation["type"] == "ov_degrees"
    vector = orientation["value"]
    region_centre_mm = cell_layout.SCATTER_CENTRE_MM
    aim = [
        region_centre_mm[i] - (translation["x"], translation["y"], translation["z"])[i]
        for i in range(3)
    ]
    ov = [vector["x"], vector["y"], vector["z"]]
    cross = [
        aim[1] * ov[2] - aim[2] * ov[1],
        aim[2] * ov[0] - aim[0] * ov[2],
        aim[0] * ov[1] - aim[1] * ov[0],
    ]
    assert all(abs(c) < 1e-9 for c in cross)  # parallel to the lens->centre ray
    assert sum(a * o for a, o in zip(aim, ov, strict=True)) > 0  # and not flipped away from it


def test_scene_camera_frames_all_three_tables():
    scene = next(c for c in _fragment()["components"] if c["name"] == "scene-cam")
    translation = scene["frame"]["translation"]
    assert translation["x"] == cell_layout.SCENE_CAMERA_POSITION_MM[0]
    assert translation["y"] == cell_layout.SCENE_CAMERA_POSITION_MM[1]
    assert translation["z"] == cell_layout.SCENE_CAMERA_POSITION_MM[2]
    assert scene["attributes"]["target"] == list(cell_layout.SCENE_CAMERA_TARGET_M)


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


def test_sim_world_geometry_matches_the_flush_three_table_slab_minus_ten_millimetres():
    """W4: the planner box rides inside sim-world's frame.geometry, sized to
    the flush three-table span with its top 10 mm below the real surface
    (R-24), centred so the box top sits at that height."""
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    table_source = cell_layout.TABLE_CENTRES_MM["table_source"][0]
    table_place = cell_layout.TABLE_CENTRES_MM["table_place"][0]
    span_x_mm = (table_place + cell_layout.TABLE_DIMS_MM[0] / 2) - (
        table_source - cell_layout.TABLE_DIMS_MM[0] / 2
    )

    geometry = world["frame"]["geometry"]
    assert geometry["type"] == "box"
    assert geometry["x"] == pytest.approx(span_x_mm)
    assert geometry["y"] == pytest.approx(cell_layout.TABLE_DIMS_MM[1])
    assert geometry["z"] == pytest.approx(cell_layout.TABLE_TOP_Z_MM - 10)

    translation = geometry["translation"]
    assert translation["x"] == 0
    assert translation["y"] == 0
    assert translation["z"] + geometry["z"] / 2 == pytest.approx(cell_layout.TABLE_TOP_Z_MM - 10)


def test_three_tables_sit_at_the_seam_centres():
    props = _world_props()
    for name, (x_mm, y_mm) in cell_layout.TABLE_CENTRES_MM.items():
        table = props[name]
        assert table["fixed"] is True
        assert table["position"][0] == pytest.approx(x_mm / 1000.0)
        assert table["position"][1] == pytest.approx(y_mm / 1000.0)
        assert table["scale"][0] == pytest.approx(cell_layout.TABLE_DIMS_MM[0] / 1000.0)
        assert table["scale"][1] == pytest.approx(cell_layout.TABLE_DIMS_MM[1] / 1000.0)


def test_pads_sit_at_the_seam_centres_with_no_overlap_and_a_level_top():
    """Every pad's footprint centre matches the seam, its top sits at
    `PAD_TOP_Z_MM`, and no two pads overlap on both axes."""
    import itertools

    props = _world_props()
    half_pad_m = cell_layout.PAD_SIDE_MM / 2000.0
    for color, (x_mm, y_mm) in cell_layout.PAD_CENTRES_MM.items():
        pad = props[cell_layout.pad_name(color)]
        assert pad["fixed"] is True
        assert pad["position"][0] == pytest.approx(x_mm / 1000.0)
        assert pad["position"][1] == pytest.approx(y_mm / 1000.0)
        assert pad["scale"][0] * pad["size"] == pytest.approx(cell_layout.PAD_SIDE_MM / 1000.0)
        assert pad["scale"][1] * pad["size"] == pytest.approx(cell_layout.PAD_SIDE_MM / 1000.0)
        assert pad["scale"][2] * pad["size"] == pytest.approx(cell_layout.PAD_THICKNESS_MM / 1000.0)
        pad_top_m = pad["position"][2] + pad["scale"][2] * pad["size"] / 2
        assert pad_top_m == pytest.approx(cell_layout.PAD_TOP_Z_MM / 1000.0)

    for color_a, color_b in itertools.combinations(cell_layout.PAD_CENTRES_MM, 2):
        ax, ay = cell_layout.PAD_CENTRES_MM[color_a]
        bx, by = cell_layout.PAD_CENTRES_MM[color_b]
        assert abs(ax - bx) >= half_pad_m * 2000.0 or abs(ay - by) >= half_pad_m * 2000.0


def test_pad_colors_match_their_block_defaults():
    props = _world_props()
    for color, default in BLOCK_COLOR_DEFAULTS.items():
        assert props[cell_layout.pad_name(color)]["color"] == default


def test_every_pooled_block_is_parked_at_its_seam_position():
    props = _world_props()
    park_positions_mm = cell_layout.park_positions_mm()
    for name, (x_mm, y_mm) in park_positions_mm.items():
        block = props[name]
        assert not block.get("fixed", False)
        assert block["position"][0] == pytest.approx(x_mm / 1000.0)
        assert block["position"][1] == pytest.approx(y_mm / 1000.0)
        assert block["position"][2] == pytest.approx(0.0305)
        assert block["size"] == 0.06


def test_the_pick_cell_roster_is_present():
    fragment = _fragment()
    world = next(c for c in fragment["components"] if c["name"] == "sim-world")
    props = {p["name"] for p in world["attributes"]["props"]}
    assert props == _expected_prop_names()

    component_names = {c["name"] for c in fragment["components"]}
    assert component_names == {
        "sim-world",
        "pick-arm",
        "pick-grip",
        "wrist-cam",
        "scene-cam",
        "side-cam",
        "block-sorter-sensor",
    }

    service_names = {(s["name"], s["api"]) for s in fragment["services"]}
    # RDK serves the builtin motion service implicitly, so the fragment may
    # carry the entry or omit it - either way the pick client's "builtin"
    # motion resource resolves.
    service_names.discard(("builtin", "rdk:service:motion"))
    assert service_names == {
        ("red-detector", "rdk:service:vision"),
        ("red-segmenter", "rdk:service:vision"),
        ("green-detector", "rdk:service:vision"),
        ("green-segmenter", "rdk:service:vision"),
        ("blue-detector", "rdk:service:vision"),
        ("blue-segmenter", "rdk:service:vision"),
        ("yellow-detector", "rdk:service:vision"),
        ("yellow-segmenter", "rdk:service:vision"),
        ("purple-detector", "rdk:service:vision"),
        ("purple-segmenter", "rdk:service:vision"),
        ("orange-detector", "rdk:service:vision"),
        ("orange-segmenter", "rdk:service:vision"),
        ("block-sorter", "rdk:service:generic"),
    }


def test_the_sorter_sensor_entry_depends_on_the_conductor_it_polls():
    fragment = _fragment()
    components = {c["name"]: c for c in fragment["components"]}
    sensor = components["block-sorter-sensor"]
    assert sensor["api"] == "rdk:component:sensor"
    assert sensor["model"] == "viam:isaac-sim-devin:sorter-sensor"
    assert sensor["attributes"]["conductor"] == "block-sorter"
    assert sensor["depends_on"] == ["block-sorter"]


def test_every_segmenter_wires_its_own_colors_detector_and_the_wrist_camera():
    fragment = _fragment()
    services = {s["name"]: s for s in fragment["services"]}
    for color in cell_layout.BLOCK_COLORS:
        segmenter = services[f"{color}-segmenter"]
        detector_name = "red-detector" if color == "red" else f"{color}-detector"
        assert segmenter["attributes"]["detector_name"] == detector_name
        assert segmenter["attributes"]["camera_name"] == "wrist-cam"


def test_the_conductor_service_entry_matches_the_phase_4_contract():
    fragment = _fragment()
    services = {s["name"]: s for s in fragment["services"]}
    conductor = services["block-sorter"]
    assert conductor["api"] == "rdk:service:generic"
    assert conductor["model"] == "viam:isaac-sim-devin:conductor"

    attrs = conductor["attributes"]
    assert attrs["world"] == "sim-world"
    assert attrs["arm"] == "pick-arm"
    assert attrs["gripper"] == "pick-grip"
    assert attrs["camera"] == "wrist-cam"
    assert attrs["side_camera"] == "side-cam"
    assert attrs["motion"] == "builtin"
    assert attrs["size_range_mm"] == [50, 80]
    assert attrs["size_range_mm"][1] <= cell_layout.MAX_BLOCK_SIZE_MM

    detectors = attrs["detectors"]
    assert set(detectors.keys()) == set(cell_layout.BLOCK_COLORS)
    for color in cell_layout.BLOCK_COLORS:
        assert detectors[color] == f"{color}-segmenter"
        assert detectors[color] in services


def test_the_nineteen_variables_ship_with_the_seam_default_values():
    found: dict[str, Any] = {}
    _collect_variables(_fragment(), found)
    assert found == EXPECTED_VARIABLE_DEFAULTS


def _hue_degrees(color: list[float]) -> float:
    import colorsys

    r, g, b = color
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def _hex_to_hue_degrees(hex_color: str) -> float:
    import colorsys

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def _hue_distance_deg(hue_a: float, hue_b: float) -> float:
    diff = abs(hue_a - hue_b) % 360
    return min(diff, 360 - diff)


# nominal-color families (PLAN.md dynamic-blocks phase-2): each block's own,
# unlit RGB default's hue must stay within its color's family band, so the
# family assignment cannot be gamed by picking a default outside it.
NOMINAL_HUE_FAMILIES_DEG: dict[str, tuple[float, float]] = {
    "red": (-10.0, 10.0),
    "orange": (20.0, 40.0),
    "yellow": (40.0, 70.0),
    "green": (90.0, 150.0),
    "blue": (200.0, 260.0),
    "purple": (260.0, 320.0),
}


@pytest.mark.parametrize("color", cell_layout.BLOCK_COLORS)
def test_each_block_colors_nominal_default_hue_stays_in_its_family(color):
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    props = {p["name"]: p for p in world["attributes"]["props"]}
    hue = _hue_degrees(props[cell_layout.pool_block_name(color, 1)]["color"])
    low, high = NOMINAL_HUE_FAMILIES_DEG[color]
    assert low <= hue <= high


@pytest.mark.parametrize("color", cell_layout.BLOCK_COLORS)
def test_each_detector_hex_hue_matches_its_measured_rendered_hue(color):
    """GPU calibration measured each color's RENDERED hue under the cell's
    lighting (`cell_layout.RENDERED_BLOCK_HUE_DEG`), which drifts off the
    nominal block-default hue. Each detector's `detect_color` must target
    that measured hue, not the nominal one, within the sampling noise."""
    fragment = _resolved_fragment()
    detector_name = "red-detector" if color == "red" else f"{color}-detector"
    detector = next(s for s in fragment["services"] if s["name"] == detector_name)
    detect_hue = _hex_to_hue_degrees(detector["attributes"]["detect_color"])
    max_sampling_noise_deg = 6
    assert _hue_distance_deg(detect_hue, cell_layout.RENDERED_BLOCK_HUE_DEG[color]) <= (
        max_sampling_noise_deg
    )


def test_every_pair_of_detector_hexes_is_separated_by_at_least_fifteen_degrees_of_hue():
    """Yellow and orange render within a few degrees of each other under the
    cell's lighting (`cell_layout.RENDERED_BLOCK_HUE_DEG`), so their bands are
    deliberately conflated: the conductor's dedup + prim-color routing tells
    them apart downstream, not hue. Every other pair keeps its margin."""
    fragment = _resolved_fragment()
    detect_hues = {
        color: _hex_to_hue_degrees(
            next(
                s
                for s in fragment["services"]
                if s["name"] == ("red-detector" if color == "red" else f"{color}-detector")
            )["attributes"]["detect_color"]
        )
        for color in cell_layout.BLOCK_COLORS
    }
    min_separation_deg = 15
    conflated_pair = {"yellow", "orange"}
    import itertools

    for color_a, color_b in itertools.combinations(cell_layout.BLOCK_COLORS, 2):
        distance = _hue_distance_deg(detect_hues[color_a], detect_hues[color_b])
        if {color_a, color_b} == conflated_pair:
            assert distance < min_separation_deg
        else:
            assert distance >= min_separation_deg


def test_blue_detectors_saturation_cutoff_clears_the_arm_silhouette_and_admits_blue_blocks():
    detector = next(s for s in _fragment()["services"] if s["name"] == "blue-detector")
    cutoff = detector["attributes"]["saturation_cutoff_pct"]
    assert cell_layout.ARM_SILHOUETTE_MAX_SATURATION < cutoff < 0.42
