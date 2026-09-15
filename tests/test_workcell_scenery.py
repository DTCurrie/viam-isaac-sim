# Every recorded payload below is transcribed by hand from two real sources: the
# get_visuals wire shape as apps/workcell/index.html's meshFromVisual/applyOV
# deserialize it (nested pose {x, y, z, o_x, o_y, o_z, theta} and nested
# color {r, g, b, opacity}), and the axis-keyed *_dims_mm and colour objects in
# the user-supplied fragment at
# scratchpad/workcell/upstream-fragment.json (hmi-cabinet.body_dims_mm,
# pick-station.infeed_box_color, caution-tape.color, pack-sequencer.box_color).
# Neither source is a live get_visuals capture, so treat this as a stand-in
# for the real wire, not a fixture recorded off a running component.

import math

import pytest

from isaac_module.spatial import ov_to_quat
from isaac_module.workcell_scenery import (
    SceneryPrimitive,
    collider_prop,
    derived_collider,
    parse_geometries,
    parse_visuals,
    render_prop,
    scenery_props,
)

BOX_VISUALS = {
    "visuals": [
        {
            "type": "box",
            "label": "base",
            "pose": {
                "x": 10.0,
                "y": 20.0,
                "z": 30.0,
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": 1.0,
                "theta": 0.0,
            },
            "dims_mm": {"x": 400.0, "y": 300.0, "z": 50.0},
            "color": {"r": 255.0, "g": 0.0, "b": 0.0, "opacity": 1.0},
        }
    ]
}

CAPSULE_VISUALS = {
    "visuals": [
        {
            "type": "capsule",
            "label": "column",
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 500.0,
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": 1.0,
                "theta": 0.0,
            },
            "radius_mm": 40.0,
            "length_mm": 900.0,
            "color": {"r": 120.0, "g": 120.0, "b": 120.0, "opacity": 1.0},
        }
    ]
}

MESH_VISUALS = {
    "visuals": [
        {
            "type": "mesh",
            "label": "shroud",
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": 1.0,
                "theta": 0.0,
            },
            "mesh_path": "assets/pedestal_shroud.usd",
        }
    ]
}

UNKNOWN_TYPE_VISUALS = {
    "visuals": [
        {
            "type": "frame",
            "label": "tcp",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        {
            "type": "box",
            "label": "base",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
            "dims_mm": {"x": 100.0, "y": 100.0, "z": 100.0},
        },
    ]
}

PALLET_GEOMETRIES = [
    {"label": "pallet-body", "box_dims_mm": {"x": 1200.0, "y": 800.0, "z": 150.0}},
]

PEDESTAL_ATTRS = {"height_mm": 150.0, "diameter_mm": 300.0}


def test_parse_visuals_box_converts_mm_to_m_and_builds_quaternion():
    [primitive] = parse_visuals(BOX_VISUALS)
    assert primitive.kind == "box"
    assert primitive.label == "base"
    assert primitive.position_m == pytest.approx((0.01, 0.02, 0.03))
    assert primitive.dims_m == pytest.approx((0.4, 0.3, 0.05))
    assert primitive.color == pytest.approx((1.0, 0.0, 0.0))
    expected_quat = ov_to_quat(0.0, 0.0, 1.0, 0.0)
    assert primitive.orientation_wxyz == pytest.approx(expected_quat)


def test_parse_visuals_capsule_converts_radius_and_length():
    [primitive] = parse_visuals(CAPSULE_VISUALS)
    assert primitive.kind == "capsule"
    assert primitive.radius_m == pytest.approx(0.04)
    assert primitive.length_m == pytest.approx(0.9)
    assert primitive.dims_m is None


def test_parse_visuals_mesh_carries_mesh_path():
    [primitive] = parse_visuals(MESH_VISUALS)
    assert primitive.kind == "mesh"
    assert primitive.mesh_path == "assets/pedestal_shroud.usd"
    assert primitive.dims_m is None
    assert primitive.radius_m is None


def test_parse_visuals_applies_theta_rotation():
    payload = {
        "visuals": [
            {
                "type": "box",
                "label": "base",
                "pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "o_x": 0.0,
                    "o_y": 0.0,
                    "o_z": 1.0,
                    "theta": 90.0,
                },
                "dims_mm": {"x": 100.0, "y": 100.0, "z": 100.0},
            }
        ]
    }
    [primitive] = parse_visuals(payload)
    expected = ov_to_quat(0.0, 0.0, 1.0, math.radians(90.0))
    assert primitive.orientation_wxyz == pytest.approx(expected)


def test_parse_visuals_skips_unknown_primitive_type():
    primitives = parse_visuals(UNKNOWN_TYPE_VISUALS)
    assert len(primitives) == 1
    assert primitives[0].label == "base"


def test_parse_geometries_converts_box_dims_mm_to_m():
    [primitive] = parse_geometries(PALLET_GEOMETRIES)
    assert primitive.kind == "box"
    assert primitive.label == "pallet-body"
    assert primitive.dims_m == pytest.approx((1.2, 0.8, 0.15))
    assert primitive.position_m == (0.0, 0.0, 0.0)
    assert primitive.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_derived_collider_robot_pedestal_from_height_and_diameter():
    primitive = derived_collider("robot-pedestal", PEDESTAL_ATTRS)
    assert primitive is not None
    assert primitive.kind == "box"
    assert primitive.dims_m == pytest.approx((0.3, 0.3, 0.15))
    assert primitive.position_m == pytest.approx((0.0, 0.0, 0.075))


def test_derived_collider_none_for_a_component_with_neither_source():
    # scan-tunnel offers no GetGeometries and is not in DERIVED_COLLIDER_MODELS,
    # so its collider is None: render-only until the GPU checklist decides
    # otherwise.
    assert derived_collider("scan-tunnel", {}) is None


def test_collider_prop_is_fixed_uncoloured_cube_sized_by_dims():
    primitive = SceneryPrimitive(
        kind="box", label="pedestal-collider", dims_m=(0.3, 0.3, 0.15), position_m=(0.0, 0.0, 0.075)
    )
    prop = collider_prop(primitive, name="arm-pedestal-collider")
    assert prop["name"] == "arm-pedestal-collider"
    assert prop["type"] == "cube"
    assert prop["fixed"] is True
    assert prop["scale"] == pytest.approx((0.3, 0.3, 0.15))
    assert prop["size"] == 1.0
    assert "color" not in prop


def test_collider_prop_approximates_a_capsule_by_its_bounding_box():
    primitive = SceneryPrimitive(kind="capsule", label="column", radius_m=0.04, length_m=0.9)
    prop = collider_prop(primitive, name="pedestal-column-collider")
    assert prop["scale"] == pytest.approx((0.08, 0.08, 0.9))


def test_render_prop_mesh_becomes_a_non_colliding_visual_prop():
    [primitive] = parse_visuals(MESH_VISUALS)
    prop = render_prop(primitive, name="pedestal-shroud")
    assert prop["type"] == "visual"
    assert prop["usd_path"] == "assets/pedestal_shroud.usd"
    assert prop["fit"] == "true"
    assert prop["collision"] is False
    assert "size" not in prop


def test_render_prop_box_becomes_a_coloured_non_colliding_cube():
    [primitive] = parse_visuals(BOX_VISUALS)
    prop = render_prop(primitive, name="pedestal-base")
    assert prop["type"] == "cube"
    assert prop["color"] == pytest.approx((1.0, 0.0, 0.0))
    assert prop["scale"] == pytest.approx((0.4, 0.3, 0.05))
    assert prop["collision"] is False


def test_scenery_props_orders_colliders_before_render_and_names_by_label():
    props = scenery_props(
        "pallet1",
        "pallet",
        visuals=BOX_VISUALS,
        geometries=PALLET_GEOMETRIES,
        attrs={},
    )
    assert [p["name"] for p in props] == ["pallet1-pallet-body", "pallet1-base"]
    assert props[0]["type"] == "cube"
    assert props[0]["fixed"] is True
    assert "collision" not in props[0]
    assert props[1]["collision"] is False


def test_scenery_props_derives_pedestal_collider_when_not_shaped():
    props = scenery_props(
        "arm-pedestal",
        "robot-pedestal",
        visuals=BOX_VISUALS,
        geometries=[],
        attrs=PEDESTAL_ATTRS,
    )
    names = [p["name"] for p in props]
    assert names[0] == "arm-pedestal-robot-pedestal-collider"
    assert names[1] == "arm-pedestal-base"


def test_scenery_props_has_no_collider_for_a_component_with_neither_source():
    props = scenery_props(
        "tunnel1",
        "scan-tunnel",
        visuals=BOX_VISUALS,
        geometries=[],
        attrs={},
    )
    # only the render prop from get_visuals, no collider prop up front
    assert [p["name"] for p in props] == ["tunnel1-base"]
    assert props[0]["type"] == "cube"
    assert props[0]["collision"] is False
