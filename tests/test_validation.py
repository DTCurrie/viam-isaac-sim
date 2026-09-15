import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld


def _config(attrs: dict) -> ComponentConfig:
    return ComponentConfig(name="isaac-world", attributes=dict_to_struct(attrs))


def test_valid_props_pass():
    cfg = _config(
        {
            "props": [
                {"name": "red_block", "type": "cube", "position": [0, 0, 0]},
                {"name": "blue_block", "size": 0.1, "color": [0, 0, 1], "fixed": True},
                {"name": "table", "type": "usd", "usd_path": "omniverse://table.usd"},
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_duplicate_names_collide_after_sanitizing():
    cfg = _config({"props": [{"name": "red_block"}, {"name": "red-block"}]})
    with pytest.raises(ValueError, match="red-block"):
        IsaacWorld.validate_config(cfg)


def test_unknown_type_rejected():
    cfg = _config({"props": [{"name": "thing", "type": "sphere"}]})
    with pytest.raises(ValueError, match="type"):
        IsaacWorld.validate_config(cfg)


def test_usd_without_usd_path_rejected():
    cfg = _config({"props": [{"name": "thing", "type": "usd"}]})
    with pytest.raises(ValueError, match="usd_path"):
        IsaacWorld.validate_config(cfg)


def test_position_wrong_length_rejected():
    cfg = _config({"props": [{"name": "thing", "position": [0, 0]}]})
    with pytest.raises(ValueError, match="position"):
        IsaacWorld.validate_config(cfg)


def test_color_out_of_range_rejected():
    cfg = _config({"props": [{"name": "thing", "color": [0, 0, 2]}]})
    with pytest.raises(ValueError, match="color"):
        IsaacWorld.validate_config(cfg)


def test_size_zero_rejected():
    cfg = _config({"props": [{"name": "thing", "size": 0}]})
    with pytest.raises(ValueError, match="size"):
        IsaacWorld.validate_config(cfg)


def test_fixed_as_string_rejected():
    cfg = _config({"props": [{"name": "thing", "fixed": "yes"}]})
    with pytest.raises(ValueError, match="fixed"):
        IsaacWorld.validate_config(cfg)


def test_entry_without_name_rejected():
    cfg = _config({"props": [{"type": "cube"}]})
    with pytest.raises(ValueError, match="props\\[0\\]"):
        IsaacWorld.validate_config(cfg)


def test_valid_lighting_passes():
    cfg = _config(
        {"lighting": {"dome": {"intensity": 1000, "color": [1, 1, 1]}, "sphere_intensity": 30000}}
    )
    IsaacWorld.validate_config(cfg)


def test_lighting_dome_color_out_of_range_rejected():
    cfg = _config({"lighting": {"dome": {"color": [1.5, 0, 0]}}})
    with pytest.raises(ValueError, match="color"):
        IsaacWorld.validate_config(cfg)


def test_lighting_negative_sphere_intensity_rejected():
    cfg = _config({"lighting": {"sphere_intensity": -1}})
    with pytest.raises(ValueError, match="sphere_intensity"):
        IsaacWorld.validate_config(cfg)


def test_lighting_unknown_key_rejected():
    cfg = _config({"lighting": {"sun": 1}})
    with pytest.raises(ValueError, match="sun"):
        IsaacWorld.validate_config(cfg)


def test_lighting_not_an_object_rejected():
    cfg = _config({"lighting": "bright"})
    with pytest.raises(ValueError, match="lighting"):
        IsaacWorld.validate_config(cfg)


def test_lighting_dome_zero_intensity_rejected():
    cfg = _config({"lighting": {"dome": {"intensity": 0}}})
    with pytest.raises(ValueError, match="intensity"):
        IsaacWorld.validate_config(cfg)


def test_valid_lighting_dome_texture_passes():
    cfg = _config(
        {
            "lighting": {
                "dome": {
                    "intensity": 1000,
                    "color": [1, 1, 1],
                    "texture": "module://assets/sky.hdr",
                    "texture_format": "latlong",
                    "rotation_deg": 90,
                }
            }
        }
    )
    IsaacWorld.validate_config(cfg)


def test_lighting_dome_texture_empty_rejected():
    cfg = _config({"lighting": {"dome": {"texture": ""}}})
    with pytest.raises(ValueError, match="texture"):
        IsaacWorld.validate_config(cfg)


def test_lighting_dome_texture_non_string_rejected():
    cfg = _config({"lighting": {"dome": {"texture": 5}}})
    with pytest.raises(ValueError, match="texture"):
        IsaacWorld.validate_config(cfg)


def test_lighting_dome_texture_format_unknown_rejected():
    cfg = _config({"lighting": {"dome": {"texture_format": "spherical"}}})
    with pytest.raises(ValueError, match="texture_format"):
        IsaacWorld.validate_config(cfg)


def test_lighting_dome_rotation_deg_bool_rejected():
    cfg = _config({"lighting": {"dome": {"rotation_deg": True}}})
    with pytest.raises(ValueError, match="rotation_deg"):
        IsaacWorld.validate_config(cfg)


def test_lighting_dome_unknown_key_rejected():
    cfg = _config({"lighting": {"dome": {"glow": 1}}})
    with pytest.raises(ValueError, match="glow"):
        IsaacWorld.validate_config(cfg)


def test_valid_ground_passes():
    cfg = _config(
        {
            "ground": {
                "kind": "plane",
                "color": [0.4, 0.4, 0.4],
                "size": 50,
                "friction": 0.5,
                "restitution": 0,
            }
        }
    )
    IsaacWorld.validate_config(cfg)


def test_ground_kind_none_passes():
    cfg = _config({"ground": {"kind": "none"}})
    IsaacWorld.validate_config(cfg)


def test_ground_empty_passes():
    cfg = _config({"ground": {}})
    IsaacWorld.validate_config(cfg)


def test_ground_unknown_key_rejected():
    cfg = _config({"ground": {"texture": "foo"}})
    with pytest.raises(ValueError, match="texture"):
        IsaacWorld.validate_config(cfg)


def test_ground_bad_kind_rejected():
    cfg = _config({"ground": {"kind": "sand"}})
    with pytest.raises(ValueError, match="kind"):
        IsaacWorld.validate_config(cfg)


def test_ground_size_zero_rejected():
    cfg = _config({"ground": {"kind": "plane", "size": 0}})
    with pytest.raises(ValueError, match="size"):
        IsaacWorld.validate_config(cfg)


def test_ground_friction_negative_rejected():
    cfg = _config({"ground": {"kind": "plane", "friction": -1}})
    with pytest.raises(ValueError, match="friction"):
        IsaacWorld.validate_config(cfg)


def test_ground_restitution_out_of_range_rejected():
    cfg = _config({"ground": {"kind": "plane", "restitution": 1.5}})
    with pytest.raises(ValueError, match="restitution"):
        IsaacWorld.validate_config(cfg)


def test_ground_color_out_of_range_rejected():
    cfg = _config({"ground": {"kind": "plane", "color": [1.5, 0, 0]}})
    with pytest.raises(ValueError, match="color"):
        IsaacWorld.validate_config(cfg)


def test_ground_plane_only_key_on_non_plane_kind_rejected():
    cfg = _config({"ground": {"kind": "grid", "size": 10}})
    with pytest.raises(ValueError, match="size"):
        IsaacWorld.validate_config(cfg)


def test_ground_matte_true_passes():
    cfg = _config({"ground": {"kind": "plane", "matte": True}})
    IsaacWorld.validate_config(cfg)


def test_ground_matte_non_bool_rejected():
    cfg = _config({"ground": {"kind": "plane", "matte": "yes"}})
    with pytest.raises(ValueError, match="matte"):
        IsaacWorld.validate_config(cfg)


def test_prop_physics_pick_cell_values_pass():
    cfg = _config(
        {
            "props": [
                {
                    "name": "block",
                    "mass": 0.05,
                    "friction": 0.7,
                    "restitution": 0.0,
                    "contact_offset": 0.005,
                }
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_prop_negative_mass_rejected():
    cfg = _config({"props": [{"name": "block", "mass": -0.1}]})
    with pytest.raises(ValueError, match="mass"):
        IsaacWorld.validate_config(cfg)


def test_prop_restitution_out_of_range_rejected():
    cfg = _config({"props": [{"name": "block", "restitution": 1.5}]})
    with pytest.raises(ValueError, match="restitution"):
        IsaacWorld.validate_config(cfg)


def test_prop_rest_offset_above_contact_offset_rejected():
    cfg = _config({"props": [{"name": "block", "rest_offset": 0.01, "contact_offset": 0.005}]})
    with pytest.raises(ValueError, match="rest_offset"):
        IsaacWorld.validate_config(cfg)


def test_prop_nonnumeric_friction_rejected():
    cfg = _config({"props": [{"name": "block", "friction": "slippery"}]})
    with pytest.raises(ValueError, match="friction"):
        IsaacWorld.validate_config(cfg)


def test_visual_prop_with_scale_passes():
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "scale": [1.2, 0.8, 0.75],
                }
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_visual_prop_with_fit_true_passes():
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "fit": "true",
                }
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_visual_prop_with_fit_collider_passes():
    cfg = _config(
        {
            "props": [
                {
                    "name": "table_cube",
                    "type": "cube",
                    "size": 1.0,
                },
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "fit": {"collider": "table_cube"},
                },
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_visual_prop_missing_usd_path_rejected():
    cfg = _config({"props": [{"name": "dressed_table", "type": "visual"}]})
    with pytest.raises(ValueError, match="usd_path"):
        IsaacWorld.validate_config(cfg)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("size", 0.1),
        ("color", [1, 0, 0]),
        ("mass", 0.05),
        ("fixed", True),
        ("box_dims", [0.1, 0.1, 0.1]),
        ("friction", 0.5),
    ],
)
def test_visual_prop_rejected_key_rejected(key: str, value: object) -> None:
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    key: value,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match=key):
        IsaacWorld.validate_config(cfg)


def test_fit_on_cube_prop_rejected():
    cfg = _config({"props": [{"name": "block", "type": "cube", "fit": "true"}]})
    with pytest.raises(ValueError, match="fit"):
        IsaacWorld.validate_config(cfg)


def test_visual_prop_scale_and_fit_together_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "scale": [1.0, 1.0, 1.0],
                    "fit": "true",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="scale"):
        IsaacWorld.validate_config(cfg)


def test_visual_prop_fit_bool_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "fit": True,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="fit"):
        IsaacWorld.validate_config(cfg)


def test_visual_prop_fit_collider_unknown_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "fit": {"collider": "no_such_prop"},
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="no_such_prop"):
        IsaacWorld.validate_config(cfg)


def test_visual_prop_fit_collider_is_usd_prop_rejected():
    cfg = _config(
        {
            "props": [
                {"name": "table_usd", "type": "usd", "usd_path": "omniverse://table.usd"},
                {
                    "name": "dressed_table",
                    "type": "visual",
                    "usd_path": "data://assets/table.usd",
                    "fit": {"collider": "table_usd"},
                },
            ]
        }
    )
    with pytest.raises(ValueError, match="table_usd"):
        IsaacWorld.validate_config(cfg)


def test_kit_log_level_accepts_a_known_value():
    cfg = _config({"kit_log_level": "verbose"})
    IsaacWorld.validate_config(cfg)


def test_kit_log_level_rejects_an_unenumerated_value():
    # PY-15: unvalidated kit_log_level lets a machine-config field inject
    # arbitrary Kit settings via the flag it becomes at boot.
    cfg = _config({"kit_log_level": "warning --/some/other/setting=x"})
    with pytest.raises(ValueError, match="kit_log_level"):
        IsaacWorld.validate_config(cfg)


def test_world_frame_translation_rejected():
    # get_geometries reports prop and floor poses in world coordinates, so a
    # translated world frame would silently shift every reported geometry.
    cfg = _config({})
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 100
    with pytest.raises(ValueError, match="frame"):
        IsaacWorld.validate_config(cfg)


def test_world_frame_orientation_rejected():
    cfg = _config({})
    cfg.frame.parent = "world"
    cfg.frame.orientation.euler_angles.yaw = 0.1
    with pytest.raises(ValueError, match="frame"):
        IsaacWorld.validate_config(cfg)


def test_world_identity_frame_accepted():
    cfg = _config({})
    cfg.frame.parent = "world"
    cfg.frame.translation.x = 0
    cfg.frame.orientation.quaternion.w = 1
    IsaacWorld.validate_config(cfg)


def test_world_with_no_frame_accepted():
    cfg = _config({})
    IsaacWorld.validate_config(cfg)


def test_visual_prop_with_empty_usd_path_is_accepted_as_a_skip():
    cfg = _config(
        {
            "props": [
                {"name": "table", "type": "cube"},
                {
                    "name": "dressing",
                    "type": "visual",
                    "usd_path": "",
                    "fit": {"collider": "table"},
                },
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_visual_prop_with_non_string_usd_path_is_rejected():
    cfg = _config({"props": [{"name": "dressing", "type": "visual", "usd_path": 3}]})
    with pytest.raises(ValueError, match="must be a string"):
        IsaacWorld.validate_config(cfg)


def test_prop_named_material_passes():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": "painted_wood"}]})
    IsaacWorld.validate_config(cfg)


def test_prop_named_material_with_color_passes():
    cfg = _config(
        {
            "props": [
                {
                    "name": "block",
                    "type": "cube",
                    "color": [0.2, 0.3, 0.4],
                    "material": "painted_wood",
                }
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_prop_unknown_named_material_rejected():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": "shiny_foo"}]})
    with pytest.raises(ValueError, match="unknown material"):
        IsaacWorld.validate_config(cfg)


def test_prop_unknown_named_material_names_bundled_sets():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": "shiny_foo"}]})
    with pytest.raises(ValueError, match="painted_wood"):
        IsaacWorld.validate_config(cfg)


def test_prop_explicit_material_with_normal_map_passes():
    cfg = _config(
        {
            "props": [
                {
                    "name": "block",
                    "type": "cube",
                    "material": {"normal": "module://materials/painted_wood/normal.png"},
                }
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_prop_explicit_material_with_tint_passes():
    cfg = _config(
        {
            "props": [
                {"name": "block", "type": "cube", "material": {"tint": [0.2, 0.2, 0.2]}},
            ]
        }
    )
    IsaacWorld.validate_config(cfg)


def test_prop_explicit_material_empty_rejected():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": {}}]})
    with pytest.raises(ValueError, match="prop "):
        IsaacWorld.validate_config(cfg)


def test_prop_material_color_and_tint_both_set_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "block",
                    "type": "cube",
                    "color": [0.2, 0.2, 0.2],
                    "material": {"tint": [0.3, 0.3, 0.3]},
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="cannot both be set"):
        IsaacWorld.validate_config(cfg)


def test_prop_material_unknown_key_rejected():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": {"shine": 1}}]})
    with pytest.raises(ValueError, match="unknown key"):
        IsaacWorld.validate_config(cfg)


def test_prop_material_tint_above_one_rejected():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": {"tint": [1.5, 0, 0]}}]})
    with pytest.raises(ValueError, match="prop "):
        IsaacWorld.validate_config(cfg)


def test_prop_material_texture_scale_zero_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "block",
                    "type": "cube",
                    "material": {"tint": [0.2, 0.2, 0.2], "texture_scale": [0, 1]},
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="texture_scale"):
        IsaacWorld.validate_config(cfg)


def test_prop_material_on_usd_prop_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "block",
                    "type": "usd",
                    "usd_path": "omniverse://table.usd",
                    "material": "painted_wood",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match='only valid on a "cube" prop'):
        IsaacWorld.validate_config(cfg)


def test_prop_material_on_visual_prop_rejected():
    cfg = _config(
        {
            "props": [
                {
                    "name": "dressing",
                    "type": "visual",
                    "usd_path": "omniverse://dressing.usd",
                    "material": "painted_wood",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match='only valid on a "cube" prop'):
        IsaacWorld.validate_config(cfg)


def test_ground_named_material_passes():
    cfg = _config({"ground": {"kind": "plane", "material": "concrete_floor"}})
    IsaacWorld.validate_config(cfg)


def test_ground_material_on_grid_kind_rejected_as_plane_only():
    cfg = _config({"ground": {"kind": "grid", "material": "concrete_floor"}})
    with pytest.raises(ValueError, match="material"):
        IsaacWorld.validate_config(cfg)


def test_ground_color_beside_material_tint_rejected():
    cfg = _config(
        {
            "ground": {
                "kind": "plane",
                "color": [0.5, 0.5, 0.5],
                "material": {"tint": [1, 1, 1]},
            }
        }
    )
    with pytest.raises(ValueError, match="ground"):
        IsaacWorld.validate_config(cfg)


def test_prop_material_errors_are_labeled_prop():
    cfg = _config({"props": [{"name": "block", "type": "cube", "material": {}}]})
    with pytest.raises(ValueError) as excinfo:
        IsaacWorld.validate_config(cfg)
    assert str(excinfo.value).startswith("prop ")


def test_ground_material_errors_are_labeled_ground():
    cfg = _config(
        {
            "ground": {
                "kind": "plane",
                "color": [0.5, 0.5, 0.5],
                "material": {"tint": [1, 1, 1]},
            }
        }
    )
    with pytest.raises(ValueError) as excinfo:
        IsaacWorld.validate_config(cfg)
    assert str(excinfo.value).startswith("ground")
