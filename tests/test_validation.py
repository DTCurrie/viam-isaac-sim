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
