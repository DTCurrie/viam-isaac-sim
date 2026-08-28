import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld


def _config(attrs: dict) -> ComponentConfig:
    return ComponentConfig(name="sim-world", attributes=dict_to_struct(attrs))


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
