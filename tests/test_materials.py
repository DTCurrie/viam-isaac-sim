import json
import logging
import sys
import types
from typing import Any

import pytest

from isaac_module.materials import (
    BUNDLED_MATERIAL_NAMES,
    OMNIPBR_INPUT_SDF_TYPES,
    MaterialSet,
    build_material,
    load_manifest,
    material_inputs,
    material_prim_path,
    material_spec,
    missing_map_paths,
    named_material_names,
    prop_display_color,
)

# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path, data: dict) -> Any:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_load_manifest_round_trips_a_valid_two_set_manifest(tmp_path):
    path = _write_manifest(
        tmp_path,
        {
            "wood": {
                "maps": {"albedo": "albedo.png", "normal": "normal.png"},
                "texture_scale": [0.5, 0.5],
                "source": "https://example.com/wood",
                "license": "CC0",
            },
            "mat": {
                "maps": {"roughness": "roughness.png"},
            },
        },
    )
    sets = load_manifest(path)
    assert set(sets) == {"wood", "mat"}
    wood = sets["wood"]
    assert isinstance(wood, MaterialSet)
    assert wood.maps == {
        "albedo": "module://materials/wood/albedo.png",
        "normal": "module://materials/wood/normal.png",
    }
    assert wood.texture_scale == (0.5, 0.5)
    assert wood.source == "https://example.com/wood"
    assert wood.license == "CC0"
    mat = sets["mat"]
    assert mat.maps == {"roughness": "module://materials/mat/roughness.png"}
    assert mat.texture_scale is None
    assert mat.source == ""
    assert mat.license == ""


def test_load_manifest_rejects_unknown_map_key(tmp_path):
    path = _write_manifest(tmp_path, {"wood": {"maps": {"bogus": "a.png"}}})
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_file_name_with_slash(tmp_path):
    path = _write_manifest(tmp_path, {"wood": {"maps": {"albedo": "sub/a.png"}}})
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_dotdot_file_name(tmp_path):
    path = _write_manifest(tmp_path, {"wood": {"maps": {"albedo": ".."}}})
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_empty_maps(tmp_path):
    path = _write_manifest(tmp_path, {"wood": {"maps": {}}})
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_unknown_set_key(tmp_path):
    path = _write_manifest(tmp_path, {"wood": {"maps": {"albedo": "a.png"}, "bogus_key": True}})
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_non_object_set(tmp_path):
    path = _write_manifest(tmp_path, {"wood": "not an object"})
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_bad_texture_scale(tmp_path):
    path = _write_manifest(
        tmp_path, {"wood": {"maps": {"albedo": "a.png"}, "texture_scale": [0.0, 1.0]}}
    )
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


def test_load_manifest_rejects_texture_scale_wrong_length(tmp_path):
    path = _write_manifest(
        tmp_path, {"wood": {"maps": {"albedo": "a.png"}, "texture_scale": [1.0]}}
    )
    with pytest.raises(ValueError, match="wood"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# named_material_names
# ---------------------------------------------------------------------------


def test_named_material_names_is_sorted_for_an_explicit_manifest():
    manifest = {
        "zebra": MaterialSet(name="zebra", maps={}, texture_scale=None, source="", license=""),
        "apple": MaterialSet(name="apple", maps={}, texture_scale=None, source="", license=""),
    }
    assert named_material_names(manifest) == ("apple", "zebra")


def test_named_material_names_matches_bundled_names():
    assert named_material_names() == tuple(sorted(BUNDLED_MATERIAL_NAMES))


# ---------------------------------------------------------------------------
# material_spec: named form
# ---------------------------------------------------------------------------


def _manifest():
    return {
        "wood": MaterialSet(
            name="wood",
            maps={
                "albedo": "module://materials/wood/albedo.png",
                "roughness": "module://materials/wood/roughness.png",
            },
            texture_scale=(0.5, 0.5),
            source="src",
            license="CC0",
        ),
        "mat": MaterialSet(
            name="mat",
            maps={"roughness": "module://materials/mat/roughness.png"},
            texture_scale=None,
            source="",
            license="",
        ),
    }


def _fake_resolve(value: str) -> str:
    return f"resolved:{value}"


def test_material_spec_named_form_resolves_maps_and_tint():
    manifest = _manifest()
    spec = material_spec("wood", color=[0.1, 0.2, 0.3], manifest=manifest, resolve=_fake_resolve)
    assert spec["name"] == "wood"
    assert spec["albedo"] == "resolved:module://materials/wood/albedo.png"
    assert spec["roughness"] == "resolved:module://materials/wood/roughness.png"
    assert spec["normal"] is None
    assert spec["metallic"] is None
    assert spec["tint"] == (0.1, 0.2, 0.3)
    assert spec["texture_scale"] == (0.5, 0.5)


def test_material_spec_named_form_default_texture_scale_flows_in_without_color():
    manifest = _manifest()
    spec = material_spec("mat", color=None, manifest=manifest, resolve=_fake_resolve)
    assert spec["texture_scale"] is None
    assert spec["tint"] is None


def test_material_spec_named_form_unknown_name_lists_bundled_names():
    manifest = _manifest()
    with pytest.raises(ValueError, match=r"wood.*mat|mat.*wood"):
        material_spec("bogus", color=None, manifest=manifest, resolve=_fake_resolve)


# ---------------------------------------------------------------------------
# material_spec: explicit form
# ---------------------------------------------------------------------------


def test_material_spec_explicit_tint_alone():
    spec = material_spec({"tint": [0.1, 0.2, 0.3]}, color=None, resolve=_fake_resolve)
    assert spec["tint"] == (0.1, 0.2, 0.3)
    assert spec["name"] is None


def test_material_spec_explicit_color_alone_becomes_tint():
    spec = material_spec({"albedo": "a.png"}, color=[0.4, 0.5, 0.6], resolve=_fake_resolve)
    assert spec["tint"] == (0.4, 0.5, 0.6)
    assert spec["albedo"] == "resolved:a.png"


def test_material_spec_explicit_color_beside_tint_rejected():
    with pytest.raises(ValueError, match="tint"):
        material_spec({"tint": [0.1, 0.2, 0.3]}, color=[0.4, 0.5, 0.6], resolve=_fake_resolve)


def test_material_spec_explicit_empty_object_rejected():
    with pytest.raises(ValueError):
        material_spec({}, color=None, resolve=_fake_resolve)


def test_material_spec_explicit_unknown_key_rejected():
    with pytest.raises(ValueError, match="bogus"):
        material_spec({"bogus": 1}, color=None, resolve=_fake_resolve)


def test_material_spec_explicit_non_string_map_rejected():
    with pytest.raises(ValueError, match="albedo"):
        material_spec({"albedo": 5}, color=None, resolve=_fake_resolve)


def test_material_spec_explicit_tint_out_of_range_rejected():
    with pytest.raises(ValueError):
        material_spec({"tint": [1.5, 0.0, 0.0]}, color=None, resolve=_fake_resolve)


def test_material_spec_explicit_texture_scale_with_zero_rejected():
    with pytest.raises(ValueError):
        material_spec(
            {"tint": [0.1, 0.2, 0.3], "texture_scale": [0.0, 1.0]},
            color=None,
            resolve=_fake_resolve,
        )


def test_material_spec_explicit_data_scheme_map_resolves_through_resolve():
    spec = material_spec({"albedo": "data://foo.png"}, color=None, resolve=_fake_resolve)
    assert spec["albedo"] == "resolved:data://foo.png"


# ---------------------------------------------------------------------------
# material_inputs
# ---------------------------------------------------------------------------


def test_material_inputs_named_set_with_color_and_no_albedo():
    spec = {
        "name": "mat",
        "albedo": None,
        "normal": "n.png",
        "roughness": "r.png",
        "metallic": None,
        "tint": (0.1, 0.2, 0.3),
        "texture_scale": None,
    }
    inputs = material_inputs(spec)
    assert inputs["diffuse_color_constant"] == (0.1, 0.2, 0.3)
    assert "diffuse_texture" not in inputs
    assert "diffuse_tint" not in inputs


def test_material_inputs_albedo_plus_tint_yields_diffuse_tint():
    spec = {
        "name": None,
        "albedo": "a.png",
        "normal": None,
        "roughness": None,
        "metallic": None,
        "tint": (0.1, 0.2, 0.3),
        "texture_scale": None,
    }
    inputs = material_inputs(spec)
    assert inputs["diffuse_tint"] == (0.1, 0.2, 0.3)
    assert "diffuse_color_constant" not in inputs
    assert inputs["diffuse_texture"] == "a.png"


def test_material_inputs_roughness_and_metallic_bring_influence_at_one():
    spec = {
        "name": None,
        "albedo": None,
        "normal": "n.png",
        "roughness": "r.png",
        "metallic": "m.png",
        "tint": None,
        "texture_scale": None,
    }
    inputs = material_inputs(spec)
    assert inputs["reflection_roughness_texture_influence"] == 1.0
    assert inputs["metallic_texture_influence"] == 1.0
    assert "normalmap_texture" in inputs
    assert "diffuse_texture" not in inputs
    influence_keys = {k for k in inputs if k.endswith("_influence")}
    assert influence_keys == {
        "reflection_roughness_texture_influence",
        "metallic_texture_influence",
    }


def test_material_inputs_texture_scale_is_a_two_tuple():
    spec = {
        "name": None,
        "albedo": None,
        "normal": None,
        "roughness": None,
        "metallic": None,
        "tint": None,
        "texture_scale": (0.5, 0.25),
    }
    inputs = material_inputs(spec)
    assert inputs["texture_scale"] == (0.5, 0.25)


def test_material_inputs_all_none_yields_empty():
    spec = {
        "name": None,
        "albedo": None,
        "normal": None,
        "roughness": None,
        "metallic": None,
        "tint": None,
        "texture_scale": None,
    }
    assert material_inputs(spec) == {}


def test_material_inputs_every_key_is_a_known_sdf_type_key():
    spec = {
        "name": None,
        "albedo": "a.png",
        "normal": "n.png",
        "roughness": "r.png",
        "metallic": "m.png",
        "tint": (0.1, 0.2, 0.3),
        "texture_scale": (1.0, 1.0),
    }
    inputs = material_inputs(spec)
    assert set(inputs).issubset(set(OMNIPBR_INPUT_SDF_TYPES))
    assert inputs  # non-trivial


# ---------------------------------------------------------------------------
# prop_display_color, material_prim_path, missing_map_paths
# ---------------------------------------------------------------------------


def test_prop_display_color_color_wins_over_material_tint():
    prop = {"color": [0.1, 0.2, 0.3], "material": {"tint": [0.9, 0.9, 0.9]}}
    assert prop_display_color(prop) == (0.1, 0.2, 0.3)


def test_prop_display_color_tint_alone():
    prop = {"material": {"tint": [0.4, 0.5, 0.6]}}
    assert prop_display_color(prop) == (0.4, 0.5, 0.6)


def test_prop_display_color_neither_returns_none():
    assert prop_display_color({}) is None


def test_prop_display_color_named_string_material_with_no_color_returns_none():
    assert prop_display_color({"material": "painted_wood"}) is None


def test_material_prim_path_shape():
    assert material_prim_path("block1") == "/World/Looks/block1_material"


def test_missing_map_paths_lists_missing_local_path():
    spec = {"albedo": "/does/not/exist.png"}
    assert missing_map_paths(spec, exists=lambda p: False) == ["/does/not/exist.png"]


def test_missing_map_paths_excludes_existing_local_path():
    spec = {"albedo": "/exists.png"}
    assert missing_map_paths(spec, exists=lambda p: True) == []


def test_missing_map_paths_never_lists_remote_paths():
    spec = {
        "albedo": "https://example.com/a.png",
        "normal": "omniverse://server/n.png",
    }
    assert missing_map_paths(spec, exists=lambda p: False) == []


# ---------------------------------------------------------------------------
# build_material, with a fake pxr and a fake OmniPBR
# ---------------------------------------------------------------------------


class _FakeVec3f:
    def __init__(self, x, y, z):
        self.values = (x, y, z)


class _FakeVec2f:
    def __init__(self, u, v):
        self.values = (u, v)


class _FakeValueTypeNames:
    Asset = "Asset"
    Float = "Float"
    Color3f = "Color3f"
    Float2 = "Float2"


class _FakeSdf:
    ValueTypeNames = _FakeValueTypeNames()


class _FakeGf:
    Vec3f = _FakeVec3f
    Vec2f = _FakeVec2f


class _FakeShaderInput:
    def __init__(self, recorder, input_name, sdf_type):
        self._recorder = recorder
        self._input_name = input_name
        self._sdf_type = sdf_type

    def Set(self, value):
        self._recorder.append((self._input_name, self._sdf_type, value))


class _FakeShader:
    def __init__(self):
        self.calls: list[tuple[str, str, Any]] = []

    def CreateInput(self, input_name, sdf_type):
        return _FakeShaderInput(self.calls, input_name, sdf_type)


class _FakeMaterial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shader = _FakeShader()
        self.shaders_list = [self.shader]


class _RaisingOmniPBR:
    def __init__(self, **kwargs):
        raise RuntimeError("boom")


class _FakeIsaacNamespace:
    def __init__(self, omnipbr):
        self.OmniPBR = omnipbr


@pytest.fixture(autouse=True)
def _fake_pxr(monkeypatch):
    fake_module = types.SimpleNamespace(Gf=_FakeGf, Sdf=_FakeSdf)
    monkeypatch.setitem(sys.modules, "pxr", fake_module)


def _full_spec() -> dict:
    return {
        "name": None,
        "albedo": None,
        "normal": "/textures/normal.png",
        "roughness": "/textures/roughness.png",
        "metallic": None,
        "tint": (0.2, 0.4, 0.6),
        "texture_scale": (0.5, 0.5),
    }


def test_build_material_happy_path_authors_every_input(tmp_path):
    isaac = _FakeIsaacNamespace(_FakeMaterial)
    spec = _full_spec()
    material = build_material(isaac, name="block1", spec=spec, exists=lambda p: True)
    assert material is not None
    assert material.kwargs == {
        "prim_path": "/World/Looks/block1_material",
        "name": "block1_material",
    }
    expected_inputs = material_inputs(spec)
    authored = {
        input_name: (sdf_type, value) for input_name, sdf_type, value in material.shader.calls
    }
    assert set(authored) == set(expected_inputs)
    for input_name, (sdf_type, _) in authored.items():
        assert sdf_type == OMNIPBR_INPUT_SDF_TYPES[input_name]
    _color_sdf_type, color_value = authored["diffuse_color_constant"]
    assert isinstance(color_value, _FakeVec3f)
    assert color_value.values == (0.2, 0.4, 0.6)
    _scale_sdf_type, scale_value = authored["texture_scale"]
    assert isinstance(scale_value, _FakeVec2f)
    assert scale_value.values == (0.5, 0.5)


def test_build_material_missing_local_map_returns_none_and_never_constructs(caplog):
    isaac = _FakeIsaacNamespace(_FakeMaterial)
    spec = _full_spec()
    with caplog.at_level(logging.WARNING):
        result = build_material(isaac, name="block1", spec=spec, exists=lambda p: False)
    assert result is None
    assert any("/textures/normal.png" in record.message for record in caplog.records)


def test_build_material_no_omnipbr_returns_none():
    isaac = _FakeIsaacNamespace(None)
    spec = _full_spec()
    assert build_material(isaac, name="block1", spec=spec, exists=lambda p: True) is None


def test_build_material_constructor_exception_returns_none(caplog):
    isaac = _FakeIsaacNamespace(_RaisingOmniPBR)
    spec = _full_spec()
    with caplog.at_level(logging.ERROR):
        result = build_material(isaac, name="block1", spec=spec, exists=lambda p: True)
    assert result is None
    assert any("block1" in record.message for record in caplog.records)
