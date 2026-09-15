import threading

import pytest

from isaac_module.materials import MATERIAL_SPEC_KEY
from isaac_module.sim_manager import GROUND_PLANE_NAME, SimConfig, SimManager
from test_world_handle import _cube, _mock_handle

# ----------------------------------------------------------------------
# mock path
# ----------------------------------------------------------------------


def test_mock_register_records_material_spec_for_explicit_material():
    handle = _mock_handle(
        [_cube("a", material={"tint": [0.2, 0.4, 0.6]}, position=[0.0, 0.0, 0.03])]
    )
    entry = handle.registry()["a"]
    assert entry[MATERIAL_SPEC_KEY]["tint"] == pytest.approx((0.2, 0.4, 0.6))
    assert entry["spawn"]["material"] == {"tint": [0.2, 0.4, 0.6]}


def test_mock_register_no_material_records_none():
    handle = _mock_handle([_cube("a", color=[1.0, 0.0, 0.0], position=[0.0, 0.0, 0.03])])
    entry = handle.registry()["a"]
    assert entry[MATERIAL_SPEC_KEY] is None


def test_mock_randomize_keeps_material_spec_and_raw_config_while_size_changes():
    handle = _mock_handle(
        [_cube("a", material={"tint": [0.2, 0.4, 0.6]}, position=[0.0, 0.0, 0.03])]
    )
    entry = handle.registry()["a"]
    original_spec = entry[MATERIAL_SPEC_KEY]
    original_material_config = entry["spawn"]["material"]
    original_size = entry["spawn"]["size"]

    handle.randomize_props(
        ["a"],
        region=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.0)),
        seed=1,
        size_range_m={"a": (0.02, 0.02)},
    )

    entry = handle.registry()["a"]
    assert entry[MATERIAL_SPEC_KEY] == original_spec
    assert entry["spawn"]["material"] == original_material_config
    assert entry["spawn"]["size"] != original_size


def test_mock_prop_geometries_color_is_tint_when_material_has_no_color():
    handle = _mock_handle(
        [_cube("a", material={"tint": [0.1, 0.2, 0.3]}, position=[0.0, 0.0, 0.03])]
    )
    geometries = {g.name: g for g in handle.prop_geometries()}
    assert geometries["a"].color == pytest.approx((0.1, 0.2, 0.3))


def test_mock_prop_geometries_color_is_plain_color_for_uncolored_prop():
    handle = _mock_handle([_cube("a", color=[0.9, 0.8, 0.7], position=[0.0, 0.0, 0.03])])
    geometries = {g.name: g for g in handle.prop_geometries()}
    assert geometries["a"].color == pytest.approx((0.9, 0.8, 0.7))


def test_mock_named_set_with_color_records_spec_tint_equal_to_color():
    handle = _mock_handle(
        [_cube("a", material="painted_wood", color=[0.5, 0.5, 0.5], position=[0.0, 0.0, 0.03])]
    )
    entry = handle.registry()["a"]
    assert entry[MATERIAL_SPEC_KEY]["name"] == "painted_wood"
    assert entry[MATERIAL_SPEC_KEY]["tint"] == pytest.approx((0.5, 0.5, 0.5))


# ----------------------------------------------------------------------
# isaac path
# ----------------------------------------------------------------------


def _isaac_manager() -> SimManager:
    from test_world_handle import _FakeIsaacNamespace, _FakeWorld

    manager = SimManager()
    manager.mock = False
    manager.world = _FakeWorld()
    manager._isaac = _FakeIsaacNamespace()
    manager._booted.set()
    manager._sim_thread_id = threading.get_ident()
    return manager


def test_isaac_spawn_with_material_built_uses_visual_material_kwarg(monkeypatch):
    sentinel = object()
    calls: list[tuple[str, dict]] = []

    def fake_build_material(isaac, *, name, spec):
        calls.append((name, dict(spec)))
        return sentinel

    monkeypatch.setattr("isaac_module.sim_manager.build_material", fake_build_material)
    manager = _isaac_manager()
    manager._spawn_prop(_cube("a", material={"tint": [0.2, 0.4, 0.6]}, position=[0.0, 0.0, 0.03]))
    obj = manager.world.scene.get_object("a")
    assert obj.kwargs["visual_material"] is sentinel
    assert "color" not in obj.kwargs
    assert len(calls) == 1
    called_name, called_spec = calls[0]
    assert called_name == "a"
    assert called_spec["tint"] == pytest.approx((0.2, 0.4, 0.6))
    assert manager._prop_specs["a"][MATERIAL_SPEC_KEY] == called_spec


def test_isaac_spawn_with_material_build_failure_falls_back_to_flat_color(monkeypatch):
    monkeypatch.setattr(
        "isaac_module.sim_manager.build_material", lambda isaac, *, name, spec: None
    )
    manager = _isaac_manager()
    manager._spawn_prop(_cube("a", material={"tint": [0.2, 0.4, 0.6]}, position=[0.0, 0.0, 0.03]))
    obj = manager.world.scene.get_object("a")
    assert obj.kwargs["color"].tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert "visual_material" not in obj.kwargs


def test_isaac_prop_specs_material_spec_recorded_and_unchanged_after_randomize(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "isaac_module.sim_manager.build_material", lambda isaac, *, name, spec: sentinel
    )
    manager = _isaac_manager()
    manager._spawn_prop(_cube("a", material={"tint": [0.2, 0.4, 0.6]}, position=[0.0, 0.0, 0.03]))
    from isaac_module.sim_manager import IsaacWorldHandle

    handle = IsaacWorldHandle(manager)
    original_spec = manager._prop_specs["a"][MATERIAL_SPEC_KEY]

    handle.randomize_props(
        ["a"],
        region=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.0)),
        seed=1,
        size_range_m={"a": (0.02, 0.02)},
    )

    assert manager._prop_specs["a"][MATERIAL_SPEC_KEY] == original_spec


def test_isaac_prop_geometries_color_is_tint(monkeypatch):
    monkeypatch.setattr(
        "isaac_module.sim_manager.build_material", lambda isaac, *, name, spec: object()
    )
    manager = _isaac_manager()
    manager._spawn_prop(_cube("a", material={"tint": [0.2, 0.4, 0.6]}, position=[0.0, 0.0, 0.03]))
    from isaac_module.sim_manager import IsaacWorldHandle

    handle = IsaacWorldHandle(manager)
    geometries = {g.name: g for g in handle.prop_geometries()}
    assert geometries["a"].color == pytest.approx((0.2, 0.4, 0.6))


# ----------------------------------------------------------------------
# ground plane
# ----------------------------------------------------------------------


class _FakeGroundObject:
    def __init__(self) -> None:
        self.applied_material = None

    def apply_visual_material(self, material) -> None:
        self.applied_material = material


class _FakeGroundScene:
    def __init__(self) -> None:
        self.add_ground_plane_kwargs: dict | None = None
        self._ground_object = _FakeGroundObject()

    def add_ground_plane(self, **kwargs) -> None:
        self.add_ground_plane_kwargs = kwargs

    def get_object(self, name):
        assert name == GROUND_PLANE_NAME
        return self._ground_object

    def add_default_ground_plane(self) -> None:
        pass


class _FakeGroundWorld:
    def __init__(self) -> None:
        self.scene = _FakeGroundScene()


def _ground_manager() -> SimManager:
    manager = SimManager()
    manager.mock = False
    manager.world = _FakeGroundWorld()
    manager._isaac = object()
    return manager


def test_add_ground_applies_built_material_and_kwargs_carry_no_material_key(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "isaac_module.sim_manager.build_material", lambda isaac, *, name, spec: sentinel
    )
    manager = _ground_manager()
    cfg = SimConfig(ground={"kind": "plane", "material": {"tint": [0.3, 0.3, 0.3]}})
    manager._add_ground(cfg)
    assert manager.world.scene._ground_object.applied_material is sentinel
    assert "material" not in manager.world.scene.add_ground_plane_kwargs


def test_add_ground_no_material_built_applies_nothing_without_raising(monkeypatch):
    monkeypatch.setattr(
        "isaac_module.sim_manager.build_material", lambda isaac, *, name, spec: None
    )
    manager = _ground_manager()
    cfg = SimConfig(ground={"kind": "plane", "material": {"tint": [0.3, 0.3, 0.3]}})
    manager._add_ground(cfg)
    assert manager.world.scene._ground_object.applied_material is None
