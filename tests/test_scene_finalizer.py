from types import SimpleNamespace
from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.utils import dict_to_struct

import isaac_module.models.scene_finalizer as finalizer_model
from isaac_module.models.scene_finalizer import IsaacSceneFinalizer


class _FakeWorkcellComponent:
    """Stands in for a viam:workcell-components dependency: answers
    get_schema, get_status, get_attributes and get_visuals the way the real
    module does, never GetGeometries, which is enough to exercise the
    wiring without pinning workcell_client's own behaviour twice."""

    def __init__(self, model: str) -> None:
        self._model = model

    async def do_command(self, command: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if "get_schema" in command:
            return {"schema": [{"key": "width_mm", "type": "number"}]}
        if "get_status" in command:
            return {"model": self._model}
        if "get_attributes" in command:
            return {"pose": {"x": 200.0, "y": 500.0, "z": 200.0}}
        return {"visuals": []}

    async def get_geometries(self, **kwargs: Any) -> list[Any]:
        return []


class _FakeWorldComponent:
    """Stands in for this cell's own world component: a generic dependency
    with no get_schema verb, so it is never treated as workcell scenery."""

    async def do_command(self, command: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("unknown command")


def _config(name: str, attrs: dict, depends_on: list[str] | None = None) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs), depends_on=depends_on or [])


def _sim_manager_stub(calls: list) -> SimpleNamespace:
    return SimpleNamespace(
        materialise_components=lambda components: calls.append(("materialise", components)),
        finalize_scene=lambda: calls.append(("finalize", None)),
    )


def test_new_finalizes_the_scene_once_and_carries_the_configured_name(monkeypatch):
    calls: list = []
    monkeypatch.setattr(finalizer_model.SimManager, "get", lambda: _sim_manager_stub(calls))

    finalizer = IsaacSceneFinalizer.new(_config("scene-ready", {}, depends_on=["isaac-world"]), {})

    assert finalizer.name == "scene-ready"
    assert [call[0] for call in calls] == ["materialise", "finalize"]


def test_reconfigure_finalizes_the_scene_again(monkeypatch):
    calls: list = []
    monkeypatch.setattr(finalizer_model.SimManager, "get", lambda: _sim_manager_stub(calls))

    finalizer = IsaacSceneFinalizer.new(_config("scene-ready", {}, depends_on=["isaac-world"]), {})
    finalizer.reconfigure(_config("scene-ready", {}, depends_on=["isaac-world"]), {})

    assert [call[0] for call in calls] == ["materialise", "finalize", "materialise", "finalize"]


def test_validate_config_returns_depends_on_as_required_dependencies():
    config = _config("scene-ready", {}, depends_on=["isaac-world", "pick-arm"])

    required, optional = IsaacSceneFinalizer.validate_config(config)

    assert list(required) == ["isaac-world", "pick-arm"]
    assert list(optional) == []


def test_validate_config_without_depends_on_raises():
    config = _config("scene-ready", {})

    with pytest.raises(ValueError, match="depends_on"):
        IsaacSceneFinalizer.validate_config(config)


def test_reconfigure_materialises_workcell_component_dependencies_before_finalizing(monkeypatch):
    calls: list = []
    monkeypatch.setattr(finalizer_model.SimManager, "get", lambda: _sim_manager_stub(calls))
    pallet_name = ResourceName(namespace="rdk", type="component", subtype="generic", name="pallet")
    world_name = ResourceName(
        namespace="rdk", type="component", subtype="generic", name="isaac-world"
    )
    dependencies = {
        pallet_name: _FakeWorkcellComponent("viam:workcell-components:pallet"),
        world_name: _FakeWorldComponent(),
    }
    config = _config("scene-ready", {}, depends_on=["isaac-world", "pallet"])

    IsaacSceneFinalizer.new(config, dependencies)

    assert [call[0] for call in calls] == ["materialise", "finalize"]
    materialised = calls[0][1]
    assert list(materialised) == ["pallet"]
    assert materialised["pallet"].frame_position_m == pytest.approx((0.2, 0.5, 0.2))


def test_reconfigure_with_no_workcell_component_dependencies_materialises_nothing(monkeypatch):
    calls: list = []
    monkeypatch.setattr(finalizer_model.SimManager, "get", lambda: _sim_manager_stub(calls))
    world_name = ResourceName(
        namespace="rdk", type="component", subtype="generic", name="isaac-world"
    )
    dependencies = {world_name: _FakeWorldComponent()}

    IsaacSceneFinalizer.new(_config("scene-ready", {}, depends_on=["isaac-world"]), dependencies)

    assert [call[0] for call in calls] == ["materialise", "finalize"]
    assert calls[0][1] == {}


def test_reconfigure_with_no_dependencies_at_all_only_finalizes_after_an_empty_materialise(
    monkeypatch,
):
    calls: list = []
    monkeypatch.setattr(finalizer_model.SimManager, "get", lambda: _sim_manager_stub(calls))

    IsaacSceneFinalizer.new(_config("scene-ready", {}, depends_on=["isaac-world"]), {})

    assert [call[0] for call in calls] == ["materialise", "finalize"]
    assert calls[0][1] == {}
