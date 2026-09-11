from types import SimpleNamespace

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

import isaac_module.models.scene_finalizer as finalizer_model
from isaac_module.models.scene_finalizer import IsaacSceneFinalizer


def _config(name: str, attrs: dict, depends_on: list[str] | None = None) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs), depends_on=depends_on or [])


def test_new_finalizes_the_scene_once_and_carries_the_configured_name(monkeypatch):
    calls = []
    monkeypatch.setattr(
        finalizer_model.SimManager,
        "get",
        lambda: SimpleNamespace(finalize_scene=lambda: calls.append(True)),
    )

    finalizer = IsaacSceneFinalizer.new(_config("scene-ready", {}, depends_on=["isaac-world"]), {})

    assert finalizer.name == "scene-ready"
    assert calls == [True]


def test_reconfigure_finalizes_the_scene_again(monkeypatch):
    calls = []
    monkeypatch.setattr(
        finalizer_model.SimManager,
        "get",
        lambda: SimpleNamespace(finalize_scene=lambda: calls.append(True)),
    )

    finalizer = IsaacSceneFinalizer.new(_config("scene-ready", {}, depends_on=["isaac-world"]), {})
    finalizer.reconfigure(_config("scene-ready", {}, depends_on=["isaac-world"]), {})

    assert calls == [True, True]


def test_validate_config_returns_depends_on_as_required_dependencies():
    config = _config("scene-ready", {}, depends_on=["isaac-world", "pick-arm"])

    required, optional = IsaacSceneFinalizer.validate_config(config)

    assert list(required) == ["isaac-world", "pick-arm"]
    assert list(optional) == []


def test_validate_config_without_depends_on_raises():
    config = _config("scene-ready", {})

    with pytest.raises(ValueError, match="depends_on"):
        IsaacSceneFinalizer.validate_config(config)
