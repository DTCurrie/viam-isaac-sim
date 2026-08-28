"""Tests for the Isaac Sim compat layer (FINDINGS XC-6)."""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from isaac_module import compat


def _isaac_is_installed() -> bool:
    return (
        importlib.util.find_spec("isaacsim") is not None
        or importlib.util.find_spec("omni") is not None
    )


@pytest.mark.skipif(
    _isaac_is_installed(),
    reason="Isaac is importable in this environment; isaac_version() would not be None",
)
def test_isaac_version_is_none_without_isaac() -> None:
    assert compat.isaac_version() is None


def test_caps_by_release_has_both_known_rows() -> None:
    assert set(compat.CAPS_BY_RELEASE) == {(4, 5), (5, 0)}

    row_45 = compat.CAPS_BY_RELEASE[(4, 5)]
    row_50 = compat.CAPS_BY_RELEASE[(5, 0)]

    assert row_45.has_depth_sensor != row_50.has_depth_sensor
    assert row_45.pointcloud_is_world_frame != row_50.pointcloud_is_world_frame
    assert row_45.camera_reads_cached_frame != row_50.camera_reads_cached_frame


@pytest.mark.parametrize(
    "version",
    [(4, 5, 0), (5, 0, 0), (4, 6, 1)],
)
def test_caps_resolves_exact_and_nearest_lower(version: tuple[int, int, int]) -> None:
    if version[:2] == (5, 0):
        assert compat.caps(version) == compat.CAPS_BY_RELEASE[(5, 0)]
    else:
        assert compat.caps(version) == compat.CAPS_BY_RELEASE[(4, 5)]


def test_caps_above_newest_known_row_uses_newest() -> None:
    assert compat.caps((6, 0, 0)) == compat.CAPS_BY_RELEASE[(5, 0)]


def test_caps_below_oldest_known_row_uses_oldest() -> None:
    assert compat.caps((4, 0, 0)) == compat.CAPS_BY_RELEASE[(4, 5)]


@pytest.mark.skipif(
    _isaac_is_installed(),
    reason="isaac_version() answers for real when Isaac is importable",
)
def test_caps_with_no_version_and_no_isaac_uses_newest() -> None:
    assert compat.caps(None) == compat.CAPS_BY_RELEASE[(5, 0)]


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch, dotted_path: str, module: types.ModuleType
) -> None:
    monkeypatch.setitem(sys.modules, dotted_path, module)


def test_probe_isaacsim_core_version_parses_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "isaacsim", types.ModuleType("isaacsim"))
    _install_fake_module(monkeypatch, "isaacsim.core", types.ModuleType("isaacsim.core"))

    fake_version_module = types.ModuleType("isaacsim.core.version")
    fake_version_module.get_version = lambda: (  # type: ignore[attr-defined]
        "5.0.0",
        "5",
        "0",
        "0",
        "",
        "107.3",
    )
    _install_fake_module(monkeypatch, "isaacsim.core.version", fake_version_module)

    assert compat.isaac_version() == (5, 0, 0)


def test_probe_omni_isaac_version_parses_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "omni", types.ModuleType("omni"))
    _install_fake_module(monkeypatch, "omni.isaac", types.ModuleType("omni.isaac"))

    fake_version_module = types.ModuleType("omni.isaac.version")
    fake_version_module.get_version = lambda: "4.5.0"  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "omni.isaac.version", fake_version_module)

    assert compat.isaac_version() == (4, 5, 0)


def test_probe_that_raises_falls_through_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "isaacsim", types.ModuleType("isaacsim"))
    _install_fake_module(monkeypatch, "isaacsim.core", types.ModuleType("isaacsim.core"))

    def _raise() -> None:
        raise RuntimeError("boom")

    fake_version_module = types.ModuleType("isaacsim.core.version")
    fake_version_module.get_version = _raise  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "isaacsim.core.version", fake_version_module)

    _install_fake_module(monkeypatch, "omni", types.ModuleType("omni"))
    _install_fake_module(monkeypatch, "omni.isaac", types.ModuleType("omni.isaac"))

    def _raise_too() -> None:
        raise RuntimeError("boom too")

    fake_omni_version_module = types.ModuleType("omni.isaac.version")
    fake_omni_version_module.get_version = _raise_too  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "omni.isaac.version", fake_omni_version_module)

    assert compat.isaac_version() is None


def test_import_isaac_moved_into_sim_manager() -> None:
    from isaac_module.sim_manager import import_isaac

    assert import_isaac is compat.import_isaac
