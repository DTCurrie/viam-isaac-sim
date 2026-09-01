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


def test_gripper_caps_follow_the_2f85_asset_per_release():
    """FINDINGS R-9 / W13: the 2F-85 finger_joint closes at 47 deg on 5.0 and
    45 deg on 4.5, with drive maxForce 26 vs 16.5 - version splits live here,
    never in models/gripper.py."""
    row_45 = compat.CAPS_BY_RELEASE[(4, 5)]
    row_50 = compat.CAPS_BY_RELEASE[(5, 0)]
    assert (row_45.gripper_closed_deg, row_45.gripper_max_force) == (45.0, 16.5)
    assert (row_50.gripper_closed_deg, row_50.gripper_max_force) == (47.0, 26.0)
    assert compat.caps((5, 0, 0)).gripper_dof_count == 6


def test_gripper_open_angle_is_zero_on_both_releases():
    """GPU run 19: with the articulation fixes in place the 5.0 2F-85 reaches
    0.4 deg at an open target of 0 (the 7.76 deg rest seen in run 12 was an
    artifact of the broken wrapper); W13's 0 stands on both releases."""
    assert compat.CAPS_BY_RELEASE[(5, 0)].gripper_open_deg == 0.0
    assert compat.CAPS_BY_RELEASE[(4, 5)].gripper_open_deg == 0.0


def test_camera_supports_annotator_device_only_on_5_0():
    """CAM-12: Camera(annotator_device=...) is a 5.0-only GPU-resident data
    path (CHANGELOG 0.4.0); 4.5 always lands in host numpy."""
    assert compat.CAPS_BY_RELEASE[(4, 5)].camera_supports_annotator_device is False
    assert compat.CAPS_BY_RELEASE[(5, 0)].camera_supports_annotator_device is True
    assert compat.caps((4, 5, 0)).camera_supports_annotator_device is False
    assert compat.caps((5, 0, 0)).camera_supports_annotator_device is True
