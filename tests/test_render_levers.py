"""CAM-12: render-cost levers, compat-gated.

World-level render levers ("motion_bvh", "disable_viewport_updates") applied
best-effort at boot (mirroring _apply_lighting), and the 5.0-only
Camera(annotator_device=...) GPU-resident data path (compat.Caps)."""

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import compat
from isaac_module.models.camera import _validate_camera_attrs
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimConfig, SimManager


def _world_config(attrs: dict) -> ComponentConfig:
    return ComponentConfig(name="sim-world", attributes=dict_to_struct(attrs))


# ----------------------------------------------------------------------
# world "render" attr validation
# ----------------------------------------------------------------------


def test_render_accepts_known_bool_keys():
    cfg = _world_config({"render": {"motion_bvh": False, "disable_viewport_updates": False}})
    IsaacWorld.validate_config(cfg)


def test_render_rejects_unknown_key():
    cfg = _world_config({"render": {"bogus": True}})
    with pytest.raises(ValueError, match="unknown key"):
        IsaacWorld.validate_config(cfg)


def test_render_rejects_non_bool_value():
    cfg = _world_config({"render": {"motion_bvh": "off"}})
    with pytest.raises(ValueError, match="motion_bvh"):
        IsaacWorld.validate_config(cfg)


def test_render_rejects_not_an_object():
    cfg = _world_config({"render": [True]})
    with pytest.raises(ValueError, match="render must be an object"):
        IsaacWorld.validate_config(cfg)


def test_disable_viewport_updates_with_livestream_is_rejected():
    cfg = _world_config({"render": {"disable_viewport_updates": True}, "livestream": True})
    with pytest.raises(ValueError, match="livestream"):
        IsaacWorld.validate_config(cfg)


def test_disable_viewport_updates_without_livestream_is_accepted():
    cfg = _world_config({"render": {"disable_viewport_updates": True}, "livestream": False})
    IsaacWorld.validate_config(cfg)


def test_disable_viewport_updates_defaults_livestream_true_and_is_rejected():
    # livestream defaults to True (models/world.py docstring); omitting it
    # must not silently allow the incompatible combination.
    cfg = _world_config({"render": {"disable_viewport_updates": True}})
    with pytest.raises(ValueError, match="livestream"):
        IsaacWorld.validate_config(cfg)


# ----------------------------------------------------------------------
# mock boot -> render config visible via status()
# ----------------------------------------------------------------------


def test_mock_boot_stores_render_config_visible_via_status():
    mgr = SimManager()
    render_cfg = {"motion_bvh": False, "disable_viewport_updates": True}
    mgr.cfg = SimConfig(mock=True, render=render_cfg)

    mgr._boot()

    assert mgr.render == render_cfg
    assert mgr.status()["render"] == render_cfg


def test_mock_boot_with_no_render_config_reports_none():
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=True)

    mgr._boot()

    assert mgr.render is None
    assert mgr.status()["render"] is None


# ----------------------------------------------------------------------
# camera "annotator_device" attr validation
# ----------------------------------------------------------------------


def test_annotator_device_accepts_non_empty_string():
    _validate_camera_attrs("cam", {"annotator_device": "cuda"})


def test_annotator_device_accepts_unset():
    _validate_camera_attrs("cam", {})


def test_annotator_device_rejects_empty_string():
    with pytest.raises(ValueError, match="annotator_device"):
        _validate_camera_attrs("cam", {"annotator_device": ""})


def test_annotator_device_rejects_non_string():
    with pytest.raises(ValueError, match="annotator_device"):
        _validate_camera_attrs("cam", {"annotator_device": 1})


# ----------------------------------------------------------------------
# caps() gating (compat.py)
# ----------------------------------------------------------------------


def test_caps_camera_supports_annotator_device_gated_by_release():
    assert compat.caps(version=(4, 5, 0)).camera_supports_annotator_device is False
    assert compat.caps(version=(5, 0, 0)).camera_supports_annotator_device is True


# ----------------------------------------------------------------------
# fake-isaac camera creation: annotator_device passed on 5.0, dropped on 4.5
# ----------------------------------------------------------------------


class _FakeCamera:
    """Records the kwargs it was constructed with (last instance wins)."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeCamera.last_kwargs = kwargs
        self._resolution = kwargs.get("resolution", (848, 480))
        self._clip = (0.05, 10.0)

    def initialize(self) -> None:
        pass

    def get_resolution(self):
        return self._resolution

    def get_horizontal_aperture(self) -> float:
        return 20.0

    def set_focal_length(self, value: float) -> None:
        pass

    def set_vertical_aperture(self, value: float) -> None:
        pass

    def set_clipping_range(self, near: float, far: float) -> None:
        self._clip = (near, far)

    def get_clipping_range(self):
        return self._clip

    def add_distance_to_image_plane_to_frame(self) -> None:
        pass

    def set_frequency(self, frequency: float) -> None:
        pass


class _FakeIsaacNamespace:
    Camera = _FakeCamera


def _camera_manager() -> SimManager:
    mgr = SimManager()
    mgr.mock = False
    mgr.cfg = SimConfig(mock=False)
    mgr._isaac = _FakeIsaacNamespace()
    return mgr


def test_annotator_device_kwarg_passed_on_5_0_caps(monkeypatch):
    monkeypatch.setattr(
        "isaac_module.sim_manager.caps", lambda version=None: compat.CAPS_BY_RELEASE[(5, 0)]
    )
    mgr = _camera_manager()

    mgr._create_camera_isaac("cam", {"annotator_device": "cuda"})

    assert _FakeCamera.last_kwargs.get("annotator_device") == "cuda"


def test_annotator_device_kwarg_dropped_on_4_5_caps(monkeypatch):
    monkeypatch.setattr(
        "isaac_module.sim_manager.caps", lambda version=None: compat.CAPS_BY_RELEASE[(4, 5)]
    )
    mgr = _camera_manager()

    mgr._create_camera_isaac("cam", {"annotator_device": "cuda"})

    assert "annotator_device" not in _FakeCamera.last_kwargs


# ----------------------------------------------------------------------
# per-camera frequency (P2) still flows through create_camera - unchanged
# by this slice, verified here that it still reaches the Isaac Camera.
# ----------------------------------------------------------------------


def test_frequency_still_flows_through_to_the_isaac_camera(monkeypatch):
    monkeypatch.setattr(
        "isaac_module.sim_manager.caps", lambda version=None: compat.CAPS_BY_RELEASE[(5, 0)]
    )
    mgr = _camera_manager()
    calls: list[float] = []

    class _RecordingCamera(_FakeCamera):
        def set_frequency(self, frequency: float) -> None:
            calls.append(frequency)

    mgr._isaac = type("_NS", (), {"Camera": _RecordingCamera})()

    mgr._create_camera_isaac("cam", {"frequency": 30.0})

    assert calls == [30.0]
