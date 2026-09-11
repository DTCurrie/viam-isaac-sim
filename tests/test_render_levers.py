import sys
import types
from typing import ClassVar

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import compat
from isaac_module.models.camera import _validate_camera_attrs
from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import SimConfig, SimManager, _boot_extra_args


def _world_config(attrs: dict) -> ComponentConfig:
    return ComponentConfig(name="isaac-world", attributes=dict_to_struct(attrs))


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
# disable_viewport_updates and limit_cpu_threads reach Isaac Sim through the
# SimulationApp launcher config at boot.
# ----------------------------------------------------------------------


class _StopAfterCapture(Exception):
    """Raised by the fake SimulationApp once it has recorded its launcher
    config, so the test doesn't have to fake the rest of a real boot."""


def _install_fake_simulation_app(monkeypatch) -> list[dict]:
    captured: list[dict] = []

    class _FakeSimulationApp:
        def __init__(self, config: dict) -> None:
            captured.append(dict(config))
            raise _StopAfterCapture

    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = _FakeSimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)
    return captured


def test_boot_passes_disable_viewport_updates_into_the_launcher_config(monkeypatch):
    captured = _install_fake_simulation_app(monkeypatch)
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=False, render={"disable_viewport_updates": True})

    with pytest.raises(_StopAfterCapture):
        mgr._boot()

    assert captured[0]["disable_viewport_updates"] is True


def test_boot_omits_the_launcher_config_key_when_render_does_not_set_it(monkeypatch):
    captured = _install_fake_simulation_app(monkeypatch)
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=False)

    with pytest.raises(_StopAfterCapture):
        mgr._boot()

    assert "disable_viewport_updates" not in captured[0]


def test_boot_sets_limit_cpu_threads_from_os_cpu_count(monkeypatch):
    # IS-14: 32 Carbonite/TBB workers (5.0's own default) oversubscribes an
    # 8 vCPU box that also runs viam-server and PhysX.
    captured = _install_fake_simulation_app(monkeypatch)
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=False)

    with pytest.raises(_StopAfterCapture):
        mgr._boot()

    assert captured[0]["limit_cpu_threads"] == 6


def test_boot_does_not_mutate_sys_argv(monkeypatch):
    # IS-23/PY-15: kit args used to be injected via a permanent sys.argv
    # mutation that viam-server's own arg handling would also see.
    _install_fake_simulation_app(monkeypatch)
    original_argv = list(sys.argv)
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=False, kit_log_level="verbose")

    with pytest.raises(_StopAfterCapture):
        mgr._boot()

    assert sys.argv == original_argv


# ----------------------------------------------------------------------
# livestream (IS-5/IS-9/IS-12): hide_ui in the launcher config, and the
# livestream port carb setting.
# ----------------------------------------------------------------------


def test_boot_passes_hide_ui_false_into_the_launcher_config_when_livestreaming(monkeypatch):
    captured = _install_fake_simulation_app(monkeypatch)
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=False, livestream=True, headless=True)

    with pytest.raises(_StopAfterCapture):
        mgr._boot()

    assert captured[0]["hide_ui"] is False


def test_boot_omits_hide_ui_when_not_livestreaming(monkeypatch):
    captured = _install_fake_simulation_app(monkeypatch)
    mgr = SimManager()
    mgr.cfg = SimConfig(mock=False, livestream=False)

    with pytest.raises(_StopAfterCapture):
        mgr._boot()

    assert "hide_ui" not in captured[0]


class _RecordingSimulationApp:
    """A SimulationApp fake that survives construction (unlike
    _StopAfterCapture's) so the livestream block after it can run, and
    records every set_setting call it receives."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.settings: dict[str, object] = {}

    def set_setting(self, path: str, value: object) -> None:
        self.settings[path] = value


class _StopAfterLivestream(Exception):
    """Raised by the faked import_isaac(), which _boot() calls right after
    the livestream block, so the test doesn't have to fake the rest of a
    real boot."""


def _install_recording_simulation_app_and_stop_after_livestream(
    monkeypatch, *, enable_extension_returns: bool = True
) -> list[_RecordingSimulationApp]:
    instances: list[_RecordingSimulationApp] = []

    def _make_recording_simulation_app(config: dict) -> _RecordingSimulationApp:
        app = _RecordingSimulationApp(config)
        instances.append(app)
        return app

    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = _make_recording_simulation_app
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)

    fake_extensions = types.ModuleType("isaacsim.core.utils.extensions")
    fake_extensions.enable_extension = lambda name: enable_extension_returns
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.extensions", fake_extensions)

    def _raise_stop():
        raise _StopAfterLivestream

    monkeypatch.setattr("isaac_module.sim_manager.import_isaac", _raise_stop)
    return instances


def test_boot_sets_the_livestream_port(monkeypatch):
    instances = _install_recording_simulation_app_and_stop_after_livestream(monkeypatch)
    mgr = SimManager()
    mgr.cfg = SimConfig(
        mock=False, livestream=True, headless=True, livestream_public_ip="203.0.113.5"
    )

    with pytest.raises(_StopAfterLivestream):
        mgr._boot()

    assert instances[0].settings["/app/livestream/port"] == 49100


def test_boot_sets_draw_mouse_when_livestreaming(monkeypatch):
    instances = _install_recording_simulation_app_and_stop_after_livestream(monkeypatch)
    mgr = SimManager()
    mgr.cfg = SimConfig(
        mock=False, livestream=True, headless=True, livestream_public_ip="203.0.113.5"
    )

    with pytest.raises(_StopAfterLivestream):
        mgr._boot()

    assert instances[0].settings["/app/window/drawMouse"] is True


# ----------------------------------------------------------------------
# _boot_extra_args: the Kit log level (IS-23/PY-15), the three motion-BVH
# settings (IS-7) and the DLSS exec mode (IS-15) fold into extra_args on the
# SimulationApp launcher config, replacing sys.argv.append and the
# post-launch carb.settings writes those keys used.
# ----------------------------------------------------------------------


def test_boot_extra_args_carries_the_kit_log_level():
    args = _boot_extra_args(SimConfig(mock=False, kit_log_level="verbose"))

    assert "--/log/outputStreamLevel=Verbose" in args


def test_boot_extra_args_disables_motion_bvh_with_all_three_settings():
    args = _boot_extra_args(SimConfig(mock=False, render={"motion_bvh": False}))

    assert "--/renderer/raytracingMotion/enabled=false" in args
    assert "--/renderer/raytracingMotion/enableHydraEngineMasking=false" in args
    assert "--/renderer/raytracingMotion/enabledForHydraEngines=" in args


def test_boot_extra_args_omits_motion_bvh_settings_when_unconfigured():
    args = _boot_extra_args(SimConfig(mock=False))

    assert not any("raytracingMotion" in arg for arg in args)


def test_boot_extra_args_sets_dlss_performance_exec_mode():
    args = _boot_extra_args(SimConfig(mock=False))

    assert "--/rtx/post/dlss/execMode=0" in args


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


def test_caps_camera_supports_annotator_device_on_5_0():
    assert compat.caps(version=(5, 0, 0)).camera_supports_annotator_device is True


# ----------------------------------------------------------------------
# fake-isaac camera creation: annotator_device passed when caps allow it, dropped otherwise
# ----------------------------------------------------------------------


class _FakeCamera:
    """Records the kwargs it was constructed with (last instance wins)."""

    last_kwargs: ClassVar[dict] = {}

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


def test_annotator_device_kwarg_dropped_when_caps_deny_it(monkeypatch):
    denied = compat.caps()._replace(camera_supports_annotator_device=False)
    monkeypatch.setattr("isaac_module.sim_manager.caps", lambda version=None: denied)
    mgr = _camera_manager()

    mgr._create_camera_isaac("cam", {"annotator_device": "cuda"})

    assert "annotator_device" not in _FakeCamera.last_kwargs


# ----------------------------------------------------------------------
