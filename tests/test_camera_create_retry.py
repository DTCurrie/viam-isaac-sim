"""_create_camera_isaac survives a cold renderer.

Observed on the first boot of a fresh install (GPU bring-up, 2026-09-02):
``Camera.initialize()`` died with
``KeyError('/Render/PostProcess/SDGPipeline/Replicator_01_LdrColorSDhostPtr')``
because the render product's SDG pipeline node only materializes after render
ticks, and the failed create left the broken render product behind, so every
5 s resource rebuild failed identically. initialize() is therefore retried
with render ticks stepped between attempts, and a camera that never
initializes is destroyed so the next build starts clean.
"""

import pytest

from isaac_module.errors import CameraInitError
from isaac_module.sim_manager import (
    CAMERA_INIT_ATTEMPTS,
    CAMERA_INIT_RENDER_TICKS,
    SimConfig,
    SimManager,
)


class _ColdRendererCamera:
    """initialize() raises KeyError until fail_times attempts have happened,
    mimicking the SDG node lookup on a renderer that has not ticked yet."""

    def __init__(self, **kwargs):
        self._resolution = kwargs.get("resolution", (848, 480))
        self._clip = (0.05, 10.0)
        self.init_calls = 0
        self.destroy_calls = 0
        self.fail_times = 0

    def initialize(self) -> None:
        self.init_calls += 1
        if self.init_calls <= self.fail_times:
            raise KeyError("/Render/PostProcess/SDGPipeline/Replicator_01_LdrColorSDhostPtr")

    def destroy(self) -> None:
        self.destroy_calls += 1

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


class _StepCountingWorld:
    def __init__(self):
        self.steps = 0

    def step(self, render: bool = False) -> None:
        self.steps += 1


def _camera_manager() -> tuple[SimManager, _StepCountingWorld]:
    mgr = SimManager()
    mgr.mock = False
    mgr.cfg = SimConfig(mock=False)
    world = _StepCountingWorld()
    mgr.world = world
    return mgr, world


def _make_namespace(camera: _ColdRendererCamera):
    return type("_NS", (), {"Camera": staticmethod(lambda **kwargs: camera)})()


def test_first_try_success_steps_no_frames():
    mgr, world = _camera_manager()
    camera = _ColdRendererCamera()
    mgr._isaac = _make_namespace(camera)

    mgr._create_camera_isaac("scene-cam", {})

    assert camera.init_calls == 1
    assert world.steps == 0
    assert camera.destroy_calls == 0


def test_cold_renderer_retries_with_render_ticks_between_attempts():
    mgr, world = _camera_manager()
    camera = _ColdRendererCamera()
    camera.fail_times = 2
    mgr._isaac = _make_namespace(camera)

    mgr._create_camera_isaac("scene-cam", {})

    assert camera.init_calls == 3
    assert world.steps == 2 * CAMERA_INIT_RENDER_TICKS
    assert camera.destroy_calls == 0


def test_exhausted_retries_destroy_the_camera_and_raise():
    mgr, world = _camera_manager()
    camera = _ColdRendererCamera()
    camera.fail_times = CAMERA_INIT_ATTEMPTS + 1
    mgr._isaac = _make_namespace(camera)

    with pytest.raises(CameraInitError) as excinfo:
        mgr._create_camera_isaac("scene-cam", {})

    assert camera.init_calls == CAMERA_INIT_ATTEMPTS
    assert world.steps == (CAMERA_INIT_ATTEMPTS - 1) * CAMERA_INIT_RENDER_TICKS
    assert camera.destroy_calls == 1
    assert isinstance(excinfo.value.__cause__, KeyError)
