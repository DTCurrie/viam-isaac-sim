"""Tests for the mock camera scene (FINDINGS CAM-14, slice 2b)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from isaac_module.encoding import depth_to_xyz, intrinsics_from_fov
from isaac_module.mock_camera import (
    MOCK_RED_BLOCK_CENTER_M,
    RED_BLOCK_DEPTH_M,
    RED_BLOCK_RGB,
    MockCameraHandle,
)


def make_handle(**attrs: object) -> MockCameraHandle:
    return MockCameraHandle("cam1", attrs)


def test_rgb_and_depth_shapes_and_dtypes() -> None:
    handle = make_handle(depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    assert rgb.shape == (480, 848, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (480, 848)
    assert depth.dtype == np.float32


def test_get_depth_raises_when_depth_disabled() -> None:
    handle = make_handle()
    with pytest.raises(RuntimeError):
        handle.get_depth()


def test_red_block_mask_size_and_depth() -> None:
    handle = make_handle(depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    mask = (rgb == np.array(RED_BLOCK_RGB)).all(-1)
    assert mask.sum() == 100 * 80
    assert np.all(depth[mask] == RED_BLOCK_DEPTH_M)


def test_nan_band_is_exactly_the_top_rows() -> None:
    handle = make_handle(depth=True)
    depth = handle.get_depth()
    nan_band_rows = 480 // 10  # 48
    assert np.all(np.isnan(depth[:nan_band_rows, :]))
    assert not np.any(np.isnan(depth[nan_band_rows:, :]))


def test_red_cluster_centroid_matches_analytic_center() -> None:
    handle = make_handle(depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    k = handle.get_intrinsics()

    xyz, mask = depth_to_xyz(depth, k)
    masked_rgb = rgb[mask]
    red_selector = (masked_rgb == np.array(RED_BLOCK_RGB)).all(-1)
    red_points = xyz[red_selector]

    expected = np.array(handle.red_block_center_m, dtype=np.float32)
    # A centred block would have produced (0, 0, 0.40); the block is
    # deliberately off-centre so this comparison is non-degenerate.
    assert not np.allclose(expected[:2], [0.0, 0.0])
    # float32 mean-of-8000-identical-values accumulation error dominates over
    # the 1e-5 m analytic tolerance; use float64 accumulation to honor it.
    centroid_f64 = red_points.astype(np.float64).mean(axis=0)
    expected_f64 = np.array(handle.red_block_center_m, dtype=np.float64)
    np.testing.assert_allclose(centroid_f64, expected_f64, atol=1e-5)
    np.testing.assert_allclose(
        centroid_f64, np.array(MOCK_RED_BLOCK_CENTER_M, dtype=np.float64), atol=1e-5
    )


def test_default_center_is_off_centre_both_ways() -> None:
    assert MOCK_RED_BLOCK_CENTER_M[0] > 0.05
    assert MOCK_RED_BLOCK_CENTER_M[1] > 0
    assert MOCK_RED_BLOCK_CENTER_M[2] == RED_BLOCK_DEPTH_M


def test_lower_resolution_handle_has_different_x_centre() -> None:
    # The block is defined by fixed pixel offsets from the principal point,
    # not scaled with resolution, while fx scales with width. So the same
    # pixel offset maps to a different metric x at a different resolution.
    default_handle = make_handle(depth=True)
    small_handle = make_handle(width=424, height=240, depth=True)
    assert small_handle.red_block_center_m[0] != default_handle.red_block_center_m[0]


def test_get_intrinsics_matches_shared_helper() -> None:
    handle = make_handle()
    assert handle.get_intrinsics() == intrinsics_from_fov(848, 480, 90.5)


def test_post_reset_increments_reset_count() -> None:
    handle = make_handle()
    assert handle.reset_count == 0
    handle.post_reset()
    handle.post_reset()
    assert handle.reset_count == 2


def test_get_frame_shares_sim_time_within_one_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = make_handle()
    fake_time = [1000.001]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

    frame1 = handle.get_frame()
    fake_time[0] += 1.0 / 120  # still within the same 1/60s tick
    frame2 = handle.get_frame()

    assert frame1.sim_time == frame2.sim_time
