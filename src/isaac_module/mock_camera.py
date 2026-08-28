"""Mock camera backend (FINDINGS CAM-14) — the mock gate's red block.

A static scene, defined in pixels relative to the principal point so it scales
with resolution: a grey ramped floor, a red (#E02020-ish) rectangle at a
constant depth whose analytic centre is ``MOCK_RED_BLOCK_CENTER_M``, and a NaN
band of "no hit" pixels along the top. Intrinsics come from
``encoding.intrinsics_from_fov``. The scene is precomputed once per handle and
never changes between frames — determinism beats the old moving-bar mock for
testing a fixed analytic centroid (see Deviations in the slice report).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .camera_base import CameraHandle, Frame
from .encoding import Intrinsics, intrinsics_from_fov

DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FOV_DEG = 90.5

# Red block's column/row span relative to the principal point (cx, cy), in
# pixels. Deliberately off-centre in both axes so a sign or axis bug in the
# back-projection changes the analytic centroid.
BLOCK_LEFT_OFFSET_PX = 40
BLOCK_RIGHT_OFFSET_PX = 140
BLOCK_TOP_OFFSET_PX = -20
BLOCK_BOTTOM_OFFSET_PX = 60

RED_BLOCK_RGB = (224, 32, 32)
RED_BLOCK_DEPTH_M = 0.40

FLOOR_FAR_M = 1.20  # depth at the top row
FLOOR_NEAR_M = 0.60  # depth at the bottom row
FLOOR_GREY_TOP = 40  # rgb grey level at the top row
FLOOR_GREY_BOTTOM = 200  # rgb grey level at the bottom row

NAN_BAND_FRACTION = 10  # top height // NAN_BAND_FRACTION rows are "no hit"
NAN_BAND_RGB = 20  # dark grey shown for the no-hit band


def _block_pixel_bounds(k: Intrinsics) -> tuple[int, int, int, int]:
    """Integer pixel bounds (u0, u1, v0, v1), u1/v1 exclusive, of the block."""
    cx_i, cy_i = int(k.cx), int(k.cy)
    u0 = cx_i + BLOCK_LEFT_OFFSET_PX
    u1 = cx_i + BLOCK_RIGHT_OFFSET_PX
    v0 = cy_i + BLOCK_TOP_OFFSET_PX
    v1 = cy_i + BLOCK_BOTTOM_OFFSET_PX
    return u0, u1, v0, v1


def _block_center_m(k: Intrinsics) -> tuple[float, float, float]:
    u0, u1, v0, v1 = _block_pixel_bounds(k)
    u_mean = (u0 + u1 - 1) / 2
    v_mean = (v0 + v1 - 1) / 2
    z = RED_BLOCK_DEPTH_M
    x = (u_mean - k.cx) * z / k.fx
    y = (v_mean - k.cy) * z / k.fy
    return (x, y, z)


class MockCameraHandle(CameraHandle):
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self._width = int(attrs.get("width", DEFAULT_WIDTH))
        self._height = int(attrs.get("height", DEFAULT_HEIGHT))
        fov_deg = float(attrs.get("fov_deg", DEFAULT_FOV_DEG))
        self.depth_enabled = bool(attrs.get("depth", False))
        self.image_format = attrs.get("image_format", "png")
        self.frequency = attrs.get("frequency")
        self.reset_count = 0

        self._k = intrinsics_from_fov(self._width, self._height, fov_deg)
        self._rgb, self._depth = self._build_scene()

    def _build_scene(self) -> tuple[np.ndarray, np.ndarray]:
        width, height = self._width, self._height

        row_grey = np.linspace(FLOOR_GREY_TOP, FLOOR_GREY_BOTTOM, height)
        row_depth = np.linspace(FLOOR_FAR_M, FLOOR_NEAR_M, height, dtype=np.float32)

        rgb = np.repeat(row_grey[:, None, None], width, axis=1).astype(np.uint8)
        rgb = np.repeat(rgb, 3, axis=2)
        depth = np.repeat(row_depth[:, None], width, axis=1).astype(np.float32)

        nan_band_rows = height // NAN_BAND_FRACTION
        rgb[:nan_band_rows, :, :] = NAN_BAND_RGB
        depth[:nan_band_rows, :] = np.nan

        u0, u1, v0, v1 = _block_pixel_bounds(self._k)
        rgb[v0:v1, u0:u1, :] = RED_BLOCK_RGB
        depth[v0:v1, u0:u1] = RED_BLOCK_DEPTH_M

        return rgb, depth

    @property
    def red_block_center_m(self) -> tuple[float, float, float]:
        return _block_center_m(self._k)

    def get_frame(self) -> Frame:
        sim_time = math.floor(time.monotonic() * 60) / 60
        depth = self._depth if self.depth_enabled else None
        return Frame(rgb=self._rgb, depth=depth, sim_time=sim_time)

    def get_intrinsics(self) -> Intrinsics:
        return self._k

    def post_reset(self) -> None:
        self.reset_count += 1


# Camera-optical-frame centre (x right, y down, z forward) of the mock's red
# block for the default 848x480 @ 90.5 deg configuration, metres. The model
# test (3a) asserts the PCD red-cluster centroid equals it within 1 mm.
MOCK_RED_BLOCK_CENTER_M: tuple[float, float, float] = MockCameraHandle(
    "_default", {}
).red_block_center_m
