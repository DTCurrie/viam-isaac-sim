"""Pure-numpy encoders for Viam's camera wire formats (FINDINGS CAM-7).

No Isaac imports, no sim-thread affinity: every function here is safe to call
from ``asyncio.to_thread``. Byte layouts are rdk-exact; the verified layouts are
in ``.claude/plans/isaac-mvp-research/research/RESEARCH.md`` §V-2 and §P-5
(sources: rdk ``rimage/depth_map_raw.go``, ``pointcloud/pointcloud_file.go``).
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import NamedTuple

import numpy as np
from PIL import Image

DEPTH_MAGIC = b"DEPTHMAP"  # big-endian uint64 4919426490892632400
DEPTH_MIME = "image/vnd.viam.dep"
PCD_MIME = "pointcloud/pcd"
MAX_DEPTH_MM = 65535  # rdk rimage.MaxDepth; deeper → 0 (invalid)


class Intrinsics(NamedTuple):
    """Pinhole intrinsics in pixels. Never zero-filled (rdk divides by fx)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Square-pixel pinhole intrinsics from a horizontal FOV.

    fx = fy = W / (2·tan(hfov/2)), cx = W/2, cy = H/2.
    intrinsics_from_fov(848, 480, 90.5) → fx = fy ≈ 420.1, cx = 424, cy = 240.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if not (0 < hfov_deg < 180):
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")

    hfov_rad = math.radians(hfov_deg)
    fx = fy = width / (2 * math.tan(hfov_rad / 2))
    cx = width / 2
    cy = height / 2
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got shape={rgb.shape} dtype={rgb.dtype}")


def rgb_to_png(rgb: np.ndarray) -> bytes:
    """(H, W, 3) uint8 → PNG bytes."""
    _validate_rgb(rgb)
    image = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """(H, W, 3) uint8 → JPEG bytes."""
    _validate_rgb(rgb)
    image = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def depth_m_to_viam_dep(depth_m: np.ndarray) -> bytes:
    """(H, W) float metres → ``image/vnd.viam.dep`` bytes.

    Layout: ``b"DEPTHMAP"`` + width as big-endian uint64 + height as big-endian
    uint64 + H·W big-endian uint16 millimetres, row-major. ``0`` = no reading.
    Non-finite, negative, or > 65.535 m → 0. Must round-trip through
    ``viam.media.video.ViamImage.bytes_to_depth_array``.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"expected a 2-D depth array, got shape={depth_m.shape}")

    height, width = depth_m.shape
    mm = depth_m.astype(np.float64) * 1000.0
    invalid = ~np.isfinite(mm) | (mm <= 0) | (mm > MAX_DEPTH_MM)
    mm = np.where(invalid, 0.0, mm)
    mm_rounded = np.rint(mm).astype(">u2")
    payload = np.ascontiguousarray(mm_rounded, dtype=">u2").tobytes()
    header = DEPTH_MAGIC + width.to_bytes(8, "big") + height.to_bytes(8, "big")
    return header + payload


def depth_to_xyz(depth_m: np.ndarray, k: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Back-project (H, W) float metres to camera-optical-frame points.

    Integer pixel grid (u, v) → x = (u − cx)·z/fx, y = (v − cy)·z/fy, z = depth.
    Frame: +X right, +Y down, +Z forward (ROS/OpenCV optical). Returns
    ``(xyz, mask)``: ``xyz`` is (N, 3) float32 metres for the valid pixels only
    (finite and > 0), ``mask`` is the (H, W) bool array selecting them so a
    caller can pick the matching colours with ``rgb[mask]``.
    """
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    mask = np.isfinite(depth_m) & (depth_m > 0)

    z = depth_m[mask]
    x = (u[mask] - k.cx) * z / k.fx
    y = (v[mask] - k.cy) * z / k.fy
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz, mask


def xyz_rgb_to_pcd(xyz: np.ndarray, rgb: np.ndarray | None) -> bytes:
    """(N, 3) float32 metres [+ (N, 3) uint8] → binary ``pointcloud/pcd`` bytes.

    Header (ASCII, ``\\n``-terminated lines, no ``# .PCD`` comment)::

        VERSION .7
        FIELDS x y z rgb          (or "x y z" when rgb is None)
        SIZE 4 4 4 4              (or "4 4 4")
        TYPE F F F I              (or "F F F")
        COUNT 1 1 1 1             (or "1 1 1")
        WIDTH <N>
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS <N>
        DATA binary

    Payload per point: little-endian float32 x, y, z, then (coloured) a
    little-endian uint32 ``r << 16 | g << 8 | b`` — 16 B/point coloured, 12 B
    uncoloured. Wire units are METRES.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N, 3) xyz, got shape={xyz.shape}")

    num_points = xyz.shape[0]
    coloured = rgb is not None
    if coloured:
        assert rgb is not None  # narrow for mypy
        if rgb.ndim != 2 or rgb.shape != (num_points, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                f"expected ({num_points}, 3) uint8 rgb, got shape={rgb.shape} dtype={rgb.dtype}"
            )

    fields = "x y z rgb" if coloured else "x y z"
    sizes = "4 4 4 4" if coloured else "4 4 4"
    types = "F F F I" if coloured else "F F F"
    counts = "1 1 1 1" if coloured else "1 1 1"
    header = (
        "VERSION .7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {num_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_points}\n"
        "DATA binary\n"
    ).encode("ascii")

    if coloured:
        assert rgb is not None
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        points = np.zeros(num_points, dtype=dtype)
        points["x"] = xyz[:, 0]
        points["y"] = xyz[:, 1]
        points["z"] = xyz[:, 2]
        packed_rgb = (
            rgb[:, 0].astype(np.uint32) << 16
            | rgb[:, 1].astype(np.uint32) << 8
            | rgb[:, 2].astype(np.uint32)
        )
        points["rgb"] = packed_rgb
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        points = np.zeros(num_points, dtype=dtype)
        points["x"] = xyz[:, 0]
        points["y"] = xyz[:, 1]
        points["z"] = xyz[:, 2]

    return header + points.tobytes()


__all__ = [
    "DEPTH_MAGIC",
    "DEPTH_MIME",
    "MAX_DEPTH_MM",
    "PCD_MIME",
    "Intrinsics",
    "depth_m_to_viam_dep",
    "depth_to_xyz",
    "intrinsics_from_fov",
    "rgb_to_jpeg",
    "rgb_to_png",
    "xyz_rgb_to_pcd",
]
