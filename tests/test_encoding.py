"""Golden-byte and round-trip tests for isaac_module.encoding (FINDINGS CAM-7)."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image
from viam.media.video import CameraMimeType, ViamImage

from isaac_module.encoding import (
    depth_m_to_viam_dep,
    depth_to_xyz,
    intrinsics_from_fov,
    rgb_to_jpeg,
    rgb_to_png,
    xyz_rgb_to_pcd,
)


def test_intrinsics_from_fov() -> None:
    # fx = fy = W / (2*tan(hfov/2)) per the encoding.py docstring formula;
    # 848 / (2*tan(radians(90.5/2))) == 420.3159532922707 (see Deviations in
    # the slice report re: the brief's stated ~420.1 / FINDINGS W18 value).
    k = intrinsics_from_fov(848, 480, 90.5)
    assert k.fx == k.fy
    assert 420.25 <= k.fx <= 420.4
    assert k.cx == 424.0
    assert k.cy == 240.0
    assert k.width == 848
    assert k.height == 480


def test_intrinsics_from_fov_rejects_bad_dims() -> None:
    with pytest.raises(ValueError):
        intrinsics_from_fov(0, 480, 90.0)
    with pytest.raises(ValueError):
        intrinsics_from_fov(848, -1, 90.0)


def test_intrinsics_from_fov_rejects_bad_hfov() -> None:
    with pytest.raises(ValueError):
        intrinsics_from_fov(848, 480, 0.0)
    with pytest.raises(ValueError):
        intrinsics_from_fov(848, 480, 180.0)


def test_dep_golden_bytes() -> None:
    depth = np.array([[0.001, 0.5, 1.0], [np.nan, -1.0, 70.0]], dtype=np.float64)
    encoded = depth_m_to_viam_dep(depth)

    expected = (
        b"DEPTHMAP"
        + (3).to_bytes(8, "big")
        + (2).to_bytes(8, "big")
        + struct.pack(">6H", 1, 500, 1000, 0, 0, 0)
    )
    assert encoded == expected
    assert len(encoded) == 24 + 2 * 3 * 2


def test_dep_round_trip_through_viam_sdk() -> None:
    # non-square (3 cols x 2 rows) so W/H swaps are caught
    depth = np.array([[0.001, 0.5, 1.0], [np.nan, -1.0, 70.0]], dtype=np.float64)
    encoded = depth_m_to_viam_dep(depth)

    image = ViamImage(encoded, CameraMimeType.VIAM_RAW_DEPTH)
    decoded = image.bytes_to_depth_array()

    assert decoded == [[1, 500, 1000], [0, 0, 0]]


def test_pcd_golden_bytes_coloured() -> None:
    xyz = np.array([[1, 2, 3], [-0.5, 0.25, 0.125]], dtype=np.float32)
    rgb = np.array([[255, 0, 0], [0, 128, 64]], dtype=np.uint8)

    encoded = xyz_rgb_to_pcd(xyz, rgb)

    header = (
        b"VERSION .7\n"
        b"FIELDS x y z rgb\n"
        b"SIZE 4 4 4 4\n"
        b"TYPE F F F I\n"
        b"COUNT 1 1 1 1\n"
        b"WIDTH 2\n"
        b"HEIGHT 1\n"
        b"VIEWPOINT 0 0 0 1 0 0 0\n"
        b"POINTS 2\n"
        b"DATA binary\n"
    )
    payload = struct.pack("<fffI", 1, 2, 3, 0xFF0000) + struct.pack(
        "<fffI", -0.5, 0.25, 0.125, 0x008040
    )
    expected = header + payload

    assert encoded == expected
    assert b"FIELDS x y z rgb\n" in encoded
    assert b"SIZE 4 4 4 4\n" in encoded
    assert b"TYPE F F F I\n" in encoded
    assert b"COUNT 1 1 1 1\n" in encoded
    assert b"WIDTH 2\n" in encoded
    assert b"HEIGHT 1\n" in encoded
    assert b"VIEWPOINT 0 0 0 1 0 0 0\n" in encoded
    assert b"POINTS 2\n" in encoded
    assert b"DATA binary\n" in encoded
    assert not encoded.startswith(b"#")
    assert len(payload) == 32


def test_pcd_uncoloured() -> None:
    xyz = np.array([[1, 2, 3], [-0.5, 0.25, 0.125]], dtype=np.float32)

    encoded = xyz_rgb_to_pcd(xyz, None)

    assert b"FIELDS x y z\n" in encoded
    assert b"SIZE 4 4 4\n" in encoded
    assert b"TYPE F F F\n" in encoded
    assert b"COUNT 1 1 1\n" in encoded

    header_end = encoded.index(b"DATA binary\n") + len(b"DATA binary\n")
    payload = encoded[header_end:]
    assert len(payload) == 12 * 2


def test_pcd_empty_is_valid() -> None:
    xyz = np.zeros((0, 3), dtype=np.float32)
    encoded = xyz_rgb_to_pcd(xyz, None)

    assert b"WIDTH 0\n" in encoded
    assert b"POINTS 0\n" in encoded
    header_end = encoded.index(b"DATA binary\n") + len(b"DATA binary\n")
    assert encoded[header_end:] == b""


def test_depth_to_xyz() -> None:
    depth = np.full((4, 4), 0.5, dtype=np.float64)
    depth[1, 1] = np.nan
    k = intrinsics_from_fov(4, 4, 90.0)

    xyz, mask = depth_to_xyz(depth, k)

    assert xyz.dtype == np.float32
    assert mask.shape == (4, 4)
    assert mask.sum() == 15
    assert xyz.shape == (15, 3)

    def point_for(u: int, v: int) -> np.ndarray:
        flat_valid_positions = np.flatnonzero(mask.ravel())
        flat_position = v * 4 + u
        index = int(np.searchsorted(flat_valid_positions, flat_position))
        assert flat_valid_positions[index] == flat_position
        return xyz[index]

    u_center, v_center = int(k.cx), int(k.cy)
    center_point = point_for(u_center, v_center)
    assert np.allclose(center_point, [0.0, 0.0, 0.5], atol=1e-6)

    right_point = point_for(u_center + 1, v_center)
    assert np.isclose(right_point[0], 0.5 / k.fx)
    assert np.isclose(right_point[1], 0.0)


def test_depth_to_xyz_drops_zero() -> None:
    depth = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    k = intrinsics_from_fov(2, 2, 90.0)

    xyz, mask = depth_to_xyz(depth, k)

    assert mask.sum() == 3
    assert xyz.shape == (3, 3)
    assert not mask[0, 0]


def test_rgb_to_png_round_trip() -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    encoded = rgb_to_png(rgb)
    from io import BytesIO

    decoded = Image.open(BytesIO(encoded))
    assert decoded.size == (8, 6)
    assert decoded.mode == "RGB"


def test_rgb_to_jpeg_round_trip() -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 255

    encoded = rgb_to_jpeg(rgb)
    from io import BytesIO

    decoded = Image.open(BytesIO(encoded))
    assert decoded.size == (8, 6)
    assert decoded.mode == "RGB"


def test_rgb_to_png_rejects_wrong_dtype() -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        rgb_to_png(rgb)


def test_rgb_to_png_rejects_wrong_shape() -> None:
    rgb = np.zeros((6, 8, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        rgb_to_png(rgb)


def test_rgb_to_jpeg_rejects_wrong_dtype() -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        rgb_to_jpeg(rgb)


def test_rgb_to_jpeg_rejects_wrong_shape() -> None:
    rgb = np.zeros((6, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        rgb_to_jpeg(rgb)
