"""Size, tallest, trust and centre-depth measurement helpers - pure numpy over
point clouds, unit-testable without a robot (see tests/test_pick_red_block.py)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pickcell.obstacles import HELD_BLOCK_PADDING_MM, KEEPOUT_MARGIN_MM

MM_PER_M = 1000.0

DEPTH_PROBE_RADIUS_MM = 20.0


def centre_depth_mm(
    points_xyz_m: np.ndarray, radius_mm: float = DEPTH_PROBE_RADIUS_MM
) -> float | None:
    """Median camera-frame depth (z, in mm) of the points within ``radius_mm``
    of the optical axis. Looking straight down from a known height, this is
    the camera's own height - so the ratio to the expected value is the depth
    scale error (GPU run 15: detections landed ~10% too far along the ray)."""
    if points_xyz_m.size == 0:
        return None
    xy_mm = points_xyz_m[:, :2] * 1000.0
    near_axis = np.hypot(xy_mm[:, 0], xy_mm[:, 1]) <= radius_mm
    if not near_axis.any():
        return None
    return float(np.median(points_xyz_m[near_axis, 2]) * 1000.0)


# a resting block's centre must sit at support + size/2; a reading farther off
# than this is not the block (GPU run 10: a gripper-shadowed cube read z 115)
DETECT_Z_TOLERANCE_MM = 15.0

# tallest-estimator trust thresholds (seam: phase-4-tallest-carry.md, "Client
# measurement API")
# the fragment segmenter's segment_size_px: 100, the cell's smallest-credible-
# object constant - fewer in-region above-support points is not a block
MIN_TALLEST_REGION_POINTS = 100
# points at/below the support plus this are sensor noise, not object height
TALLEST_SUPPORT_EPSILON_MM = 1.0
# a 30 mm face at 900 mm range through 848 px / 90.5 deg intrinsics yields
# hundreds of points, so requiring 5 within this band of the max z is
# conservative - a lone stray point is not a block top
TALLEST_TOP_BAND_MM = 10.0
MIN_TALLEST_TOP_POINTS = 5


@dataclass
class TallestEstimate:
    tallest_mm: float
    points: int  # in-region points above the support plane
    trusted: bool
    reasons: list[str]  # empty when trusted


def tallest_in_region_mm(
    xyz_world_mm: np.ndarray,
    region_mm: tuple[Sequence[float], Sequence[float]] | None,
    support_z_mm: float,
    size_range_mm: tuple[float, float],
) -> TallestEstimate:
    """Tallest object height above ``support_z_mm`` in ``xyz_world_mm``
    (world frame, mm): clipped to ``region_mm``'s x/y footprint when given
    (None skips the clip and the quadrant-coverage check below), points at or
    below the support dropped before taking the max. Four independent trust
    checks each append a distinct reason on failure; ``trusted`` is true only
    when none do."""
    if region_mm is not None:
        (x0, y0, _z0), (x1, y1, _z1) = region_mm
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        in_region = (
            (xyz_world_mm[:, 0] >= lo_x)
            & (xyz_world_mm[:, 0] <= hi_x)
            & (xyz_world_mm[:, 1] >= lo_y)
            & (xyz_world_mm[:, 1] <= hi_y)
        )
        region_points = xyz_world_mm[in_region]
    else:
        region_points = xyz_world_mm

    reasons: list[str] = []

    if region_mm is not None:
        mid_x = (lo_x + hi_x) / 2.0
        mid_y = (lo_y + hi_y) / 2.0
        # counted BEFORE the support drop: a quadrant shadowed by a near
        # block has no support returns either - the side view's real failure
        quadrant_counts = [
            int(
                (
                    (region_points[:, 0] < mid_x if left else region_points[:, 0] >= mid_x)
                    & (region_points[:, 1] < mid_y if bottom else region_points[:, 1] >= mid_y)
                ).sum()
            )
            for left in (True, False)
            for bottom in (True, False)
        ]
        if any(count < 1 for count in quadrant_counts):
            reasons.append("region-quadrant coverage: a footprint quadrant has no in-region points")

    above_support = region_points[:, 2] > support_z_mm + TALLEST_SUPPORT_EPSILON_MM
    above_points = region_points[above_support]
    points = int(len(above_points))
    if points < MIN_TALLEST_REGION_POINTS:
        reasons.append(
            f"point floor: {points} in-region above-support points < {MIN_TALLEST_REGION_POINTS}"
        )

    tallest_mm = float(above_points[:, 2].max()) - support_z_mm if points > 0 else 0.0

    lo_size, hi_size = size_range_mm
    widened_lo = lo_size - DETECT_Z_TOLERANCE_MM
    widened_hi = hi_size + DETECT_Z_TOLERANCE_MM
    if not (widened_lo <= tallest_mm <= widened_hi):
        reasons.append(
            f"size window: tallest {tallest_mm:.1f} mm outside [{widened_lo:.1f}, {widened_hi:.1f}]"
        )

    near_top = (
        int((above_points[:, 2] >= float(above_points[:, 2].max()) - TALLEST_TOP_BAND_MM).sum())
        if points > 0
        else 0
    )
    if near_top < MIN_TALLEST_TOP_POINTS:
        reasons.append(
            f"lone-point top: {near_top} points within {TALLEST_TOP_BAND_MM} mm of the max z "
            f"< {MIN_TALLEST_TOP_POINTS}"
        )

    return TallestEstimate(
        tallest_mm=tallest_mm, points=points, trusted=not reasons, reasons=reasons
    )


# keep-out/carry derivation (seam): tallest + held-cube hang + margin. The
# hang fraction reproduces today's GPU-validated 60 mm-block numbers
# (keepout_height_mm(60, 60) == 130, carry_clear_above_support_mm(60, 60) ==
# 200); a held cube of a different size re-validates on GPU (phase 4
# checklist item 2).
KEEPOUT_HELD_HANG_FRACTION = 1.0 / 3.0
# the held padded cube's bottom clears the keep-out ceiling by this much once
# carried (GPU run 12: "~20 mm to spare" for the 60 mm case)
CARRY_KEEPOUT_CLEARANCE_MM = 21.0
# believed-vs-physical TCP gap already named at CARRY_CLEAR_ABOVE_SUPPORT_MM's
# definition (ARM-10)
CARRY_TCP_TO_CUBE_BOTTOM_OFFSET_MM = 9.0


def keepout_height_mm(tallest_mm: float, held_size_mm: float) -> float:
    """Pick-area keep-out ceiling height above the support: the tallest
    scattered object, plus room for the held cube's hang, plus
    KEEPOUT_MARGIN_MM reused as vertical margin."""
    return tallest_mm + held_size_mm * KEEPOUT_HELD_HANG_FRACTION + KEEPOUT_MARGIN_MM


def carry_clear_above_support_mm(tallest_mm: float, held_size_mm: float) -> float:
    """TCP height for the free-carry hop: the keep-out ceiling top, plus
    CARRY_KEEPOUT_CLEARANCE_MM, plus the held padded cube's own half-height
    and TCP offset so its bottom face clears the ceiling."""
    half_padded_held_mm = (held_size_mm + HELD_BLOCK_PADDING_MM) / 2.0
    return (
        keepout_height_mm(tallest_mm, held_size_mm)
        + CARRY_KEEPOUT_CLEARANCE_MM
        + CARRY_TCP_TO_CUBE_BOTTOM_OFFSET_MM
        + half_padded_held_mm
    )


def is_red_point(rgb: tuple[int, int, int], threshold: float = 0.5) -> bool:
    """Same rule as gpu_checklist_camera.is_red_pixel: r is at least 100 and
    both g and b are at most `threshold` fractions of r."""
    r, g, b = rgb
    return r >= 100 and g <= r * threshold and b <= r * threshold


def red_centroid_m(
    points_xyz: np.ndarray, colors_rgb: np.ndarray, threshold: float = 0.5
) -> tuple[float, float, float]:
    """Mean xyz (metres) of the points whose colour is "red" (see
    is_red_point). Raises ValueError when no point matches."""
    red_mask = np.array(
        [is_red_point((int(r), int(g), int(b)), threshold) for r, g, b in colors_rgb]
    )
    if not red_mask.any():
        raise ValueError("no red points found in the point cloud")
    centroid = points_xyz[red_mask].mean(axis=0)
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))


# points within this much of the nearest depth are "the top face"; 10 mm let the
# near side face's top edge in and pulled x 30 mm toward the camera (GPU run 18)
TOP_FACE_BAND_M = 0.002
MIN_BLOCK_DEPTH_M = 0.15  # nearer than this is the gripper itself, not the scene
MIN_RED_BAND_FRACTION = 0.3  # red points in the band are trusted only when there are enough


def top_face_centre_m(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    band_m: float = TOP_FACE_BAND_M,
    red_threshold: float = 0.5,
) -> tuple[float, float, float] | None:
    """Camera-frame centre (m) of the block's TOP FACE from a segment point
    cloud: keep red points (a detections-to-segments box also contains floor
    around the block, which drags a plain centroid away from the camera and
    down - GPU runs 13-16), then the nearest-depth band, i.e. the face seen
    straight down from the look pose. None when nothing red is left."""
    far_enough = xyz[:, 2] >= MIN_BLOCK_DEPTH_M  # the fingers sit ~90 mm in front of the camera
    if not far_enough.any():
        return None
    nearest = float(xyz[far_enough, 2].min())
    in_band = far_enough & (xyz[:, 2] <= nearest + band_m)
    chosen = in_band
    if rgb is not None and len(rgb) == len(xyz):
        # the lit top face can wash out past the red test (GPU run 21: only a
        # vertical face passed it), so red is a tie-breaker inside the band,
        # never the way to find the band
        r = rgb[:, 0].astype(float)
        red = (r >= 100) & (rgb[:, 1] <= r * red_threshold) & (rgb[:, 2] <= r * red_threshold)
        red_in_band = in_band & red
        if red_in_band.sum() >= MIN_RED_BAND_FRACTION * in_band.sum():
            chosen = red_in_band
    centre = xyz[chosen].mean(axis=0)
    return (float(centre[0]), float(centre[1]), float(centre[2]))


FOOTPRINT_TRIM_PCT_LO = 2.0
FOOTPRINT_TRIM_PCT_HI = 98.0
# a cube's three independent size readings (footprint x, footprint y,
# height) should agree; a bigger spread means a shadowed or edge-on view,
# not a block to grasp on (seam decision, phase 3)
MEASURED_SIZE_DEGENERATE_FRACTION = 0.25


def footprint_extents_mm(
    xyz: np.ndarray, band_m: float = TOP_FACE_BAND_M
) -> tuple[float, float] | None:
    """Top-face x/y footprint (mm) of the focused segment: the segment is
    already the detected block (the segmenter cuts it out of the detector's
    box), so measure the nearest-depth band - the same points
    top_face_centre_m trusts - each axis trimmed to the 2nd-98th percentile
    so a stray point cannot blow out the extent. Never select by redness:
    the lit top face washes out past any red test (GPU run 21; the phase-3
    checklist run saw red: 0 on every top-down scan). None when no point
    clears MIN_BLOCK_DEPTH_M."""
    far_enough = xyz[:, 2] >= MIN_BLOCK_DEPTH_M
    if not far_enough.any():
        return None
    nearest = float(xyz[far_enough, 2].min())
    band_xy = xyz[far_enough & (xyz[:, 2] <= nearest + band_m), :2]
    lo = np.percentile(band_xy, FOOTPRINT_TRIM_PCT_LO, axis=0)
    hi = np.percentile(band_xy, FOOTPRINT_TRIM_PCT_HI, axis=0)
    # the trim shaves 4% of a uniformly sampled extent (GPU: 57.3 mm measured
    # on a true 60 mm face) - rescale so a clean face measures true size
    trim_fraction = (FOOTPRINT_TRIM_PCT_HI - FOOTPRINT_TRIM_PCT_LO) / 100.0
    extent_mm = (hi - lo) * MM_PER_M / trim_fraction
    return (float(extent_mm[0]), float(extent_mm[1]))


def measured_block_size_mm(estimates: Sequence[float]) -> tuple[float, list[float]] | None:
    """Cube-prior size estimate: the median of independent size readings (a
    real detection uses footprint x, footprint y and height). None (a
    degenerate view) when any estimate strays more than
    MEASURED_SIZE_DEGENERATE_FRACTION of the median from it - advance the
    scan ladder instead of grasping on a bad number."""
    values = [float(v) for v in estimates]
    if any(v <= 0 for v in values):
        return None
    size_mm = float(np.median(values))
    if any(abs(v - size_mm) > MEASURED_SIZE_DEGENERATE_FRACTION * size_mm for v in values):
        return None
    return size_mm, values


def segment_stats(
    xyz: np.ndarray, rgb: np.ndarray | None, band_m: float = TOP_FACE_BAND_M
) -> dict[str, Any]:
    """What a segment is made of, camera frame: point counts (all / red /
    nearest-depth band) and the red points' extents in mm. Printed at detect
    so a biased block pose can be read off the segment's shape."""
    stats: dict[str, Any] = {"points": int(len(xyz)), "red": 0, "band": 0}
    if xyz.size == 0:
        return stats
    far_enough = xyz[:, 2] >= MIN_BLOCK_DEPTH_M
    scene = xyz[far_enough] if far_enough.any() else xyz
    nearest = float(scene[:, 2].min())
    in_band = scene[:, 2] <= nearest + band_m
    stats["band"] = int(in_band.sum())
    stats["band_min_mm"] = [round(float(v) * 1000.0, 1) for v in scene[in_band].min(axis=0)]
    stats["band_max_mm"] = [round(float(v) * 1000.0, 1) for v in scene[in_band].max(axis=0)]
    if rgb is not None and len(rgb) == len(xyz):
        r = rgb[:, 0].astype(float)
        red = (r >= 100) & (rgb[:, 1] <= r * 0.5) & (rgb[:, 2] <= r * 0.5)
        stats["red"] = int(red.sum())
        if red.any():
            stats["red_min_mm"] = [round(float(v) * 1000.0, 1) for v in xyz[red].min(axis=0)]
            stats["red_max_mm"] = [round(float(v) * 1000.0, 1) for v in xyz[red].max(axis=0)]
    return stats


def parse_pcd(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse binary `pointcloud/pcd` bytes written by
    isaac_module.encoding.xyz_rgb_to_pcd back into (xyz metres, rgb uint8 or
    None). Mirrors that function's header/body layout exactly."""
    data_marker = b"DATA binary\n"
    header_end = data.index(data_marker) + len(data_marker)
    header_fields: dict[str, list[str]] = {}
    for line in data[:header_end].decode("ascii").splitlines():
        key, _, rest = line.partition(" ")
        header_fields[key] = rest.split()

    field_names = header_fields["FIELDS"]
    num_points = int(header_fields["POINTS"][0])
    coloured = "rgb" in field_names

    if coloured:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

    points = np.frombuffer(data, dtype=dtype, count=num_points, offset=header_end)
    xyz = np.stack([points["x"], points["y"], points["z"]], axis=-1).astype(np.float32)
    if not coloured:
        return xyz, None

    packed = points["rgb"].astype(np.uint32)
    rgb = np.stack([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF], axis=-1).astype(
        np.uint8
    )
    return xyz, rgb
