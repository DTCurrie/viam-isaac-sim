"""Pointing-down/scan/focus/grasp pose builders and the constants they derive
from - the geometry a pick needs, with no client-SDK or CLI dependency."""

from __future__ import annotations

from collections.abc import Sequence

from viam.proto.common import Pose

PRE_GRASP_STANDOFF_MM = 100.0
POINTING_DOWN_O_Z = -1.0


def _pointing_down(x: float, y: float, z: float, theta_deg: float = 0.0) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=POINTING_DOWN_O_Z, theta=theta_deg)


# the wide scan height above measures poorly; a detection focuses down to
# this height instead (checklist item 3)
FOCUS_HEIGHT_ABOVE_SUPPORT_MM = 350.0
# the 848x480 wrist camera's fov_deg is horizontal, so vertical half-coverage
# is tan(fov/2) * 480/848 of depth; at this height (depth ~= 650 - 60 to the
# block top) that is 0.571 * 590 =~ 337 mm, covering the 650x600 zone's edge
# blocks (|y| <= 270 plus half a face) (checklist item 3, GPU run 20: the
# 350 mm scan height clipped an edge block's top face and never recovered)
SCAN_HEIGHT_ABOVE_SUPPORT_MM = 650.0


def look_pose_from(xyz_mm: str) -> Pose:
    """The wrist-camera pose to detect from: at ``x,y,z`` (mm, world) with the
    optical axis pointing straight down (o_z = -1), so the block region is in
    view. At the UR5e zero pose the camera looks along world -Y, away from it."""
    x, y, z = (float(v) for v in xyz_mm.split(","))
    return Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=POINTING_DOWN_O_Z, theta=0.0)


def pre_grasp_pose(block_pose: Pose, standoff_mm: float = PRE_GRASP_STANDOFF_MM) -> Pose:
    """The stationary pose the arm detects from and lifts back to: directly
    above the block by ``standoff_mm``, gripper pointing straight down."""
    return _pointing_down(block_pose.x, block_pose.y, block_pose.z + standoff_mm)


# Measured on the GPU (phase 3, item 4): the 2F-85 pads reach 153 mm along the
# tool axis with the TCP at their centre, 134 mm, so the fingertips extend 19 mm
# past the TCP. A grasp lower than support + overhang drives them into the table.
FINGERTIP_OVERHANG_MM = 19.0
# GPU run 19: the arm model and Isaac disagree by ~10-15 mm in z at the grasp
# configuration, and the pads (38 mm tall) still cover 2/3 of a 60 mm block
# when the TCP sits 39 mm up - so leave real room above the support.
FINGERTIP_CLEARANCE_MM = 20.0


def grasp_height_mm(
    detected_z_mm: float,
    block_size_mm: float,
    support_z_mm: float,
    fingertip_overhang_mm: float = FINGERTIP_OVERHANG_MM,
    clearance_mm: float = FINGERTIP_CLEARANCE_MM,
) -> float:
    """The TCP height to grasp at. A block resting on its support cannot have
    its centre below support + size/2 (a depth centroid seen from above lands
    low), and the TCP cannot go below support + overhang + clearance without
    the fingertips hitting the support."""
    centre_floor = support_z_mm + block_size_mm / 2.0
    fingertip_floor = support_z_mm + fingertip_overhang_mm + clearance_mm
    return max(detected_z_mm, centre_floor, fingertip_floor)


# The frame system and Isaac disagree by ~17-19 mm in z at the grasp
# configuration while agreeing at the look pose (GPU runs 19-22, ARM-10
# follow-up). Measured at the pre-grasp pose (believed TCP vs the physical pad
# centre) and applied to the grasp/lift targets; capped so a bad reading
# cannot command a wild pose.
TCP_CORRECTION_CAP_MM = 40.0


def corrected_pose(
    pose: Pose, delta_mm: tuple[float, float, float], cap_mm: float = TCP_CORRECTION_CAP_MM
) -> Pose:
    """``pose`` shifted by the measured believed-minus-physical TCP offset, so
    the physical pads land where the plan intended. Each axis is clamped to
    +/- cap_mm."""
    dx, dy, dz = (max(-cap_mm, min(cap_mm, v)) for v in delta_mm)
    return Pose(
        x=pose.x + dx,
        y=pose.y + dy,
        z=pose.z + dz,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )


def with_z(pose: Pose, z_mm: float) -> Pose:
    return Pose(
        x=pose.x, y=pose.y, z=z_mm, o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta
    )


def grasp_pose(block_pose: Pose) -> Pose:
    """The block's own pose, gripper pointing straight down - no standoff."""
    return _pointing_down(block_pose.x, block_pose.y, block_pose.z)


# retry ladder for a gripper-shadowed detection: the fingers hang ~150 mm
# below the wrist camera, ~80 mm off its axis, so a quarter wrist turn sweeps
# the shadow to a new direction while the camera stays put; stepping the
# camera sideways is the last resort. Entries are (x offset, y offset, theta).
SCAN_ATTEMPTS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 90.0),
    (0.0, 0.0, 180.0),
    (0.0, 0.0, 270.0),
    (150.0, 150.0, 0.0),
    (-150.0, -150.0, 0.0),
)

TALLEST_SWEEP_CORNER_INSET_MM = 50.0


def tallest_sweep_attempts(
    region_mm: tuple[Sequence[float], Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """The wrist-sweep fallback ladder for tallest measurement: 4 region-corner
    vantages (offsets from the region centre, inset
    TALLEST_SWEEP_CORNER_INSET_MM from each corner, theta 0) FIRST, then
    SCAN_ATTEMPTS. Corners lead because a region-centre vantage hangs the
    camera's own gripper inside the region footprint, where it reads as a
    ~274 mm object (GPU phase-4 run 1: all four centre vantages discarded);
    a corner vantage keeps the arm outside the footprint."""
    (x0, y0, _z0), (x1, y1, _z1) = region_mm
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    centre_x = (lo_x + hi_x) / 2.0
    centre_y = (lo_y + hi_y) / 2.0
    inset = TALLEST_SWEEP_CORNER_INSET_MM
    corners = (
        (lo_x + inset - centre_x, lo_y + inset - centre_y, 0.0),
        (hi_x - inset - centre_x, lo_y + inset - centre_y, 0.0),
        (lo_x + inset - centre_x, hi_y - inset - centre_y, 0.0),
        (hi_x - inset - centre_x, hi_y - inset - centre_y, 0.0),
    )
    return corners + SCAN_ATTEMPTS


def _poses_match_mm(pose: Pose | None, other: Pose, tolerance_mm: float) -> bool:
    """Whether ``pose`` and ``other`` sit within ``tolerance_mm`` on every axis."""
    if pose is None:
        return False
    return (
        abs(pose.x - other.x) <= tolerance_mm
        and abs(pose.y - other.y) <= tolerance_mm
        and abs(pose.z - other.z) <= tolerance_mm
    )


def _pose_to_dict(pose: Pose) -> dict[str, float]:
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "o_x": pose.o_x,
        "o_y": pose.o_y,
        "o_z": pose.o_z,
        "theta": pose.theta,
    }
