"""Frame-system parts translated into static colliders."""

from __future__ import annotations

import logging
import math

import pytest
from viam.proto.common import Geometry, Pose, PoseInFrame, RectangularPrism, Transform, Vector3
from viam.proto.robot import FrameSystemConfig

from isaac_module.frame_system import collider_props, world_parts
from isaac_module.spatial import quat_rotate

LOGGER = logging.getLogger(__name__)

# an orientation vector in degrees, the shape a frame block's ov_degrees
# arrives in: (o_x, o_y, o_z, theta)
Ov = tuple[float, float, float, float]

YAW_90: Ov = (0.0, 0.0, 1.0, 90.0)


def _pose(xyz: tuple[float, float, float], ov: Ov | None) -> Pose:
    if ov is None:
        return Pose(x=xyz[0], y=xyz[1], z=xyz[2])
    return Pose(x=xyz[0], y=xyz[1], z=xyz[2], o_x=ov[0], o_y=ov[1], o_z=ov[2], theta=ov[3])


def _part(
    name: str,
    *,
    parent: str = "world",
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: Ov | None = None,
    dims: tuple[float, float, float] | None = None,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    centre_orientation: Ov | None = None,
) -> FrameSystemConfig:
    geometry = Geometry()
    if dims is not None:
        geometry.box.CopyFrom(RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])))
        geometry.center.CopyFrom(_pose(centre, centre_orientation))
    return FrameSystemConfig(
        frame=Transform(
            reference_frame=name,
            pose_in_observer_frame=PoseInFrame(
                reference_frame=parent, pose=_pose(pose, orientation)
            ),
            physical_object=geometry,
        )
    )


def _yaw_deg(quat_wxyz: tuple[float, float, float, float]) -> float:
    """Where the collider's own +x axis points, in degrees about world z."""
    x_axis = quat_rotate(quat_wxyz, (1.0, 0.0, 0.0))
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


def test_a_part_with_no_geometry_contributes_no_collider():
    props = collider_props([_part("scene-cam", pose=(0.0, 2500.0, 1500.0))], logger=LOGGER)
    assert props == []


def test_a_box_part_becomes_a_fixed_cube_at_its_world_pose_in_metres():
    part = _part("pallet", pose=(200.0, 500.0, 200.0), dims=(500.0, 350.0, 100.0))
    (prop,) = collider_props([part], logger=LOGGER)
    assert prop["name"] == "frame-pallet"
    assert prop["position"] == (0.2, 0.5, 0.2)
    assert prop["orientation_wxyz"] == (1.0, 0.0, 0.0, 0.0)
    assert prop["scale"] == (0.5, 0.35, 0.1)
    assert prop["fixed"] is True


def test_a_frame_turned_about_z_turns_its_collider_with_it():
    # fence-left as the vendored fragment declares it: a 1200 mm panel whose
    # frame turns 90 degrees about z, so the panel runs along y. The GPU
    # viewport of 2026-09-22 showed it running along x, across the cell,
    # because the frame's orientation never reached the collider.
    part = _part(
        "fence-left",
        pose=(-1150.0, -350.0, 0.0),
        orientation=YAW_90,
        dims=(1200.0, 36.0, 1180.0),
        centre=(0.0, 0.0, 625.0),
    )
    (prop,) = collider_props([part], logger=LOGGER)
    assert prop["position"] == pytest.approx((-1.15, -0.35, 0.625))
    assert prop["scale"] == (1.2, 0.036, 1.18)
    assert _yaw_deg(prop["orientation_wxyz"]) == pytest.approx(90.0)
    # the panel's long axis, its own x, ends up along world y
    long_axis = quat_rotate(prop["orientation_wxyz"], (1.0, 0.0, 0.0))
    assert long_axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_the_geometry_centre_is_an_offset_in_the_turned_frame():
    # a centre 100 mm along the frame's own x lands along world y once the
    # frame turns 90 degrees. Adding translations would put it along world x.
    part = _part(
        "turned",
        pose=(0.0, 0.0, 0.0),
        orientation=YAW_90,
        dims=(10.0, 10.0, 10.0),
        centre=(100.0, 0.0, 0.0),
    )
    (prop,) = collider_props([part], logger=LOGGER)
    assert prop["position"] == pytest.approx((0.0, 0.1, 0.0), abs=1e-9)


def test_the_geometry_own_orientation_composes_onto_the_frame():
    # a frame turned 90 degrees carrying a box turned another 45 within it
    # is a box turned 135 degrees in the world
    part = _part(
        "twice",
        orientation=YAW_90,
        dims=(10.0, 10.0, 10.0),
        centre_orientation=(0.0, 0.0, 1.0, 45.0),
    )
    (prop,) = collider_props([part], logger=LOGGER)
    assert _yaw_deg(prop["orientation_wxyz"]) == pytest.approx(135.0)


def test_the_geometry_centre_offsets_the_collider_from_the_frame_origin():
    # robot-pedestal stands ON its frame origin, so its box centre is half its
    # height above it. Ignoring the centre would sink it into the floor.
    part = _part(
        "robot-pedestal", pose=(0.0, 0.0, 0.0), dims=(220.0, 220.0, 150.0), centre=(0.0, 0.0, 75.0)
    )
    (prop,) = collider_props([part], logger=LOGGER)
    assert prop["position"] == (0.0, 0.0, 0.075)


def test_a_zero_sized_box_is_not_a_collider():
    part = _part("caution-tape", dims=(0.0, 0.0, 0.0))
    assert collider_props([part], logger=LOGGER) == []


def test_world_parts_drops_anything_posed_against_another_frame():
    # wrist-cam declares a geometry but is parented to arm-1, so its pose is
    # in the ARM's frame. Treating it as a world pose puts a box at the origin.
    parts = [
        _part("pallet", pose=(200.0, 500.0, 200.0), dims=(500.0, 350.0, 100.0)),
        _part("wrist-cam", parent="arm-1", pose=(0.0, 0.0, 60.0), dims=(90.0, 25.0, 25.0)),
    ]
    kept = [p.frame.reference_frame for p in world_parts(parts)]
    assert kept == ["pallet"]
