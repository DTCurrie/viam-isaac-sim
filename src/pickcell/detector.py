"""``RealDetector``: the ``Detector`` protocol implemented over a vision
segmenter segment chosen by ``select_segment``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from viam.proto.common import Pose, PoseInFrame
from viam.services.vision import VisionClient

from pickcell.measurement import (
    footprint_extents_mm,
    measured_block_size_mm,
    parse_pcd,
    segment_stats,
    top_face_centre_m,
)
from pickcell.poses import _pose_to_dict, with_z

if TYPE_CHECKING:
    from viam.robot.client import RobotClient


# with centred selection: a segment farther than this from the camera's
# boresight is never the target (a census pose error is <= ~40 mm while the
# pool's minimum block spacing is 120 mm, so 100 splits them; GPU seed-7: a
# bigger same-band NEIGHBOR at the frame edge out-competed the centred target
# under largest-segment selection and wedged the pick)
CENTRED_SEGMENT_MAX_OFFSET_MM = 100.0


def select_segment(objects: list[Any], prefer_centred: bool) -> Any:
    """The segment a detection locks onto: the largest point cloud by
    default, or with ``prefer_centred`` the one nearest the camera's
    boresight (planar camera-frame offset), for callers whose look pose is
    already centred on the target.

    @throws RuntimeError with ``prefer_centred`` when no segment sits within
    CENTRED_SEGMENT_MAX_OFFSET_MM of the boresight."""
    if not prefer_centred:
        return max(objects, key=lambda obj: len(obj.point_cloud))

    def boresight_offset_mm(obj: Any) -> float:
        centre = obj.geometries.geometries[0].center
        return float((centre.x**2 + centre.y**2) ** 0.5)

    nearest = min(objects, key=boresight_offset_mm)
    offset = boresight_offset_mm(nearest)
    if offset > CENTRED_SEGMENT_MAX_OFFSET_MM:
        raise RuntimeError(
            f"no segment within {CENTRED_SEGMENT_MAX_OFFSET_MM:.0f} mm of the look axis "
            f"(nearest at {offset:.0f} mm) - the looked-at block is missing or moved"
        )
    return nearest


class RealDetector:
    """Block pose from a vision-service segment: the color points' top face
    gives x/y and the top height; the block centre is size/2 below it. The
    segmenter's own centre is printed for comparison.

    Segment choice is ``select_segment``: largest by default;
    ``prefer_centred`` selects nearest the boresight instead, for callers
    (the conductor's per-pick verify) whose look pose is centred on the
    target and whose frame may contain bigger neighbors.

    ``block_size_mm`` None measures the size from the same segment
    (footprint x/y extents plus top-face-minus-support height, cube-prior
    median) instead of trusting a caller-supplied value; the measurement is
    cached for ``last_measurement`` (None = explicit size, no measurement)."""

    def __init__(
        self,
        robot: RobotClient,
        vision: VisionClient,
        camera_name: str,
        block_size_mm: float | None,
        support_z_mm: float = 0.0,
        prefer_centred: bool = False,
    ) -> None:
        self._robot = robot
        self._vision = vision
        self._camera_name = camera_name
        self._block_size_mm = block_size_mm
        self._support_z_mm = support_z_mm
        self._prefer_centred = prefer_centred
        self._last_measurement: dict[str, Any] | None = None

    def last_measurement(self) -> dict[str, Any] | None:
        return self._last_measurement

    async def block_pose_world(self) -> Pose:
        objects = await self._vision.get_object_point_clouds(self._camera_name)
        if not objects:
            raise RuntimeError(
                f"vision service returned no segments for camera {self._camera_name!r}"
            )
        selected = select_segment(list(objects), self._prefer_centred)
        segment_centre = selected.geometries.geometries[0].center
        xyz, rgb = parse_pcd(selected.point_cloud)
        print(f"  segment (camera frame): {segment_stats(xyz, rgb)}")
        top = top_face_centre_m(xyz, rgb)
        if top is None:
            raise RuntimeError("segment has no points to take a top face from")
        top_world = await self._to_world(top[0] * 1000.0, top[1] * 1000.0, top[2] * 1000.0)
        segment_world = await self._to_world(segment_centre.x, segment_centre.y, segment_centre.z)
        print(
            f"  segmenter centre (world, mm): {_pose_to_dict(segment_world)}; "
            f"top face centre: {_pose_to_dict(top_world)} from {len(xyz)} points"
        )
        if self._block_size_mm is not None:
            self._last_measurement = None
            size_mm = self._block_size_mm
        else:
            footprint_mm = footprint_extents_mm(xyz)
            height_mm = top_world.z - self._support_z_mm
            measured = (
                measured_block_size_mm([footprint_mm[0], footprint_mm[1], height_mm])
                if footprint_mm is not None
                else None
            )
            if measured is None:
                if footprint_mm is None:
                    print("  size estimates: no top-face band points")
                else:
                    print(
                        f"  size estimates (mm): footprint [{footprint_mm[0]:.1f}, "
                        f"{footprint_mm[1]:.1f}], height {height_mm:.1f} (top z - "
                        f"support z {self._support_z_mm:.0f})"
                    )
                self._last_measurement = None
                size_mm = 0.0  # degenerate: the caller re-scans and discards this pose
            else:
                assert footprint_mm is not None
                size_mm, estimates = measured
                self._last_measurement = {
                    "footprint_mm": [footprint_mm[0], footprint_mm[1]],
                    "height_mm": estimates[2],
                    "size_mm": size_mm,
                }
        return with_z(top_world, top_world.z - size_mm / 2.0)

    async def _to_world(self, x_mm: float, y_mm: float, z_mm: float) -> Pose:
        camera_pif = PoseInFrame(
            reference_frame=self._camera_name,
            pose=Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        )
        return (await self._robot.transform_pose(camera_pif, "world")).pose
