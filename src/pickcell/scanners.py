"""Real-camera ``TallestScanner`` collaborators: the fixed side camera
(primary) and, aimed at a wrist-sweep vantage, the wrist camera (fallback),
plus the camera->world affine both are built from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from viam.proto.common import Pose, PoseInFrame

from pickcell.measurement import MM_PER_M, parse_pcd

if TYPE_CHECKING:
    from viam.robot.client import RobotClient


async def camera_world_transform_mm(robot: Any, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(R 3x3, t mm) camera-to-world affine from 4 ``transform_pose`` probes:
    the camera-frame origin and +100 mm x/y/z offsets. Applied as
    ``world_mm = xyz_cam_m * 1000 @ R.T + t`` - no orientation-vector math,
    just the measured effect of the frame transform on known offsets."""

    async def probe(x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
        camera_pif = PoseInFrame(
            reference_frame=camera_name,
            pose=Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        )
        pose = (await robot.transform_pose(camera_pif, "world")).pose
        return np.array([pose.x, pose.y, pose.z])

    origin = await probe(0.0, 0.0, 0.0)
    x_probe = await probe(100.0, 0.0, 0.0)
    y_probe = await probe(0.0, 100.0, 0.0)
    z_probe = await probe(0.0, 0.0, 100.0)
    rotation = np.column_stack(
        [(x_probe - origin) / 100.0, (y_probe - origin) / 100.0, (z_probe - origin) / 100.0]
    )
    return rotation, origin


async def _camera_client(robot: RobotClient, camera_name: str) -> Any:
    """The resource list is a snapshot taken at connect; a module that was
    still rebuilding the camera then (a redeploy) is missing from it until a
    refresh. On a real miss, name the cameras the machine does report."""
    from viam.components.camera import Camera
    from viam.errors import ResourceNotFoundError

    await robot.refresh()
    try:
        return Camera.from_robot(robot, camera_name)
    except ResourceNotFoundError:
        cameras = sorted(
            name.name
            for name in robot.resource_names
            if name.subtype == Camera.API.resource_subtype
        )
        raise RuntimeError(
            f"camera {camera_name!r} is not in the machine's resource list (cameras present: "
            f"{cameras}) - check the component's status on the machine page"
        ) from None


class FixedCameraScanner:
    """A ``TallestScanner`` over a real camera: grabs its point cloud, then
    transforms the camera-frame points to world mm through the measured
    camera->world affine (``camera_world_transform_mm``). Used both for the
    fixed side camera (primary) and, aimed at a wrist-sweep vantage, the
    wrist camera (fallback)."""

    def __init__(self, robot: RobotClient, camera_name: str) -> None:
        self._robot = robot
        self._camera_name = camera_name

    async def scan_world_mm(self) -> np.ndarray:
        camera = await _camera_client(self._robot, self._camera_name)
        pcd_bytes, _mime = await camera.get_point_cloud()
        xyz_m, _rgb = parse_pcd(pcd_bytes)
        rotation, translation = await camera_world_transform_mm(self._robot, self._camera_name)
        world_mm = xyz_m * MM_PER_M @ rotation.T + translation
        valid = ~np.isnan(world_mm).any(axis=1)
        return world_mm[valid]
