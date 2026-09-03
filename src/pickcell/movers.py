"""``RealMover``: the ``Mover`` protocol implemented over the motion service."""

from __future__ import annotations

from viam.proto.common import Pose, PoseInFrame, WorldState
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from pickcell.poses import _pose_to_dict

LINEAR_LINE_TOLERANCE_MM = 10.0
LINEAR_ORIENTATION_TOLERANCE_DEG = 10.0


class RealMover:
    """Drives the motion service. At viam-sdk 0.80 ``MotionClient.move`` takes
    the component NAME as a plain string (``MoveRequest.component_name`` is a
    string; the ResourceName form is ``component_name_deprecated``)."""

    def __init__(self, motion: MotionClient, gripper_name: str, camera_name: str) -> None:
        self._motion = motion
        self._gripper_name = gripper_name
        self._camera_name = camera_name

    async def look_from(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        await self._move_frame(self._camera_name, pose, world_state, linear)

    async def move_to(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        await self._move_frame(self._gripper_name, pose, world_state, linear)

    async def _move_frame(
        self, component_name: str, pose: Pose, world_state: WorldState, linear: bool = False
    ) -> None:
        destination = PoseInFrame(reference_frame="world", pose=pose)
        constraints = (
            Constraints(
                linear_constraint=[
                    LinearConstraint(
                        line_tolerance_mm=LINEAR_LINE_TOLERANCE_MM,
                        orientation_tolerance_degs=LINEAR_ORIENTATION_TOLERANCE_DEG,
                    )
                ]
            )
            if linear
            else None
        )
        success = await self._motion.move(
            component_name=component_name,
            destination=destination,
            world_state=world_state,
            constraints=constraints,
        )
        if not success:
            raise RuntimeError(
                f"motion move of {component_name!r} to {_pose_to_dict(pose)} reported failure"
            )
