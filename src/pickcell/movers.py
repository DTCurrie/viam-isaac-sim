"""``RealMover``: the ``Mover`` protocol implemented over the motion service."""

from __future__ import annotations

from typing import Any

from viam.proto.common import Pose, PoseInFrame, WorldState
from viam.proto.service.motion import Constraints, LinearConstraint, OrientationConstraint
from viam.services.motion import MotionClient

from pickcell.poses import _pose_to_dict

LINEAR_LINE_TOLERANCE_MM = 10.0
LINEAR_ORIENTATION_TOLERANCE_DEG = 10.0

# how far a level carry may tip on the way. A cup keeps hold of a box tipped
# this far, and what the constraint exists to forbid is the wrist turning the
# tool over: a free plan between two pointing-down poses whose wrist 2
# solutions differ by 180 degrees is one segment that swings the payload
# over the top, through the arm's own forearm (measured 2026-09-22).
CARRY_ORIENTATION_TOLERANCE_DEG = 15.0

# how long the planner may spend on a straight-line move before it counts as
# refused, in seconds, handed to the motion service as extra["timeout"]
# (PlannerOptions.Timeout in RDK's motionplan/armplanning, default 300). A
# straight line that exists plans in a second or two: the 2026-09-22 descents
# did. One that does not, from whichever wrist or base branch the standoff
# landed in, is what the caller's free fallback is for, and on 2026-09-24 the
# default let the constrained planner search for the full five minutes and
# the run died before the fallback could move. Thirty seconds is an order of
# magnitude over a real plan and a tenth of the default.
LINEAR_PLAN_TIMEOUT_S = 30.0
# and for a free move: every free plan that has ever succeeded in this cell
# did so inside 8 s, and on 2026-09-24 a carry the planner could not route
# round a keep-out ran the full 300 s default before the box was reported
# failed. A minute is an order of magnitude over the real plans.
FREE_PLAN_TIMEOUT_S = 60.0


def move_constraints(linear: bool, level: bool) -> Constraints | None:
    """The planner constraints for one move. A straight line carries its own
    orientation tolerance, so `level` only adds one when the path is free:
    the tool's orientation stays within CARRY_ORIENTATION_TOLERANCE_DEG of
    the line between its start and goal orientations all the way along.
    Neither flag means an unconstrained move."""
    if linear:
        return Constraints(
            linear_constraint=[
                LinearConstraint(
                    line_tolerance_mm=LINEAR_LINE_TOLERANCE_MM,
                    orientation_tolerance_degs=LINEAR_ORIENTATION_TOLERANCE_DEG,
                )
            ]
        )
    if level:
        return Constraints(
            orientation_constraint=[
                OrientationConstraint(orientation_tolerance_degs=CARRY_ORIENTATION_TOLERANCE_DEG)
            ]
        )
    return None


def move_extra(
    linear: bool, planner_seed: int | None = None, plan_timeout_s: float | None = None
) -> dict[str, Any]:
    """The motion service's `extra` for one move: a planning time bound,
    short on a straight line (whose refusal has a fallback waiting) and a
    minute on a free move (whose refusal is the box's failure).

    The planner seeds every plan with 0 unless `rseed` says otherwise, so
    identical inputs give an identical plan, and a plan that timed out times
    out again. `planner_seed` is how a caller asks for another draw.
    `plan_timeout_s` replaces the per-kind bound for this one move."""
    if plan_timeout_s is None:
        plan_timeout_s = LINEAR_PLAN_TIMEOUT_S if linear else FREE_PLAN_TIMEOUT_S
    extra: dict[str, Any] = {"timeout": plan_timeout_s}
    if planner_seed is not None:
        extra["rseed"] = planner_seed
    return extra


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

    async def move_to(
        self,
        pose: Pose,
        world_state: WorldState,
        linear: bool = False,
        level: bool = False,
        *,
        planner_seed: int | None = None,
        plan_timeout_s: float | None = None,
    ) -> None:
        await self._move_frame(
            self._gripper_name, pose, world_state, linear, level, planner_seed, plan_timeout_s
        )

    async def _move_frame(
        self,
        component_name: str,
        pose: Pose,
        world_state: WorldState,
        linear: bool = False,
        level: bool = False,
        planner_seed: int | None = None,
        plan_timeout_s: float | None = None,
    ) -> None:
        destination = PoseInFrame(reference_frame="world", pose=pose)
        constraints = move_constraints(linear, level)
        success = await self._motion.move(
            component_name=component_name,
            destination=destination,
            world_state=world_state,
            constraints=constraints,
            extra=move_extra(linear, planner_seed, plan_timeout_s),
        )
        if not success:
            raise RuntimeError(
                f"motion move of {component_name!r} to {_pose_to_dict(pose)} reported failure"
            )
