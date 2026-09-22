"""``RealMover``'s constraints: what each move mode asks the planner for."""

from __future__ import annotations

from typing import Any

from viam.proto.common import Pose, WorldState

from pickcell.movers import (
    CARRY_ORIENTATION_TOLERANCE_DEG,
    LINEAR_LINE_TOLERANCE_MM,
    LINEAR_ORIENTATION_TOLERANCE_DEG,
    RealMover,
    move_constraints,
)


class _FakeMotion:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def move(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        return True


POSE = Pose(x=1.0, y=2.0, z=3.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)


def test_a_free_move_carries_no_constraint():
    assert move_constraints(linear=False, level=False) is None


def test_a_linear_move_carries_the_line_and_its_own_orientation_tolerance():
    constraints = move_constraints(linear=True, level=False)
    assert constraints is not None
    (line,) = constraints.linear_constraint
    assert line.line_tolerance_mm == LINEAR_LINE_TOLERANCE_MM
    assert line.orientation_tolerance_degs == LINEAR_ORIENTATION_TOLERANCE_DEG
    assert list(constraints.orientation_constraint) == []


def test_a_level_move_keeps_the_tools_orientation_and_leaves_the_path_free():
    """A free plan between two pointing-down poses whose wrist solutions
    differ by 180 degrees is one segment that turns the payload over the arm.
    Level forbids the turn without asking for a straight line."""
    constraints = move_constraints(linear=False, level=True)
    assert constraints is not None
    (orientation,) = constraints.orientation_constraint
    assert orientation.orientation_tolerance_degs == CARRY_ORIENTATION_TOLERANCE_DEG
    assert list(constraints.linear_constraint) == []


def test_a_linear_move_is_not_also_given_the_level_constraint():
    constraints = move_constraints(linear=True, level=True)
    assert constraints is not None
    assert len(constraints.linear_constraint) == 1
    assert list(constraints.orientation_constraint) == []


async def test_real_mover_hands_the_level_constraint_to_the_motion_service():
    motion = _FakeMotion()
    mover = RealMover(motion, "gripper-1", "camera-1")  # type: ignore[arg-type]

    await mover.move_to(POSE, WorldState(), level=True)

    (call,) = motion.calls
    assert call["component_name"] == "gripper-1"
    (orientation,) = call["constraints"].orientation_constraint
    assert orientation.orientation_tolerance_degs == CARRY_ORIENTATION_TOLERANCE_DEG


async def test_real_mover_default_move_is_unconstrained():
    motion = _FakeMotion()
    mover = RealMover(motion, "gripper-1", "camera-1")  # type: ignore[arg-type]

    await mover.move_to(POSE, WorldState())

    (call,) = motion.calls
    assert call["constraints"] is None
