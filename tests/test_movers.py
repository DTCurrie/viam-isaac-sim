"""``RealMover``'s constraints: what each move mode asks the planner for."""

from __future__ import annotations

from typing import Any

from viam.proto.common import Pose, WorldState

from pickcell.movers import (
    CARRY_ORIENTATION_TOLERANCE_DEG,
    FREE_PLAN_TIMEOUT_S,
    LINEAR_LINE_TOLERANCE_MM,
    LINEAR_ORIENTATION_TOLERANCE_DEG,
    LINEAR_PLAN_TIMEOUT_S,
    RealMover,
    move_constraints,
    move_extra,
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


def test_a_straight_line_bounds_the_planners_time_shorter_than_a_free_move():
    assert move_extra(linear=True) == {"timeout": LINEAR_PLAN_TIMEOUT_S}
    assert move_extra(linear=False) == {"timeout": FREE_PLAN_TIMEOUT_S}
    assert LINEAR_PLAN_TIMEOUT_S < FREE_PLAN_TIMEOUT_S


async def test_real_mover_hands_the_planning_bound_to_the_motion_service_on_a_line():
    """On 2026-09-24 a straight descent from a wound standoff ran the
    constrained planner to its 300 s default and the run died before the
    free fallback could move. The bound is what gives the fallback its turn."""
    motion = _FakeMotion()
    mover = RealMover(motion, "gripper-1", "camera-1")  # type: ignore[arg-type]

    await mover.move_to(POSE, WorldState(), linear=True)
    await mover.move_to(POSE, WorldState())

    linear_call, free_call = motion.calls
    assert linear_call["extra"] == {"timeout": LINEAR_PLAN_TIMEOUT_S}
    assert free_call["extra"] == {"timeout": FREE_PLAN_TIMEOUT_S}


def test_a_planner_seed_rides_in_the_extra_only_when_given():
    assert move_extra(linear=False, planner_seed=None) == {"timeout": FREE_PLAN_TIMEOUT_S}
    assert move_extra(linear=False, planner_seed=1) == {
        "timeout": FREE_PLAN_TIMEOUT_S,
        "rseed": 1,
    }
    assert move_extra(linear=True, planner_seed=0) == {
        "timeout": LINEAR_PLAN_TIMEOUT_S,
        "rseed": 0,
    }


async def test_real_mover_hands_the_planner_seed_to_the_motion_service():
    motion = _FakeMotion()
    mover = RealMover(motion, "gripper-1", "camera-1")  # type: ignore[arg-type]

    await mover.move_to(POSE, WorldState(), planner_seed=7)
    await mover.move_to(POSE, WorldState())

    seeded, unseeded = motion.calls
    assert seeded["extra"] == {"timeout": FREE_PLAN_TIMEOUT_S, "rseed": 7}
    assert unseeded["extra"] == {"timeout": FREE_PLAN_TIMEOUT_S}


def test_a_plan_timeout_replaces_the_per_kind_bound():
    assert move_extra(False, plan_timeout_s=10.0) == {"timeout": 10.0}
    assert move_extra(False, planner_seed=1, plan_timeout_s=10.0) == {
        "timeout": 10.0,
        "rseed": 1,
    }


async def test_real_mover_hands_the_plan_timeout_to_the_motion_service():
    motion = _FakeMotion()
    mover = RealMover(motion, "gripper-1", "camera-1")  # type: ignore[arg-type]

    await mover.move_to(POSE, WorldState(), plan_timeout_s=10.0)

    assert motion.calls[0]["extra"] == {"timeout": 10.0}
