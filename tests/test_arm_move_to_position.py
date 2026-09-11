import asyncio
import json
import math
from pathlib import Path

import pytest
from grpclib import Status
from viam.components.arm import JointPositions, Pose
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.kinematics import Chain
from isaac_module.models.arm import (
    IsaacArm,
    JointTargetOutOfLimitsError,
    KinematicsUnavailableError,
    PoseUnreachableError,
)
from isaac_module.spatial import ov_to_quat, quat_to_ov

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "kinematics"
UR5E_PATH = FIXTURES_DIR / "ur5e.json"
XARM7_PATH = FIXTURES_DIR / "xarm7.json"

_POSITION_TOL_M = 1e-3
_ANGULAR_TOL_DEG = 0.5


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _ur5e_arm(world, name: str = "arm-move-to-position") -> IsaacArm:
    return IsaacArm.new(
        _config(
            name,
            {
                "world": "isaac-world",
                "asset": "ur5e",
                "mock_dof": 6,
                "kinematics_url": UR5E_PATH.as_uri(),
            },
        ),
        {},
    )


def _quat_angle_diff_deg(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return math.degrees(2 * math.acos(min(1.0, abs(dot))))


def _pose_from_fk(chain: Chain, q_deg: list[float]) -> Pose:
    pos, quat = chain.fk([math.radians(v) for v in q_deg])
    ox, oy, oz, theta = quat_to_ov(quat)
    return Pose(
        x=pos[0] * 1000.0,
        y=pos[1] * 1000.0,
        z=pos[2] * 1000.0,
        o_x=ox,
        o_y=oy,
        o_z=oz,
        theta=math.degrees(theta),
    )


def _assert_fk_matches(chain: Chain, current_deg: list[float], target: Pose) -> None:
    pos, quat = chain.fk([math.radians(v) for v in current_deg])
    target_pos = (target.x / 1000.0, target.y / 1000.0, target.z / 1000.0)
    target_quat = ov_to_quat(target.o_x, target.o_y, target.o_z, math.radians(target.theta))
    assert math.dist(pos, target_pos) < _POSITION_TOL_M
    assert _quat_angle_diff_deg(quat, target_quat) < _ANGULAR_TOL_DEG


def test_move_to_position_reachable_pose_reaches_target(world):
    arm = _ur5e_arm(world, "arm-mtp-reachable")
    chain = Chain.from_sva(UR5E_PATH.read_bytes())
    q = [10.0, -20.0, 45.0, 5.0, 10.0, -5.0]
    target = _pose_from_fk(chain, q)

    async def scenario():
        offset = [v + 3.0 for v in q]
        await arm.move_to_joint_positions(JointPositions(values=offset))

        await arm.move_to_position(target)
        end = await arm.get_joint_positions()
        assert end.values == pytest.approx(q, abs=_ANGULAR_TOL_DEG)
        _assert_fk_matches(chain, end.values, target)

    asyncio.run(scenario())


def test_move_to_position_already_there_is_a_no_op(world):
    arm = _ur5e_arm(world, "arm-mtp-already-there")
    chain = Chain.from_sva(UR5E_PATH.read_bytes())
    q = [10.0, -20.0, 45.0, 5.0, 10.0, -5.0]
    target = _pose_from_fk(chain, q)

    async def scenario():
        await arm.move_to_joint_positions(JointPositions(values=q))
        before = (await arm.get_joint_positions()).values

        handle = arm._h()
        original_set_joint_targets = handle.set_joint_targets
        calls = []
        handle.set_joint_targets = lambda *a, **kw: (
            calls.append((a, kw)),
            original_set_joint_targets(*a, **kw),
        )[1]
        try:
            await arm.move_to_position(target)
        finally:
            handle.set_joint_targets = original_set_joint_targets

        after = (await arm.get_joint_positions()).values
        assert calls == []
        assert after == pytest.approx(before, abs=1e-6)

    asyncio.run(scenario())


def test_move_to_position_unreachable_pose_raises(world):
    arm = _ur5e_arm(world, "arm-mtp-unreachable")

    async def scenario():
        far_away = Pose(x=3000.0, y=3000.0, z=3000.0, o_x=0, o_y=0, o_z=1, theta=0)
        with pytest.raises(ValueError) as excinfo:
            await arm.move_to_position(far_away)
        assert isinstance(excinfo.value, PoseUnreachableError)
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_move_to_position_out_of_limits_raises(world, tmp_path):
    sva = json.loads(UR5E_PATH.read_text())
    for joint in sva["joints"]:
        if joint["id"] == "elbow_joint":
            joint["min"], joint["max"] = -10.0, 10.0
    narrowed_path = tmp_path / "ur5e-narrow-elbow.json"
    narrowed_path.write_text(json.dumps(sva))

    arm = IsaacArm.new(
        _config(
            "arm-mtp-limits",
            {
                "world": "isaac-world",
                "asset": "ur5e",
                "mock_dof": 6,
                "kinematics_url": narrowed_path.as_uri(),
            },
        ),
        {},
    )
    chain = Chain.from_sva(UR5E_PATH.read_bytes())
    target = _pose_from_fk(chain, [10.0, -20.0, 45.0, 5.0, 10.0, -5.0])

    async def scenario():
        with pytest.raises(ValueError) as excinfo:
            await arm.move_to_position(target)
        assert isinstance(excinfo.value, JointTargetOutOfLimitsError)
        assert "elbow_joint" in str(excinfo.value)

    asyncio.run(scenario())


def test_move_to_position_nearest_solution_keeps_elbow_sign(world):
    arm = _ur5e_arm(world, "arm-mtp-nearest")
    chain = Chain.from_sva(UR5E_PATH.read_bytes())
    q = [10.0, -20.0, 45.0, 5.0, 10.0, -5.0]
    target = _pose_from_fk(chain, q)

    async def scenario():
        await arm.move_to_joint_positions(JointPositions(values=[v + 3.0 for v in q]))
        await arm.move_to_position(target)
        end = await arm.get_joint_positions()
        assert (end.values[2] > 0) == (q[2] > 0)

    asyncio.run(scenario())


def test_move_to_position_seven_dof_reaches_target(world):
    arm = IsaacArm.new(
        _config(
            "arm-mtp-xarm7",
            {
                "world": "isaac-world",
                "asset": "franka",
                "mock_dof": 7,
                "kinematics_url": XARM7_PATH.as_uri(),
            },
        ),
        {},
    )
    chain = Chain.from_sva(XARM7_PATH.read_bytes())
    q = [10.0, 20.0, -15.0, 60.0, 10.0, 30.0, -10.0]
    target = _pose_from_fk(chain, q)

    async def scenario():
        offset = [v + 3.0 for v in q]
        await arm.move_to_joint_positions(JointPositions(values=offset))

        await arm.move_to_position(target)
        end = await arm.get_joint_positions()
        _assert_fk_matches(chain, end.values, target)

    asyncio.run(scenario())


def test_move_to_position_no_kinematics_raises(world):
    arm = IsaacArm.new(
        _config(
            "arm-mtp-no-kinematics", {"world": "isaac-world", "asset": "franka", "mock_dof": 6}
        ),
        {},
    )

    async def scenario():
        with pytest.raises(RuntimeError) as excinfo:
            await arm.move_to_position(Pose())
        assert isinstance(excinfo.value, KinematicsUnavailableError)
        assert excinfo.value.grpc_code == Status.FAILED_PRECONDITION

    asyncio.run(scenario())
