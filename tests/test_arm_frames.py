"""Base-frame correction on spawn and base-frame end pose reporting
(FINDINGS XC-1, ARM-10)."""

import asyncio
import math

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.errors import PrimNotFoundError
from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import (
    KNOWN_ASSETS,
    MockArmHandle,
    anchor_fixed_joint_frame,
    compose_pose,
    pose_in_frame,
    spawn_orientation,
    viam_base_frame,
)
from isaac_module.spatial import quat_rotate


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _rz(deg: float) -> tuple[float, float, float, float]:
    """A (w,x,y,z) quaternion rotating deg around +Z."""
    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def test_spawn_orientation_defaults_to_identity_without_correction():
    assert spawn_orientation({}, {}) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_spawn_orientation_applies_known_asset_correction():
    # ur5e's base_frame_correction is Rz(180deg) = (0,0,0,1)
    got = spawn_orientation({}, KNOWN_ASSETS["ur5e"])
    assert got == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


def test_spawn_orientation_composes_frame_then_correction():
    # Rx(90deg) and Rz(180deg) do not commute, so this pins the composition
    # order (frame * correction) the seam requires: reversing the order
    # would rotate (0, 1, 0) to (0, 0, +1) instead of (0, 0, -1).
    q_frame = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # Rx(90deg)
    meta = KNOWN_ASSETS["ur5e"]  # base_frame_correction is Rz(180deg)
    got = spawn_orientation({"orientation_wxyz": list(q_frame)}, meta)
    rotated = quat_rotate(got, (0.0, 1.0, 0.0))
    assert rotated == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_pose_in_frame_translates_and_rotates_into_the_base():
    base_pos = (1.0, 2.0, 3.0)
    base_quat = _rz(90.0)
    world_point = (1.0, 3.0, 3.0)
    # end effector with the same orientation as the base -> identity relative
    local_pos, local_quat = pose_in_frame(base_pos, base_quat, world_point, base_quat)
    assert local_pos == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert local_quat == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)


def test_mock_arm_end_position_is_base_frame_regardless_of_spawn_pose(world):
    arm_at_origin = IsaacArm.new(_config("arm-origin", {"world": "sim-world", "asset": "ur5e"}), {})

    config_with_frame = _config("arm-offset", {"world": "sim-world", "asset": "ur5e"})
    config_with_frame.frame.translation.x = 500.0
    config_with_frame.frame.translation.y = -250.0
    config_with_frame.frame.translation.z = 750.0
    qw, qx, qy, qz = _rz(90.0)
    config_with_frame.frame.orientation.quaternion.w = qw
    config_with_frame.frame.orientation.quaternion.x = qx
    config_with_frame.frame.orientation.quaternion.y = qy
    config_with_frame.frame.orientation.quaternion.z = qz
    arm_offset = IsaacArm.new(config_with_frame, {})

    async def scenario():
        pose_origin = await arm_at_origin.get_end_position()
        pose_offset = await arm_offset.get_end_position()
        for pose in (pose_origin, pose_offset):
            assert pose.x == pytest.approx(300.0)
            assert pose.y == pytest.approx(0.0)
            assert pose.z == pytest.approx(300.0)

    asyncio.run(scenario())

    assert arm_offset._h().spawn_orientation != pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_viam_base_frame_undoes_a_matching_root_correction():
    # root spawned as frame * correction; when root == correction (frame ==
    # identity), viam_base_frame must recover identity.
    root_quat = _rz(180.0)
    correction = _rz(180.0)
    pos, quat = viam_base_frame((1.0, 2.0, 3.0), root_quat, correction)
    assert pos == pytest.approx((1.0, 2.0, 3.0))
    # a quaternion and its negation represent the same rotation
    assert quat == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9) or quat == pytest.approx(
        (-1.0, 0.0, 0.0, 0.0), abs=1e-9
    )


def test_compose_pose_round_trips_with_pose_in_frame():
    base_pos = (1.0, 2.0, 3.0)
    base_quat = _rz(90.0)
    local_pos = (0.5, -0.25, 0.75)
    local_quat = _rz(30.0)

    world_pos, world_quat = compose_pose(base_pos, base_quat, local_pos, local_quat)
    round_trip_pos, round_trip_quat = pose_in_frame(base_pos, base_quat, world_pos, world_quat)

    assert round_trip_pos == pytest.approx(local_pos, abs=1e-9)
    assert round_trip_quat == pytest.approx(local_quat, abs=1e-9)


def test_mock_arm_end_pose_is_in_viam_frame_not_the_raw_isaac_root(world):
    # regression for the defect found 2026-08-28: get_end_pose must be
    # reported in Viam's arm frame (the root un-rotated by
    # base_frame_correction), not the raw Isaac articulation root.
    handle = MockArmHandle("ur5e-origin", {"world": "sim-world", "asset": "ur5e"})

    end_pos, end_quat = handle.get_end_pose()
    assert end_pos == pytest.approx(MockArmHandle.FIXED_LOCAL_EE[0], abs=1e-9)
    assert end_quat == pytest.approx(MockArmHandle.FIXED_LOCAL_EE[1], abs=1e-9)

    ee_prim_path = f"{handle._prim_path}/wrist_3_link"
    ee_world_pos, ee_world_quat = handle.get_prim_world_pose(ee_prim_path)
    # expressed in the raw (rotated) Isaac root frame instead, the same
    # world pose comes out off by exactly the base_frame_correction
    raw_root_pos, raw_root_quat = pose_in_frame(
        handle.spawn_position, handle.spawn_orientation, ee_world_pos, ee_world_quat
    )
    assert raw_root_pos == pytest.approx((-0.3, 0.0, 0.3), abs=1e-9)
    assert raw_root_quat == pytest.approx(_rz(180.0), abs=1e-9) or raw_root_quat == pytest.approx(
        tuple(-c for c in _rz(180.0)), abs=1e-9
    )


def test_anchor_fixed_joint_frame_identity_spawn_leaves_authored_frame_unchanged():
    authored_pos = (0.1, -0.2, 0.05)
    authored_quat = _rz(30.0)
    pos, quat = anchor_fixed_joint_frame(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), authored_pos, authored_quat
    )
    assert pos == pytest.approx(authored_pos, abs=1e-9)
    assert quat == pytest.approx(authored_quat, abs=1e-9)


def test_anchor_fixed_joint_frame_ur5e_spawn_composes_to_spawn_position_and_identity_rotation():
    spawn_pos = (0.15, -0.25, 0.75)
    spawn_quat = spawn_orientation({}, KNOWN_ASSETS["ur5e"])  # Rz(180deg)
    authored_pos = (0.0, 0.0, 0.0)
    authored_quat = _rz(180.0)

    pos, quat = anchor_fixed_joint_frame(spawn_pos, spawn_quat, authored_pos, authored_quat)

    assert pos == pytest.approx(spawn_pos, abs=1e-9)
    # a quaternion and its negation represent the same rotation
    assert quat == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9) or quat == pytest.approx(
        (-1.0, 0.0, 0.0, 0.0), abs=1e-9
    )


def test_anchor_fixed_joint_frame_rotates_the_authored_offset_into_the_spawn_frame():
    spawn_pos = (1.0, 2.0, 3.0)
    spawn_quat = _rz(90.0)
    authored_pos = (0.1, 0.0, 0.0)
    authored_quat = (1.0, 0.0, 0.0, 0.0)

    pos, quat = anchor_fixed_joint_frame(spawn_pos, spawn_quat, authored_pos, authored_quat)

    assert pos == pytest.approx((1.0, 2.1, 3.0), abs=1e-9)
    assert quat == pytest.approx(spawn_quat, abs=1e-9)


def test_mock_arm_get_prim_world_pose_rejects_unknown_prim(world):
    handle = MockArmHandle("ur5e-unknown-prim", {"world": "sim-world", "asset": "ur5e"})

    with pytest.raises(PrimNotFoundError):
        handle.get_prim_world_pose("/World/does_not_exist")

    assert issubclass(PrimNotFoundError, ValueError)
