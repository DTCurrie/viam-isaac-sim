"""GPU acceptance checklist for the Isaac Sim UR5e arm.

Connects to a running Viam machine (the module running on the Isaac GPU box)
and checks that the arm's joint positions, its end-effector pose, and its
tool axis agree with what the Viam motion service reports, then exercises a
motion-service move over a target pose. Prints PASS/FAIL and the raw numbers
for each check.

Depends only on the stdlib and viam-sdk: it runs on a laptop against a
remote machine, not inside the module process.

Usage::

    python examples/gpu_checklist_arm.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id>
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from viam.components.arm import Arm
from viam.components.generic import Generic
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

# python examples/gpu_checklist_arm.py (standalone, no PYTHONPATH set) needs the
# repo's src/ on sys.path before isaac_module is importable. pytest already adds
# it (pyproject pythonpath = ["src"]), so this is a no-op there.
try:
    import isaac_module  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isaac_module.spatial import ov_to_quat, quat_conj, quat_mul, quat_rotate

# pure helpers, unit-tested without a robot in tests/test_gpu_checklist_arm.py

PoseTuple = tuple[float, float, float, float, float, float, float]
"""(x, y, z, o_x, o_y, o_z, theta_deg). x/y/z share the length unit of the two
poses being compared. This script always compares mm to mm."""

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)


def pose_delta_mm_deg(pose_a: PoseTuple, pose_b: PoseTuple) -> tuple[float, float]:
    """Translation distance (mm) and rotation angle (deg) between two Viam
    Pose-like tuples (x, y, z, o_x, o_y, o_z, theta_deg)."""
    ax, ay, az, aox, aoy, aoz, atheta = pose_a
    bx, by, bz, box_, boy, boz, btheta = pose_b
    translation_mm = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    qa = ov_to_quat(aox, aoy, aoz, math.radians(atheta))
    qb = ov_to_quat(box_, boy, boz, math.radians(btheta))
    relative = quat_mul(quat_conj(qa), qb)
    w = max(-1.0, min(1.0, abs(relative[0])))
    rotation_deg = math.degrees(2.0 * math.acos(w))
    return translation_mm, rotation_deg


def axis_from_quaternion(q: Quat, axis: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    """Rotate `axis` (default +Z) by a (w,x,y,z) quaternion."""
    return quat_rotate(q, axis)


def angle_between_deg(v1: Vec3, v2: Vec3) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    n1 = math.sqrt(sum(c * c for c in v1))
    n2 = math.sqrt(sum(c * c for c in v2))
    cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_theta))


def verdict(name: str, ok: bool, detail: str) -> str:
    """Format one checklist line: "[PASS|FAIL] name: detail"."""
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {name}: {detail}"


# checklist items, each returns (name, ok) for the summary table

TRANSLATION_TOLERANCE_MM = 1.0
ROTATION_TOLERANCE_DEG = 0.1
MOVE_TOLERANCE_MM = 10.0
JOINT_ZERO_TOLERANCE_DEG = 0.5


@dataclass
class Args:
    address: str
    api_key: str | None
    api_key_id: str | None
    arm: str
    motion: str
    world: str
    block_xyz_m: tuple[float, float, float]
    pregrasp_clearance_mm: float
    skip_move: bool


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--address", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--world", default="isaac-world", help="the isaac-sim world component name")
    parser.add_argument("--block-xyz-m", default="0.60,0.10,0.7755")
    parser.add_argument("--pregrasp-clearance-mm", type=float, default=100.0)
    parser.add_argument("--skip-move", action="store_true")
    ns = parser.parse_args()
    bx, by, bz = (float(v) for v in ns.block_xyz_m.split(","))
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        arm=ns.arm,
        motion=ns.motion,
        world=ns.world,
        block_xyz_m=(bx, by, bz),
        pregrasp_clearance_mm=ns.pregrasp_clearance_mm,
        skip_move=ns.skip_move,
    )


async def _connect(args: Args) -> RobotClient:
    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    return await RobotClient.at_address(args.address, opts)


def _pose_to_tuple(pose: Pose) -> PoseTuple:
    return (pose.x, pose.y, pose.z, pose.o_x, pose.o_y, pose.o_z, pose.theta)


async def _check_joints_zero(arm: Arm) -> tuple[str, bool]:
    positions = await arm.get_joint_positions()
    ok = all(abs(v) <= JOINT_ZERO_TOLERANCE_DEG for v in positions.values)
    print(f"  joint positions (deg): {list(positions.values)}")
    max_abs_deg = max((abs(v) for v in positions.values), default=0.0)
    line = verdict("joints zero at boot", ok, f"max |deg| = {max_abs_deg:.3f}")
    print(line)
    return line, ok


async def _check_dof_names(arm: Arm, world: Generic) -> tuple[str, bool]:
    result = await world.do_command({"command": "dof_names", "name": arm.name})
    names = list(result["dof_names"])  # type: ignore[arg-type]
    print(f"  dof_names ({len(names)}): {names}")
    line = verdict("dof_names logged", True, f"{len(names)} dofs: {names}")
    print(line)
    return line, True


async def _prim_world_pose(arm: Arm, world: Generic) -> dict:
    return dict(await world.do_command({"command": "prim_pose", "name": arm.name}))


async def _check_world_pose(
    arm: Arm, motion: MotionClient, world: Generic
) -> tuple[str, bool, dict]:
    prim_result = await _prim_world_pose(arm, world)
    isaac_pose: PoseTuple = (
        *prim_result["position_mm"],
        prim_result["orientation_vector"]["o_x"],
        prim_result["orientation_vector"]["o_y"],
        prim_result["orientation_vector"]["o_z"],
        prim_result["orientation_vector"]["theta_deg"],
    )
    viam_pose_in_frame = await motion.get_pose(component_name=arm.name, destination_frame="world")
    viam_pose = _pose_to_tuple(viam_pose_in_frame.pose)
    translation_mm, rotation_deg = pose_delta_mm_deg(isaac_pose, viam_pose)
    ok = translation_mm <= TRANSLATION_TOLERANCE_MM and rotation_deg <= ROTATION_TOLERANCE_DEG
    print(f"  isaac prim world pose (mm/deg): {isaac_pose}")
    print(f"  viam motion.get_pose(world) (mm/deg): {viam_pose}")
    line = verdict(
        "isaac prim pose matches viam world pose",
        ok,
        f"delta {translation_mm:.3f} mm / {rotation_deg:.4f} deg",
    )
    print(line)
    return line, ok, prim_result


async def _check_end_position(arm: Arm, motion: MotionClient) -> tuple[str, bool]:
    end_position = await arm.get_end_position()
    # The arm component's own frame is its end effector (GetPose(arm, dest=arm) is
    # the identity). The arm base frame the motion service exposes is "<arm>_origin".
    viam_pose_in_frame = await motion.get_pose(
        component_name=arm.name, destination_frame=f"{arm.name}_origin"
    )
    isaac_pose = _pose_to_tuple(end_position)
    viam_pose = _pose_to_tuple(viam_pose_in_frame.pose)
    translation_mm, rotation_deg = pose_delta_mm_deg(isaac_pose, viam_pose)
    ok = translation_mm <= TRANSLATION_TOLERANCE_MM and rotation_deg <= ROTATION_TOLERANCE_DEG
    print(f"  arm.get_end_position() (mm/deg): {isaac_pose}")
    print(f"  viam motion.get_pose(arm, dest=<arm>_origin) (mm/deg): {viam_pose}")
    line = verdict(
        "get_end_position matches viam arm-frame pose",
        ok,
        f"delta {translation_mm:.3f} mm / {rotation_deg:.4f} deg",
    )
    print(line)
    return line, ok


async def _check_tool_axis(prim_result: dict) -> tuple[str, bool]:
    quat_wxyz: Quat = tuple(prim_result["quaternion_wxyz"])
    isaac_z_axis = axis_from_quaternion(quat_wxyz)
    ov = prim_result["orientation_vector"]
    viam_axis: Vec3 = (ov["o_x"], ov["o_y"], ov["o_z"])
    angle_deg = angle_between_deg(isaac_z_axis, viam_axis)
    ok = angle_deg <= ROTATION_TOLERANCE_DEG
    print(f"  isaac prim +Z axis: {isaac_z_axis}")
    print(f"  viam orientation vector: {viam_axis}")
    line = verdict("tool axis is isaac's +Z", ok, f"angle {angle_deg:.4f} deg")
    print(line)
    return line, ok


async def _check_move(arm: Arm, motion: MotionClient, args: Args) -> tuple[str, bool]:
    bx, by, bz = args.block_xyz_m
    target = Pose(
        x=bx * 1000.0,
        y=by * 1000.0,
        z=bz * 1000.0 + args.pregrasp_clearance_mm,
        o_x=0.0,
        o_y=0.0,
        o_z=-1.0,
        theta=0.0,
    )
    try:
        success = await motion.move(
            component_name=arm.name,
            destination=PoseInFrame(reference_frame="world", pose=target),
        )
        print(f"  motion.move plan result: success={success}")
        end_position = await arm.get_end_position()
        viam_pose_in_frame = await motion.get_pose(
            component_name=arm.name, destination_frame="world"
        )
        actual = _pose_to_tuple(end_position)
        viam_world = _pose_to_tuple(viam_pose_in_frame.pose)
        target_tuple: PoseTuple = (
            target.x,
            target.y,
            target.z,
            target.o_x,
            target.o_y,
            target.o_z,
            target.theta,
        )
        translation_mm, _ = pose_delta_mm_deg(viam_world, target_tuple)
        ok = bool(success) and translation_mm <= MOVE_TOLERANCE_MM
        print(f"  target (mm): {target_tuple}")
        print(f"  arrived (mm): {actual} / world (mm): {viam_world}")
        detail = f"arrival delta {translation_mm:.3f} mm"
        line = verdict("move over block_red succeeds", ok, detail)
    except Exception as exc:  # noqa: BLE001 - never crash the checklist run
        ok = False
        line = verdict("move over block_red succeeds", ok, f"exception: {exc!r}")
    print(line)
    return line, ok


def _check_pip_check_reminder() -> tuple[str, bool]:
    line = verdict(
        "pip check",
        True,
        "informational: read `pip check` from the module's run.sh logs in viam-server",
    )
    print(line)
    return line, True


async def main() -> None:
    args = _parse_args()
    machine = await _connect(args)
    try:
        arm = Arm.from_robot(machine, args.arm)
        motion = MotionClient.from_robot(machine, args.motion)
        world = Generic.from_robot(machine, args.world)

        results: list[tuple[str, bool]] = []

        print("\n-- joints zero at boot --")
        results.append(await _check_joints_zero(arm))

        print("\n-- dof_names --")
        results.append(await _check_dof_names(arm, world))

        print("\n-- isaac prim world pose vs viam world pose --")
        line, ok, prim_result = await _check_world_pose(arm, motion, world)
        results.append((line, ok))

        print("\n-- get_end_position vs viam arm-frame pose --")
        results.append(await _check_end_position(arm, motion))

        print("\n-- tool axis --")
        results.append(await _check_tool_axis(prim_result))

        print("\n-- move over block_red --")
        if args.skip_move:
            print("  skipped (--skip-move)")
            results.append((verdict("move over block_red succeeds", True, "skipped"), True))
        else:
            results.append(await _check_move(arm, motion, args))

        print("\n-- pip check --")
        results.append(_check_pip_check_reminder())

        print("\n== summary ==")
        for line, _ in results:
            print(line)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
