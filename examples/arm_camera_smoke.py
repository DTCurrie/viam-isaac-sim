"""Arm/camera/gripper smoke test for a Viam machine.

Connects to a running machine and exercises the arm, camera, motion and
gripper APIs: reads joint positions and end position off the arm, pulls
images off the camera, drives the arm with a motion service ``Move`` to a
pose offset along the arm's own z axis, then opens and closes the gripper.
Every step is caught individually and recorded, so one failure never stops
the rest, and the script never raises to the user. It prints one JSON report
to stdout and exits 0 only when every step succeeded, 1 otherwise.

This script names no world and no simulation verb. It is the same script a
real machine and a sim machine both run to prove the arm, camera, motion and
gripper resources answer the Viam APIs a client expects.

Usage::

    python examples/arm_camera_smoke.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id> \\
        --arm pick-arm --gripper pick-grip --camera wrist-cam \\
        --motion builtin --offset-mm 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient


@dataclass
class Args:
    address: str
    api_key: str | None
    api_key_id: str | None
    arm: str
    gripper: str
    camera: str
    motion: str
    offset_mm: float


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--address", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--gripper", default="pick-grip")
    parser.add_argument("--camera", default="wrist-cam")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--offset-mm", type=float, default=50)
    ns = parser.parse_args(argv)
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        arm=ns.arm,
        gripper=ns.gripper,
        camera=ns.camera,
        motion=ns.motion,
        offset_mm=ns.offset_mm,
    )


async def _connect(args: Args) -> RobotClient:
    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    return await RobotClient.at_address(args.address, opts)


def _pose_to_dict(pose: Pose) -> dict[str, float]:
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "o_x": pose.o_x,
        "o_y": pose.o_y,
        "o_z": pose.o_z,
        "theta": pose.theta,
    }


async def _step_joint_positions(arm: Arm) -> dict[str, Any]:
    positions = await arm.get_joint_positions()
    return {"ok": True, "values": list(positions.values)}


async def _step_end_position(arm: Arm) -> dict[str, Any]:
    end_position = await arm.get_end_position()
    return {"ok": True, "pose": _pose_to_dict(end_position)}


async def _step_images(camera: Camera) -> dict[str, Any]:
    images, _metadata = await camera.get_images()
    return {
        "ok": True,
        "images": [
            {"mime_type": str(image.mime_type), "size_bytes": len(image.data)} for image in images
        ],
    }


async def _step_move(arm: Arm, motion: MotionClient, offset_mm: float) -> dict[str, Any]:
    start_pose = await arm.get_end_position()
    target = Pose(
        x=start_pose.x,
        y=start_pose.y,
        z=start_pose.z + offset_mm,
        o_x=start_pose.o_x,
        o_y=start_pose.o_y,
        o_z=start_pose.o_z,
        theta=start_pose.theta,
    )
    # GetEndPosition is in the arm's base frame, which the frame system names
    # "<arm>_origin". The bare component name is the end-effector frame.
    success = await motion.move(
        component_name=arm.name,
        destination=PoseInFrame(reference_frame=f"{arm.name}_origin", pose=target),
    )
    end_position = await arm.get_end_position()
    return {
        "ok": bool(success),
        "target": _pose_to_dict(target),
        "end_position_after": _pose_to_dict(end_position),
    }


async def _step_gripper(gripper: Gripper) -> dict[str, Any]:
    await gripper.open()
    grabbed = await gripper.grab()
    return {"ok": True, "grabbed": bool(grabbed)}


async def _run_step(coro: Any) -> dict[str, Any]:
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - every step is recorded, never raised
        return {"ok": False, "error": repr(exc)}


async def _run(args: Args) -> dict[str, Any]:
    report: dict[str, Any] = {"steps": {}}
    machine = await _connect(args)
    try:
        arm = Arm.from_robot(machine, args.arm)
        gripper = Gripper.from_robot(machine, args.gripper)
        camera = Camera.from_robot(machine, args.camera)
        motion = MotionClient.from_robot(machine, args.motion)

        report["steps"]["joint_positions"] = await _run_step(_step_joint_positions(arm))
        report["steps"]["end_position"] = await _run_step(_step_end_position(arm))
        report["steps"]["images"] = await _run_step(_step_images(camera))
        report["steps"]["move"] = await _run_step(_step_move(arm, motion, args.offset_mm))
        report["steps"]["gripper"] = await _run_step(_step_gripper(gripper))
    finally:
        await machine.close()

    report["ok"] = all(step.get("ok") for step in report["steps"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(_run(args))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
