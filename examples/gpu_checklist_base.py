"""GPU acceptance checklist for the Isaac Sim jetbot base.

Connects to a running Viam machine (the module running on the Isaac GPU box)
and checks that the base's reported wheel geometry is sane, then drives it
with ``SetVelocity``, ``MoveStraight`` and ``Spin`` and checks the resulting
motion, read back from the sim through the world component's ``prim_pose``
DoCommand (the same verb the arm checklist uses), against what each command
asked for. Prints PASS/FAIL and the raw numbers for each check.

Depends only on the stdlib and viam-sdk: it runs on a laptop against a
remote machine, not inside the module process.

Usage::

    python examples/gpu_checklist_base.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id>
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viam.components.base import Base
from viam.components.generic import Generic
from viam.proto.common import Vector3
from viam.robot.client import RobotClient

# python examples/gpu_checklist_base.py (standalone, no PYTHONPATH set) needs the
# repo's src/ on sys.path before isaac_module is importable. pytest already adds
# it (pyproject pythonpath = ["src"]), so this is a no-op there.
try:
    import isaac_module  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isaac_module.spatial import ov_to_quat, quat_rotate

# pure helpers, unit-tested without a robot in tests/test_gpu_checklist_base.py

PoseTuple = tuple[float, float, float, float, float, float, float]
"""(x, y, z, o_x, o_y, o_z, theta_deg), all in mm/degrees, matching the
tuple shape gpu_checklist_arm.py compares poses with."""

Vec3 = tuple[float, float, float]


def pose_from_prim_result(result: Mapping[str, Any]) -> PoseTuple:
    """A world ``{"command": "prim_pose"}`` result's ``position_mm`` and
    ``orientation_vector`` as a `PoseTuple`, the shape
    ``component_diagnostics.prim_pose`` returns."""
    x, y, z = result["position_mm"]
    ov = result["orientation_vector"]
    return (x, y, z, ov["o_x"], ov["o_y"], ov["o_z"], ov["theta_deg"])


def forward_axis(pose: PoseTuple) -> Vec3:
    """The base's own +x axis, in the world frame, at `pose`."""
    _, _, _, ox, oy, oz, theta_deg = pose
    quat = ov_to_quat(ox, oy, oz, math.radians(theta_deg))
    return quat_rotate(quat, (1.0, 0.0, 0.0))


def forward_distance_mm(before: PoseTuple, after: PoseTuple) -> float:
    """Distance travelled from `before` to `after` (mm), projected onto the
    base's own +x axis at `before` (its heading before the move). Positive
    means the base moved forward along the heading it started with."""
    axis = forward_axis(before)
    dx, dy, dz = after[0] - before[0], after[1] - before[1], after[2] - before[2]
    return dx * axis[0] + dy * axis[1] + dz * axis[2]


def planar_distance_mm(a: PoseTuple, b: PoseTuple) -> float:
    """Euclidean distance (mm) between two poses' positions."""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def heading_delta_deg(a: PoseTuple, b: PoseTuple) -> float:
    """Absolute change in `theta_deg` between two poses, wrapped to the
    shorter way around the circle. Valid for a base that stays level, where
    `theta_deg` alone is the heading."""
    delta = (b[6] - a[6] + 180.0) % 360.0 - 180.0
    return abs(delta)


def within_relative_tolerance(observed: float, expected: float, tolerance_fraction: float) -> bool:
    """True when `observed` is within `tolerance_fraction` of `expected`,
    e.g. a fraction of 0.3 allows 30 percent either side."""
    return abs(observed - expected) <= abs(expected) * tolerance_fraction


def verdict(name: str, ok: bool, detail: str) -> str:
    """Format one checklist line: "[PASS|FAIL] name: detail"."""
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {name}: {detail}"


# checklist items, each returns (name, ok) for the summary table

SET_VELOCITY_LINEAR_MMPS = 200.0
SET_VELOCITY_DURATION_S = 1.0
SET_VELOCITY_TOLERANCE_FRACTION = 0.30

MOVE_STRAIGHT_DISTANCE_MM = 300.0
MOVE_STRAIGHT_VELOCITY_MMPS = 150.0
MOVE_STRAIGHT_TOLERANCE_FRACTION = 0.20

SPIN_ANGLE_DEG = 90.0
SPIN_VELOCITY_DEGPS = 45.0
SPIN_TOLERANCE_DEG = 15.0

STOP_SETTLE_DURATION_S = 1.0
STOP_POSE_TOLERANCE_MM = 5.0


@dataclass
class Args:
    address: str
    api_key: str | None
    api_key_id: str | None
    base: str
    world: str


def _parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--address", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--base", default="jetbot-base")
    parser.add_argument("--world", default="isaac-world", help="the isaac-sim world component name")
    ns = parser.parse_args(argv)
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        base=ns.base,
        world=ns.world,
    )


async def _connect(args: Args) -> RobotClient:
    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    return await RobotClient.at_address(args.address, opts)


async def _base_pose(base: Base, world: Generic) -> PoseTuple:
    # no prim_path: the sim picks the base's own root prim
    result = await world.do_command({"command": "prim_pose", "name": base.name})
    return pose_from_prim_result(result)


async def _check_wheel_geometry(base: Base) -> tuple[str, bool]:
    props = await base.get_properties()
    wheel_radius_m = props.wheel_circumference_meters / (2.0 * math.pi)
    ok = props.width_meters > 0.0 and wheel_radius_m > 0.0
    print(f"  width_meters (wheel base): {props.width_meters:.4f} m")
    print(f"  wheel_circumference_meters: {props.wheel_circumference_meters:.4f} m")
    print(f"  wheel_radius: {wheel_radius_m:.4f} m")
    line = verdict(
        "GetProperties reports the asset's wheel base and radius",
        ok,
        f"wheel_base={props.width_meters:.4f} m wheel_radius={wheel_radius_m:.4f} m",
    )
    print(line)
    return line, ok


async def _check_set_velocity(base: Base, world: Generic) -> tuple[str, bool]:
    before = await _base_pose(base, world)
    await base.set_velocity(
        linear=Vector3(x=0.0, y=SET_VELOCITY_LINEAR_MMPS, z=0.0),
        angular=Vector3(x=0.0, y=0.0, z=0.0),
    )
    try:
        await asyncio.sleep(SET_VELOCITY_DURATION_S)
    finally:
        await base.stop()
    after = await _base_pose(base, world)
    observed_mm = forward_distance_mm(before, after)
    expected_mm = SET_VELOCITY_LINEAR_MMPS * SET_VELOCITY_DURATION_S
    ok = within_relative_tolerance(observed_mm, expected_mm, SET_VELOCITY_TOLERANCE_FRACTION)
    print(f"  before (mm/deg): {before}")
    print(f"  after (mm/deg): {after}")
    print(f"  forward distance: {observed_mm:.1f} mm, expected {expected_mm:.1f} mm")
    line = verdict(
        "SetVelocity forward for one second moves the base along its own +x",
        ok,
        f"distance {observed_mm:.1f} mm vs expected {expected_mm:.1f} mm",
    )
    print(line)
    return line, ok


async def _check_move_straight(base: Base, world: Generic) -> tuple[str, bool]:
    before = await _base_pose(base, world)
    await base.move_straight(
        distance=int(MOVE_STRAIGHT_DISTANCE_MM), velocity=MOVE_STRAIGHT_VELOCITY_MMPS
    )
    after = await _base_pose(base, world)
    observed_mm = forward_distance_mm(before, after)
    ok = within_relative_tolerance(
        observed_mm, MOVE_STRAIGHT_DISTANCE_MM, MOVE_STRAIGHT_TOLERANCE_FRACTION
    )
    print(f"  before (mm/deg): {before}")
    print(f"  after (mm/deg): {after}")
    print(f"  distance travelled: {observed_mm:.1f} mm, commanded {MOVE_STRAIGHT_DISTANCE_MM} mm")
    line = verdict(
        "MoveStraight for a set distance ends within tolerance of it",
        ok,
        f"distance {observed_mm:.1f} mm vs commanded {MOVE_STRAIGHT_DISTANCE_MM} mm",
    )
    print(line)
    return line, ok


async def _check_spin(base: Base, world: Generic) -> tuple[str, bool]:
    before = await _base_pose(base, world)
    await base.spin(angle=SPIN_ANGLE_DEG, velocity=SPIN_VELOCITY_DEGPS)
    after = await _base_pose(base, world)
    observed_deg = heading_delta_deg(before, after)
    ok = abs(observed_deg - SPIN_ANGLE_DEG) <= SPIN_TOLERANCE_DEG
    print(f"  before (mm/deg): {before}")
    print(f"  after (mm/deg): {after}")
    print(f"  heading change: {observed_deg:.2f} deg, commanded {SPIN_ANGLE_DEG} deg")
    line = verdict(
        "Spin by 90 degrees ends within tolerance of it",
        ok,
        f"heading change {observed_deg:.2f} deg vs commanded {SPIN_ANGLE_DEG} deg",
    )
    print(line)
    return line, ok


async def _check_stop(base: Base, world: Generic) -> tuple[str, bool]:
    await base.stop()
    before = await _base_pose(base, world)
    await asyncio.sleep(STOP_SETTLE_DURATION_S)
    after = await _base_pose(base, world)
    observed_mm = planar_distance_mm(before, after)
    ok = observed_mm <= STOP_POSE_TOLERANCE_MM
    print(f"  before (mm/deg): {before}")
    print(f"  after (mm/deg): {after}")
    print(f"  pose delta over {STOP_SETTLE_DURATION_S:.0f} s: {observed_mm:.3f} mm")
    line = verdict(
        "Stop leaves the base stationary",
        ok,
        f"pose delta {observed_mm:.3f} mm over {STOP_SETTLE_DURATION_S:.0f} s",
    )
    print(line)
    return line, ok


async def main() -> None:
    args = _parse_args()
    machine = await _connect(args)
    try:
        base = Base.from_robot(machine, args.base)
        world = Generic.from_robot(machine, args.world)

        results: list[tuple[str, bool]] = []

        print("\n-- wheel geometry --")
        results.append(await _check_wheel_geometry(base))

        print("\n-- SetVelocity forward --")
        results.append(await _check_set_velocity(base, world))

        print("\n-- MoveStraight --")
        results.append(await _check_move_straight(base, world))

        print("\n-- Spin --")
        results.append(await _check_spin(base, world))

        print("\n-- Stop --")
        results.append(await _check_stop(base, world))

        print("\n== summary ==")
        for line, _ in results:
            print(line)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
