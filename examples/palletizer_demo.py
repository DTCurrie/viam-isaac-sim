"""Run one pack on the palletizer cell and watch it: the box-palletizer service
picks each box off the pick station and places it on the pallet while this
script prints the service's status once a second. Made for filming the cell.

    .venv/bin/python examples/palletizer_demo.py --address "$VIAM_MACHINE_ADDRESS" \
        --api-key "$VIAM_API_KEY" --api-key-id "$VIAM_API_KEY_ID"

Before the pack starts, the first box in the service's pick order is put back
on the pick station at the cell's pick spot (400, -300, 270 mm by default), so
the demo can be run again after a pack has emptied the station. `--no-reset`
skips that, `--stop` cancels a running pack between motions instead.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FINAL_STATES = ("complete", "failed", "idle")
POLL_S = 1.0
RESET_SETTLE_S = 1.5


@dataclass(frozen=True)
class Args:
    address: str
    api_key: str | None
    api_key_id: str | None
    world: str
    palletizer: str
    box_prop: str
    pick_xyz_mm: tuple[float, float, float]
    reset: bool
    stop: bool
    timeout_s: float


def parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--address", required=True, help="machine address")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--world", default="isaac-world", help="the world component name")
    parser.add_argument(
        "--palletizer", default="box-palletizer", help="the box-palletizer service name"
    )
    parser.add_argument(
        "--box-prop", default="infeed_box_1", help="the box put back on the pick station first"
    )
    parser.add_argument("--pick-x-mm", type=float, default=400.0)
    parser.add_argument("--pick-y-mm", type=float, default=-300.0)
    parser.add_argument("--pick-z-mm", type=float, default=270.0)
    parser.add_argument(
        "--no-reset", action="store_true", help="start without putting the box back first"
    )
    parser.add_argument("--stop", action="store_true", help="cancel a running pack and exit")
    parser.add_argument(
        "--timeout-s", type=float, default=600.0, help="give up waiting for the pack after this"
    )
    ns = parser.parse_args(argv)
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        world=ns.world,
        palletizer=ns.palletizer,
        box_prop=ns.box_prop,
        pick_xyz_mm=(ns.pick_x_mm, ns.pick_y_mm, ns.pick_z_mm),
        reset=not ns.no_reset,
        stop=ns.stop,
        timeout_s=ns.timeout_s,
    )


def record_line(record: Mapping[str, Any]) -> str:
    """One line per placed box, from the service's own record: which box, how
    it ended, where it landed against where it was sent, and how long it took."""
    measured = record.get("measured_pose_mm") or {}
    target = record.get("target_pose_mm") or {}
    where = ""
    if measured and target:
        error_mm = (
            (float(measured.get("x", 0.0)) - float(target.get("x", 0.0))) ** 2
            + (float(measured.get("y", 0.0)) - float(target.get("y", 0.0))) ** 2
        ) ** 0.5
        where = f", landed {error_mm:.1f} mm from its slot"
    duration = record.get("duration_s")
    took = f" in {float(duration):.1f} s" if duration is not None else ""
    return f"{record.get('box_prop', '?')}: {record.get('outcome', '?')}{where}{took}"


async def watch_pack(palletizer: Any, timeout_s: float, poll_s: float = POLL_S) -> str:
    """Polls the service's status until the pack ends, printing each change of
    state and each new record as it appears. Returns the final state."""
    seen = 0
    last_state = None
    waited = 0.0
    while True:
        status = await palletizer.do_command({"command": "status"})
        state = str(status.get("state", "?"))
        if state != last_state:
            print(f"  state: {state}")
            last_state = state
        records = list(status.get("records") or [])
        for record in records[seen:]:
            print(f"  {record_line(record)}")
        seen = len(records)
        if state in FINAL_STATES and (state != "idle" or waited > 0.0):
            reason = status.get("reason")
            if reason:
                print(f"  reason: {reason}")
            return state
        if waited >= timeout_s:
            print(f"  gave up after {timeout_s:.0f} s in state {state}")
            return state
        await asyncio.sleep(poll_s)
        waited += poll_s


async def run(args: Args) -> int:
    from viam.components.generic import Generic
    from viam.robot.client import RobotClient
    from viam.services.generic import Generic as GenericService

    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    robot = await RobotClient.at_address(args.address, opts)
    try:
        world = Generic.from_robot(robot, args.world)
        palletizer = GenericService.from_robot(robot, args.palletizer)

        if args.stop:
            reply = await palletizer.do_command({"command": "stop"})
            print(f"stop: {reply}")
            return 0

        if args.reset:
            x_mm, y_mm, z_mm = args.pick_xyz_mm
            await world.do_command(
                {
                    "command": "set_prop_pose",
                    "name": args.box_prop,
                    "position": [x_mm, y_mm, z_mm],
                    "orientation_rpy_deg": [0.0, 0.0, 0.0],
                }
            )
            await asyncio.sleep(RESET_SETTLE_S)
            print(
                f"{args.box_prop} back on the pick station at "
                f"({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f}) mm"
            )

        reply = await palletizer.do_command({"command": "start"})
        print(f"start: {reply}")
        if not reply.get("ok", False):
            return 1
        final = await watch_pack(palletizer, args.timeout_s)
        return 0 if final == "complete" else 1
    finally:
        await robot.close()


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
