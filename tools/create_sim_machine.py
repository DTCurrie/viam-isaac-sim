#!/usr/bin/env python3
"""Create a sim machine for a real machine's config, end to end.

Usage: `create_sim_machine.py <real-config.json> --name NAME --location-id ID
[--org-id ID] [--project P] [--zone Z] [--machine-type T] [--image-family F]
[--allow-unmatched] [--dry-run]`

Steps, in order: resolve the real config into the sim machine's config with
`isaac_module.config_resolver`; create the machine in the location through
the Viam app API; read its main part's id and secret; write a `viam.json`
cloud credentials file; launch a GCP GPU instance from the image family with
a startup script that installs that file at `/etc/viam.json` and restarts
viam-agent; wait until the part reports online; push the resolved config to
the part. `--dry-run` performs the resolve and prints every remote call it
would make, verbatim, without making any. Exit 0 on success, 2 for unmatched
hardware, 3 when the part never comes online, 1 for any other error.

Authentication: `VIAM_API_KEY` and `VIAM_API_KEY_ID` in the environment for the
app API, and the active `gcloud` login for the instance.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_PROJECT_HINT = "the active gcloud project"
DEFAULT_ZONE = "us-central1-a"
DEFAULT_MACHINE_TYPE = "g2-standard-8"
DEFAULT_IMAGE_FAMILY = "viam-isaac-sim"
APP_ADDRESS = "https://app.viam.com:443"
ONLINE_TIMEOUT_S = 900.0
ONLINE_POLL_S = 10.0


@dataclass(frozen=True)
class Plan:
    """Everything the run will do, computed before any remote call.

    `gcloud_args` is the full `gcloud compute instances create` argument
    vector. `credentials` is the `viam.json` document, with the part secret
    filled in only after the machine exists (empty in a dry run).
    """

    machine_name: str
    resolved_config: dict[str, Any]
    gcloud_args: tuple[str, ...]
    credentials: dict[str, Any]


def build_plan(args: Any, resolved_config: dict[str, Any], part_id: str, secret: str) -> Plan:
    """The plan for one run. Pure, so tests can check it without gcloud or the app API."""
    document = {"cloud": {"app_address": APP_ADDRESS, "id": part_id, "secret": secret}}
    startup_script_path = str(Path(tempfile.gettempdir()) / f"viam-startup-{args.name}.sh")
    startup_script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "install -m 600 /dev/stdin /etc/viam.json <<'VIAM_JSON'\n"
        f"{json.dumps(document, indent=2)}\n"
        "VIAM_JSON\n"
        "systemctl restart viam-agent\n"
    )
    gcloud_args: tuple[str, ...] = (
        "gcloud",
        "compute",
        "instances",
        "create",
        args.name,
        "--zone",
        args.zone,
        "--machine-type",
        args.machine_type,
        "--image-family",
        args.image_family,
        "--accelerator",
        "type=nvidia-l4,count=1",
        "--maintenance-policy",
        "TERMINATE",
        "--metadata-from-file",
        f"startup-script={startup_script_path}",
    )
    project = getattr(args, "project", None)
    if project:
        gcloud_args = gcloud_args + ("--image-project", project, "--project", project)
    return Plan(
        machine_name=args.name,
        resolved_config=resolved_config,
        gcloud_args=gcloud_args,
        credentials={"document": document, "startup_script": startup_script},
    )


def _build_parser() -> Any:
    parser = argparse.ArgumentParser(
        description="Create a sim machine for a real machine's config, end to end"
    )
    parser.add_argument("config", help="path to the real machine's config JSON")
    parser.add_argument("--name", required=True, help="name of the sim machine to create")
    parser.add_argument("--location-id", required=True)
    parser.add_argument("--org-id", default=None)
    parser.add_argument(
        "--project",
        default=None,
        help=f"GCP project for the instance (default: {DEFAULT_PROJECT_HINT})",
    )
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE)
    parser.add_argument("--image-family", default=DEFAULT_IMAGE_FAMILY)
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaac_module.config_resolver import UnmatchedHardwareError, resolve

    args = _build_parser().parse_args(argv)

    try:
        config = json.loads(Path(args.config).read_text())
        table = json.loads((REPO / "simulates.json").read_text())
        world_fragment = json.loads((REPO / "fragments" / "isaac-sim-world.json").read_text())
        resolution = resolve(config, table, world_fragment, allow_unmatched=args.allow_unmatched)
    except UnmatchedHardwareError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary, reported as a one-line error
        print(str(exc), file=sys.stderr)
        return 1

    if args.dry_run:
        _print_dry_run(args, resolution)
        return 0

    try:
        return _create_and_launch(args, resolution)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, reported as a one-line error
        print(str(exc), file=sys.stderr)
        return 1


def _print_dry_run(args: Any, resolution: Any) -> None:
    plan = build_plan(args, resolution.config, part_id="<pending>", secret="")
    print(f"swapped: {', '.join(resolution.swapped) or '(none)'}")
    print(f"passed through: {', '.join(resolution.passed_through) or '(none)'}")
    print(f"placeholders: {', '.join(resolution.placeholders) or '(none)'}")
    print(" ".join(plan.gcloud_args))
    redacted_document = copy.deepcopy(plan.credentials["document"])
    redacted_document["cloud"]["secret"] = "<secret>"
    print(json.dumps(redacted_document, indent=2))
    print(f"push target: main part of machine {args.name!r} in location {args.location_id!r}")


def _create_and_launch(args: Any, resolution: Any) -> int:
    return asyncio.run(_create_and_launch_async(args, resolution))


async def _create_and_launch_async(args: Any, resolution: Any) -> int:
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions

    dial_options = DialOptions.with_api_key(
        os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]
    )
    client = await ViamClient.create_from_dial_options(dial_options, APP_ADDRESS)
    try:
        app = client.app_client
        robot_id = await app.new_robot(args.name, args.location_id)
        main_part = await _main_part(app, robot_id)
        plan = build_plan(args, resolution.config, main_part.id, main_part.secret)
        _write_startup_script(plan)
        subprocess.run(plan.gcloud_args, check=True)

        deadline = time.monotonic() + ONLINE_TIMEOUT_S
        while True:
            main_part = await _main_part(app, robot_id)
            if _is_online(main_part):
                break
            if time.monotonic() >= deadline:
                print(f"part {main_part.id} never came online", file=sys.stderr)
                return 3
            await asyncio.sleep(ONLINE_POLL_S)

        await app.update_robot_part(main_part.id, main_part.name, resolution.config)
        print(f"{APP_ADDRESS}/machine/{robot_id}")
        print("livestream: open the machine's isaac-world component in the app's CONTROL tab")
        return 0
    finally:
        client.close()


async def _main_part(app: Any, robot_id: str) -> Any:
    parts = await app.get_robot_parts(robot_id)
    return next(part for part in parts if part.main_part)


def _is_online(part: Any) -> bool:
    """`part.last_access` (`viam.app.app_client.RobotPart`) is already a
    `datetime.datetime` converted from the proto `Timestamp` by the SDK, naive
    and in UTC. Attach the UTC tzinfo before comparing against an aware `now`.
    """
    if part.last_access is None:
        return False
    last_access = part.last_access
    if last_access.tzinfo is None:
        last_access = last_access.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last_access < timedelta(seconds=2 * ONLINE_POLL_S)


def _write_startup_script(plan: Plan) -> None:
    gcloud_args = list(plan.gcloud_args)
    metadata_arg = gcloud_args[gcloud_args.index("--metadata-from-file") + 1]
    path = metadata_arg.split("=", 1)[1]
    Path(path).write_text(plan.credentials["startup_script"])


if __name__ == "__main__":
    raise SystemExit(main())
