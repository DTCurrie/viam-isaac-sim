#!/usr/bin/env python3
"""CLI entry point: resolve a real machine config into its sim twin.

Usage: `simulate_config.py <real-config.json> [--table PATH]
[--world-fragment PATH] [--allow-unmatched] [--only a,b] [--out PATH]`.

Writes the resolved config as two-space-indented JSON with a trailing newline
to stdout or `--out`, and reports which components were swapped, passed
through, or replaced with a placeholder on stderr. Exit 0 on success, 2 for
unmatched hardware, 1 for any other error, each with a one-line message on
stderr.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    from isaac_module.config_resolver import Resolution, UnmatchedHardwareError, resolve
except ModuleNotFoundError as missing:
    raise SystemExit(
        f"{missing}. Run this tool with the repo's interpreter: "
        f"{REPO}/.venv/bin/python {Path(__file__).name} ..."
    ) from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a real machine config into the config of its sim machine"
    )
    parser.add_argument("config", help="path to the real machine's config JSON")
    parser.add_argument("--table", default=str(REPO / "simulates.json"))
    parser.add_argument(
        "--world-fragment", default=str(REPO / "fragments" / "isaac-sim-world.json")
    )
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated component names")
    parser.add_argument("--out", default=None, help="write to this path instead of stdout")
    return parser


def _report(resolution: Resolution) -> None:
    print(f"swapped: {', '.join(resolution.swapped) or '(none)'}", file=sys.stderr)
    print(f"passed through: {', '.join(resolution.passed_through) or '(none)'}", file=sys.stderr)
    print(f"placeholders: {', '.join(resolution.placeholders) or '(none)'}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config = json.loads(Path(args.config).read_text())
        table = json.loads(Path(args.table).read_text())
        world_fragment = json.loads(Path(args.world_fragment).read_text())
        only = set(args.only.split(",")) if args.only else None
        resolution = resolve(
            config, table, world_fragment, allow_unmatched=args.allow_unmatched, only=only
        )
    except UnmatchedHardwareError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary, reported as a one-line error
        print(str(exc), file=sys.stderr)
        return 1

    _report(resolution)
    output = json.dumps(resolution.config, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(output)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
