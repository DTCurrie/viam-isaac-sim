#!/usr/bin/env python3
"""CLI entry point: resolve a real machine config into its sim twin.

Usage: `simulate_config.py <real-config.json> [--table PATH]
[--world-fragment PATH] [--allow-unmatched] [--only a,b] [--out PATH]`.
See `isaac_module.config_resolver.main` for the full contract.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    from isaac_module.config_resolver import main
except ModuleNotFoundError as missing:
    raise SystemExit(
        f"{missing}. Run this tool with the repo's interpreter: "
        f"{REPO}/.venv/bin/python {Path(__file__).name} ..."
    ) from None

if __name__ == "__main__":
    raise SystemExit(main())
