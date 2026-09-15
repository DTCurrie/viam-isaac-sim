"""Lists NVIDIA-hosted material trees against the Isaac assets root, for phase
3's item 0 (`.claude/plans/photoreal-workcell/phase-3-pbr-materials.md`): is
there a hosted wood or rubber PBR set that beats the bundled ambientCG ones.

Runs INSIDE Isaac Sim's python (it boots a headless ``SimulationApp`` to get
the asset resolver and ``omni.client``), e.g. on the GPU machine::

    $ISAAC_SIM_PATH/python.sh examples/list_nvidia_materials.py \\
        --out nvidia_materials_listing.json

Prints one line per listed entry, then the folders that look like a complete
material set (a normal map and a roughness map under the same parent), then
writes every listed row to ``--out`` as JSON. A root that fails to list
(missing on this release, or unreachable) prints ``not found`` and the script
carries on with the rest.

The pure helpers at the top take plain records so they are unit-tested on a
laptop (see tests/test_list_nvidia_materials.py). Kit/omni imports live
inside ``main`` only, so importing this module never needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

# omni.client.ItemFlags.CAN_HAVE_CHILDREN
FOLDER_FLAG = 4
DEFAULT_OUT_PATH = "nvidia_materials_listing.json"
# the assets server was unreachable, so nothing could be listed
EXIT_NO_ASSETS_ROOT = 2
NORMAL_MAP_FRAGMENT = "normal"
ROUGHNESS_MAP_FRAGMENT = "rough"
ISAAC_TREE_MARKER = "/Isaac/"


def candidate_roots(assets_root: str) -> list[str]:
    """The Isaac and NVIDIA material trees to list, in order: the ambientCG
    pattern sets under the Isaac tree, its parent Materials folder, and the
    NVIDIA/Materials/Base sibling trees under whatever holds the Isaac tree
    (cut ``assets_root`` at its last ``/Isaac/``; without one, ``assets_root``
    itself is that parent)."""
    if ISAAC_TREE_MARKER in assets_root:
        parent = assets_root.rsplit(ISAAC_TREE_MARKER, 1)[0]
    else:
        parent = assets_root
    return [
        f"{assets_root}/Isaac/Materials/Textures/Patterns",
        f"{assets_root}/Isaac/Materials",
        f"{parent}/NVIDIA/Materials/Base",
        f"{parent}/NVIDIA/Materials/2023_1/Base",
    ]


def listing_rows(root: str, entries: Sequence[Any]) -> list[dict[str, Any]]:
    """One row per duck-typed listing entry (``relative_path``, ``size``,
    ``flags``), folders identified by the ``CAN_HAVE_CHILDREN`` bit."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        is_folder = bool(entry.flags & FOLDER_FLAG)
        rows.append(
            {
                "root": root,
                "path": entry.relative_path,
                "size": entry.size,
                "is_folder": is_folder,
            }
        )
    return rows


def format_listing(rows: Sequence[dict[str, Any]]) -> str:
    """One line per row, ready to print."""
    lines = []
    for row in rows:
        kind = "dir " if row["is_folder"] else "file"
        lines.append(f"  [{kind}] {row['root']}/{row['path']}  ({row['size']} bytes)")
    return "\n".join(lines)


def usable_sets(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Folders (``root``/``path``) whose files, siblings under the same root
    and parent folder, include one name with ``normal`` and one with
    ``rough`` in it, case-insensitive."""
    names_by_folder: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row["is_folder"]:
            continue
        parent = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
        names_by_folder.setdefault((row["root"], parent), []).append(row["path"])

    folders: list[str] = []
    for (root, parent), names in names_by_folder.items():
        lowered = [name.lower() for name in names]
        has_normal = any(NORMAL_MAP_FRAGMENT in name for name in lowered)
        has_roughness = any(ROUGHNESS_MAP_FRAGMENT in name for name in lowered)
        if has_normal and has_roughness:
            folders.append(f"{root}/{parent}" if parent else root)
    return sorted(folders)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    from isaacsim import SimulationApp  # only importable inside Isaac Sim's python

    app = SimulationApp({"headless": True})
    try:
        import omni.client

        try:
            from isaacsim.storage.native import get_assets_root_path
        except ImportError:
            from omni.isaac.core.utils.nucleus import get_assets_root_path

        assets_root = get_assets_root_path()
        if assets_root is None:
            print("FAILED: could not reach the isaac assets server")
            return EXIT_NO_ASSETS_ROOT

        roots = candidate_roots(assets_root) + list(args.root)
        all_rows: list[dict[str, Any]] = []
        for root in roots:
            result, entries = omni.client.list(root)
            if result != omni.client.Result.OK:
                print(f"{root}: not found")
                continue
            rows = listing_rows(root, entries)
            all_rows.extend(rows)
            print(f"{root}:")
            print(format_listing(rows))

        print("usable sets (normal + roughness maps present):")
        for folder in usable_sets(all_rows):
            print(f"  {folder}")

        with open(args.out, "w") as out_file:
            json.dump(all_rows, out_file, indent=2, sort_keys=True)
        print(f"wrote {len(all_rows)} rows to {args.out}")
        return 0
    finally:
        app.close()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="path to write the listing JSON to")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="an extra root to list, beside the candidate Isaac/NVIDIA roots (repeatable)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
