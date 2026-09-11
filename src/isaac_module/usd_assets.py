"""USD asset path repair and composition introspection.

Rewrites references off the unresolvable dev host, strips nested
articulation roots, and reports what actually composed under a spawned
asset.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from viam.logging import getLogger

LOGGER = getLogger(__name__)

# The host the 5.0 2F-85 sub-references point at. It is NXDOMAIN outside
# NVIDIA, so anything referencing it composes empty.
UNRESOLVABLE_ASSET_HOST = "isaac-dev"
ASSETS_PATH_MARKER = "/Isaac/"  # every asset path is rooted here under both hosts

# Prepared asset layers (SimManager._prepared_asset_layer) are cached across
# module restarts under viam-server's per-module data dir when it sets one,
# else the system temp dir. Bump the version when the preparation rules
# change so stale copies are redone.
ASSET_CACHE_DIRNAME = "viam-isaac-sim-assets"
ASSET_LAYER_CACHE_VERSION = 2

# How many composition-dependency levels _prepared_asset_layer walks looking
# for unresolvable-host references. The 5.0 2F-85 needs 2 (root asset ->
# payload -> parts). The cap only bounds pathological reference chains.
ASSET_PREPARE_MAX_DEPTH = 4


def _is_layer_path(asset_path: str) -> bool:
    """A dependency worth recursing into: an explicit path or URL to a USD
    layer. A bare search path (``OmniPBR.mdl``) belongs to the MDL resolver,
    and textures/MDLs cannot carry references."""
    if not (asset_path.startswith(("./", "../", "/")) or "://" in asset_path):
        return False
    return asset_path.rsplit("?", 1)[0].lower().endswith((".usd", ".usda", ".usdc"))


def asset_cache_dir() -> Path:
    base = os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()
    return Path(base) / ASSET_CACHE_DIRNAME


def _bucket_candidate(asset_path: str, assets_root: str) -> str | None:
    """``assets_root`` + the ``/Isaac/...`` tail of a path on the unresolvable
    host. None for a path anywhere else."""
    marker_at = asset_path.find(ASSETS_PATH_MARKER)
    if UNRESOLVABLE_ASSET_HOST not in asset_path or marker_at < 0:
        return None
    return assets_root.rstrip("/") + asset_path[marker_at:]


def _anchor_asset_path(asset_path: str, anchor_layer: str) -> str:
    """A layer-relative path (``./x`` or ``../x``) resolved against the layer
    at ``anchor_layer``. Every other path - absolute, URL, or a search path
    such as an MDL module name - is returned unchanged, since only the
    explicitly relative forms break when the layer is copied elsewhere."""
    if not (asset_path.startswith("./") or asset_path.startswith("../")):
        return asset_path
    segments = anchor_layer.rsplit("/", 1)[0].split("/")
    # never pop into the scheme/host of a URL ("https:", "", host) or past
    # the leading "" of an absolute path
    floor = 3 if "://" in anchor_layer else 1
    for part in asset_path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if len(segments) > floor:
                segments.pop()
            continue
        segments.append(part)
    return "/".join(segments)


def _prepared_asset_path(
    asset_path: str,
    anchor_layer: str,
    assets_root: str,
    exists: Callable[[str], bool | None],
    report: dict[str, Any],
) -> str:
    """What a prepared copy of ``anchor_layer`` holds in place of
    ``asset_path``: a path on the unresolvable host moves onto ``assets_root``
    (when the bucket has the file, or that cannot be checked - None), a
    layer-relative path becomes absolute. Records the (old, new) pairs under
    ``report["applied"]`` / ``["anchored"]`` and bucket files that provably
    do not exist under ``report["missing"]`` - those references are DROPPED
    (empty path removes them): a dead-host reference can never load and
    costs a ~12 s composition stall each (GPU 2026-09-04), so a missing
    visual is strictly better."""
    candidate = _bucket_candidate(asset_path, assets_root)
    if candidate is not None:
        if exists(candidate) is False:
            report["missing"].append(candidate)
            return ""
        report["applied"].append((asset_path, candidate))
        return candidate
    anchored = _anchor_asset_path(asset_path, anchor_layer)
    if anchored != asset_path:
        report["anchored"].append((asset_path, anchored))
    return anchored


# The 2F-85 asset has no `*_pad` links. The fingertip geometry that must
# collide is the `..._fingertipsstep_..` mesh under `left/right_inner_finger`.
PAD_PRIM_NAME_FRAGMENTS = ("pad", "fingertip", "inner_finger")


def _rewritten_references(
    sdf: Any,
    items: Sequence[Any],
    assets_root: str,
    exists: Callable[[str], bool | None],
) -> tuple[list[Any], list[tuple[str, str]], list[str]]:
    """Map every reference on the unresolvable host onto ``assets_root`` +
    its ``/Isaac/...`` path, keeping the reference's primPath/layerOffset.
    Returns (the full new item list, the (old, new) pairs changed, the bucket
    candidates that provably do not exist). An item whose bucket file does
    not exist (``exists`` returned False) is kept as-is - None means "could
    not check", so it is tried."""
    new_items: list[Any] = []
    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for item in items:
        asset_path = str(getattr(item, "assetPath", ""))
        candidate = _bucket_candidate(asset_path, assets_root)
        if candidate is None:
            new_items.append(item)
            continue
        if exists(candidate) is False:
            LOGGER.warning("assets root has no %s, keeping %s", candidate, asset_path)
            new_items.append(item)
            missing.append(candidate)
            continue
        new_items.append(sdf.Reference(candidate, item.primPath, item.layerOffset))
        pairs.append((asset_path, candidate))
    return new_items, pairs, missing


def _remove_articulation_roots(
    usd: Any, usd_physics: Any, physx_schema: Any, root_prim: Any
) -> list[str]:
    """Remove ``UsdPhysics.ArticulationRootAPI`` (and PhysX's companion
    ``PhysxArticulationAPI``) from every prim under ``root_prim`` that carries
    it, as a root-layer override (the asset layer is read-only). Returns the
    prim paths changed. A gripper attached under an arm must not bring its
    own articulation root: PhysX rejects a nested root and drops the whole
    articulation."""
    removed: list[str] = []
    for prim in _prim_range(usd, root_prim):
        if prim.IsInstanceProxy() or not prim.HasAPI(usd_physics.ArticulationRootAPI):
            continue
        prim.RemoveAPI(usd_physics.ArticulationRootAPI)
        if physx_schema is not None and prim.HasAPI(physx_schema.PhysxArticulationAPI):
            prim.RemoveAPI(physx_schema.PhysxArticulationAPI)
        removed.append(str(prim.GetPath()))
    return removed


def _pad_collision_status(
    usd: Any, usd_physics: Any, root_prim: Any
) -> list[tuple[str, bool, bool]]:
    """(path, has CollisionAPI itself, has it anywhere in its subtree) for
    every prim under ``root_prim`` whose name contains "pad". Pure over
    the pxr modules passed in, so it is testable with fakes."""
    status: list[tuple[str, bool, bool]] = []
    for prim in _prim_range(usd, root_prim):
        prim_name = prim.GetName().lower()
        if not any(fragment in prim_name for fragment in PAD_PRIM_NAME_FRAGMENTS):
            continue
        on_self = bool(prim.HasAPI(usd_physics.CollisionAPI))
        in_subtree = on_self or any(
            bool(child.HasAPI(usd_physics.CollisionAPI)) for child in _prim_range(usd, prim)
        )
        status.append((str(prim.GetPath()), on_self, in_subtree))
    return status


def _prim_range(usd: Any, root_prim: Any) -> Any:
    """Traverse ``root_prim``'s subtree INCLUDING instance proxies: Isaac's
    robot assets mark link meshes instanceable, and a default PrimRange
    stops at an instance, hiding the collision meshes under it."""
    traverse_instances = getattr(usd, "TraverseInstanceProxies", None)
    if traverse_instances is None:
        return usd.PrimRange(root_prim)
    return usd.PrimRange(root_prim, traverse_instances())


SAMPLE_PRIM_PATHS = 60


def _describe_composition(usd: Any, sdf: Any, root_prim: Any, usd_path: str) -> dict[str, Any]:
    """What composed under ``root_prim`` after referencing ``usd_path``, plus
    what the referenced layer itself declares - enough to tell "the asset
    did not load" from "it loaded with names we did not expect".
    Best-effort: every field degrades to an empty value rather than raising."""
    out: dict[str, Any] = {
        "children": [],
        "prim_count": 0,
        "instanceable_count": 0,
        "sample_paths": [],
        "layer_default_prim": "",
        "layer_root_prims": [],
    }
    try:
        out["children"] = [child.GetName() for child in root_prim.GetChildren()]
        prims = list(_prim_range(usd, root_prim))
        out["prim_count"] = len(prims)
        out["instanceable_count"] = sum(1 for prim in prims if prim.IsInstanceable())
        root_path = str(root_prim.GetPath())
        out["sample_paths"] = [
            str(prim.GetPath())[len(root_path) :] for prim in prims[1 : SAMPLE_PRIM_PATHS + 1]
        ]
    except Exception:
        LOGGER.exception("could not describe the prims under %s", root_prim)
    try:
        layer = sdf.Layer.FindOrOpen(usd_path)
        if layer is not None:
            out["layer_default_prim"] = str(layer.defaultPrim)
            out["layer_root_prims"] = [str(spec.path) for spec in layer.rootPrims]
    except Exception:
        LOGGER.exception("could not open the referenced layer %s", usd_path)
    return out


def _reference_asset_paths(usd: Any, root_prim: Any) -> list[str]:
    """Every reference/payload asset path authored on prims under
    ``root_prim`` (which hosts the composed asset actually pulls from).
    Best-effort - metadata shapes vary across USD versions, so failures
    yield an empty list rather than breaking the attach."""
    paths: list[str] = []
    for prim in _prim_range(usd, root_prim):
        for key in ("references", "payload"):
            try:
                list_op = prim.GetMetadata(key)
                items = list_op.GetAddedOrExplicitItems() if list_op is not None else []
            except Exception:  # noqa: BLE001 - metadata shapes vary across USD versions, failures yield an empty list
                items = []
            for item in items:
                asset_path = getattr(item, "assetPath", "")
                if asset_path:
                    paths.append(str(asset_path))
    return paths


def _version_string(version: tuple[int, int, int] | None) -> str | None:
    return None if version is None else ".".join(str(part) for part in version)


def _local_ip() -> str:
    """Best-effort primary local IP (no traffic is actually sent)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:  # noqa: BLE001 - best-effort IP probe, any failure falls back to empty string
        return ""
