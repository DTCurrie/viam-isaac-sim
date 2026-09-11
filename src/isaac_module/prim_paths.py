"""USD prim path naming and defaults shared across component handles."""

from __future__ import annotations

from typing import Any

from .asset_catalog import KNOWN_ASSETS


def prim_name(name: str) -> str:
    """Component names may contain characters USD prim names can't."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def default_ee_prim_path(attrs: dict[str, Any], name: str) -> str | None:
    """The EE prim an arm's prim_pose falls back to when no explicit
    prim_path is given, matching the normalisation SimManager uses to
    spawn/mock the arm's prim. None when the arm's asset is unknown or
    declares no ee_prim, so a caller can skip rather than guess wrong."""
    asset = attrs.get("asset")
    ee_prim = KNOWN_ASSETS[asset].get("ee_prim") if asset and asset in KNOWN_ASSETS else None
    if not ee_prim:
        return None
    prim_path = attrs.get("prim_path") or f"/World/{prim_name(name)}"
    return f"{prim_path}/{ee_prim}"


def default_base_prim_path(attrs: dict[str, Any], name: str) -> str:
    """The prim_pose verb's default path for a base: the asset's body prim
    under the base's root when the asset declares one (the jetbot's root is
    a plain Xform physics never moves, its chassis is what drives), else the
    root itself, matching the normalisation SimManager uses to spawn/mock
    it."""
    root = attrs.get("prim_path") or f"/World/{prim_name(name)}"
    asset = attrs.get("asset")
    body_prim = KNOWN_ASSETS[asset].get("body_prim") if asset and asset in KNOWN_ASSETS else None
    return f"{root}/{body_prim}" if body_prim else root
