"""Asset path schemes for textures, HDRIs and USD files named in world config.

Two module schemes sit beside whatever Isaac's own resolver accepts (absolute
paths, ``http(s)://``, ``omniverse://``):

* ``module://<rel>`` resolves under the module's bundled ``assets/`` directory,
  shipped inside ``module.tar.gz``. Only CC0 assets at 1k/2k live there.
* ``data://<rel>`` resolves under ``$VIAM_MODULE_DATA/assets/<rel>``, which
  survives module upgrades and holds anything too big or not licensed to
  commit.

Anything else passes through untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

MODULE_SCHEME = "module://"
DATA_SCHEME = "data://"
# paths Isaac's own resolver fetches; nothing local checks their existence
REMOTE_ASSET_SCHEMES = ("http://", "https://", "omniverse://")
ASSETS_DIR_NAME = "assets"
# mirrors run.sh: the module data directory when viam-server does not set one
DEFAULT_DATA_ROOT = "/opt/viam-isaac-sim"
# the repo or tarball root: run.sh cds here, and src/isaac_module/assets.py
# sits two levels below it. Derived from __file__, never from the CWD.
MODULE_ROOT = Path(__file__).resolve().parent.parent.parent


def data_root() -> Path:
    """``$VIAM_MODULE_DATA`` when set, else ``DEFAULT_DATA_ROOT``. The
    ``assets`` segment is appended by ``resolve_asset_path``, not here."""
    env_value = os.environ.get("VIAM_MODULE_DATA")
    if env_value:
        return Path(env_value)
    return Path(DEFAULT_DATA_ROOT)


def _resolve_scheme(relative_path: str, scheme: str, base: Path) -> str:
    if not relative_path or ".." in Path(relative_path).parts:
        raise ValueError(f"invalid path for {scheme} scheme: {relative_path!r}")
    return str(base / ASSETS_DIR_NAME / relative_path)


def resolve_asset_path(value: str, *, module_root: Path, data_root: Path) -> str:
    """Pure. ``module://rel`` -> ``<module_root>/assets/rel``; ``data://rel`` ->
    ``<data_root>/assets/rel``; any other string is returned unchanged. An
    empty ``rel`` or one containing a ``..`` segment raises ``ValueError``."""
    if value.startswith(MODULE_SCHEME):
        return _resolve_scheme(value[len(MODULE_SCHEME) :], MODULE_SCHEME, module_root)
    if value.startswith(DATA_SCHEME):
        return _resolve_scheme(value[len(DATA_SCHEME) :], DATA_SCHEME, data_root)
    return value


def resolve_asset(value: str) -> str:
    """``resolve_asset_path`` with the module's own roots (``MODULE_ROOT`` and
    ``data_root()``)."""
    return resolve_asset_path(value, module_root=MODULE_ROOT, data_root=data_root())
