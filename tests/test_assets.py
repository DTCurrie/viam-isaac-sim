from pathlib import Path

import pytest

from isaac_module import assets

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
MAX_ASSET_BYTES = 5 * 1024 * 1024


def test_resolve_module_scheme():
    module_root = Path("/mod")
    data_root = Path("/data")
    resolved = assets.resolve_asset_path(
        "module://hdri/foo.hdr", module_root=module_root, data_root=data_root
    )
    assert resolved == str(module_root / "assets" / "hdri" / "foo.hdr")


def test_resolve_data_scheme():
    module_root = Path("/mod")
    data_root = Path("/data")
    resolved = assets.resolve_asset_path(
        "data://textures/bar.png", module_root=module_root, data_root=data_root
    )
    assert resolved == str(data_root / "assets" / "textures" / "bar.png")


def test_resolve_module_scheme_empty_raises():
    with pytest.raises(ValueError, match="module://"):
        assets.resolve_asset_path("module://", module_root=Path("/mod"), data_root=Path("/data"))


def test_resolve_data_scheme_dotdot_raises():
    with pytest.raises(ValueError, match="data://"):
        assets.resolve_asset_path("data://../x", module_root=Path("/mod"), data_root=Path("/data"))


def test_resolve_passthrough_absolute_path():
    value = "/absolute/path/to/asset.hdr"
    resolved = assets.resolve_asset_path(value, module_root=Path("/mod"), data_root=Path("/data"))
    assert resolved == value


def test_resolve_passthrough_https_url():
    value = "https://example.com/asset.hdr"
    resolved = assets.resolve_asset_path(value, module_root=Path("/mod"), data_root=Path("/data"))
    assert resolved == value


def test_resolve_passthrough_omniverse_url():
    value = "omniverse://server/asset.usd"
    resolved = assets.resolve_asset_path(value, module_root=Path("/mod"), data_root=Path("/data"))
    assert resolved == value


def test_data_root_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", "/custom/data/root")
    assert assets.data_root() == Path("/custom/data/root")


def test_data_root_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    assert assets.data_root() == Path(assets.DEFAULT_DATA_ROOT)


def test_module_root_derivation_points_at_repo_root():
    assert (assets.MODULE_ROOT / "Makefile").exists()


def test_bundled_assets_are_within_size_bounds():
    for path in ASSETS_DIR.rglob("*"):
        if path.is_dir() or path.name == "LICENSE.md":
            continue
        size = path.stat().st_size
        assert 1 <= size <= MAX_ASSET_BYTES, f"{path} is {size} bytes"


def test_bundled_hdri_is_radiance_format():
    hdri_path = ASSETS_DIR / "hdri" / "empty_warehouse_01_1k.hdr"
    with open(hdri_path, "rb") as f:
        header = f.read(10)
    assert header.startswith(b"#?RADIANCE")


def test_hdri_license_exists_and_is_cc0():
    license_path = ASSETS_DIR / "hdri" / "LICENSE.md"
    assert license_path.exists()
    assert "CC0" in license_path.read_text()


def test_makefile_tar_line_includes_assets():
    makefile_text = (REPO_ROOT / "Makefile").read_text()
    tar_line = next(
        line for line in makefile_text.splitlines() if line.startswith("\ttar czf module.tar.gz")
    )
    args = tar_line.split()
    assert "assets" in args
