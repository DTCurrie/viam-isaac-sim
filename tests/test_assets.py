from pathlib import Path

import pytest
from PIL import Image

from isaac_module import assets
from isaac_module.assets import resolve_asset
from isaac_module.materials import (
    BUNDLED_MATERIAL_NAMES,
    MANIFEST_PATH,
    MAX_MATERIAL_FILE_BYTES,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
MATERIALS_DIR = ASSETS_DIR / "materials"
MAX_ASSET_BYTES = 5 * 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SQUARE_SIDES = {512, 1024}


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


def test_bundled_material_manifest_names_match_constant():
    manifest = load_manifest(MANIFEST_PATH)
    assert set(manifest) == set(BUNDLED_MATERIAL_NAMES)


def test_bundled_material_maps_exist_and_are_sized_png_files():
    manifest = load_manifest(MANIFEST_PATH)
    for material_set in manifest.values():
        for scheme_path in material_set.maps.values():
            path = Path(resolve_asset(scheme_path))
            assert path.is_file(), f"{path} is not a regular file"
            size = path.stat().st_size
            assert 1 <= size <= MAX_MATERIAL_FILE_BYTES, f"{path} is {size} bytes"
            with open(path, "rb") as f:
                assert f.read(len(PNG_MAGIC)) == PNG_MAGIC, f"{path} is not a PNG"


def test_bundled_material_maps_are_square_pillow_images():
    manifest = load_manifest(MANIFEST_PATH)
    for material_set in manifest.values():
        for scheme_path in material_set.maps.values():
            path = Path(resolve_asset(scheme_path))
            with Image.open(path) as img:
                width, height = img.size
                assert width == height, f"{path} is not square ({img.size})"
                assert width in SQUARE_SIDES, f"{path} side {width} not in {SQUARE_SIDES}"


def test_bundled_material_sets_have_cc0_license_files():
    manifest = load_manifest(MANIFEST_PATH)
    for name in manifest:
        license_path = MATERIALS_DIR / name / "LICENSE.md"
        assert license_path.exists(), f"{license_path} is missing"
        assert "CC0" in license_path.read_text()


def test_painted_sets_carry_no_albedo_and_concrete_floor_does():
    manifest = load_manifest(MANIFEST_PATH)
    assert "albedo" not in manifest["painted_wood"].maps
    assert "albedo" not in manifest["painted_mat"].maps
    assert "albedo" in manifest["concrete_floor"].maps


def test_material_set_directories_hold_only_manifest_maps_and_license():
    manifest = load_manifest(MANIFEST_PATH)
    for name, material_set in manifest.items():
        set_dir = MATERIALS_DIR / name
        expected = {Path(resolve_asset(p)).name for p in material_set.maps.values()}
        expected.add("LICENSE.md")
        actual = {p.name for p in set_dir.iterdir() if p.is_file()}
        assert actual == expected, f"{set_dir}: {actual} != {expected}"


def test_bundled_material_png_count_is_seven():
    png_paths = list(MATERIALS_DIR.rglob("*.png"))
    assert len(png_paths) == 7, [str(p) for p in png_paths]
