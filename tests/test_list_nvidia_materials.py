import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "list_nvidia_materials.py"
_spec = importlib.util.spec_from_file_location("list_nvidia_materials", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
lister = importlib.util.module_from_spec(_spec)
sys.modules["list_nvidia_materials"] = lister
_spec.loader.exec_module(lister)

FOLDER_FLAG = 4


@dataclass(frozen=True)
class _Entry:
    relative_path: str
    size: int
    flags: int


def test_candidate_roots_yields_the_isaac_and_nvidia_trees():
    assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
    roots = lister.candidate_roots(assets_root)
    assert roots == [
        f"{assets_root}/Isaac/Materials/Textures/Patterns",
        f"{assets_root}/Isaac/Materials",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
        "/NVIDIA/Materials/Base",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
        "/NVIDIA/Materials/2023_1/Base",
    ]


def test_candidate_roots_falls_back_to_the_assets_root_without_an_isaac_tree():
    assets_root = "https://example.com/Materials"
    roots = lister.candidate_roots(assets_root)
    assert roots[2] == "https://example.com/Materials/NVIDIA/Materials/Base"
    assert roots[3] == "https://example.com/Materials/NVIDIA/Materials/2023_1/Base"


def test_listing_rows_marks_folders_from_the_flag_bit():
    entries = [
        _Entry("Wood_Oak", 0, FOLDER_FLAG),
        _Entry("Wood_Oak.mdl", 4096, 0),
    ]
    rows = lister.listing_rows("root", entries)
    assert rows == [
        {"root": "root", "path": "Wood_Oak", "size": 0, "is_folder": True},
        {"root": "root", "path": "Wood_Oak.mdl", "size": 4096, "is_folder": False},
    ]


def test_format_listing_has_one_line_per_row():
    rows = lister.listing_rows("root", [_Entry("a", 1, 0), _Entry("b", 2, FOLDER_FLAG)])
    formatted = lister.format_listing(rows)
    assert len(formatted.splitlines()) == 2
    assert "root/a" in formatted
    assert "root/b" in formatted


def test_usable_sets_finds_a_folder_with_normal_and_roughness_maps():
    rows = lister.listing_rows(
        "root",
        [
            _Entry("Rubber/X_BaseColor.png", 10, 0),
            _Entry("Rubber/X_Normal.png", 10, 0),
            _Entry("Rubber/X_Roughness.png", 10, 0),
        ],
    )
    assert lister.usable_sets(rows) == ["root/Rubber"]


def test_usable_sets_skips_a_folder_with_only_a_colour_map():
    rows = lister.listing_rows(
        "root",
        [
            _Entry("Wood/X_BaseColor.png", 10, 0),
        ],
    )
    assert lister.usable_sets(rows) == []


def test_out_default_parses():
    args = lister._parse_args([])
    assert args.out == lister.DEFAULT_OUT_PATH
    assert args.root == []
