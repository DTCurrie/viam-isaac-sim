import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

_MODULE_PATH = _EXAMPLES_DIR / "gpu_checklist_photoreal.py"
_spec = importlib.util.spec_from_file_location("gpu_checklist_photoreal", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
checklist = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_photoreal"] = checklist
_spec.loader.exec_module(checklist)

HUE_TOLERANCE_DEG = 0.5


def test_hue_degrees_of_pure_primaries():
    assert abs(checklist.hue_degrees([255.0, 0.0, 0.0]) - 0.0) <= HUE_TOLERANCE_DEG
    assert abs(checklist.hue_degrees([0.0, 255.0, 0.0]) - 120.0) <= HUE_TOLERANCE_DEG
    assert abs(checklist.hue_degrees([0.0, 0.0, 255.0]) - 240.0) <= HUE_TOLERANCE_DEG


def test_saturation_of_gray_is_zero_and_of_pure_red_is_one():
    assert checklist.saturation([128.0, 128.0, 128.0]) == 0.0
    assert checklist.saturation([255.0, 0.0, 0.0]) == 1.0


def test_rendered_hue_block_round_trips_with_the_six_block_colors_in_order():
    from isaac_module.cell_layout import BLOCK_COLORS

    samples = {
        "red": [255.0, 0.0, 0.0],
        "green": [0.0, 255.0, 0.0],
        "blue": [0.0, 0.0, 255.0],
        "yellow": [255.0, 255.0, 0.0],
        "purple": [128.0, 0.0, 128.0],
        "orange": [255.0, 128.0, 0.0],
    }

    rendered = checklist.rendered_hue_block(samples)
    assert rendered.startswith("RENDERED_BLOCK_HUE_DEG = {")

    body = rendered[len("RENDERED_BLOCK_HUE_DEG = ") :]
    parsed = ast.literal_eval(body)

    assert isinstance(parsed, dict)
    assert list(parsed.keys()) == list(BLOCK_COLORS)
    for colour, hue in parsed.items():
        assert abs(hue - checklist.hue_degrees(samples[colour])) <= HUE_TOLERANCE_DEG


def test_ready_time_s_returns_elapsed_time_to_the_first_ready_sample():
    samples = [
        (100.0, {"ready": False}),
        (100.5, {"ready": False}),
        (101.0, {"ready": True}),
        (101.5, {"ready": True}),
    ]
    assert checklist.ready_time_s(samples) == 1.0


def test_ready_time_s_returns_none_when_never_ready():
    samples = [
        (100.0, {"ready": False}),
        (100.5, {"ready": False}),
    ]
    assert checklist.ready_time_s(samples) is None


def test_ready_time_s_returns_none_for_an_empty_sequence():
    assert checklist.ready_time_s([]) is None


def test_load_regions_accepts_a_full_six_colour_mapping():
    import json

    from isaac_module.cell_layout import BLOCK_COLORS

    text = json.dumps({colour: [0, 0, 10, 10] for colour in BLOCK_COLORS})
    regions = checklist.load_regions(text)
    assert regions == {colour: (0, 0, 10, 10) for colour in BLOCK_COLORS}


def test_load_regions_rejects_a_missing_colour():
    import json

    from isaac_module.cell_layout import BLOCK_COLORS

    incomplete = {colour: [0, 0, 10, 10] for colour in BLOCK_COLORS if colour != "purple"}
    with pytest.raises(ValueError, match="purple"):
        checklist.load_regions(json.dumps(incomplete))


def test_load_regions_rejects_a_box_where_x0_is_not_less_than_x1():
    import json

    from isaac_module.cell_layout import BLOCK_COLORS

    boxes = {colour: [0, 0, 10, 10] for colour in BLOCK_COLORS}
    boxes["red"] = [10, 0, 10, 10]
    with pytest.raises(ValueError, match="red"):
        checklist.load_regions(json.dumps(boxes))


def test_parse_args_accepts_regions():
    args = checklist._parse_args(["--mock", "--regions", "/tmp/regions.json"])
    assert args.regions == "/tmp/regions.json"


def test_parse_args_accepts_mock_and_rejects_missing_address_without_it():
    args = checklist._parse_args(["--mock"])
    assert args.mock is True

    args = checklist._parse_args(["--address", "10.0.0.1"])
    assert args.mock is False
    assert args.address == "10.0.0.1"


def test_main_reports_failure_and_exits_nonzero_when_neither_mock_nor_address_given(capsys):
    exit_code = checklist.main([])
    assert exit_code == 1
    assert "--address is required" in capsys.readouterr().out


def test_visual_top_offsets_mm_reports_the_offset_above_the_collider_top():
    visual_props = [
        {
            "name": "table_source_visual",
            "collider": "table_source",
            "bounds_m": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.7515]},
        }
    ]
    geometries = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 375.0},
        }
    ]
    offsets = checklist.visual_top_offsets_mm(visual_props, geometries)
    assert offsets["table_source_visual"] == pytest.approx(1.5, abs=1e-6)


def test_visual_top_offsets_mm_is_none_for_unmeasured_bounds():
    visual_props = [{"name": "table_source_visual", "collider": "table_source", "bounds_m": None}]
    geometries = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": 0.0, "y": 0.0, "z": 375.0},
        }
    ]
    offsets = checklist.visual_top_offsets_mm(visual_props, geometries)
    assert offsets["table_source_visual"] is None


def test_visual_top_offsets_mm_is_none_for_a_missing_collider():
    visual_props = [
        {
            "name": "table_source_visual",
            "collider": "table_source",
            "bounds_m": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.7515]},
        }
    ]
    offsets = checklist.visual_top_offsets_mm(visual_props, geometries=[])
    assert offsets["table_source_visual"] is None


def test_geometry_diff_of_identical_geometries_is_empty():
    geometries = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 375.0},
        }
    ]
    assert checklist.geometry_diff(geometries, geometries) == []


def test_geometry_diff_reports_a_one_millimeter_z_move():
    baseline = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 375.0},
        }
    ]
    current = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 376.0},
        }
    ]
    diff = checklist.geometry_diff(baseline, current)
    assert len(diff) == 1
    assert "table_source" in diff[0]


def test_geometry_diff_reports_a_renamed_prop_on_both_sides():
    baseline = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 375.0},
        }
    ]
    current = [
        {
            "name": "table_source_renamed",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 375.0},
        }
    ]
    diff = checklist.geometry_diff(baseline, current)
    assert any("only in baseline" in line and "table_source" in line for line in diff)
    assert any("only in current" in line and "table_source_renamed" in line for line in diff)


def test_geometry_diff_ignores_a_sub_tolerance_jitter():
    baseline = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 375.0},
        }
    ]
    current = [
        {
            "name": "table_source",
            "box_dims_mm": [1200.0, 800.0, 750.0],
            "pose_in_world_mm": {"x": -1200.0, "y": 0.0, "z": 375.2},
        }
    ]
    assert checklist.geometry_diff(baseline, current) == []


def test_parse_args_default_phase_is_three():
    args = checklist._parse_args(["--mock"])
    assert args.phase == 3


def test_parse_args_accepts_phase_one():
    args = checklist._parse_args(["--mock", "--phase", "1"])
    assert args.phase == 1


def test_parse_args_accepts_phase_three():
    args = checklist._parse_args(["--mock", "--phase", "3"])
    assert args.phase == 3


def test_phase_3_items_has_seven_entries_numbered_zero_to_six():
    assert len(checklist.PHASE_3_ITEMS) == 7
    for index, item in enumerate(checklist.PHASE_3_ITEMS):
        assert item.startswith(f"{index}.")


def test_main_runs_phase_3_against_the_mock_and_exits_zero(capsys):
    exit_code = checklist.main(["--mock", "--phase", "3"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "list_nvidia_materials.py" in out
    assert "painted_wood" in out


def test_parse_args_accepts_geometry_baseline_and_dump_geometries():
    args = checklist._parse_args(
        [
            "--mock",
            "--geometry-baseline",
            "/tmp/baseline.json",
            "--dump-geometries",
            "/tmp/dump.json",
        ]
    )
    assert args.geometry_baseline == "/tmp/baseline.json"
    assert args.dump_geometries == "/tmp/dump.json"
