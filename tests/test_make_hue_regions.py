import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "make_hue_regions.py"
sys.path.insert(0, str(_MODULE_PATH.parent))
_spec = importlib.util.spec_from_file_location("make_hue_regions", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
regions = importlib.util.module_from_spec(_spec)
# dataclasses resolve their own module out of sys.modules at class-creation
# time, so the module has to be registered before it is executed
sys.modules["make_hue_regions"] = regions
_spec.loader.exec_module(regions)


def _intrinsics(width: int = 640, height: int = 480) -> object:
    return regions.Intrinsics(
        width=width, height=height, fx=400.0, fy=400.0, cx=width / 2, cy=height / 2
    )


def _looking_along_world_x() -> tuple[np.ndarray, np.ndarray]:
    """A camera 1 m back along world -x looking toward +x, its image x along
    world +y and its image y along world -z."""
    rotation = np.column_stack(
        [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])]
    )
    return rotation, np.array([-1000.0, 0.0, 800.0])


def _geometry(
    name: str, x: float, y: float, z: float, edge_mm: float = 60.0, color=(1.0, 0.0, 0.0)
):
    return {
        "name": name,
        "box_dims_mm": [edge_mm, edge_mm, edge_mm],
        "pose_in_world_mm": {"x": x, "y": y, "z": z},
        "color": list(color),
    }


# ----------------------------------------------------------------------
# colour helpers
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rgb", "expected_deg"),
    [((1.0, 0.0, 0.0), 0.0), ((0.0, 1.0, 0.0), 120.0), ((0.0, 0.0, 1.0), 240.0)],
)
def test_hue_degrees_of_pure_primaries(rgb, expected_deg):
    assert regions.hue_degrees(rgb) == pytest.approx(expected_deg)


def test_hue_degrees_accepts_both_0_1_and_0_255_scales():
    assert regions.hue_degrees([255.0, 0.0, 0.0]) == pytest.approx(regions.hue_degrees([1.0, 0, 0]))


def test_saturation_of_gray_is_zero_and_of_a_primary_is_one():
    assert regions.saturation([128.0, 128.0, 128.0]) == pytest.approx(0.0)
    assert regions.saturation([255.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_hue_gap_takes_the_shorter_way_round_the_circle():
    assert regions.hue_gap_degrees(350.0, 10.0) == pytest.approx(20.0)
    assert regions.hue_gap_degrees(10.0, 350.0) == pytest.approx(20.0)
    assert regions.hue_gap_degrees(0.0, 180.0) == pytest.approx(180.0)


# ----------------------------------------------------------------------
# projection
# ----------------------------------------------------------------------


def test_project_puts_a_boresight_point_at_the_principal_point():
    rotation, translation = _looking_along_world_x()
    u, v, depth = regions.project((0.0, 0.0, 800.0), rotation, translation, _intrinsics())
    assert (u, v, depth) == pytest.approx((320.0, 240.0, 1000.0))


def test_project_moves_u_with_world_y_and_v_against_world_z():
    rotation, translation = _looking_along_world_x()
    intrinsics = _intrinsics()
    u_right, v_right, _ = regions.project((0.0, 100.0, 800.0), rotation, translation, intrinsics)
    u_up, v_up, _ = regions.project((0.0, 0.0, 900.0), rotation, translation, intrinsics)
    assert (u_right, v_right) == pytest.approx((360.0, 240.0))
    assert (u_up, v_up) == pytest.approx((320.0, 200.0))


def test_project_returns_nan_pixels_for_a_point_behind_the_camera():
    rotation, translation = _looking_along_world_x()
    u, v, depth = regions.project((-2000.0, 0.0, 800.0), rotation, translation, _intrinsics())
    assert np.isnan(u) and np.isnan(v)
    assert depth < 0


def test_project_scales_with_depth():
    rotation, translation = _looking_along_world_x()
    intrinsics = _intrinsics()
    near_u, _, near_depth = regions.project((0.0, 100.0, 800.0), rotation, translation, intrinsics)
    far_u, _, far_depth = regions.project((1000.0, 100.0, 800.0), rotation, translation, intrinsics)
    assert far_depth == pytest.approx(2 * near_depth)
    assert far_u - intrinsics.cx == pytest.approx((near_u - intrinsics.cx) / 2)


# ----------------------------------------------------------------------
# box placement
# ----------------------------------------------------------------------


def test_box_around_is_centred_and_a_fraction_of_the_projection():
    box = regions.box_around(320.0, 240.0, 60.0, _intrinsics())
    assert box == (308, 228, 332, 252)


def test_box_around_never_shrinks_below_the_minimum():
    x0, y0, x1, y1 = regions.box_around(320.0, 240.0, 1.0, _intrinsics())
    assert (x1 - x0, y1 - y0) == (regions.MIN_BOX_PX, regions.MIN_BOX_PX)


@pytest.mark.parametrize("u,v", [(2.0, 240.0), (638.0, 240.0), (320.0, 2.0), (320.0, 478.0)])
def test_box_around_rejects_a_box_that_would_clip_the_frame(u, v):
    assert regions.box_around(u, v, 60.0, _intrinsics()) is None


# ----------------------------------------------------------------------
# candidate ranking
# ----------------------------------------------------------------------


def test_rank_candidates_orders_the_nearest_block_first():
    rotation, translation = _looking_along_world_x()
    near = _geometry("block_red_1", 0.0, 0.0, 800.0)
    far = _geometry("block_red_2", 500.0, 0.0, 800.0)
    ranked, rejected = regions.rank_candidates([far, near], rotation, translation, _intrinsics())
    assert [c.name for c in ranked] == ["block_red_1", "block_red_2"]
    assert not any(rejected.values())


def test_rank_candidates_counts_a_block_behind_the_camera():
    rotation, translation = _looking_along_world_x()
    ranked, rejected = regions.rank_candidates(
        [_geometry("block_red_1", -3000.0, 0.0, 800.0)], rotation, translation, _intrinsics()
    )
    assert ranked == []
    assert rejected[regions.BEHIND_CAMERA] == 1


def test_rank_candidates_counts_a_block_too_far_to_sample():
    rotation, translation = _looking_along_world_x()
    ranked, rejected = regions.rank_candidates(
        [_geometry("block_red_1", 500000.0, 0.0, 800.0)], rotation, translation, _intrinsics()
    )
    assert ranked == []
    assert rejected[regions.TOO_SMALL] == 1


def test_rank_candidates_counts_a_block_whose_box_leaves_the_frame():
    rotation, translation = _looking_along_world_x()
    ranked, rejected = regions.rank_candidates(
        [_geometry("block_red_1", 0.0, 900.0, 800.0)], rotation, translation, _intrinsics()
    )
    assert ranked == []
    assert rejected[regions.OUTSIDE_FRAME] == 1


def test_rank_candidates_carries_the_configured_colour_through():
    rotation, translation = _looking_along_world_x()
    (candidate,), _ = regions.rank_candidates(
        [_geometry("block_blue_1", 0.0, 0.0, 800.0, color=(0.05, 0.1, 0.9))],
        rotation,
        translation,
        _intrinsics(),
    )
    assert candidate.colour_rgb == pytest.approx((0.05, 0.1, 0.9))


def test_rank_candidates_measures_the_top_face_not_the_centre():
    """The box belongs on the face the camera sees, so a taller block of the
    same footprint projects to a different pixel."""
    rotation, translation = _looking_along_world_x()
    intrinsics = _intrinsics()
    (short,), _ = regions.rank_candidates(
        [_geometry("block_red_1", 0.0, 0.0, 800.0, edge_mm=60.0)],
        rotation,
        translation,
        intrinsics,
    )
    (tall,), _ = regions.rank_candidates(
        [_geometry("block_red_1", 0.0, 0.0, 800.0, edge_mm=200.0)],
        rotation,
        translation,
        intrinsics,
    )
    assert short.box[1] != tall.box[1]


# ----------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------


def _measurement(mean_rgb, colour_rgb=(0.9, 0.1, 0.1)):
    candidate = regions.Candidate(
        name="block_red_1",
        box=(0, 0, 4, 4),
        colour_rgb=colour_rgb,
        projected_px=40.0,
        centre_offset_px=0.0,
    )
    return regions.Measurement(
        candidate=candidate,
        srgb_hex="#000000",
        mean_rgb=mean_rgb,
        hue_deg=regions.hue_degrees(mean_rgb),
        saturation=regions.saturation(mean_rgb),
    )


def test_sample_verdict_accepts_a_saturated_sample_of_the_right_hue():
    assert regions.sample_verdict(_measurement((222.0, 103.0, 101.0))) == "ok"


def test_sample_verdict_rejects_a_washed_out_sample_whatever_its_hue():
    """A gray patch has an arbitrary hue, so the hue check alone would pass it.
    This is the GPU case where a box landed on the arm, not the block."""
    grey = _measurement((122.0, 123.0, 121.0), colour_rgb=(0.05, 0.65, 0.1))
    assert regions.hue_gap_degrees(regions.hue_degrees(grey.candidate.colour_rgb), grey.hue_deg) < (
        regions.HUE_TOLERANCE_DEG
    )
    assert regions.sample_verdict(grey).startswith("REJECT washed out")


def test_sample_verdict_rejects_a_saturated_sample_of_the_wrong_hue():
    verdict = regions.sample_verdict(_measurement((70.0, 100.0, 221.0)))
    assert verdict.startswith("REJECT hue")


# ----------------------------------------------------------------------
# output shapes
# ----------------------------------------------------------------------


def test_blocks_by_colour_keeps_only_pool_blocks():
    grouped = regions.blocks_by_colour(
        [
            _geometry("block_red_1", 0, 0, 0),
            _geometry("block_red_2", 0, 0, 0),
            _geometry("place_pad_red", 0, 0, 0),
            _geometry("table_source", 0, 0, 0),
        ]
    )
    assert sorted(grouped) == ["red"]
    assert [g["name"] for g in grouped["red"]] == ["block_red_1", "block_red_2"]


def test_rendered_hue_block_round_trips_through_the_cell_layout_colours():
    from isaac_module.cell_layout import BLOCK_COLORS

    measurements = {colour: _measurement((222.0, 103.0, 101.0)) for colour in BLOCK_COLORS}
    block = regions.rendered_hue_block(measurements, BLOCK_COLORS)
    namespace: dict = {}
    exec(block, namespace)
    assert list(namespace["RENDERED_BLOCK_HUE_DEG"]) == list(BLOCK_COLORS)


def test_parse_args_defaults_to_auto_camera_and_no_motion():
    args = regions._parse_args([])
    assert (args.camera, args.look, args.scatter_seed) == ("auto", False, None)
    assert args.out == regions.DEFAULT_OUT_PATH


def test_parse_args_accepts_a_look_and_a_scatter_seed():
    args = regions._parse_args(["--camera", "wrist-cam", "--look", "--scatter-seed", "20"])
    assert (args.camera, args.look, args.scatter_seed) == ("wrist-cam", True, 20)


# ----------------------------------------------------------------------
# joint winding
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wound_deg", "expected_deg"),
    [(0.0, 0.0), (90.0, 90.0), (360.0, 0.0), (-360.0, 0.0), (3789.86, -170.14), (540.0, -180.0)],
)
def test_unwound_degrees_removes_whole_turns(wound_deg, expected_deg):
    assert regions.unwound_degrees(wound_deg) == pytest.approx(expected_deg, abs=1e-6)


def test_unwound_degrees_keeps_the_same_physical_orientation():
    for angle in (3789.86, -725.0, 1080.5):
        assert (regions.unwound_degrees(angle) - angle) % 360.0 == pytest.approx(0.0, abs=1e-6)


def test_wound_joint_indices_flags_only_joints_past_the_limit():
    positions = [137.7, -150.2, -47.7, 107.4, 360.0, -87.1]
    assert regions.wound_joint_indices(positions) == [4]


def test_wound_joint_indices_is_empty_for_an_ordinary_pose():
    assert regions.wound_joint_indices([137.7, -150.2, -47.7, 107.4, -0.03, -87.1]) == []


def test_wound_joint_indices_honours_an_explicit_limit():
    assert regions.wound_joint_indices([0.0, 190.0], limit_deg=180.0) == [1]
    assert regions.wound_joint_indices([0.0, 190.0], limit_deg=200.0) == []
