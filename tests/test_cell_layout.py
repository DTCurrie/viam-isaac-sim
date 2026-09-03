"""The zone constants are safe by construction: every point the arm must
visit sits inside 80% of the ur20's reach, every zone sits on the surface
that supports it, and the park grid stays clear of the cell."""

import itertools
import math

from isaac_module import cell_layout


def _planar_radius_mm(x: float, y: float) -> float:
    base_x, base_y = cell_layout.ARM_BASE_XY_MM
    return math.hypot(x - base_x, y - base_y)


def test_safe_reach_is_eighty_percent_of_the_ur20_envelope():
    assert cell_layout.UR20_REACH_MM == 1750.0
    assert cell_layout.MAX_PLANAR_REACH_MM == 1400.0


def test_every_scatter_corner_is_inside_safe_reach():
    for x, y in cell_layout.scatter_corners_mm():
        assert _planar_radius_mm(x, y) <= cell_layout.MAX_PLANAR_REACH_MM


def test_every_pad_centre_is_inside_safe_reach():
    for x, y in cell_layout.PAD_CENTRES_MM.values():
        assert _planar_radius_mm(x, y) <= cell_layout.MAX_PLANAR_REACH_MM


def test_the_scatter_zone_sits_on_the_source_table_with_margin():
    """Lower bound on the zone: inside the source-table footprint by at least
    50 mm, so a block scattered at the very edge still rests fully on the
    table (the reach test above is the upper bound)."""
    margin_mm = 50.0
    centre_x, centre_y = cell_layout.TABLE_CENTRES_MM["table_source"]
    half_x = cell_layout.TABLE_DIMS_MM[0] / 2.0 - margin_mm
    half_y = cell_layout.TABLE_DIMS_MM[1] / 2.0 - margin_mm
    x_low, x_high = cell_layout.SCATTER_ZONE_X_MM
    y_low, y_high = cell_layout.SCATTER_ZONE_Y_MM
    assert x_low < x_high
    assert y_low < y_high
    assert centre_x - half_x <= x_low
    assert x_high <= centre_x + half_x
    assert centre_y - half_y <= y_low
    assert y_high <= centre_y + half_y


def test_pads_sit_on_the_table_row_and_never_overlap():
    """Every pad footprint stays on the flush three-table surface (a pad may
    straddle the middle/place seam, the tops are level), and any two pads
    clear each other on at least one axis."""
    table_x_min = cell_layout.TABLE_CENTRES_MM["table_source"][0] - cell_layout.TABLE_DIMS_MM[0] / 2
    table_x_max = cell_layout.TABLE_CENTRES_MM["table_place"][0] + cell_layout.TABLE_DIMS_MM[0] / 2
    table_y_half = cell_layout.TABLE_DIMS_MM[1] / 2
    half_pad = cell_layout.PAD_SIDE_MM / 2.0

    assert set(cell_layout.PAD_CENTRES_MM) == set(cell_layout.BLOCK_COLORS)
    for x, y in cell_layout.PAD_CENTRES_MM.values():
        assert table_x_min <= x - half_pad
        assert x + half_pad <= table_x_max
        assert -table_y_half <= y - half_pad
        assert y + half_pad <= table_y_half
        # pads live on the place side, clear of the arm's table centre
        assert x - half_pad >= cell_layout.TABLE_DIMS_MM[0] / 2 - 50.0

    for (ax, ay), (bx, by) in itertools.combinations(cell_layout.PAD_CENTRES_MM.values(), 2):
        assert abs(ax - bx) >= cell_layout.PAD_SIDE_MM or abs(ay - by) >= cell_layout.PAD_SIDE_MM


def test_the_park_grid_holds_all_eighteen_blocks_clear_of_the_cell():
    """18 unique names (3 per color), every position behind the table row by
    at least 500 mm and spaced at least 150 mm from its neighbours, so parked
    blocks never touch the cell or each other."""
    positions = cell_layout.park_positions_mm()
    assert len(positions) == len(cell_layout.BLOCK_COLORS) * cell_layout.POOL_BLOCKS_PER_COLOR
    for color in cell_layout.BLOCK_COLORS:
        for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1):
            assert cell_layout.pool_block_name(color, index) in positions

    table_y_min = -cell_layout.TABLE_DIMS_MM[1] / 2
    for x, y in positions.values():
        assert y <= table_y_min - 500.0
        assert abs(x) <= cell_layout.TABLE_DIMS_MM[0] * 1.5  # within the cell's x span

    minimum_spacing_mm = 150.0
    for (ax, ay), (bx, by) in itertools.combinations(positions.values(), 2):
        assert math.hypot(ax - bx, ay - by) >= minimum_spacing_mm


def test_pad_slots_hold_max_size_blocks_on_the_pad_and_apart():
    """Both bounds: every slot keeps a MAX_BLOCK_SIZE_MM block fully on the
    300 mm pad, and any two slots keep two such blocks from touching."""
    half_block = cell_layout.MAX_BLOCK_SIZE_MM / 2
    half_pad = cell_layout.PAD_SIDE_MM / 2
    assert len(cell_layout.PAD_SLOT_OFFSETS_MM) == cell_layout.POOL_BLOCKS_PER_COLOR
    for x, y in cell_layout.PAD_SLOT_OFFSETS_MM:
        assert abs(x) + half_block <= half_pad
        assert abs(y) + half_block <= half_pad
    for (ax, ay), (bx, by) in itertools.combinations(cell_layout.PAD_SLOT_OFFSETS_MM, 2):
        assert math.hypot(ax - bx, ay - by) >= cell_layout.MAX_BLOCK_SIZE_MM * 2


def test_metre_helpers_mirror_the_millimetre_constants():
    (x0, y0, z0), (x1, y1, z1) = cell_layout.scatter_region_m()
    assert (x0 * 1000, x1 * 1000) == cell_layout.SCATTER_ZONE_X_MM
    assert (y0 * 1000, y1 * 1000) == cell_layout.SCATTER_ZONE_Y_MM
    assert z0 == z1 == cell_layout.TABLE_TOP_Z_MM / 1000

    park_m = cell_layout.park_positions_m()
    park_mm = cell_layout.park_positions_mm()
    assert set(park_m) == set(park_mm)
    for name, (x_mm, y_mm) in park_mm.items():
        assert park_m[name] == (x_mm / 1000, y_mm / 1000)


def test_the_side_camera_watches_the_scatter_zone_from_past_its_far_edge():
    """The lens sits past the zone's -x edge (facing the arm, so the zone is
    never behind the camera's own body), below 1000 mm so it stays
    tabletop-adjacent, and the scatter centre it aims at is the zone's own
    midpoint on the table top."""
    x, y, z = cell_layout.SIDE_CAMERA_POSITION_MM
    assert x < cell_layout.SCATTER_ZONE_X_MM[0]
    assert cell_layout.TABLE_TOP_Z_MM < z < 1000.0
    centre_x, centre_y, centre_z = cell_layout.SCATTER_CENTRE_MM
    assert centre_x == (cell_layout.SCATTER_ZONE_X_MM[0] + cell_layout.SCATTER_ZONE_X_MM[1]) / 2
    assert centre_y == (cell_layout.SCATTER_ZONE_Y_MM[0] + cell_layout.SCATTER_ZONE_Y_MM[1]) / 2
    assert centre_z == cell_layout.TABLE_TOP_Z_MM
    assert y == centre_y  # dead-on in y: the aim ray tilts only in x/z


def test_place_zone_envelopes_every_pad_exactly():
    assert cell_layout.PLACE_ZONE_X_MM == (570.0, 1530.0)
    assert cell_layout.PLACE_ZONE_Y_MM == (-330.0, 330.0)
    half = cell_layout.PAD_SIDE_MM / 2.0
    for x, y in cell_layout.PAD_CENTRES_MM.values():
        assert cell_layout.PLACE_ZONE_X_MM[0] <= x - half
        assert x + half <= cell_layout.PLACE_ZONE_X_MM[1]
        assert cell_layout.PLACE_ZONE_Y_MM[0] <= y - half
        assert y + half <= cell_layout.PLACE_ZONE_Y_MM[1]
