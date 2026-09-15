import pytest

from isaac_module import cell_layout
from isaac_module.models.conductor import (
    CENSUS_KEEPOUT_CLEARANCE_MM,
    _census_keepouts,
    _scatter_region_mm,
    census_keepout_height_mm,
)
from pickcell.obstacles import KEEPOUT_HEIGHT_MM, KEEPOUT_MARGIN_MM


def _prop(name: str, x: float, y: float, z: float, edge_mm: float = 60.0) -> dict:
    return {
        "name": name,
        "box_dims_mm": [edge_mm, edge_mm, edge_mm],
        "pose_in_world_mm": {"x": x, "y": y, "z": z},
        "color": [1.0, 0.0, 0.0],
        "fixed": False,
    }


def _block_on_table(name: str, x: float, y: float, edge_mm: float = 60.0) -> dict:
    """A block resting on the table top, the way a scattered block sits."""
    return _prop(name, x, y, cell_layout.TABLE_TOP_Z_MM + edge_mm / 2.0, edge_mm)


def _scatter_centre() -> tuple[float, float]:
    x_mm, y_mm, _z = cell_layout.SCATTER_CENTRE_MM
    return x_mm, y_mm


def _pad_centre() -> tuple[float, float]:
    return next(iter(cell_layout.PAD_CENTRES_MM.values()))


# ----------------------------------------------------------------------
# keep-out height
# ----------------------------------------------------------------------


def test_height_is_none_for_an_empty_zone():
    assert census_keepout_height_mm([], _scatter_region_mm()) is None


def test_height_is_none_when_every_prop_sits_outside_the_zone():
    outside = [_block_on_table("block_red_1", 2000.0, 2000.0)]
    assert census_keepout_height_mm(outside, _scatter_region_mm()) is None


def test_height_floors_at_the_pipeline_default_for_ordinary_blocks():
    x_mm, y_mm = _scatter_centre()
    geometries = [_block_on_table("block_red_1", x_mm, y_mm, edge_mm=60.0)]
    assert census_keepout_height_mm(geometries, _scatter_region_mm()) == KEEPOUT_HEIGHT_MM


def test_height_clears_a_block_taller_than_the_default():
    """A stack taller than the 130 mm default has to raise the box, or the
    guard the census plans against passes through the thing it guards."""
    x_mm, y_mm = _scatter_centre()
    tall_mm = KEEPOUT_HEIGHT_MM + 100.0
    geometries = [_block_on_table("block_red_1", x_mm, y_mm, edge_mm=tall_mm)]
    height = census_keepout_height_mm(geometries, _scatter_region_mm())
    assert height == pytest.approx(tall_mm + CENSUS_KEEPOUT_CLEARANCE_MM)
    assert height > KEEPOUT_HEIGHT_MM


def test_height_takes_the_tallest_block_in_the_zone():
    x_mm, y_mm = _scatter_centre()
    tall_mm = KEEPOUT_HEIGHT_MM + 100.0
    geometries = [
        _block_on_table("block_red_1", x_mm, y_mm, edge_mm=60.0),
        _block_on_table("block_blue_1", x_mm + 100.0, y_mm, edge_mm=tall_mm),
    ]
    assert census_keepout_height_mm(geometries, _scatter_region_mm()) == pytest.approx(
        tall_mm + CENSUS_KEEPOUT_CLEARANCE_MM
    )


def test_a_table_topping_out_at_the_surface_adds_nothing_above_it():
    """The scatter zone sits on a table, and the table is itself a prop in
    ``prop_geometries``. Its top is the surface, so it must not inflate the
    box, and it must not be mistaken for an empty zone either."""
    table_dims = cell_layout.TABLE_DIMS_MM
    table = {
        "name": "table_source",
        "box_dims_mm": list(table_dims),
        "pose_in_world_mm": {
            "x": _scatter_centre()[0],
            "y": _scatter_centre()[1],
            "z": cell_layout.TABLE_TOP_Z_MM - table_dims[2] / 2.0,
        },
        "color": None,
        "fixed": True,
    }
    assert census_keepout_height_mm([table], _scatter_region_mm()) == KEEPOUT_HEIGHT_MM


# ----------------------------------------------------------------------
# the boxes the census plans against
# ----------------------------------------------------------------------


def test_no_keepouts_when_both_zones_are_empty():
    assert _census_keepouts([]) == []


def test_a_scattered_block_raises_the_pick_zone_guard_only():
    x_mm, y_mm = _scatter_centre()
    keepouts = _census_keepouts([_block_on_table("block_red_1", x_mm, y_mm)])
    assert [k.label for k in keepouts] == ["pick_area_keepout"]


def test_a_placed_block_raises_the_place_zone_guard_only():
    pad_x, pad_y = _pad_centre()
    placed = _prop("block_red_1", pad_x, pad_y, cell_layout.PAD_TOP_Z_MM + 30.0)
    keepouts = _census_keepouts([placed])
    assert [k.label for k in keepouts] == ["place_area_keepout"]


def test_both_guards_go_up_when_both_zones_hold_blocks():
    scatter_x, scatter_y = _scatter_centre()
    pad_x, pad_y = _pad_centre()
    keepouts = _census_keepouts(
        [
            _block_on_table("block_red_1", scatter_x, scatter_y),
            _prop("block_blue_1", pad_x, pad_y, cell_layout.PAD_TOP_Z_MM + 30.0),
        ]
    )
    assert sorted(k.label for k in keepouts) == ["pick_area_keepout", "place_area_keepout"]


def test_the_pick_guard_spans_the_scatter_zone_and_starts_at_the_table_top():
    scatter_x, scatter_y = _scatter_centre()
    (keepout,) = _census_keepouts([_block_on_table("block_red_1", scatter_x, scatter_y)])
    x_lo, x_hi = cell_layout.SCATTER_ZONE_X_MM
    y_lo, y_hi = cell_layout.SCATTER_ZONE_Y_MM
    assert keepout.box.dims_mm.x == pytest.approx(x_hi - x_lo + 2 * KEEPOUT_MARGIN_MM)
    assert keepout.box.dims_mm.y == pytest.approx(y_hi - y_lo + 2 * KEEPOUT_MARGIN_MM)
    assert keepout.box.dims_mm.z == pytest.approx(KEEPOUT_HEIGHT_MM)
    bottom_z = keepout.center.z - keepout.box.dims_mm.z / 2.0
    assert bottom_z == pytest.approx(cell_layout.TABLE_TOP_Z_MM)


def test_the_guard_tops_out_well_below_the_census_look_height():
    """The census looks from TABLE_TOP + SCAN_HEIGHT. A guard reaching that
    high would box the look pose itself out and every scan would fail."""
    from pickcell.poses import SCAN_HEIGHT_ABOVE_SUPPORT_MM

    scatter_x, scatter_y = _scatter_centre()
    tallest = _block_on_table("block_red_1", scatter_x, scatter_y, cell_layout.MAX_BLOCK_SIZE_MM)
    (keepout,) = _census_keepouts([tallest])
    top_z = keepout.center.z + keepout.box.dims_mm.z / 2.0
    assert top_z < cell_layout.TABLE_TOP_Z_MM + SCAN_HEIGHT_ABOVE_SUPPORT_MM
