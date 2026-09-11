import math

import pytest

from isaac_module import cell_layout
from isaac_module.sim_manager import (
    POOL_SCATTER_MIN_SEPARATION_M,
    MockWorldHandle,
    SimManager,
)

# scatter_cell/clear_cell no longer know about the demo cell: this test
# builds the same payload the conductor sends (names_by_color, region,
# park_positions_m), sourced from cell_layout the way the conductor's own
# copy is, and passes it explicitly on every call.
NAMES_BY_COLOR: dict[str, list[str]] = {
    color: [
        cell_layout.pool_block_name(color, index)
        for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
    ]
    for color in cell_layout.BLOCK_COLORS
}
POOL_NAMES = [name for names in NAMES_BY_COLOR.values() for name in names]
REGION = cell_layout.scatter_region_m()
PARK_XY = cell_layout.park_positions_m()


def _pool_block(name: str) -> dict:
    x, y = PARK_XY[name]
    return {
        "type": "cube",
        "name": name,
        "size": 0.06,
        "position": [x, y, 0.03 + 0.0005],
    }


def _mock_cell(extra_props: list[dict] | None = None) -> MockWorldHandle:
    manager = SimManager()
    manager.mock = True
    props = [_pool_block(name) for name in POOL_NAMES] + (extra_props or [])
    return MockWorldHandle(manager, props)


def _scatter(handle: MockWorldHandle, seed: int, **kwargs):
    return handle.scatter_cell(NAMES_BY_COLOR, REGION, PARK_XY, seed, **kwargs)


def _clear(handle: MockWorldHandle):
    return handle.clear_cell(NAMES_BY_COLOR, PARK_XY)


# ----------------------------------------------------------------------
# count draw
# ----------------------------------------------------------------------


def test_scatter_cell_default_counts_are_in_range_across_many_seeds():
    for seed in range(50):
        handle = _mock_cell()
        result = _scatter(handle, seed)
        assert set(result.counts) == set(cell_layout.BLOCK_COLORS)
        for count in result.counts.values():
            assert 1 <= count <= cell_layout.POOL_BLOCKS_PER_COLOR
        assert len(result.parked) + len(result.positions_m) == 18
        assert result.seed == seed


def test_scatter_cell_counts_vary_across_seeds():
    counts_by_seed = {
        seed: tuple(_scatter(_mock_cell(), seed).counts.values()) for seed in range(10)
    }
    assert len(set(counts_by_seed.values())) > 1


def test_scatter_cell_same_seed_is_deterministic():
    result_a = _scatter(_mock_cell(), 7)
    result_b = _scatter(_mock_cell(), 7)
    assert result_a.counts == result_b.counts
    assert result_a.positions_m == result_b.positions_m
    assert result_a.sizes_m == result_b.sizes_m
    assert sorted(result_a.parked) == sorted(result_b.parked)


def test_scatter_cell_counts_override_full_pool_scatters_everything():
    counts = {color: cell_layout.POOL_BLOCKS_PER_COLOR for color in cell_layout.BLOCK_COLORS}
    result = _scatter(_mock_cell(), 1, counts=counts)
    assert result.counts == counts
    assert len(result.positions_m) == 18
    assert result.parked == []


def test_scatter_cell_counts_override_empty_color_parks_all_three():
    counts = {"red": 0}
    result = _scatter(_mock_cell(), 1, counts=counts)
    assert result.counts["red"] == 0
    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1):
        name = cell_layout.pool_block_name("red", index)
        assert name in result.parked
        assert name not in result.positions_m


def test_scatter_cell_counts_override_unknown_color_raises_value_error():
    with pytest.raises(ValueError):
        _scatter(_mock_cell(), 1, counts={"magenta": 1})


def test_scatter_cell_counts_override_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        _scatter(_mock_cell(), 1, counts={"red": 4})
    with pytest.raises(ValueError):
        _scatter(_mock_cell(), 1, counts={"red": -1})


# ----------------------------------------------------------------------
# placement
# ----------------------------------------------------------------------


def test_scattered_blocks_land_inside_the_scatter_region():
    (x0, y0, _z0), (x1, y1, _z1) = REGION
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    for seed in range(20):
        result = _scatter(_mock_cell(), seed)
        for x, y, _z in result.positions_m.values():
            assert lo_x <= x <= hi_x
            assert lo_y <= y <= hi_y


def test_scattered_blocks_respect_minimum_separation():
    for seed in range(20):
        result = _scatter(_mock_cell(), seed)
        positions = list(result.positions_m.values())
        for i, (x0, y0, _z0) in enumerate(positions):
            for x1, y1, _z1 in positions[i + 1 :]:
                assert math.hypot(x0 - x1, y0 - y1) >= POOL_SCATTER_MIN_SEPARATION_M


def test_scatter_cell_missing_pool_prim_raises_value_error():
    manager = SimManager()
    manager.mock = True
    handle = MockWorldHandle(manager, [_pool_block(name) for name in POOL_NAMES[:-1]])
    with pytest.raises(ValueError, match="missing pool blocks"):
        _scatter(handle, 1)


def test_scatter_cell_missing_park_position_raises_value_error_naming_it():
    missing_name = POOL_NAMES[-1]
    incomplete_park_xy = {name: xy for name, xy in PARK_XY.items() if name != missing_name}
    handle = _mock_cell()
    with pytest.raises(ValueError, match=missing_name):
        handle.scatter_cell(NAMES_BY_COLOR, REGION, incomplete_park_xy, 1)


def test_clear_cell_missing_park_position_raises_value_error_naming_it():
    missing_name = POOL_NAMES[0]
    incomplete_park_xy = {name: xy for name, xy in PARK_XY.items() if name != missing_name}
    handle = _mock_cell()
    with pytest.raises(ValueError, match=missing_name):
        handle.clear_cell(NAMES_BY_COLOR, incomplete_park_xy)


# ----------------------------------------------------------------------
# park round-trip
# ----------------------------------------------------------------------


def test_scatter_then_clear_then_scatter_reproduces_identical_positions():
    handle = _mock_cell()
    first = _scatter(handle, 3)
    _clear(handle)
    second = _scatter(handle, 3)
    assert first.positions_m == second.positions_m
    assert first.counts == second.counts


def test_after_clear_every_pool_block_sits_at_its_park_xy():
    handle = _mock_cell()
    _scatter(handle, 3)
    result = _clear(handle)
    assert sorted(result.parked) == sorted(POOL_NAMES)
    registry = handle.registry()
    for name in POOL_NAMES:
        x, y, _z = registry[name]["position"]
        expected_x, expected_y = PARK_XY[name]
        assert x == expected_x
        assert y == expected_y


def test_clear_cell_parks_all_eighteen_pool_blocks():
    handle = _mock_cell()
    _scatter(handle, 5)
    result = _clear(handle)
    assert sorted(result.parked) == sorted(POOL_NAMES)
