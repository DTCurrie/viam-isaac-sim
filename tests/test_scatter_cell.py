"""scatter_cell / clear_cell (phase-2 pooled scatter): the count draw, the
seeded one-stream contract, placement inside the seam's scatter region, and
the park round-trip - mock backend only (isaac shares the same draw/park
math by construction, only the prim-write layer differs)."""

import math

from isaac_module import cell_layout
from isaac_module.sim_manager import (
    POOL_SCATTER_MIN_SEPARATION_M,
    MockWorldHandle,
    SimManager,
)

POOL_NAMES = [
    cell_layout.pool_block_name(color, index)
    for color in cell_layout.BLOCK_COLORS
    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
]


def _pool_block(name: str) -> dict:
    x, y = cell_layout.park_positions_m()[name]
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


# ----------------------------------------------------------------------
# count draw
# ----------------------------------------------------------------------


def test_scatter_cell_default_counts_are_in_range_across_many_seeds():
    for seed in range(50):
        handle = _mock_cell()
        result = handle.scatter_cell(seed)
        assert set(result.counts) == set(cell_layout.BLOCK_COLORS)
        for count in result.counts.values():
            assert 1 <= count <= cell_layout.POOL_BLOCKS_PER_COLOR
        assert len(result.parked) + len(result.positions_m) == 18
        assert result.seed == seed


def test_scatter_cell_counts_vary_across_seeds():
    counts_by_seed = {
        seed: tuple(_mock_cell().scatter_cell(seed).counts.values()) for seed in range(10)
    }
    assert len(set(counts_by_seed.values())) > 1


def test_scatter_cell_same_seed_is_deterministic():
    result_a = _mock_cell().scatter_cell(7)
    result_b = _mock_cell().scatter_cell(7)
    assert result_a.counts == result_b.counts
    assert result_a.positions_m == result_b.positions_m
    assert result_a.sizes_m == result_b.sizes_m
    assert sorted(result_a.parked) == sorted(result_b.parked)


def test_scatter_cell_counts_override_full_pool_scatters_everything():
    counts = {color: cell_layout.POOL_BLOCKS_PER_COLOR for color in cell_layout.BLOCK_COLORS}
    result = _mock_cell().scatter_cell(1, counts=counts)
    assert result.counts == counts
    assert len(result.positions_m) == 18
    assert result.parked == []


def test_scatter_cell_counts_override_empty_color_parks_all_three():
    counts = {"red": 0}
    result = _mock_cell().scatter_cell(1, counts=counts)
    assert result.counts["red"] == 0
    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1):
        name = cell_layout.pool_block_name("red", index)
        assert name in result.parked
        assert name not in result.positions_m


def test_scatter_cell_counts_override_unknown_color_raises_value_error():
    import pytest

    with pytest.raises(ValueError):
        _mock_cell().scatter_cell(1, counts={"magenta": 1})


def test_scatter_cell_counts_override_out_of_range_raises_value_error():
    import pytest

    with pytest.raises(ValueError):
        _mock_cell().scatter_cell(1, counts={"red": 4})
    with pytest.raises(ValueError):
        _mock_cell().scatter_cell(1, counts={"red": -1})


# ----------------------------------------------------------------------
# placement
# ----------------------------------------------------------------------


def test_scattered_blocks_land_inside_the_scatter_region():
    (x0, y0, _z0), (x1, y1, _z1) = cell_layout.scatter_region_m()
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    for seed in range(20):
        result = _mock_cell().scatter_cell(seed)
        for x, y, _z in result.positions_m.values():
            assert lo_x <= x <= hi_x
            assert lo_y <= y <= hi_y


def test_scattered_blocks_respect_minimum_separation():
    for seed in range(20):
        result = _mock_cell().scatter_cell(seed)
        positions = list(result.positions_m.values())
        for i, (x0, y0, _z0) in enumerate(positions):
            for x1, y1, _z1 in positions[i + 1 :]:
                assert math.hypot(x0 - x1, y0 - y1) >= POOL_SCATTER_MIN_SEPARATION_M


def test_scatter_cell_missing_pool_prim_raises_value_error():
    import pytest

    manager = SimManager()
    manager.mock = True
    handle = MockWorldHandle(manager, [_pool_block(name) for name in POOL_NAMES[:-1]])
    with pytest.raises(ValueError):
        handle.scatter_cell(1)


# ----------------------------------------------------------------------
# park round-trip
# ----------------------------------------------------------------------


def test_scatter_then_clear_then_scatter_reproduces_identical_positions():
    handle = _mock_cell()
    first = handle.scatter_cell(3)
    handle.clear_cell()
    second = handle.scatter_cell(3)
    assert first.positions_m == second.positions_m
    assert first.counts == second.counts


def test_after_clear_every_pool_block_sits_at_its_park_xy():
    handle = _mock_cell()
    handle.scatter_cell(3)
    result = handle.clear_cell()
    assert sorted(result.parked) == sorted(POOL_NAMES)
    park_xy = cell_layout.park_positions_m()
    registry = handle.registry()
    for name in POOL_NAMES:
        x, y, _z = registry[name]["position"]
        expected_x, expected_y = park_xy[name]
        assert x == expected_x
        assert y == expected_y


def test_clear_cell_parks_all_eighteen_pool_blocks():
    handle = _mock_cell()
    handle.scatter_cell(5)
    result = handle.clear_cell()
    assert sorted(result.parked) == sorted(POOL_NAMES)
