import pytest

from isaac_module.cell_layout import BLOCK_COLORS, POOL_BLOCKS_PER_COLOR, pool_block_name
from isaac_module.sim_manager import MockWorldHandle
from isaac_module.visual_props import fit_scale, visual_scale
from test_world_handle import _cube, _mock_handle


def _visual(name: str, **extra) -> dict:
    return {"type": "visual", "name": name, "usd_path": "module://table/table.usd", **extra}


# ----------------------------------------------------------------------
# fit arithmetic
# ----------------------------------------------------------------------


def test_fit_scale_computes_per_axis_quotient():
    assert fit_scale((1.2, 0.8, 0.75), (1.6, 1.0, 0.9)) == pytest.approx(
        (0.75, 0.8, 0.8333333333333334)
    )


def test_fit_scale_raises_on_zero_mesh_dim():
    with pytest.raises(ValueError):
        fit_scale((1.2, 0.8, 0.75), (0.0, 1.0, 0.9))


def test_visual_scale_returns_configured_scale():
    prop = {"scale": [2.0, 3.0, 4.0]}
    assert visual_scale(prop, None, None) == (2.0, 3.0, 4.0)


def test_visual_scale_is_unit_scale_for_fit_true():
    assert visual_scale({"fit": "true"}, None, None) == (1.0, 1.0, 1.0)


def test_visual_scale_is_unit_scale_when_neither_scale_nor_fit_given():
    assert visual_scale({}, None, None) == (1.0, 1.0, 1.0)


def test_visual_scale_is_none_for_fit_collider_with_unknown_mesh():
    prop = {"fit": {"collider": "table"}}
    assert visual_scale(prop, None, None) is None


# ----------------------------------------------------------------------
# mock registration
# ----------------------------------------------------------------------


def test_visual_prop_registers_with_collider_dims_when_collider_listed_first():
    handle = _mock_handle(
        [
            _visual("dressing", fit={"collider": "table"}),
            _cube("table", position=[0.0, 0.0, 0.375], size=1.0, scale=[1.2, 0.8, 0.75]),
        ]
    )
    record = handle._sim._visual_props["dressing"]
    assert record["collider_dims_m"] == pytest.approx((1.2, 0.8, 0.75))
    assert record["scale"] is None


def test_visual_prop_is_absent_from_registry_and_geometries():
    handle = _mock_handle(
        [
            _visual("dressing", fit={"collider": "table"}),
            _cube("table", scale=[1.2, 0.8, 0.75]),
        ]
    )
    assert "dressing" not in handle.registry()
    assert {g.name for g in handle.prop_geometries()} == {"table"}


def test_status_lists_visual_prop_as_a_row():
    handle = _mock_handle(
        [
            _visual("dressing", fit={"collider": "table"}),
            _cube("table", scale=[1.2, 0.8, 0.75]),
        ]
    )
    rows = handle._sim.status()["visual_props"]
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "dressing"
    assert row["scale"] is None
    assert row["bounds_m"] is None
    assert isinstance(row["collider_dims_m"], list)
    assert isinstance(row["position"], list)


def test_unknown_collider_raises():
    with pytest.raises(ValueError):
        _mock_handle([_visual("dressing", fit={"collider": "no-such-prop"})])


def test_usd_collider_raises():
    with pytest.raises(ValueError):
        _mock_handle(
            [
                _visual("dressing", fit={"collider": "crate"}),
                {"type": "usd", "name": "crate", "usd_path": "module://crate.usd"},
            ]
        )


def test_duplicate_visual_name_raises():
    with pytest.raises(ValueError):
        _mock_handle([_visual("dressing"), _visual("dressing")])


def test_duplicate_name_across_registry_and_visual_props_raises():
    with pytest.raises(ValueError):
        _mock_handle([_cube("dressing"), _visual("dressing")])


# ----------------------------------------------------------------------
# verb rejection
# ----------------------------------------------------------------------


def _handle_with_visual() -> MockWorldHandle:
    return _mock_handle([_visual("dressing"), _cube("block", position=[0.0, 0.0, 0.03])])


def test_set_prop_pose_rejects_a_visual_name():
    handle = _handle_with_visual()
    with pytest.raises(ValueError, match="is a visual prop"):
        handle.set_prop_pose("dressing", (0.0, 0.0, 0.0))


def test_randomize_props_rejects_a_visual_name():
    handle = _handle_with_visual()
    with pytest.raises(ValueError, match="is a visual prop"):
        handle.randomize_props(["dressing"], ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), seed=1)


NAMES_BY_COLOR: dict[str, list[str]] = {
    color: [pool_block_name(color, index) for index in range(1, POOL_BLOCKS_PER_COLOR + 1)]
    for color in BLOCK_COLORS
}
POOL_NAMES = [name for names in NAMES_BY_COLOR.values() for name in names]


def _pool_block(name: str) -> dict:
    return {"type": "cube", "name": name, "size": 0.06, "position": [0.0, 0.0, 0.0305]}


def _park_positions() -> dict[str, tuple[float, float]]:
    return {name: (0.0, 0.0) for name in POOL_NAMES}


def _names_by_color_with_visual(visual_name: str) -> dict[str, list[str]]:
    color = next(iter(NAMES_BY_COLOR))
    injected = dict(NAMES_BY_COLOR)
    injected[color] = [visual_name, *NAMES_BY_COLOR[color]]
    return injected


def test_scatter_cell_rejects_a_visual_name_in_the_pool():
    visual_name = "dressing"
    handle = _mock_handle([_pool_block(name) for name in POOL_NAMES] + [_visual(visual_name)])
    names_by_color = _names_by_color_with_visual(visual_name)
    park_positions = _park_positions()
    park_positions[visual_name] = (0.0, 0.0)
    with pytest.raises(ValueError, match="is a visual prop"):
        handle.scatter_cell(
            names_by_color, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), park_positions, seed=1
        )


def test_clear_cell_rejects_a_visual_name_in_the_pool():
    visual_name = "dressing"
    handle = _mock_handle([_pool_block(name) for name in POOL_NAMES] + [_visual(visual_name)])
    names_by_color = _names_by_color_with_visual(visual_name)
    park_positions = _park_positions()
    park_positions[visual_name] = (0.0, 0.0)
    with pytest.raises(ValueError, match="is a visual prop"):
        handle.clear_cell(names_by_color, park_positions)


def test_sized_randomize_leaves_visual_props_unchanged():
    handle = _mock_handle(
        [
            _visual("dressing", fit={"collider": "table"}),
            _cube("table", scale=[1.2, 0.8, 0.75], position=[0.0, 0.0, 0.375]),
        ]
    )
    before = dict(handle._sim._visual_props["dressing"])
    handle.randomize_props(
        ["table"],
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        seed=1,
        size_range_m={"table": (0.5, 0.5)},
    )
    assert handle._sim._visual_props["dressing"] == before


def test_spawn_prop_registers_a_visual_on_the_mock_handle():
    handle = _mock_handle([_cube("table", scale=[1.2, 0.8, 0.75])])
    handle.spawn_prop(_visual("dressing", fit={"collider": "table"}))
    assert "dressing" in handle._sim._visual_props
    assert "dressing" not in handle.registry()


def test_record_marks_a_fitted_collider_hidden_and_a_free_visual_not():
    handle = _mock_handle(
        [
            _cube("table", position=[0.0, 0.0, 0.375], size=1.0, scale=[1.2, 0.8, 0.75]),
            _visual("dressing", fit={"collider": "table"}),
            _visual("ornament", fit="true"),
        ]
    )
    assert handle._sim._visual_props["dressing"]["collider_hidden"] is True
    assert handle._sim._visual_props["ornament"]["collider_hidden"] is False


def test_visual_with_empty_usd_path_is_skipped_without_error():
    handle = _mock_handle(
        [
            _cube("table", position=[0.0, 0.0, 0.375], size=1.0, scale=[1.2, 0.8, 0.75]),
            {**_visual("dressing", fit={"collider": "table"}), "usd_path": ""},
        ]
    )
    assert "dressing" not in handle._sim._visual_props
    assert handle._sim.status()["visual_props"] == []
