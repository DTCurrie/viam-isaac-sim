"""WorldHandle seam contract (SCN-16): the scene mutations the world
component's DoCommand verbs drive through PropGeometry / prop_spawn_orientation
/ prop_box_dims / sample_prop_positions / MockWorldHandle / IsaacWorldHandle."""

import math
import threading

import numpy as np
import pytest

from isaac_module.sim_manager import (
    DEFAULT_MIN_SEPARATION_M,
    IsaacWorldHandle,
    MockWorldHandle,
    PropGeometry,
    SimConfig,
    SimManager,
    prop_box_dims,
    sample_prop_positions,
)
from isaac_module.spatial import quat_from_euler_deg

SIM_THREAD_JOIN_TIMEOUT_S = 5


def _cube(name: str, **extra) -> dict:
    return {"type": "cube", "name": name, "size": 0.05, **extra}


def _mock_handle(props: list[dict]) -> MockWorldHandle:
    manager = SimManager()
    return MockWorldHandle(manager, props)


# ----------------------------------------------------------------------
# registry / prop_geometries
# ----------------------------------------------------------------------


def test_registered_props_appear_in_registry_with_spawn_attrs():
    handle = _mock_handle([_cube("block", position=[0.1, 0.2, 0.03])])
    registry = handle.registry()
    assert set(registry) == {"block"}
    entry = registry["block"]
    assert entry["spawn_position"] == (0.1, 0.2, 0.03)
    assert entry["spawn_orientation"] == (1.0, 0.0, 0.0, 0.0)
    assert entry["position"] == entry["spawn_position"]


def test_prop_geometries_one_entry_per_prop_matching_config():
    handle = _mock_handle(
        [
            _cube("a", position=[0.0, 0.0, 0.03], color=[1.0, 0.0, 0.0]),
            _cube("b", position=[0.2, 0.0, 0.03], size=0.1, fixed=True),
        ]
    )
    geoms = {g.name: g for g in handle.prop_geometries()}
    assert set(geoms) == {"a", "b"}

    a = geoms["a"]
    assert isinstance(a, PropGeometry)
    assert a.box_dims_m == (0.05, 0.05, 0.05)
    assert a.position_m == (0.0, 0.0, 0.03)
    assert a.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert a.color == (1.0, 0.0, 0.0)
    assert a.fixed is False

    b = geoms["b"]
    assert b.box_dims_m == (0.1, 0.1, 0.1)
    assert b.fixed is True
    assert b.color is None


def test_prop_with_rpy_orientation_reports_rotated_pose():
    handle = _mock_handle([_cube("tilted", orientation_rpy_deg=[0.0, 0.0, 90.0])])
    expected = quat_from_euler_deg(0.0, 0.0, 90.0)
    (geom,) = handle.prop_geometries()
    assert geom.orientation_wxyz == expected


# ----------------------------------------------------------------------
# set_prop_pose / reset
# ----------------------------------------------------------------------


def test_set_prop_pose_then_reset_restores_configured_pose():
    handle = _mock_handle([_cube("block", position=[0.0, 0.0, 0.03])])
    handle.set_prop_pose("block", (1.0, 1.0, 1.0), (0.0, 1.0, 0.0, 0.0))

    moved = handle.registry()["block"]
    assert moved["position"] == (1.0, 1.0, 1.0)
    assert moved["orientation"] == (0.0, 1.0, 0.0, 0.0)

    handle.reset(soft=True)

    restored = handle.registry()["block"]
    assert restored["position"] == (0.0, 0.0, 0.03)
    assert restored["orientation"] == (1.0, 0.0, 0.0, 0.0)


def test_set_prop_pose_keeps_orientation_when_none_passed():
    handle = _mock_handle([_cube("block", orientation_rpy_deg=[0.0, 0.0, 45.0])])
    expected_orientation = quat_from_euler_deg(0.0, 0.0, 45.0)

    handle.set_prop_pose("block", (0.5, 0.5, 0.03))

    entry = handle.registry()["block"]
    assert entry["position"] == (0.5, 0.5, 0.03)
    assert entry["orientation"] == expected_orientation


def test_set_prop_pose_unknown_name_raises_value_error():
    handle = _mock_handle([_cube("block")])
    with pytest.raises(ValueError):
        handle.set_prop_pose("no-such-prop", (0.0, 0.0, 0.0))


def test_soft_reset_restores_poses_without_full_sim_reset():
    handle = _mock_handle([_cube("block", position=[0.0, 0.0, 0.03])])
    handle.set_prop_pose("block", (2.0, 2.0, 2.0))
    handle.reset(soft=True)
    assert handle.registry()["block"]["position"] == (0.0, 0.0, 0.03)


# ----------------------------------------------------------------------
# spawn_prop
# ----------------------------------------------------------------------


def test_spawn_prop_adds_prop_that_appears_in_geometries_and_survives_reset():
    handle = _mock_handle([_cube("existing")])
    handle.spawn_prop(_cube("new-block", position=[0.4, 0.4, 0.03]))

    names = {g.name for g in handle.prop_geometries()}
    assert names == {"existing", "new_block"}

    handle.set_prop_pose("new_block", (9.0, 9.0, 9.0))
    handle.reset(soft=True)

    restored = {g.name: g for g in handle.prop_geometries()}["new_block"]
    assert restored.position_m == (0.4, 0.4, 0.03)


def test_spawn_prop_duplicate_name_raises_value_error():
    handle = _mock_handle([_cube("dup")])
    with pytest.raises(ValueError):
        handle.spawn_prop(_cube("dup"))


# ----------------------------------------------------------------------
# randomize_props
# ----------------------------------------------------------------------


REGION = ((0.0, 0.0, 0.5), (1.0, 1.0, 0.5))


def test_randomize_props_is_deterministic_for_a_given_seed():
    handle = _mock_handle([_cube("a"), _cube("b")])
    first = handle.randomize_props(["a", "b"], REGION, seed=1)

    handle2 = _mock_handle([_cube("a"), _cube("b")])
    second = handle2.randomize_props(["a", "b"], REGION, seed=1)

    assert first == second


def test_randomize_props_respects_region_and_separation_bounds():
    handle = _mock_handle([_cube("a"), _cube("b"), _cube("c")])
    dims = prop_box_dims(_cube("a"))
    placed = handle.randomize_props(["a", "b", "c"], REGION, seed=1)

    (lo_x, lo_y, z0), (hi_x, hi_y, z1) = REGION
    face_z = (z0 + z1) / 2.0
    half = dims[0] / 2.0

    positions = list(placed.values())
    for x, y, z in positions:
        assert lo_x + half <= x <= hi_x - half
        assert lo_y + half <= y <= hi_y - half
        assert z == pytest.approx(face_z + dims[2] / 2.0)

    for i, (x0, y0, _z0) in enumerate(positions):
        for x1, y1, _z1 in positions[i + 1 :]:
            assert math.hypot(x0 - x1, y0 - y1) >= DEFAULT_MIN_SEPARATION_M


def test_randomize_props_unknown_name_raises_value_error():
    handle = _mock_handle([_cube("a")])
    with pytest.raises(ValueError):
        handle.randomize_props(["a", "ghost"], REGION, seed=1)


# ----------------------------------------------------------------------
# sample_prop_positions (unit-level)
# ----------------------------------------------------------------------


def test_sample_prop_positions_is_deterministic():
    dims = {"a": (0.05, 0.05, 0.05), "b": (0.05, 0.05, 0.05)}
    first = sample_prop_positions(dims, REGION, seed=7)
    second = sample_prop_positions(dims, REGION, seed=7)
    assert first == second


def test_sample_prop_positions_raises_when_region_too_small_for_footprint():
    dims = {"huge": (10.0, 10.0, 0.05)}
    with pytest.raises(ValueError):
        sample_prop_positions(dims, REGION, seed=1)


def test_sample_prop_positions_raises_when_crowded_region_has_no_room():
    tiny_region = ((0.0, 0.0, 0.5), (0.2, 0.2, 0.5))
    dims = {f"p{i}": (0.05, 0.05, 0.05) for i in range(20)}
    with pytest.raises(ValueError):
        sample_prop_positions(dims, tiny_region, seed=1, min_separation_m=0.15)


def test_sample_prop_positions_placements_respect_both_bounds():
    dims = {"a": (0.05, 0.05, 0.05), "b": (0.05, 0.05, 0.05), "c": (0.05, 0.05, 0.05)}
    placed = sample_prop_positions(dims, REGION, seed=3)
    (lo_x, lo_y, z0), (hi_x, hi_y, z1) = REGION
    half = 0.025
    positions = list(placed.values())
    for x, y, _z in positions:
        assert lo_x + half <= x <= hi_x - half
        assert lo_y + half <= y <= hi_y - half
    for i, (x0, y0, _z0) in enumerate(positions):
        for x1, y1, _z1 in positions[i + 1 :]:
            assert math.hypot(x0 - x1, y0 - y1) >= DEFAULT_MIN_SEPARATION_M


# ----------------------------------------------------------------------
# boot-recording: a fresh SimManager boots mock with props (own thread)
# ----------------------------------------------------------------------


def test_fresh_sim_manager_boot_registers_configured_props():
    manager = SimManager()
    sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
    sim_thread.start()
    try:
        manager.ensure_booted(
            SimConfig(mock=True, props=[_cube("boot-block", position=[0.1, 0.1, 0.03])])
        )
        registry = manager.world_handle().registry()
        assert "boot_block" in registry
        assert registry["boot_block"]["spawn_position"] == (0.1, 0.1, 0.03)
    finally:
        manager.request_stop()
        sim_thread.join(timeout=SIM_THREAD_JOIN_TIMEOUT_S)


# ----------------------------------------------------------------------
# IsaacWorldHandle, driven with a fake isaac namespace (cheap coverage only:
# spawn-orientation plumbing, prop_geometries, set_prop_pose, spawn_prop,
# reset(soft=True) - no fake PhysX).
# ----------------------------------------------------------------------


class _FakeXForm:
    """Shared pose store keyed by prim_path; stands in for
    isaacsim's SingleXFormPrim AND the Dynamic/FixedCuboid constructors
    (both end up recording a pose the same way here)."""

    _STORE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __init__(self, prim_path: str, name: str = "", position=None, orientation=None, **_ignored):
        self.prim_path = prim_path
        if position is not None or orientation is not None:
            self.set_world_pose(position=position, orientation=orientation)
        elif prim_path not in self._STORE:
            self._STORE[prim_path] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    def set_world_pose(self, position=None, orientation=None) -> None:
        pos, quat = self._STORE.get(self.prim_path, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        if position is not None:
            pos = np.array([float(v) for v in position])
        if orientation is not None:
            quat = np.array([float(v) for v in orientation])
        self._STORE[self.prim_path] = (pos, quat)

    def get_world_pose(self):
        return self._STORE[self.prim_path]


class _FakeScene:
    def add(self, obj) -> None:
        pass

    def get_object(self, _name):
        return None


class _FakeWorld:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeIsaacNamespace:
    SingleXFormPrim = _FakeXForm
    DynamicCuboid = _FakeXForm
    FixedCuboid = _FakeXForm
    PhysicsMaterial = None
    PhysxSchema = None

    @staticmethod
    def add_reference_to_stage(usd_path: str, prim_path: str) -> None:
        pass


@pytest.fixture
def isaac_handle():
    _FakeXForm._STORE.clear()
    manager = SimManager()
    manager.mock = False
    manager.world = _FakeWorld()
    manager._isaac = _FakeIsaacNamespace()
    manager._booted.set()
    manager._sim_thread_id = threading.get_ident()
    for prop in [_cube("a", position=[0.0, 0.0, 0.03])]:
        manager._spawn_prop(prop)
    return IsaacWorldHandle(manager)


def test_isaac_handle_spawn_orientation_is_recorded_on_the_prim(isaac_handle):
    isaac_handle._sim._spawn_prop(_cube("tilted", orientation_rpy_deg=[0.0, 0.0, 90.0]))
    (geom,) = [g for g in isaac_handle.prop_geometries() if g.name == "tilted"]
    expected = quat_from_euler_deg(0.0, 0.0, 90.0)
    assert geom.orientation_wxyz == pytest.approx(expected)


def test_isaac_handle_prop_geometries_matches_config(isaac_handle):
    (geom,) = isaac_handle.prop_geometries()
    assert geom.name == "a"
    assert geom.box_dims_m == (0.05, 0.05, 0.05)
    assert geom.position_m == pytest.approx((0.0, 0.0, 0.03))


def test_isaac_handle_set_prop_pose_then_reset_restores_spawn_pose(isaac_handle):
    isaac_handle.set_prop_pose("a", (5.0, 5.0, 5.0))
    moved = isaac_handle.prop_geometries()[0]
    assert moved.position_m == pytest.approx((5.0, 5.0, 5.0))

    isaac_handle.reset(soft=True)

    restored = isaac_handle.prop_geometries()[0]
    assert restored.position_m == pytest.approx((0.0, 0.0, 0.03))


def test_isaac_handle_set_prop_pose_unknown_name_raises_value_error(isaac_handle):
    with pytest.raises(ValueError):
        isaac_handle.set_prop_pose("ghost", (0.0, 0.0, 0.0))


def test_isaac_handle_spawn_prop_appears_and_survives_reset(isaac_handle):
    isaac_handle.spawn_prop(_cube("new-block", position=[0.2, 0.2, 0.03]))
    names = {g.name for g in isaac_handle.prop_geometries()}
    assert "new_block" in names
    assert isaac_handle._sim.world.reset_calls == 1


def test_isaac_handle_spawn_prop_duplicate_name_raises_value_error(isaac_handle):
    with pytest.raises(ValueError):
        isaac_handle.spawn_prop(_cube("a"))


def test_sample_prop_positions_restarts_stranded_layouts():
    """GPU run 8: with per-prop-only retries, seed 6 strands the third cube in
    the demo cell's exact region. A stranded layout must be redrawn whole."""
    dims = {
        name: (0.06, 0.06, 0.06) for name in ("pick_cube", "ignore_cube_green", "ignore_cube_blue")
    }
    region = ((0.45, -0.25, 0.0), (0.7, 0.25, 0.0))
    for seed in range(50):
        placed = sample_prop_positions(dims, region, seed=seed, min_separation_m=0.2)
        assert set(placed) == set(dims)
        positions = list(placed.values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                gap = math.hypot(
                    positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]
                )
                assert gap >= 0.2
