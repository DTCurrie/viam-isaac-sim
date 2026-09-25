"""geometry_in_world and read_store_obstacles against a fake world state
store, whose two calls (list_uuids, get_transform) are exactly what a broken
implementation would get wrong: composing the geometry's own centre onto
the transform's pose, skipping anything not parented to world, and skipping
anything with no shape."""

from typing import Any

import pytest
from viam.proto.common import Geometry, Pose, PoseInFrame, RectangularPrism, Transform, Vector3

from isaac_module.store_relay import WORLD_FRAME, geometry_in_world, read_store_obstacles


def _pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=0.0, theta=0.0) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)


def _box_transform(
    name: str,
    pose: Pose,
    *,
    frame: str = WORLD_FRAME,
    centre: Pose | None = None,
    dims: tuple[float, float, float] = (200.0, 150.0, 100.0),
) -> Transform:
    return Transform(
        reference_frame=name,
        pose_in_observer_frame=PoseInFrame(reference_frame=frame, pose=pose),
        physical_object=Geometry(
            center=centre if centre is not None else _pose(),
            box=RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])),
            label=name,
        ),
    )


def test_a_box_centred_at_the_pose_keeps_the_pose():
    pose = _pose(x=100.0, y=200.0, z=350.0, o_z=1.0)
    transform = _box_transform("box-1-v1", pose)

    geometry = geometry_in_world(transform)

    assert geometry is not None
    assert (geometry.center.x, geometry.center.y, geometry.center.z) == (100.0, 200.0, 350.0)
    assert geometry.label == "box-1-v1"
    assert (geometry.box.dims_mm.x, geometry.box.dims_mm.y, geometry.box.dims_mm.z) == (
        200.0,
        150.0,
        100.0,
    )


def test_a_geometry_centre_offset_below_the_pose_lands_lower():
    pose = _pose(x=0.0, y=0.0, z=350.0, o_z=1.0)
    centre = _pose(z=-50.0)
    transform = _box_transform("box-1-v1", pose, centre=centre)

    geometry = geometry_in_world(transform)

    assert geometry is not None
    assert geometry.center.z == pytest.approx(300.0)


def test_a_transform_parented_to_a_pallet_group_is_skipped():
    pose = _pose(x=0.0, y=0.0, z=0.0, o_z=1.0)
    transform = _box_transform("fixture-1", pose, frame="pallet/group")

    assert geometry_in_world(transform) is None


def test_a_transform_with_no_shape_is_skipped():
    transform = Transform(
        reference_frame="empty-1",
        pose_in_observer_frame=PoseInFrame(
            reference_frame=WORLD_FRAME, pose=_pose(x=1.0, y=2.0, z=3.0, o_z=1.0)
        ),
        physical_object=Geometry(),
    )

    assert geometry_in_world(transform) is None


def test_a_pose_pointing_down_composes_the_offset_along_the_axis_it_now_points():
    """o_z=-1 flips the pose upside down, so an offset that was +z in the
    local frame lands BELOW the pose in the world, not above it - the axis a
    naive translation-only composition would get backwards."""
    pose = _pose(x=0.0, y=0.0, z=350.0, o_z=-1.0)
    centre = _pose(z=50.0)
    transform = _box_transform("box-1-v1", pose, centre=centre)

    geometry = geometry_in_world(transform)

    assert geometry is not None
    assert geometry.center.z == pytest.approx(300.0)


def test_a_pose_rotated_90_degrees_composes_the_offset_into_the_horizontal_plane():
    """theta=90 about +z turns a local +x offset into world +y, not world
    +x - the axis a composition that ignored theta would leave unmoved."""
    pose = _pose(x=0.0, y=0.0, z=0.0, o_z=1.0, theta=90.0)
    centre = _pose(x=50.0)
    transform = _box_transform("box-1-v1", pose, centre=centre)

    geometry = geometry_in_world(transform)

    assert geometry is not None
    assert geometry.center.x == pytest.approx(0.0, abs=1e-6)
    assert geometry.center.y == pytest.approx(50.0)


class _FakeStore:
    def __init__(self, transforms: dict[str, Transform]) -> None:
        self._transforms = transforms

    async def list_uuids(self, *, extra: Any = None, timeout: float | None = None) -> list[str]:
        return list(self._transforms.keys())

    async def get_transform(
        self, uuid: Any, *, extra: Any = None, timeout: float | None = None
    ) -> Transform:
        return self._transforms[uuid]


async def test_read_store_obstacles_counts_skips_and_relays():
    world_pose = _pose(x=10.0, y=20.0, z=30.0, o_z=1.0)
    store = _FakeStore(
        {
            "box-1": _box_transform("box-1-v1", world_pose),
            "fixture-1": _box_transform("fixture-1", world_pose, frame="pallet/group"),
        }
    )

    reading = await read_store_obstacles([("sequencer", store)])

    assert reading.relayed == 1
    assert reading.skipped_not_in_world == 1
    assert reading.skipped_no_geometry == 0
    assert [geometry.label for geometry in reading.obstacles] == ["box-1-v1"]


async def test_read_store_obstacles_counts_a_shapeless_transform_as_skipped():
    world_pose = _pose(x=0.0, y=0.0, z=0.0, o_z=1.0)
    empty = Transform(
        reference_frame="empty-1",
        pose_in_observer_frame=PoseInFrame(reference_frame=WORLD_FRAME, pose=world_pose),
        physical_object=Geometry(),
    )
    store = _FakeStore({"box-1": _box_transform("box-1-v1", world_pose), "empty-1": empty})

    reading = await read_store_obstacles([("sequencer", store)])

    assert reading.relayed == 1
    assert reading.skipped_no_geometry == 1
    assert reading.skipped_not_in_world == 0


async def test_read_store_obstacles_sums_counts_across_stores():
    world_pose = _pose(x=0.0, y=0.0, z=0.0, o_z=1.0)
    store_a = _FakeStore({"box-1": _box_transform("box-1-v1", world_pose)})
    store_b = _FakeStore({"box-2": _box_transform("box-2-v1", world_pose)})

    reading = await read_store_obstacles([("sequencer", store_a), ("scene", store_b)])

    assert reading.relayed == 2
    assert [geometry.label for geometry in reading.obstacles] == ["box-1-v1", "box-2-v1"]
    assert reading.counts() == {
        "relayed": 2,
        "skipped_not_in_world": 0,
        "skipped_no_geometry": 0,
    }


class _NilStore:
    """pack-sequencer 0.3.0 with nothing placed: ListUUIDs hands RDK a nil
    slice and the client sees this error, on every pack's first read."""

    async def list_uuids(self, *, extra=None, timeout=None):
        raise RuntimeError(
            "(<Status.UNKNOWN: 2>, 'world state store service returned a nil response', None)"
        )

    async def get_transform(self, uuid, *, extra=None, timeout=None):
        raise AssertionError("nothing to fetch from an empty store")


class _BrokenStore:
    async def list_uuids(self, *, extra=None, timeout=None):
        raise RuntimeError("connection reset")

    async def get_transform(self, uuid, *, extra=None, timeout=None):
        raise AssertionError("unreachable")


async def test_a_store_answering_nil_for_an_empty_list_is_read_as_empty():
    reading = await read_store_obstacles([("pack-sequencer", _NilStore())])
    assert reading.obstacles == []
    assert reading.counts() == {"relayed": 0, "skipped_not_in_world": 0, "skipped_no_geometry": 0}


async def test_any_other_store_error_still_stops_the_read():
    with pytest.raises(RuntimeError, match="connection reset"):
        await read_store_obstacles([("pack-sequencer", _BrokenStore())])
