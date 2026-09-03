"""The place-zone keep-out (phase 5, GPU round): with ``place_region_mm`` and
``placed_tallest_mm`` set, the carry plans against a ``place_area_keepout``
box over the pads and the pre-place hover is floored so the held block's
bottom clears it. With the fields at their defaults the pipeline's world
states and poses are unchanged."""

import asyncio

import pytest
from viam.proto.common import Pose

from pickcell import pipeline as pipeline_module
from pickcell.obstacles import KEEPOUT_MARGIN_MM, pick_area_keepout
from pickcell.pipeline import PLACED_KEEPOUT_CLEARANCE_MM, PickPipeline

BLOCK_SIZE_MM = 60.0
BLOCK_TOP_Z_MM = 30.0
PAD_TOP_Z_MM = 10.0
PLACE_REGION_MM = ((500.0, -400.0, PAD_TOP_Z_MM), (900.0, -300.0, PAD_TOP_Z_MM))
# tall enough that the hover floor exceeds the standoff-derived hover, so the
# test fails if the floor stops binding
PLACED_TALLEST_MM = 120.0


def _prop_geometry(name, dims, x_mm=600.0, y_mm=100.0, z_mm=40.0, fixed=False):
    return {
        "name": name,
        "fixed": fixed,
        "box_dims_mm": list(dims),
        "pose_in_world_mm": {
            "x": x_mm,
            "y": y_mm,
            "z": z_mm,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": 1.0,
            "theta": 0.0,
        },
    }


class _Fakes:
    def __init__(self, **pipeline_overrides):
        self.moves: list[tuple[Pose, bool, object]] = []
        self.opens = 0
        outer = self

        class World:
            async def do_command(self, command):
                return {
                    "geometries": [
                        _prop_geometry("pick_cube", [60.0, 60.0, 60.0]),
                        _prop_geometry(
                            "place_pad",
                            [200.0, 200.0, 10.0],
                            x_mm=700.0,
                            y_mm=-350.0,
                            z_mm=5.0,
                            fixed=True,
                        ),
                    ]
                }

        class Detector:
            async def block_pose_world(self):
                return Pose(
                    x=600.0, y=100.0, z=BLOCK_TOP_Z_MM, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0
                )

        class Mover:
            async def look_from(self, pose, world_state, linear=False):
                return None

            async def move_to(self, pose, world_state, linear=False):
                outer.moves.append((pose, linear, world_state))

        class Gripper:
            async def open(self):
                outer.opens += 1

            async def grab(self):
                return True

            async def is_holding_something(self):
                return True

        self.pipeline = PickPipeline(
            detector=Detector(),
            mover=Mover(),
            gripper=Gripper(),
            block_name="pick_cube",
            block_size_mm=BLOCK_SIZE_MM,
            gripper_name="pick-grip",
            world=World(),
            target_prop_name="pick_cube",
            place_prop_name="place_pad",
            **pipeline_overrides,
        )


def _labels_in(world_state) -> list[str]:
    return [geometry.label for frame in world_state.obstacles for geometry in frame.geometries]


def _run(fakes: _Fakes) -> None:
    asyncio.run(fakes.pipeline.run())


def test_place_keepout_boxes_the_pad_zone_and_floors_the_hover(monkeypatch):
    monkeypatch.setattr(pipeline_module, "PLACE_SETTLE_S", 0.0)
    fakes = _Fakes(place_region_mm=PLACE_REGION_MM, placed_tallest_mm=PLACED_TALLEST_MM)
    _run(fakes)

    keepout_states = [
        (pose, state)
        for pose, _linear, state in fakes.moves
        if "place_area_keepout" in _labels_in(state)
    ]
    assert keepout_states, "no move planned against the place-zone keep-out"
    _pose, state = keepout_states[0]
    box = next(
        geometry
        for frame in state.obstacles
        for geometry in frame.geometries
        if geometry.label == "place_area_keepout"
    )
    height_mm = PLACED_TALLEST_MM + PLACED_KEEPOUT_CLEARANCE_MM
    # box top sits exactly at pad top + tallest + clearance, no higher
    assert box.center.z == pytest.approx(PAD_TOP_Z_MM + height_mm / 2.0)
    assert box.box.dims_mm.z == pytest.approx(height_mm)
    assert box.box.dims_mm.x == pytest.approx(400.0 + 2 * KEEPOUT_MARGIN_MM)
    assert box.box.dims_mm.y == pytest.approx(100.0 + 2 * KEEPOUT_MARGIN_MM)

    # the hover floor: held block bottom (TCP - centre_below_tcp - size/2)
    # clears the box top exactly
    grasp_pose = fakes.moves[2][0]
    centre_below_tcp_mm = grasp_pose.z - BLOCK_TOP_Z_MM
    expected_floor = PAD_TOP_Z_MM + height_mm + centre_below_tcp_mm + BLOCK_SIZE_MM / 2.0
    assert fakes.pipeline.place_clear_tcp_z_mm == pytest.approx(expected_floor)
    pre_place_pose = keepout_states[0][0]
    assert pre_place_pose.z == pytest.approx(expected_floor)


def test_place_keepout_absent_by_default(monkeypatch):
    monkeypatch.setattr(pipeline_module, "PLACE_SETTLE_S", 0.0)
    fakes = _Fakes()
    _run(fakes)

    assert fakes.pipeline.place_clear_tcp_z_mm is None
    for _pose, _linear, state in fakes.moves:
        assert "place_area_keepout" not in _labels_in(state)
    # the legacy shape: waypoint, pre-grasp, grasp, lift, raise, carry, place, retreat
    assert len(fakes.moves) == 8


def test_keepout_label_override_names_the_box():
    keepout = pick_area_keepout(
        ((0.0, 0.0, 0.0), (100.0, 100.0, 0.0)), 50.0, label="place_area_keepout"
    )
    assert keepout.label == "place_area_keepout"
    assert pick_area_keepout(((0.0, 0.0, 0.0), (100.0, 100.0, 0.0))).label == "pick_area_keepout"
