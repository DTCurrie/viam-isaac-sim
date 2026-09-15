"""Palletizer lifecycle and pure geometry, over fake dependencies: no motion
or gripper client is real here - the one pick-and-place's own orchestration
(start/stop/status, the grasp/place poses, grab failure, cancellation) is
what's under test."""

from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.utils import dict_to_struct

from isaac_module.models.palletizer import (
    CUP_APPROACH_GAP_MM,
    IsaacPalletizer,
    pick_grasp_pose,
    pick_grasp_standoff_pose,
    place_release_pose,
    place_release_standoff_pose,
)
from isaac_module.sort_plan import OUTCOME_FAILED, OUTCOME_PLACED

WORLD_NAME = "world-1"
ARM_NAME = "arm-1"
GRIPPER_NAME = "gripper-1"
MOTION_NAME = "builtin"
BOX_PROP = "infeed-box-1"
PLACE_POSE_MM = {"x": 700.0, "y": 0.0, "z": 354.0}

BOX_GEOMETRY = {
    "name": BOX_PROP,
    "box_dims_mm": [400.0, 300.0, 200.0],
    "pose_in_world_mm": {
        "x": -900.0,
        "y": 0.0,
        "z": 100.0,
        "o_x": 0.0,
        "o_y": 0.0,
        "o_z": 1.0,
        "theta": 0.0,
    },
    "color": None,
    "fixed": False,
}
# the box's known top-face centre: pose z (its own centre) plus half its height
BOX_TOP_FACE_XYZ_MM = (-900.0, 0.0, 100.0 + 200.0 / 2.0)


def _config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _valid_attrs(**overrides: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "world": WORLD_NAME,
        "arm": ARM_NAME,
        "gripper": GRIPPER_NAME,
        "motion": MOTION_NAME,
        "box_prop": BOX_PROP,
        "place_pose_mm": PLACE_POSE_MM,
    }
    attrs.update(overrides)
    return attrs


class FakeWorld:
    def __init__(self, geometries: list[dict[str, Any]] | None = None) -> None:
        self.commands: list[dict[str, Any]] = []
        self._geometries = geometries if geometries is not None else [BOX_GEOMETRY]

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        self.commands.append(dict(command))
        if command["command"] == "prop_geometries":
            return {"geometries": self._geometries}
        return {"ok": True}


class FakeGripper:
    def __init__(self, grab_result: bool = True) -> None:
        self.grab_result = grab_result
        self.opened = False
        self.grabbed = False

    async def open(self) -> None:
        self.opened = True

    async def grab(self) -> bool:
        self.grabbed = True
        return self.grab_result

    async def is_holding_something(self) -> Any:
        raise NotImplementedError


class _RecordingMover:
    def __init__(self) -> None:
        self.calls: list[tuple[Pose, bool]] = []

    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        raise NotImplementedError

    async def move_to(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        self.calls.append((pose, linear))


class _NoopMover:
    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        return None

    async def move_to(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        return None


def _dependencies(world: Any, gripper: Any) -> dict[ResourceName, Any]:
    return {
        ResourceName(name=WORLD_NAME): world,
        ResourceName(name=ARM_NAME): object(),
        ResourceName(name=GRIPPER_NAME): gripper,
        ResourceName(name=MOTION_NAME): object(),
    }


def _make_palletizer(
    world: Any | None = None, gripper: Any | None = None, **attr_overrides: Any
) -> IsaacPalletizer:
    world = world if world is not None else FakeWorld()
    gripper = gripper if gripper is not None else FakeGripper()
    config = _config("palletizer-1", _valid_attrs(**attr_overrides))
    return IsaacPalletizer.new(config, _dependencies(world, gripper))


# ----------------------------------------------------------------------
# pure geometry
# ----------------------------------------------------------------------


def test_pick_grasp_pose_stops_just_short_of_the_box_top_face():
    """A cup driven onto a box makes contact the arm cannot push through, and
    it stalls short of its commanded pose. The gap has to be positive and no
    wider than the vacuum's own payload window, or the cup stops somewhere it
    can no longer see the payload."""
    from isaac_module.handles.vacuum import DEFAULT_MAX_PAYLOAD_GAP_M

    pose = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    x, y, top_face_z = BOX_TOP_FACE_XYZ_MM
    assert (pose.x, pose.y) == (x, y)
    assert pose.z == top_face_z + CUP_APPROACH_GAP_MM
    assert 0.0 < CUP_APPROACH_GAP_MM <= DEFAULT_MAX_PAYLOAD_GAP_M * 1000.0
    assert pose.o_z == -1.0


def test_pick_grasp_standoff_sits_above_the_grasp_pose():
    grasp = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    standoff = pick_grasp_standoff_pose(BOX_TOP_FACE_XYZ_MM)
    assert standoff.z > grasp.z
    assert (standoff.x, standoff.y) == (grasp.x, grasp.y)


def test_place_release_pose_is_at_the_configured_place_pose():
    pose = place_release_pose(PLACE_POSE_MM)
    assert (pose.x, pose.y, pose.z) == (
        PLACE_POSE_MM["x"],
        PLACE_POSE_MM["y"],
        PLACE_POSE_MM["z"],
    )


def test_place_release_standoff_sits_above_the_release_pose():
    release = place_release_pose(PLACE_POSE_MM)
    standoff = place_release_standoff_pose(PLACE_POSE_MM)
    assert standoff.z > release.z
    assert (standoff.x, standoff.y) == (release.x, release.y)


# ----------------------------------------------------------------------
# validate_config
# ----------------------------------------------------------------------


def test_validate_config_returns_every_named_dependency():
    config = _config("palletizer-1", _valid_attrs())
    dependencies, optional = IsaacPalletizer.validate_config(config)
    assert set(dependencies) == {WORLD_NAME, ARM_NAME, GRIPPER_NAME, MOTION_NAME}
    assert optional == []


@pytest.mark.parametrize("missing", ["arm", "gripper", "motion", "box_prop"])
def test_validate_config_requires_arm_gripper_motion_box_prop(missing: str):
    attrs = _valid_attrs()
    del attrs[missing]
    with pytest.raises(ValueError, match=missing):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


def test_validate_config_rejects_a_place_pose_missing_an_axis():
    attrs = _valid_attrs(place_pose_mm={"x": 700.0, "y": 0.0})
    with pytest.raises(ValueError, match="place_pose_mm"):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


def test_validate_config_rejects_a_missing_place_pose():
    attrs = _valid_attrs()
    del attrs["place_pose_mm"]
    with pytest.raises(ValueError, match="place_pose_mm"):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


def test_validate_config_accepts_a_full_place_pose():
    attrs = _valid_attrs(place_pose_mm={"x": 800.0, "y": -100.0, "z": 400.0})
    dependencies, _optional = IsaacPalletizer.validate_config(_config("palletizer-1", attrs))
    assert set(dependencies) == {WORLD_NAME, ARM_NAME, GRIPPER_NAME, MOTION_NAME}


# ----------------------------------------------------------------------
# status before any run
# ----------------------------------------------------------------------


async def test_status_before_any_run_is_idle_with_no_records():
    palletizer = _make_palletizer()
    status = await palletizer.do_command({"command": "status"})
    assert status == {"state": "idle", "records": []}


# ----------------------------------------------------------------------
# a full run against fakes
# ----------------------------------------------------------------------


async def test_full_run_places_the_box_and_reports_one_placed_record():
    gripper = FakeGripper(grab_result=True)
    palletizer = _make_palletizer(gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    started = await palletizer.do_command({"command": "start"})
    assert started == {"ok": True, "state": "running"}
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert len(status["records"]) == 1
    record = status["records"][0]
    assert record["box_prop"] == BOX_PROP
    assert record["outcome"] == OUTCOME_PLACED
    assert gripper.grabbed is True
    assert gripper.opened is True


async def test_run_fails_when_the_box_prop_has_no_known_geometry():
    world = FakeWorld(geometries=[])
    palletizer = _make_palletizer(world=world)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    record = status["records"][0]
    assert record["outcome"] == OUTCOME_FAILED
    assert "no known geometry" in record["reason"]


async def test_second_start_while_running_is_a_noop():
    palletizer = _make_palletizer()
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    started = await palletizer.do_command({"command": "start"})
    assert started == {"ok": True, "state": "running"}

    second = await palletizer.do_command({"command": "start"})
    assert second == {"ok": False, "state": "running"}

    await palletizer.wait_until_done()
    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"


# ----------------------------------------------------------------------
# grab() returning False
# ----------------------------------------------------------------------


async def test_grab_failure_ends_the_run_failed_with_a_named_reason():
    gripper = FakeGripper(grab_result=False)
    palletizer = _make_palletizer(gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert len(status["records"]) == 1
    record = status["records"][0]
    assert record["outcome"] == OUTCOME_FAILED
    assert "no object grasped" in record["reason"]
    assert gripper.opened is False


# ----------------------------------------------------------------------
# stop between motions
# ----------------------------------------------------------------------


async def test_stop_between_motions_halts_before_the_next_move():
    palletizer = _make_palletizer()
    mover = _RecordingMover()

    original_move_to = mover.move_to

    async def move_to_then_stop(pose: Any, world_state: Any, linear: bool = False) -> None:
        await original_move_to(pose, world_state, linear=linear)
        if len(mover.calls) == 1:
            await palletizer.do_command({"command": "stop"})

    mover.move_to = move_to_then_stop  # type: ignore[method-assign]
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "idle"
    # exactly the one motion that ran before "stop" landed - the descent
    # onto the box never happened, and neither did the grab
    assert len(mover.calls) == 1
    record = status["records"][0]
    assert record["outcome"] == OUTCOME_FAILED
    assert "stopped" in record["reason"]
