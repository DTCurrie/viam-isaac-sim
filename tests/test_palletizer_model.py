"""Palletizer lifecycle over fake dependencies: no motion, gripper or
sequencer client is real here. The sequencer drives which box goes where
(a fake ``SequencerClient`` of the same shape stands in for it), and this
service's own job - drive, execute, report the measured pose back - is
what's under test."""

from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.utils import dict_to_struct

from isaac_module.models.palletizer import (
    CUP_APPROACH_GAP_MM,
    PICK_KEEPOUT_HEADROOM_MM,
    PLACE_RELEASE_CLEARANCE_MM,
    IsaacPalletizer,
    approach_and_descend,
    box_has_settled,
    box_keepout,
    held_box_transform,
    is_execution_failure,
    move_linear_or_free,
    pick_grasp_pose,
    pick_grasp_standoff_pose,
    place_release_pose,
    support_keepout,
    touched_down,
)
from isaac_module.sort_plan import OUTCOME_FAILED, OUTCOME_PLACED
from pickcell.poses import PRE_GRASP_STANDOFF_MM

WORLD_NAME = "world-1"
ARM_NAME = "arm-1"
GRIPPER_NAME = "gripper-1"
MOTION_NAME = "builtin"
SEQUENCER_NAME = "sequencer-1"
BOX_PROPS = [f"infeed_box_{i}" for i in range(1, 9)]

BOX_TOP_FACE_XYZ_MM = (-900.0, 0.0, 100.0 + 200.0 / 2.0)


def _box_geometry(name: str, x: float = -900.0) -> dict[str, Any]:
    return {
        "name": name,
        "box_dims_mm": [400.0, 300.0, 200.0],
        "pose_in_world_mm": {
            "x": x,
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


def _place_pose(seq: int) -> Pose:
    return Pose(x=700.0, y=seq * 10.0, z=354.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)


def _place_start_pose(seq: int) -> Pose:
    end = _place_pose(seq)
    return Pose(x=end.x, y=end.y, z=end.z + 150.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)


def _config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _valid_attrs(**overrides: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "world": WORLD_NAME,
        "arm": ARM_NAME,
        "gripper": GRIPPER_NAME,
        "motion": MOTION_NAME,
        "sequencer": SEQUENCER_NAME,
        "box_props": list(BOX_PROPS),
    }
    attrs.update(overrides)
    return attrs


class FakeWorld:
    """A ``WorldApi`` fake. When ``settled_offset_mm`` (dx, dy, dtheta) is
    given, a box the gripper has released reads at the sequencer's own target
    for its seq, offset by that amount, standing in for physics settling
    somewhere other than exactly where the plan aimed.

    Keyed off releases rather than off how many times the service reads. The
    service reads repeatedly while it waits for a box to stop moving, and a
    fake that alternated per read would never let one settle."""

    def __init__(
        self,
        geometries: list[dict[str, Any]] | None = None,
        settled_offset_mm: tuple[float, float, float] | None = None,
    ) -> None:
        self.commands: list[dict[str, Any]] = []
        self._geometries = geometries if geometries is not None else [_box_geometry(BOX_PROPS[0])]
        self._settled_offset_mm = settled_offset_mm
        self.released = 0

    def mark_released(self) -> None:
        """One more box has been let go, so its next reading is its settled
        one. Called by FakeGripper.open()."""
        self.released += 1

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        self.commands.append(dict(command))
        if command["command"] != "prop_geometries":
            return {"ok": True}
        if self._settled_offset_mm is None or self.released == 0:
            return {"geometries": self._geometries}
        box_index = self.released - 1
        seq = box_index + 1
        target = _place_pose(seq)
        dx, dy, dtheta = self._settled_offset_mm
        settled_pose_mm = {
            "x": target.x + dx,
            "y": target.y + dy,
            "z": target.z,
            "o_x": target.o_x,
            "o_y": target.o_y,
            "o_z": target.o_z,
            "theta": target.theta + dtheta,
        }
        settled_geometries = [dict(g) for g in self._geometries]
        settled_geometries[box_index] = {
            **settled_geometries[box_index],
            "pose_in_world_mm": settled_pose_mm,
        }
        return {"geometries": settled_geometries}


class FakeGripper:
    def __init__(self, grab_result: bool = True, world: "FakeWorld | None" = None) -> None:
        self.grab_result = grab_result
        self.opened = False
        self.open_count = 0
        self.grabbed = False
        self._holding = False
        self._world = world

    async def open(self) -> None:
        self.opened = True
        self.open_count += 1
        # letting go of nothing releases nothing: the run's opening open() must
        # not read as a box landing on its slot
        if self._holding and self._world is not None:
            self._world.mark_released()
        self._holding = False

    async def grab(self) -> bool:
        self.grabbed = True
        self._holding = self.grab_result
        return self.grab_result

    async def is_holding_something(self) -> Any:
        raise NotImplementedError


class _NoopMover:
    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        return None

    async def move_to(
        self, pose: Any, world_state: Any, linear: bool = False, level: bool = False
    ) -> None:
        return None


class _RecordingMover:
    def __init__(self) -> None:
        self.calls: list[tuple[Pose, bool]] = []

    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        raise NotImplementedError

    async def move_to(
        self, pose: Any, world_state: Any, linear: bool = False, level: bool = False
    ) -> None:
        self.calls.append((pose, linear))


class _CapturingMover:
    """Like ``_NoopMover``, but keeps the ``world_state`` and the ``level``
    flag handed to every move - the pieces ``_NoopMover`` and
    ``_RecordingMover`` both drop, which is exactly what a test of
    ``obstacle_source`` or of the carry needs to see."""

    def __init__(self) -> None:
        self.world_states: list[Any] = []
        self.levels: list[bool] = []

    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        raise NotImplementedError

    async def move_to(
        self, pose: Any, world_state: Any, linear: bool = False, level: bool = False
    ) -> None:
        self.world_states.append(world_state)
        self.levels.append(level)


class FakeNextBox:
    """Stands in for ``sequencer_client.NextBox`` - same field names, built
    directly rather than through the sibling slice's still-unimplemented
    ``parse_next_box``."""

    def __init__(self, seq: int, *, is_complete: bool = False) -> None:
        self.seq = None if is_complete else seq
        self.is_complete = is_complete
        self.place_start_in_world = None if is_complete else _place_start_pose(seq)
        self.place_end_in_world = None if is_complete else _place_pose(seq)


class FakeSequencer:
    """A fake of ``SequencerClient``'s shape: a cursor that advances on a
    successful ``report_placement`` and stays put on a failed one, exactly
    the behaviour the real ``viam:pack-sequencer:sequencer`` documents."""

    def __init__(self, box_count: int) -> None:
        self.box_count = box_count
        self.cursor = 1
        self.report_calls: list[tuple[int, bool, str]] = []
        self.skip_calls: list[tuple[int, str]] = []
        self.transforms: list[tuple[int, Pose]] = []

    async def next_box(self) -> FakeNextBox:
        if self.cursor > self.box_count:
            return FakeNextBox(0, is_complete=True)
        return FakeNextBox(self.cursor)

    async def report_placement(self, seq: int, *, success: bool, error: str = "") -> None:
        self.report_calls.append((seq, success, error))
        if success:
            self.cursor += 1

    async def skip_box(self, seq: int, *, reason: str = "") -> None:
        self.skip_calls.append((seq, reason))
        self.cursor += 1

    async def set_box_transform(
        self, seq: int, pose: Pose, *, parent: str = "", color: Any = None
    ) -> str:
        self.transforms.append((seq, pose))
        return f"uuid-{seq}"


def _dependencies(world: Any, gripper: Any, motion: Any | None = None) -> dict[ResourceName, Any]:
    return {
        ResourceName(name=WORLD_NAME): world,
        ResourceName(name=ARM_NAME): object(),
        ResourceName(name=GRIPPER_NAME): gripper,
        ResourceName(name=MOTION_NAME): motion if motion is not None else object(),
        ResourceName(name=SEQUENCER_NAME): object(),
    }


def _make_palletizer(
    world: Any | None = None,
    gripper: Any | None = None,
    motion: Any | None = None,
    **attr_overrides: Any,
) -> IsaacPalletizer:
    world = world if world is not None else FakeWorld()
    gripper = gripper if gripper is not None else FakeGripper(world=world)
    config = _config("palletizer-1", _valid_attrs(**attr_overrides))
    return IsaacPalletizer.new(config, _dependencies(world, gripper, motion))


def _all_box_geometries() -> list[dict[str, Any]]:
    return [_box_geometry(prop, x=-900.0 + i * 10.0) for i, prop in enumerate(BOX_PROPS)]


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


# ----------------------------------------------------------------------
# validate_config
# ----------------------------------------------------------------------


def test_validate_config_returns_every_named_dependency():
    config = _config("palletizer-1", _valid_attrs())
    dependencies, optional = IsaacPalletizer.validate_config(config)
    assert set(dependencies) == {WORLD_NAME, ARM_NAME, GRIPPER_NAME, MOTION_NAME, SEQUENCER_NAME}
    assert optional == []


@pytest.mark.parametrize("missing", ["arm", "gripper", "motion", "sequencer"])
def test_validate_config_requires_arm_gripper_motion_sequencer(missing: str):
    attrs = _valid_attrs()
    del attrs[missing]
    with pytest.raises(ValueError, match=missing):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


def test_validate_config_rejects_a_missing_box_props():
    attrs = _valid_attrs()
    del attrs["box_props"]
    with pytest.raises(ValueError, match="box_props"):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


def test_validate_config_rejects_an_empty_box_props():
    attrs = _valid_attrs(box_props=[])
    with pytest.raises(ValueError, match="box_props"):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


def test_validate_config_rejects_an_unknown_obstacle_source():
    attrs = _valid_attrs(obstacle_source="made_up_source")
    with pytest.raises(ValueError, match="obstacle_source"):
        IsaacPalletizer.validate_config(_config("palletizer-1", attrs))


@pytest.mark.parametrize("obstacle_source", ["world_state_store", "prop_geometries"])
def test_validate_config_accepts_both_obstacle_sources(obstacle_source: str):
    attrs = _valid_attrs(obstacle_source=obstacle_source)
    dependencies, _optional = IsaacPalletizer.validate_config(_config("palletizer-1", attrs))
    assert set(dependencies) == {WORLD_NAME, ARM_NAME, GRIPPER_NAME, MOTION_NAME, SEQUENCER_NAME}


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


async def _run_full_pack(
    obstacle_source: str = "world_state_store",
    world: Any | None = None,
    mover: Any | None = None,
) -> tuple[IsaacPalletizer, FakeSequencer, FakeGripper]:
    world = world if world is not None else FakeWorld(geometries=_all_box_geometries())
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper, obstacle_source=obstacle_source)
    mover = mover if mover is not None else _NoopMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    sequencer = FakeSequencer(box_count=len(BOX_PROPS))
    palletizer._sequencer = sequencer  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()
    return palletizer, sequencer, gripper


async def test_full_pack_places_all_eight_boxes_in_sequencer_order():
    palletizer, _sequencer, gripper = await _run_full_pack()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    records = status["records"]
    assert len(records) == len(BOX_PROPS)
    assert [record["seq"] for record in records] == list(range(1, len(BOX_PROPS) + 1))
    assert [record["box_prop"] for record in records] == BOX_PROPS
    assert all(record["outcome"] == OUTCOME_PLACED for record in records)
    assert gripper.grabbed is True
    assert gripper.opened is True


async def test_full_pack_reports_each_outcome_to_the_sequencer():
    _palletizer, sequencer, _gripper = await _run_full_pack()

    assert [seq for seq, _success, _error in sequencer.report_calls] == list(
        range(1, len(BOX_PROPS) + 1)
    )
    assert all(success for _seq, success, _error in sequencer.report_calls)


MEASURED_POSE_OFFSET_MM = (3.0, -2.0, 5.0)  # (dx, dy, dtheta): where physics settled it


async def test_full_pack_publishes_the_settled_pose_not_the_sequencers_target():
    """The whole point of set_box_transform: the viewer must show where
    physics put the box, not where the plan said it would go. A code path
    that republished the target instead of the world's reading would leave
    every one of these assertions true only by accident - the offset below
    is what tells the two apart."""
    world = FakeWorld(geometries=_all_box_geometries(), settled_offset_mm=MEASURED_POSE_OFFSET_MM)
    palletizer, sequencer, _gripper = await _run_full_pack(world=world)

    assert [seq for seq, _pose in sequencer.transforms] == list(range(1, len(BOX_PROPS) + 1))
    dx, dy, dtheta = MEASURED_POSE_OFFSET_MM

    status = await palletizer.do_command({"command": "status"})
    for (seq, published_pose), record in zip(sequencer.transforms, status["records"], strict=True):
        target = record["target_pose_mm"]
        measured = record["measured_pose_mm"]
        assert measured is not None
        assert measured["x"] == pytest.approx(target["x"] + dx)
        assert measured["y"] == pytest.approx(target["y"] + dy)
        assert measured["theta"] == pytest.approx(target["theta"] + dtheta)
        assert measured != target
        # set_box_transform got exactly the settled reading, not the target
        assert (published_pose.x, published_pose.y, published_pose.theta) == (
            measured["x"],
            measured["y"],
            measured["theta"],
        )
        assert (published_pose.x, published_pose.y, published_pose.theta) != (
            target["x"],
            target["y"],
            target["theta"],
        )
        assert seq == record["seq"]


@pytest.mark.parametrize("obstacle_source", ["world_state_store", "prop_geometries"])
async def test_full_pack_works_under_both_obstacle_sources(obstacle_source: str):
    _palletizer, sequencer, _gripper = await _run_full_pack(obstacle_source=obstacle_source)
    assert sequencer.cursor > len(BOX_PROPS)


def _labels(world_state) -> set[str]:
    return {geometry.label for frame in world_state.obstacles for geometry in frame.geometries}


async def test_obstacle_source_selects_what_the_motion_service_sees():
    """world_state_store hands the motion service nothing but this service's
    own keep-outs (the frame system and the store own the cell's obstacles
    instead). prop_geometries hands it the support slab plus a geometry per
    other prop, and the keep-outs on top. A test that cannot tell the two
    apart protects nothing, since the GPU run's own checklist item is this
    exact A/B."""
    store_mover = _CapturingMover()
    await _run_full_pack(obstacle_source="world_state_store", mover=store_mover)
    assert store_mover.world_states
    # the only transform is the box on the cup, and only while it is there
    assert all(
        {transform.reference_frame for transform in ws.transforms} <= set(BOX_PROPS)
        for ws in store_mover.world_states
    )
    assert all(
        _labels(ws) <= {"pick_box_keepout", "place_support_keepout"}
        for ws in store_mover.world_states
    )
    assert "support" not in {label for ws in store_mover.world_states for label in _labels(ws)}

    props_mover = _CapturingMover()
    await _run_full_pack(obstacle_source="prop_geometries", mover=props_mover)
    assert props_mover.world_states
    for world_state in props_mover.world_states:
        labels = _labels(world_state)
        assert "support" in labels
        # the held box is excluded, the other seven are obstacles, and any
        # keep-out this leg closes is on top of them
        assert len(labels - {"pick_box_keepout", "place_support_keepout"}) == 1 + (
            len(BOX_PROPS) - 1
        )


async def test_restage_moves_every_later_box_to_the_first_boxs_infeed_pose():
    """The pick station only fits one box_props entry at a time. box_props[0]
    is left exactly where it starts (the infeed pose IS its pose), and every
    later box is re-posed there through set_prop_pose before the
    prop_geometries read that drives its own pick - never a configured pick
    pose, only the sim's own reading of where the first box sat."""
    world = FakeWorld(geometries=_all_box_geometries())
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=len(BOX_PROPS))  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    restage_commands = [
        command for command in world.commands if command["command"] == "set_prop_pose"
    ]
    restaged_names = [command["name"] for command in restage_commands]
    # box_props[0] is never re-posed - the first read of the world already
    # reads its own resting pose as the infeed pose
    assert BOX_PROPS[0] not in restaged_names
    # every later box is re-posed exactly once, in pick order
    assert restaged_names == BOX_PROPS[1:]

    infeed_pose_mm = _all_box_geometries()[0]["pose_in_world_mm"]
    expected_position = [infeed_pose_mm["x"], infeed_pose_mm["y"], infeed_pose_mm["z"]]
    assert all(command["position"] == expected_position for command in restage_commands)

    # the command order with consecutive reads collapsed: box_props[0] reads,
    # then every later box RE-POSES and only then reads. A re-pose landing
    # after the read that drives the grasp would move here. How many reads a
    # turn takes is the settle poll's business, not this test's.
    order = [command["command"] for command in world.commands]
    collapsed = [name for i, name in enumerate(order) if i == 0 or name != order[i - 1]]
    expected_command_order = ["prop_geometries"]
    for _box_prop in BOX_PROPS[1:]:
        expected_command_order += ["set_prop_pose", "prop_geometries"]
    assert collapsed == expected_command_order


async def test_place_motions_use_the_sequencers_poses_with_no_added_standoff():
    """place_start_in_world already carries the sequencer's own approach
    standoff, clamped so a descending box cannot plow through a neighbour.
    Stacking PRE_GRASP_STANDOFF_MM on top - or any other half-box
    correction - would move the descent/retreat poses off these exact
    values."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _RecordingMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    place_start = _place_start_pose(1)
    place_end = _place_pose(1)

    def _poses_equal(a: Pose, b: Pose) -> bool:
        return (a.x, a.y, a.z, a.o_x, a.o_y, a.o_z, a.theta) == (
            b.x,
            b.y,
            b.z,
            b.o_x,
            b.o_y,
            b.o_z,
            b.theta,
        )

    release = place_release_pose(place_end)
    place_start_calls = [
        (pose, linear) for pose, linear in mover.calls if _poses_equal(pose, place_start)
    ]
    release_calls = [(pose, linear) for pose, linear in mover.calls if _poses_equal(pose, release)]
    # the approach (non-linear) and the retreat (linear) both land exactly at
    # place_start_in_world
    assert (place_start, False) in place_start_calls
    assert (place_start, True) in place_start_calls
    # the descent lands at the release pose, linear: place_end_in_world's x, y
    # and orientation, raised by the grasp gap and the release clearance
    assert (release, True) in release_calls
    assert not any(_poses_equal(pose, place_end) for pose, _linear in mover.calls)


async def test_the_box_rides_the_gripper_between_the_grab_and_the_release():
    """The planner knows the arm and nothing hanging from it unless told. Six
    legs per box: approach, descent, lift, cross, place descent, retreat. The
    box is on the cup for the middle three and nowhere else."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _CapturingMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    holding = [bool(ws.transforms) for ws in mover.world_states]
    assert holding == [False, False, True, True, True, False]
    # and every leg with the box on the cup is a level one
    assert mover.levels == holding
    (held,) = mover.world_states[2].transforms
    assert held.reference_frame == BOX_PROPS[0]
    assert held.pose_in_observer_frame.reference_frame == GRIPPER_NAME
    dims = held.physical_object.box.dims_mm
    assert (dims.x, dims.y, dims.z) == (400.0, 300.0, 200.0)


def test_held_box_transform_hangs_the_box_below_the_cup_by_the_grasp_gap():
    held = held_box_transform("infeed_box_1", (150.0, 200.0, 100.0), "gripper-1")
    assert held.reference_frame == "infeed_box_1"
    assert held.pose_in_observer_frame.reference_frame == "gripper-1"
    centre = held.pose_in_observer_frame.pose
    # the gripper's z is the tool axis, pointing down at a grasp, so below
    # the cup is +z: the grasp gap, then half the box's height
    assert (centre.x, centre.y) == (0.0, 0.0)
    assert centre.z == pytest.approx(CUP_APPROACH_GAP_MM + 50.0)
    dims = held.physical_object.box.dims_mm
    assert (dims.x, dims.y, dims.z) == (150.0, 200.0, 100.0)
    assert held.physical_object.label == "infeed_box_1"


async def test_a_stop_landing_before_a_boxs_restage_leaves_that_box_where_it_was():
    """The restage is a teleport, not a motion, so the between-motions stop
    check comes after it. A run stopped once its first box was placed used to
    put the second box on the station anyway, and every later reset of the
    first box landed on top of it."""

    class _StopOnSecond(FakeSequencer):
        def __init__(self, palletizer: IsaacPalletizer) -> None:
            super().__init__(box_count=2)
            self._palletizer = palletizer

        async def next_box(self) -> FakeNextBox:
            if self.cursor == 2:
                self._palletizer._cancel_requested = True
            return await super().next_box()

    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0]), _box_geometry(BOX_PROPS[1])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = _StopOnSecond(palletizer)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    restaged = [
        command["name"] for command in world.commands if command["command"] == "set_prop_pose"
    ]
    assert restaged == []
    status = await palletizer.do_command({"command": "status"})
    assert [record["outcome"] for record in status["records"]] == [OUTCOME_PLACED, OUTCOME_FAILED]
    assert "stopped" in status["records"][1]["reason"]


class _FakeMotion:
    """The one motion-service call the palletizer makes directly: where the
    gripper's frame is after a stall."""

    def __init__(self, cup_pose: Pose) -> None:
        self._cup_pose = cup_pose

    async def get_pose(self, component_name: str, destination_frame: str) -> Any:
        from viam.proto.common import PoseInFrame

        assert destination_frame == "world"
        return PoseInFrame(reference_frame="world", pose=self._cup_pose)


class _StallsOnThePlaceDescent(_CapturingMover):
    """Every move succeeds except the linear one made with a box on the cup
    after the cross, which is the place descent, and that one stalls the way
    the arm reports a box meeting the deck."""

    async def move_to(
        self, pose: Any, world_state: Any, linear: bool = False, level: bool = False
    ) -> None:
        await super().move_to(pose, world_state, linear=linear, level=level)
        if level and linear and len(self.world_states) == 5:
            raise RuntimeError(
                "error moving component arm-1 to inputs [...]: rpc error: code = Aborted desc = "
                "arm arm-1 stalled at waypoint 12/12 (1 consecutive, stuck joints: j3: at -98.7 "
                "want -100.1)"
            )


async def _run_one_box_with_a_stalled_descent(cup_pose: Pose) -> tuple[Any, FakeGripper]:
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper, motion=_FakeMotion(cup_pose))
    palletizer._build_mover = lambda: _StallsOnThePlaceDescent()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()
    return await palletizer.do_command({"command": "status"}), gripper


async def test_a_place_descent_that_stalls_at_the_slot_releases_the_box():
    """Two descents onto a slot flush with the pallet's corner stalled on
    their last waypoint with the cup 7 mm above the release height. The box
    was down. Letting go is what the descent was for."""
    release = place_release_pose(_place_pose(1))
    cup_seven_mm_low = Pose(
        x=release.x + 4.0,
        y=release.y - 8.0,
        z=release.z - 7.0,
        o_x=0.0,
        o_y=0.0,
        o_z=-1.0,
        theta=0.0,
    )

    status, gripper = await _run_one_box_with_a_stalled_descent(cup_seven_mm_low)

    assert status["state"] == "complete"
    assert [record["outcome"] for record in status["records"]] == [OUTCOME_PLACED]
    assert gripper.opened is True


async def test_a_place_descent_that_stalls_far_from_the_slot_is_a_failure():
    release = place_release_pose(_place_pose(1))
    cup_elsewhere = Pose(
        x=release.x, y=release.y, z=release.z + 200.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0
    )

    status, _gripper = await _run_one_box_with_a_stalled_descent(cup_elsewhere)

    assert status["state"] == "failed"
    assert "stalled at waypoint 12/12" in status["reason"]


def test_touched_down_is_a_distance_to_the_release_pose():
    release = Pose(x=50.0, y=400.0, z=360.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    near = Pose(x=46.2, y=392.4, z=353.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    far = Pose(x=50.0, y=400.0, z=390.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    assert touched_down(near, release) is True
    assert touched_down(far, release) is False


def test_place_release_pose_lifts_place_end_by_the_grasp_gap_and_the_clearance():
    """place_end_in_world is the cup at the box's top face with the box on
    its slot. The cup holds the box CUP_APPROACH_GAP_MM above that face, so a
    descent to place_end drives the box that far into the deck. Measured on
    2026-09-22: the arm stalled with the box on the deck and the cup 5 mm
    above place_end."""
    place_end = Pose(x=59.3, y=374.5, z=350.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=-109.5)
    release = place_release_pose(place_end)
    assert (release.x, release.y) == (59.3, 374.5)
    assert release.z == pytest.approx(350.0 + CUP_APPROACH_GAP_MM + PLACE_RELEASE_CLEARANCE_MM)
    assert (release.o_x, release.o_y, release.o_z, release.theta) == (0.0, 0.0, 1.0, -109.5)


async def test_a_run_begins_by_letting_go_of_whatever_the_cup_holds():
    """A run that failed between grab and release leaves its box welded to
    the cup, and a held prop ignores the restage that starts the next run."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=0)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    assert gripper.opened is True
    assert gripper.grabbed is False


# ----------------------------------------------------------------------
# a box the arm cannot pick
# ----------------------------------------------------------------------


async def test_run_fails_the_box_with_no_known_geometry_and_skips_it():
    world = FakeWorld(geometries=[])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    sequencer = FakeSequencer(box_count=1)
    palletizer._sequencer = sequencer  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    # retried once (the sequencer's cursor stayed put on the first failure),
    # then skip_box'ed on the second
    assert len(status["records"]) == 2
    assert all(record["outcome"] == OUTCOME_FAILED for record in status["records"])
    assert "no known geometry" in status["records"][0]["reason"]
    assert sequencer.skip_calls == [(1, status["records"][0]["reason"])]


async def test_second_start_while_running_is_a_noop():
    palletizer = _make_palletizer()
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]
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


async def test_grab_failure_reports_failed_retries_once_then_skips():
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=False, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    sequencer = FakeSequencer(box_count=1)
    palletizer._sequencer = sequencer  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert len(status["records"]) == 2
    for record in status["records"]:
        assert record["outcome"] == OUTCOME_FAILED
        assert "no object grasped" in record["reason"]
    # the run's opening open() and nothing more: a failed grab releases
    # nothing, since there is nothing to release
    assert gripper.open_count == 1
    assert [seq for seq, success, _error in sequencer.report_calls] == [1, 1]
    assert all(success is False for _seq, success, _error in sequencer.report_calls)
    assert [seq for seq, _reason in sequencer.skip_calls] == [1]
    assert "no object grasped" in sequencer.skip_calls[0][1]


# ----------------------------------------------------------------------
# stop between motions
# ----------------------------------------------------------------------


async def test_stop_between_motions_halts_before_the_next_move():
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    palletizer = _make_palletizer(world=world)
    sequencer = FakeSequencer(box_count=1)
    palletizer._sequencer = sequencer  # type: ignore[assignment]
    mover = _RecordingMover()

    original_move_to = mover.move_to

    async def move_to_then_stop(
        pose: Any, world_state: Any, linear: bool = False, level: bool = False
    ) -> None:
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
    # a stop is not a placement failure - the sequencer never hears about it
    assert sequencer.report_calls == []
    assert sequencer.skip_calls == []


class _RecordingApproach:
    """Records every (pose, linear) it is asked for, and refuses the first
    `linear_refusals` straight-line moves the way the planner refuses one it
    cannot produce from the configuration the approach landed in."""

    def __init__(self, linear_refusals: int = 0, stop_before: int | None = None):
        self.calls: list[tuple[Pose, bool]] = []
        self._linear_refusals = linear_refusals
        self._stop_before = stop_before

    async def __call__(self, pose: Pose, linear: bool) -> bool:
        if self._stop_before is not None and len(self.calls) == self._stop_before:
            return False
        self.calls.append((pose, linear))
        if linear and self._linear_refusals > 0:
            self._linear_refusals -= 1
            raise RuntimeError("motion planner failed to find path")
        return True


async def test_move_linear_or_free_stays_straight_when_the_straight_line_plans():
    grasp = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    approach = _RecordingApproach()

    assert await move_linear_or_free(approach, grasp, "grasp descent") is True
    assert approach.calls == [(grasp, True)]


async def test_move_linear_or_free_drops_the_constraint_when_the_line_is_refused():
    """From the configuration the pick standoff kept landing in, no
    straight-line descent exists at any distance or tolerance, while an
    unconstrained one plans first time. Falling back is the recovery."""
    grasp = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    approach = _RecordingApproach(linear_refusals=1)

    assert await move_linear_or_free(approach, grasp, "grasp descent") is True
    assert approach.calls == [(grasp, True), (grasp, False)]


async def test_move_linear_or_free_reraises_a_stall_rather_than_retrying_free():
    """A straight line that planned and then stalled on the arm is the arm
    blocked by contact. The 2026-09-22 place descent stalled on waypoint 13 of
    13 with the box on the deck, and the free retry planned a move of under a
    degree from the blocked pose, stalled the same way, and reported that
    instead."""
    place = Pose(x=59.3, y=374.5, z=350.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    calls: list[tuple[Pose, bool]] = []

    async def stalls(pose: Pose, linear: bool) -> bool:
        calls.append((pose, linear))
        raise RuntimeError(
            "error moving component arm-1 to inputs [...]: rpc error: code = Aborted desc = "
            "arm arm-1 stalled at waypoint 13/13 (1 consecutive, stuck joints: j2: at -115.1 "
            "want -115.7, j3: at -79.2 want -78.4)"
        )

    with pytest.raises(RuntimeError, match="stalled at waypoint 13/13"):
        await move_linear_or_free(stalls, place, "place descent")
    assert calls == [(place, True)]


def test_is_execution_failure_tells_a_stall_from_a_planner_refusal():
    assert is_execution_failure(RuntimeError("arm arm-1 stalled at waypoint 2/2 (...)")) is True
    assert (
        is_execution_failure(RuntimeError("arm arm-1 did not reach final waypoint within 10.0s"))
        is True
    )
    assert is_execution_failure(RuntimeError("motion planner failed to find path")) is False
    assert is_execution_failure(RuntimeError("zero IK solutions produced")) is False


async def test_move_linear_or_free_raises_when_the_free_move_is_refused_too():
    grasp = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)

    async def refuse_everything(pose: Pose, linear: bool) -> bool:
        raise RuntimeError("zero IK solutions produced")

    with pytest.raises(RuntimeError, match="zero IK solutions produced"):
        await move_linear_or_free(refuse_everything, grasp, "grasp descent")


async def test_approach_and_descend_keeps_the_two_legs_on_their_own_obstacle_sets():
    """The approach leg and the descent leg see different worlds: the box is
    an obstacle for one and the payload for the other. Sharing a mover between
    them is what let the arm swing through the box on its way to the
    standoff."""
    standoff = pick_grasp_standoff_pose(BOX_TOP_FACE_XYZ_MM)
    grasp = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    approach = _RecordingApproach()
    descend = _RecordingApproach()

    assert await approach_and_descend(approach, descend, standoff, grasp) is True
    assert approach.calls == [(standoff, False)]
    assert descend.calls == [(grasp, True)]


async def test_approach_and_descend_reports_a_stop_rather_than_descending():
    standoff = pick_grasp_standoff_pose(BOX_TOP_FACE_XYZ_MM)
    grasp = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    approach = _RecordingApproach(stop_before=0)
    descend = _RecordingApproach()

    assert await approach_and_descend(approach, descend, standoff, grasp) is False
    assert approach.calls == []
    assert descend.calls == []


_PICK_GEOMETRIES = [
    {
        "name": "infeed_box_1",
        "pose_in_world_mm": {"x": 400.0, "y": -300.0, "z": 270.0},
        "box_dims_mm": [200.0, 150.0, 100.0],
    },
    {
        "name": "frame_pallet",
        "pose_in_world_mm": {"x": 200.0, "y": 500.0, "z": 200.0},
        "box_dims_mm": [500.0, 350.0, 100.0],
    },
]


def test_box_keepout_covers_the_box_and_stops_below_the_standoff():
    """The keep-out has to close the airspace the arm swung through, and stay
    clear of the pose it descends from, or the approach becomes unplannable."""
    zone = box_keepout(_PICK_GEOMETRIES, "infeed_box_1")
    assert zone is not None
    ceiling_mm = zone.center.z + zone.box.dims_mm.z / 2.0
    floor_mm = zone.center.z - zone.box.dims_mm.z / 2.0
    box_top_mm = 270.0 + 100.0 / 2.0
    standoff_mm = box_top_mm + CUP_APPROACH_GAP_MM + PRE_GRASP_STANDOFF_MM

    assert floor_mm == pytest.approx(270.0 - 100.0 / 2.0)
    assert ceiling_mm == pytest.approx(box_top_mm + PICK_KEEPOUT_HEADROOM_MM)
    assert ceiling_mm < standoff_mm
    # grown sideways, so a plan that grazes the box's own face is out too
    assert zone.box.dims_mm.x > 200.0
    assert zone.box.dims_mm.y > 150.0


def test_box_keepout_is_none_when_the_box_is_not_in_the_scene():
    assert box_keepout(_PICK_GEOMETRIES, "infeed_box_7") is None


def test_support_keepout_rises_from_the_supports_top_to_the_ceiling_it_is_given():
    zone = support_keepout(_PICK_GEOMETRIES, "frame_pallet", 400.0)
    assert zone is not None
    pallet_top_mm = 200.0 + 100.0 / 2.0
    assert zone.center.z - zone.box.dims_mm.z / 2.0 == pytest.approx(pallet_top_mm)
    assert zone.center.z + zone.box.dims_mm.z / 2.0 == pytest.approx(400.0)


def test_support_keepout_is_none_when_the_ceiling_is_at_or_below_the_support():
    """A pallet whose approach pose sits below its own deck would otherwise
    produce an inverted box, which is an obstacle nothing can plan around."""
    assert support_keepout(_PICK_GEOMETRIES, "frame_pallet", 250.0) is None


def test_support_keepout_is_none_when_the_support_is_not_in_the_scene():
    assert support_keepout(_PICK_GEOMETRIES, "frame_tray_dock", 400.0) is None


def test_box_has_settled_accepts_two_readings_in_the_same_place():
    pose = {"x": 400.0, "y": -300.0, "z": 270.0}
    assert box_has_settled(pose, dict(pose)) is True


def test_box_has_settled_rejects_a_box_still_falling():
    """The 2026-09-16 shape: the box was read a moment after a 100 mm release
    and the cup was sent 5 mm under a top face the box had already left."""
    assert (
        box_has_settled(
            {"x": 400.0, "y": -300.0, "z": 370.0}, {"x": 400.0, "y": -300.0, "z": 310.0}
        )
        is False
    )


def test_box_has_settled_rejects_a_box_that_registered_no_pose():
    assert box_has_settled(None, {"x": 400.0, "y": -300.0, "z": 270.0}) is False
    assert box_has_settled({"x": 400.0, "y": -300.0, "z": 270.0}, None) is False
