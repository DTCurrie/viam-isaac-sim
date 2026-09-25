"""Palletizer lifecycle over fake dependencies: no motion, gripper or
sequencer client is real here. The sequencer drives which box goes where
(a fake ``SequencerClient`` of the same shape stands in for it), and this
service's own job - drive, execute, report the measured pose back - is
what's under test."""

import asyncio
from typing import Any

import pytest
from grpclib import Status
from grpclib.exceptions import GRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    ResourceName,
    Transform,
    Vector3,
)
from viam.utils import dict_to_struct

from isaac_module.models import palletizer
from isaac_module.models.palletizer import (
    CUP_APPROACH_GAP_MM,
    FIRST_FREE_PLAN_TIMEOUT_S,
    PICK_KEEPOUT_HEADROOM_MM,
    PLACE_APPROACH_CLEARANCE_MM,
    PLACE_RELEASE_CLEARANCE_MM,
    RESTAGE_MAX_ATTEMPTS,
    IsaacPalletizer,
    approach_and_descend,
    box_keepout,
    carry_height_pose,
    held_box_transform,
    is_execution_failure,
    is_planner_timeout,
    move_linear_or_free,
    pick_grasp_pose,
    pick_grasp_standoff_pose,
    place_approach_pose,
    place_release_pose,
    support_keepout,
    touched_down,
)
from isaac_module.sort_plan import OUTCOME_FAILED, OUTCOME_PLACED
from isaac_module.stack_stability import DEFAULT_SETTLE_POLICY, SettlePolicy
from pickcell.poses import PRE_GRASP_STANDOFF_MM

WORLD_NAME = "world-1"
ARM_NAME = "arm-1"
GRIPPER_NAME = "gripper-1"
MOTION_NAME = "builtin"
SEQUENCER_NAME = "sequencer-1"
BOX_PROPS = [f"infeed_box_{i}" for i in range(1, 9)]

BOX_TOP_FACE_XYZ_MM = (-900.0, 0.0, 100.0 + 200.0 / 2.0)

_FAST_SETTLE_POLICY = SettlePolicy(tolerance_mm=1.0, window_s=0.02, poll_s=0.005, timeout_s=0.1)


_FAST_RESTAGE_TIMEOUT_S = 0.05


@pytest.fixture(autouse=True)
def _fast_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test polls the settle policy at least once per box, so a full
    pack of eight would otherwise cost the production policy's own window
    and timeout eight times over. A restage that never lands waits out
    RESTAGE_TIMEOUT_S too, so that is cut down here alongside it."""
    monkeypatch.setattr(palletizer, "SETTLE_POLICY", _FAST_SETTLE_POLICY)
    monkeypatch.setattr(palletizer, "RESTAGE_TIMEOUT_S", _FAST_RESTAGE_TIMEOUT_S)


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
        never_settle: bool = False,
        never_settle_while_carried: bool = False,
        restage_lands_after: int = 0,
        never_restages: bool = False,
    ) -> None:
        self.commands: list[dict[str, Any]] = []
        source = geometries if geometries is not None else [_box_geometry(BOX_PROPS[0])]
        self._geometries = [dict(g) for g in source]
        self._settled_offset_mm = settled_offset_mm
        # a box that keeps moving on every read once released, so the settle
        # policy never sees a trailing window agree and times out
        self._never_settle = never_settle
        self._moving_reads_since_release = 0
        self.released = 0
        # a box that keeps moving on every read from the moment it is
        # grabbed and until it is released, standing in for a carry the box
        # never stops swinging through
        self._never_settle_while_carried = never_settle_while_carried
        self._moving_reads_since_grab = 0
        self.grabbed_count = 0
        # a restage that leaves the box parked for this many set_prop_pose
        # calls before actually moving it, standing in for the read-after-
        # teleport race the world's own set_prop_pose has
        self._restage_lands_after = restage_lands_after
        self._never_restages = never_restages
        self._restage_calls: dict[str, int] = {}

    def mark_grabbed(self) -> None:
        """One more box is now on the cup, so a ``never_settle_while_carried``
        world starts jittering its pose on every later read until it is
        released. Called by ``FakeGripper.grab()`` on a successful grab."""
        self.grabbed_count += 1

    def mark_released(self) -> None:
        """One more box has been let go, so its next reading is its settled
        one. Called by FakeGripper.open()."""
        self.released += 1

    def _teleport(self, name: str, position: list[float]) -> None:
        """A real move: lands ``name`` at ``position`` in ``self._geometries``,
        so a restage's read-back can see it, unless ``never_restages`` or
        ``restage_lands_after`` says this call does not count yet."""
        if self._never_restages:
            return
        calls_so_far = self._restage_calls.get(name, 0)
        self._restage_calls[name] = calls_so_far + 1
        if calls_so_far < self._restage_lands_after:
            return
        for geometry in self._geometries:
            if geometry["name"] != name:
                continue
            x, y, z = position
            pose = dict(geometry["pose_in_world_mm"])
            pose["x"], pose["y"], pose["z"] = x, y, z
            geometry["pose_in_world_mm"] = pose
            return

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        self.commands.append(dict(command))
        if command["command"] == "set_prop_pose":
            self._teleport(command["name"], command["position"])
            return {"ok": True}
        if command["command"] != "prop_geometries":
            return {"ok": True}
        if self._never_settle_while_carried and self.grabbed_count > self.released:
            box_index = self.grabbed_count - 1
            self._moving_reads_since_grab += 1
            moving_geometries = [dict(g) for g in self._geometries]
            pose = dict(moving_geometries[box_index]["pose_in_world_mm"])
            pose["x"] = pose["x"] + self._moving_reads_since_grab * 5.0
            moving_geometries[box_index] = {
                **moving_geometries[box_index],
                "pose_in_world_mm": pose,
            }
            return {"geometries": moving_geometries}
        if self._never_settle and self.released > 0:
            box_index = self.released - 1
            self._moving_reads_since_release += 1
            moving_geometries = [dict(g) for g in self._geometries]
            pose = dict(moving_geometries[box_index]["pose_in_world_mm"])
            pose["x"] = pose["x"] + self._moving_reads_since_release * 5.0
            moving_geometries[box_index] = {
                **moving_geometries[box_index],
                "pose_in_world_mm": pose,
            }
            return {"geometries": moving_geometries}
        if self._settled_offset_mm is None or self.released == 0:
            return {"geometries": self._geometries}
        box_index = self.released - 1
        seq = box_index + 1
        target = _place_pose(seq)
        dx, dy, dtheta = self._settled_offset_mm
        # the target is the box's top face, in the sequencer's convention;
        # prop_geometries reports the body's centre, half a height below it
        _dim_x, _dim_y, dim_z = self._geometries[box_index]["box_dims_mm"]
        settled_pose_mm = {
            "x": target.x + dx,
            "y": target.y + dy,
            "z": target.z - dim_z / 2.0,
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
        if self.grab_result and self._world is not None:
            self._world.mark_grabbed()
        return self.grab_result

    async def is_holding_something(self) -> Any:
        raise NotImplementedError


class _NoopMover:
    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        return None

    async def move_to(
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
    ) -> None:
        return None


class _RecordingMover:
    def __init__(self) -> None:
        self.calls: list[tuple[Pose, bool]] = []

    async def look_from(self, pose: Any, world_state: Any, linear: bool = False) -> None:
        raise NotImplementedError

    async def move_to(
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
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
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
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


class _EmptyStoreResource:
    """Stands in for the dependency resource behind ``sequencer``. Most tests
    swap ``palletizer._sequencer`` for a ``FakeSequencer`` and never call into
    this directly, but ``world_state_stores`` defaults to the sequencer name
    and reads it as a world state store, so it needs an empty ``list_uuids``
    even where the sequencer verbs are never exercised."""

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def list_uuids(self, *, extra: Any = None, timeout: float | None = None) -> list[Any]:
        return []

    async def get_transform(
        self, uuid: Any, *, extra: Any = None, timeout: float | None = None
    ) -> Any:
        raise NotImplementedError


def _dependencies(world: Any, gripper: Any, motion: Any | None = None) -> dict[ResourceName, Any]:
    return {
        ResourceName(name=WORLD_NAME): world,
        ResourceName(name=ARM_NAME): object(),
        ResourceName(name=GRIPPER_NAME): gripper,
        ResourceName(name=MOTION_NAME): motion if motion is not None else object(),
        ResourceName(name=SEQUENCER_NAME): _EmptyStoreResource(),
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
    it stalls short of its commanded pose. The gap has to be positive and
    shorter than the cups' own raycast, or the ray from the cups never
    reaches past it to find the payload."""
    from isaac_module.surface_gripper import DEFAULT_MAX_GRIP_DISTANCE_MM

    pose = pick_grasp_pose(BOX_TOP_FACE_XYZ_MM)
    x, y, top_face_z = BOX_TOP_FACE_XYZ_MM
    assert (pose.x, pose.y) == (x, y)
    assert pose.z == top_face_z + CUP_APPROACH_GAP_MM
    assert 0.0 < CUP_APPROACH_GAP_MM < DEFAULT_MAX_GRIP_DISTANCE_MM
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
        # both poses are the box's top face, the sequencer's convention: the
        # world reports the centre, and a centre published as-is sits half a
        # box low in the viewer and reads as a 50 mm error against the target
        assert measured["z"] == pytest.approx(target["z"])
        # set_box_transform gets the box's CENTRE pose - the store centres its
        # own geometry on the pose it is given - while measured_pose_mm in the
        # record stays the top face
        _dim_x, _dim_y, dim_z = _all_box_geometries()[seq - 1]["box_dims_mm"]
        assert published_pose.x == pytest.approx(measured["x"])
        assert published_pose.y == pytest.approx(measured["y"])
        assert published_pose.theta == pytest.approx(measured["theta"])
        assert published_pose.z == pytest.approx(measured["z"] - dim_z / 2.0)
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


_KEEPOUT_LABELS = {
    "pick_box_keepout",
    "place_support_keepout",
    "place_support_keepout_empty",
    "placed_box_keepout",
}


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
    assert all(_labels(ws) <= _KEEPOUT_LABELS for ws in store_mover.world_states)
    assert "support" not in {label for ws in store_mover.world_states for label in _labels(ws)}

    props_mover = _CapturingMover()
    await _run_full_pack(obstacle_source="prop_geometries", mover=props_mover)
    assert props_mover.world_states
    for world_state in props_mover.world_states:
        labels = _labels(world_state)
        assert "support" in labels
        # the held box is excluded, the other seven are obstacles, and any
        # keep-out this leg closes is on top of them
        assert len(labels - _KEEPOUT_LABELS) == 1 + (len(BOX_PROPS) - 1)


class _FakeRelayStore:
    """A world state store fake: ``list_uuids``/``get_transform`` over a
    fixed dict, the same two calls ``read_store_obstacles`` makes."""

    def __init__(self, transforms: dict[str, Transform]) -> None:
        self._transforms = transforms

    async def list_uuids(self, *, extra: Any = None, timeout: float | None = None) -> list[str]:
        return list(self._transforms.keys())

    async def get_transform(
        self, uuid: Any, *, extra: Any = None, timeout: float | None = None
    ) -> Transform:
        return self._transforms[uuid]


def _world_box_transform(name: str, x: float) -> Transform:
    return Transform(
        reference_frame=name,
        pose_in_observer_frame=PoseInFrame(
            reference_frame="world",
            pose=Pose(x=x, y=0.0, z=300.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        ),
        physical_object=Geometry(
            center=Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
            box=RectangularPrism(dims_mm=Vector3(x=200.0, y=150.0, z=100.0)),
            label=name,
        ),
    )


async def test_store_mode_carries_the_stores_boxes_and_prop_mode_does_not():
    """The sequencer's store publishes the placed boxes in the world frame,
    which reach the planner only through the relay; a fixture parented to
    the scene's own frame tree reaches it through the frame system instead,
    so the relay skips and counts it rather than resolving it."""
    store = _FakeRelayStore(
        {
            "box-1": _world_box_transform("box-1-v1", x=100.0),
            "box-2": _world_box_transform("box-2-v1", x=200.0),
            "fixture-1": Transform(
                reference_frame="fixture-1",
                pose_in_observer_frame=PoseInFrame(
                    reference_frame="pallet/group",
                    pose=Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
                ),
                physical_object=Geometry(
                    center=Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
                    box=RectangularPrism(dims_mm=Vector3(x=50.0, y=50.0, z=50.0)),
                    label="fixture-1",
                ),
            ),
        }
    )

    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper, obstacle_source="world_state_store")
    palletizer._stores = [("sequencer-1", store)]
    mover = _CapturingMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    relayed_labels = {label for ws in mover.world_states for label in _labels(ws)} & {
        "box-1-v1",
        "box-2-v1",
        "fixture-1",
    }
    assert relayed_labels == {"box-1-v1", "box-2-v1"}

    status = await palletizer.do_command({"command": "status"})
    obstacles = status["records"][0]["planner_obstacles"]
    assert obstacles["relayed"] == 2
    assert obstacles["skipped_not_in_world"] == 1

    props_world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    props_gripper = FakeGripper(grab_result=True, world=props_world)
    props_palletizer = _make_palletizer(
        world=props_world, gripper=props_gripper, obstacle_source="prop_geometries"
    )
    props_palletizer._stores = [("sequencer-1", store)]
    props_mover = _CapturingMover()
    props_palletizer._build_mover = lambda: props_mover  # type: ignore[method-assign]
    props_palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await props_palletizer.do_command({"command": "start"})
    await props_palletizer.wait_until_done()

    props_relayed_labels = {label for ws in props_mover.world_states for label in _labels(ws)} & {
        "box-1-v1",
        "box-2-v1",
        "fixture-1",
    }
    assert props_relayed_labels == set()

    props_status = await props_palletizer.do_command({"command": "status"})
    props_obstacles = props_status["records"][0]["planner_obstacles"]
    assert props_obstacles["relayed"] == 0


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


async def test_a_restage_that_misses_once_retries_the_same_teleport_and_lands():
    """`set_prop_pose` returns before physics has moved the prop, and the read
    straight after it can still show the box at its old spot. A restage reads
    the box back and, on a miss, sends the same teleport again rather than
    picking against a pose the box was never at."""
    geometries = [_box_geometry(BOX_PROPS[0], x=-900.0), _box_geometry(BOX_PROPS[1], x=-100.0)]
    world = FakeWorld(geometries=geometries, restage_lands_after=1)
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=2)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert [record["outcome"] for record in status["records"]] == [OUTCOME_PLACED, OUTCOME_PLACED]

    seq_two_restages = [
        command
        for command in world.commands
        if command["command"] == "set_prop_pose" and command["name"] == BOX_PROPS[1]
    ]
    assert len(seq_two_restages) == 2


async def test_a_restage_that_lands_only_on_the_third_teleport_is_placed():
    geometries = [_box_geometry(BOX_PROPS[0], x=-900.0), _box_geometry(BOX_PROPS[1], x=-100.0)]
    world = FakeWorld(geometries=geometries, restage_lands_after=2)
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=2)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert [record["outcome"] for record in status["records"]] == [OUTCOME_PLACED, OUTCOME_PLACED]
    sends = [
        command
        for command in world.commands
        if command["command"] == "set_prop_pose" and command["name"] == BOX_PROPS[1]
    ]
    assert len(sends) == 3


async def test_a_restage_that_never_lands_fails_the_seq_and_the_run_skips_it():
    """A box that never reads back at the pose it was sent to - a stuck prop,
    a sim that rejected the teleport - fails the pick that was waiting on it
    rather than driving the cup down over nothing, and the sequencer's own
    retry-then-skip still moves the pack on."""
    geometries = [_box_geometry(BOX_PROPS[0], x=-900.0), _box_geometry(BOX_PROPS[1], x=-100.0)]
    world = FakeWorld(geometries=geometries, never_restages=True)
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    sequencer = FakeSequencer(box_count=2)
    palletizer._sequencer = sequencer  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    seq_two_records = [record for record in status["records"] if record["seq"] == 2]
    assert len(seq_two_records) == 2
    sent_mm = (-900.0, 0.0, 100.0)
    for record in seq_two_records:
        assert record["outcome"] == OUTCOME_FAILED
        assert BOX_PROPS[1] in record["reason"]
        assert str(sent_mm) in record["reason"]
        assert f"{RESTAGE_MAX_ATTEMPTS} attempts" in record["reason"]
    assert sequencer.skip_calls == [(2, seq_two_records[-1]["reason"])]
    sends = [
        command
        for command in world.commands
        if command["command"] == "set_prop_pose" and command["name"] == BOX_PROPS[1]
    ]
    assert len(sends) == RESTAGE_MAX_ATTEMPTS * len(seq_two_records)
    # the pack goes on to the box after the one it gave up on
    assert sequencer.cursor > 2


async def test_a_seq_after_one_with_no_infeed_pose_captured_fails_rather_than_restaging_nothing():
    """A run started at seq > 1 - the sequencer's own cursor moved on without
    a fresh box_props[0] ever being read - has no pose to send this box to.
    Restaging nothing used to pick it wherever it happened to be parked."""
    world = FakeWorld()

    class _StartsAtSeqTwo(FakeSequencer):
        async def next_box(self) -> FakeNextBox:
            if self.cursor > self.box_count:
                return FakeNextBox(0, is_complete=True)
            return FakeNextBox(2)

    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = _StartsAtSeqTwo(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["records"]
    for record in status["records"]:
        assert record["outcome"] == OUTCOME_FAILED
        assert "no infeed pose captured yet" in record["reason"]
    assert world.commands == []


async def test_place_motions_use_the_sequencers_poses_with_no_added_standoff():
    """place_start_in_world already carries the sequencer's own approach
    standoff, clamped so a descending box cannot plow through a neighbour.
    Stacking PRE_GRASP_STANDOFF_MM on top - or any other half-box
    correction - would move the descent/retreat poses off these exact
    values. The carry itself ends PLACE_APPROACH_CLEARANCE_MM above
    place_start, where the box stops swinging, and comes down onto
    place_start in a straight line before the descent proper."""
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
    approach = place_approach_pose(place_start)
    approach_calls = [
        (pose, linear) for pose, linear in mover.calls if _poses_equal(pose, approach)
    ]
    place_start_calls = [
        (pose, linear) for pose, linear in mover.calls if _poses_equal(pose, place_start)
    ]
    release_calls = [(pose, linear) for pose, linear in mover.calls if _poses_equal(pose, release)]
    # the carry (non-linear) ends straight above place_start_in_world, by the
    # approach clearance and nothing else
    assert approach_calls == [(approach, False)]
    assert approach.z == place_start.z + PLACE_APPROACH_CLEARANCE_MM
    assert (approach.x, approach.y) == (place_start.x, place_start.y)
    # the settle onto place_start and the retreat back to it are both linear,
    # and nothing lands there by a free move
    assert place_start_calls == [(place_start, True), (place_start, True)]
    # the lift runs straight up over the pick to the approach's height, so
    # the box clears the place keep-out before the sweep, and the pick
    # standoff is not a stop on the way
    climb = carry_height_pose(BOX_TOP_FACE_XYZ_MM, approach.z)
    assert (climb, True) in mover.calls
    assert climb.z == approach.z
    pick_standoff = pick_grasp_standoff_pose(BOX_TOP_FACE_XYZ_MM)
    linear_standoff_calls = [
        (pose, linear)
        for pose, linear in mover.calls
        if _poses_equal(pose, pick_standoff) and linear
    ]
    assert linear_standoff_calls == []
    # the descent lands at the release pose, linear: place_end_in_world's x, y
    # and orientation, raised by the release clearance, which is zero for a
    # release at contact, so the descent ends on place_end itself
    assert (release, True) in release_calls
    assert release.z == place_end.z + PLACE_RELEASE_CLEARANCE_MM


async def test_the_box_rides_the_gripper_between_the_grab_and_the_release():
    """The planner knows the arm and nothing hanging from it unless told.
    Seven legs per box: approach, descent, lift to carry height, cross,
    settle over the slot, place descent, retreat. The box is on the cup for
    the middle four and nowhere else."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _CapturingMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    holding = [bool(ws.transforms) for ws in mover.world_states]
    assert holding == [False, False, True, True, True, True, False]
    # and every leg with the box on the cup is a level one
    assert mover.levels == holding
    (held,) = mover.world_states[2].transforms
    assert held.reference_frame == BOX_PROPS[0]
    assert held.pose_in_observer_frame.reference_frame == GRIPPER_NAME
    dims = held.physical_object.box.dims_mm
    assert (dims.x, dims.y, dims.z) == (400.0, 300.0, 200.0)


def test_held_box_transform_hangs_the_box_below_the_cup_by_half_its_height():
    held = held_box_transform("infeed_box_1", (150.0, 200.0, 100.0), "gripper-1")
    assert held.reference_frame == "infeed_box_1"
    assert held.pose_in_observer_frame.reference_frame == "gripper-1"
    centre = held.pose_in_observer_frame.pose
    # the gripper's z is the tool axis, pointing down at a grasp, so below
    # the cup is +z: half the box's height, its top face at the cup
    assert (centre.x, centre.y) == (0.0, 0.0)
    assert centre.z == pytest.approx(50.0)
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
    """Every move succeeds except the place descent, the sixth leg: after
    the approach, the grasp descent, the lift to carry height, the cross and
    the settle onto place_start. That one stalls the way the arm reports a
    box meeting the deck."""

    async def move_to(
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
    ) -> None:
        await super().move_to(pose, world_state, linear=linear, level=level)
        if level and linear and len(self.world_states) == 6:
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


def test_place_release_pose_lifts_place_end_by_the_release_clearance():
    """place_end_in_world is the cup at the box's top face with the box on
    its slot, so a descent all the way there sets the box down. The release
    pose adds only the arm's own tracking allowance at the end of a descent,
    PLACE_RELEASE_CLEARANCE_MM."""
    place_end = Pose(x=59.3, y=374.5, z=350.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=-109.5)
    release = place_release_pose(place_end)
    assert (release.x, release.y) == (59.3, 374.5)
    assert release.z == pytest.approx(350.0 + PLACE_RELEASE_CLEARANCE_MM)
    assert (release.o_x, release.o_y, release.o_z, release.theta) == (0.0, 0.0, 1.0, -109.5)


async def test_a_run_begins_by_letting_go_of_whatever_the_cup_holds():
    """A run that failed between grab and release leaves its box held to
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
# stop_after_seq and the post-release settle
# ----------------------------------------------------------------------


async def test_a_bounded_run_stops_after_its_seq_and_a_later_start_finishes_the_pack():
    world = FakeWorld(geometries=_all_box_geometries())
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    sequencer = FakeSequencer(box_count=len(BOX_PROPS))
    palletizer._sequencer = sequencer  # type: ignore[assignment]

    await palletizer.do_command({"command": "start", "stop_after_seq": 4})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["stopped_after_seq"] == 4
    assert [record["seq"] for record in status["records"]] == [1, 2, 3, 4]
    # the sequencer's own cursor kept advancing, so a later start resumes it
    assert sequencer.cursor == 5

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    second_status = await palletizer.do_command({"command": "status"})
    assert second_status["state"] == "complete"
    assert "stopped_after_seq" not in second_status
    assert [record["seq"] for record in second_status["records"]] == [5, 6, 7, 8]
    assert sequencer.cursor > len(BOX_PROPS)


async def test_stop_after_seq_accepts_a_whole_float_from_the_protobuf_round_trip():
    """A protobuf Struct carries every number as a double, so a caller's
    stop_after_seq of 4 crosses DoCommand as 4.0."""
    world = FakeWorld(geometries=_all_box_geometries())
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=len(BOX_PROPS))  # type: ignore[assignment]

    await palletizer.do_command({"command": "start", "stop_after_seq": 4.0})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["stopped_after_seq"] == 4
    assert [record["seq"] for record in status["records"]] == [1, 2, 3, 4]


async def test_infeed_pose_outlives_a_bounded_run_and_restages_the_next_starts_first_box():
    """Only a part restart parks a fresh box_props[0] on the pick station - a
    bounded run's own stop must not forget where seq 1 read it, or the start
    that resumes the pack has nothing to restage its next box onto."""
    world = FakeWorld(geometries=_all_box_geometries())
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=len(BOX_PROPS))  # type: ignore[assignment]

    await palletizer.do_command({"command": "start", "stop_after_seq": 4})
    await palletizer.wait_until_done()
    infeed_after_first_run = palletizer._infeed_position_mm
    assert infeed_after_first_run is not None

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    assert palletizer._infeed_position_mm is infeed_after_first_run

    seq_five_restage = next(
        command
        for command in world.commands
        if command["command"] == "set_prop_pose" and command["name"] == BOX_PROPS[4]
    )
    infeed_pose_mm = _all_box_geometries()[0]["pose_in_world_mm"]
    assert seq_five_restage["position"] == [
        infeed_pose_mm["x"],
        infeed_pose_mm["y"],
        infeed_pose_mm["z"],
    ]


@pytest.mark.parametrize("bad_value", [0, "4", True, 4.5])
async def test_start_rejects_a_bad_stop_after_seq(bad_value: Any):
    palletizer = _make_palletizer()
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="stop_after_seq"):
        await palletizer.do_command({"command": "start", "stop_after_seq": bad_value})


async def test_an_unbounded_run_carries_no_stopped_after_seq_key():
    palletizer, _sequencer, _gripper = await _run_full_pack()
    status = await palletizer.do_command({"command": "status"})
    assert "stopped_after_seq" not in status


async def test_placed_records_carry_a_measured_settle_time():
    palletizer, _sequencer, _gripper = await _run_full_pack()
    status = await palletizer.do_command({"command": "status"})
    for record in status["records"]:
        assert record["outcome"] == OUTCOME_PLACED
        assert isinstance(record["settle_s"], float)
        assert record["settle_s"] >= 0.0


async def test_placed_records_carry_a_measured_swing_time():
    palletizer, _sequencer, _gripper = await _run_full_pack()
    status = await palletizer.do_command({"command": "status"})
    for record in status["records"]:
        assert record["outcome"] == OUTCOME_PLACED
        assert isinstance(record["swing_s"], float)
        assert record["swing_s"] >= 0.0


async def test_settle_sampling_begins_at_release_not_after_the_retreat_leg():
    """The regression this guards: settle_s used to start ticking only once
    the retreat motion had finished, so a box that stopped moving during the
    one-to-two second retreat read settle_s as 0. `events` interleaves the
    world's prop_geometries reads with the retreat leg's own start and finish,
    so a read landing between them shows the sampler was already running."""
    events: list[str] = []
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    original_do_command = world.do_command

    async def recording_do_command(command: dict[str, Any]) -> dict[str, Any]:
        result = await original_do_command(command)
        if command["command"] == "prop_geometries":
            events.append("prop_geometries")
        return result

    world.do_command = recording_do_command  # type: ignore[method-assign]
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)

    class _YieldingMover(_NoopMover):
        async def move_to(
            self,
            pose: Any,
            world_state: Any,
            linear: bool = False,
            level: bool = False,
            *,
            plan_timeout_s: float | None = None,
        ) -> None:
            events.append("move-start")
            await asyncio.sleep(0)
            events.append("move-done")

    palletizer._build_mover = lambda: _YieldingMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["records"][0]["outcome"] == OUTCOME_PLACED

    # the retreat is the last leg of a one-box run, so its start/finish are
    # the last such pair in the interleaved event order
    last_start = max(i for i, event in enumerate(events) if event == "move-start")
    last_done = max(i for i, event in enumerate(events) if event == "move-done")
    assert last_done > last_start
    assert any(event == "prop_geometries" for event in events[last_start + 1 : last_done])


async def test_a_box_that_never_stops_moving_is_recorded_with_no_settle_time():
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])], never_settle=True)
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["records"][0]["outcome"] == OUTCOME_PLACED
    assert "settle_s" not in status["records"][0]


async def test_a_carried_box_that_never_stops_swinging_times_out_but_still_places(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A box that keeps moving on every read through the whole carry never
    lets the trailing window agree, so the wait gives up after
    SETTLE_POLICY.timeout_s rather than blocking the pack forever - the
    place still happens, against the box's last reading, with a warning
    naming it."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])], never_settle_while_carried=True)
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        await palletizer.do_command({"command": "start"})
        await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["records"][0]["outcome"] == OUTCOME_PLACED
    assert "swing_s" not in status["records"][0]
    assert any(
        record.levelname == "WARNING" and BOX_PROPS[0] in record.message
        for record in caplog.records
    )


def test_the_default_settle_policy_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """`monkeypatch` here is the same fixture instance `_fast_settle` used,
    so undoing it restores the real module attribute this test pins."""
    monkeypatch.undo()
    assert palletizer.SETTLE_POLICY is DEFAULT_SETTLE_POLICY


# ----------------------------------------------------------------------
# the leg timeline
# ----------------------------------------------------------------------


_LEG_NAMES_IN_ORDER = (
    "approach to the pick standoff",
    "descent onto the box",
    "grab",
    "lift to carry height",
    "carry to the place approach",
    "settle over the slot",
    "place descent",
    "release",
    "retreat off the place",
)


async def test_every_leg_logs_its_own_start_and_done_in_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    with caplog.at_level("INFO"):
        await palletizer.do_command({"command": "start"})
        await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["records"][0]["outcome"] == OUTCOME_PLACED

    def _index_of(message: str) -> int:
        return next(i for i, record in enumerate(caplog.records) if record.message == message)

    def _index_of_prefix(prefix: str) -> int:
        return next(
            i for i, record in enumerate(caplog.records) if record.message.startswith(prefix)
        )

    start_indices = [
        _index_of(f"palletizer seq 1 leg {name}: start") for name in _LEG_NAMES_IN_ORDER
    ]
    done_indices = [
        _index_of_prefix(f"palletizer seq 1 leg {name}: done in") for name in _LEG_NAMES_IN_ORDER
    ]
    for start_index, done_index in zip(start_indices, done_indices, strict=True):
        assert done_index > start_index
    assert start_indices == sorted(start_indices)
    assert done_indices == sorted(done_indices)


async def test_a_leg_that_stalls_logs_failed_after_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release = place_release_pose(_place_pose(1))
    cup_elsewhere = Pose(
        x=release.x, y=release.y, z=release.z + 200.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0
    )

    with caplog.at_level("WARNING"):
        status, _gripper = await _run_one_box_with_a_stalled_descent(cup_elsewhere)

    assert status["state"] == "failed"
    assert any(
        record.levelname == "WARNING"
        and record.message.startswith("palletizer seq 1 leg place descent: failed after")
        for record in caplog.records
    )


async def test_the_swing_wait_runs_between_the_carry_legs_done_and_the_settle_legs_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: _NoopMover()  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    with caplog.at_level("INFO"):
        await palletizer.do_command({"command": "start"})
        await palletizer.wait_until_done()

    carry_done = next(
        i
        for i, record in enumerate(caplog.records)
        if record.message.startswith("palletizer seq 1 leg carry to the place approach: done in")
    )
    swing_wait = next(
        i
        for i, record in enumerate(caplog.records)
        if record.message.startswith("palletizer seq 1: carried box still after")
    )
    settle_start = next(
        i
        for i, record in enumerate(caplog.records)
        if record.message == "palletizer seq 1 leg settle over the slot: start"
    )
    assert carry_done < swing_wait < settle_start


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
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
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


def test_support_keepout_is_labelled_as_asked_and_by_default_as_the_place_keepout():
    default = support_keepout(_PICK_GEOMETRIES, "frame_pallet", 400.0)
    raised = support_keepout(
        _PICK_GEOMETRIES, "frame_pallet", 400.0, label="place_support_keepout_empty"
    )
    assert default is not None and raised is not None
    assert default.label == "place_support_keepout"
    assert raised.label == "place_support_keepout_empty"
    assert raised.center == default.center
    assert raised.box.dims_mm == default.box.dims_mm


def test_a_labelled_support_keepout_is_none_when_the_raised_ceiling_still_leaves_no_room():
    """The raised keep-out sits the approach clearance above the flush one. A
    layer top that lies further under the deck than that clearance still
    leaves it no room, and no inverted box comes out; one that lies closer
    than that gets a zone of the difference."""
    pallet_top_mm = 200.0 + 100.0 / 2.0
    layer_top_under_deck_mm = pallet_top_mm - 2.0 * PLACE_APPROACH_CLEARANCE_MM
    assert (
        support_keepout(
            _PICK_GEOMETRIES,
            "frame_pallet",
            layer_top_under_deck_mm + PLACE_APPROACH_CLEARANCE_MM,
            label="place_support_keepout_empty",
        )
        is None
    )
    layer_top_near_deck_mm = pallet_top_mm - PLACE_APPROACH_CLEARANCE_MM / 2.0
    zone = support_keepout(
        _PICK_GEOMETRIES,
        "frame_pallet",
        layer_top_near_deck_mm + PLACE_APPROACH_CLEARANCE_MM,
        label="place_support_keepout_empty",
    )
    assert zone is not None
    assert zone.box.dims_mm.z == pytest.approx(PLACE_APPROACH_CLEARANCE_MM / 2.0)


def test_support_keepout_is_none_when_the_support_is_not_in_the_scene():
    assert support_keepout(_PICK_GEOMETRIES, "frame_tray_dock", 400.0) is None


async def test_the_place_keepout_reaches_the_top_of_the_layer_being_placed():
    """The keep-out over the pallet covers the whole stack: its ceiling is the
    sequencer's place_end.z, the top face of the layer being placed, not
    20 mm under it. Until 2026-09-24 the placed boxes' top 20 mm stood out of
    the zone, which is where a carried box's corner clipped them."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0]), *_PICK_GEOMETRIES])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _CapturingMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    place_end = _place_pose(1)
    zones = [
        geometry
        for ws in mover.world_states
        for obstacles in ws.obstacles
        for geometry in obstacles.geometries
        if geometry.label == "place_support_keepout"
    ]
    assert zones, "no leg carried the place keep-out"
    for zone in zones:
        top_z = zone.center.z + zone.box.dims_mm.z / 2.0
        assert top_z == pytest.approx(place_end.z)


def _zone_labelled(world_state, label: str):
    return next(
        geometry
        for obstacles in world_state.obstacles
        for geometry in obstacles.geometries
        if geometry.label == label
    )


async def _legs_with_a_pallet_in_the_scene(mover: Any) -> None:
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0]), *_PICK_GEOMETRIES])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()


async def test_the_approach_to_the_pick_standoff_plans_under_a_keepout_raised_by_the_clearance():
    """The arm runs this leg empty with the stack nearby, and a keep-out
    reaching exactly to the top face of the last box placed gave the planner
    no margin over it: on the GPU run of 2026-10-01 the cups swept the box
    just released. The ceiling here is PLACE_APPROACH_CLEARANCE_MM higher
    than the one every leg with the box on the cups plans under."""
    mover = _CapturingMover()
    await _legs_with_a_pallet_in_the_scene(mover)

    approach_labels = _labels(mover.world_states[0])
    assert "place_support_keepout_empty" in approach_labels
    assert "place_support_keepout" not in approach_labels

    def top_of(zone) -> float:
        return zone.center.z + zone.box.dims_mm.z / 2.0

    raised = _zone_labelled(mover.world_states[0], "place_support_keepout_empty")
    flush = _zone_labelled(mover.world_states[2], "place_support_keepout")
    assert top_of(raised) == pytest.approx(_place_pose(1).z + PLACE_APPROACH_CLEARANCE_MM)
    assert top_of(raised) - top_of(flush) == pytest.approx(PLACE_APPROACH_CLEARANCE_MM)


async def test_every_leg_with_the_box_on_the_cups_keeps_the_keepout_flush_with_the_layer():
    """With the box on the cups its bottom rides PLACE_APPROACH_CLEARANCE_MM
    plus 10 mm above the layer top. A keep-out raised by the clearance would
    leave it 10 mm clear, so the lift and the carry plan under the flush one,
    and the settle and the place descent, which descend onto the stack, under
    neither."""
    mover = _CapturingMover()
    await _legs_with_a_pallet_in_the_scene(mover)

    # grasp descent, lift to carry height, carry to the place approach
    for leg in (1, 2, 3):
        labels = _labels(mover.world_states[leg])
        assert "place_support_keepout" in labels
        assert "place_support_keepout_empty" not in labels
        zone = _zone_labelled(mover.world_states[leg], "place_support_keepout")
        assert zone.center.z + zone.box.dims_mm.z / 2.0 == pytest.approx(_place_pose(1).z)
    assert mover.world_states[2].transforms and mover.world_states[3].transforms
    # settle over the slot, place descent
    for leg in (4, 5):
        labels = _labels(mover.world_states[leg])
        assert "place_support_keepout" not in labels
        assert "place_support_keepout_empty" not in labels


async def test_the_retreat_plans_without_the_place_keepout_and_its_fallback_under_the_raised_one():
    mover = _RefusesTheRetreatOverThePlacedBox()
    await _legs_with_a_pallet_in_the_scene(mover)

    labels_per_leg = [_labels(ws) for ws in mover.world_states]
    # the retreat starts inside the place keep-out's airspace, so it has none
    assert "placed_box_keepout" in labels_per_leg[-2]
    assert not labels_per_leg[-2] & {"place_support_keepout", "place_support_keepout_empty"}
    # the fallback has lost the placed box as an obstacle and keeps the margin
    assert labels_per_leg[-1] == {"pick_box_keepout", "place_support_keepout_empty"}
    raised = _zone_labelled(mover.world_states[-1], "place_support_keepout_empty")
    assert raised.center.z + raised.box.dims_mm.z / 2.0 == pytest.approx(
        _place_pose(1).z + PLACE_APPROACH_CLEARANCE_MM
    )


# ----------------------------------------------------------------------
# the retreat's keep-out over the box it just placed
# ----------------------------------------------------------------------


async def test_retreat_leg_carries_a_keepout_over_the_box_just_placed():
    """The retreat's straight line off the slot has to stay off the box that
    just landed there, or it clips it. Only the retreat leg needs to: every
    earlier leg either has not placed the box yet or is still holding it."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _CapturingMover()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    labels_per_leg = [_labels(ws) for ws in mover.world_states]
    assert "placed_box_keepout" in labels_per_leg[-1]
    assert all("placed_box_keepout" not in labels for labels in labels_per_leg[:-1])

    placed_zone = next(
        geometry
        for obstacles in mover.world_states[-1].obstacles
        for geometry in obstacles.geometries
        if geometry.label == "placed_box_keepout"
    )
    box_geometry = _box_geometry(BOX_PROPS[0])
    pose_mm = box_geometry["pose_in_world_mm"]
    _dim_x, _dim_y, dim_z = box_geometry["box_dims_mm"]
    assert placed_zone.center.x == pytest.approx(pose_mm["x"])
    assert placed_zone.center.y == pytest.approx(pose_mm["y"])
    bottom_z = pose_mm["z"] - dim_z / 2.0
    assert placed_zone.center.z - placed_zone.box.dims_mm.z / 2.0 == pytest.approx(bottom_z)


class _RefusesTheRetreatOverThePlacedBox(_CapturingMover):
    """Refuses only the leg whose world state carries `placed_box_keepout`,
    the way a planner refuses a straight line it cannot find a path for -
    never a stall."""

    async def move_to(
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
    ) -> None:
        await super().move_to(pose, world_state, linear=linear, level=level)
        if "placed_box_keepout" in _labels(world_state):
            raise RuntimeError("motion planner failed to find path")


async def test_a_refused_retreat_over_the_placed_box_falls_back_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A planner that refuses the retreat with the placed box as an obstacle
    - not a stall - gets the free retreat run without it, and a warning
    naming the box, so a clip that follows is not a silent surprise."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _RefusesTheRetreatOverThePlacedBox()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        await palletizer.do_command({"command": "start"})
        await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "complete"
    assert status["records"][0]["outcome"] == OUTCOME_PLACED

    labels_per_leg = [_labels(ws) for ws in mover.world_states]
    # the refused attempt (with the keep-out) and the fallback (without it)
    # both ran, as the last two legs recorded
    assert "placed_box_keepout" in labels_per_leg[-2]
    assert labels_per_leg[-1] == {"pick_box_keepout"}

    assert any(
        record.levelname == "WARNING"
        and BOX_PROPS[0] in record.message
        and "retreat" in record.message
        for record in caplog.records
    )


class _StallsOnTheRetreatOverThePlacedBox(_CapturingMover):
    """Stalls the leg whose world state carries `placed_box_keepout`, the way
    the arm reports a move that planned and then could not be executed."""

    async def move_to(
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        plan_timeout_s: float | None = None,
    ) -> None:
        await super().move_to(pose, world_state, linear=linear, level=level)
        if "placed_box_keepout" in _labels(world_state):
            raise RuntimeError(
                "error moving component arm-1 to inputs [...]: rpc error: code = Aborted desc = "
                "arm arm-1 stalled at waypoint 12/12 (1 consecutive, stuck joints: j3: at -98.7 "
                "want -100.1)"
            )


async def test_a_stalled_retreat_over_the_placed_box_propagates_without_a_fallback():
    """A retreat that planned around the placed box and then stalled on the
    arm is not a planner refusal, and does not get the free-retreat fallback
    that would bury the stall under a second, unrelated move."""
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    mover = _StallsOnTheRetreatOverThePlacedBox()
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]

    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()

    status = await palletizer.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "stalled at waypoint 12/12" in status["reason"]
    # no fallback retreat was attempted
    assert len(mover.world_states) == 7


_PLANNER_TIMEOUT = "cbirrt timeout context deadline exceeded"


class _TimesOutOnce(_CapturingMover):
    """Raises on the first ``failures`` attempts of the first move whose
    ``linear`` flag equals ``on_linear``, then plans normally. Keeps every
    call's ``planner_seed`` (None when the keyword was not passed)."""

    def __init__(self, *, on_linear: bool, failures: int = 1, message: str = _PLANNER_TIMEOUT):
        super().__init__()
        self.on_linear = on_linear
        self.failures_left = failures
        self.message = message
        self.seeds: list[tuple[bool, int | None]] = []
        self.timeouts: list[tuple[bool, float | None]] = []

    async def move_to(
        self,
        pose: Any,
        world_state: Any,
        linear: bool = False,
        level: bool = False,
        *,
        planner_seed: int | None = None,
        plan_timeout_s: float | None = None,
    ) -> None:
        self.seeds.append((linear, planner_seed))
        self.timeouts.append((linear, plan_timeout_s))
        if linear == self.on_linear and self.failures_left > 0:
            self.failures_left -= 1
            raise GRPCError(Status.DEADLINE_EXCEEDED, self.message)
        await super().move_to(pose, world_state, linear=linear, level=level)


async def _run_one_box_with(mover: Any) -> dict[str, Any]:
    world = FakeWorld(geometries=[_box_geometry(BOX_PROPS[0])])
    gripper = FakeGripper(grab_result=True, world=world)
    palletizer = _make_palletizer(world=world, gripper=gripper)
    palletizer._build_mover = lambda: mover  # type: ignore[method-assign]
    palletizer._sequencer = FakeSequencer(box_count=1)  # type: ignore[assignment]
    await palletizer.do_command({"command": "start"})
    await palletizer.wait_until_done()
    return await palletizer.do_command({"command": "status"})  # type: ignore[no-any-return]


async def test_a_free_plan_that_times_out_is_planned_once_more_under_another_seed():
    mover = _TimesOutOnce(on_linear=False)

    status = await _run_one_box_with(mover)

    assert [record["outcome"] for record in status["records"]] == [OUTCOME_PLACED]
    first_free = [i for i, (linear, _seed) in enumerate(mover.seeds) if not linear][:2]
    assert mover.seeds[first_free[0]] == (False, None)
    assert mover.seeds[first_free[1]] == (False, 1)
    assert [seed for _linear, seed in mover.seeds if seed is not None] == [1]


async def test_the_first_attempt_of_a_free_move_is_bounded_and_the_retry_is_not():
    mover = _TimesOutOnce(on_linear=False)

    await _run_one_box_with(mover)

    free = [
        (seed, timeout)
        for (linear, seed), (_l, timeout) in zip(mover.seeds, mover.timeouts, strict=True)
        if not linear
    ]
    assert free[0] == (None, FIRST_FREE_PLAN_TIMEOUT_S)
    assert free[1] == (1, None)
    assert all(timeout == FIRST_FREE_PLAN_TIMEOUT_S for seed, timeout in free if seed is None)


async def test_a_straight_line_move_carries_neither_planner_keyword():
    mover = _TimesOutOnce(on_linear=False, failures=0)

    await _run_one_box_with(mover)

    linear = [
        (seed, t)
        for (is_linear, seed), (_l, t) in zip(mover.seeds, mover.timeouts, strict=True)
        if is_linear
    ]
    assert linear
    assert all(entry == (None, None) for entry in linear)


async def test_two_planner_timeouts_in_a_row_fail_the_run():
    mover = _TimesOutOnce(on_linear=False, failures=2)

    status = await _run_one_box_with(mover)

    assert status["state"] == "failed"
    assert "cbirrt timeout" in status["reason"]


async def test_a_linear_timeout_is_not_retried_in_place():
    mover = _TimesOutOnce(on_linear=True)

    status = await _run_one_box_with(mover)

    assert [record["outcome"] for record in status["records"]] == [OUTCOME_PLACED]
    assert [seed for _linear, seed in mover.seeds if seed is not None] == []


async def test_a_free_move_that_failed_on_the_arm_is_never_retried():
    mover = _TimesOutOnce(
        on_linear=False, message="arm arm-1 did not reach final waypoint 12/12 in time"
    )

    status = await _run_one_box_with(mover)

    assert status["state"] == "failed"
    assert [seed for _linear, seed in mover.seeds if seed is not None] == []


def test_is_planner_timeout_tells_a_planner_timeout_from_an_arm_timeout():
    assert is_planner_timeout(GRPCError(Status.DEADLINE_EXCEEDED, _PLANNER_TIMEOUT))
    assert not is_planner_timeout(
        GRPCError(Status.DEADLINE_EXCEEDED, "did not reach final waypoint")
    )
    assert not is_planner_timeout(GRPCError(Status.ABORTED, _PLANNER_TIMEOUT))
    assert not is_planner_timeout(RuntimeError(_PLANNER_TIMEOUT))
