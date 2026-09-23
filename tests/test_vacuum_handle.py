import math
import time
from typing import ClassVar

import pytest

from isaac_module.asset_catalog import CUP_APPROACH_GAP_MM
from isaac_module.epick import attachment_points_tool_m
from isaac_module.handles.vacuum import (
    DEFAULT_GRAB_DELAY_MS,
    DEFAULT_RELEASE_DELAY_MS,
    IsaacVacuumHandle,
    StagePoseReader,
)
from isaac_module.spatial import quat_from_axis_angle
from isaac_module.surface_gripper import CupCompliance, HoldLoad

# --- protocol conformance against the mock ---------------------------------


def test_create_vacuum_gripper_unknown_arm_raises(sim):
    with pytest.raises(ValueError, match="not attached to the sim"):
        sim.create_vacuum_gripper("vacuum-bad-arm", {"world": "isaac-world", "arm": "no-such-arm"})


def test_grab_with_attach_prop_holds(sim):
    sim.create_arm("vacuum-arm-a", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-a",
        {"world": "isaac-world", "arm": "vacuum-arm-a", "mock_attach_prop": "box-1"},
    )

    assert vacuum.is_holding() is False
    vacuum.grab()
    assert vacuum.is_holding() is True


def test_grab_with_no_attach_prop_never_holds(sim):
    sim.create_arm("vacuum-arm-b", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper("vacuum-b", {"world": "isaac-world", "arm": "vacuum-arm-b"})

    vacuum.grab()
    assert vacuum.is_holding() is False


def test_open_after_grab_releases(sim):
    sim.create_arm("vacuum-arm-c", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-c",
        {"world": "isaac-world", "arm": "vacuum-arm-c", "mock_attach_prop": "box-1"},
    )

    vacuum.grab()
    assert vacuum.is_holding() is True
    vacuum.open()
    assert vacuum.is_holding() is False


def test_is_moving_false_before_grab_and_after_zero_delay_grab(sim):
    sim.create_arm("vacuum-arm-d", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-d",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-d",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 0,
        },
    )

    assert vacuum.is_moving() is False
    vacuum.grab()
    assert vacuum.is_moving() is False


def test_is_moving_true_during_grab_delay_window_then_false(sim):
    sim.create_arm("vacuum-arm-dd", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-dd",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-dd",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 50,
        },
    )

    assert vacuum.is_moving() is False
    vacuum.grab()
    assert vacuum.is_moving() is True
    time.sleep(0.1)
    assert vacuum.is_moving() is False
    # the weld itself is instant, so holding is already reportable during the window
    assert vacuum.is_holding() is True


def test_grab_delay_defaults_to_the_epick_default(sim):
    sim.create_arm("vacuum-arm-dg", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-dg",
        {"world": "isaac-world", "arm": "vacuum-arm-dg", "mock_attach_prop": "box-1"},
    )

    # the EPick's own gripping time from its manual
    assert DEFAULT_GRAB_DELAY_MS == 150
    vacuum.grab()
    assert vacuum.is_moving() is True
    assert vacuum._grab_delay_s == pytest.approx(0.15)


def test_open_clears_the_grab_window(sim):
    sim.create_arm("vacuum-arm-do", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-do",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-do",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 5000,
        },
    )

    vacuum.grab()
    assert vacuum.is_moving() is True
    vacuum.open()
    assert vacuum.is_moving() is False


def test_dof_names_is_empty(sim):
    sim.create_arm("vacuum-arm-e", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper("vacuum-e", {"world": "isaac-world", "arm": "vacuum-arm-e"})

    assert vacuum.dof_names() == []


def test_poll_state_matches_is_moving_and_is_holding(sim):
    sim.create_arm("vacuum-arm-f", {"world": "isaac-world", "asset": "ur5e"})
    vacuum = sim.create_vacuum_gripper(
        "vacuum-f",
        {
            "world": "isaac-world",
            "arm": "vacuum-arm-f",
            "mock_attach_prop": "box-1",
            "grab_delay_ms": 0,
        },
    )

    vacuum.grab()
    moving, holding = vacuum.poll_state()
    assert (moving, holding) == (vacuum.is_moving(), vacuum.is_holding())
    assert moving is False
    assert holding is True


# --- IsaacVacuumHandle over a fake surface gripper interface ---------------


class _FakeWorld:
    """The two physics-callback methods release() drives through _sim.run."""

    def __init__(self) -> None:
        self._callbacks: set[str] = set()

    def add_physics_callback(self, name, callback):
        self._callbacks.add(name)

    def physics_callback_exists(self, name):
        return name in self._callbacks

    def remove_physics_callback(self, name):
        self._callbacks.discard(name)


class _FakePoseReader:
    """A stand-in for StagePoseReader: world_pose() reads whatever the test
    last set for that prim's path, in a dict shared with the sim, counts
    begin_step() calls, records every read in order, and raises `error` on a
    read when a test sets one."""

    def __init__(self, poses) -> None:
        self._poses = poses
        self.begin_steps = 0
        self.reads: list[str] = []
        self.error: Exception | None = None

    def begin_step(self) -> None:
        self.begin_steps += 1

    def world_pose(self, path):
        self.reads.append(path)
        if self.error is not None:
            raise self.error
        return self._poses[path]


class _FakeIsaac:
    """The monitor must never build a prim view, so this namespace has no
    SingleXFormPrim: any attempt raises AttributeError."""


class _FakeSim:
    """run(fn) calls fn() directly, in the spirit of the mock's synchronous
    handles: nothing here actually needs a sim thread hop."""

    def __init__(self) -> None:
        self.world = _FakeWorld()
        # prim path -> (position, orientation-wxyz) and prim path -> half box
        # dims, both set directly by a test
        self.poses: dict[str, tuple] = {}
        self.half_dims: dict[str, tuple[float, float, float]] = {}
        self.pose_reader = _FakePoseReader(self.poses)
        self._isaac = _FakeIsaac()

    def run(self, fn):
        return fn()

    def unregister_post_reset(self, name):
        pass


class _FakeGripperInterface:
    """A stand-in for what acquire_surface_gripper_interface() returns.
    close_gripper needs two calls to settle on a payload, Closing then
    Closed, matching a real gripper's retry against its own raycast; with
    nothing under the cup it never gets past Closing."""

    def __init__(self, has_payload: bool) -> None:
        self._has_payload = has_payload
        self.status = "Open"
        self.objects: list[str] = []
        self.refuse_close = False
        self.close_calls = 0
        self.open_calls = 0
        self.status_reads = 0

    def close_gripper(self, path: str) -> bool:
        self.close_calls += 1
        if self.refuse_close:
            return False
        if self._has_payload:
            self.status = "Closed"
            self.objects = ["/World/box-1"]
        else:
            self.status = "Closing"
            self.objects = []
        return True

    def open_gripper(self, path: str) -> bool:
        self.open_calls += 1
        self.status = "Open"
        self.objects = []
        return True

    def get_gripper_status(self, path: str):
        self.status_reads += 1
        return self.status

    def get_gripped_objects(self, path: str):
        return self.objects


TOOL_PATH = "/World/Arm/EPick"
OBJECT_PATH = "/World/box-1"
IDENTITY = (1.0, 0.0, 0.0, 0.0)
CUPS = attachment_points_tool_m()
HALF = (0.2, 0.15, 0.125)
CLEARANCE_M = CUP_APPROACH_GAP_MM / 1000.0


def _make_handle(iface, sim=None, **kwargs) -> IsaacVacuumHandle:
    if sim is None:
        sim = _FakeSim()
    kwargs.setdefault("pose_reader", sim.pose_reader)
    kwargs.setdefault("prop_half_dims", lambda: sim.half_dims)
    return IsaacVacuumHandle(
        sim,
        "vacuum-x",
        "/World/Arm/SurfaceGripper",
        TOOL_PATH,
        "/World/Arm/wrist_3_link",
        iface,
        **kwargs,
    )


def _coaxial_kwargs(limit_n=20.0):
    return dict(
        compliance=CupCompliance(),
        coaxial_force_limit_n=limit_n,
        physics_dt=1 / 120,
        cup_points_tool_m=CUPS,
    )


def _place_box(sim, gap_m, quat=IDENTITY, x=0.0, y=0.0, path=OBJECT_PATH, half=HALF):
    """Puts a box under the tool with its near face `gap_m` below the cup
    plane. The tool sits at the world origin looking along +Z, so with the
    box's frame aligned to the tool's its -Z face is the one the cups see."""
    sim.poses[TOOL_PATH] = ((0.0, 0.0, 0.0), IDENTITY)
    sim.poses[path] = ((x, y, gap_m + half[2]), quat)
    sim.half_dims[path] = half


def test_grab_with_a_payload_reads_closed_holding_and_engaged():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    handle.grab()

    assert handle.is_holding() is True
    assert handle.is_engaged() is True
    status, objects = handle.gripper_status()
    assert status == "Closed"
    assert objects == ["/World/box-1"]


def test_grab_with_nothing_under_the_cup_reads_closing_not_holding():
    iface = _FakeGripperInterface(has_payload=False)
    handle = _make_handle(iface)

    handle.grab()

    assert handle.is_engaged() is True
    assert handle.is_holding() is False
    status, _objects = handle.gripper_status()
    assert status == "Closing"


def test_open_reads_open_not_holding_not_engaged():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    handle.grab()
    handle.open()

    assert handle.is_holding() is False
    assert handle.is_engaged() is False
    status, objects = handle.gripper_status()
    assert status == "Open"
    assert objects == []


def test_a_mid_carry_loss_reads_not_holding_with_no_call_in_between():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    handle.grab()
    assert handle.is_holding() is True

    # the plugin itself broke the hold under load, with nothing commanded here
    iface.status = "Open"
    iface.objects = []

    assert handle.is_holding() is False


def test_stop_opens():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    handle.grab()
    handle.stop()

    assert handle.is_engaged() is False
    status, _objects = handle.gripper_status()
    assert status == "Open"


@pytest.mark.parametrize(
    ("status", "objects", "expected"),
    [
        ("Closed", ["/World/box-1"], True),
        ("Closed", [], False),
        ("Closing", ["/World/box-1"], False),
        ("Open", ["/World/box-1"], False),
    ],
)
def test_is_holding_truth_table(status, objects, expected):
    iface = _FakeGripperInterface(has_payload=False)
    iface.status = status
    iface.objects = objects
    handle = _make_handle(iface)

    assert handle.is_holding() is expected


def test_post_reset_recloses_when_engaged():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    handle.grab()
    calls_before = iface.close_calls
    handle.post_reset()

    assert iface.close_calls == calls_before + 1


def test_post_reset_does_nothing_when_not_engaged():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    handle.post_reset()

    assert iface.close_calls == 0
    assert iface.open_calls == 0


def test_release_delay_s_defaults_to_the_epick_release_time():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface)

    # the EPick's own release time from its manual
    assert DEFAULT_RELEASE_DELAY_MS == 180
    assert handle.release_delay_s == pytest.approx(0.18)


def test_release_delay_s_follows_release_delay_ms():
    iface = _FakeGripperInterface(has_payload=True)
    handle = _make_handle(iface, release_delay_ms=250)

    assert handle.release_delay_s == pytest.approx(0.25)


def test_a_refused_close_leaves_not_holding_and_still_engaged():
    iface = _FakeGripperInterface(has_payload=True)
    iface.refuse_close = True
    handle = _make_handle(iface)

    handle.grab()

    assert handle.is_engaged() is True
    assert handle.is_holding() is False


# --- the coaxial monitor: _on_physics_step ----------------------------------


class _RaisingInterface:
    """Fails any call, so a test that expects the monitor to be off can
    prove it never touches the interface."""

    def get_gripper_status(self, path):
        raise AssertionError("interface read while the coaxial monitor should be off")

    def get_gripped_objects(self, path):
        raise AssertionError("interface read while the coaxial monitor should be off")

    def open_gripper(self, path):
        raise AssertionError("open_gripper called while the coaxial monitor should be off")


class _EnumLike:
    def __init__(self, name: str) -> None:
        self.name = name


def test_status_name_normalises_int_enum_and_str():
    from isaac_module.surface_gripper import status_name

    assert status_name(2) == "Closed"
    assert status_name(_EnumLike("Closing")) == "Closing"
    assert status_name("Open") == "Open"


class _OffInterface:
    """Raises on any read: the monitor must not touch the interface while off."""

    def get_gripper_status(self, path):
        raise AssertionError("interface read while the coaxial monitor should be off")

    def get_gripped_objects(self, path):
        raise AssertionError("interface read while the coaxial monitor should be off")


def _armed_handle(limit_n=20.0, gap_before_grab_m=CLEARANCE_M):
    """A handle that grabbed box-1 with the monitor on, the box under the cups
    at `gap_before_grab_m` when the grab was read (the clearance itself by
    default, so the face rests at the cup plane), ready for face samples."""
    iface = _FakeGripperInterface(has_payload=True)
    sim = _FakeSim()
    handle = _make_handle(iface, sim=sim, **_coaxial_kwargs(limit_n))
    _place_box(sim, gap_before_grab_m)
    handle.grab()
    return handle, iface, sim


def test_zero_limit_turns_the_monitor_off():
    handle = _make_handle(_OffInterface(), **_coaxial_kwargs(limit_n=0.0))

    handle._on_physics_step(1 / 120)

    assert handle.hold_load().monitor == "off"


def test_monitor_is_off_without_a_pose_reader_or_prop_dimensions():
    no_reader = _make_handle(_OffInterface(), pose_reader=None, **_coaxial_kwargs())
    no_dims = _make_handle(_OffInterface(), prop_half_dims=None, **_coaxial_kwargs())

    no_reader._on_physics_step(1 / 120)
    no_dims._on_physics_step(1 / 120)

    assert no_reader.hold_load().monitor == "off"
    assert no_dims.hold_load().monitor == "off"


def test_grab_reads_where_the_plugin_will_leave_the_face_at_rest():
    handle, _, sim = _armed_handle(gap_before_grab_m=0.0057)

    assert handle._cup_rest_offsets_m == pytest.approx((0.0014,) * 4)
    assert sim.pose_reader.reads[0] == TOOL_PATH
    assert OBJECT_PATH in sim.pose_reader.reads


@pytest.mark.parametrize("gap_m", [0.040, -0.002])
def test_grab_with_no_box_in_reach_leaves_the_rest_at_the_cup_plane(gap_m):
    handle, _, _ = _armed_handle(gap_before_grab_m=gap_m)

    assert handle._cup_rest_offsets_m is None


def test_grab_picks_the_nearest_of_two_boxes_under_the_cups():
    iface = _FakeGripperInterface(has_payload=True)
    sim = _FakeSim()
    handle = _make_handle(iface, sim=sim, **_coaxial_kwargs())
    _place_box(sim, 0.009, path="/World/box-far")
    _place_box(sim, 0.006)

    handle.grab()

    assert handle._cup_rest_offsets_m == pytest.approx((0.002,) * 4)


def test_a_failing_read_before_the_grab_is_logged_and_the_grab_still_closes(caplog):
    iface = _FakeGripperInterface(has_payload=True)
    sim = _FakeSim()
    handle = _make_handle(iface, sim=sim, **_coaxial_kwargs())
    sim.pose_reader.error = RuntimeError("stage gone")

    with caplog.at_level("ERROR"):
        handle.grab()

    assert iface.close_calls == 1
    assert handle._cup_rest_offsets_m is None
    assert any("could not read the box under the cups" in r.message for r in caplog.records)


def test_open_status_resets_the_window_and_never_opens():
    iface = _FakeGripperInterface(has_payload=True)
    sim = _FakeSim()
    handle = _make_handle(iface, sim=sim, **_coaxial_kwargs())
    handle._load_window.push(30.0)
    handle._hold_active = True

    handle._on_physics_step(1 / 120)

    assert handle._hold_active is False
    assert handle.hold_load().coaxial_load_n == 0.0
    assert handle.hold_load().monitor == "idle"
    assert iface.open_calls == 0
    assert sim.pose_reader.begin_steps == 0


def test_the_face_5_5mm_below_the_plane_reads_25n_and_opens_on_the_twelfth_step():
    handle, iface, sim = _armed_handle()
    _place_box(sim, 0.0055)

    for _ in range(11):
        handle._on_physics_step(1 / 120)
    assert iface.open_calls == 0
    assert handle.hold_load().coaxial_load_n == 0.0
    assert handle.hold_load().monitor == "armed"

    handle._on_physics_step(1 / 120)

    assert iface.open_calls == 1
    assert iface.status == "Open"
    assert handle.hold_load().released_load_n == pytest.approx(25.0)
    assert handle.hold_load().peak_coaxial_load_n == pytest.approx(25.0)
    assert handle.hold_load().monitor == "idle"
    assert handle.is_holding() is False

    handle._on_physics_step(1 / 120)
    assert iface.open_calls == 1


def test_the_stretch_is_read_against_where_the_face_rests():
    handle, _, sim = _armed_handle(gap_before_grab_m=0.0057)
    # rest 1.4 mm, dead band 0.5 mm, spring 4.5 mm: 22.5 N
    _place_box(sim, 0.0014 + 0.0005 + 0.0045)

    handle._on_physics_step(1 / 120)

    assert handle.hold_load().peak_coaxial_load_n == pytest.approx(22.5)


def test_a_face_above_the_plane_reads_no_load():
    handle, _, sim = _armed_handle()
    _place_box(sim, -0.003)

    handle._on_physics_step(1 / 120)

    assert handle.hold_load().peak_coaxial_load_n == 0.0


def test_one_onset_step_over_the_limit_does_not_open_but_raises_the_peak():
    handle, iface, sim = _armed_handle()
    _place_box(sim, 0.002)
    for _ in range(11):
        handle._on_physics_step(1 / 120)
    _place_box(sim, 0.008)

    handle._on_physics_step(1 / 120)

    assert iface.open_calls == 0
    assert handle.hold_load().peak_coaxial_load_n == pytest.approx(37.5)
    assert handle.hold_load().coaxial_load_n == pytest.approx((11 * 7.5 + 37.5) / 12)


def test_a_tilted_box_loads_its_far_cups_and_the_step_reads_the_worst_one():
    handle, _, sim = _armed_handle()
    tilt = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(3.0))
    _place_box(sim, 0.003, quat=tilt)
    handle._on_physics_step(1 / 120)
    tilted_peak = handle.hold_load().peak_coaxial_load_n

    handle.grab()
    _place_box(sim, 0.003)
    handle._on_physics_step(1 / 120)
    flat_peak = handle.hold_load().peak_coaxial_load_n

    assert tilted_peak > flat_peak
    assert flat_peak == pytest.approx(12.5)


def test_a_held_object_with_no_dimensions_is_not_watched_and_warns_once(caplog):
    handle, iface, sim = _armed_handle()
    del sim.half_dims[OBJECT_PATH]

    with caplog.at_level("WARNING"):
        handle._on_physics_step(1 / 120)
        handle._on_physics_step(1 / 120)

    assert handle.hold_load().monitor == "idle"
    assert iface.open_calls == 0
    warnings = [r.message for r in caplog.records if "cannot read" in r.message]
    assert len(warnings) == 1
    assert "no box dimensions" in warnings[0]


def test_a_held_box_no_cup_sits_over_is_not_watched_and_warns_once(caplog):
    handle, _, sim = _armed_handle()
    _place_box(sim, 0.005, x=0.6)

    with caplog.at_level("WARNING"):
        handle._on_physics_step(1 / 120)
        handle._on_physics_step(1 / 120)

    assert handle.hold_load().monitor == "idle"
    warnings = [r.message for r in caplog.records if "cannot read" in r.message]
    assert len(warnings) == 1
    assert "no cup sits over" in warnings[0]


def test_grab_resets_the_readings_and_open_keeps_them():
    handle, _, sim = _armed_handle()
    _place_box(sim, 0.0055)
    for _ in range(12):
        handle._on_physics_step(1 / 120)
    assert handle.hold_load().released_load_n == pytest.approx(25.0)

    handle.open()
    assert handle.hold_load().peak_coaxial_load_n == pytest.approx(25.0)
    assert handle.hold_load().released_load_n == pytest.approx(25.0)

    handle.grab()
    assert handle.hold_load() == HoldLoad(monitor="idle")
    assert handle._hold_active is False


def test_release_removes_the_physics_callback():
    iface = _FakeGripperInterface(has_payload=True)
    sim = _FakeSim()
    handle = _make_handle(iface, sim=sim, **_coaxial_kwargs())
    sim.world.add_physics_callback(handle.coaxial_callback_name, handle._on_physics_step)

    handle.release()

    assert not sim.world.physics_callback_exists(handle.coaxial_callback_name)


def test_monitor_state_reads_off_idle_armed_and_back_to_idle():
    assert _make_handle(_FakeGripperInterface(has_payload=True)).hold_load().monitor == "off"

    handle, iface, sim = _armed_handle()
    assert handle.hold_load().monitor == "idle"
    _place_box(sim, 0.001)
    handle._on_physics_step(1 / 120)
    assert handle.hold_load().monitor == "armed"

    iface.open_gripper("/World/Arm/SurfaceGripper")
    handle._on_physics_step(1 / 120)
    assert handle.hold_load().monitor == "idle"


def test_monitor_begins_one_pose_step_per_monitored_physics_step():
    handle, _, sim = _armed_handle()
    _place_box(sim, 0.001)
    sim.pose_reader.begin_steps = 0
    sim.pose_reader.reads.clear()

    for _ in range(3):
        handle._on_physics_step(1 / 120)

    assert sim.pose_reader.begin_steps == 3
    assert sim.pose_reader.reads == [TOOL_PATH, OBJECT_PATH] * 3


def test_a_raising_pose_read_stops_the_monitor_after_one_logged_error(caplog):
    handle, iface, sim = _armed_handle()
    _place_box(sim, 0.001)
    sim.pose_reader.error = RuntimeError("stage gone")

    with caplog.at_level("ERROR"):
        handle._on_physics_step(1 / 120)
        handle._on_physics_step(1 / 120)

    assert handle.hold_load().monitor == "stopped"
    assert iface.open_calls == 0
    assert sum("coaxial monitor stopped" in r.message for r in caplog.records) == 1


class _FakeQuat:
    def __init__(self, w, x, y, z) -> None:
        self._w, self._xyz = w, (x, y, z)

    def GetReal(self):
        return self._w

    def GetImaginary(self):
        return self._xyz


class _FakeMatrix:
    """A stand-in for the Gf.Matrix4d an XformCache returns: a translation and
    a rotation, with RemoveScaleShear() a no-op on it."""

    def __init__(self, translation, quat_wxyz) -> None:
        self._translation = translation
        self._quat = quat_wxyz

    def ExtractTranslation(self):
        return self._translation

    def RemoveScaleShear(self):
        return self

    def ExtractRotationQuat(self):
        return _FakeQuat(*self._quat)


class _FakeXformCache:
    instances: ClassVar[list["_FakeXformCache"]] = []

    def __init__(self) -> None:
        self.clears = 0
        _FakeXformCache.instances.append(self)

    def Clear(self) -> None:
        self.clears += 1

    def GetLocalToWorldTransform(self, prim):
        return _FakeMatrix(*prim)


class _FakeUsdGeom:
    XformCache = _FakeXformCache


def test_stage_pose_reader_reads_a_prim_through_one_xform_cache_cleared_per_step():
    _FakeXformCache.instances.clear()
    prims = {"/World/box-1": ((1.0, 2.0, 3.0), (0.5, 0.5, 0.5, 0.5))}
    reader = StagePoseReader(_FakeUsdGeom, prims.__getitem__)

    reader.begin_step()
    position, orientation = reader.world_pose("/World/box-1")
    reader.begin_step()
    reader.world_pose("/World/box-1")

    assert position == (1.0, 2.0, 3.0)
    assert orientation == (0.5, 0.5, 0.5, 0.5)
    assert len(_FakeXformCache.instances) == 1
    assert _FakeXformCache.instances[0].clears == 1


def test_stage_pose_reader_builds_its_cache_on_a_read_before_any_step():
    _FakeXformCache.instances.clear()
    prims = {"/World/box-1": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))}
    reader = StagePoseReader(_FakeUsdGeom, prims.__getitem__)

    assert reader.world_pose("/World/box-1") == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    assert len(_FakeXformCache.instances) == 1
