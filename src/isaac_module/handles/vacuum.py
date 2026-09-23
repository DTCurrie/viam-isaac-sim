from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from viam.logging import getLogger

from ..asset_catalog import CUP_APPROACH_GAP_MM, EPICK
from ..cup_contact import cup_contact, cup_gaps_before_grab, rest_offsets_m
from ..spatial import Quat, Vec3, pose_in_frame
from ..surface_gripper import (
    COAXIAL_LOAD_WINDOW_S,
    DEFAULT_MAX_GRIP_DISTANCE_MM,
    CupCompliance,
    HoldLoad,
    LoadWindow,
    cup_pull_load_n,
    status_name,
)
from .arm import ArmHandle
from .gripper import GripperHandle

if TYPE_CHECKING:
    from ..sim_manager import SimManager

LOGGER = getLogger(__name__)

# the EPick's own gripping and release times for one cup, from its manual
# (asset_catalog.EPICK): how long a grab takes to build vacuum before the
# gripper can say whether it caught anything, and how long a release takes to
# bleed it. The demo machine's overlay sets grab_delay_ms to 250 on the
# module's own simulated model, and that still applies over this default
DEFAULT_GRAB_DELAY_MS = int(EPICK["grip_time_ms"])
DEFAULT_RELEASE_DELAY_MS = int(EPICK["release_time_ms"])


class VacuumGripperHandle(GripperHandle):
    """The suction extension of the core protocol. A cup that ran with nothing
    under it is engaged and holding nothing, which a jaw cannot be, so the two
    states are separate here. A caller that needs the command rather than the
    outcome has to narrow to this type, the same way a jaw caller narrows to
    JawGripperHandle."""

    _grab_delay_s: float = DEFAULT_GRAB_DELAY_MS / 1000.0
    _engaged_at: float | None = None
    # how long a release takes to bleed the vacuum, which the model waits out
    # after open(). The mock releases instantly
    release_delay_s: float = 0.0

    def is_engaged(self) -> bool:
        """Whether the cup was last commanded to take hold. True after grab()
        even when nothing was found, False after open()."""
        raise NotImplementedError

    def gripper_status(self) -> tuple[str, list[str]]:
        """What the gripper itself says: its status, one of Open, Closing or
        Closed, and the prim paths of the objects it holds. is_holding() is
        derived from these (Closed with at least one object), and the
        model's is_holding_something carries both in its meta."""
        raise NotImplementedError

    def _start_grab_window(self) -> None:
        """Record that a grab was just commanded, so is_moving() can report
        True until grab_delay_s has passed."""
        self._engaged_at = time.monotonic()

    def _clear_grab_window(self) -> None:
        self._engaged_at = None

    def is_moving(self) -> bool:
        """True while a commanded grab is still within its grab_delay_s
        window, matching the time a real epick takes to build suction."""
        if self._engaged_at is None:
            return False
        return (time.monotonic() - self._engaged_at) < self._grab_delay_s

    def hold_load(self) -> HoldLoad:
        """The module's own reading of the pull on the cups. The mock has no
        bellows to read, so the base answer is no load."""
        return HoldLoad()


class PoseReader(Protocol):
    """World poses for the coaxial monitor, read on the sim thread inside a
    physics step: ``begin_step`` once per step, then ``world_pose`` per prim."""

    def begin_step(self) -> None: ...

    def world_pose(self, prim_path: str) -> tuple[Vec3, Quat]: ...


class StagePoseReader:
    """World poses read straight off the stage through a ``UsdGeom.XformCache``,
    cleared once per physics step, the rotation taken with the scale removed.
    A pure read: building a prim view inside a physics step rewrote the prim's
    xform ops, which rebuilt the articulation and left every physics view
    answering "failed to get ... from backend" for the rest of the run (GPU
    machine, 2026-09-23)."""

    def __init__(self, usd_geom: Any, get_prim_at_path: Callable[[str], Any]) -> None:
        self._usd_geom = usd_geom
        self._get_prim_at_path = get_prim_at_path
        self._cache: Any = None

    def begin_step(self) -> None:
        if self._cache is None:
            self._cache = self._usd_geom.XformCache()
        else:
            self._cache.Clear()

    def world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        if self._cache is None:
            self.begin_step()
        matrix = self._cache.GetLocalToWorldTransform(self._get_prim_at_path(prim_path))
        translation = matrix.ExtractTranslation()
        rotation = matrix.RemoveScaleShear().ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        return (
            (float(translation[0]), float(translation[1]), float(translation[2])),
            (
                float(rotation.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ),
        )


class IsaacVacuumHandle(VacuumGripperHandle):
    """Drives Isaac's ``IsaacSurfaceGripper`` prim over the interface
    ``acquire_surface_gripper_interface()`` returns. The gripper raycasts from
    its own attachment points on close and holds whatever it hits. The hold
    breaks two ways with no command from the caller: the plugin releases it
    once a sideways load exceeds the prim's shear force limit, and this
    handle's own monitor (``_on_physics_step``, a physics callback the sim
    manager registers under ``coaxial_callback_name``) opens the gripper once
    the pull on any one cup, read off how far the held box's face has dropped
    below that cup and averaged over ``COAXIAL_LOAD_WINDOW_S``, exceeds
    ``coaxial_force_limit_n``. The plugin attaches its joints inside PhysX and
    leaves the stage's joints untouched, so the face is the only thing on the
    stage that moves with the cups' stretch. is_moving()
    still reports True for grab_delay_s after grab() is commanded, matching
    the time a real epick takes to build pressure, independent of whatever
    the gripper prim itself reports at that instant. Every interface call
    runs on the sim thread."""

    def __init__(
        self,
        sim: SimManager,
        name: str,
        gripper_prim_path: str,
        tool_prim_path: str,
        parent_prim_path: str,
        interface: Any,
        grab_delay_ms: float = DEFAULT_GRAB_DELAY_MS,
        release_delay_ms: float = DEFAULT_RELEASE_DELAY_MS,
        *,
        compliance: CupCompliance | None = None,
        coaxial_force_limit_n: float = 0.0,
        physics_dt: float = 1.0 / 60.0,
        cup_points_tool_m: Sequence[Vec3] = (),
        pose_reader: PoseReader | None = None,
        prop_half_dims: Callable[[], Mapping[str, Vec3]] | None = None,
        max_grip_distance_m: float = DEFAULT_MAX_GRIP_DISTANCE_MM / 1000.0,
        clearance_offset_m: float = CUP_APPROACH_GAP_MM / 1000.0,
    ) -> None:
        self._sim = sim
        self._name = name
        self._gripper_prim_path = gripper_prim_path
        self._tool_prim_path = tool_prim_path
        self.parent_prim_path = parent_prim_path
        self._interface = interface
        self._grab_delay_s = float(grab_delay_ms) / 1000.0
        self.release_delay_s = float(release_delay_ms) / 1000.0
        # whether the cup was last COMMANDED to take hold, which is not the
        # same as whether it is holding anything
        self._engaged = False
        # the coaxial monitor. A limit of 0, no compliance, no cup points, no
        # pose reader or no prop dimensions turns it off, and the sim manager
        # passes the world's own physics step (the default here is the world
        # config's default)
        self._pose_reader = pose_reader
        self._prop_half_dims = prop_half_dims
        self._compliance = compliance
        self._coaxial_force_limit_n = float(coaxial_force_limit_n)
        self._cup_points_tool_m: tuple[Vec3, ...] = tuple(cup_points_tool_m)
        self._max_grip_distance_m = float(max_grip_distance_m)
        self._clearance_offset_m = float(clearance_offset_m)
        self._load_window = LoadWindow(COAXIAL_LOAD_WINDOW_S, physics_dt)
        self._peak_coaxial_load_n = 0.0
        self._released_load_n: float | None = None
        # where the plugin leaves the held face at rest below each cup, read
        # before the grab from the box under the cups (cup_contact); None when
        # no registered box sat there, and the rest is then the cup plane
        self._cup_rest_offsets_m: tuple[float, ...] | None = None
        # True from the first step a hold is read until it ends, so the
        # monitor logs once per hold that it is watching
        self._hold_active = False
        # the monitor's state as HoldLoad reports it. Off-switches are fixed at
        # construction; stopped is set once it has raised, after logging the
        # traceback, so it stays off for the rest of the run rather than fail
        # every step
        monitor_on = (
            pose_reader is not None
            and prop_half_dims is not None
            and self._coaxial_force_limit_n > 0.0
            and compliance is not None
            and bool(self._cup_points_tool_m)
        )
        self._monitor_state = "idle" if monitor_on else "off"
        # one warning per hold the monitor cannot read: the held object has no
        # box dimensions registered, or no cup sits over its face
        self._unreadable_hold_logged = False

    @property
    def coaxial_callback_name(self) -> str:
        """The physics callback the sim manager registers for
        ``_on_physics_step``, and release() removes."""
        return f"{self._name}_coaxial_load"

    def _on_physics_step(self, step_size: float) -> None:
        """Sim thread, every physics step: the coaxial monitor.

        Off (return at once) when the limit is 0, there is no compliance, no
        cup points, no pose reader or no prop dimensions, or the monitor has
        raised before (it logs once and stays off). Otherwise read the
        gripper's status and gripped objects straight off the interface (no
        ``_sim.run`` here, the callback is already on the sim thread).
        Anything but Closed with an object clears the window and returns; the
        peak stays until the next grab().

        Holding: read the tool's and the held box's world poses off the stage
        (``begin_step`` first), pose the box in the tool frame and, for each
        cup, find the point on the box's near face straight under it
        (``cup_contact``). That point's distance below the cup plane, less
        where the plugin left the face at rest (``_cup_rest_offsets_m``, read
        before the grab), is the cup's stretch; ``cup_pull_load_n`` turns it
        into newtons past the dead band, and the step's load is the most
        loaded cup. A cup past the face's edge is skipped. Raise the peak,
        push the load through the window, and once the window is filled and
        its mean exceeds the limit call ``open_gripper`` on the interface,
        record the mean as ``_released_load_n``, log one warning naming the
        objects, the load, the limit and the window, and clear the window.
        A held object with no registered box dimensions, or one no cup sits
        over, is not watched: the state reads idle and one warning says why.
        """
        if (
            self._monitor_state in ("off", "stopped")
            or self._pose_reader is None
            or self._prop_half_dims is None
        ):
            return
        try:
            self._monitor_step(self._pose_reader, self._prop_half_dims())
        except Exception:
            self._monitor_state = "stopped"
            LOGGER.exception(
                "vacuum %r coaxial monitor stopped after an error; the hold is no longer "
                "released on load for the rest of this run",
                self._name,
            )

    def _monitor_step(self, pose_reader: PoseReader, half_dims_by_path: Mapping[str, Vec3]) -> None:
        """One step of the monitor, on the sim thread, with the off-switches
        already checked by ``_on_physics_step``."""
        status = status_name(self._interface.get_gripper_status(self._gripper_prim_path))
        objects = self._interface.get_gripped_objects(self._gripper_prim_path)
        if status != "Closed" or not objects:
            self._hold_active = False
            self._unreadable_hold_logged = False
            self._monitor_state = "idle"
            self._load_window.reset()
            return

        gripped = sorted(str(path) for path in objects)
        # this rig grips one box: only the first gripped object is monitored
        held_path = gripped[0]
        half_dims_m = half_dims_by_path.get(held_path)
        if half_dims_m is None:
            self._unreadable_hold(f"{held_path} has no box dimensions registered")
            return

        pose_reader.begin_step()
        tool_pos, tool_quat = pose_reader.world_pose(self._tool_prim_path)
        box_pos, box_quat = pose_reader.world_pose(held_path)
        box_pos_tool, box_quat_tool = pose_in_frame(tool_pos, tool_quat, box_pos, box_quat)
        step_load_n = self._most_loaded_cup_n(box_pos_tool, box_quat_tool, half_dims_m)
        if step_load_n is None:
            self._unreadable_hold(f"no cup sits over {held_path}'s face")
            return
        if not self._hold_active:
            self._hold_active = True
            self._monitor_state = "armed"
            rest_mm = (
                "the cup plane"
                if self._cup_rest_offsets_m is None
                else ", ".join(f"{offset * 1000.0:.1f}" for offset in self._cup_rest_offsets_m)
                + " mm below the cups"
            )
            LOGGER.info(
                "vacuum %r coaxial monitor armed on %s, face at rest %s",
                self._name,
                ", ".join(gripped),
                rest_mm,
            )

        self._peak_coaxial_load_n = max(self._peak_coaxial_load_n, step_load_n)
        mean_n = self._load_window.push(step_load_n)
        if not self._load_window.filled or mean_n <= self._coaxial_force_limit_n:
            return

        self._interface.open_gripper(self._gripper_prim_path)
        self._released_load_n = mean_n
        LOGGER.warning(
            "vacuum %r released %s: %.1f N per cup over %.1f N for %.2f s",
            self._name,
            ", ".join(gripped),
            mean_n,
            self._coaxial_force_limit_n,
            COAXIAL_LOAD_WINDOW_S,
        )
        self._hold_active = False
        self._monitor_state = "idle"
        self._load_window.reset()

    def _unreadable_hold(self, reason: str) -> None:
        """A hold the monitor cannot turn into a load: idle, window cleared,
        one warning per hold."""
        self._monitor_state = "idle"
        self._load_window.reset()
        if not self._unreadable_hold_logged:
            self._unreadable_hold_logged = True
            LOGGER.warning(
                "vacuum %r holds a load the coaxial monitor cannot read (%s); it is not "
                "watched for release",
                self._name,
                reason,
            )

    def _most_loaded_cup_n(
        self, box_pos_tool_m: Vec3, box_quat_tool: Quat, half_dims_m: Vec3
    ) -> float | None:
        """The pull on the most stretched cup over the held box's face, or None
        when no cup sits over it."""
        assert self._compliance is not None
        load_n: float | None = None
        for index, cup_point in enumerate(self._cup_points_tool_m):
            contact = cup_contact(cup_point, box_pos_tool_m, box_quat_tool, half_dims_m)
            if not contact.inside_face:
                continue
            rest_m = 0.0 if self._cup_rest_offsets_m is None else self._cup_rest_offsets_m[index]
            cup_load_n = cup_pull_load_n(contact.gap_m - rest_m, self._compliance)
            load_n = cup_load_n if load_n is None else max(load_n, cup_load_n)
        return load_n

    def _rest_offsets_before_close(self) -> tuple[float, ...] | None:
        """Sim thread, right before close_gripper: the gap from each cup to the
        face of the box under the cups, turned into where the plugin will
        leave that face at rest (``cup_contact.rest_offsets_m``). None when the
        monitor is off or no registered box sits under every cup within the
        grip distance; the nearest box wins when several do. An error here is
        logged and leaves the rest at the cup plane rather than block the
        grab."""
        if (
            self._monitor_state == "off"
            or self._pose_reader is None
            or self._prop_half_dims is None
        ):
            return None
        try:
            self._pose_reader.begin_step()
            tool_pos, tool_quat = self._pose_reader.world_pose(self._tool_prim_path)
            best_gaps: list[float] | None = None
            for prim_path, half_dims_m in self._prop_half_dims().items():
                box_pos, box_quat = self._pose_reader.world_pose(prim_path)
                box_pos_tool, box_quat_tool = pose_in_frame(tool_pos, tool_quat, box_pos, box_quat)
                gaps = cup_gaps_before_grab(
                    self._cup_points_tool_m,
                    box_pos_tool,
                    box_quat_tool,
                    half_dims_m,
                    self._max_grip_distance_m,
                )
                if gaps is not None and (best_gaps is None or sum(gaps) < sum(best_gaps)):
                    best_gaps = gaps
        except Exception:
            LOGGER.exception(
                "vacuum %r could not read the box under the cups before the grab; the "
                "coaxial monitor takes the face's rest at the cup plane",
                self._name,
            )
            return None
        if best_gaps is None:
            return None
        return tuple(rest_offsets_m(best_gaps, self._clearance_offset_m))

    def hold_load(self) -> HoldLoad:
        """Read from any thread: three plain floats the sim thread writes."""
        return HoldLoad(
            coaxial_load_n=self._load_window.mean_n,
            peak_coaxial_load_n=self._peak_coaxial_load_n,
            released_load_n=self._released_load_n,
            monitor=self._monitor_state,
        )

    @property
    def tool_prim_path(self) -> str:
        """The tool geometry's prim, for diagnostics that hang things off it."""
        return self._tool_prim_path

    def grab(self) -> None:
        self._engaged = True
        self._start_grab_window()
        self._peak_coaxial_load_n = 0.0
        self._released_load_n = None
        self._load_window.reset()
        self._hold_active = False
        self._unreadable_hold_logged = False

        def _grab() -> None:
            self._cup_rest_offsets_m = self._rest_offsets_before_close()
            accepted = self._interface.close_gripper(self._gripper_prim_path)
            if not accepted:
                LOGGER.warning(
                    "vacuum %r close_gripper refused for %r", self._name, self._gripper_prim_path
                )

        self._sim.run(_grab)

    def open(self) -> None:
        self._engaged = False
        self._clear_grab_window()
        self._hold_active = False
        self._load_window.reset()

        def _open() -> None:
            self._interface.open_gripper(self._gripper_prim_path)

        self._sim.run(_open)

    def stop(self) -> None:
        """The real driver's stop sets GTO 0, which halts vacuum regulation
        and drops whatever the cup held, so it is an open() here too."""
        self.open()

    def is_engaged(self) -> bool:
        return self._engaged

    def gripper_status(self) -> tuple[str, list[str]]:
        def _status() -> tuple[str, list[str]]:
            status = status_name(self._interface.get_gripper_status(self._gripper_prim_path))
            objects = [str(p) for p in self._interface.get_gripped_objects(self._gripper_prim_path)]
            return status, objects

        return self._sim.run(_status)

    def is_holding(self) -> bool:
        """False the instant a load breaks the hold past the prim's force
        limits, with nothing commanded from here: the next status read finds
        the gripper already back to Open with no gripped objects."""
        status, objects = self.gripper_status()
        return status == "Closed" and len(objects) > 0

    def poll_state(self) -> tuple[bool, bool]:
        return self.is_moving(), self.is_holding()

    def dof_names(self) -> list[str]:
        """A vacuum tool adds no articulated joint of its own."""
        return []

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        def _poses() -> dict[str, tuple[Vec3, Quat]]:
            out: dict[str, tuple[Vec3, Quat]] = {}
            if self.parent_prim_path:
                pos, quat = self._sim._isaac.SingleXFormPrim(self.parent_prim_path).get_world_pose()
                out["parent"] = (
                    (float(pos[0]), float(pos[1]), float(pos[2])),
                    (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
            pos, quat = self._sim._isaac.SingleXFormPrim(self._tool_prim_path).get_world_pose()
            out["tool"] = (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
            return out

        return self._sim.run(_poses)

    def post_reset(self) -> None:
        """A world reset reopens the gripper and puts every prop back where it
        spawned, so a cup engaged before the reset re-commands close_gripper,
        which retries against whatever is under it now. An open cup has
        nothing to redo."""
        if not self._engaged:
            return

        def _redo() -> None:
            self._interface.close_gripper(self._gripper_prim_path)

        self._sim.run(_redo)

    def release(self) -> None:
        """Drop the post-reset hook create_vacuum_gripper registered under
        this component's name, and the coaxial monitor's physics callback.
        The prim stays attached to the arm."""
        self._sim.unregister_post_reset(self._name)

        def _remove_callback() -> None:
            name = self.coaxial_callback_name
            if self._sim.world.physics_callback_exists(name):
                self._sim.world.remove_physics_callback(name)

        self._sim.run(_remove_callback)


class MockVacuumHandle(VacuumGripperHandle):
    """attrs["mock_attach_prop"] names the prop the cup finds under it.
    Unset means nothing is there, so grab() leaves is_holding() False - the
    mock counterpart to a real grab finding no payload within reach. The
    weld itself is instant, but is_moving() reports True for grab_delay_ms
    after grab() is commanded, matching IsaacVacuumHandle."""

    def __init__(self, name: str, attrs: dict[str, Any], arm: ArmHandle) -> None:
        self.name = name
        self._arm = arm
        self._attach_prop: str | None = attrs.get("mock_attach_prop")
        self.tcp_offset_m = float(attrs.get("tcp_offset_m", EPICK["tcp_offset_m"]))
        self._grab_delay_s = float(attrs.get("grab_delay_ms", DEFAULT_GRAB_DELAY_MS)) / 1000.0
        self._holding = False
        self._engaged = False

    def grab(self) -> None:
        self._engaged = True
        self._start_grab_window()
        self._holding = self._attach_prop is not None

    def open(self) -> None:
        self._engaged = False
        self._clear_grab_window()
        self._holding = False

    def stop(self) -> None:
        return None

    def is_engaged(self) -> bool:
        return self._engaged

    def is_holding(self) -> bool:
        return self._holding

    def gripper_status(self) -> tuple[str, list[str]]:
        """Closed on the prop it was told is there, Closing while engaged on
        nothing (the automatic mode retrying), Open otherwise."""
        if self._holding and self._attach_prop is not None:
            return "Closed", [f"/World/{self._attach_prop}"]
        if self._engaged:
            return "Closing", []
        return "Open", []

    def poll_state(self) -> tuple[bool, bool]:
        return self.is_moving(), self._holding

    def dof_names(self) -> list[str]:
        return []

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        """Synthetic: the mount link at the origin, the tool straddling the
        TCP along +Z, in the spirit of MockGripperHandle.link_world_poses."""
        identity: Quat = (1.0, 0.0, 0.0, 0.0)
        return {
            "parent": ((0.0, 0.0, 0.0), identity),
            "tool": ((0.0, 0.0, self.tcp_offset_m), identity),
        }

    def post_reset(self) -> None:
        return None

    def release(self) -> None:
        return None
