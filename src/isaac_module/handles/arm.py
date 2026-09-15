from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from viam.logging import getLogger

from ..asset_catalog import KNOWN_ASSETS
from ..errors import PrimNotFoundError
from ..prim_paths import prim_name
from ..spatial import (
    Quat,
    Vec3,
    _as_quat,
    compose_pose,
    pose_in_frame,
    spawn_orientation,
    viam_base_frame,
)

if TYPE_CHECKING:
    from ..sim_manager import SimManager

LOGGER = getLogger(__name__)


def resolve_joint_indices(
    dof_names: Sequence[str], joint_names: Sequence[str] | None
) -> list[int] | None:
    """Map an asset's declared arm joint names onto the articulation's PhysX
    dof order, by name rather than position (attaching a
    gripper later can add/reorder dofs, so a positional slice would silently
    pick up the wrong joints). Returns None when the asset declares no joint
    names, meaning "all dofs, in PhysX order"."""
    if joint_names is None:
        return None
    indices = []
    missing = []
    for name in joint_names:
        try:
            indices.append(dof_names.index(name))
        except ValueError:
            missing.append(name)
    if missing:
        raise ValueError(
            f"joint(s) not found in articulation: {missing}, actual dof_names: {list(dof_names)}"
        )
    return indices


# The two "arrived" notions - velocity-based IsMoving
# and position-based move completion - share these constants and one
# predicate on every backend.
# |joint velocity| at or below this counts as still. A 12-DOF PhysX articulation
# at 120 Hz idles with ~1e-3 rad/s of residual jitter, so 1e-3
# never settled. 1e-2 rad/s (0.6 deg/s) is still far below any real motion.
VEL_EPS_RAD_S = 1e-2
SETTLE_TOL_RAD = math.radians(0.5)  # commanded-vs-measured gap that counts as arrived
SETTLE_WINDOW_STEPS = 5  # consecutive physics steps the predicate must hold
# wait_for_settle's wall-clock guard: timeout_s * SETTLE_GUARD_MULTIPLE +
# SETTLE_GUARD_SLACK_S, generous enough that a paused sim can't hang forever.
SETTLE_GUARD_MULTIPLE = 4
SETTLE_GUARD_SLACK_S = 5.0
# A blocked arm under contact vibrates above VEL_EPS_RAD_S and never reads
# still (observed over 30 s of pushing into a block), so a stall is also
# declared when the worst joint error stops improving for this many steps
# (1 s at physics_dt 1/120) by at least STALL_PROGRESS_EPS_RAD.
STALL_NO_PROGRESS_STEPS = 120
STALL_PROGRESS_EPS_RAD = math.radians(0.1)


class SettleOutcome(str, Enum):
    """Result of ArmHandle.wait_for_settle."""

    REACHED = "reached"  # every named joint within tolerance for SETTLE_WINDOW_STEPS steps
    STALLED = "stalled"  # still (|v| <= VEL_EPS_RAD_S) for the window with a joint off target
    TIMED_OUT = "timed_out"  # the sim-time deadline passed with joints still converging


class ArmHandle:
    """The interface the arm component model talks to. Every public method
    is safe to call from any thread."""

    def dof_names(self) -> list[str]:
        """Names of the arm's named joints, in the asset's declared order
        (all DOFs, in PhysX order, when the asset declares none)."""
        raise NotImplementedError

    def all_dof_names(self) -> list[str]:
        """Every DOF of the articulation in PhysX order - the arm's joints
        plus anything attached under it (a gripper). The GPU checklist's
        `len == 12`. The mock includes its padding dofs."""
        raise NotImplementedError

    def joint_state(self) -> list[dict[str, Any]]:
        """Per DOF of the whole articulation, in PhysX order: ``name``,
        ``position`` and ``velocity`` (rad, rad/s), the drive ``target`` the
        physics is actually holding (rad, None when unreadable) and ``named``
        (True for the arm's own joints). The diagnostic that separates "wrong
        target" from "physics fought the target"."""
        raise NotImplementedError

    def get_joint_positions(self) -> list[float]:  # radians
        """Positions of the arm's named joints, in the asset's declared
        order (all DOFs when the asset declares none)."""
        raise NotImplementedError

    def set_joint_targets(self, positions: list[float], max_vel_rad_s: float | None = None) -> None:
        """Targets for the arm's named joints, in the asset's declared
        order (all DOFs when the asset declares none). ``max_vel_rad_s`` caps
        every named joint's speed for this move (MoveOptions
        max_vel_degs_per_sec, converted by the model). None = the drive's
        own limit."""
        raise NotImplementedError

    def home(self, positions: list[float]) -> None:
        """Place the arm's named joints AT ``positions`` (radians) without
        driving there, and hold them.

        A UR asset's own default pose is every joint at zero, which is the arm
        fully extended horizontally. In a cell with anything tall in front of
        it that is the arm lying across its own workspace, and the first
        commanded move sweeps whatever is there aside before any of it can be
        picked up. Spawning at a folded pose is the fix, and it has to be a
        state write rather than a move, since there is nothing to plan around
        yet and a driven sweep is the very thing being avoided."""
        raise NotImplementedError

    def is_moving(self) -> bool:
        """True while any named joint's |velocity| > VEL_EPS_RAD_S OR any
        |target - measured| > SETTLE_TOL_RAD. A stalled arm that
        never reached its target therefore keeps reporting True."""
        raise NotImplementedError

    def wait_for_settle(
        self, timeout_s: float, tolerance_rad: float = SETTLE_TOL_RAD
    ) -> SettleOutcome:
        """Block the calling (non-sim) thread until the last commanded targets
        are REACHED (within ``tolerance_rad`` for SETTLE_WINDOW_STEPS
        consecutive steps), the arm STALLED (still for the window, outside
        tolerance), or ``timeout_s`` of SIM time has elapsed (TIMED_OUT).
        Backends: Isaac via a physics-step callback + threading.Event, the
        mock via its interpolation clock. The model never polls wall clock."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def post_reset(self) -> None:
        """Re-apply anything a world.reset() undoes (solver
        iteration count, controller gains) and re-command the last targets,
        so a reset mid-move holds position instead of teleporting to the
        default pose. No-op by default (the mock has no such
        state)."""
        return None

    def release(self) -> None:
        """Called by SimManager.release_handle when the owning component
        closes. Isaac backends drop the world.scene registry entry
        (registry_only - the prim stays) and any physics callbacks, so a
        later create_arm for the same name can re-attach."""
        return None

    def get_end_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """((x,y,z) meters, (w,x,y,z) quaternion) of the end effector, in
        Viam's arm frame - the Isaac root un-rotated by the asset's
        base_frame_correction, if any."""
        raise NotImplementedError

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        """((x,y,z) meters, (w,x,y,z) quaternion) world pose of an arbitrary
        prim on the stage."""
        raise NotImplementedError


class IsaacArmHandle(ArmHandle):
    def __init__(
        self,
        sim: SimManager,
        articulation,
        ee_prim,
        joint_names: Sequence[str] | None = None,
        base_correction: Quat = (1.0, 0.0, 0.0, 0.0),
        prim_path: str = "",
    ) -> None:
        self._sim = sim
        self._art = articulation
        self._ee = ee_prim
        self._joint_names = joint_names
        self._base_correction: Quat = base_correction
        self._prim_path = prim_path
        self._dof_names: list[str] = list(articulation.dof_names)
        LOGGER.info(
            "arm %r articulation dof_names: %s",
            getattr(articulation, "name", ""),
            self._dof_names,
        )
        self._joint_indices: list[int] | None = resolve_joint_indices(self._dof_names, joint_names)
        # last commanded targets for the named joints (the second arrived term)
        self._targets: list[float] | None = None
        # the previous per-joint velocity cap, restored when a move
        # arrives with max_vel_rad_s=None after one that set it.
        self._saved_max_joint_velocities: Any | None = None
        # snapshotted by create_arm's factory, re-applied by
        # post_reset after a world.reset() undoes them.
        self._solver_iterations: int | None = None
        self._gains: Any | None = None
        # the in-flight wait_for_settle's event/outcome, so stop() can
        # short-circuit it (None when no wait is in flight).
        self._active_settle: dict[str, Any] | None = None
        self._active_settle_lock = threading.Lock()

    def refresh_dofs(self) -> None:
        """Re-read dof_names and re-resolve the named-joint indices after the
        articulation's topology changed - a gripper was attached under it.
        Raises ValueError if the named joints are no longer all
        present."""
        previous_count = len(self._dof_names)
        if not getattr(self._art, "handles_initialized", True):
            # a wrapper registered after the world's first reset may not have
            # been initialized by the scene yet
            self._art.initialize()
        names_now = list(self._art.dof_names)
        initialize = getattr(self._art, "initialize", None)
        if len(names_now) != previous_count and initialize is not None:
            # the wrapper captured its default joint state before the topology
            # changed (observed: "setDofActuationForces expected 12, received
            # 6" at every later reset). Re-initialize so it is re-captured
            initialize()
            names_now = list(self._art.dof_names)
        self._dof_names = names_now
        LOGGER.info("arm %r articulation dof_names now: %s", self._art.name, self._dof_names)
        self._joint_indices = resolve_joint_indices(self._dof_names, self._joint_names)
        if self._gains is None:
            # a fresh wrapper (replace_articulation) has no snapshot yet: take
            # it from the new topology and re-apply the solver count
            if self._solver_iterations is not None:
                set_iterations = getattr(self._art, "set_solver_position_iteration_count", None)
                if set_iterations is not None:
                    set_iterations(self._solver_iterations)
            self._gains = self._art.get_articulation_controller().get_gains()

    def replace_articulation(self, articulation: Any) -> None:
        """Swap in a fresh SingleArticulation wrapper before a reset that
        changes the articulation's topology (a gripper joining it). The old
        wrapper's default joint state and gains snapshot are sized for the old
        DOF count, and the scene's post_reset would push them into the new
        articulation and fail ("Failed to set DOF actuation forces"). The
        gains snapshot is dropped here and retaken by refresh_dofs()."""
        self._art = articulation
        self._gains = None

    def all_dof_names(self) -> list[str]:
        return list(self._dof_names)

    def joint_state(self) -> list[dict[str, Any]]:
        def _state() -> list[dict[str, Any]]:
            positions = [float(v) for v in self._art.get_joint_positions()]
            velocities = [float(v) for v in self._art.get_joint_velocities()]
            targets: list[float | None] = [None] * len(positions)
            view = getattr(self._art, "_articulation_view", None)
            try:
                applied = view.get_applied_actions() if view is not None else None
                if applied is not None and applied.joint_positions is not None:
                    row = applied.joint_positions
                    row = row[0] if len(getattr(row, "shape", ())) == 2 else row
                    targets = [float(v) for v in row]
            except Exception:  # a diagnostic read, any failure is logged, not fatal
                LOGGER.exception("could not read the applied drive targets")
            named = set(self._joint_indices or range(len(positions)))
            return [
                {
                    "name": name,
                    "position": positions[i],
                    "velocity": velocities[i],
                    "target": targets[i],
                    "named": i in named,
                }
                for i, name in enumerate(self._dof_names)
            ]

        return self._sim.run(_state)

    def dof_names(self) -> list[str]:
        if self._joint_indices is None:
            return list(self._dof_names)
        return [self._dof_names[i] for i in self._joint_indices]

    def get_joint_positions(self) -> list[float]:
        def _get():
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            return [float(v) for v in positions]

        return self._sim.run(_get)

    def set_joint_targets(self, positions: list[float], max_vel_rad_s: float | None = None) -> None:
        self._sim.run(lambda: self._apply_joint_targets_on_sim_thread(positions, max_vel_rad_s))

    def home(self, positions: list[float]) -> None:
        def _place() -> None:
            values = np.array(positions, dtype=float)
            set_positions = getattr(self._art, "set_joint_positions", None)
            if set_positions is not None:
                set_positions(values, joint_indices=self._joint_indices)
            set_velocities = getattr(self._art, "set_joint_velocities", None)
            if set_velocities is not None:
                set_velocities(np.zeros(len(positions), dtype=float), self._joint_indices)
            # holding the same values keeps gravity from folding the arm back
            # down, and records them as the targets post_reset restores
            self._apply_joint_targets_on_sim_thread(positions, None)

        self._sim.run(_place, allow_during_initialization=True)

    def _apply_joint_targets_on_sim_thread(
        self, positions: list[float], max_vel_rad_s: float | None
    ) -> None:
        """Sim-thread body shared by set_joint_targets and stop's hold-in-place."""
        self._apply_velocity_cap(max_vel_rad_s)
        action = self._sim._isaac.ArticulationAction(
            joint_positions=np.array(positions, dtype=float),
            joint_indices=self._joint_indices,
        )
        self._art.apply_action(action)
        self._targets = list(positions)

    def _apply_velocity_cap(self, max_vel_rad_s: float | None) -> None:
        """Cap the named joints' max velocity for this move. SingleArticulation
        has no set_max_joint_velocities in 5.0; the batched articulation view
        it wraps does, spelled set_max_joint_velocities / get_joint_max_velocities
        (the getter does not mirror the setter's name). Restores the pre-cap
        values read via get_joint_max_velocities when a later move passes None."""
        view = getattr(self._art, "_articulation_view", None)
        set_max = getattr(view, "set_max_joint_velocities", None) if view is not None else None
        if set_max is None:
            LOGGER.warning(
                "articulation view has no set_max_joint_velocities; velocity cap dropped"
            )
            return
        if max_vel_rad_s is not None:
            if self._saved_max_joint_velocities is None:
                get_max = getattr(view, "get_joint_max_velocities", None)
                if get_max is not None:
                    saved = get_max(joint_indices=self._joint_indices)
                    self._saved_max_joint_velocities = (
                        saved[0] if len(getattr(saved, "shape", ())) == 2 else saved
                    )
            count = (
                len(self._joint_indices)
                if self._joint_indices is not None
                else len(self._dof_names)
            )
            set_max(np.array([max_vel_rad_s] * count), joint_indices=self._joint_indices)
        elif self._saved_max_joint_velocities is not None:
            set_max(self._saved_max_joint_velocities, joint_indices=self._joint_indices)
            self._saved_max_joint_velocities = None

    def is_moving(self) -> bool:
        def _check():
            vels = self._art.get_joint_velocities(joint_indices=self._joint_indices)
            is_moving = vels is not None and bool(max(abs(float(v)) for v in vels) > VEL_EPS_RAD_S)
            if is_moving or self._targets is None:
                return is_moving
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            return any(
                abs(float(p) - t) > SETTLE_TOL_RAD
                for p, t in zip(positions, self._targets, strict=True)
            )

        return self._sim.run(_check)

    def _settle_callback_name(self) -> str:
        return f"{getattr(self._art, 'name', '')}_settle"

    def wait_for_settle(
        self, timeout_s: float, tolerance_rad: float = SETTLE_TOL_RAD
    ) -> SettleOutcome:
        # A physics-step callback evaluates the settle predicate on
        # the sim thread every step and signals a threading.Event. This
        # (caller) thread only waits on it, so no wall-clock polling.
        if self._targets is None:
            return SettleOutcome.REACHED

        targets = list(self._targets)
        joint_indices = self._joint_indices
        event = threading.Event()
        outcome: list[SettleOutcome] = []
        counters: dict[str, float] = {
            "within": 0,
            "still_off_target": 0,
            "sim_time": 0.0,
            "max_speed": 0.0,
            "best_error": math.inf,
            "no_progress": 0,
        }
        settle_state: dict[str, Any] = {"event": event, "outcome": outcome}

        def _on_step(step_size: float) -> None:
            # sim thread only: touch the counters/Event, never self._sim.run.
            if event.is_set():
                return
            velocities = self._art.get_joint_velocities(joint_indices=joint_indices)
            max_speed = max((abs(float(v)) for v in velocities), default=0.0)
            counters["max_speed"] = max(counters["max_speed"], max_speed)
            is_still = max_speed <= VEL_EPS_RAD_S
            positions = self._art.get_joint_positions(joint_indices=joint_indices)
            errors = [abs(float(p) - t) for p, t in zip(positions, targets, strict=True)]
            worst_error = max(errors, default=0.0)
            is_within = worst_error <= tolerance_rad
            # REACHED is a position criterion held over the window - holding
            # within tolerance for SETTLE_WINDOW_STEPS steps IS "settled".
            # Velocity only decides stalls: PhysX never reads exactly still.
            counters["within"] = counters["within"] + 1 if is_within else 0
            counters["still_off_target"] = (
                counters["still_off_target"] + 1 if (is_still and not is_within) else 0
            )
            # no-progress stall: the worst error has not improved for a while
            if worst_error < counters["best_error"] - STALL_PROGRESS_EPS_RAD:
                counters["best_error"] = worst_error
                counters["no_progress"] = 0
            else:
                counters["no_progress"] += 1

            if counters["within"] >= SETTLE_WINDOW_STEPS:
                outcome.append(SettleOutcome.REACHED)
                event.set()
                return
            if counters["still_off_target"] >= SETTLE_WINDOW_STEPS or (
                not is_within and counters["no_progress"] >= STALL_NO_PROGRESS_STEPS
            ):
                outcome.append(SettleOutcome.STALLED)
                event.set()
                return

            counters["sim_time"] += step_size
            if counters["sim_time"] >= timeout_s:
                LOGGER.warning(
                    "arm %r settle timed out after %.2fs sim time: within tolerance=%s, "
                    "still=%s, max |v| seen=%.4f rad/s (VEL_EPS %.4f)",
                    getattr(self._art, "name", ""),
                    counters["sim_time"],
                    is_within,
                    is_still,
                    counters["max_speed"],
                    VEL_EPS_RAD_S,
                )
                outcome.append(SettleOutcome.TIMED_OUT)
                event.set()

        callback_name = self._settle_callback_name()

        def _register() -> None:
            self._sim.world.add_physics_callback(callback_name, _on_step)

        def _remove() -> None:
            if self._sim.world.physics_callback_exists(callback_name):
                self._sim.world.remove_physics_callback(callback_name)

        with self._active_settle_lock:
            self._active_settle = settle_state
        self._sim.run(_register)
        try:
            # a generous wall-clock guard so a paused sim can't hang forever.
            wall_clock_guard_s = timeout_s * SETTLE_GUARD_MULTIPLE + SETTLE_GUARD_SLACK_S
            event_was_set = event.wait(timeout=wall_clock_guard_s)
            if not event_was_set:
                return SettleOutcome.TIMED_OUT
            return outcome[0]
        finally:
            with self._active_settle_lock:
                if self._active_settle is settle_state:
                    self._active_settle = None
            self._sim.run(_remove)

    def stop(self) -> None:
        def _hold() -> None:
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            current = [float(v) for v in positions]
            self._apply_joint_targets_on_sim_thread(current, None)

        self._sim.run(_hold, allow_during_initialization=True)
        # the new target IS the current position, so any in-flight
        # wait_for_settle should read as having reached it, not stalled.
        with self._active_settle_lock:
            active = self._active_settle
            if active is not None and not active["event"].is_set():
                active["outcome"].append(SettleOutcome.REACHED)
                active["event"].set()

    def get_end_pose(self):
        if self._ee is None:
            raise NotImplementedError(
                "set end_effector_prim in the arm config to report end position"
            )

        def _pose():
            root_pos, root_quat = self._art.get_world_pose()
            pos, quat = self._ee.get_world_pose()
            root_pos_t = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
            root_quat_t = (
                float(root_quat[0]),
                float(root_quat[1]),
                float(root_quat[2]),
                float(root_quat[3]),
            )
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            base_pos, base_quat = viam_base_frame(root_pos_t, root_quat_t, self._base_correction)
            return pose_in_frame(base_pos, base_quat, pos_t, quat_t)

        return self._sim.run(_pose)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        def _pose() -> tuple[Vec3, Quat]:
            self._sim._require_prim(prim_path)
            pos, quat = self._sim._isaac.SingleXFormPrim(prim_path).get_world_pose()
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            return pos_t, quat_t

        return self._sim.run(_pose)

    def post_reset(self) -> None:
        """A world.reset() resets the solver iteration count
        and controller gains to the prim's authored defaults, and teleports
        the articulation to its default pose - re-apply the snapshotted
        solver count/gains, teleport the joints back to the last targets
        (zero velocity), then re-command those targets so the arm holds
        position across the reset instead of sweeping back unplanned."""

        def _redo() -> None:
            set_iterations = getattr(self._art, "set_solver_position_iteration_count", None)
            if set_iterations is not None and self._solver_iterations is not None:
                set_iterations(self._solver_iterations)
            if self._gains is not None:
                kps, kds = self._gains
                self._art.get_articulation_controller().set_gains(kps=kps, kds=kds)
            if self._targets is not None:
                positions = np.array(self._targets, dtype=float)
                # State write BEFORE the hold action: the reset left the arm
                # at its default pose, and driving from there to the targets
                # is an unplanned full-speed sweep through the scene.
                set_positions = getattr(self._art, "set_joint_positions", None)
                if set_positions is not None:
                    set_positions(positions, joint_indices=self._joint_indices)
                set_velocities = getattr(self._art, "set_joint_velocities", None)
                if set_velocities is not None:
                    set_velocities(
                        np.zeros(len(self._targets), dtype=float),
                        joint_indices=self._joint_indices,
                    )
                action = self._sim._isaac.ArticulationAction(
                    joint_positions=positions,
                    joint_indices=self._joint_indices,
                )
                self._art.apply_action(action)

        self._sim.run(_redo)

    def release(self) -> None:
        """Remove the settle callback if one is registered and drop the
        scene-registry entry (registry_only - the prim stays), so a later
        create_arm for this name can re-attach."""

        def _release() -> None:
            world = self._sim.world
            callback_name = self._settle_callback_name()
            if world.physics_callback_exists(callback_name):
                world.remove_physics_callback(callback_name)
            name = getattr(self._art, "name", "")
            if world.scene.get_object(name) is not None:
                world.scene.remove_object(name, registry_only=True)

        self._sim.run(_release)


class MockArmHandle(ArmHandle):
    """Joints move linearly toward their targets at a fixed speed. Total dof
    count is mock_dof (default: the number of declared joint names, else 6).
    The arm's named joints are selected by index the same way the Isaac
    handle does, and any remaining dofs are padding
    that never moves."""

    SPEED = 1.0  # rad/s per joint
    STEP_S = 1.0 / 120.0  # the mock's "physics step" for wait_for_settle polling

    # the mock's end effector, fixed in Viam's arm frame (public,
    # deterministic value, unchanged by spawn pose or base_frame_correction).
    FIXED_LOCAL_EE: tuple[Vec3, Quat] = ((0.3, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))

    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        from ..spatial import to_vec3

        self.name = name
        # test knob ("stalled vs timed out"): when set, every move stops
        # after this fraction of its travel, like an arm blocked by an obstacle
        stall = attrs.get("mock_stall_fraction")
        self.mock_stall_fraction: float | None = None if stall is None else float(stall)
        self._speed = self.SPEED
        meta = KNOWN_ASSETS.get(str(attrs.get("asset", "")), {})
        joint_names: Sequence[str] | None = meta.get("joint_names")
        default_dof = len(joint_names) if joint_names else 6
        dof = int(attrs.get("mock_dof", default_dof))
        if joint_names:
            names = list(joint_names) + [f"mock_extra_{i}" for i in range(dof - len(joint_names))]
            self._joint_indices: list[int] | None = list(range(len(joint_names)))
        else:
            names = [f"mock_joint_{i}" for i in range(dof)]
            self._joint_indices = None
        self._dof_names = names
        self._lock = threading.Lock()
        self._start = [0.0] * dof
        self._target = [0.0] * dof
        self._t0 = time.monotonic()
        self.spawn_position = to_vec3(attrs.get("position"))
        self.spawn_orientation = spawn_orientation(attrs, meta)
        correction = meta.get("base_frame_correction")
        self._base_correction: Quat = (
            _as_quat(correction)
            if correction is not None
            else (
                1.0,
                0.0,
                0.0,
                0.0,
            )
        )
        self._prim_path = attrs.get("prim_path") or f"/World/{prim_name(name)}"

    def dof_names(self) -> list[str]:
        return list(self._dof_names)

    def all_dof_names(self) -> list[str]:
        return list(self._dof_names)

    def _selected(self) -> list[int]:
        if self._joint_indices is None:
            return list(range(len(self._dof_names)))
        return self._joint_indices

    def _travel_limit(self, delta: float) -> float:
        """How far a joint may travel toward its target this move: all the
        way, or mock_stall_fraction of it when the mock is told to stall."""
        if self.mock_stall_fraction is None:
            return abs(delta)
        return abs(delta) * self.mock_stall_fraction

    def _positions_at(self, now: float) -> list[float]:
        out = []
        dt = max(0.0, now - self._t0)
        for s, t in zip(self._start, self._target, strict=True):
            delta = t - s
            travel = min(self._speed * dt, self._travel_limit(delta))
            if travel >= abs(delta):
                out.append(t)
            else:
                out.append(s + math.copysign(travel, delta))
        return out

    def _velocities_at(self, now: float) -> list[float]:
        dt = max(0.0, now - self._t0)
        return [
            0.0 if self._speed * dt >= self._travel_limit(t - s) else self._speed
            for s, t in zip(self._start, self._target, strict=True)
        ]

    def get_all_joint_positions(self) -> list[float]:
        """Test-only accessor for the full (unselected) dof array."""
        with self._lock:
            return self._positions_at(time.monotonic())

    def joint_state(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            positions = self._positions_at(now)
            velocities = self._velocities_at(now)
            targets = list(self._target)
        named = set(self._selected())
        return [
            {
                "name": name,
                "position": positions[i],
                "velocity": velocities[i],
                "target": targets[i],
                "named": i in named,
            }
            for i, name in enumerate(self._dof_names)
        ]

    def get_joint_positions(self) -> list[float]:
        with self._lock:
            all_pos = self._positions_at(time.monotonic())
        return [all_pos[i] for i in self._selected()]

    def set_joint_targets(self, positions: list[float], max_vel_rad_s: float | None = None) -> None:
        with self._lock:
            now = time.monotonic()
            all_pos = self._positions_at(now)
            selected = self._selected()
            if len(positions) != len(selected):
                raise ValueError(f"expected {len(selected)} joint positions, got {len(positions)}")
            self._start = all_pos
            self._target = list(all_pos)
            for i, p in zip(selected, positions, strict=True):
                self._target[i] = p
            self._t0 = now
            self._speed = self.SPEED if max_vel_rad_s is None else min(self.SPEED, max_vel_rad_s)

    def home(self, positions: list[float]) -> None:
        with self._lock:
            selected = self._selected()
            if len(positions) != len(selected):
                raise ValueError(f"expected {len(selected)} joint positions, got {len(positions)}")
            placed = self._positions_at(time.monotonic())
            for i, p in zip(selected, positions, strict=True):
                placed[i] = p
            # start and target both at the home pose, so the arm is there now
            # rather than travelling toward it
            self._start = list(placed)
            self._target = list(placed)
            self._t0 = time.monotonic()

    def is_moving(self) -> bool:
        with self._lock:
            now = time.monotonic()
            pos = self._positions_at(now)
            vel = self._velocities_at(now)
        selected = self._selected()
        return any(
            abs(vel[i]) > VEL_EPS_RAD_S or abs(pos[i] - self._target[i]) > SETTLE_TOL_RAD
            for i in selected
        )

    def wait_for_settle(
        self, timeout_s: float, tolerance_rad: float = SETTLE_TOL_RAD
    ) -> SettleOutcome:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = time.monotonic()
                pos = self._positions_at(now)
                vel = self._velocities_at(now)
                target = list(self._target)
            selected = self._selected()
            is_within_tolerance = all(abs(pos[i] - target[i]) <= tolerance_rad for i in selected)
            is_still = all(abs(vel[i]) <= VEL_EPS_RAD_S for i in selected)
            # REACHED means settled, not merely close: is_moving() must read
            # False the instant this returns.
            if is_within_tolerance and is_still:
                return SettleOutcome.REACHED
            if is_still:
                return SettleOutcome.STALLED
            if now >= deadline:
                return SettleOutcome.TIMED_OUT
            time.sleep(self.STEP_S)

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            self._target = list(self._start)
            self._t0 = now

    def _ee_world_pose(self) -> tuple[Vec3, Quat]:
        """The mock's simulated Isaac root is (spawn_position,
        spawn_orientation) - already composed with base_frame_correction -
        so its end effector's world pose is FIXED_LOCAL_EE expressed in
        Viam's arm frame, then re-composed onto that rotated root."""
        base_pos, base_quat = viam_base_frame(
            self.spawn_position, self.spawn_orientation, self._base_correction
        )
        local_pos, local_quat = self.FIXED_LOCAL_EE
        return compose_pose(base_pos, base_quat, local_pos, local_quat)

    def get_end_pose(self):
        # a fixed, deterministic pose for testing, defined in Viam's arm
        # frame - it must not change with spawn_position/
        # spawn_orientation/base_frame_correction.
        base_pos, base_quat = viam_base_frame(
            self.spawn_position, self.spawn_orientation, self._base_correction
        )
        ee_pos, ee_quat = self._ee_world_pose()
        return pose_in_frame(base_pos, base_quat, ee_pos, ee_quat)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        ee_prim_path = f"{self._prim_path}/wrist_3_link"
        if prim_path != ee_prim_path:
            raise PrimNotFoundError(f"prim not found: {prim_path}")
        return self._ee_world_pose()
