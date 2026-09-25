from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
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
# a joint closing on its target faster than this is making progress whatever
# the error window says. Above the 0.05 rad/s (2.9 deg/s) jitter of a joint
# pinned against a block, which the no-progress rule exists to catch; far
# under the 20 to 60 deg/s a back-driven wrist recovered at on 2026-09-24,
# when the window ending at the bottom of its forty degree setback read as
# no progress and stopped the arm mid-recovery.
STALL_CONVERGING_SPEED_RAD_S = math.radians(6.0)
# how many consecutive physics steps an off-target arm must read still before
# it is a stall. A joint reversing direction dwells under VEL_EPS for a few
# steps as its velocity crosses zero (the 2026-09-24 wrist back-drives turned
# around inside 0.1 s and were called stalled at five), while a joint pinned
# against a block stays still for as long as the drive pushes. Half a second
# at 60 Hz: an order of magnitude over a reversal, a quarter of the
# no-progress window.
STALL_STILL_STEPS = 30


class SettleOutcome(str, Enum):
    """Result of ArmHandle.wait_for_settle."""

    REACHED = "reached"  # every named joint within tolerance for SETTLE_WINDOW_STEPS steps
    STALLED = (
        "stalled"  # still (|v| <= VEL_EPS_RAD_S) for STALL_STILL_STEPS with a joint off target
    )
    TIMED_OUT = "timed_out"  # the sim-time deadline passed with joints still converging


class _StepTracker:
    """The per-step arrival and stall rules every wait on the drive shares.

    ``step`` takes one physics step of joint state against the targets being
    waited on and returns the outcome once there is one, None while the wait
    goes on. ``reset`` starts the rules afresh for the next target, so a path
    followed point by point is judged by the same rules as a single wait."""

    def __init__(
        self, arm_name: str, timeout_s: float, tolerance_rad: float, window_steps: int
    ) -> None:
        self._arm_name = arm_name
        self._timeout_s = timeout_s
        self._tolerance_rad = tolerance_rad
        self._window_steps = window_steps
        self.reset()

    def reset(self) -> None:
        self._within = 0
        self._still_off_target = 0
        self._sim_time = 0.0
        self._max_speed = 0.0
        # the worst error at each of the last STALL_NO_PROGRESS_STEPS steps,
        # oldest first, so progress is judged over a window rather than
        # against the best value ever seen. On 2026-09-24 the base's swing
        # back-drove wrist 3 forty degrees with a box on the cups. The joint
        # was recovering at 60 degrees per second when its all-time best, two
        # seconds behind it, read as no progress and the arm was stopped a
        # third of the way through the recovery, on every carry from that
        # branch.
        self._recent_worst_errors: deque[float] = deque(maxlen=STALL_NO_PROGRESS_STEPS + 1)

    def step(
        self,
        positions: Sequence[float],
        velocities: Sequence[float],
        targets: Sequence[float],
        step_size: float,
    ) -> SettleOutcome | None:
        max_speed = max((abs(float(v)) for v in velocities), default=0.0)
        self._max_speed = max(self._max_speed, max_speed)
        is_still = max_speed <= VEL_EPS_RAD_S
        errors = [abs(float(p) - t) for p, t in zip(positions, targets, strict=True)]
        worst_error = max(errors, default=0.0)
        is_within = worst_error <= self._tolerance_rad
        # REACHED is a position criterion held over the window, so holding
        # within tolerance for window_steps steps IS "settled". Velocity only
        # decides stalls, because PhysX never reads exactly still.
        self._within = self._within + 1 if is_within else 0
        self._still_off_target = self._still_off_target + 1 if (is_still and not is_within) else 0
        # no-progress stall: the worst error is no better than it was
        # STALL_NO_PROGRESS_STEPS steps ago. A joint pinned against a block
        # vibrates above VEL_EPS and never reads still, and this is what
        # catches it. A joint losing ground and winning it back is making
        # progress over any window that ends in its recovery.
        self._recent_worst_errors.append(worst_error)
        # the fastest any off-target joint is closing on its target, where a
        # joint moving away or a joint at rest contributes nothing
        closing_speed = max(
            (
                float(v) * math.copysign(1.0, t - float(p))
                for p, v, t in zip(positions, velocities, targets, strict=True)
                if abs(float(p) - t) > self._tolerance_rad
            ),
            default=0.0,
        )
        no_progress = (
            len(self._recent_worst_errors) > STALL_NO_PROGRESS_STEPS
            and worst_error > self._recent_worst_errors[0] - STALL_PROGRESS_EPS_RAD
            and closing_speed < STALL_CONVERGING_SPEED_RAD_S
        )

        if self._within >= self._window_steps:
            return SettleOutcome.REACHED
        if self._still_off_target >= STALL_STILL_STEPS or (not is_within and no_progress):
            # the numbers the decision was made on, since a stall the arm
            # reports is otherwise indistinguishable from one it caused
            LOGGER.warning(
                "arm settle STALLED by %s after %d steps (%.2f s sim): worst error %.1f deg, "
                "error %d steps ago %.1f deg, max speed %.1f deg/s, closing speed %.1f deg/s, "
                "still-off-target run %d",
                "still" if self._still_off_target >= STALL_STILL_STEPS else "no-progress",
                len(self._recent_worst_errors),
                self._sim_time,
                math.degrees(worst_error),
                STALL_NO_PROGRESS_STEPS,
                math.degrees(self._recent_worst_errors[0]),
                math.degrees(max_speed),
                math.degrees(closing_speed),
                self._still_off_target,
            )
            return SettleOutcome.STALLED

        self._sim_time += step_size
        if self._sim_time >= self._timeout_s:
            LOGGER.warning(
                "arm %r settle timed out after %.2fs sim time: within tolerance=%s, "
                "still=%s, max |v| seen=%.4f rad/s (VEL_EPS %.4f)",
                self._arm_name,
                self._sim_time,
                is_within,
                is_still,
                self._max_speed,
                VEL_EPS_RAD_S,
            )
            return SettleOutcome.TIMED_OUT
        return None


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

    def set_drive_gains(self, gains: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        """Override the position gain (``kp``) and damping (``kd``) of the
        named joints, e.g. ``{"wrist_3_joint": {"kp": 225781, "kd": 90}}``,
        and keep them across a world reset. Returns the table
        ``drive_gains`` reports, after the change. The asset's own tuning is
        for the arm carrying itself; a wrist holding a payload can need more,
        and the wrist_3 drive of the UR5e asset (kp 57300, kd 0.25 against
        225781 and 90 one joint up) was back-driven forty degrees by a base
        swing with a 2 kg box on the cups on 2026-09-24."""
        raise NotImplementedError

    def prim_attrs(self, prim_path: str) -> dict[str, Any]:
        """Every authored attribute of the prim at ``prim_path`` as
        ``{name: value}`` plus its children's names, for reading what a
        joint's drive, limits and physics really are on the stage. The
        articulation controller's gain table is not the drive: on 2026-09-24
        wrist 3's controller gains were raised fourfold with no change in
        its motion."""
        raise NotImplementedError

    def drive_gains(self) -> dict[str, Any]:
        """The articulation's per-DOF position gain and damping, in PhysX
        order, alongside `all_dof_names`.

        Nothing in this module authors these: an asset's own values apply, and
        they are tuned for the arm carrying itself. A wrist that visibly sags
        or swings under a payload is asking whether they are high enough for
        this cell, and the answer is a number rather than a judgment."""
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

    def wait_until_within(self, timeout_s: float, tolerance_rad: float) -> SettleOutcome:
        """For a caller passing through a point rather than stopping at it:
        REACHED on the first physics step at which every named joint is within
        ``tolerance_rad`` of the last commanded targets, with no window of
        steps held. STALLED and TIMED_OUT follow the same rules as
        ``wait_for_settle``."""
        raise NotImplementedError

    def follow_path(
        self,
        points: list[list[float]],
        max_vel_rad_s: float | None,
        tolerance_rad: float,
        point_timeout_s: float,
    ) -> tuple[SettleOutcome, int]:
        """Drive the named joints through ``points`` in order, blocking the
        calling (non-sim) thread. A point counts as passed on the first physics
        step every joint is within ``tolerance_rad`` of it, and on that step the
        next point becomes the drive target, so the sim thread is not crossed
        between points.

        Returns ``(REACHED, len(points))`` once the last point is passed.
        Otherwise it returns ``(STALLED or TIMED_OUT, i)`` at the first point
        ``i`` not passed, by the same rules as ``wait_until_within`` with
        ``point_timeout_s`` of sim time per point, and the drive target stays at
        ``points[i]``. A ``stop()`` during the follow ends it with
        ``(REACHED, i)`` for the point in flight, with the drive target held
        where the arm was. An empty list returns ``(REACHED, 0)``."""
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
        drive_gains: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self._sim = sim
        self._art = articulation
        self._ee = ee_prim
        self._joint_names = joint_names
        self._base_correction: Quat = base_correction
        self._prim_path = prim_path
        # per-joint kp/kd overrides, applied whenever the articulation's
        # gains are (re)read so a reset keeps them
        self._drive_gain_overrides: dict[str, dict[str, float]] = {
            str(name): {k: float(v) for k, v in spec.items()}
            for name, spec in (drive_gains or {}).items()
        }
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
            if self._drive_gain_overrides:
                self._apply_drive_gains(self._drive_gain_overrides)

    def _apply_drive_gains(self, gains: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        """Sim-thread body of set_drive_gains: the controller's current gains
        with the named joints' kp/kd replaced, written back and snapshotted
        so post_reset restores the overridden values rather than the
        asset's."""
        controller = self._art.get_articulation_controller()
        kps, kds = controller.get_gains()
        kps = np.array(kps, dtype=float).copy()
        kds = np.array(kds, dtype=float).copy()
        for name, spec in gains.items():
            if name not in self._dof_names:
                raise ValueError(
                    f"no joint {name!r} in this articulation; joints are {self._dof_names}"
                )
            index = self._dof_names.index(name)
            if "kp" in spec:
                kps[index] = float(spec["kp"])
            if "kd" in spec:
                kds[index] = float(spec["kd"])
            self._drive_gain_overrides.setdefault(str(name), {}).update(
                {k: float(v) for k, v in spec.items()}
            )
        controller.set_gains(kps=kps, kds=kds)
        self._gains = (kps, kds)
        # armature: extra rotor inertia on the joint, the standard remedy for
        # an articulation drive that goes soft under a jointed payload
        view = getattr(self._art, "_articulation_view", None)
        armatures = {name: spec["armature"] for name, spec in gains.items() if "armature" in spec}
        if armatures:
            get_arm = getattr(view, "get_armatures", None) if view is not None else None
            set_arm = getattr(view, "set_armatures", None) if view is not None else None
            if get_arm is None or set_arm is None:
                raise ValueError("this articulation view cannot set joint armature")
            current = get_arm(joint_indices=self._joint_indices)
            if len(getattr(current, "shape", ())) == 2:
                current = current[0]
            values = np.array(current, dtype=float).copy()
            for name, armature in armatures.items():
                values[self._dof_names.index(name)] = float(armature)
            set_arm(values, joint_indices=self._joint_indices)
        table: dict[str, Any] = {
            "dof_names": list(self._dof_names),
            "kp": [float(value) for value in kps],
            "kd": [float(value) for value in kds],
        }
        get_arm = getattr(view, "get_armatures", None) if view is not None else None
        if get_arm is not None:
            try:
                current = get_arm(joint_indices=self._joint_indices)
                if len(getattr(current, "shape", ())) == 2:
                    current = current[0]
                table["armature"] = [float(value) for value in current]
            except Exception:  # noqa: BLE001 - the readback is a courtesy
                pass
        return table

    def set_drive_gains(self, gains: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        return self._sim.run(lambda: self._apply_drive_gains(gains))

    def prim_attrs(self, prim_path: str) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise PrimNotFoundError(f"no prim at {prim_path!r}")
            attrs: dict[str, Any] = {}
            for attr in prim.GetAttributes():
                if not attr.HasAuthoredValue() and not attr.HasValue():
                    continue
                try:
                    value = attr.Get()
                except Exception:  # noqa: BLE001 - an unreadable attribute is reported as such
                    value = "<unreadable>"
                if hasattr(value, "__iter__") and not isinstance(value, str | bytes):
                    try:
                        value = [float(v) if isinstance(v, int | float) else str(v) for v in value]
                    except Exception:  # noqa: BLE001
                        value = str(value)
                elif not isinstance(value, int | float | str | bool) and value is not None:
                    value = str(value)
                attrs[attr.GetName()] = value
            return {
                "prim_path": prim_path,
                "type": prim.GetTypeName(),
                "schemas": [str(schema) for schema in prim.GetAppliedSchemas()],
                "attrs": attrs,
                "children": [child.GetName() for child in prim.GetChildren()],
            }

        return self._sim.run(_read)

    def drive_gains(self) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            kps, kds = self._art.get_articulation_controller().get_gains()
            table: dict[str, Any] = {
                "dof_names": list(self._dof_names),
                "kp": [float(value) for value in kps],
                "kd": [float(value) for value in kds],
            }
            # the drive's force cap, which a stiffness override cannot lift:
            # a saturated drive is back-driven the same at any kp
            view = getattr(self._art, "_articulation_view", None)
            get_max_efforts = getattr(view, "get_max_efforts", None) if view is not None else None
            if get_max_efforts is not None:
                try:
                    efforts = get_max_efforts(joint_indices=self._joint_indices)
                    if len(getattr(efforts, "shape", ())) == 2:
                        efforts = efforts[0]
                    table["max_effort"] = [float(value) for value in efforts]
                except Exception as error:  # noqa: BLE001 - a diagnostic that cannot read stays absent
                    LOGGER.info("arm %r max efforts unreadable: %s", self._art.name, error)
            get_armatures = getattr(view, "get_armatures", None) if view is not None else None
            if get_armatures is not None:
                try:
                    armatures = get_armatures(joint_indices=self._joint_indices)
                    if len(getattr(armatures, "shape", ())) == 2:
                        armatures = armatures[0]
                    table["armature"] = [float(value) for value in armatures]
                except Exception as error:  # noqa: BLE001 - same courtesy as max_effort
                    LOGGER.info("arm %r armatures unreadable: %s", self._art.name, error)
            return table

        return self._sim.run(_read)

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
        self._write_targets_on_sim_thread(positions)

    def _write_targets_on_sim_thread(self, positions: list[float]) -> None:
        """Command the named joints to ``positions`` and record them as the
        last targets, leaving the velocity cap as it is."""
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
        return self._wait_within_window(timeout_s, tolerance_rad, SETTLE_WINDOW_STEPS)

    def wait_until_within(self, timeout_s: float, tolerance_rad: float) -> SettleOutcome:
        return self._wait_within_window(timeout_s, tolerance_rad, 1)

    def _judge_step(
        self, tracker: _StepTracker, targets: Sequence[float], step_size: float
    ) -> SettleOutcome | None:
        """One physics step of the named joints' state against ``targets``,
        judged by ``tracker``. Sim thread only."""
        velocities = self._art.get_joint_velocities(joint_indices=self._joint_indices)
        positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
        return tracker.step(positions, velocities, targets, step_size)

    def _wait_within_window(
        self, timeout_s: float, tolerance_rad: float, window_steps: int
    ) -> SettleOutcome:
        # A physics-step callback evaluates the settle predicate on
        # the sim thread every step and signals a threading.Event. This
        # (caller) thread only waits on it, so no wall-clock polling.
        if self._targets is None:
            return SettleOutcome.REACHED

        targets = list(self._targets)
        event = threading.Event()
        outcome: list[SettleOutcome] = []
        tracker = _StepTracker(
            getattr(self._art, "name", ""), timeout_s, tolerance_rad, window_steps
        )
        settle_state: dict[str, Any] = {"event": event, "outcome": outcome}

        def _on_step(step_size: float) -> None:
            # sim thread only: touch the tracker/Event, never self._sim.run.
            if event.is_set():
                return
            result = self._judge_step(tracker, targets, step_size)
            if result is not None:
                outcome.append(result)
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

    def follow_path(
        self,
        points: list[list[float]],
        max_vel_rad_s: float | None,
        tolerance_rad: float,
        point_timeout_s: float,
    ) -> tuple[SettleOutcome, int]:
        # One sim-thread task sets the first target and registers the
        # callback. After that the callback steps the target along the path
        # on the sim thread, so the sim thread is crossed once per path
        # rather than three times per point.
        if not points:
            return SettleOutcome.REACHED, 0

        path = [list(point) for point in points]
        event = threading.Event()
        outcome: list[SettleOutcome] = []
        # the index of the point in flight, and len(path) once the last is passed
        progress = {"index": 0}
        tracker = _StepTracker(getattr(self._art, "name", ""), point_timeout_s, tolerance_rad, 1)
        settle_state: dict[str, Any] = {"event": event, "outcome": outcome}

        def _on_step(step_size: float) -> None:
            # sim thread only: touch the tracker/Event, never self._sim.run.
            if event.is_set():
                return
            result = self._judge_step(tracker, path[progress["index"]], step_size)
            if result is None:
                return
            if result is not SettleOutcome.REACHED:
                outcome.append(result)
                event.set()
                return
            progress["index"] += 1
            if progress["index"] == len(path):
                outcome.append(SettleOutcome.REACHED)
                event.set()
                return
            self._write_targets_on_sim_thread(path[progress["index"]])
            tracker.reset()

        callback_name = self._settle_callback_name()

        def _start() -> None:
            self._apply_joint_targets_on_sim_thread(path[0], max_vel_rad_s)
            self._sim.world.add_physics_callback(callback_name, _on_step)

        def _remove() -> None:
            if self._sim.world.physics_callback_exists(callback_name):
                self._sim.world.remove_physics_callback(callback_name)

        with self._active_settle_lock:
            self._active_settle = settle_state
        self._sim.run(_start)
        try:
            # a wall-clock guard per point rather than per path: a point that
            # takes this long without the sim stepping onward means a paused
            # sim, and the path as a whole may legitimately take far longer.
            wall_clock_guard_s = point_timeout_s * SETTLE_GUARD_MULTIPLE + SETTLE_GUARD_SLACK_S
            seen_index = progress["index"]
            while not event.wait(timeout=wall_clock_guard_s):
                if progress["index"] == seen_index:
                    break
                seen_index = progress["index"]
        finally:
            with self._active_settle_lock:
                if self._active_settle is settle_state:
                    self._active_settle = None
            # after this no callback can run, so outcome and progress are final
            self._sim.run(_remove)
        return (outcome[0] if outcome else SettleOutcome.TIMED_OUT), progress["index"]

    def stop(self) -> None:
        def _hold() -> None:
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            current = [float(v) for v in positions]
            self._apply_joint_targets_on_sim_thread(current, None)
            # The new target IS the current position, so any in-flight wait
            # should read as having reached it, not stalled. This happens in
            # the same sim-thread task as the hold, so no physics step between
            # the two can advance a followed path to its next point.
            with self._active_settle_lock:
                active = self._active_settle
                if active is not None and not active["event"].is_set():
                    active["outcome"].append(SettleOutcome.REACHED)
                    active["event"].set()

        self._sim.run(_hold, allow_during_initialization=True)

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
            # armature lives on the view, which the reset rebuilds from the
            # asset's zeros; the overrides put it back with the gains
            if self._drive_gain_overrides:
                self._apply_drive_gains(self._drive_gain_overrides)
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
        # how many times stop() has run, so a follow_path in flight can tell it
        # was stopped
        self._stop_count = 0
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

    # a mock articulation has no drives, so it reports the shape of the answer
    def prim_attrs(self, prim_path: str) -> dict[str, Any]:
        """The mock has no stage: an empty attribute set for any path."""
        return {"prim_path": prim_path, "type": "", "schemas": [], "attrs": {}, "children": []}

    def set_drive_gains(self, gains: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        """The mock has no drives: the overrides are kept and reported back,
        None where a joint was never given a value."""
        names = list(self._dof_names)
        for name in gains:
            if name not in names:
                raise ValueError(f"no joint {name!r} in this articulation; joints are {names}")
        overrides = getattr(self, "_drive_gain_overrides", {})
        for name, spec in gains.items():
            overrides.setdefault(str(name), {}).update({k: float(v) for k, v in spec.items()})
        self._drive_gain_overrides = overrides
        return {
            "dof_names": names,
            "kp": [overrides.get(n, {}).get("kp") for n in names],
            "kd": [overrides.get(n, {}).get("kd") for n in names],
        }

    # with none of the numbers. A caller checking the verb's wiring gets a
    # reply; one reading gains gets Nones rather than an invented stiffness.
    def drive_gains(self) -> dict[str, Any]:
        return {
            "dof_names": list(self._dof_names),
            "kp": [None] * len(self._dof_names),
            "kd": [None] * len(self._dof_names),
        }

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

    def wait_until_within(self, timeout_s: float, tolerance_rad: float) -> SettleOutcome:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = time.monotonic()
                pos = self._positions_at(now)
                vel = self._velocities_at(now)
                target = list(self._target)
            selected = self._selected()
            if all(abs(pos[i] - target[i]) <= tolerance_rad for i in selected):
                return SettleOutcome.REACHED
            if all(abs(vel[i]) <= VEL_EPS_RAD_S for i in selected):
                return SettleOutcome.STALLED
            if now >= deadline:
                return SettleOutcome.TIMED_OUT
            time.sleep(self.STEP_S)

    def follow_path(
        self,
        points: list[list[float]],
        max_vel_rad_s: float | None,
        tolerance_rad: float,
        point_timeout_s: float,
    ) -> tuple[SettleOutcome, int]:
        with self._lock:
            stops_at_start = self._stop_count
        for index, point in enumerate(points):
            self.set_joint_targets(point, max_vel_rad_s)
            outcome = self.wait_until_within(point_timeout_s, tolerance_rad)
            if outcome is not SettleOutcome.REACHED:
                return outcome, index
            with self._lock:
                was_stopped = self._stop_count != stops_at_start
            if was_stopped:
                return SettleOutcome.REACHED, index
        return SettleOutcome.REACHED, len(points)

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            self._target = list(self._start)
            self._t0 = now
            self._stop_count += 1

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
