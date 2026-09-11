from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from viam.logging import getLogger

from ..asset_catalog import KNOWN_ASSETS, UR_JOINT_NAMES
from ..compat import caps
from ..spatial import Quat, Vec3
from ..usd_assets import _prim_range
from .arm import ArmHandle, MockArmHandle

if TYPE_CHECKING:
    from ..sim_manager import SimManager

LOGGER = getLogger(__name__)


GRIPPER_HOLDING_STEPS = 5  # consecutive still steps outside tolerance before is_holding() flips
# The 2F-85's five passive joints follow finger_joint through PhysxMimicJointAPI
# in the standalone asset. Attached under the arm those mimics fail to create
# (parsed before the joints join the articulation) and the
# passive drives then hold the linkage open against the finger drive, so the
# handle commands them itself: sign x finger angle, signs read off the
# linkage at rest.
GRIPPER_COUPLED_JOINT_SIGNS: dict[str, float] = {
    "right_outer_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
# The real 2F-85 has ONE motor. The linkage (loop-closing fixed joints in the
# asset) moves the other joints. Stiff drives on all six over-constrain the
# loops and the jaw buzzes in place (observed: ±85 deg/s at 1.7 deg), so the
# passive joints' drives are released to a little damping and finger_joint
# alone is driven.
PASSIVE_JOINT_DAMPING = 0.1
# A jaw pressed onto an object vibrates (observed: +/-90 deg/s at the contact
# angle), so stillness can never gate the stall/holding predicates. Like the
# arm's settle rule: a stall is "the gap to the target stopped improving by
# this much over GRIPPER_HOLDING_STEPS consecutive checks".
JAW_PROGRESS_EPS_RAD = math.radians(0.5)
GRIPPER_OPEN_WIDTH_M = 0.085  # 2F-85 jaw opening at the open angle. Linear to 0 at closed
DEFAULT_HOLDING_TOLERANCE_DEG = 2.0  # holding_tolerance_deg attrs default


class GripperHandle:
    """A parallel-jaw gripper riding an arm. Angles are radians on the drive
    joint (finger_joint on the 2F-85), increasing from open toward closed.
    The Viam edge (models/gripper.py) owns the [0,1]-normalised inputs and
    degrees. All methods are safe from any thread."""

    def jaw_limits(self) -> tuple[float, float]:
        """(open_rad, closed_rad) of the drive joint - the ends of the [0,1]
        input range."""
        raise NotImplementedError

    def get_jaw(self) -> float:
        """Measured drive-joint angle, radians."""
        raise NotImplementedError

    def set_jaw(self, rad: float) -> None:
        """Command the drive joint (clamped to jaw_limits). Returns at once."""
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Hold the current jaw angle."""
        raise NotImplementedError

    def is_moving(self) -> bool:
        """True while the jaw is travelling. False once it has settled, whether
        at its target or stalled on an object."""
        raise NotImplementedError

    def is_holding(self) -> bool:
        """Stall predicate, version-neutral (no Isaac contact query):
        the jaw is still AND |commanded - measured| > holding tolerance for
        GRIPPER_HOLDING_STEPS consecutive steps - i.e. it closed onto
        something short of its target. False while moving or when the jaw
        reached its target."""
        raise NotImplementedError

    def poll_state(self) -> tuple[float, bool, bool]:
        """(get_jaw(), is_moving(), is_holding()) in one round trip - grab()'s
        poll loop reads all three every iteration and this collapses that to
        a single call instead of three."""
        raise NotImplementedError

    def dof_names(self) -> list[str]:
        """The gripper's own DOF names as PhysX reports them after attach
        (finger_joint first). The mock returns its drive joint only."""
        raise NotImplementedError

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        """World poses ((x,y,z) m, (w,x,y,z)) of the mount link the gripper is
        bolted to ("parent") and its two fingertip links ("left_inner_finger",
        "right_inner_finger") - the GPU checklist's TCP measurement (item 4).
        The mock returns a synthetic set consistent with tcp_offset_m."""
        raise NotImplementedError

    def fingertip_world_bounds(self) -> dict[str, tuple[Vec3, Vec3]]:
        """World-space axis-aligned bounds (min, max) in meters of the two
        fingertip PAD meshes, keyed "left"/"right". The 2F-85 asset authors
        every link frame at the base, so link origins say nothing about where
        the pads are - the mesh bounds do (item 4, see jaw_box_mm)."""
        raise NotImplementedError

    def post_reset(self) -> None:
        """Re-command the last commanded jaw target after a world
        reset, so a reset mid-pick doesn't drop the object. No-op by
        default (the mock has no such state)."""
        return None

    def release(self) -> None:
        """Drop callbacks. The prim stays attached to the arm."""
        return None


class IsaacGripperHandle(GripperHandle):
    """Drives the finger_joint DOF of the ARM's articulation: the gripper is
    referenced with articulationEnabled=False, so its joints join the arm's
    DOF list and are addressed by name, never by position."""

    def __init__(
        self,
        sim: SimManager,
        articulation: Any,
        drive_joint: str,
        open_rad: float,
        closed_rad: float,
        holding_tolerance_rad: float,
        prim_path: str,
    ) -> None:
        self._sim = sim
        self._art = articulation
        self._drive_joint = drive_joint
        self._open_rad = open_rad
        self._closed_rad = closed_rad
        self._holding_tolerance_rad = holding_tolerance_rad
        self._prim_path = prim_path
        # set by _create_gripper_isaac: the link base_link is bolted to
        self.parent_prim_path: str | None = None
        dof_names = list(articulation.dof_names)
        try:
            self._idx = dof_names.index(drive_joint)
        except ValueError as exc:
            raise ValueError(
                f"gripper drive joint {drive_joint!r} not found in articulation "
                f"dof_names: {dof_names}"
            ) from exc
        # last commanded target, plus the progress window for the stall-based
        # is_moving/is_holding predicates. The latch carries a detected hold
        # across slow polls: once the jaw stalls outside tolerance it stays
        # "holding" until it reaches its target (nothing left between the
        # jaws) or a new set_jaw.
        self._target: float | None = None
        self._best_gap_rad: float | None = None
        self._no_progress_count = 0
        self._held_latch = False
        self._coupled: list[tuple[int, float]] = [
            (dof_names.index(name), sign)
            for name, sign in GRIPPER_COUPLED_JOINT_SIGNS.items()
            if name in dof_names
        ]
        LOGGER.info(
            "gripper drive %r at dof %d, %d passive linkage joints: %s",
            drive_joint,
            self._idx,
            len(self._coupled),
            [dof_names[i] for i, _sign in self._coupled],
        )
        self._release_passive_drives()

    def _release_passive_drives(self) -> None:
        """Zero the passive linkage joints' drive stiffness (small damping) so
        the loop closures, not competing drives, couple them to finger_joint.
        Logs the authored gains so the asset's tuning stays on record."""
        try:
            controller = self._art.get_articulation_controller()
            kps, kds = controller.get_gains()
            kps = np.array(kps, dtype=float).copy()
            kds = np.array(kds, dtype=float).copy()
            LOGGER.info(
                "gripper passive joint gains before release (kp, kd): %s",
                [(float(kps[i]), float(kds[i])) for i, _sign in self._coupled],
            )
            for index, _sign in self._coupled:
                kps[index] = 0.0
                kds[index] = PASSIVE_JOINT_DAMPING
            controller.set_gains(kps=kps, kds=kds)
        except Exception:  # best-effort gain release, any failure is logged, not fatal
            LOGGER.exception("could not release the gripper's passive joint drives")

    def jaw_limits(self) -> tuple[float, float]:
        return (self._open_rad, self._closed_rad)

    def get_jaw(self) -> float:
        def _get() -> float:
            return float(self._art.get_joint_positions(joint_indices=[self._idx])[0])

        return self._sim.run(_get)

    def set_jaw(self, rad: float) -> None:
        self._sim.run(lambda: self._apply_jaw_target_on_sim_thread(rad))

    def _apply_jaw_target_on_sim_thread(self, rad: float) -> None:
        """Sim-thread body shared by set_jaw and stop's hold-in-place. Clamps
        here, not in each caller, so a physics overshoot measured by stop's
        hold never becomes a target past the jaw's travel limits."""
        rad = min(max(rad, self._open_rad), self._closed_rad)
        # finger_joint only: the linkage carries the passive joints
        action = self._sim._isaac.ArticulationAction(
            joint_positions=np.array([rad], dtype=float),
            joint_indices=[self._idx],
        )
        self._art.apply_action(action)
        self._target = rad
        self._best_gap_rad = None
        self._no_progress_count = 0
        self._held_latch = False

    def open(self) -> None:
        self.set_jaw(self._open_rad)

    def close(self) -> None:
        self.set_jaw(self._closed_rad)

    def stop(self) -> None:
        def _hold() -> None:
            measured = float(self._art.get_joint_positions(joint_indices=[self._idx])[0])
            self._apply_jaw_target_on_sim_thread(measured)

        self._sim.run(_hold, allow_during_initialization=True)

    def _gap_and_stall(self) -> tuple[float, bool]:
        """(|target - measured|, stalled): the jaw is stalled when the gap has
        not improved by JAW_PROGRESS_EPS_RAD for GRIPPER_HOLDING_STEPS
        consecutive checks. Velocity plays no part: a jaw pressed onto an
        object vibrates and never reads still. The window counts CALLS, so
        grab()'s 120 Hz poll detects the stall but a client sampling at 1 Hz
        never re-accumulates it once the jaw creeps - a stall
        outside tolerance therefore latches _held_latch, cleared only when
        the jaw reaches its target or on a new set_jaw. Sim thread only."""
        if self._target is None:
            return 0.0, False
        measured = float(self._art.get_joint_positions(joint_indices=[self._idx])[0])
        gap = abs(self._target - measured)
        if self._best_gap_rad is None or gap < self._best_gap_rad - JAW_PROGRESS_EPS_RAD:
            self._best_gap_rad = gap
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1
        stalled = self._no_progress_count >= GRIPPER_HOLDING_STEPS
        if gap <= self._holding_tolerance_rad:
            self._held_latch = False  # the jaw reached its target: nothing is held
        elif stalled:
            self._held_latch = True
        return gap, stalled

    def is_moving(self) -> bool:
        def _check() -> bool:
            gap, stalled = self._gap_and_stall()
            return gap > self._holding_tolerance_rad and not stalled and not self._held_latch

        return self._sim.run(_check)

    def is_holding(self) -> bool:
        def _check() -> bool:
            gap, stalled = self._gap_and_stall()
            return (stalled or self._held_latch) and gap > self._holding_tolerance_rad

        return self._sim.run(_check)

    def poll_state(self) -> tuple[float, bool, bool]:
        def _poll() -> tuple[float, bool, bool]:
            jaw = float(self._art.get_joint_positions(joint_indices=[self._idx])[0])
            gap, stalled = self._gap_and_stall()
            moving = gap > self._holding_tolerance_rad and not stalled and not self._held_latch
            holding = (stalled or self._held_latch) and gap > self._holding_tolerance_rad
            return jaw, moving, holding

        return self._sim.run(_poll)

    def dof_names(self) -> list[str]:
        def _names() -> list[str]:
            return [n for n in self._art.dof_names if n not in UR_JOINT_NAMES]

        return self._sim.run(_names)

    def post_reset(self) -> None:
        """Re-command the last commanded jaw target - a world.reset()
        can otherwise let a held object drop."""

        def _redo() -> None:
            if self._target is None:
                return
            action = self._sim._isaac.ArticulationAction(
                joint_positions=np.array([self._target], dtype=float),
                joint_indices=[self._idx],
            )
            self._art.apply_action(action)

        self._sim.run(_redo)

    def release(self) -> None:
        """The gripper drives a DOF of the arm's articulation and owns
        no scene-registry entry or physics callback of its own (post_reset is
        a post-reset hook, not a physics callback) - nothing to release."""
        return None

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        def _poses() -> dict[str, tuple[Vec3, Quat]]:
            from pxr import Usd, UsdPhysics

            # every rigid-body link under the gripper, keyed by link name (the
            # GPU checklist reads the whole chain, not only the fingertips)
            paths: dict[str, str] = {}
            root = self._sim._isaac.get_prim_at_path(self._prim_path)
            for prim in _prim_range(Usd, root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() not in paths:
                    paths[prim.GetName()] = str(prim.GetPath())
            if self.parent_prim_path:
                paths["parent"] = self.parent_prim_path
            out: dict[str, tuple[Vec3, Quat]] = {}
            for key, path in paths.items():
                pos, quat = self._sim._isaac.SingleXFormPrim(path).get_world_pose()
                out[key] = (
                    (float(pos[0]), float(pos[1]), float(pos[2])),
                    (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
            return out

        return self._sim.run(_poses)

    def fingertip_world_bounds(self) -> dict[str, tuple[Vec3, Vec3]]:
        def _bounds() -> dict[str, tuple[Vec3, Vec3]]:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics

            time = Usd.TimeCode.Default()
            root = self._sim._isaac.get_prim_at_path(self._prim_path)
            out: dict[str, tuple[Vec3, Vec3]] = {}
            for mesh in _prim_range(Usd, root):
                if "fingertip" not in mesh.GetName().lower():
                    continue
                link = mesh.GetParent()
                while link.IsValid() and not link.HasAPI(UsdPhysics.RigidBodyAPI):
                    link = link.GetParent()
                if not link.IsValid():
                    continue
                # mesh-in-link from USD (static), link-in-world from the
                # physics-aware pose: robust whether PhysX writes to USD or Fabric
                mesh_in_link = (
                    UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(time)
                    * UsdGeom.Xformable(link).ComputeLocalToWorldTransform(time).GetInverse()
                )
                pos, quat = self._sim._isaac.SingleXFormPrim(str(link.GetPath())).get_world_pose()
                rotate = Gf.Matrix4d().SetRotate(
                    Gf.Quatd(
                        float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))
                    )
                )
                translate = Gf.Matrix4d().SetTranslate(
                    Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                )
                mesh_world = mesh_in_link * rotate * translate
                extent = UsdGeom.Boundable(mesh).GetExtentAttr().Get(time)
                if extent is None or len(extent) != 2:
                    continue
                corners = [
                    mesh_world.Transform(Gf.Vec3d(x, y, z))
                    for x in (extent[0][0], extent[1][0])
                    for y in (extent[0][1], extent[1][1])
                    for z in (extent[0][2], extent[1][2])
                ]
                low = tuple(min(float(c[i]) for c in corners) for i in range(3))
                high = tuple(max(float(c[i]) for c in corners) for i in range(3))
                side = "left" if "left" in str(mesh.GetPath()).lower() else "right"
                out[side] = ((low[0], low[1], low[2]), (high[0], high[1], high[2]))
            return out

        return self._sim.run(_bounds)


class MockGripperHandle(GripperHandle):
    """The jaw interpolates at MockArmHandle.SPEED toward its target. With
    attrs["mock_object_width_m"] set, the jaw stalls at the angle where the
    jaws would touch that object (GRIPPER_OPEN_WIDTH_M at open_rad, linear
    to 0 at closed_rad) and is_holding() flips true after
    GRIPPER_HOLDING_STEPS. Unset = nothing between the jaws, so close()
    reaches closed_rad and is_holding() stays False."""

    def __init__(self, name: str, attrs: dict[str, Any], arm: ArmHandle) -> None:
        self.name = name
        self._arm = arm
        default_meta = KNOWN_ASSETS["robotiq_2f_85"]
        meta = KNOWN_ASSETS.get(str(attrs.get("asset", "robotiq_2f_85")), default_meta)
        self._drive_joint = meta.get("drive_joint", "finger_joint")
        self.open_rad = math.radians(attrs.get("open_deg", meta.get("open_deg", 0.0)))
        self.closed_rad = math.radians(attrs.get("closed_deg", caps().gripper_closed_deg))
        self.holding_tolerance_rad = math.radians(
            attrs.get("holding_tolerance_deg", DEFAULT_HOLDING_TOLERANCE_DEG)
        )
        self.mock_object_width_m: float | None = attrs.get("mock_object_width_m")
        self.tcp_offset_m = float(attrs.get("tcp_offset_m", meta.get("tcp_offset_m", 0.134)))
        self._speed = MockArmHandle.SPEED
        self._lock = threading.Lock()
        now = time.monotonic()
        self._start = self.open_rad
        self._target = self.open_rad
        self._t0 = now

    def _contact_angle(self) -> float:
        """The drive-joint angle at which the jaws would touch
        mock_object_width_m - GRIPPER_OPEN_WIDTH_M at open_rad, linearly to
        closed_rad at width 0 (also the value used when nothing is set, since
        there is then nothing to stop the jaw short of closed_rad)."""
        if self.mock_object_width_m is None:
            return self.closed_rad
        width = min(max(self.mock_object_width_m, 0.0), GRIPPER_OPEN_WIDTH_M)
        fraction_closed = 1.0 - width / GRIPPER_OPEN_WIDTH_M
        return self.open_rad + (self.closed_rad - self.open_rad) * fraction_closed

    def _effective_target(self) -> float:
        """The commanded target, clamped short of an object in the way."""
        return min(self._target, self._contact_angle())

    def _arrival_time(self) -> float:
        """The monotonic time the jaw reaches _effective_target, given the
        move that started at (_start, _t0)."""
        delta = self._effective_target() - self._start
        return self._t0 + abs(delta) / self._speed

    def _jaw_at(self, now: float) -> float:
        start = self._start
        target = self._effective_target()
        delta = target - start
        travel = min(self._speed * max(0.0, now - self._t0), abs(delta))
        if travel >= abs(delta):
            return target
        return start + math.copysign(travel, delta)

    def jaw_limits(self) -> tuple[float, float]:
        return (self.open_rad, self.closed_rad)

    def get_jaw(self) -> float:
        with self._lock:
            return self._jaw_at(time.monotonic())

    def set_jaw(self, rad: float) -> None:
        rad = min(max(rad, self.open_rad), self.closed_rad)
        with self._lock:
            now = time.monotonic()
            self._start = self._jaw_at(now)
            self._target = rad
            self._t0 = now

    def open(self) -> None:
        self.set_jaw(self.open_rad)

    def close(self) -> None:
        self.set_jaw(self.closed_rad)

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            current = self._jaw_at(now)
            self._start = current
            self._target = current
            self._t0 = now

    def is_moving(self) -> bool:
        with self._lock:
            return time.monotonic() < self._arrival_time()

    def is_holding(self) -> bool:
        with self._lock:
            now = time.monotonic()
            arrival = self._arrival_time()
            if now < arrival:
                return False
            measured = self._effective_target()
            if abs(self._target - measured) <= self.holding_tolerance_rad:
                return False
            still_duration = now - arrival
            return still_duration >= GRIPPER_HOLDING_STEPS * MockArmHandle.STEP_S

    def poll_state(self) -> tuple[float, bool, bool]:
        with self._lock:
            now = time.monotonic()
            jaw = self._jaw_at(now)
            arrival = self._arrival_time()
            moving = now < arrival
            holding = False
            if not moving:
                measured = self._effective_target()
                if abs(self._target - measured) > self.holding_tolerance_rad:
                    holding = (now - arrival) >= GRIPPER_HOLDING_STEPS * MockArmHandle.STEP_S
            return jaw, moving, holding

    def dof_names(self) -> list[str]:
        return [self._drive_joint]

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        """Synthetic: the mount link at the origin, fingertips straddling the
        TCP at tcp_offset_m along +Z, GRIPPER_OPEN_WIDTH_M apart."""
        identity: Quat = (1.0, 0.0, 0.0, 0.0)
        half_width = GRIPPER_OPEN_WIDTH_M / 2.0
        return {
            "parent": ((0.0, 0.0, 0.0), identity),
            "base_link": ((0.0, 0.0, 0.0), identity),
            "left_inner_finger": ((half_width, 0.0, self.tcp_offset_m), identity),
            "right_inner_finger": ((-half_width, 0.0, self.tcp_offset_m), identity),
        }

    FINGERTIP_PAD_HALF_EXTENT_M: tuple[float, float, float] = (0.005, 0.011, 0.019)

    def fingertip_world_bounds(self) -> dict[str, tuple[Vec3, Vec3]]:
        """Synthetic pads centered at tcp_offset_m along +Z, straddling the jaw."""
        half_width = GRIPPER_OPEN_WIDTH_M / 2.0
        hx, hy, hz = self.FINGERTIP_PAD_HALF_EXTENT_M
        out: dict[str, tuple[Vec3, Vec3]] = {}
        for side, cx in (("left", half_width), ("right", -half_width)):
            cz = self.tcp_offset_m
            out[side] = ((cx - hx, -hy, cz - hz), (cx + hx, hy, cz + hz))
        return out
