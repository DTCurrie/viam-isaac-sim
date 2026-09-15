"""The simulated arm model."""

import asyncio
import hashlib
import json
import math
import os
import tempfile
import threading
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.arm import Arm, JointPositions, KinematicsFileFormat, Pose
from viam.errors import MethodNotImplementedError, ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.proto.component.arm import MoveOptions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..kinematics import Chain, JointLimitError, UnreachablePoseError
from ..length_units import MM_PER_M
from ..sim_manager import (
    KNOWN_ASSETS,
    SETTLE_TOL_RAD,
    ArmHandle,
    SettleOutcome,
    SimManager,
)
from ..spatial import ov_to_quat, quat_to_ov
from .component_frame_pose import apply_frame_to_attrs, get_attrs
from .sim_component_validation import validate_sim_component

_TOLERANCE_RAD = SETTLE_TOL_RAD
_WAYPOINT_TOLERANCE_RAD = math.radians(2.0)
_WAYPOINT_DEADLINE_S = 10.0
# How many waypoints in a row may stall before a trajectory is abandoned. One
# stall is a dense path's normal residual; a run of them is a blocked arm.
_MAX_CONSECUTIVE_WAYPOINT_STALLS = 5
# observed settle drift past a limit is ~3e-5 deg (wrist_2 at
# -360.00003). 0.01 covers it by orders of magnitude while a genuinely wrong
# target still raises
_JOINT_LIMIT_TOLERANCE_DEG = 0.01
# the real driver skips a MoveToPosition when already within these
# thresholds of the target (docs/PARITY.md, Arm move_to_position row)
_POSITION_SKIP_TOL_M = 1e-3
_ANGULAR_SKIP_TOL_RAD = math.radians(0.06)


class JointTargetOutOfLimitsError(ViamGRPCError, ValueError):
    """A commanded joint target is outside the SVA's declared limits, or the
    number of joint values doesn't match the arm's DOF count."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.INVALID_ARGUMENT)
        Exception.__init__(self, message)


class ArmMoveStalledError(ViamGRPCError):
    """The arm stopped moving (velocities settled) before reaching its
    commanded target - e.g. blocked by an obstacle."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.ABORTED)
        Exception.__init__(self, message)


class ArmMoveTimeoutError(ViamGRPCError, TimeoutError):
    """The move deadline passed while the arm was still converging on its
    target."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.DEADLINE_EXCEEDED)
        Exception.__init__(self, message)


class PoseUnreachableError(ViamGRPCError, ValueError):
    """No joint solution reaches the requested pose within the kinematics
    solver's tolerances."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.INVALID_ARGUMENT)
        Exception.__init__(self, message)


class KinematicsUnavailableError(ViamGRPCError, RuntimeError):
    """The arm has no kinematics file to solve MoveToPosition against (no
    kinematics_url and no known asset kinematics)."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.FAILED_PRECONDITION)
        Exception.__init__(self, message)


def _pose_within_tolerance(
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    target_pos: tuple[float, float, float],
    target_quat: tuple[float, float, float, float],
) -> bool:
    """True when `pos`/`quat` are within `_POSITION_SKIP_TOL_M` and
    `_ANGULAR_SKIP_TOL_RAD` of the target, matching the "already there" skip
    the real driver applies before commanding a move."""
    position_error = math.dist(pos, target_pos)
    dot = sum(a * b for a, b in zip(quat, target_quat, strict=True))
    angular_error = 2 * math.acos(min(1.0, abs(dot)))
    return position_error < _POSITION_SKIP_TOL_M and angular_error < _ANGULAR_SKIP_TOL_RAD


def _wrapped_into_range(deg: float, min_deg: float, max_deg: float) -> float:
    """``deg`` moved by whole turns into ``[min_deg, max_deg]`` when a turn
    lands it there, else unchanged.

    PhysX reports a revolute joint's accumulated angle, so an arm that keeps
    rotating the same way reads past the range its kinematics declares: a
    sorting run measured 4135.62 degrees on a joint limited to 360. The motion
    service then plans from a state its own solver calls illegal and refuses
    every later move through that joint. A whole turn is the identity for a
    revolute joint's pose, so reporting the wrapped angle describes the same
    physical arm in terms the planner accepts. A value no turn can bring into
    range is left alone, since that is a genuinely out-of-range joint and the
    caller needs to see it."""
    turn = 360.0
    if max_deg <= min_deg or min_deg <= deg <= max_deg:
        return deg
    fewest_turns = math.ceil((min_deg - deg) / turn)
    most_turns = math.floor((max_deg - deg) / turn)
    if fewest_turns > most_turns:
        return deg
    # a range wider than a turn accepts several, so take the one landing
    # nearest the middle of the range rather than nearest whichever end
    middle = (min_deg + max_deg) / 2.0
    turns = min(range(fewest_turns, most_turns + 1), key=lambda k: abs(deg + k * turn - middle))
    return deg + turns * turn


def _stuck_joint_detail(
    current: Sequence[float], targets: Sequence[float], tolerance_rad: float
) -> str:
    """ "jN: at <deg> want <deg>" for every joint outside tolerance."""
    return ", ".join(
        f"j{i}: at {math.degrees(c):.1f} want {math.degrees(t):.1f}"
        for i, (c, t) in enumerate(zip(current, targets, strict=True))
        if abs(c - t) > tolerance_rad
    )


def _validate_arm_attrs(name: str, attrs: Mapping[str, Any]) -> None:
    max_vel_degs_per_sec = attrs.get("max_vel_degs_per_sec")
    if max_vel_degs_per_sec is not None:
        if isinstance(max_vel_degs_per_sec, bool) or not isinstance(
            max_vel_degs_per_sec, (int, float)
        ):
            raise ValueError(
                f'{name}: "max_vel_degs_per_sec" must be a positive number, '
                f"got {max_vel_degs_per_sec!r}"
            )
        if max_vel_degs_per_sec <= 0:
            raise ValueError(
                f'{name}: "max_vel_degs_per_sec" must be a positive number, '
                f"got {max_vel_degs_per_sec!r}"
            )


class IsaacArm(Arm, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    """viam:isaac-sim-devin:arm, a simulated arm.

    GetEndPosition reports the end effector pose in the arm base frame (not
    world frame) as of this release.

    Move completion: moves settle via ArmHandle.wait_for_settle, with no
    wall-clock polling, and raise one of:
      JointTargetOutOfLimitsError (ValueError, INVALID_ARGUMENT)  - a target is
        outside the SVA's declared joint limits, or the joint count doesn't
        match the arm's DOF count.
      ArmMoveStalledError (ABORTED)      - the arm stopped moving before
        reaching its target (e.g. blocked by an obstacle).
      ArmMoveTimeoutError (TimeoutError, DEADLINE_EXCEEDED) - the move deadline
        (move_timeout_sec, capped by the SDK's timeout= kwarg) passed while the
        arm was still converging.
    A dropped RPC holds the arm at its current position instead of leaving it
    driving toward a target nothing is waiting on any more.
    MoveToPosition solves against the served kinematics and drives the joint
    path, so the frame system, the motion service and the sim agree by
    construction.
    move_through_joint_positions honors MoveOptions.max_vel_degs_per_sec_joints
    (the min across joints) when set, else MoveOptions.max_vel_degs_per_sec,
    else the configured max_vel_degs_per_sec attribute. move_to_joint_positions
    takes no MoveOptions, so it always uses the configured attribute. The
    acceleration fields and max_tcp_speed are logged once and not honored.
    DoCommand answers no sim-only verbs. The world component's DoCommand
    (joint_state/dof_names/prim_pose) reads this arm's sim state instead.

    close() releases the handle and its post-reset hooks. The prim stays in the
    stage."""

    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: ArmHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._move_timeout = 30.0
        self._default_max_vel_rad_s: float | None = None
        self._kinematics: tuple[KinematicsFileFormat.ValueType, bytes] | None = None
        self._kinematics_load_has_failed = False
        self._has_warned_options = False
        self._kinematics_chain: Chain | None = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        arm = cls(config.name)
        arm.reconfigure(config, dependencies)
        return arm

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        """Attributes:
        world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
        asset (string)             - known robot, e.g. "ur20", "ur5e", "franka"
        usd_path (string)          - explicit USD to spawn instead of a known asset
        prim_path (string)         - where to place it (default /World/<name>), or
                                     an existing articulation in the stage
        position ([x,y,z] meters)  - spawn position
        end_effector_prim (string) - prim whose pose, in the arm base frame, is
                                     reported by GetEndPosition (default
                                     <arm prim>/wrist_3_link for UR assets)
        home_joints_deg ([deg])    - joint angles the arm is PLACED at on build,
                                     rather than driven to, and that a world
                                     reset returns it to. Unset = the asset's
                                     own default, which for a UR is every joint
                                     at zero, i.e. fully extended horizontally.
                                     A cell with anything tall in front of the
                                     arm wants this set, or the arm boots lying
                                     across its own workspace and the first
                                     move sweeps whatever is there aside.
        move_timeout_sec (float)   - max time to wait for a move (default 30)
        max_vel_degs_per_sec (float, positive) - default velocity cap for a move
                                     that carries no MoveOptions cap of its own
                                     (e.g. a real UR's speed_degs_per_sec).
                                     Unset = the drive's own limit.
        kinematics_url (string)    - where to fetch the kinematics file served by
                                     GetKinematics (.json = SVA, .urdf = URDF;
                                     file:// URLs work). Known assets with official
                                     viam kinematics (ur3e/ur5e/ur7e/ur20) fetch them
                                     automatically.
        """
        deps, opt_deps = validate_sim_component(config)
        _validate_arm_attrs(config.name, get_attrs(config))
        return deps, opt_deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._move_timeout = float(attrs.get("move_timeout_sec", 30.0))
        max_vel_degs_per_sec = attrs.get("max_vel_degs_per_sec")
        self._default_max_vel_rad_s = (
            math.radians(float(max_vel_degs_per_sec)) if max_vel_degs_per_sec is not None else None
        )
        self._attrs = attrs
        self._handle = SimManager.get().create_arm(self.name, attrs)
        self._kinematics_chain = None
        self._prefetch_kinematics()

    def _prefetch_kinematics(self) -> None:
        """Loads the configured kinematics file on a background thread kicked
        off from reconfigure, not the first RPC, so a fresh machine's first
        GetKinematics/MoveToPosition never pays for a kinematics_url fetch
        inline. Failure here is silent: the first RPC that needs kinematics
        retries the load and raises normally."""
        if not self._kinematics_url():
            return
        threading.Thread(target=self._prefetch_kinematics_worker, daemon=True).start()

    def _prefetch_kinematics_worker(self) -> None:
        try:
            self._kinematics = self._load_kinematics()
        # Prefetch is best-effort: any load failure here should fall back
        # to the normal on-demand load path (_chain/get_kinematics), not
        # raise from a background thread with nothing to catch it.
        except Exception:  # noqa: BLE001
            self.logger.warning(
                "arm %s: kinematics prefetch failed; will retry on first use", self.name
            )

    async def close(self) -> None:
        """Release the handle (hooks, callbacks). The prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    def _deadline_s(self, timeout: float | None) -> float:
        """move_timeout_sec, capped by the SDK's timeout= kwarg when given."""
        return self._move_timeout if timeout is None else min(self._move_timeout, timeout)

    async def get_end_position(self, **kwargs) -> Pose:
        (x, y, z), quat = await asyncio.to_thread(self._h().get_end_pose)
        ox, oy, oz, theta = quat_to_ov(quat)
        return Pose(
            x=x * MM_PER_M,
            y=y * MM_PER_M,
            z=z * MM_PER_M,
            o_x=ox,
            o_y=oy,
            o_z=oz,
            theta=math.degrees(theta),
        )

    async def _chain(self) -> Chain:
        """The served kinematics as a `Chain`, loaded and parsed once."""
        if self._kinematics_chain is not None:
            return self._kinematics_chain
        if self._kinematics is None:
            if not self._kinematics_url():
                raise KinematicsUnavailableError(
                    f"arm {self.name}: no kinematics file to solve move_to_position "
                    'against; set the "kinematics_url" attribute'
                )
            self._kinematics = await asyncio.to_thread(self._load_kinematics)
        fmt, data = self._kinematics
        chain = Chain.from_kinematics(fmt, data)
        self._kinematics_chain = chain
        return chain

    async def move_to_position(
        self,
        pose: Pose,
        *,
        extra: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """Solves against the served kinematics and drives the joint path, so
        the frame system, the motion service and the sim agree by
        construction."""
        chain = await self._chain()
        target_pos = (pose.x / MM_PER_M, pose.y / MM_PER_M, pose.z / MM_PER_M)
        target_quat = ov_to_quat(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta))

        handle = self._h()
        current = await asyncio.to_thread(handle.get_joint_positions)
        current_pos, current_quat = chain.fk(current)
        if _pose_within_tolerance(current_pos, current_quat, target_pos, target_quat):
            return

        try:
            solution = chain.ik(target_pos, target_quat, current)
        except UnreachablePoseError:
            raise PoseUnreachableError(
                f"arm {self.name}: no joint solution reaches pose {pose}"
            ) from None
        except JointLimitError as e:
            raise JointTargetOutOfLimitsError(str(e)) from None

        await self.move_to_joint_positions(
            JointPositions(values=[math.degrees(v) for v in solution]), timeout=timeout
        )

    async def _joint_limits_deg(self) -> list[tuple[str, float, float]] | None:
        """(joint id, min_deg, max_deg) per joint from the SVA kinematics, in
        SVA joint order - the order set_joint_targets/move_to_joint_positions
        already expect. None when limits aren't available: no kinematics
        configured, URDF format, or the file failed to load (logged once)."""
        kinematics = self._kinematics
        if kinematics is None:
            url = self._kinematics_url()
            if not url:
                return None
            try:
                kinematics = await asyncio.to_thread(self._load_kinematics)
                self._kinematics = kinematics
            # Joint-limit checking is best-effort: any load failure (missing file, bad
            # format, IO error) should skip it, not crash the arm.
            except Exception:  # noqa: BLE001
                if not self._kinematics_load_has_failed:
                    self.logger.warning(
                        "arm %s: could not load kinematics for joint-limit checking; skipping",
                        self.name,
                    )
                    self._kinematics_load_has_failed = True
                return None

        fmt, data = kinematics
        if fmt != KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA:
            return None
        try:
            joints = json.loads(data).get("joints", [])
            return [
                (j.get("id", f"joint{i}"), float(j["min"]), float(j["max"]))
                for i, j in enumerate(joints)
            ]
        # Joint-limit checking is best-effort: any parse failure (malformed JSON,
        # missing keys, wrong types) should skip it, not crash the arm.
        except Exception:  # noqa: BLE001
            if not self._kinematics_load_has_failed:
                self.logger.warning(
                    "arm %s: could not parse SVA kinematics for joint-limit checking; skipping",
                    self.name,
                )
                self._kinematics_load_has_failed = True
            return None

    async def _clamped_joint_targets(self, targets_rad: Sequence[float]) -> list[float]:
        """The targets with any boundary value within _JOINT_LIMIT_TOLERANCE_DEG
        of an SVA limit clamped onto that limit; raises
        JointTargetOutOfLimitsError beyond the tolerance. Physics
        settle drifts a joint micro-degrees past its limit and the motion
        service echoes that reported state back as a plan waypoint
        (wrist_2 at -360.00003 deg wedged every subsequent plan), so
        an exact-boundary target must execute, not raise. Targets pass through
        unchecked when limits aren't available."""
        limits = await self._joint_limits_deg()
        if limits is None:
            return list(targets_rad)
        clamped = list(targets_rad)
        for i, t in enumerate(targets_rad):
            if i >= len(limits):
                break
            joint_id, min_deg, max_deg = limits[i]
            deg = math.degrees(t)
            if (
                deg < min_deg - _JOINT_LIMIT_TOLERANCE_DEG
                or deg > max_deg + _JOINT_LIMIT_TOLERANCE_DEG
            ):
                raise JointTargetOutOfLimitsError(
                    f"arm {self.name}: joint {joint_id} target {deg:.2f} deg "
                    f"out of range [{min_deg:.2f}, {max_deg:.2f}]"
                )
            if deg < min_deg or deg > max_deg:
                clamped[i] = math.radians(min(max(deg, min_deg), max_deg))
        return clamped

    async def _settle_or_raise(
        self,
        handle: ArmHandle,
        targets: Sequence[float],
        deadline_s: float,
        tolerance_rad: float,
        *,
        detail_prefix: str,
    ) -> None:
        outcome = await asyncio.to_thread(handle.wait_for_settle, deadline_s, tolerance_rad)
        if outcome is SettleOutcome.REACHED:
            return
        # Hold where we are: leaving the drive target at an unreachable pose
        # keeps the arm pushing into whatever blocked it, which on hardware has
        # launched the block and wound the elbow up.
        await asyncio.to_thread(handle.stop)
        current = await asyncio.to_thread(handle.get_joint_positions)
        detail = _stuck_joint_detail(current, targets, tolerance_rad)
        if outcome is SettleOutcome.STALLED:
            raise ArmMoveStalledError(f"{detail_prefix} stalled (stuck joints: {detail})")
        raise ArmMoveTimeoutError(
            f"{detail_prefix} did not reach target within {deadline_s:.1f}s "
            f"(stuck joints: {detail})"
        )

    async def move_to_joint_positions(
        self, positions: JointPositions, *, timeout: float | None = None, **kwargs
    ) -> None:
        targets = [math.radians(v) for v in positions.values]
        handle = self._h()
        current = await asyncio.to_thread(handle.get_joint_positions)
        if len(current) != len(targets):
            raise JointTargetOutOfLimitsError(
                f"arm {self.name}: expected {len(current)} joint values, got {len(targets)}"
            )
        targets = await self._clamped_joint_targets(targets)
        await asyncio.to_thread(handle.set_joint_targets, targets, self._default_max_vel_rad_s)

        try:
            await self._settle_or_raise(
                handle,
                targets,
                self._deadline_s(timeout),
                _TOLERANCE_RAD,
                detail_prefix=f"arm {self.name}",
            )
        except asyncio.CancelledError:
            # a dropped RPC holds position instead of leaving the drive
            # target at an unreached, now-unwatched pose
            await asyncio.to_thread(handle.stop)
            raise

    def _max_vel_rad_s(self, options: MoveOptions | None) -> float | None:
        """The per-joint max_vel_degs_per_sec_joints, when non-empty, is the
        ONLY velocity limit honored (the scalar is ignored in that
        case); the handle only takes one scalar, so the min across joints is
        used. Otherwise MoveOptions.max_vel_degs_per_sec applies when set.
        Falls back to the configured max_vel_degs_per_sec attribute
        (None = the drive's own limit) when the options carry neither.
        Acceleration fields and max_tcp_speed follow the same per-joint-wins
        precedence but aren't honored at all - logged once."""
        if options is None:
            return self._default_max_vel_rad_s

        has_acceleration_limit = len(options.max_acc_degs_per_sec2_joints) > 0 or (
            options.HasField("max_acc_degs_per_sec2")
        )
        if not self._has_warned_options and (
            has_acceleration_limit or options.HasField("max_tcp_speed")
        ):
            self.logger.info(
                "arm %s: MoveOptions acceleration limits and max_tcp_speed are not honored",
                self.name,
            )
            self._has_warned_options = True

        if len(options.max_vel_degs_per_sec_joints) > 0:
            return math.radians(min(options.max_vel_degs_per_sec_joints))
        if options.HasField("max_vel_degs_per_sec"):
            return math.radians(options.max_vel_degs_per_sec)
        return self._default_max_vel_rad_s

    async def move_through_joint_positions(
        self,
        positions: Sequence[JointPositions],
        options: MoveOptions | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """Execute a trajectory - this is what the motion service calls to run
        its planned paths. Intermediate waypoints use a loose tolerance so the
        arm flows through them and a short deadline that warns and continues
        on timeout (an obstacle blocking the path raises, since it won't
        clear itself); the final waypoint settles tight against the move
        deadline."""
        handle = self._h()
        waypoints = list(positions)
        max_vel_rad_s = self._max_vel_rad_s(options)
        move_deadline_s = self._deadline_s(timeout)
        consecutive_stalls = 0
        try:
            for i, wp in enumerate(waypoints):
                targets = [math.radians(v) for v in wp.values]
                current = await asyncio.to_thread(handle.get_joint_positions)
                if len(current) != len(targets):
                    raise JointTargetOutOfLimitsError(
                        f"arm {self.name}: expected {len(current)} joint values, got {len(targets)}"
                    )
                targets = await self._clamped_joint_targets(targets)
                await asyncio.to_thread(handle.set_joint_targets, targets, max_vel_rad_s)

                last = i == len(waypoints) - 1
                tolerance = _TOLERANCE_RAD if last else _WAYPOINT_TOLERANCE_RAD
                deadline_s = move_deadline_s if last else _WAYPOINT_DEADLINE_S

                outcome = await asyncio.to_thread(handle.wait_for_settle, deadline_s, tolerance)
                if outcome is SettleOutcome.REACHED:
                    consecutive_stalls = 0
                    continue

                current = await asyncio.to_thread(handle.get_joint_positions)
                detail = _stuck_joint_detail(current, targets, tolerance)
                stalled = outcome is SettleOutcome.STALLED
                consecutive_stalls = consecutive_stalls + 1 if stalled else 0
                # A constrained path arrives as dozens of waypoints a couple of
                # millimetres apart. The arm reaches each one, goes still, and
                # sits on a residual a little outside the loose waypoint
                # tolerance, which is indistinguishable from a blocked arm at
                # that one waypoint. Only a run of them tells the two apart, so
                # an isolated stall flows on to the next waypoint and a run of
                # them fails. Measured on a 46-waypoint 100 mm lift, where
                # single waypoints stalled 2.5 degrees outside a 2 degree
                # tolerance and the path was clear.
                if not last and consecutive_stalls < _MAX_CONSECUTIVE_WAYPOINT_STALLS:
                    self.logger.warning(
                        "%s: waypoint %d/%d not reached, continuing (%s)",
                        self.name,
                        i + 1,
                        len(waypoints),
                        detail,
                    )
                    continue

                # hold here rather than keep pushing at the unreachable target
                await asyncio.to_thread(handle.stop)
                if stalled:
                    raise ArmMoveStalledError(
                        f"arm {self.name} stalled at waypoint {i + 1}/{len(waypoints)} "
                        f"({consecutive_stalls} consecutive, stuck joints: {detail})"
                    )
                raise ArmMoveTimeoutError(
                    f"arm {self.name} did not reach final waypoint within "
                    f"{deadline_s:.1f}s (stuck joints: {detail})"
                )
        except asyncio.CancelledError:
            # a dropped RPC holds position instead of continuing toward
            # whatever waypoint was in flight
            await asyncio.to_thread(handle.stop)
            raise

    async def get_joint_positions(self, **kwargs) -> JointPositions:
        radians = await asyncio.to_thread(self._h().get_joint_positions)
        degrees = [math.degrees(r) for r in radians]
        limits = await self._joint_limits_deg()
        if limits is not None:
            for i, deg in enumerate(degrees):
                if i >= len(limits):
                    break
                _joint_id, min_deg, max_deg = limits[i]
                # physics settle drifts micro-degrees past a limit; report the
                # limit itself so the planner never plans from an illegal
                # state. Drift is checked before winding, since a joint resting
                # a hair past its limit is at that limit, not a turn away.
                if min_deg - _JOINT_LIMIT_TOLERANCE_DEG <= deg < min_deg:
                    deg = min_deg
                elif max_deg < deg <= max_deg + _JOINT_LIMIT_TOLERANCE_DEG:
                    deg = max_deg
                else:
                    deg = _wrapped_into_range(deg, min_deg, max_deg)
                degrees[i] = deg
        return JointPositions(values=degrees)

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    def _kinematics_url(self) -> str | None:
        url = self._attrs.get("kinematics_url")
        if url:
            return str(url)
        asset = self._attrs.get("asset")
        if asset and asset in KNOWN_ASSETS:
            return KNOWN_ASSETS[asset].get("kinematics")
        return None

    def _load_kinematics(self) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        url = self._kinematics_url()
        if not url:
            raise NotImplementedError(
                f"no kinematics file known for arm {self.name}; set the "
                '"kinematics_url" attribute (SVA .json or .urdf)'
            )
        ext = os.path.splitext(url)[1].lower()
        fmt = (
            KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
            if ext in (".urdf", ".xml", ".xacro")
            else KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        )

        cache_dir = os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()
        cache = os.path.join(
            cache_dir,
            f"kinematics-{hashlib.sha1(url.encode()).hexdigest()[:12]}{ext}",
        )
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return fmt, f.read()

        self.logger.info("fetching kinematics for %s from %s", self.name, url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cache + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, cache)
        except OSError as e:
            self.logger.warning("could not cache kinematics for %s at %s: %s", self.name, cache, e)
        return fmt, data

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            self._kinematics = await asyncio.to_thread(self._load_kinematics)
        return self._kinematics

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        """Deliberately empty. rdk builds arm geometry from GetKinematics (the
        SVA already carries the link capsules) and never calls Geometries for
        arms."""
        return []

    async def get_3d_models(self, **kwargs) -> Mapping[str, Any]:
        """The RDK API routes this method, and the Python SDK's ArmRPCService
        at 0.80.0 serves no handler for it. Defined so IsaacArm stays
        instantiable on an SDK that declares it abstract."""
        return {}

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        """No sim-only verbs live here. The world's DoCommand answers
        joint_state/dof_names/prim_pose so a real driver's do_command never
        grows a verb it can't answer."""
        raise MethodNotImplementedError("do_command")
