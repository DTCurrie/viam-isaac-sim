"""viam:isaac-sim-devin:arm - a simulated arm.

Attributes:
  world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
  asset (string)             - known robot, e.g. "ur20", "ur10", "franka"
  usd_path (string)          - explicit USD to spawn instead of a known asset
  prim_path (string)         - where to place it (default /World/<name>), or
                               an existing articulation in the stage
  position ([x,y,z] meters)  - spawn position
  end_effector_prim (string) - prim whose pose, in the arm base frame, is
                               reported by GetEndPosition (default
                               <arm prim>/wrist_3_link for UR assets)
  move_timeout_sec (float)   - max time to wait for a move (default 30)
  kinematics_url (string)    - where to fetch the kinematics file served by
                               GetKinematics (.json = SVA, .urdf = URDF;
                               file:// URLs work). Known assets with official
                               viam kinematics (ur3e/ur5e/ur20) fetch them
                               automatically.

Note: GetEndPosition reports the end effector pose in the arm base frame
(not world frame) as of this release.

Move completion: moves settle via
ArmHandle.wait_for_settle - no wall-clock polling - and raise one of:
  JointTargetOutOfLimitsError (ValueError, INVALID_ARGUMENT)  - a target is
    outside the SVA's declared joint limits, or the joint count doesn't
    match the arm's DOF count.
  ArmMoveStalledError (ABORTED)      - the arm stopped moving before
    reaching its target (e.g. blocked by an obstacle).
  ArmMoveTimeoutError (TimeoutError, DEADLINE_EXCEEDED) - the move deadline
    (move_timeout_sec, capped by the SDK's timeout= kwarg) passed while the
    arm was still converging.
MoveToPosition solves against the served kinematics and drives the joint
path, so the frame system, the motion service and the sim agree by
construction.
move_through_joint_positions honours MoveOptions.max_vel_degs_per_sec_joints
(the min across joints) when set, else max_vel_degs_per_sec; the
acceleration fields and max_tcp_speed are logged once and not honoured.
DoCommand answers no sim-only verbs; the world component's DoCommand
(joint_state/dof_names/prim_pose) reads this arm's sim state instead.

close() releases the handle and its post-reset hooks. The prim stays
in the stage. A reconfigure that changes a spawn attribute (asset, usd_path,
prim_path, position, or the frame it derives from) after the arm is already
attached raises ValueError - restart the module to apply it.
"""

import asyncio
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.arm import Arm, JointPositions, KinematicsFileFormat, Pose
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.proto.component.arm import MoveOptions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..kinematics import Chain, JointLimitError, UnreachablePoseError
from ..sim_manager import (
    KNOWN_ASSETS,
    SETTLE_TOL_RAD,
    ArmHandle,
    SettleOutcome,
    SimManager,
)
from ..spatial import ov_to_quat, quat_to_ov
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_TOLERANCE_RAD = SETTLE_TOL_RAD
_WAYPOINT_TOLERANCE_RAD = math.radians(2.0)
_WAYPOINT_DEADLINE_S = 10.0
# observed settle drift past a limit is ~3e-5 deg (wrist_2 at
# -360.00003); 0.01 covers it by orders of magnitude while a genuinely wrong
# target still raises
_JOINT_LIMIT_TOLERANCE_DEG = 0.01
_MM_PER_M = 1000.0
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


def _stuck_joint_detail(
    current: Sequence[float], targets: Sequence[float], tolerance_rad: float
) -> str:
    """ "jN: at <deg> want <deg>" for every joint outside tolerance."""
    return ", ".join(
        f"j{i}: at {math.degrees(c):.1f} want {math.degrees(t):.1f}"
        for i, (c, t) in enumerate(zip(current, targets, strict=True))
        if abs(c - t) > tolerance_rad
    )


class IsaacArm(Arm, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: ArmHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._move_timeout = 30.0
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
        return validate_sim_component(config)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._move_timeout = float(attrs.get("move_timeout_sec", 30.0))
        self._attrs = attrs
        self._handle = SimManager.get().create_arm(self.name, attrs)
        self._kinematics_chain = None

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
            x=x * 1000.0,
            y=y * 1000.0,
            z=z * 1000.0,
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
        target_pos = (pose.x / _MM_PER_M, pose.y / _MM_PER_M, pose.z / _MM_PER_M)
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
            except Exception:
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
        except Exception:
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
        # keeps the arm pushing into whatever blocked it (GPU run 15 launched
        # the block and wound the elbow up that way).
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
        await asyncio.to_thread(handle.set_joint_targets, targets, None)

        await self._settle_or_raise(
            handle,
            targets,
            self._deadline_s(timeout),
            _TOLERANCE_RAD,
            detail_prefix=f"arm {self.name}",
        )

    def _max_vel_rad_s(self, options: MoveOptions | None) -> float | None:
        """The per-joint max_vel_degs_per_sec_joints, when non-empty, is the
        ONLY velocity limit honoured (the scalar is ignored in that
        case); the handle only takes one scalar, so the min across joints is
        used. Otherwise MoveOptions.max_vel_degs_per_sec applies when set.
        None = the drive's own limit. Acceleration fields and max_tcp_speed
        follow the same per-joint-wins precedence but aren't honoured at
        all - logged once."""
        if options is None:
            return None

        has_acceleration_limit = len(options.max_acc_degs_per_sec2_joints) > 0 or (
            options.HasField("max_acc_degs_per_sec2")
        )
        if not self._has_warned_options and (
            has_acceleration_limit or options.HasField("max_tcp_speed")
        ):
            self.logger.info(
                "arm %s: MoveOptions acceleration limits and max_tcp_speed are not honoured",
                self.name,
            )
            self._has_warned_options = True

        if len(options.max_vel_degs_per_sec_joints) > 0:
            return math.radians(min(options.max_vel_degs_per_sec_joints))
        if options.HasField("max_vel_degs_per_sec"):
            return math.radians(options.max_vel_degs_per_sec)
        return None

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
                continue

            current = await asyncio.to_thread(handle.get_joint_positions)
            detail = _stuck_joint_detail(current, targets, tolerance)
            if outcome is SettleOutcome.STALLED or last:
                # hold here rather than keep pushing at the unreachable target
                await asyncio.to_thread(handle.stop)
            if outcome is SettleOutcome.STALLED:
                raise ArmMoveStalledError(
                    f"arm {self.name} stalled at waypoint {i + 1}/{len(waypoints)} "
                    f"(stuck joints: {detail})"
                )
            # TIMED_OUT
            if last:
                raise ArmMoveTimeoutError(
                    f"arm {self.name} did not reach final waypoint within {deadline_s:.1f}s "
                    f"(stuck joints: {detail})"
                )
            self.logger.warning(
                "%s: waypoint %d/%d not reached, continuing (%s)",
                self.name,
                i + 1,
                len(waypoints),
                detail,
            )

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
                # limit itself so the planner never plans from an illegal state
                if min_deg - _JOINT_LIMIT_TOLERANCE_DEG <= deg < min_deg:
                    degrees[i] = min_deg
                elif max_deg < deg <= max_deg + _JOINT_LIMIT_TOLERANCE_DEG:
                    degrees[i] = max_deg
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
        except OSError:
            pass  # caching is best-effort
        return fmt, data

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            self._kinematics = await asyncio.to_thread(self._load_kinematics)
        return self._kinematics

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        # Deliberately empty: rdk builds arm geometry from GetKinematics (the
        # SVA already carries the link capsules) and never calls Geometries
        # for arms.
        return []

    # Abstract on viam-sdk "main" but absent at the 0.80.0 floor pinned in
    # requirements.txt; implementing it keeps IsaacArm instantiable across
    # that bound.
    async def get_3d_models(self, **kwargs) -> Mapping[str, Any]:
        return {}

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        # No sim-only verbs live here: the world's DoCommand answers
        # joint_state/dof_names/prim_pose so a real driver's do_command
        # never grows a verb it can't answer.
        raise ValueError(f"unknown command: {command}")
