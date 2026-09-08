"""Forward and inverse kinematics over the kinematics file an arm serves.

The chain is parsed from the same SVA JSON or URDF bytes `GetKinematics`
returns, so the joints the sim drives to for a pose are the joints Viam's
own kinematics say correspond to it. `GetEndPosition`, the frame system and
the motion service then agree by construction. Pure Python and numpy. No
Isaac import, so the mock test suite covers all of it.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from viam.components.arm import KinematicsFileFormat

from .spatial import (
    Quat,
    Vec3,
    ov_to_quat,
    quat_conj,
    quat_from_axis_angle,
    quat_from_euler_deg,
    quat_mul,
    quat_rotate,
)

IK_MAX_ITERATIONS = 200
IK_POSITION_TOL_M = 1e-4
IK_ORIENTATION_TOL_RAD = 1e-3
IK_DAMPING = 1e-2
# Largest change of any one joint per iteration. From a singular start (a UR at zero joints
# is fully stretched) an uncapped damped-least-squares step winds a joint through several turns.
IK_MAX_STEP_RAD = 0.5

_MM_PER_M = 1000.0
_IDENTITY_POS: Vec3 = (0.0, 0.0, 0.0)
_IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
_JACOBIAN_STEP_RAD = 1e-6
_JOINT_LIMIT_SLACK_RAD = 1e-6
_URDF_CONTINUOUS_LIMIT_RAD = 2 * math.pi


class IKError(ValueError):
    """Base class for inverse-kinematics failures."""


class UnreachablePoseError(IKError):
    """The solver did not converge on the target pose within the iteration cap."""


class JointLimitError(IKError):
    """The solver converged, but the solution lies outside a joint's declared limits."""


@dataclass(frozen=True)
class Joint:
    """One actuated joint of the chain, in the order the file declares them.

    `axis` is a unit vector in the joint's parent link frame. Limits are in
    radians. `type` is `"revolute"` or `"prismatic"`.
    """

    id: str
    type: str
    axis: Vec3
    min_rad: float
    max_rad: float


# The fixed transform (metres, quaternion) from the previous joint's frame
# (or the base, for the first entry) up to and including a joint's own
# origin, where the joint's variable motion is applied.
_FixedFrame = tuple[Vec3, Quat]
_DEFAULT_FRAME: _FixedFrame = (_IDENTITY_POS, _IDENTITY_QUAT)


@dataclass(frozen=True)
class Chain:
    """A serial kinematic chain from the base link to the end-effector link.

    Positions are metres in the arm's base frame (the frame `GetEndPosition`
    reports in). Orientations are `(w, x, y, z)` quaternions. Joint vectors
    are radians in `joints` order, which is the order `GetJointPositions`
    and `MoveToJointPositions` use for the same file.
    """

    joints: tuple[Joint, ...]
    # One more entry than `joints`: `_frames[i]` precedes `joints[i]`'s
    # variable transform, `_frames[-1]` is the fixed offset to the
    # end-effector link after the last joint.
    _frames: tuple[_FixedFrame, ...] = field(default_factory=lambda: (_DEFAULT_FRAME,))

    @classmethod
    def from_kinematics(cls, fmt: KinematicsFileFormat.ValueType, data: bytes) -> Chain:
        """Parse whichever format `GetKinematics` served."""
        if fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA:
            return cls.from_sva(data)
        if fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF:
            return cls.from_urdf(data)
        raise ValueError(f"unsupported kinematics file format: {fmt}")

    @classmethod
    def from_sva(cls, data: bytes) -> Chain:
        """Parse Viam's SVA JSON (`kinematic_param_type: "SVA"`, `links`, `joints`)."""
        parsed = json.loads(data)
        links = parsed.get("links", [])
        joints = parsed.get("joints", [])

        roots = [link for link in links if link.get("parent") == "world"]
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one root link with parent 'world', found {len(roots)}"
            )

        frames: list[_FixedFrame] = []
        chain_joints: list[Joint] = []
        current = roots[0]
        while True:
            frames.append(_sva_link_frame(current))
            outgoing = [j for j in joints if j.get("parent") == current["id"]]
            if not outgoing:
                break
            if len(outgoing) > 1:
                raise ValueError(f"branching kinematic chain at link {current['id']!r}")
            joint = outgoing[0]
            children = [link for link in links if link.get("parent") == joint["id"]]
            if len(children) != 1:
                raise ValueError(f"branching kinematic chain at joint {joint['id']!r}")
            chain_joints.append(
                Joint(
                    id=joint["id"],
                    type=joint["type"],
                    axis=_unit_vec3(joint["axis"], joint["id"]),
                    min_rad=math.radians(joint["min"]),
                    max_rad=math.radians(joint["max"]),
                )
            )
            current = children[0]

        return cls(joints=tuple(chain_joints), _frames=tuple(frames))

    @classmethod
    def from_urdf(cls, data: bytes) -> Chain:
        """Parse a URDF with stdlib `xml`, joints in tree order from the root link."""
        root = ET.fromstring(data)
        joint_elements = root.findall("joint")
        link_names = {link.get("name") for link in root.findall("link")}
        child_names = {_urdf_link_name(j, "child") for j in joint_elements}

        root_candidates = link_names - child_names
        if len(root_candidates) != 1:
            raise ValueError(f"expected exactly one root link, found {len(root_candidates)}")
        current_name = next(iter(root_candidates))

        frames: list[_FixedFrame] = []
        chain_joints: list[Joint] = []
        accumulated_pos, accumulated_quat = _IDENTITY_POS, _IDENTITY_QUAT
        while True:
            outgoing = [j for j in joint_elements if _urdf_link_name(j, "parent") == current_name]
            if not outgoing:
                frames.append((accumulated_pos, accumulated_quat))
                break
            if len(outgoing) > 1:
                raise ValueError(f"branching kinematic chain at link {current_name!r}")
            joint = outgoing[0]
            joint_name = _urdf_joint_name(joint)
            origin_pos, origin_quat = _urdf_origin(joint.find("origin"))
            accumulated_pos, accumulated_quat = _compose(
                accumulated_pos, accumulated_quat, origin_pos, origin_quat
            )
            joint_type = joint.get("type")
            child_name = _urdf_link_name(joint, "child")
            if joint_type == "fixed":
                current_name = child_name
                continue

            frames.append((accumulated_pos, accumulated_quat))
            axis_element = joint.find("axis")
            axis = (
                _unit_vec3(_parse_xyz(axis_element.get("xyz")), joint_name)
                if axis_element is not None
                else (0.0, 0.0, 1.0)
            )
            if joint_type == "continuous":
                min_rad, max_rad = -_URDF_CONTINUOUS_LIMIT_RAD, _URDF_CONTINUOUS_LIMIT_RAD
            else:
                limit_element = joint.find("limit")
                min_rad = (
                    float(limit_element.get("lower", 0.0)) if limit_element is not None else 0.0
                )
                max_rad = (
                    float(limit_element.get("upper", 0.0)) if limit_element is not None else 0.0
                )
            chain_type = "prismatic" if joint_type == "prismatic" else "revolute"
            chain_joints.append(
                Joint(id=joint_name, type=chain_type, axis=axis, min_rad=min_rad, max_rad=max_rad)
            )
            accumulated_pos, accumulated_quat = _IDENTITY_POS, _IDENTITY_QUAT
            current_name = child_name

        return cls(joints=tuple(chain_joints), _frames=tuple(frames))

    @property
    def dof(self) -> int:
        return len(self.joints)

    def fk(self, q_rad: Sequence[float]) -> tuple[Vec3, Quat]:
        """End-effector pose in the base frame for the joint vector `q_rad`."""
        pos, quat = _IDENTITY_POS, _IDENTITY_QUAT
        for joint, angle, frame in zip(self.joints, q_rad, self._frames, strict=False):
            pos, quat = _compose(pos, quat, frame[0], frame[1])
            var_pos, var_quat = _joint_transform(joint, angle)
            pos, quat = _compose(pos, quat, var_pos, var_quat)
        pos, quat = _compose(pos, quat, self._frames[-1][0], self._frames[-1][1])
        return pos, quat

    def ik(self, target_pos: Vec3, target_quat: Quat, q0_rad: Sequence[float]) -> list[float]:
        """Joint vector whose `fk` matches the target within the tolerances above.

        Damped least squares starting from `q0_rad`, so of several solutions
        the one nearest the current joints is returned by construction.
        Raises `UnreachablePoseError` when the solver does not converge in
        `IK_MAX_ITERATIONS`, and `JointLimitError` when it converges outside
        a joint's limits. Limits are never clamped.
        """
        q0 = np.array(q0_rad, dtype=float)
        q = q0.copy()
        damping_sq_eye = (IK_DAMPING**2) * np.eye(6)
        revolute = np.array([joint.type == "revolute" for joint in self.joints], dtype=bool)

        for _ in range(IK_MAX_ITERATIONS):
            pos, quat = self.fk(q.tolist())
            error = _pose_error(pos, quat, target_pos, target_quat)
            if (
                np.linalg.norm(error[:3]) < IK_POSITION_TOL_M
                and np.linalg.norm(error[3:]) < IK_ORIENTATION_TOL_RAD
            ):
                return self._checked_joint_vector(q, q0)
            jacobian = self._numeric_jacobian(q)
            dq: np.ndarray = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping_sq_eye, error
            )
            largest = float(np.max(np.abs(dq)))
            if largest > IK_MAX_STEP_RAD:
                dq = dq * (IK_MAX_STEP_RAD / largest)
            q = q + dq
            # A revolute joint's pose is periodic, so keep its angle within one half turn
            # of the start. The pose error is unchanged and the angle cannot wind up.
            q = np.where(revolute, q0 + _wrap_to_pi(q - q0), q)

        raise UnreachablePoseError(f"IK did not converge within {IK_MAX_ITERATIONS} iterations")

    def _numeric_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Central-difference Jacobian: column i is d(pose)/d(q_i)."""
        jacobian = np.zeros((6, self.dof))
        for i in range(self.dof):
            q_plus = q.copy()
            q_plus[i] += _JACOBIAN_STEP_RAD
            q_minus = q.copy()
            q_minus[i] -= _JACOBIAN_STEP_RAD
            pos_plus, quat_plus = self.fk(q_plus.tolist())
            pos_minus, quat_minus = self.fk(q_minus.tolist())
            d_pos = np.array(pos_plus) - np.array(pos_minus)
            d_rot = np.array(_rotation_vector(quat_mul(quat_plus, quat_conj(quat_minus))))
            jacobian[:, i] = np.concatenate([d_pos, d_rot]) / (2 * _JACOBIAN_STEP_RAD)
        return jacobian

    def _checked_joint_vector(self, q: np.ndarray, q0: np.ndarray) -> list[float]:
        """The solution with every joint inside its limits, or `JointLimitError`.

        A revolute angle is the same pose at every multiple of two pi, so of the
        equivalents inside the joint's limits the one nearest the start is kept.
        Limits are never clamped: a joint with no in-limit equivalent raises.
        """
        checked: list[float] = []
        for joint, angle, start in zip(self.joints, q, q0, strict=True):
            candidates = [float(angle)]
            if joint.type == "revolute":
                candidates = [float(angle) + 2.0 * math.pi * k for k in range(-3, 4)]
            in_limits = [
                c
                for c in candidates
                if joint.min_rad - _JOINT_LIMIT_SLACK_RAD
                <= c
                <= joint.max_rad + _JOINT_LIMIT_SLACK_RAD
            ]
            if not in_limits:
                raise JointLimitError(
                    f"joint {joint.id!r} solution {float(angle):.6f} rad has no equivalent inside "
                    f"[{joint.min_rad:.6f}, {joint.max_rad:.6f}]"
                )
            checked.append(min(in_limits, key=lambda c: abs(c - float(start))))
        return checked


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def _compose(pos: Vec3, quat: Quat, local_pos: Vec3, local_quat: Quat) -> tuple[Vec3, Quat]:
    rx, ry, rz = quat_rotate(quat, local_pos)
    new_pos: Vec3 = (pos[0] + rx, pos[1] + ry, pos[2] + rz)
    return new_pos, quat_mul(quat, local_quat)


def _joint_transform(joint: Joint, angle: float) -> tuple[Vec3, Quat]:
    if joint.type == "revolute":
        return _IDENTITY_POS, quat_from_axis_angle(joint.axis, angle)
    if joint.type == "prismatic":
        ax, ay, az = joint.axis
        translation: Vec3 = (ax * angle, ay * angle, az * angle)
        return translation, _IDENTITY_QUAT
    raise ValueError(f"unknown joint type {joint.type!r} for joint {joint.id!r}")


def _rotation_vector(q: Quat) -> Vec3:
    """Axis*angle (radians) for the rotation `q` represents, shortest way round."""
    w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    v_norm = math.sqrt(x * x + y * y + z * z)
    if v_norm < 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2 * math.atan2(v_norm, w)
    scale = angle / v_norm
    return (x * scale, y * scale, z * scale)


def _pose_error(pos: Vec3, quat: Quat, target_pos: Vec3, target_quat: Quat) -> np.ndarray:
    pos_error = np.array(target_pos) - np.array(pos)
    rot_error = np.array(_rotation_vector(quat_mul(target_quat, quat_conj(quat))))
    return np.concatenate([pos_error, rot_error])


def _unit_vec3(value: dict | Sequence[float], context: str) -> Vec3:
    if isinstance(value, dict):
        x, y, z = float(value["x"]), float(value["y"]), float(value["z"])
    else:
        x, y, z = (float(v) for v in value)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        raise ValueError(f"zero-length axis for {context!r}")
    return (x / norm, y / norm, z / norm)


def _sva_link_frame(link: dict) -> _FixedFrame:
    translation = link.get("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    pos = (
        translation.get("x", 0.0) / _MM_PER_M,
        translation.get("y", 0.0) / _MM_PER_M,
        translation.get("z", 0.0) / _MM_PER_M,
    )
    orientation = link.get("orientation")
    quat = _parse_sva_orientation(orientation, link["id"]) if orientation else _IDENTITY_QUAT
    return pos, quat


def _parse_sva_orientation(orientation: dict, context: str) -> Quat:
    kind = orientation["type"]
    value = orientation["value"]
    if kind == "ov_degrees":
        return ov_to_quat(value["x"], value["y"], value["z"], math.radians(value["th"]))
    if kind == "ov_radians":
        return ov_to_quat(value["x"], value["y"], value["z"], value["th"])
    if kind == "quaternion":
        w = value.get("w", value.get("W"))
        x = value.get("x", value.get("X"))
        y = value.get("y", value.get("Y"))
        z = value.get("z", value.get("Z"))
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        return (w / norm, x / norm, y / norm, z / norm)
    if kind == "euler_angles":
        return quat_from_euler_deg(
            math.degrees(value["roll"]), math.degrees(value["pitch"]), math.degrees(value["yaw"])
        )
    if kind == "axis_angles":
        return quat_from_axis_angle((value["x"], value["y"], value["z"]), value["th"])
    raise ValueError(f"unknown orientation type {kind!r} for {context!r}")


def _parse_xyz(text: str | None) -> Vec3:
    if not text:
        return _IDENTITY_POS
    x, y, z = (float(v) for v in text.split())
    return (x, y, z)


def _urdf_joint_name(joint: ET.Element) -> str:
    name = joint.get("name")
    if name is None:
        raise ValueError("URDF joint element missing 'name' attribute")
    return name


def _urdf_link_name(joint: ET.Element, tag: str) -> str:
    element = joint.find(tag)
    if element is None:
        raise ValueError(f"joint {_urdf_joint_name(joint)!r} missing <{tag}>")
    link = element.get("link")
    if link is None:
        raise ValueError(f"joint {_urdf_joint_name(joint)!r} <{tag}> missing 'link' attribute")
    return link


def _urdf_origin(origin_element: ET.Element | None) -> _FixedFrame:
    if origin_element is None:
        return _IDENTITY_POS, _IDENTITY_QUAT
    xyz = _parse_xyz(origin_element.get("xyz"))
    roll, pitch, yaw = _parse_xyz(origin_element.get("rpy"))
    quat = quat_from_euler_deg(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    return xyz, quat
