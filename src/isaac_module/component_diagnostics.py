"""Per-component sim readouts for the world's diagnostic DoCommand verbs.

The world is the one resource allowed to know it is a simulator, so its
per-component verbs (joint_state, dof_names, prim_pose, tcp_pose, jaw_deg)
live here rather than on the arm/gripper models, which keep only the verbs a
real driver could answer. Every function here takes the (attrs, handle) pair
`SimManager.handle_entry(name)` returns and returns the payload dict for the
matching DoCommand verb.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from .length_units import MM_PER_M, to_millimeters
from .models.gripper import DEFAULT_TCP_OFFSET_M
from .sim_manager import ArmHandle, GripperHandle
from .spatial import Quat, Vec3, quat_rotate, quat_to_ov


class _PrimPoseHandle(Protocol):
    """Anything the prim_pose verb can read a world pose from: today an
    ArmHandle or a BaseHandle."""

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]: ...


def joint_state(attrs: dict[str, Any], handle: ArmHandle) -> dict[str, Any]:
    state = handle.joint_state()
    joints = [
        {
            "name": entry["name"],
            "named": entry["named"],
            "position_deg": math.degrees(entry["position"]),
            "velocity_deg_s": math.degrees(entry["velocity"]),
            "target_deg": (None if entry["target"] is None else math.degrees(entry["target"])),
        }
        for entry in state
    ]
    return {"joints": joints}


def dof_names(
    attrs: dict[str, Any], handle: ArmHandle | GripperHandle, *, all_dofs: bool = False
) -> dict[str, Any]:
    if all_dofs:
        assert isinstance(handle, ArmHandle)
        return {"dof_names": list(handle.all_dof_names())}
    return {"dof_names": list(handle.dof_names())}


def prim_pose(attrs: dict[str, Any], handle: _PrimPoseHandle, prim_path: str) -> dict[str, Any]:
    (x, y, z), quat = handle.get_prim_world_pose(prim_path)
    ox, oy, oz, theta = quat_to_ov(quat)
    return {
        "prim_path": prim_path,
        "position_mm": [to_millimeters(x), to_millimeters(y), to_millimeters(z)],
        "quaternion_wxyz": list(quat),
        "orientation_vector": {
            "o_x": ox,
            "o_y": oy,
            "o_z": oz,
            "theta_deg": math.degrees(theta),
        },
    }


def jaw_deg(attrs: dict[str, Any], handle: GripperHandle) -> dict[str, Any]:
    open_rad, closed_rad = handle.jaw_limits()
    return {
        "jaw_deg": math.degrees(handle.get_jaw()),
        "open_deg": math.degrees(open_rad),
        "closed_deg": math.degrees(closed_rad),
    }


def tcp_pose(attrs: dict[str, Any], handle: GripperHandle) -> dict[str, Any]:
    """The fingertip midpoint's offset from the mount link along the tool
    +Z, in mm, next to the configured tcp_offset_m, so the TCP is corrected
    in one place if they differ."""
    tcp_offset_m = float(attrs.get("tcp_offset_m", DEFAULT_TCP_OFFSET_M))
    poses = handle.link_world_poses()
    out: dict[str, Any] = {"jaw_deg": math.degrees(handle.get_jaw())}
    out |= {
        key: {
            "position_mm": [to_millimeters(v) for v in pos],
            "quaternion_wxyz": list(quat),
        }
        for key, (pos, quat) in poses.items()
    }
    parent = poses.get("parent")
    if parent is None:
        out["error"] = "mount link pose unavailable"
        return out
    tool_axis = quat_rotate(parent[1], (0.0, 0.0, 1.0))

    def along_tool(point: tuple[float, ...]) -> float:
        return sum((q - p) * a for q, p, a in zip(point, parent[0], tool_axis, strict=True))

    left = poses.get("left_inner_finger")
    right = poses.get("right_inner_finger")
    if left is not None and right is not None:
        origin_mid = tuple((a + b) / 2.0 for a, b in zip(left[0], right[0], strict=True))
        # informational: this asset authors link frames at the base
        out["inner_finger_origin_offset_mm"] = to_millimeters(along_tool(origin_mid))

    bounds = handle.fingertip_world_bounds()
    out["fingertips"] = {
        side: {
            "min_mm": [to_millimeters(v) for v in low],
            "max_mm": [to_millimeters(v) for v in high],
            "center_mm": [(a + b) / 2.0 * MM_PER_M for a, b in zip(low, high, strict=True)],
        }
        for side, (low, high) in bounds.items()
    }
    if "left" not in bounds or "right" not in bounds:
        out["error"] = "fingertip pad meshes not found under the gripper"
        return out
    centers = [
        tuple((a + b) / 2.0 for a, b in zip(low, high, strict=True))
        for low, high in (bounds["left"], bounds["right"])
    ]
    pad_mid = tuple((a + b) / 2.0 for a, b in zip(centers[0], centers[1], strict=True))
    corners = [
        corner
        for low, high in bounds.values()
        for corner in (
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        )
    ]
    measured = along_tool(pad_mid)
    out["pad_center_midpoint_mm"] = [to_millimeters(v) for v in pad_mid]
    out["fingertip_reach_mm"] = to_millimeters(max(along_tool(c) for c in corners))
    out["jaw_gap_mm"] = to_millimeters(
        math.dist(centers[0], centers[1])
    )  # pad-center to pad-center, across the jaw
    out["measured_tcp_offset_mm"] = to_millimeters(measured)
    out["configured_tcp_offset_mm"] = to_millimeters(tcp_offset_m)
    out["delta_mm"] = to_millimeters(measured - tcp_offset_m)
    return out
