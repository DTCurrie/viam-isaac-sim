import math
from typing import Any

from viam.proto.app.robot import ComponentConfig
from viam.utils import struct_to_dict

from ..spatial import Quat, Vec3, ov_to_quat, quat_from_axis_angle, quat_mul


def get_attrs(config: ComponentConfig) -> dict[str, Any]:
    return struct_to_dict(config.attributes)


def frame_pose(config: ComponentConfig) -> tuple[Vec3 | None, Quat | None]:
    """Spawn pose from the component's standard frame config: position in
    meters (frame translations are mm) and a (w,x,y,z) quaternion, or None
    for whatever isn't set. The frame system is the preferred way to place
    isaac-sim components - it keeps viam's view of the machine (e.g. the
    motion service) consistent with where prims actually are in the sim."""
    if not config.HasField("frame"):
        return None, None
    frame = config.frame
    t = frame.translation
    position = (t.x / 1000.0, t.y / 1000.0, t.z / 1000.0)

    quat: Quat | None = None
    o = frame.orientation
    which = o.WhichOneof("type")
    if which == "quaternion":
        q = o.quaternion
        quat = (q.w, q.x, q.y, q.z)
    elif which == "vector_radians":
        v = o.vector_radians
        quat = ov_to_quat(v.x, v.y, v.z, v.theta)
    elif which == "vector_degrees":
        v = o.vector_degrees
        quat = ov_to_quat(v.x, v.y, v.z, math.radians(v.theta))
    elif which == "euler_angles":
        e = o.euler_angles  # radians, applied as Rz(yaw)*Ry(pitch)*Rx(roll)
        qz = (math.cos(e.yaw / 2), 0.0, 0.0, math.sin(e.yaw / 2))
        qy = (math.cos(e.pitch / 2), 0.0, math.sin(e.pitch / 2), 0.0)
        qx = (math.cos(e.roll / 2), math.sin(e.roll / 2), 0.0, 0.0)
        quat = quat_mul(qz, quat_mul(qy, qx))
    elif which == "axis_angles":
        a = o.axis_angles
        quat = quat_from_axis_angle((a.x, a.y, a.z), a.theta)
    return position, quat


def apply_frame_to_attrs(config: ComponentConfig, attrs: dict[str, Any]) -> dict[str, Any]:
    """Fold the frame config into the spawn attrs (frame wins over the
    legacy position/orientation attributes).

    When "parent_prim" is set the component rides another prim, so the frame
    describes a LOCAL pose relative to that prim, not a world pose: it is
    written to local_position/local_orientation_wxyz instead of
    position/orientation_wxyz. A frame and a legacy pose attribute are
    mutually exclusive, so attrs never carries both a local and a world
    pose key."""
    position, quat = frame_pose(config)
    if attrs.get("parent_prim"):
        if position is not None:
            attrs["local_position"] = list(position)
        if quat is not None or config.HasField("frame"):
            attrs["local_orientation_wxyz"] = list(quat or (1.0, 0.0, 0.0, 0.0))
        return attrs
    if position is not None:
        attrs["position"] = list(position)
    if quat is not None:
        attrs["orientation_wxyz"] = list(quat)
    return attrs


def _prim_root(parent_prim: str) -> str:
    """The prim segment that owns parent_prim: the segment right after a
    leading /World/, or the first segment when the path doesn't start with
    /World/."""
    parts = [p for p in parent_prim.split("/") if p]
    if parent_prim.startswith("/World/") and len(parts) >= 2:
        return parts[1]
    return parts[0]
