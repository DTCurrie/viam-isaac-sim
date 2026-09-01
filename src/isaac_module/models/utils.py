import math
from collections.abc import Sequence
from typing import Any

from viam.proto.app.robot import ComponentConfig
from viam.utils import struct_to_dict

from .. import FAMILY, NAMESPACE
from ..sim_manager import _prim_name
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
    position/orientation_wxyz (CAM-10 - mixing the two meant a world pose
    landed on a mounted camera)."""
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


def _validate_parent_prim_frame(config: ComponentConfig, attrs: dict[str, Any]) -> None:
    """A component riding another prim (parent_prim set) must agree with
    Viam about who owns that prim: frame.parent names the component whose
    prim it is, so viam's view (e.g. the motion service) matches the sim."""
    parent_prim = attrs["parent_prim"]
    if not config.HasField("frame") or config.frame.parent in ("", "world"):
        raise ValueError(
            f'{config.name}: "parent_prim" is set, so set frame.parent to the '
            f"{NAMESPACE}:{FAMILY} component that owns that prim (frame.parent "
            'cannot be "world" for a mounted component)'
        )
    parent = config.frame.parent
    owner = parent.split(":")[0]
    root = _prim_root(parent_prim)
    if _prim_name(owner) != root:
        raise ValueError(
            f"{config.name}: frame.parent {parent!r} does not own parent_prim "
            f"{parent_prim!r} (root prim {root!r}); set frame.parent to the "
            "component whose prim that is"
        )


def validate_sim_component(
    config: ComponentConfig, needs_source: bool = True
) -> tuple[Sequence[str], Sequence[str]]:
    """Shared validation for arm/camera/base: they must name their world
    component (so viam-server starts it first) and, when they spawn a prim,
    say what to spawn. A component riding another prim (parent_prim) also
    depends on the component that owns that prim, so viam-server builds the
    owner first - siblings build concurrently, and a mounted camera built
    before its arm fails with PrimNotFoundError (seen on the GPU, phase 3)."""
    attrs = struct_to_dict(config.attributes)
    world = attrs.get("world")
    if not world or not isinstance(world, str):
        raise ValueError(
            f'{config.name}: set the "world" attribute to the name of your '
            f"{NAMESPACE}:{FAMILY}:world component"
        )
    if needs_source and not (attrs.get("asset") or attrs.get("usd_path") or attrs.get("prim_path")):
        raise ValueError(
            f'{config.name}: set "asset" (e.g. "ur20"), "usd_path", or '
            '"prim_path" (to attach to something already in the stage)'
        )
    parent_prim = attrs.get("parent_prim")
    if parent_prim:
        _validate_parent_prim_frame(config, attrs)
        owner = config.frame.parent.split(":")[0]
        return [world, owner], []
    if config.HasField("frame"):
        parent = config.frame.parent
        if parent not in ("", "world"):
            raise ValueError(
                f'{config.name}: frame.parent must be "world" for spawned isaac-sim '
                f"components in this release (got {parent!r}); to ride another prim "
                'set "parent_prim"'
            )
    return [world], []
