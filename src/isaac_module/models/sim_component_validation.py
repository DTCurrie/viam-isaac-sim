from collections.abc import Sequence
from typing import Any

from viam.proto.app.robot import ComponentConfig
from viam.utils import struct_to_dict

from .. import DEFAULT_WORLD_NAME, FAMILY, NAMESPACE
from ..sim_manager import prim_name
from .component_frame_pose import _prim_root


def _validate_parent_prim_frame(config: ComponentConfig, attrs: dict[str, Any]) -> None:
    """A component riding another prim (parent_prim set) must agree with
    Viam about who owns that prim: frame.parent names the component whose
    prim it is, so viam's view (e.g. the motion service) matches the sim.
    When frame.parent also carries a link (<arm-name>:<link-name>), that
    link must be the same one parent_prim ends in, or Viam and the sim would
    silently disagree about which joint the mount rides."""
    parent_prim = attrs["parent_prim"]
    if not config.HasField("frame") or config.frame.parent in ("", "world"):
        raise ValueError(
            f'{config.name}: "parent_prim" is set, so set frame.parent to the '
            f"{NAMESPACE}:{FAMILY} component that owns that prim (frame.parent "
            'cannot be "world" for a mounted component)'
        )
    parent = config.frame.parent
    owner, _, link = parent.partition(":")
    root = _prim_root(parent_prim)
    if prim_name(owner) != root:
        raise ValueError(
            f"{config.name}: frame.parent {parent!r} does not own parent_prim "
            f"{parent_prim!r} (root prim {root!r}); set frame.parent to the "
            "component whose prim that is"
        )
    if link:
        last_segment = parent_prim.rstrip("/").rsplit("/", 1)[-1]
        if link != last_segment:
            raise ValueError(
                f"{config.name}: frame.parent {parent!r} names link {link!r}, "
                f"but parent_prim {parent_prim!r} ends in {last_segment!r}; "
                "the link half of frame.parent must match parent_prim's last segment"
            )


def validate_sim_component(
    config: ComponentConfig, needs_source: bool = True
) -> tuple[Sequence[str], Sequence[str]]:
    """Shared validation for arm/camera/base: they must name their world
    component (so viam-server starts it first, defaulting to
    DEFAULT_WORLD_NAME) and, when they spawn a prim, say what to spawn. A
    component riding another prim (parent_prim) also depends on the component
    that owns that prim, so viam-server builds the owner first - siblings
    build concurrently, and a mounted camera built before its arm fails with
    PrimNotFoundError. A bare arm parent (no ":<link>" suffix) means the
    arm's end-effector frame; see _validate_parent_prim_frame for the link
    form."""
    attrs = struct_to_dict(config.attributes)
    world = attrs.get("world", DEFAULT_WORLD_NAME)
    if not world or not isinstance(world, str):
        raise ValueError(
            f'{config.name}: "world" defaults to "{DEFAULT_WORLD_NAME}" and, when set, '
            "must be a non-empty string naming your "
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
        _reject_pose_attributes_beside_frame(config, attrs)
    return [world], []


# The spawn pose comes from the frame when one is set, so a pose attribute beside
# it would be silently ignored: a camera configured at [1.2, 1.2, 0.9] with a
# translation-less frame spawned at the origin, inside the arm, and rendered
# black with no error (GPU machine, 2026-09-08).
_POSE_ATTRIBUTES = ("position", "orientation_rpy_deg", "orientation_wxyz")


def _reject_pose_attributes_beside_frame(config: ComponentConfig, attrs: dict[str, Any]) -> None:
    present = [key for key in _POSE_ATTRIBUTES if key in attrs]
    if present:
        raise ValueError(
            f"{config.name}: {', '.join(present)} would be ignored because a frame is set. "
            "Put the pose in frame.translation (mm) and frame.orientation, or remove the frame"
        )
