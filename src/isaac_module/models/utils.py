from typing import Any, Dict, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.utils import struct_to_dict


def get_attrs(config: ComponentConfig) -> Dict[str, Any]:
    return struct_to_dict(config.attributes)


def validate_sim_component(
    config: ComponentConfig, needs_source: bool = True
) -> Tuple[Sequence[str], Sequence[str]]:
    """Shared validation for arm/camera/base: they must name their world
    component (so viam-server starts it first) and, when they spawn a prim,
    say what to spawn."""
    attrs = struct_to_dict(config.attributes)
    world = attrs.get("world")
    if not world or not isinstance(world, str):
        raise ValueError(
            f'{config.name}: set the "world" attribute to the name of your '
            "erh:isaac-sim:world component"
        )
    if needs_source and not (
        attrs.get("asset") or attrs.get("usd_path") or attrs.get("prim_path")
    ):
        raise ValueError(
            f'{config.name}: set "asset" (e.g. "ur20"), "usd_path", or '
            '"prim_path" (to attach to something already in the stage)'
        )
    return [world], []
