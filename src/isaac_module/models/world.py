"""dtcurrie:isaac-sim:world - the generic component that owns the simulator.

Configure exactly one of these per machine. All other isaac-sim components
name it in their "world" attribute; their validate_config returns it as an
implicit dependency so viam-server boots the world first.

Attributes:
  mock (bool, default false)        - run without isaac sim (for dev/testing)
  headless (bool, default true)     - run kit without a local GUI window
  livestream (bool, default true)   - enable WebRTC livestreaming (view with
                                      the Isaac Sim WebRTC Streaming Client)
  livestream_public_ip (string)     - IP advertised to streaming clients;
                                      auto-detected if unset
  usd_stage (string)                - USD file/omniverse URL to open; if unset
                                      an empty stage with a ground plane is used
  physics_dt / rendering_dt (float) - sim step sizes, default 1/60
  boot_timeout_sec (float)          - how long to wait for kit to boot
  kit_log_level (string)            - kit console verbosity, default "warning"
  props (list)                      - objects spawned into the scene at boot:
                                      {"name": non-empty str, unique after
                                        sanitizing to a USD prim name,
                                       "type": "cube"|"usd" (default "cube"),
                                       "position": [x,y,z] meters (3 numbers),
                                       "size" (m, > 0), "scale" [sx,sy,sz]
                                        (3 numbers),
                                       "color" [r,g,b] each in [0, 1],
                                       "fixed" (bool),
                                       "usd_path" (non-empty str, required
                                        when type is "usd")}
  lighting (object)                 - scene lights to configure at boot:
                                      {"dome": {"intensity": 1000,
                                                 "color": [1, 1, 1]},
                                       "sphere_intensity": 30000}. Both keys
                                      optional; unset means leave the stage's
                                      lights alone.

DoCommand:
  {"command": "status"} | {"command": "play"} | {"command": "pause"} |
  {"command": "reset"} |
  {"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
   "position": [x, y, z]}
"""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from typing_extensions import Self
from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..sim_manager import SimConfig, SimManager, _prim_name
from ..spatial import to_vec3


def _prop_label(prop: object, index: int) -> str:
    if isinstance(prop, Mapping) and prop.get("name"):
        return str(prop["name"])
    return f"props[{index}]"


def _validate_number_triple(prop_label: str, key: str, value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise ValueError(f"prop {prop_label}: {key!r} must be a list of 3 numbers")
    for v in value:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"prop {prop_label}: {key!r} must be a list of 3 numbers")


def _validate_props(props: object) -> None:
    if not isinstance(props, Sequence) or isinstance(props, str):
        raise ValueError("props must be a list")
    seen_prim_names: set[str] = set()
    for index, prop in enumerate(props):
        label = _prop_label(prop, index)
        if not isinstance(prop, Mapping):
            raise ValueError(f"prop {label}: must be an object")
        name = prop.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"prop {label}: 'name' must be a non-empty string")
        prim_name = _prim_name(name)
        if prim_name in seen_prim_names:
            raise ValueError(
                f"prop {label}: 'name' collides with another prop after sanitizing "
                f"to a USD prim name ({prim_name!r})"
            )
        seen_prim_names.add(prim_name)

        kind = prop.get("type", "cube")
        if kind not in ("cube", "usd"):
            raise ValueError(f'prop {label}: \'type\' must be "cube" or "usd" (got {kind!r})')
        if kind == "usd" and not prop.get("usd_path"):
            raise ValueError(f"prop {label}: 'usd_path' is required when 'type' is \"usd\"")

        if "position" in prop:
            _validate_number_triple(label, "position", prop["position"])
        if "scale" in prop:
            _validate_number_triple(label, "scale", prop["scale"])
        if "color" in prop:
            color = prop["color"]
            _validate_number_triple(label, "color", color)
            if isinstance(color, Sequence) and not isinstance(color, str):
                for v in color:
                    is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
                    if is_number and not (0 <= v <= 1):
                        raise ValueError(f"prop {label}: 'color' values must be in [0, 1]")
        if "size" in prop:
            size = prop["size"]
            if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"prop {label}: 'size' must be a positive number")
        if "fixed" in prop and not isinstance(prop["fixed"], bool):
            raise ValueError(f"prop {label}: 'fixed' must be a bool")


_LIGHTING_KEYS = {"dome", "sphere_intensity"}


def _validate_lighting(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("lighting must be an object")
    for key in value:
        if key not in _LIGHTING_KEYS:
            raise ValueError(f"lighting: unknown key {key!r}")

    dome = value.get("dome")
    if dome is not None:
        if not isinstance(dome, Mapping):
            raise ValueError("lighting.dome must be an object")
        if "intensity" in dome:
            intensity = dome["intensity"]
            is_number = isinstance(intensity, (int, float)) and not isinstance(intensity, bool)
            if not is_number or intensity <= 0:
                raise ValueError("lighting.dome.intensity must be a positive number")
        if "color" in dome:
            _validate_number_triple("lighting.dome", "color", dome["color"])
            for v in dome["color"]:
                is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
                if is_number and not (0 <= v <= 1):
                    raise ValueError("lighting.dome.color values must be in [0, 1]")

    sphere_intensity = value.get("sphere_intensity")
    if sphere_intensity is not None:
        is_number = isinstance(sphere_intensity, (int, float)) and not isinstance(
            sphere_intensity, bool
        )
        if not is_number or sphere_intensity < 0:
            raise ValueError("lighting.sphere_intensity must be a non-negative number")


class IsaacWorld(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "world")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        w = cls(config.name)
        w.reconfigure(config, dependencies)
        return w

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        for key in ("physics_dt", "rendering_dt", "boot_timeout_sec"):
            if key in attrs and float(attrs[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if "props" in attrs:
            _validate_props(attrs["props"])
        if "lighting" in attrs:
            _validate_lighting(attrs["lighting"])
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        cfg = SimConfig(
            mock=bool(attrs.get("mock", False)),
            headless=bool(attrs.get("headless", True)),
            livestream=bool(attrs.get("livestream", True)),
            usd_stage=attrs.get("usd_stage") or None,
            physics_dt=float(attrs.get("physics_dt", 1.0 / 60.0)),
            rendering_dt=float(attrs.get("rendering_dt", 1.0 / 60.0)),
            boot_timeout=float(attrs.get("boot_timeout_sec", 300.0)),
            kit_log_level=str(attrs.get("kit_log_level", "warning")),
            livestream_public_ip=str(attrs.get("livestream_public_ip", "")),
            props=[dict(p) for p in attrs.get("props", [])],
            lighting=dict(attrs["lighting"]) if attrs.get("lighting") is not None else None,
        )
        SimManager.get().ensure_booted(cfg)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        sim = SimManager.get()
        cmd = str(command.get("command", ""))
        if cmd == "status":
            return sim.status()
        if cmd == "play":
            sim.play()
            return {"ok": True}
        if cmd == "pause":
            sim.pause()
            return {"ok": True}
        if cmd == "reset":
            sim.reset()
            return {"ok": True}
        if cmd == "add_usd":
            usd_path = str(command.get("usd_path", ""))
            prim_path = str(command.get("prim_path", ""))
            if not usd_path or not prim_path:
                raise ValueError("add_usd requires usd_path and prim_path")
            position = cast("Sequence[float]", command.get("position") or [0.0, 0.0, 0.0])
            sim.add_usd_reference(usd_path, prim_path, to_vec3(position))
            return {"ok": True}
        raise ValueError(f"unknown command {cmd!r}; supported: status, play, pause, reset, add_usd")
