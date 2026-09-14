"""The generic component that owns the simulator."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.generic import Generic
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, RectangularPrism, ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..length_units import MM_PER_M
from ..sim_manager import SimConfig, SimManager, WorldHandle
from .world_commands import COMMAND_HANDLERS, _pose_mm_from_m
from .world_config_validation import (
    FLOOR_LABEL,
    FLOOR_SIDE_MM,
    FLOOR_THICKNESS_MM,
    validate_ground,
    validate_kit_log_level,
    validate_lighting,
    validate_props,
    validate_render,
    validate_world_frame,
)

_SUPPORTED_COMMANDS = (
    "status",
    "play",
    "pause",
    "reset",
    "add_usd",
    "prop_geometries",
    "spawn_prop",
    "set_prop_pose",
    "randomize_props",
    "ignore_props",
    "scatter_cell",
    "clear_cell",
    "joint_state",
    "dof_names",
    "prim_pose",
    "tcp_pose",
    "jaw_deg",
)

# scatter_cell and randomize_props each draw sizes/positions from the
# registered props before handing the result to the sim thread; run two of
# either concurrently off the event loop and their Python-side setup can
# race on that shared state, so do_command serializes just these two.
_SERIALIZED_COMMANDS = frozenset({"scatter_cell", "randomize_props"})


def sim_config_from_attrs(attrs: Mapping[str, Any]) -> SimConfig:
    """The world component's attributes as the SimConfig the sim boots from.
    Shared with tools/warm_shader_cache.py, which boots the same world
    without viam-server, so the two cannot drift."""
    return SimConfig(
        mock=bool(attrs.get("mock", False)),
        headless=bool(attrs.get("headless", True)),
        livestream=bool(attrs.get("livestream", True)),
        usd_stage=attrs.get("usd_stage") or None,
        physics_dt=float(attrs.get("physics_dt", 1.0 / 60.0)),
        rendering_dt=float(attrs.get("rendering_dt", 1.0 / 60.0)),
        boot_timeout=float(attrs.get("boot_timeout_sec", 110.0)),
        wait_for_finalizer=bool(attrs.get("wait_for_finalizer", False)),
        kit_log_level=str(attrs.get("kit_log_level", "warning")),
        livestream_public_ip=str(attrs.get("livestream_public_ip", "")),
        props=[dict(p) for p in attrs.get("props", [])],
        lighting=dict(attrs["lighting"]) if attrs.get("lighting") is not None else None,
        render=dict(attrs["render"]) if attrs.get("render") is not None else None,
        ground=dict(attrs["ground"]) if attrs.get("ground") is not None else None,
    )


class IsaacWorld(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    """viam:isaac-sim-devin:world, the generic component that owns the simulator.

    Configure exactly one of these per machine. All other isaac-sim components
    name it in their "world" attribute; their validate_config returns it as an
    implicit dependency so viam-server boots the world first.

    DoCommand:
      {"command": "status"} | {"command": "play"} | {"command": "pause"} |
      {"command": "reset", "soft"?: bool (default false)} |
      {"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
       "position": [x, y, z] meters, "orientation_rpy_deg"?: [r, p, y] degrees} |
      {"command": "prop_geometries"} ->
        {"geometries": [{"name", "box_dims_mm": [x,y,z],
                          "pose_in_world_mm": {"x","y","z","o_x","o_y","o_z",
                                                "theta"} (theta in degrees),
                          "color": [r,g,b] or None, "fixed": bool}]} |
      {"command": "spawn_prop", "prop": {...same schema as the props config
       attr...}} |
      {"command": "set_prop_pose", "name": "...", "position": [x,y,z] mm,
       "orientation_rpy_deg"?: [r,p,y] degrees} |
      {"command": "randomize_props", "names": [...],
       "region": [[x0,y0,z],[x1,y1,z]] mm, "seed": int,
       "min_separation"?: mm (default 150),
       "size_range_mm"?: [lo, hi] (applies to every named prop) or
         {name: [lo, hi]} (keys must be a subset of "names"); cube props
         only, 0 < lo <= hi. Redraws that prop's size (one uniform(lo, hi)
         scalar applied to all three axes) before placing it, from the same
         seeded stream as the positions, so sizes and positions both
         reproduce for a given seed} ->
        {"positions": {name: [x,y,z] mm}, "sizes_mm": {name: [x,y,z] mm}}
        ("sizes_mm" is always present: the drawn dims for a ranged prop, its
         current dims otherwise) |
      {"command": "ignore_props", "names": [...]} -> {"ignored": [...]}
        (empty list clears; excludes named props from get_geometries, which
         excludes the pick target while grasping) |
      {"command": "scatter_cell", "names_by_color": {color: [names, ...]}
       (each color's pool prop names, in draw order; a color's per-color
       max is the length of its list), "region": [[x0,y0,z],[x1,y1,z]] mm
       (the scatter rectangle), "park_positions_mm": {name: [x,y]} mm
       (every named prop's re-park spot), "seed": int, "size_range_mm"?:
       [lo, hi] (applies to every drawn block; 0 < lo <= hi; omit to keep
       current sizes), "counts"?: {color: int} (overrides the default
       1..len(names_by_color[color]) per-color draw for that color)} ->
       {"seed", "counts": {color: n}, "positions": {name: [x,y,z] mm},
       "sizes_mm": {name: [x,y,z] mm}, "parked": [names]} (log-only
       evidence, never control input) |
      {"command": "clear_cell", "names_by_color": {color: [names, ...]},
       "park_positions_mm": {name: [x,y]} mm} -> {"parked": [names]}
       (log-only evidence, never control input)
    """

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
        """Attributes:
        mock (bool, default false)        - run without isaac sim (for dev/testing)
        headless (bool, default true)     - run kit without a local GUI window
        livestream (bool, default true)   - enable WebRTC livestreaming (view with
                                            the Isaac Sim WebRTC Streaming Client)
        livestream_public_ip (string)     - IP advertised to streaming clients;
                                            auto-detected if unset
        usd_stage (string)                - USD file/omniverse URL to open; if unset
                                            an empty stage with a ground plane is used
        physics_dt / rendering_dt (float) - sim step sizes, default 1/60
        boot_timeout_sec (float)          - how long to wait for kit to boot (default
                                            110). viam-server wraps this whole
                                            reconfigure in its own
                                            VIAM_RESOURCE_CONFIGURATION_TIMEOUT (2
                                            minutes by default); 110 leaves 10s of
                                            headroom under that ceiling so a slow cold
                                            boot raises this module's own timeout
                                            error instead of viam-server's. Raise both
                                            (VIAM_RESOURCE_CONFIGURATION_TIMEOUT is a
                                            viam-server env var, set on the machine,
                                            not a module attribute) if a box's cold
                                            boot routinely runs longer.
        kit_log_level (string)            - kit console verbosity: "verbose", "info",
                                            "warning" or "error" (default "warning")
        wait_for_finalizer (bool, default false) - defer world stepping until the
                                            scene-finalizer component reports every
                                            scene-populating component built; operational
                                            calls answer UNAVAILABLE until it has stepped a
                                            few times past that point
        props (list)                      - objects spawned into the scene at boot:
                                            {"name": non-empty str, unique after
                                              sanitizing to a USD prim name,
                                             "type": "cube"|"usd"|"visual" (default
                                              "cube"; "visual" is a referenced USD
                                              with a pose and a scale and no physics,
                                              absent from every prop verb),
                                             "position": [x,y,z] meters (3 numbers),
                                             "size" (m, > 0), "scale" [sx,sy,sz]
                                              (3 numbers),
                                             "color" [r,g,b] each in [0, 1],
                                             "fixed" (bool),
                                             "usd_path" (non-empty str, required
                                              when type is "usd" or "visual"; a
                                              "visual" path may use module:// or
                                              data://),
                                             "fit" ("visual" only, exclusive with
                                              "scale"): "true" keeps the asset's
                                              authored size, {"collider": "<cube
                                              prop name>"} scales its bounds onto
                                              that cube's size x scale box,
                                             "orientation_rpy_deg" [r,p,y] degrees,
                                             "orientation_wxyz" [w,x,y,z] (not all
                                              zero); at most one of the two,
                                             "box_dims" [x,y,z] meters, each > 0
                                              (used by "usd" props whose geometry
                                              this module can't infer),
                                             "mass" (kg, > 0), "friction" (unitless,
                                              static = dynamic, >= 0), "restitution"
                                              (unitless, in [0, 1]), "contact_offset"
                                              (m, >= 0), "rest_offset" (m, >= 0,
                                              <= contact_offset when both are set)}
        lighting (object)                 - scene lights to configure at boot:
                                            {"dome": {"intensity": 1000,
                                                       "color": [1, 1, 1]},
                                             "sphere_intensity": 30000}. Both keys
                                            optional; unset means leave the stage's
                                            lights alone. dome also takes "texture"
                                            (a path, URL, module:// or data://
                                            HDRI), "texture_format" (UsdLux dome
                                            format, default "latlong") and
                                            "rotation_deg" (yaw about Z).
        ground (object)                   - the floor the module adds when it owns
                                            the stage: {"kind": "grid"|"plane"|
                                            "none", "color": [r,g,b], "size": m,
                                            "friction": f, "restitution": r}.
                                            Unset means "grid" (today's default
                                            environment). Ignored with a warning
                                            when usd_stage is set.
        render (object)                   - render-cost levers applied at boot,
                                            best-effort: {"motion_bvh":
                                            bool, "disable_viewport_updates": bool,
                                            "viewport_grid": bool}.
                                            All keys optional; unset means leave
                                            the renderer's defaults alone.
                                            disable_viewport_updates: true requires
                                            livestream: false (the livestream needs
                                            viewport updates). viewport_grid: false
                                            hides the viewport grid overlay.
        """
        validate_world_frame(config)
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        for key in ("physics_dt", "rendering_dt", "boot_timeout_sec"):
            if key in attrs and float(attrs[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if "props" in attrs:
            validate_props(attrs["props"])
        if "lighting" in attrs:
            validate_lighting(attrs["lighting"])
        if "render" in attrs:
            validate_render(attrs["render"], bool(attrs.get("livestream", True)))
        if "ground" in attrs:
            validate_ground(attrs["ground"])
        if "kit_log_level" in attrs:
            validate_kit_log_level(attrs["kit_log_level"])
        if "wait_for_finalizer" in attrs and not isinstance(attrs["wait_for_finalizer"], bool):
            raise ValueError("wait_for_finalizer must be a boolean")
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        if attrs.get("usd_stage") and "lighting" not in attrs:
            self.logger.warning(
                "usd_stage is set without lighting: stage must provide floor and lights"
            )
        SimManager.get().ensure_booted(sim_config_from_attrs(attrs))
        # the module adds a ground plane only when it owns the stage
        self._serves_floor = not attrs.get("usd_stage")
        if not hasattr(self, "_ignored_props"):
            self._ignored_props: set[str] = set()
        if not hasattr(self, "_serialized_verb_lock"):
            self._serialized_verb_lock = asyncio.Lock()

    async def close(self) -> None:
        """Unregister this component's own post-reset hooks.

        Deliberately does not stop Kit: once SimulationApp is constructed it
        cannot be re-created in the same process (see sim_manager._boot), so
        removing the world component only forgets its own state here. Isaac
        Sim stays booted and stepping until viam-server ends the module
        process.
        """
        SimManager.get().unregister_post_reset(self.name)

    def _handle(self) -> WorldHandle:
        return SimManager.get().world_handle()

    async def get_geometries(self, **kwargs: Any) -> list[Geometry]:
        ignored: set[str] = getattr(self, "_ignored_props", set())
        geometries: list[Geometry] = []
        if getattr(self, "_serves_floor", False) and FLOOR_LABEL not in ignored:
            geometries.append(
                Geometry(
                    center=Pose(
                        x=0.0,
                        y=0.0,
                        z=-FLOOR_THICKNESS_MM / 2.0,
                        o_x=0.0,
                        o_y=0.0,
                        o_z=1.0,
                        theta=0.0,
                    ),
                    box=RectangularPrism(
                        dims_mm=Vector3(x=FLOOR_SIDE_MM, y=FLOOR_SIDE_MM, z=FLOOR_THICKNESS_MM)
                    ),
                    label=FLOOR_LABEL,
                )
            )
        props = await asyncio.to_thread(self._handle().prop_geometries)
        for prop in props:
            if prop.name in ignored:
                continue
            if prop.box_dims_m == (0.0, 0.0, 0.0):
                continue
            dims_mm = tuple(d * MM_PER_M for d in prop.box_dims_m)
            pose_mm = _pose_mm_from_m(prop.position_m, prop.orientation_wxyz)
            geometries.append(
                Geometry(
                    center=Pose(**pose_mm),
                    box=RectangularPrism(dims_mm=Vector3(x=dims_mm[0], y=dims_mm[1], z=dims_mm[2])),
                    label=prop.name,
                )
            )
        return geometries

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        handle = self._handle()
        cmd = str(command.get("command", ""))
        handler = COMMAND_HANDLERS.get(cmd)
        if handler is None:
            raise ViamGRPCError(
                f"unknown command {cmd!r}; supported: {', '.join(_SUPPORTED_COMMANDS)}",
                Status.INVALID_ARGUMENT,
            )
        if cmd in _SERIALIZED_COMMANDS:
            async with self._serialized_verb_lock:
                return await asyncio.to_thread(handler, self, handle, command)
        return await asyncio.to_thread(handler, self, handle, command)
