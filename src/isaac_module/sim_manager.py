"""The singleton that owns Isaac Sim.

Isaac Sim (Omniverse Kit) wants to be created and stepped from a single
thread, so the module runs it on the process main thread (see main.py) and
everything else - the Viam module server, component handlers - submits work
to that thread through a queue. Handles returned by create_arm/create_camera/
create_base wrap that queue so component models can stay simple.

A "mock" backend (world attribute: {"mock": true}) implements the same
handle interfaces with plain python so the module can run and be tested on
machines without Isaac Sim installed.
"""

import concurrent.futures
import hashlib
import math
import os
import queue
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, cast

from viam.logging import getLogger

from . import FAMILY, NAMESPACE
from .asset_catalog import KNOWN_ASSETS, VACUUM_TOOL, VACUUM_TOOL_PRIM
from .asset_catalog import UR_JOINT_NAMES as UR_JOINT_NAMES
from .assets import REMOTE_ASSET_SCHEMES
from .compat import IsaacAPI, caps, import_isaac, isaac_version
from .errors import (
    CameraInitError,
    PrimNotFoundError,
    SimInitializingError,
    SimNotBootedError,
    SimTimeoutError,
)
from .handles.arm import SETTLE_TOL_RAD as SETTLE_TOL_RAD

# models/arm.py re-exports SETTLE_TOL_RAD and SettleOutcome through sim_manager rather than
# reading them off handles.arm directly. Re-imported "as" themselves so ruff (F401) and mypy
# (implicit re-export) both see the deliberate re-export.
from .handles.arm import ArmHandle, IsaacArmHandle, MockArmHandle
from .handles.arm import SettleOutcome as SettleOutcome
from .handles.base import BaseHandle, IsaacBaseHandle, MockBaseHandle
from .handles.camera import DEFAULT_CAMERA_FOV_DEG as DEFAULT_CAMERA_FOV_DEG
from .handles.camera import DEFAULT_CLIP_FAR_M as DEFAULT_CLIP_FAR_M
from .handles.camera import DEFAULT_CLIP_NEAR_M as DEFAULT_CLIP_NEAR_M
from .handles.camera import (
    CameraHandle,
    IsaacCameraHandle,
    MockCameraHandle,
    _camera_prim_path,
    _configure_camera_optics,
    _place_camera,
)
from .handles.gripper import (
    DEFAULT_HOLDING_TOLERANCE_DEG,
    IsaacGripperHandle,
    JawGripperHandle,
    MockGripperHandle,
)

# re-exported for the world's diagnostic verbs, which narrow on the
# mechanism-neutral protocol; nothing in this module names it
from .handles.gripper import (
    GripperHandle as GripperHandle,
)
from .handles.vacuum import (
    DEFAULT_GRAB_DELAY_MS,
    DEFAULT_MAX_PAYLOAD_GAP_M,
    IsaacVacuumHandle,
    MockVacuumHandle,
    VacuumGripperHandle,
)
from .handles.world import IsaacWorldHandle, MockWorldHandle, WorldHandle
from .materials import MATERIAL_SPEC_KEY, build_material, material_spec, prop_display_color
from .physics import ARM_SOLVER_POSITION_ITERATIONS, apply_prop_physics
from .prim_paths import prim_name
from .prop_scatter import (
    DEFAULT_MIN_SEPARATION_M as DEFAULT_MIN_SEPARATION_M,
)
from .prop_scatter import (
    POOL_SCATTER_MIN_SEPARATION_M as POOL_SCATTER_MIN_SEPARATION_M,
)
from .prop_scatter import (
    PROP_EDGE_CLEARANCE_M as PROP_EDGE_CLEARANCE_M,
)
from .prop_scatter import (
    PROP_REST_EPSILON_M as PROP_REST_EPSILON_M,
)
from .prop_scatter import (
    ClearCellResult as ClearCellResult,
)
from .prop_scatter import (
    PropGeometry as PropGeometry,
)
from .prop_scatter import (
    RandomizeResult as RandomizeResult,
)
from .prop_scatter import (
    ScatterCellResult as ScatterCellResult,
)
from .prop_scatter import (
    _draw_pool_counts as _draw_pool_counts,
)
from .prop_scatter import (
    _draw_sizes_and_positions as _draw_sizes_and_positions,
)
from .prop_scatter import (
    _park_pose_m as _park_pose_m,
)
from .prop_scatter import (
    _require_cube_prop as _require_cube_prop,
)
from .prop_scatter import (
    _require_pool_blocks as _require_pool_blocks,
)
from .prop_scatter import (
    _scattered_block_names as _scattered_block_names,
)
from .prop_scatter import (
    _validate_size_range_names as _validate_size_range_names,
)
from .prop_scatter import (
    prop_box_dims as prop_box_dims,
)
from .prop_scatter import (
    prop_spawn_orientation,
)
from .prop_scatter import (
    sample_prop_positions as sample_prop_positions,
)
from .spatial import (
    Quat,
    Vec3,
    _as_quat,
    anchor_fixed_joint_frame,
    quat_from_euler_deg,
    spawn_orientation,
    to_vec3,
)
from .spatial import (
    compose_pose as compose_pose,
)
from .spatial import (
    pose_in_frame as pose_in_frame,
)
from .spatial import (
    viam_base_frame as viam_base_frame,
)
from .usd_assets import (
    ASSET_CACHE_DIRNAME as ASSET_CACHE_DIRNAME,
)
from .usd_assets import (
    ASSET_LAYER_CACHE_VERSION,
    ASSET_PREPARE_MAX_DEPTH,
    UNRESOLVABLE_ASSET_HOST,
    _anchor_asset_path,
    _describe_composition,
    _is_layer_path,
    _local_ip,
    _pad_collision_status,
    _prepared_asset_path,
    _prim_range,
    _reference_asset_paths,
    _remove_articulation_roots,
    _rewritten_references,
    _version_string,
    asset_cache_dir,
)
from .usd_assets import (
    ASSETS_PATH_MARKER as ASSETS_PATH_MARKER,
)
from .usd_assets import (
    PAD_PRIM_NAME_FRAGMENTS as PAD_PRIM_NAME_FRAGMENTS,
)
from .usd_assets import (
    _bucket_candidate as _bucket_candidate,
)
from .visual_props import VISUAL_PROP_KIND, status_row
from .workcell_scenery import scenery_props

LOGGER = getLogger("viam-isaac-sim")


@dataclass
class SimConfig:
    """Sim boot configuration, built by the world component's reconfigure
    and handed to SimManager.ensure_booted."""

    mock: bool = False
    headless: bool = True
    livestream: bool = True
    usd_stage: str | None = None
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    # viam-server's own VIAM_RESOURCE_CONFIGURATION_TIMEOUT (2 minutes by
    # default) wraps this component's whole reconfigure; 110 leaves 10s of
    # headroom under that ceiling so a slow cold boot raises this module's
    # own SimTimeoutError instead of viam-server's own timeout error.
    boot_timeout: float = 110.0
    # IP the livestream advertises to clients. Auto-detected if empty
    livestream_public_ip: str = ""
    # props to spawn into the scene at boot. Each entry:
    #   {"type": "cube"|"usd", "name": ..., "position": [x,y,z] (m),
    #    "size": edge_m, "scale": [sx,sy,sz], "color": [r,g,b] 0-1,
    #    "material": "<set>" | {albedo, normal, roughness, metallic, tint,
    #    texture_scale}, "fixed": bool, "usd_path": ...}
    props: list[dict[str, Any]] = field(default_factory=list)
    # kit console verbosity (verbose/info/warning/error). Kit prints thousands
    # of lines at info, and viam-server records the module's stderr as
    # error-level logs, so default to warning.
    kit_log_level: str = "warning"
    # scene lighting. None = leave the stage's lights
    # alone. Shape: {"dome": {"intensity": 1000, "color": [1, 1, 1]},
    # "sphere_intensity": 30000}. Isaac creates a DomeLight prim and rescales
    # /World/SphereLight, the mock records the config for tests.
    lighting: dict[str, Any] | None = None
    # render-cost levers. None = leave the renderer's
    # defaults alone. Shape: {"motion_bvh": bool, "disable_viewport_updates":
    # bool}. motion_bvh folds into extra_args on the SimulationApp launcher
    # config (Kit settings can't be safely re-applied after launch) and
    # disable_viewport_updates is a launcher-config key, both at boot. The
    # mock records the config for tests.
    render: dict[str, Any] | None = None
    # the floor the module adds when it owns the stage. None = today's grid
    # environment. Shape: {"kind": "grid"|"plane"|"none", "color": [r,g,b],
    # "size": m, "friction": f, "restitution": r}, the plane-only keys with
    # defaults in models/world_config_validation.py (GROUND_DEFAULTS). Ignored
    # with a warning when usd_stage is set. The mock records it for tests.
    ground: dict[str, Any] | None = None
    # defer world steps until a scene-finalizer component runs. The finalizer's
    # depends_on names every scene-populating component, so the renderer's
    # first (slow, shader-compiling) steps happen after every resource is built
    # instead of while viam-server is still constructing them. Operational
    # calls answer UNAVAILABLE (SimInitializingError) until
    # POST_FINALIZER_WARMUP_STEPS steps have completed after finalization.
    wait_for_finalizer: bool = False


# Completed world steps after finalize_scene() before the world reports ready.
# Each cold step completes only after its shader compile, so three completed
# steps means the compiles the populated scene provoked are behind us.
POST_FINALIZER_WARMUP_STEPS = 3

# Materialising a full workcell is one sim-thread task covering every
# component's prims. The stage gate is not stepping while it runs, so a long
# task costs nothing, but it must not inherit run()'s ordinary 30 s budget.
MATERIALISE_TIMEOUT_S = 300.0
# A world.step that takes longer than this is logged with what it means (a
# cold shader compile, expected on the first steps after the scene changes).
# A warm step is ~1/60 s, so anything past a few seconds is never a normal step.
SLOW_STEP_WARN_S = 5.0
# How long the sim thread waits on the scene gate between task drains while
# stepping is deferred, so a queued create_* call is picked up promptly.
SCENE_GATE_POLL_S = 0.01


def _boot_extra_args(cfg: SimConfig) -> list[str]:
    """Kit CLI args for SimulationApp's ``extra_args`` launcher key, built
    once before Kit starts rather than injected via ``sys.argv`` (which
    mutates process-global state viam-server's own arg handling shares) or
    written through ``carb.settings`` after launch (which the 5.0 handbook
    documents as leaving the cost in place for the motion-BVH settings).

    kit_log_level -> "/log/outputStreamLevel". render["motion_bvh"] ->
    the three raytracingMotion settings the handbook requires together to
    actually disable it; True leaves Kit's own (enabled) default alone
    apart from the explicit "enabled" flag. The DLSS exec mode is always
    pinned to Performance: Auto tends to pick Quality below 720p, and the
    shipped cell's wrist/side cameras render at 848x480."""
    args = [f"--/log/outputStreamLevel={cfg.kit_log_level.capitalize()}"]

    motion_bvh = (cfg.render or {}).get("motion_bvh")
    if motion_bvh is not None:
        args.append(f"--/renderer/raytracingMotion/enabled={'true' if motion_bvh else 'false'}")
        if not motion_bvh:
            args += [
                "--/renderer/raytracingMotion/enableHydraEngineMasking=false",
                "--/renderer/raytracingMotion/enabledForHydraEngines=",
            ]

    args.append("--/rtx/post/dlss/execMode=0")
    return args


# GCP 1:1-NATs a VM's external IP, so it never appears on a local interface,
# the metadata server is the only way to read it from inside the VM.
_GCP_METADATA_EXTERNAL_IP_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "network-interfaces/0/access-configs/0/external-ip"
)


def _public_ip() -> str:
    """The address the livestream should advertise to clients: the GCP
    metadata server's external IP, falling back to ``_local_ip()`` off GCP
    (or if the metadata query fails). Logs which source won."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        _GCP_METADATA_EXTERNAL_IP_URL, headers={"Metadata-Flavor": "Google"}
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            ip = response.read().decode("ascii").strip()
        if ip:
            LOGGER.info("livestream public IP %s (source: GCP metadata server)", ip)
            return ip
    except (urllib.error.URLError, OSError, ValueError):
        pass
    ip = _local_ip()
    LOGGER.info("livestream public IP %s (source: local network interface)", ip or "<none>")
    return ip


# lighting defaults: the DomeLight the module adds, and the prim
# paths in default_environment.usd it adjusts.
DEFAULT_DOME_INTENSITY = 1000.0
DEFAULT_DOME_COLOR = (1.0, 1.0, 1.0)
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
SPHERE_LIGHT_PRIM_PATH = "/World/SphereLight"
# the plain floor ground.kind "plane" authors in place of the grid environment
GROUND_PLANE_PRIM_PATH = "/World/groundPlane"
GROUND_PLANE_NAME = "ground_plane"
# RTX "Matte Object" post-process: a prim carrying the primvar is invisible to
# primary rays but still receives shadows, so the dome texture shows through a
# ground.matte plane. Setting paths and the primvar were confirmed on Isaac Sim
# 5.0.0 on 2026-09-14: the plane vanished and kept its shadows.
MATTE_OBJECT_PRIMVAR = "primvars:isMatteObject"
MATTE_OBJECT_SETTING = "/rtx/post/matteObject/enabled"
SHADOW_CATCHER_SETTING = "/rtx/post/matteObject/enableShadowCatcher"


def dome_light_settings(dome: Mapping[str, Any], resolve: Callable[[str], str]) -> dict[str, Any]:
    """Pure. Turns a validated ``lighting.dome`` mapping into the values
    ``_apply_lighting`` authors on the DomeLight prim: the resolved texture
    path (or None), the texture format (default DEFAULT_DOME_TEXTURE_FORMAT),
    and the ``(0, 0, yaw)`` rotation - None unless rotation_deg is present, so
    a dome without an explicit yaw keeps today's identity xform. Kit already
    orients a DomeLight's pole to the Z-up stage (GPU-observed 2026-09-14, a
    270 degree X tilt stood the HDRI floor up in front of the camera), so no
    up-axis correction is authored here."""
    from .models.world_config_validation import DEFAULT_DOME_TEXTURE_FORMAT

    texture_value = dome.get("texture")
    rotation_deg = dome.get("rotation_deg")
    rotate_xyz: tuple[float, float, float] | None = None
    if rotation_deg is not None:
        rotate_xyz = (0.0, 0.0, float(rotation_deg))
    return {
        "texture": resolve(texture_value) if texture_value else None,
        "texture_format": dome.get("texture_format", DEFAULT_DOME_TEXTURE_FORMAT),
        "rotate_xyz": rotate_xyz,
    }


def missing_texture_warning(
    path: str, exists: Callable[[str], bool] = os.path.exists
) -> str | None:
    """Pure given ``exists``. A resolved local texture path that is not on disk
    gets a warning string; remote URLs are left to Isaac's resolver and return
    None. USD authors a missing asset path without complaint and the renderer
    falls back to an untextured dome, so this is the only place the mistake
    is named."""
    if path.startswith(REMOTE_ASSET_SCHEMES) or exists(path):
        return None
    return f"dome texture not found, the dome renders untextured: {path}"


def ground_plan(
    ground: Mapping[str, Any] | None, usd_stage: str | None
) -> tuple[str, dict[str, Any]]:
    """Pure. Decides what ``_boot`` should author for the floor: ``"skip"``
    (with a ``reason`` key) when a ``ground`` config is set alongside a user's
    own ``usd_stage`` (DEC-W5's ignore-with-a-warning case), ``"grid"`` for no
    config or ``kind: "grid"``, ``"none"`` for ``kind: "none"``, or
    ``"plane"`` with the ``scene.add_ground_plane`` kwargs (size, color,
    static/dynamic friction from ``friction``, restitution), each field
    filled from GROUND_DEFAULTS when the config omits it."""
    from .models.world_config_validation import GROUND_DEFAULTS

    if usd_stage is not None and ground is not None:
        return "skip", {"reason": f"usd_stage {usd_stage!r} is set"}
    if ground is None:
        return "grid", {}

    kind = ground.get("kind", GROUND_DEFAULTS["kind"])
    if kind == "grid":
        return "grid", {}
    if kind == "none":
        return "none", {}

    friction = float(ground.get("friction", GROUND_DEFAULTS["friction"]))
    color = ground.get("color", GROUND_DEFAULTS["color"])
    kwargs = {
        "size": float(ground.get("size", GROUND_DEFAULTS["size"])),
        "color": [float(v) for v in color],
        "static_friction": friction,
        "dynamic_friction": friction,
        "restitution": float(ground.get("restitution", GROUND_DEFAULTS["restitution"])),
    }
    return "plane", kwargs


def ground_is_matte(ground: Mapping[str, Any] | None) -> bool:
    """Pure. Whether ``ground.matte`` is set, falling back to
    GROUND_DEFAULTS when the config omits it or ``ground`` is None."""
    from .models.world_config_validation import GROUND_DEFAULTS

    if ground is None:
        return bool(GROUND_DEFAULTS["matte"])
    return bool(ground.get("matte", GROUND_DEFAULTS["matte"]))


# create_camera attrs contract defaults (CameraHandle class docstring).
# A freshly booted renderer creates a render product's SDG pipeline nodes
# only after render ticks, so Camera.initialize()'s immediate node lookup can
# die with KeyError('/Render/PostProcess/SDGPipeline/..._LdrColorSDhostPtr')
# - observed on the first cold boot of a fresh install (empty shader cache,
# every component building at once). 5 attempts x 30 ticks is ~2 s of
# stepping at 60 Hz, well inside create_camera's 120 s budget even with
# shader compilation on top.
CAMERA_INIT_ATTEMPTS = 5
CAMERA_INIT_RENDER_TICKS = 30

DEFAULT_CAMERA_WIDTH = 848
DEFAULT_CAMERA_HEIGHT = 480


def _forget_scene_object(scene: Any, name: str) -> None:
    """Drop ``name`` from the scene registry if it is registered, leaving the
    prim in place. A re-attached arm is bound with ``initialize()`` and never
    re-added to the registry, and its released handle already dropped the
    entry, so an unconditional ``remove_object`` raises ``Cannot remove
    object ... since it doesn't exist`` and a gripper rebuilt after an arm
    attribute edit never comes up."""
    if scene.get_object(name) is not None:
        scene.remove_object(name, registry_only=True)


def _home_joints_rad(attrs: dict[str, Any]) -> list[float] | None:
    """``home_joints_deg`` in radians, or None when the arm keeps its asset's
    own default pose.

    Raises ValueError for anything that is not a list of numbers, since a
    malformed home pose would otherwise spawn the arm somewhere nobody asked
    for and the failure would look like a physics problem."""
    value = attrs.get("home_joints_deg")
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f'"home_joints_deg" must be a list of joint angles in degrees, got {value!r}'
        )
    out: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(f'"home_joints_deg" entries must be numbers in degrees, got {entry!r}')
        out.append(math.radians(float(entry)))
    if not out:
        raise ValueError('"home_joints_deg" must name at least one joint')
    return out


@dataclass(frozen=True)
class ComponentScenery:
    """One configured workcell component, as ``materialise_components`` needs
    it: its model, its two shape replies, its attributes, and the frame it
    sits in.

    ``frame_position_m`` and ``frame_orientation_wxyz`` are the component's
    own frame in the cell (metres, world-frame quaternion), since
    ``workcell_scenery.scenery_props`` returns every primitive's pose in the
    component's own frame and only the caller knows where that frame sits.
    """

    model: str
    visuals: Mapping[str, Any]
    geometries: Sequence[Mapping[str, Any]]
    attrs: Mapping[str, Any]
    frame_position_m: Vec3 = (0.0, 0.0, 0.0)
    frame_orientation_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)


class SimManager:
    """Owns the sim thread. Get the process-wide instance via SimManager.get()."""

    _instance: "SimManager | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SimManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SimManager()
            return cls._instance

    def __init__(self) -> None:
        self._tasks: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._boot_requested = threading.Event()
        self._booted = threading.Event()
        # the scene gate. _scene_finalized: every scene-populating component
        # has been built (set by the scene-finalizer model's finalize_scene()).
        # _ready: POST_FINALIZER_WARMUP_STEPS steps have completed since. Both
        # start set and are cleared by ensure_booted when cfg.wait_for_finalizer
        # is true, so a world without the field never gates anything.
        self._scene_finalized = threading.Event()
        self._scene_finalized.set()
        self._ready = threading.Event()
        self._ready.set()
        # completed world steps since finalize_scene(), counted toward
        # POST_FINALIZER_WARMUP_STEPS. Only meaningful while _ready is clear.
        self._warmup_steps_completed = 0
        # monotonic start time of the world.step currently in flight on the
        # sim thread, or None between steps. Lets run()'s timeout message
        # say how long a slow (shader-compiling) step has been running.
        self._step_started_at: float | None = None
        self._boot_error: BaseException | None = None
        self._stop = threading.Event()
        # set once main_loop() returns, so a call arriving after shutdown
        # fails immediately instead of waiting run()'s full timeout.
        self._stopped = False
        self._sim_thread_id: int | None = None

        self.cfg: SimConfig | None = None
        self.mock = False
        # Isaac objects are created at boot. Typed Any because the isaacsim
        # modules are not importable (or type-checkable) outside Isaac Sim.
        self._sim_app: Any = None
        self.world: Any = None
        self._isaac: IsaacAPI = cast(IsaacAPI, None)  # populated by _boot()
        self._step_callbacks: dict[str, Callable[[float], None]] = {}
        # scene lighting config from the world component.
        # stored so status() and tests can read it even in mock mode.
        self.lighting: dict[str, Any] | None = None
        # render-cost levers config from the world component.
        # stored so status() and tests can read it even in mock mode.
        self.render: dict[str, Any] | None = None
        # floor config from the world component.
        # stored so status() and tests can read it even in mock mode.
        self.ground: dict[str, Any] | None = None
        # hooks fired (in registration order) after every world reset,
        # so component handles can re-anchor state that resets undo.
        # Each is (owner component name or None, hook). The owner lets
        # release_handle drop a closed component's hooks.
        self._post_reset_hooks: list[tuple[str | None, Callable[[], None]]] = []
        self._post_reset_lock = threading.Lock()
        # the seam the world component's verbs drive the scene through,
        # created at boot (Isaac or Mock flavour)
        self._world_handle: WorldHandle | None = None
        # spawn spec per registered prop (sanitized name -> attrs), the
        # Isaac-side scene registry
        self._prop_specs: dict[str, dict[str, Any]] = {}
        # visual-only props (type "visual"), sanitized name ->
        # visual_props.visual_prop_record. Never in _prop_specs or the mock
        # registry, so no scene verb sees one; status() lists them under
        # "visual_props". Filled by the Isaac path (_spawn_visual_prop) and
        # by MockWorldHandle in mock mode.
        self._visual_props: dict[str, dict[str, Any]] = {}
        # component name -> (spawn attrs, handle). release_handle (close())
        # pops a name so the next create_* re-runs the factory, re-attaching
        # to the prim that release_handle deliberately left in the stage.
        self._handles: dict[str, tuple[dict[str, Any], Any]] = {}
        # component names whose prim has been spawned into the scene at
        # least once this Kit process, never cleared by release_handle (the
        # prim outlives the handle). _create_arm_isaac/_create_base_isaac
        # check this to skip a duplicate world.scene.add and a full
        # _reset_world() on a re-attach, only re-wrapping the surviving prim.
        self._prim_spawned: set[str] = set()

    def ensure_booted(self, cfg: SimConfig) -> None:
        """Called by the world component's reconfigure. Boots the sim on the
        sim thread the first time. Subsequent calls with a different config
        log that a module restart is required (Kit can't be re-created)."""
        if self._booted.is_set():
            if self.cfg != cfg:
                LOGGER.warning(
                    "Isaac Sim is already running, changes to world config "
                    "(stage/headless/etc) require restarting the module"
                )
            return
        if self._boot_error is not None:
            raise RuntimeError(f"Isaac Sim failed to boot previously: {self._boot_error}")

        self.cfg = cfg
        if cfg.wait_for_finalizer:
            self._scene_finalized.clear()
            self._ready.clear()
        else:
            self._scene_finalized.set()
            self._ready.set()
        self._boot_requested.set()
        if not self._booted.wait(timeout=cfg.boot_timeout):
            raise SimTimeoutError(f"Isaac Sim did not boot within {cfg.boot_timeout}s")
        if self._boot_error is not None:
            raise RuntimeError(f"Isaac Sim failed to boot: {self._boot_error}")

    def finalize_scene(self) -> None:
        """Release world stepping. Called by the scene-finalizer model once
        every component it depends on has been built."""
        self._scene_finalized.set()

    def require_ready(self) -> None:
        """Raise SimInitializingError (UNAVAILABLE) until the post-finalizer
        warm-up steps have completed. Handles whose operational verbs bypass
        run() (the base's velocity command) call this themselves."""
        if not self._ready.is_set():
            raise SimInitializingError()

    def request_stop(self) -> None:
        self._stop.set()

    def register_post_reset(self, hook: Callable[[], None], owner: str | None = None) -> None:
        """Register a hook that fires (in registration order) after every
        world reset - boot, a component spawn, or an explicit reset command -
        on whichever thread performs the reset (the sim thread in practice).
        ``owner`` is the component name the hook belongs to, so closing that
        component can drop it. None = lives for the process."""
        with self._post_reset_lock:
            self._post_reset_hooks.append((owner, hook))

    def unregister_post_reset(self, owner: str) -> None:
        """Drop every hook registered under ``owner``."""
        with self._post_reset_lock:
            self._post_reset_hooks = [(o, hook) for o, hook in self._post_reset_hooks if o != owner]

    def _reset_world(self) -> None:
        """The single chokepoint for resetting the Isaac world: resets it
        (skipped in mock mode) then runs every registered post-reset hook,
        isolating each hook's failures so one can't block the rest."""
        if not self.mock:
            self.world.reset()
        with self._post_reset_lock:
            hooks = [hook for _owner, hook in self._post_reset_hooks]
        for hook in hooks:
            try:
                hook()
            except Exception:
                LOGGER.exception("post-reset hook failed")

    def main_loop(self) -> None:
        """Run forever on the owning (main) thread: wait for a boot request,
        boot, then step the sim while draining queued tasks. Sets
        ``_stopped`` on every exit path, so callers still holding a
        reference after this returns fail fast instead of waiting out
        run()'s full timeout."""
        try:
            self._sim_thread_id = threading.get_ident()

            while not self._stop.is_set() and not self._boot_requested.wait(timeout=0.1):
                self._drain_tasks()

            if self._stop.is_set():
                return

            try:
                self._boot()
            except BaseException as e:  # SimulationApp failures can be SystemExit
                LOGGER.exception("failed to boot Isaac Sim")
                self._boot_error = e
                self._booted.set()
                return
            self._booted.set()

            self._step_loop()

            if self._sim_app is not None:
                try:
                    self._sim_app.close()
                except Exception:
                    LOGGER.exception("error closing Isaac Sim")
        finally:
            self._stopped = True

    def _step_loop(self) -> None:
        """Step the sim while draining queued tasks, until request_stop() or
        - non-mock only - the Isaac app itself stops running. A renderer
        crash or a stage close (File->Exit on a remote desktop) leaves
        self._sim_app.is_running() False; without checking it here the loop
        used to spin on a dead app instead of reporting a boot/run failure."""
        last = time.monotonic()
        while not self._stop.is_set() and (self.mock or self._sim_app.is_running()):
            if not self._scene_finalized.is_set():
                # scene-populating components are still being built. Drain
                # tasks (a create_* call is allowed through the gate) but
                # skip stepping so the renderer's first, slow shader-compiling
                # steps happen once, after the scene is complete.
                self._drain_tasks()
                self._scene_finalized.wait(timeout=SCENE_GATE_POLL_S)
                last = time.monotonic()
                continue

            self._drain_tasks()
            now = time.monotonic()
            dt = now - last
            last = now
            if self.mock:
                for callback in list(self._step_callbacks.values()):
                    callback(dt)
                time.sleep(0.01)
            else:
                self._step_world_once()

            if not self._ready.is_set():
                self._warmup_steps_completed += 1
                if self._warmup_steps_completed >= POST_FINALIZER_WARMUP_STEPS:
                    self._ready.set()

        if not self.mock and self._sim_app is not None and not self._stop.is_set():
            self._boot_error = RuntimeError(
                "Isaac Sim's app stopped running outside a requested shutdown"
            )
            LOGGER.error("Isaac Sim's app stopped running; the world's status verb reports it")

    def _step_world_once(self) -> None:
        """One rendered Isaac step, timed. Records the in-flight start so
        run()'s timeout message can say how long the step has been running,
        and warns when a step passes SLOW_STEP_WARN_S, which on a cold shader
        cache is the first steps after the scene changes."""
        step_started = time.monotonic()
        self._step_started_at = step_started
        self.world.step(render=True)
        self._step_started_at = None
        elapsed = time.monotonic() - step_started
        if elapsed > SLOW_STEP_WARN_S:
            LOGGER.warning(
                "world.step took %.1fs: the first steps after the scene "
                "changes compile shaders; with the gate on, calls answer "
                "UNAVAILABLE until the world is ready",
                elapsed,
            )

    def _drain_tasks(self) -> None:
        while True:
            try:
                task, fut = self._tasks.get_nowait()
            except queue.Empty:
                return
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(task())
                except BaseException as e:  # noqa: BLE001 - task() may raise anything, it must reach the caller via the future, not be swallowed here
                    fut.set_exception(e)

    def run(
        self,
        task: Callable[[], Any],
        timeout: float = 30.0,
        *,
        allow_during_initialization: bool = False,
    ) -> Any:
        """Run task on the sim thread and return its result. Raises
        immediately, without waiting ``timeout``, once main_loop() has
        exited, and with SimInitializingError (UNAVAILABLE) while the
        scene gate is closed unless ``allow_during_initialization`` is set,
        which the component factories and the stop verbs pass so a cold
        start can still build resources and halt motion."""
        if self._stopped:
            raise SimNotBootedError("Isaac Sim has stopped - the module is shutting down")
        if threading.get_ident() == self._sim_thread_id:
            # already on the sim thread, so nothing here can queue behind a
            # slow step. The gate is for callers that would. Post-reset hooks
            # and the create factories' own nested calls land here while the
            # scene is still initializing.
            return task()
        if not allow_during_initialization:
            self.require_ready()
        fut: Future = Future()
        self._tasks.put((task, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            message = f"sim-thread call timed out after {timeout}s"
            step_started = self._step_started_at
            if step_started is not None:
                message += (
                    "; the sim thread has been inside world.step for "
                    f"{time.monotonic() - step_started:.1f}s"
                )
            raise SimTimeoutError(message) from exc

    def _boot(self) -> None:
        cfg = self.cfg
        assert cfg is not None
        self.lighting = cfg.lighting
        self.render = cfg.render
        self.ground = cfg.ground
        if cfg.mock:
            LOGGER.info("booting in MOCK mode - no Isaac Sim")
            self.mock = True
            self._world_handle = MockWorldHandle(self, cfg.props)
            self._reset_world()
            return

        LOGGER.info("booting Isaac Sim (headless=%s)...", cfg.headless)
        from isaacsim import SimulationApp

        launcher_config: dict[str, Any] = {
            "headless": cfg.headless,
            "extra_args": _boot_extra_args(cfg),
            # 32 Carbonite/TBB workers (5.0's own default) oversubscribes an
            # 8 vCPU box that is also running viam-server and PhysX.
            "limit_cpu_threads": max(2, (os.cpu_count() or 4) - 2),
        }
        disable_viewport_updates = (cfg.render or {}).get("disable_viewport_updates")
        if disable_viewport_updates is not None:
            launcher_config["disable_viewport_updates"] = bool(disable_viewport_updates)
        livestreaming = cfg.livestream and cfg.headless
        if livestreaming:
            # headless=True makes SimulationApp append --/app/window/hideUi=1;
            # override it here so the WebRTC client sees the full UI, matching
            # the 5.0 livestream sample ("hide_ui": False,  # Show the GUI).
            launcher_config["hide_ui"] = False
        # SimulationApp.__init__ unconditionally replaces SIGINT with a
        # handler that calls sys.exit(0) directly, bypassing
        # main.py's own shutdown path. Put the caller's handler straight
        # back so SIGINT after boot goes through the same path SIGTERM does.
        prior_sigint_handler = signal.getsignal(signal.SIGINT)
        self._sim_app = SimulationApp(launcher_config)
        signal.signal(signal.SIGINT, prior_sigint_handler)

        if livestreaming:
            try:
                from isaacsim.core.utils.extensions import enable_extension

                ip = cfg.livestream_public_ip or _public_ip()
                self._sim_app.set_setting("/app/livestream/enabled", True)
                self._sim_app.set_setting("/app/livestream/port", 49100)
                if ip:
                    self._sim_app.set_setting("/app/livestream/publicEndpointAddress", ip)
                self._sim_app.set_setting("/app/window/drawMouse", True)
                if enable_extension("omni.kit.livestream.webrtc"):
                    LOGGER.info(
                        "livestream enabled - connect the 'Isaac Sim WebRTC Streaming "
                        "Client' app to %s (TCP 49100 + UDP 47998 must be reachable)",
                        ip or "<this machine's IP>",
                    )
                else:
                    LOGGER.warning(
                        "enable_extension('omni.kit.livestream.webrtc') returned False, "
                        "the livestream client will not be able to connect"
                    )
            except Exception:
                LOGGER.exception("could not enable livestream, continuing without it")

        self._isaac = import_isaac()

        if cfg.usd_stage:
            LOGGER.info("opening stage %s", cfg.usd_stage)
            self._open_stage_and_wait(cfg.usd_stage)

        self.world = self._isaac.World(
            physics_dt=cfg.physics_dt,
            rendering_dt=cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        self._add_ground(cfg)
        if cfg.render is not None and "viewport_grid" in cfg.render:
            self._apply_viewport_grid(bool(cfg.render["viewport_grid"]))
        # non-visual props spawn first so a visual's fit.collider always
        # finds its cube already in _prop_specs
        ordered_props = [p for p in cfg.props if str(p.get("type", "cube")) != VISUAL_PROP_KIND]
        ordered_props += [p for p in cfg.props if str(p.get("type", "cube")) == VISUAL_PROP_KIND]
        for prop in ordered_props:
            try:
                self._spawn_prop(prop)
            except Exception:
                LOGGER.exception("failed to spawn prop %s", prop.get("name"))
        if cfg.lighting is not None:
            self._apply_lighting(cfg.lighting)
        self._world_handle = IsaacWorldHandle(self)
        self._reset_world()
        LOGGER.info("Isaac Sim world ready")

    def _open_stage_and_wait(self, usd_stage: str) -> None:
        """Open usd_stage and drain the async load the 5.0 sample documents
        (standalone_examples/api/isaacsim.simulation_app/load_stage.py):
        is_stage_loading() stays true while USD composition keeps running in
        the background after open_stage() returns, and reset_render_settings()
        re-applies render config the new stage otherwise drops. open_stage()
        returns False on a bad path instead of raising; that used to leave an
        empty stage (no ground plane, no error) instead of failing boot."""
        opened = self._isaac.open_stage(usd_stage)
        if not opened:
            raise RuntimeError(f"failed to open USD stage: {usd_stage}")
        from isaacsim.core.utils.stage import is_stage_loading

        while is_stage_loading():
            self._sim_app.update()
        self._sim_app.reset_render_settings()

    def _apply_lighting(self, lighting: dict[str, Any]) -> None:
        """Configure scene lights. Best-effort: never
        raises, so bad/unavailable lighting config can't block boot."""
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdLux

            from .assets import resolve_asset

            stage = omni.usd.get_context().get_stage()

            dome = lighting.get("dome")
            if dome is not None:
                dome_light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
                dome_light.CreateIntensityAttr(float(dome.get("intensity", DEFAULT_DOME_INTENSITY)))
                color = dome.get("color", DEFAULT_DOME_COLOR)
                dome_light.CreateColorAttr(Gf.Vec3f(*[float(v) for v in color]))

                settings = dome_light_settings(dome, resolve_asset)
                if settings["texture"] is not None:
                    warning = missing_texture_warning(settings["texture"])
                    if warning:
                        LOGGER.warning(warning)
                    dome_light.CreateTextureFileAttr(Sdf.AssetPath(settings["texture"]))
                    dome_light.CreateTextureFormatAttr(settings["texture_format"])
                if settings["rotate_xyz"] is not None:
                    xformable = UsdGeom.Xformable(dome_light)
                    # clear any xform ops a prior reconfigure authored so a
                    # re-applied rotation replaces rather than stacks on top.
                    xformable.ClearXformOpOrder()
                    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*settings["rotate_xyz"]))

            sphere_intensity = lighting.get("sphere_intensity")
            if sphere_intensity is not None:
                sphere_prim = stage.GetPrimAtPath(SPHERE_LIGHT_PRIM_PATH)
                if sphere_prim.IsValid():
                    UsdLux.SphereLight(sphere_prim).GetIntensityAttr().Set(float(sphere_intensity))
        except Exception:
            LOGGER.exception("failed to apply scene lighting")

    def _add_ground(self, cfg: SimConfig) -> None:
        """Author the floor ``ground_plan`` decided on. Runs on the sim thread
        before props spawn, so a block always has something to land on."""
        ground_kind, ground_kwargs = ground_plan(cfg.ground, cfg.usd_stage)
        if ground_kind == "skip":
            LOGGER.warning("ground config ignored: %s", ground_kwargs["reason"])
            return
        if ground_kind == "none":
            return
        if ground_kind == "plane":
            import numpy as np

            # PreviewSurface calls color.tolist(), so the colour must be an array
            ground_kwargs["color"] = np.array(ground_kwargs["color"], dtype=float)
            try:
                self.world.scene.add_ground_plane(
                    name=GROUND_PLANE_NAME,
                    prim_path=GROUND_PLANE_PRIM_PATH,
                    z_position=0.0,
                    **ground_kwargs,
                )
                self._apply_ground_material(cfg.ground or {})
                if ground_is_matte(cfg.ground):
                    self._make_ground_matte()
            except Exception:
                LOGGER.exception("failed to add ground plane, falling back to the default grid")
                self.world.scene.add_default_ground_plane()
            return
        # "grid": today's behaviour is a floor only when the module owns the
        # stage, an unowned usd_stage keeps its own floor.
        if not cfg.usd_stage:
            self.world.scene.add_default_ground_plane()

    def _apply_ground_material(self, ground_config: Mapping[str, Any]) -> None:
        """Bind ``ground.material`` to the plane just added (sim thread). An
        explicit ``ground.color`` is the tint of a named set. A material that
        fails to build leaves the plane's flat colour."""
        ground_material = ground_config.get("material")
        if ground_material is None:
            return
        spec = material_spec(ground_material, color=ground_config.get("color"))
        material = build_material(self._isaac, name=GROUND_PLANE_NAME, spec=spec)
        if material is not None:
            self.world.scene.get_object(GROUND_PLANE_NAME).apply_visual_material(material)

    def _make_ground_matte(self) -> None:
        """Flag the ground plane's mesh prims as RTX "Matte Object"s so the
        plane is invisible to primary rays but still receives shadows, and
        turn on the matte-object/shadow-catcher render settings. Best-effort:
        never raises, so a render-settings-API change can't block boot."""
        try:
            import carb
            import omni.usd
            from pxr import Sdf, Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            root_prim = stage.GetPrimAtPath(GROUND_PLANE_PRIM_PATH)
            flagged: list[str] = []
            prims_to_check = list(Usd.PrimRange(root_prim)) if root_prim.IsValid() else []
            # primvars inherit down the namespace, so the root covers whatever
            # geometry the plane composes; meshes get it directly as well.
            for prim in prims_to_check:
                if prim == root_prim or prim.IsA(UsdGeom.Mesh):
                    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                        "isMatteObject", Sdf.ValueTypeNames.Bool
                    ).Set(True)
                    flagged.append(str(prim.GetPath()))
            settings = carb.settings.get_settings()
            settings.set(MATTE_OBJECT_SETTING, True)
            settings.set(SHADOW_CATCHER_SETTING, True)
            LOGGER.info(
                "flagged matte prims %s, set %s and %s",
                flagged,
                MATTE_OBJECT_SETTING,
                SHADOW_CATCHER_SETTING,
            )
        except Exception:
            LOGGER.exception("failed to make the ground plane matte")

    def _apply_viewport_grid(self, show_grid: bool) -> None:
        """Toggle the viewport's grid overlay through carb.settings, the only
        post-launch (not launcher-config) render lever so far. Best-effort:
        never raises, so a settings-API change can't block boot."""
        setting_path = "/app/viewport/grid/enabled"
        try:
            import carb

            carb.settings.get_settings().set(setting_path, show_grid)
            LOGGER.info("set %s to %s", setting_path, show_grid)
        except Exception:
            LOGGER.exception("failed to apply render.viewport_grid")

    def _spawn_prop(self, prop: dict[str, Any]) -> None:
        """Add a configured prop to the scene (runs on the sim thread,
        before the initial world.reset)."""
        import numpy as np

        from .spatial import to_vec3

        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = prim_name(str(prop["name"]))
        prim_path = f"/World/{name}"
        position = list(to_vec3(prop.get("position")))
        kind = str(prop.get("type", "cube"))
        orientation = prop_spawn_orientation(prop)

        if kind == VISUAL_PROP_KIND:
            self._spawn_visual_prop(prop, name, position, orientation)
            return

        if kind == "usd":
            usd_path = prop.get("usd_path")
            if not usd_path:
                raise ValueError(f"prop {name}: type 'usd' needs usd_path")
            if self._usd_exists(usd_path) is False:
                raise ValueError(f"prop {name}: usd not found: {usd_path}")
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(
                position=position, orientation=list(orientation)
            )
            apply_prop_physics(self._isaac, self.world, prim_path, prop)
            self._prop_specs[name] = {
                **prop,
                "name": name,
                "position": tuple(position),
                "spawn_orientation": orientation,
            }
            return

        if kind != "cube":
            raise ValueError(f"prop {name}: unknown type {kind!r} (cube or usd)")

        if prop.get("collision") is False:
            self._spawn_render_only_cube(prop, name, prim_path, position, orientation)
            return

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            orientation=np.array(orientation),
            size=float(prop.get("size", 0.05)),
        )
        if prop.get("scale") is not None:
            kwargs["scale"] = np.array([float(v) for v in prop["scale"]])
        display = prop_display_color(prop)
        if display is not None:
            kwargs["color"] = np.array(display)
        spec, material = self._cube_material(name, prop)
        if material is not None:
            kwargs["visual_material"] = material
            kwargs.pop("color", None)
        cls = self._isaac.FixedCuboid if prop.get("fixed") else self._isaac.DynamicCuboid
        self.world.scene.add(cls(**kwargs))
        # explicit material + offsets when the prop names them
        apply_prop_physics(self._isaac, self.world, prim_path, prop)
        self._prop_specs[name] = {
            **prop,
            "name": name,
            "position": tuple(position),
            "spawn_orientation": orientation,
            MATERIAL_SPEC_KEY: spec,
        }

    def _spawn_render_only_cube(
        self,
        prop: dict[str, Any],
        name: str,
        prim_path: str,
        position: list[float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """A box-shaped prop carrying ``"collision": False``: posed and
        coloured on the stage, with no collider and no rigid body.

        Built the same way a fixed cube is, then stripped of the two USD
        Physics schemas that make it collide, so PhysX never sees it. Like
        ``_spawn_visual_prop``'s referenced mesh, it is never added to
        ``world.scene``, never run through ``apply_prop_physics``, and never
        registered in ``_prop_specs``: a render prop has nothing for a scene
        verb (``set_prop_pose``, ``randomize_props``, ...) to find by name,
        the same split ``visual_props.py`` keeps for a visual prop.
        """
        import numpy as np

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            orientation=np.array(orientation),
            size=float(prop.get("size", 0.05)),
        )
        if prop.get("scale") is not None:
            kwargs["scale"] = np.array([float(v) for v in prop["scale"]])
        display = prop_display_color(prop)
        if display is not None:
            kwargs["color"] = np.array(display)
        self._isaac.FixedCuboid(**kwargs)

        get_prim = getattr(self._isaac, "get_prim_at_path", None)
        prim = get_prim(prim_path) if get_prim is not None else None
        if prim is not None and self._isaac.UsdPhysics is not None:
            prim.RemoveAPI(self._isaac.UsdPhysics.CollisionAPI)
            prim.RemoveAPI(self._isaac.UsdPhysics.RigidBodyAPI)

    def _cube_material(
        self, name: str, prop: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, Any | None]:
        """The normalised ``material_spec`` record and the built material for a
        cube prop, both ``None`` without a ``material`` key. A record with a
        ``None`` material means the build failed and the cube keeps its flat
        colour."""
        if prop.get("material") is None:
            return None, None
        spec = material_spec(prop["material"], color=prop.get("color"))
        return spec, build_material(self._isaac, name=name, spec=spec)

    def _spawn_visual_prop(
        self,
        prop: dict[str, Any],
        name: str,
        position: list[float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """Reference a visual-only prop (sim thread, before the initial
        world.reset). Contract:

        1. ``resolve_asset(prop["usd_path"])``; ``_usd_exists`` False ->
           ``ValueError`` naming the prop and the resolved path.
        2. ``add_reference_to_stage`` at ``/World/<name>``.
        3. ``fit.collider`` -> the named cube must already be in
           ``_prop_specs`` (boot spawns every non-visual prop first) and be
           ``type: cube``, else ``ValueError``; ``collider_dims_m`` is its
           ``prop_box_dims``. Mesh dims: ``UsdGeom.BBoxCache`` over the
           default and render purposes, ``ComputeUntransformedBound`` (the
           asset's own extent, before our pose and scale).
        4. ``SingleXFormPrim(prim_path).set_world_pose(position, orientation)``
           then ``set_local_scale`` with ``visual_scale(...)``.
        5. ``bounds_m`` = ``ComputeWorldBound`` after pose and scale, as
           ``{"min": [x, y, z], "max": [x, y, z]}``.
        6. ``self._visual_props[name] = visual_prop_record(...)``.

        No ``apply_prop_physics``, no ``world.scene.add``, never touches
        ``_prop_specs``. A name already in ``_prop_specs`` or
        ``_visual_props`` is a ``ValueError``."""
        import numpy as np
        import omni.usd
        from pxr import Usd, UsdGeom

        from .assets import resolve_asset
        from .visual_props import fit_collider_name, visual_prop_record, visual_scale

        if name in self._prop_specs or name in self._visual_props:
            raise ValueError(f"prop {name!r} already exists")
        usd_path = prop.get("usd_path")
        if not usd_path:
            LOGGER.info("visual prop %s skipped: usd_path is empty, nothing to reference", name)
            return
        resolved_path = resolve_asset(str(usd_path))
        if self._usd_exists(resolved_path) is False:
            raise ValueError(f"prop {name}: usd not found: {resolved_path}")
        prim_path = f"/World/{name}"
        self._isaac.add_reference_to_stage(usd_path=resolved_path, prim_path=prim_path)

        collider_name = fit_collider_name(prop)
        collider_dims_m: tuple[float, float, float] | None = None
        if collider_name is not None:
            collider_spec = self._prop_specs.get(prim_name(collider_name))
            if collider_spec is None or str(collider_spec.get("type", "cube")) != "cube":
                raise ValueError(
                    f"prop {name}: fit.collider {collider_name!r} must name an existing cube prop"
                )
            collider_dims_m = prop_box_dims(collider_spec)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        mesh_size = cache.ComputeUntransformedBound(prim).ComputeAlignedRange().GetSize()
        mesh_dims_m: tuple[float, float, float] = (
            float(mesh_size[0]),
            float(mesh_size[1]),
            float(mesh_size[2]),
        )
        if collider_dims_m is not None and any(dim <= 0 for dim in mesh_dims_m):
            raise ValueError(f"prop {name}: mesh has no bounds to fit against its collider")

        if collider_name is not None:
            # the mesh dresses the collider, so the grey cube stops rendering.
            # Visibility is a render attribute: PhysX keeps the collider.
            collider_prim = stage.GetPrimAtPath(f"/World/{prim_name(collider_name)}")
            UsdGeom.Imageable(collider_prim).MakeInvisible()
            LOGGER.info("hid collider %s behind visual prop %s", collider_name, name)

        xform = self._isaac.SingleXFormPrim(prim_path)
        xform.set_world_pose(position=position, orientation=list(orientation))
        scale = visual_scale(prop, collider_dims_m, mesh_dims_m)
        if scale is not None:
            xform.set_local_scale(np.array(scale))

        cache.Clear()
        world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        bounds_m = {
            "min": [float(v) for v in world_range.GetMin()],
            "max": [float(v) for v in world_range.GetMax()],
        }

        self._visual_props[name] = visual_prop_record(
            prop,
            name=name,
            resolved_path=resolved_path,
            position=(position[0], position[1], position[2]),
            orientation=orientation,
            collider_dims_m=collider_dims_m,
            mesh_dims_m=mesh_dims_m,
            bounds_m=bounds_m,
        )

    def materialise_frame_system(self, props: Sequence[Mapping[str, Any]]) -> None:
        """Spawn one static collider per frame-system part that declares a box.

        These are the machine's own furniture: a component that declares
        ``frame.geometry`` has told the planner its shape, and this makes the
        simulator agree. Props the world spawns are not part of this, and
        neither is anything a component merely draws.
        """
        if not props:
            LOGGER.info("frame system declared no colliders")
            return
        LOGGER.info("materialising %d frame-system colliders", len(props))
        self.run(
            lambda: self._spawn_frame_system_on_sim_thread(props),
            timeout=MATERIALISE_TIMEOUT_S,
            allow_during_initialization=True,
        )

    def _spawn_frame_system_on_sim_thread(self, props: Sequence[Mapping[str, Any]]) -> None:
        for prop in props:
            LOGGER.info("  spawning frame-system collider %r", prop.get("name"))
            self._spawn_prop(dict(prop))
        LOGGER.info("materialised every frame-system collider")

    def materialise_components(self, components: Mapping[str, ComponentScenery]) -> None:
        """Turn every configured workcell component's scenery into stage
        prims (runs on the sim thread, before the initial world.reset).

        One component at a time, in ``components``' iteration order:
        ``scenery_props`` gives colliders first and render props after, and
        that order is kept so a render prop's ``fit.collider`` always finds
        its cube already spawned. Each primitive's pose comes back in the
        component's own frame, so it is composed onto the component's world
        frame (``spatial.compose_pose``) before it reaches ``_spawn_prop``.
        """
        LOGGER.info("materialising %d components: %s", len(components), list(components))
        # _spawn_prop touches USD directly and does not marshal, so every one
        # of these has to execute on the sim thread. Called from a module
        # thread it is a native crash with no Python traceback, which is what
        # killed the module process the first time this path ever ran on
        # hardware. run() executes inline when already on the sim thread.
        self.run(
            lambda: self._materialise_on_sim_thread(components),
            timeout=MATERIALISE_TIMEOUT_S,
            allow_during_initialization=True,
        )

    def _materialise_on_sim_thread(self, components: Mapping[str, ComponentScenery]) -> None:
        for component, scenery in components.items():
            started = time.monotonic()
            spawned = 0
            for prop in scenery_props(
                component,
                scenery.model,
                visuals=scenery.visuals,
                geometries=scenery.geometries,
                attrs=scenery.attrs,
            ):
                local_position = to_vec3(prop.get("position"))
                local_orientation = prop_spawn_orientation(prop)
                world_position, world_orientation = compose_pose(
                    scenery.frame_position_m,
                    scenery.frame_orientation_wxyz,
                    local_position,
                    local_orientation,
                )
                LOGGER.info("  spawning %r prop %r", component, prop.get("name"))
                self._spawn_prop(
                    {
                        **prop,
                        "position": world_position,
                        "orientation_wxyz": world_orientation,
                    }
                )
                spawned += 1
            LOGGER.info(
                "materialised %r: %d prims in %.2fs", component, spawned, time.monotonic() - started
            )
        LOGGER.info("materialised every component")

    def _require_booted(self) -> None:
        if self._stopped:
            raise SimNotBootedError("Isaac Sim has stopped - the module is shutting down")
        if not self._booted.is_set():
            raise SimNotBootedError(
                "Isaac Sim world is not running - configure a "
                f"{NAMESPACE}:{FAMILY}:world component and depend on it"
            )
        if self._boot_error is not None:
            raise SimNotBootedError(f"Isaac Sim failed to boot: {self._boot_error}")

    def play(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.play())

    def pause(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.pause())

    def reset(self) -> None:
        self._require_booted()
        self.run(lambda: self._reset_world())

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "booted": self._booted.is_set(),
            "mock": self.mock,
            "error": str(self._boot_error) if self._boot_error else "",
            "lighting": self.lighting,
            "render": self.render,
            "ground": self.ground,
            # GPU checklist item 6: None in mock or when no probe answers
            "isaac_version": _version_string(isaac_version()),
            "ready": self._ready.is_set(),
            "visual_props": [status_row(record) for record in self._visual_props.values()],
        }
        if self._booted.is_set() and not self.mock and self._ready.is_set():
            out["playing"] = self.run(lambda: bool(self.world.is_playing()))
            out["sim_time"] = self.run(lambda: float(self.world.current_time))
        return out

    def world_handle(self) -> "WorldHandle":
        """The WorldHandle the world component's DoCommand verbs drive
        the scene through. Every booted world has one."""
        self._require_booted()
        assert self._world_handle is not None
        return self._world_handle

    def add_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_wxyz: "Quat | None" = None,
    ) -> None:
        self._require_booted()
        if self.mock:
            return

        def _add():
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            prim = self._isaac.SingleXFormPrim(prim_path)
            kwargs: dict[str, Any] = {"position": list(position)}
            if orientation_wxyz is not None:
                kwargs["orientation"] = list(orientation_wxyz)
            prim.set_world_pose(**kwargs)

        self.run(_add, timeout=60.0)

    def _cached_handle(self, name: str, attrs: dict[str, Any], factory: Callable[[], Any]) -> Any:
        """One handle per component name for the life of the process.

        viam-server's modular resource lifecycle never calls reconfigure()
        twice on the same live instance for a changed attribute: it removes
        the resource (close() -> release_handle, popping ``name`` here) and
        builds a fresh one, whose reconfigure() calls create_* again. So the
        branch below only returns a handle already live under ``name`` -
        e.g. a runtime-only attribute change that never went through
        close(). After release_handle the name is forgotten and the next
        create_* re-runs ``factory``, which re-attaches to the prim
        release_handle deliberately left in the stage (see _prim_spawned)."""
        if name in self._handles:
            _old_attrs, handle = self._handles[name]
            return handle
        handle = factory()
        self._handles[name] = (dict(attrs), handle)
        return handle

    def handle_entry(self, name: str) -> tuple[dict[str, Any], Any]:
        """The (spawn attrs, handle) pair registered under a component name.

        The world's per-component diagnostic verbs reach a component's sim
        state through this, so the arm, gripper and camera models never grow
        a verb a real driver could not have. Raises ValueError for a name no
        live component holds."""
        entry = self._handles.get(name)
        if entry is None:
            raise ValueError(f"no sim component named {name!r}, known: {sorted(self._handles)}")
        return entry

    def release_handle(self, name: str) -> None:
        """Called from a model's close(). Forgets the cached handle, drops the
        post-reset hooks registered under ``name`` and calls handle.release().
        The prim stays in the stage: isaacsim.core.utils.prims.delete_prim
        could remove it, but a delete plus re-add needs a world.reset(), and
        re-adding after an articulation topology change is the failure class
        the gripper attach sequence fights. A later create_* for the same
        name re-attaches to the surviving prim instead. Idempotent."""
        entry = self._handles.pop(name, None)
        self.unregister_post_reset(name)
        if entry is None:
            return
        _attrs, handle = entry
        try:
            handle.release()
        except Exception:
            LOGGER.exception("%r: handle.release() failed", name)

    def _usd_exists(self, path: str) -> bool | None:
        """True/False if we can check, None if omni.client is unavailable."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, _ = client.stat(path)
            return result == client.Result.OK
        except Exception:  # noqa: BLE001 - stat failure means we can't check, matching the documented None case
            return None

    def _resolve_usd(self, attrs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Return (absolute usd path or None, known-asset metadata)."""
        meta: dict[str, Any] = {}
        usd = attrs.get("usd_path")
        asset = attrs.get("asset")
        if asset:
            if asset not in KNOWN_ASSETS:
                raise ValueError(
                    f"unknown asset {asset!r}, known: {sorted(KNOWN_ASSETS)} "
                    "(or set usd_path directly)"
                )
            meta = KNOWN_ASSETS[asset]
            if not usd:
                # 5.0's get_assets_root_path() raises RuntimeError instead of
                # returning None on failure; re-raise with our own message so
                # it reaches the user instead of the raw Kit error.
                try:
                    root = self._isaac.get_assets_root_path()
                except RuntimeError as exc:
                    raise RuntimeError("could not reach the Isaac Sim assets server") from exc
                if root is None:
                    raise RuntimeError("could not reach the Isaac Sim assets server")
                candidates = meta["usd"]
                for rel in candidates:
                    if self._usd_exists(root + rel) is not False:
                        usd = root + rel
                        break
                if usd is None:
                    raise ValueError(
                        f"asset {asset!r}: none of {candidates} exist under {root}. "
                        "The asset layout may have changed in this Isaac release"
                    )
        # a USD reference to a missing file "succeeds" but leaves an empty
        # prim, which later fails with confusing physics-tensor errors -
        # catch it here instead
        if usd and self._usd_exists(usd) is False:
            raise ValueError(f"usd not found: {usd}")
        return usd, meta

    def create_arm(self, name: str, attrs: dict[str, Any]) -> "ArmHandle":
        self._require_booted()
        home_rad = _home_joints_rad(attrs)
        if self.mock:

            def factory():
                handle = MockArmHandle(name, attrs)
                if home_rad is not None:
                    handle.home(home_rad)
                return handle
        else:

            def factory():
                def _spawn():
                    handle = self._create_arm_isaac(name, attrs)
                    # snapshot the controller gains once and
                    # apply the solver iteration count, so post_reset (below)
                    # can re-apply both after a world.reset() undoes them.
                    handle._solver_iterations = ARM_SOLVER_POSITION_ITERATIONS
                    handle._art.set_solver_position_iteration_count(ARM_SOLVER_POSITION_ITERATIONS)
                    handle._gains = handle._art.get_articulation_controller().get_gains()
                    return handle

                handle = self.run(_spawn, timeout=120.0, allow_during_initialization=True)
                if home_rad is not None:
                    handle.home(home_rad)
                    LOGGER.info(
                        "arm %r placed at its home pose (deg): %s",
                        name,
                        [round(math.degrees(v), 2) for v in home_rad],
                    )
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle(name, attrs, factory)

    def _place_root_xform(self, prim_path: str, position: Vec3, orientation: Quat) -> bool:
        """Author the spawn pose as plain USD xform ops on the referenced asset's
        root prim, BEFORE any Isaac prim wrapper exists for it.

        Passing position/orientation through SingleArticulation does not work
        once a physics sim view exists (the world was reset at boot): the
        wrapper routes the write to a physics handle that has not parsed the
        new articulation yet, drops it, and captures identity as the default
        state (Isaac 5.0 xform_prim.py:150-175). Writing the ops here makes the
        USD pose the truth PhysX parses on the next world.reset(). Never raises."""
        try:
            from pxr import Gf, UsdGeom

            stage = self._isaac.get_prim_at_path(prim_path).GetStage()
            prim = stage.GetPrimAtPath(prim_path)
            xformable = UsdGeom.Xformable(prim)
            # ClearXformOpOrder drops the order, not the op attributes: a
            # referenced asset may already carry xformOp:orient as quatd or
            # quatf, and AddOrientOp raises if the requested precision differs.
            # Match whatever precision is already authored (default double).
            xformable.ClearXformOpOrder()
            double = UsdGeom.XformOp.PrecisionDouble
            single = UsdGeom.XformOp.PrecisionFloat
            translate_attr = prim.GetAttribute("xformOp:translate")
            translate_is_float = (
                bool(translate_attr) and str(translate_attr.GetTypeName()) == "float3"
            )
            orient_attr = prim.GetAttribute("xformOp:orient")
            orient_is_float = bool(orient_attr) and str(orient_attr.GetTypeName()) == "quatf"
            scale_attr = prim.GetAttribute("xformOp:scale")
            scale_is_double = bool(scale_attr) and str(scale_attr.GetTypeName()) == "double3"

            px, py, pz = (float(v) for v in position)
            translate_op = xformable.AddTranslateOp(single if translate_is_float else double)
            translate_op.Set(Gf.Vec3f(px, py, pz) if translate_is_float else Gf.Vec3d(px, py, pz))

            w, x, y, z = (float(v) for v in orientation)
            orient_op = xformable.AddOrientOp(single if orient_is_float else double)
            orient_op.Set(
                Gf.Quatf(w, Gf.Vec3f(x, y, z))
                if orient_is_float
                else Gf.Quatd(w, Gf.Vec3d(x, y, z))
            )

            scale_op = xformable.AddScaleOp(double if scale_is_double else single)
            scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0) if scale_is_double else Gf.Vec3f(1.0, 1.0, 1.0))
            LOGGER.info(
                "placed %s via usd xform ops: position=%s orientation=%s",
                prim_path,
                position,
                orientation,
            )
            return True
        except Exception:
            LOGGER.exception("failed to author spawn pose on %s", prim_path)
            return False

    def _anchor_fixed_base(self, prim_path: str, position: Vec3, orientation: Quat) -> bool:
        """Re-anchor a world-anchored fixed-base joint under prim_path to the
        spawn pose (position, orientation). The UR assets fix their base to
        the world frame with a FixedJoint whose localPos0/localRot0 are
        authored in world coordinates. PhysX re-syncs the root xform to that
        joint frame on world.reset(), silently undoing the spawn pose passed
        to SingleArticulation. Never raises: a failure
        here should not fail the spawn, only leave the pose un-anchored."""
        try:
            from pxr import Gf, Sdf, Usd, UsdPhysics

            stage = self._isaac.get_prim_at_path(prim_path).GetStage()
            root_prim = stage.GetPrimAtPath(prim_path)
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                joint = UsdPhysics.Joint(prim)
                if joint.GetBody0Rel().GetTargets():
                    continue

                pos_attr = prim.GetAttribute("physics:localPos0")
                rot_attr = prim.GetAttribute("physics:localRot0")
                authored_pos_gf = pos_attr.Get() if pos_attr else None
                authored_rot_gf = rot_attr.Get() if rot_attr else None
                authored_pos: Vec3 = (
                    (authored_pos_gf[0], authored_pos_gf[1], authored_pos_gf[2])
                    if authored_pos_gf is not None
                    else (0.0, 0.0, 0.0)
                )
                authored_quat: Quat = (
                    (
                        authored_rot_gf.GetReal(),
                        authored_rot_gf.GetImaginary()[0],
                        authored_rot_gf.GetImaginary()[1],
                        authored_rot_gf.GetImaginary()[2],
                    )
                    if authored_rot_gf is not None
                    else (1.0, 0.0, 0.0, 0.0)
                )

                new_pos, new_quat = anchor_fixed_joint_frame(
                    position, orientation, authored_pos, authored_quat
                )
                if pos_attr is None:
                    pos_attr = prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f)
                if rot_attr is None:
                    rot_attr = prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf)
                pos_attr.Set(Gf.Vec3f(*new_pos))
                rot_attr.Set(Gf.Quatf(new_quat[0], Gf.Vec3f(*new_quat[1:])))
                LOGGER.info(
                    "re-anchored fixed base joint %s to position=%s orientation=%s",
                    prim.GetPath(),
                    new_pos,
                    new_quat,
                )
                return True
            LOGGER.warning(
                "no world-anchored fixed joint found under %s, spawn pose relies on the prim xform",
                prim_path,
            )
            return False
        except Exception:
            LOGGER.exception("failed to re-anchor fixed base joint under %s", prim_path)
            return False

    def _create_arm_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacArmHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{prim_name(name)}"
        position = to_vec3(attrs.get("position"))
        orientation = spawn_orientation(attrs, meta)
        # A re-attach (release_handle popped the cache but deliberately left
        # the prim in the stage) skips re-adding the USD reference and the
        # scene registration below: PhysX already has this articulation's
        # physics view, and _place_root_xform's own docstring is why the
        # spawn pose can't be re-authored once that view exists anyway.
        reattaching = name in self._prim_spawned
        if usd and not reattaching:
            self._isaac.add_reference_to_stage(usd_path=usd, prim_path=prim_path)
            # Spawn pose goes into USD first (see _place_root_xform) and the
            # world-anchored base joint is moved with it. The wrapper below is
            # constructed WITHOUT a pose on purpose.
            self._place_root_xform(prim_path, position, orientation)
            self._anchor_fixed_base(prim_path, position, orientation)

        art = self._isaac.SingleArticulation(prim_path=prim_path, name=name)
        if reattaching:
            # bind the fresh wrapper to the already-live physics view instead
            # of a duplicate scene.add + a world reset that would undo every
            # other component's state (refresh_dofs uses the same call for a
            # wrapper registered after the world's first reset).
            art.initialize()
        else:
            self.world.scene.add(art)
            self._reset_world()
        self._prim_spawned.add(name)
        try:
            root_pos, root_quat = art.get_world_pose()
            LOGGER.info(
                "arm %r root pose after reset: position=%s orientation_wxyz=%s (requested %s / %s)",
                name,
                [round(float(v), 4) for v in root_pos],
                [round(float(v), 4) for v in root_quat],
                position,
                orientation,
            )
        except Exception:
            LOGGER.exception("could not read root pose for arm %r after reset", name)

        end_effector = None
        asset = attrs.get("asset")
        known_ee_prim = (
            KNOWN_ASSETS[asset].get("ee_prim") if asset and asset in KNOWN_ASSETS else None
        )
        ee_path = attrs.get("end_effector_prim") or (
            f"{prim_path}/{known_ee_prim}" if known_ee_prim else None
        )
        if ee_path:
            end_effector = self._isaac.SingleXFormPrim(ee_path)
        correction = (
            _as_quat(meta["base_frame_correction"])
            if meta.get("base_frame_correction") is not None
            else (1.0, 0.0, 0.0, 0.0)
        )
        return IsaacArmHandle(
            self,
            art,
            end_effector,
            meta.get("joint_names"),
            base_correction=correction,
            prim_path=prim_path,
        )

    def create_camera(self, name: str, attrs: dict[str, Any]) -> "CameraHandle":
        self._require_booted()
        # wired once per handle (not in the model) so both backends
        # drop their cache / re-arm acquisition after every world.reset().
        # Registered inside factory() - which _cached_handle only calls on
        # first construction - because viam-server re-runs reconfigure ->
        # create_camera on every config change and _cached_handle returns the
        # same handle each time. Registering outside factory() would append
        # a duplicate hook per reconfigure. Dispatched dynamically (not a
        # bound-method reference captured now) so tests can monkeypatch
        # handle.post_reset after creation.
        if self.mock:

            def factory():
                handle = MockCameraHandle(name, attrs)
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle
        else:

            def factory():
                handle = self.run(
                    lambda: self._create_camera_isaac(name, attrs),
                    timeout=120.0,
                    allow_during_initialization=True,
                )
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle(name, attrs, factory)

    def _create_camera_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacCameraHandle":
        parent = attrs.get("parent_prim")
        if parent:
            self._require_prim(parent)
        prim_path = _camera_prim_path(name, attrs)
        width = int(attrs.get("width", DEFAULT_CAMERA_WIDTH))
        height = int(attrs.get("height", DEFAULT_CAMERA_HEIGHT))

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            resolution=(width, height),
        )
        if attrs.get("position") is not None:
            kwargs["position"] = list(to_vec3(attrs.get("position")))
        if attrs.get("orientation_rpy_deg") is not None:
            roll, pitch, yaw = to_vec3(attrs.get("orientation_rpy_deg"))
            kwargs["orientation"] = list(quat_from_euler_deg(roll, pitch, yaw))

        annotator_device = attrs.get("annotator_device")
        if annotator_device is not None:
            if caps().camera_supports_annotator_device:
                kwargs["annotator_device"] = annotator_device
            else:
                LOGGER.info(
                    "camera %s: annotator_device %r ignored - the running Isaac "
                    "release has no GPU-resident annotator path (5.0 only)",
                    name,
                    annotator_device,
                )

        cam = self._isaac.Camera(**kwargs)
        # 4.5: get_resolution()/apertures only read back correctly once the
        # render product exists, so initialize() must come first.
        self._initialize_camera(name, cam)

        _place_camera(cam, attrs)
        _configure_camera_optics(cam, attrs)

        return IsaacCameraHandle(
            self,
            cam,
            depth_enabled=bool(attrs.get("depth")),
            image_format=attrs.get("image_format", "png"),
            frequency=attrs.get("frequency"),
        )

    def _initialize_camera(self, name: str, cam: Any) -> None:
        """Bounded retry around ``Camera.initialize()`` for a cold renderer
        (CAMERA_INIT_ATTEMPTS's comment has the failure). Runs on the sim
        thread, where the loop is paused while we execute, so the render
        ticks between attempts are stepped here. On final failure the
        half-created render product is destroyed so the next resource build
        starts clean instead of wrapping the corpse."""
        last_error: Exception | None = None
        for attempt in range(CAMERA_INIT_ATTEMPTS):
            if attempt:
                for _ in range(CAMERA_INIT_RENDER_TICKS):
                    self.world.step(render=True)
            try:
                cam.initialize()
                return
            except Exception as exc:  # noqa: BLE001 - any init failure must be retried, not just anticipated exception types
                last_error = exc
                LOGGER.warning(
                    "camera %s: initialize attempt %d/%d failed: %r",
                    name,
                    attempt + 1,
                    CAMERA_INIT_ATTEMPTS,
                    exc,
                )
        destroy = getattr(cam, "destroy", None)
        if destroy is not None:
            try:
                destroy()
            except Exception:
                LOGGER.exception("camera %s: destroy() after failed initialize", name)
        raise CameraInitError(
            f"camera {name}: initialize() failed after {CAMERA_INIT_ATTEMPTS} attempts "
            f"with {CAMERA_INIT_RENDER_TICKS} render ticks between them"
        ) from last_error

    def _require_prim(self, prim_path: str) -> None:
        """Raise a helpful error if prim_path doesn't exist in the stage."""
        get_prim = getattr(self._isaac, "get_prim_at_path", None)
        if get_prim is None:
            return
        prim_path = prim_path.strip()  # a pasted path with stray whitespace is never valid
        prim = get_prim(prim_path)
        if prim is None or not prim.IsValid():
            parent_path = prim_path.rsplit("/", 1)[0] or "/"
            hint = ""
            parent = get_prim(parent_path)
            if parent is not None and parent.IsValid():
                children = [c.GetName() for c in parent.GetChildren()]
                hint = f", children of {parent_path}: {children}"
            raise PrimNotFoundError(f"prim not found: {prim_path}{hint}")

    def create_base(self, name: str, attrs: dict[str, Any]) -> "BaseHandle":
        self._require_booted()
        if self.mock:

            def factory():
                return MockBaseHandle(name, attrs)
        else:

            def factory():
                return self.run(
                    lambda: self._create_base_isaac(name, attrs),
                    timeout=120.0,
                    allow_during_initialization=True,
                )

        return self._cached_handle(name, attrs, factory)

    def _create_base_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacBaseHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{prim_name(name)}"
        wheel_joints = attrs.get("wheel_joints") or meta.get("wheel_joints")
        if not wheel_joints or len(wheel_joints) != 2:
            raise ValueError(
                "base needs wheel_joints: [left_joint_name, right_joint_name] "
                "(known assets like 'jetbot' provide defaults)"
            )
        wheel_radius = float(attrs.get("wheel_radius", meta.get("wheel_radius", 0.05)))
        wheel_base = float(attrs.get("wheel_base", meta.get("wheel_base", 0.3)))
        position = to_vec3(attrs.get("position"))

        # see _create_arm_isaac: a re-attach skips re-creating the USD
        # reference and the scene registration below, both already done for
        # the surviving prim.
        reattaching = name in self._prim_spawned
        base_kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=list(wheel_joints),
            create_robot=(usd is not None) and not reattaching,
            usd_path=usd,
            position=list(position),
        )
        if attrs.get("orientation_wxyz") is not None:
            base_kwargs["orientation"] = [float(v) for v in attrs["orientation_wxyz"]]
        robot = self._isaac.WheeledRobot(**base_kwargs)
        if reattaching:
            robot.initialize()
        else:
            self.world.scene.add(robot)
            self._reset_world()
        self._prim_spawned.add(name)

        # models/base.py clamps the commanded linear/angular velocity; only
        # the controller can clamp the resulting per-wheel speed, which is
        # what actually saturates the drive. max_wheel_speed is the
        # combined-motion bound: the wheel angular velocity needed to hit
        # max_linear_mps while simultaneously turning at max_angular_rps.
        max_linear_mps = float(attrs.get("max_linear_mps", 0.5))
        max_angular_rps = float(attrs.get("max_angular_rps", 2.0))
        max_wheel_speed = (max_linear_mps + max_angular_rps * wheel_base / 2.0) / wheel_radius
        controller = self._isaac.DifferentialController(
            name=f"{name}_controller",
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
            max_linear_speed=max_linear_mps,
            max_angular_speed=max_angular_rps,
            max_wheel_speed=max_wheel_speed,
        )
        handle = IsaacBaseHandle(self, robot, controller, wheel_radius, wheel_base)
        self.world.add_physics_callback(f"{name}_drive", handle._on_physics_step)
        return handle

    def create_gripper(self, name: str, attrs: dict[str, Any]) -> "JawGripperHandle":
        """Attach a gripper to an arm that is already in the sim.

        attrs: world, arm (Viam name of the arm it rides - validate_config
        lists it as a dependency so viam-server builds the arm first), asset
        (default "robotiq_2f_85", must have kind "gripper" in KNOWN_ASSETS),
        parent_prim (default the arm's ee_prim, <arm prim>/wrist_3_link for a
        known UR asset), local_position / local_orientation_rpy_deg (mount
        pose of the gripper's base_link on parent_prim, default identity -
        NOT the frame, see models/gripper.py), open_deg, closed_deg,
        holding_tolerance_deg, mock_object_width_m."""
        self._require_booted()
        arm_name = str(attrs.get("arm", ""))
        arm_entry = self._handles.get(arm_name)
        if arm_entry is None:
            raise ValueError(
                f"gripper {name!r}: arm {arm_name!r} is not attached to the sim "
                '(set "arm" to the name of the isaac-sim arm component it rides)'
            )
        arm_attrs, arm_handle = arm_entry
        if not isinstance(arm_handle, ArmHandle):
            raise ValueError(f"gripper {name!r}: {arm_name!r} is not an arm")
        asset = attrs.get("asset", "robotiq_2f_85")
        known_asset = isinstance(asset, str) and asset in KNOWN_ASSETS
        kind = KNOWN_ASSETS[asset].get("kind") if known_asset else None
        if kind != "gripper":
            raise ValueError(
                f"gripper {name!r}: asset {asset!r} has kind {kind!r}, not the "
                '"gripper" mechanism create_gripper builds'
            )
        if self.mock:

            def factory():
                return MockGripperHandle(name, attrs, arm_handle)
        else:

            def factory():
                handle = self.run(
                    lambda: self._create_gripper_isaac(name, attrs, arm_attrs, arm_handle),
                    timeout=120.0,
                    allow_during_initialization=True,
                )
                # GPU checklist item 6: re-command the last commanded
                # jaw target after a reset mid-pick, so it doesn't drop
                # whatever it was holding.
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle(name, attrs, factory)

    def _create_gripper_isaac(
        self, name: str, attrs: dict[str, Any], arm_attrs: dict[str, Any], arm: "ArmHandle"
    ) -> "IsaacGripperHandle":
        """Sim thread. Reference the asset under f"{arm prim}/Gripper" with
        articulationEnabled=False on the gripper root so its joints join the
        ARM's articulation. Author a PhysicsFixedJoint parent_prim (wrist_3_link)
        <-> gripper base_link at the local mount pose. Assert the pad prims
        exist and carry PhysicsCollisionAPI and raise a clear
        error otherwise. ONE reset via _reset_world(), then arm.refresh_dofs()
        and log the full dof_names, refusing if the six UR joint names are no
        longer resolvable."""
        from pxr import Gf, Sdf, Usd, UsdPhysics

        if not isinstance(arm, IsaacArmHandle):
            raise ValueError(
                f"gripper {name!r}: arm handle for {attrs.get('arm')!r} is not an Isaac arm handle"
            )

        gripper_attrs = dict(attrs)
        gripper_attrs.setdefault("asset", "robotiq_2f_85")
        usd, meta = self._resolve_usd(gripper_attrs)
        if usd is None:
            raise ValueError(f"gripper {name!r}: no usd_path or known asset resolved")
        try:
            from pxr import UsdUtils as usd_utils
        except ImportError:  # pragma: no cover - older Kit builds
            usd_utils = None
        usd, prepared = self._prepared_asset_layer(Sdf, usd_utils, usd)
        LOGGER.info("gripper %r asset layer %s: %s", name, prepared["reason"], prepared)

        arm_prim = arm._prim_path
        gripper_prim = f"{arm_prim}/Gripper"
        self._isaac.add_reference_to_stage(usd_path=usd, prim_path=gripper_prim)

        stage = self._isaac.get_prim_at_path(gripper_prim).GetStage()
        gripper_root = stage.GetPrimAtPath(gripper_prim)

        # diagnostics: what actually composed under the reference. A layer
        # without a defaultPrim composes NOTHING through AddReference(path), so
        # re-reference its first root prim explicitly before giving up.
        composed = _describe_composition(Usd, Sdf, gripper_root, usd)
        LOGGER.info("gripper %r composed under %s: %s", name, gripper_prim, composed)
        if not composed["children"] and composed["layer_root_prims"]:
            if not composed["layer_default_prim"]:
                root_prim_path = composed["layer_root_prims"][0]
                references = gripper_root.GetReferences()
                references.ClearReferences()
                references.AddReference(Sdf.Reference(usd, Sdf.Path(root_prim_path)))
                composed = _describe_composition(Usd, Sdf, gripper_root, usd)
                LOGGER.info(
                    "gripper %r: layer has no defaultPrim, re-referenced %s explicitly: %s",
                    name,
                    root_prim_path,
                    composed,
                )
        if not composed["children"]:
            raise ValueError(
                f"gripper {name!r}: {usd} composed no prims under {gripper_prim} "
                f"(layer defaultPrim={composed['layer_default_prim']!r}, root prims="
                f"{composed['layer_root_prims']}). The asset did not load - check the module "
                "log for omni.client/USD resolver errors and the asset path"
            )

        rewrite = self._rewrite_unresolvable_references(Usd, Sdf, gripper_root)
        LOGGER.info(
            "gripper %r: %s reference rewrite - applied %d, missing in bucket %d, "
            "de-instanced %d: %s",
            name,
            UNRESOLVABLE_ASSET_HOST,
            len(rewrite["applied"]),
            len(rewrite["missing"]),
            len(rewrite["de_instanced"]),
            rewrite,
        )

        # The gripper's own articulation root must GO, not merely be disabled:
        # it now sits inside the arm root's subtree, and PhysX drops an
        # articulation that contains a nested root (seen on the GPU as the
        # arm's view losing its backend). Its joints then join the arm's
        # articulation through the fixed joint below. The API is on the
        # asset's default prim (Gripper/Robotiq_2F_85), not on our Gripper prim.
        removed_roots = _remove_articulation_roots(
            Usd, UsdPhysics, self._isaac.PhysxSchema, gripper_root
        )
        if removed_roots:
            LOGGER.info("gripper %r: removed articulation root API from %s", name, removed_roots)
        else:
            LOGGER.warning("gripper %r: no ArticulationRootAPI found under %s", name, gripper_prim)

        base_link_prim = None
        for prim in Usd.PrimRange(gripper_root):
            if prim.GetName() == "base_link":
                base_link_prim = prim
                break
        if base_link_prim is None:
            raise ValueError(f"gripper {name!r}: base_link prim not found under {gripper_prim}")

        parent_prim = attrs.get("parent_prim") or f"{arm_prim}/wrist_3_link"
        local_position = to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.0))
        roll, pitch, yaw = to_vec3(attrs.get("local_orientation_rpy_deg"), default=(0.0, 0.0, 0.0))
        local_quat = quat_from_euler_deg(roll, pitch, yaw)

        joint = UsdPhysics.FixedJoint.Define(stage, f"{gripper_prim}/WristFixedJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_prim)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(str(base_link_prim.GetPath()))])
        px, py, pz = (float(v) for v in local_position)
        joint.CreateLocalPos0Attr(Gf.Vec3f(px, py, pz))
        quat_w, quat_x, quat_y, quat_z = (float(v) for v in local_quat)
        joint.CreateLocalRot0Attr(Gf.Quatf(quat_w, Gf.Vec3f(quat_x, quat_y, quat_z)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        LOGGER.info(
            "authored gripper wrist fixed joint %s/WristFixedJoint: body0=%s body1=%s",
            gripper_prim,
            parent_prim,
            base_link_prim.GetPath(),
        )

        # The 2F-85 pad prims must carry PhysicsCollisionAPI -
        # verify this BEFORE resetting so a missing-asset failure is clear.
        # A pad counts as collidable when it OR any descendant (the asset
        # nests collision meshes under the link) carries the API. Everything
        # observed is logged before the refusal, so the GPU checklist can
        # record it from the module log alone.
        pad_status = _pad_collision_status(Usd, UsdPhysics, gripper_root)
        asset_refs = _reference_asset_paths(Usd, gripper_root)
        unresolved_refs = [path for path in asset_refs if UNRESOLVABLE_ASSET_HOST in path]
        LOGGER.info(
            "gripper %r pad prims (path, collision on self, collision in subtree): %s",
            name,
            pad_status,
        )
        LOGGER.info(
            "gripper %r asset references under %s: %d total, %d on %s: %s",
            name,
            gripper_prim,
            len(asset_refs),
            len(unresolved_refs),
            UNRESOLVABLE_ASSET_HOST,
            unresolved_refs,
        )
        if not any(in_subtree for _path, _on_self, in_subtree in pad_status):
            raise ValueError(
                f"gripper {name!r}: 2F-85 pad prims missing PhysicsCollisionAPI. "
                f"Pads seen: {pad_status}. References on {UNRESOLVABLE_ASSET_HOST}: "
                f"{unresolved_refs}. Rewrite: {rewrite}. Composed: {composed}. "
                "Rewrite references to the assets bucket's parts/*.usd, "
                "or use a module-authored parallel-jaw USD"
            )

        dof_count_before = len(arm.all_dof_names())
        # The arm's SingleArticulation was initialized for the arm alone. Its
        # default joint state is sized for that DOF count and the scene's
        # post_reset would push it into the new, larger articulation and fail.
        # Register a fresh wrapper BEFORE the reset so reset() initializes it
        # against the new topology (the prim, and the arm's name, stay).
        arm_object_name = arm._art.name
        _forget_scene_object(self.world.scene, arm_object_name)
        fresh_articulation = self._isaac.SingleArticulation(
            prim_path=arm._prim_path, name=arm_object_name
        )
        self.world.scene.add(fresh_articulation)
        arm.replace_articulation(fresh_articulation)
        self._reset_world()
        arm.refresh_dofs()
        all_names = arm.all_dof_names()
        LOGGER.info(
            "arm %r articulation dof_names after gripper attach (%d): %s",
            arm_prim,
            len(all_names),
            all_names,
        )

        added_dof_count = len(all_names) - dof_count_before
        expected_dof_count = caps().gripper_dof_count
        if added_dof_count != expected_dof_count:
            LOGGER.warning(
                "gripper %r added %d dofs to the arm articulation, expected %d for this Isaac "
                "release",
                name,
                added_dof_count,
                expected_dof_count,
            )

        drive_joint = meta.get("drive_joint", "finger_joint")
        # Defaults for open/closed come from the drive joint's authored limits
        # (observed: this asset rests OPEN at its lower limit of ~7.8 deg, not
        # at 0 as the paper spec assumed, and closes at 47). Explicit attrs
        # still win, and the paper values remain the fallback when limits
        # cannot be read.
        lower_deg, upper_deg = self._drive_joint_limits_deg(arm, drive_joint)
        open_deg_default = lower_deg if lower_deg is not None else caps().gripper_open_deg
        closed_deg_default = upper_deg if upper_deg is not None else caps().gripper_closed_deg
        open_rad = math.radians(attrs.get("open_deg", open_deg_default))
        closed_rad = math.radians(attrs.get("closed_deg", closed_deg_default))
        LOGGER.info(
            "gripper %r drive joint %r: open %.2f deg, closed %.2f deg (authored limits %s..%s)",
            name,
            drive_joint,
            math.degrees(open_rad),
            math.degrees(closed_rad),
            lower_deg,
            upper_deg,
        )
        holding_tolerance_rad = math.radians(
            attrs.get("holding_tolerance_deg", DEFAULT_HOLDING_TOLERANCE_DEG)
        )
        handle = IsaacGripperHandle(
            self,
            arm._art,
            drive_joint,
            open_rad,
            closed_rad,
            holding_tolerance_rad,
            gripper_prim,
        )
        handle.parent_prim_path = parent_prim
        # the gripper handle just released the passive drives. The arm's
        # post-reset hook re-applies its gains snapshot, so retake it now
        try:
            arm._gains = arm._art.get_articulation_controller().get_gains()
        except Exception:
            LOGGER.exception("could not re-snapshot the arm gains after the gripper attach")
        return handle

    @staticmethod
    def _drive_joint_limits_deg(
        arm: "IsaacArmHandle", drive_joint: str
    ) -> tuple[float | None, float | None]:
        """(lower, upper) authored limits of the gripper's drive joint in
        degrees, read off the arm articulation after attach. (None, None)
        when the joint is absent or the API is unavailable. Sim thread."""
        names = arm.all_dof_names()
        if drive_joint not in names:
            return None, None
        index = names.index(drive_joint)
        art = arm._art
        try:
            # SingleArticulation (5.0) exposes the view. Older wrappers had
            # get_dof_limits / dof_properties directly
            view = getattr(art, "_articulation_view", None)
            if view is not None and hasattr(view, "get_dof_limits"):
                limits = view.get_dof_limits()
                row = limits[0][index] if len(getattr(limits, "shape", ())) == 3 else limits[index]
                return math.degrees(float(row[0])), math.degrees(float(row[1]))
            if hasattr(art, "get_dof_limits"):
                row = art.get_dof_limits()[index]
                return math.degrees(float(row[0])), math.degrees(float(row[1]))
            properties = getattr(art, "dof_properties", None)
            if properties is not None:
                return (
                    math.degrees(float(properties["lower"][index])),
                    math.degrees(float(properties["upper"][index])),
                )
        except Exception:
            LOGGER.exception("could not read the drive joint limits for %r", drive_joint)
        return None, None

    def create_vacuum_gripper(self, name: str, attrs: dict[str, Any]) -> "VacuumGripperHandle":
        """Attach a vacuum cup to an arm that is already in the sim.

        attrs: world, arm (Viam name of the arm it rides), parent_prim
        (default the arm's ee_prim, <arm prim>/wrist_3_link for a known UR
        asset), local_position / local_orientation_rpy_deg (mount pose of
        the tool on parent_prim, default identity), tcp_offset_m (default
        VACUUM_TOOL's), max_payload_gap_m (default DEFAULT_MAX_PAYLOAD_GAP_M),
        mock_attach_prop (mock only - the prop name grab() finds under the
        cup)."""
        self._require_booted()
        arm_name = str(attrs.get("arm", ""))
        arm_entry = self._handles.get(arm_name)
        if arm_entry is None:
            raise ValueError(
                f"vacuum gripper {name!r}: arm {arm_name!r} is not attached to the sim "
                '(set "arm" to the name of the isaac-sim arm component it rides)'
            )
        _arm_attrs, arm_handle = arm_entry
        if not isinstance(arm_handle, ArmHandle):
            raise ValueError(f"vacuum gripper {name!r}: {arm_name!r} is not an arm")

        if self.mock:

            def factory():
                return MockVacuumHandle(name, attrs, arm_handle)
        else:

            def factory():
                handle = self.run(
                    lambda: self._create_vacuum_gripper_isaac(name, attrs, arm_handle),
                    timeout=120.0,
                    allow_during_initialization=True,
                )
                # mirrors create_gripper's GPU checklist item 6: re-weld the
                # suction joint after a reset mid-pick, so it doesn't drop
                # whatever it was holding.
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle(name, attrs, factory)

    def _create_vacuum_gripper_isaac(
        self, name: str, attrs: dict[str, Any], arm: "ArmHandle"
    ) -> "IsaacVacuumHandle":
        """Sim thread. Author the tool as plain USD geometry parented under
        parent_prim, with no physics of its own.

        The tool needs no rigid body and no collider. It is not what holds a
        payload: grab() welds the payload straight to the arm LINK, and it
        decides what to weld from prim poses rather than from contact. Giving
        the tool physics instead costs twice. A collider sitting exactly where
        the cup meets a box makes PhysX push the two apart while the weld holds
        them together, which tilts the payload and drags the arm. And a rigid
        body bolted on with its own fixed joint puts two maximal-coordinate
        joints in series off the end of an articulation, which PhysX resolves
        with an impulse big enough to throw the arm across the cell. As a plain
        USD child of the link it simply inherits the link's transform, which is
        all a tool has to do."""
        from pxr import Gf, UsdGeom

        if not isinstance(arm, IsaacArmHandle):
            raise ValueError(
                f"vacuum gripper {name!r}: arm handle for {attrs.get('arm')!r} "
                "is not an Isaac arm handle"
            )

        arm_prim = arm._prim_path
        parent_prim = attrs.get("parent_prim") or f"{arm_prim}/wrist_3_link"
        tool_prim = f"{parent_prim}/{VACUUM_TOOL_PRIM}"
        # The cube's prim origin is its CENTRE, so placing it at the flange
        # would sink half of it into the link and leave the cup face floating
        # half a tool-length past its own body. Hanging it by half its length
        # puts the face exactly at tcp_offset_m, which is what the frame and
        # the planner are told.
        tool_half_length_m = float(VACUUM_TOOL["box_mm"][2]) / 2000.0
        local_position = to_vec3(
            attrs.get("local_position"), default=(0.0, 0.0, tool_half_length_m)
        )
        roll, pitch, yaw = to_vec3(attrs.get("local_orientation_rpy_deg"), default=(0.0, 0.0, 0.0))
        local_quat = quat_from_euler_deg(roll, pitch, yaw)

        stage = self._isaac.get_prim_at_path(parent_prim).GetStage()
        cube = UsdGeom.Cube.Define(stage, tool_prim)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        px, py, pz = (float(v) for v in local_position)
        xform.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        quat_w, quat_x, quat_y, quat_z = (float(v) for v in local_quat)
        xform.AddOrientOp().Set(Gf.Quatf(quat_w, Gf.Vec3f(quat_x, quat_y, quat_z)))
        size_m = [float(v) / 1000.0 for v in VACUUM_TOOL["box_mm"]]
        xform.AddScaleOp().Set(Gf.Vec3f(*size_m))
        LOGGER.info(
            "authored vacuum tool geometry %s under %s at local %s",
            tool_prim,
            parent_prim,
            (px, py, pz),
        )

        cup_side_m = float(VACUUM_TOOL["cup_side_mm"]) / 1000.0
        max_payload_gap_m = float(attrs.get("max_payload_gap_m", DEFAULT_MAX_PAYLOAD_GAP_M))
        grab_delay_ms = float(attrs.get("grab_delay_ms", DEFAULT_GRAB_DELAY_MS))
        handle = IsaacVacuumHandle(
            self,
            name,
            tool_prim,
            cup_side_m,
            max_payload_gap_m,
            tool_half_length_m,
            grab_delay_ms=grab_delay_ms,
        )
        handle.parent_prim_path = parent_prim
        return handle

    def _prepared_asset_layer(
        self, sdf: Any, usd_utils: Any, usd: str
    ) -> tuple[str, dict[str, Any]]:
        """Applied BEFORE composition. Measured on the GPU
        (2026-09-03): the 2F-85 layer's 11 ``omniverse://isaac-dev`` part
        references each stalled omni.client ~12 s while the stage composed
        them - 131 s per module start, past viam-server's 2-minute resource
        timeout, so the gripper only came up on the retry. Opening the asset's
        own layer resolves none of its sub-references, so when that layer
        points at the unresolvable host, write a copy with those references
        moved onto the assets root (and its layer-relative paths made
        absolute, so the copy can live anywhere) under asset_cache_dir(), and
        reference the copy. Later module starts find the copy and skip even
        this. Returns (path to reference, report). The original path when
        nothing needs rewriting or the rewrite is unavailable. Sim thread."""
        report: dict[str, Any] = {"source": usd, "applied": [], "missing": [], "anchored": []}
        if usd_utils is None:
            report["reason"] = "pxr.UsdUtils unavailable"
            return usd, report
        prepared = self._prepare_layer(sdf, usd_utils, usd, report, {}, ASSET_PREPARE_MAX_DEPTH)
        if prepared == usd and "reason" not in report:
            if usd in report.get("unopened", ()):
                report["reason"] = "layer did not open"
            else:
                report["reason"] = f"no {UNRESOLVABLE_ASSET_HOST} references in the closure"
        return prepared, report

    def _prepare_layer(
        self,
        sdf: Any,
        usd_utils: Any,
        usd: str,
        report: dict[str, Any],
        prepared_by_source: dict[str, str],
        depth_left: int,
    ) -> str:
        """The recursive step of ``_prepared_asset_layer`` (GPU 2026-09-04:
        the 5.0 ``Robotiq_2F_85_edit.usd`` carries its isaac-dev references
        one layer DOWN, in its payload ``Robotiq_2F_85_base.usda``, so the
        old single-layer check said "no references" and every module start
        re-paid ~13 dead-host stalls - 2 m 11 s, past the RDK's hard
        2-minute AddResource deadline, which is why reload-local could never
        reconfigure). Returns the path to reference for ``usd``: a cached
        rewritten copy when its composition closure touches the unresolvable
        host (parents are rewritten to reference their children's copies),
        else ``usd`` unchanged. Sim thread."""
        if usd in prepared_by_source:
            return prepared_by_source[usd]
        prepared_by_source[usd] = usd  # cycle guard, overwritten on a rewrite
        stem = usd.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        digest = hashlib.sha1(usd.encode()).hexdigest()[:12]
        local = asset_cache_dir() / f"{stem}-{digest}-v{ASSET_LAYER_CACHE_VERSION}.usd"
        if local.exists():
            report["reason"] = "cached"
            prepared_by_source[usd] = str(local)
            return str(local)
        layer = sdf.Layer.FindOrOpen(usd)
        if layer is None:
            report.setdefault("unopened", []).append(usd)
            return usd
        dependencies_of = (
            getattr(layer, "GetCompositionAssetDependencies", None) or layer.GetExternalReferences
        )
        dependencies = [str(path) for path in dependencies_of()]
        child_copies: dict[str, str] = {}
        if depth_left > 0:
            for dependency in dependencies:
                if UNRESOLVABLE_ASSET_HOST in dependency or not _is_layer_path(dependency):
                    continue
                absolute = _anchor_asset_path(dependency, usd)
                prepared_child = self._prepare_layer(
                    sdf, usd_utils, absolute, report, prepared_by_source, depth_left - 1
                )
                if prepared_child != absolute:
                    child_copies[dependency] = prepared_child
        has_unresolvable = any(UNRESOLVABLE_ASSET_HOST in dependency for dependency in dependencies)
        if not child_copies and not has_unresolvable:
            return usd
        try:
            assets_root = self._isaac.get_assets_root_path()
        except RuntimeError:
            assets_root = None
        if not assets_root:
            report["reason"] = "no assets root"
            return usd
        copy = sdf.Layer.CreateAnonymous(f"{stem}-prepared.usd")
        copy.TransferContent(layer)

        def rewrite(path: Any) -> str:
            raw = str(path)
            if raw in child_copies:
                report.setdefault("relinked", []).append((raw, child_copies[raw]))
                return child_copies[raw]
            return _prepared_asset_path(raw, usd, assets_root, self._usd_exists, report)

        usd_utils.ModifyAssetPaths(copy, rewrite)
        local.parent.mkdir(parents=True, exist_ok=True)
        if not copy.Export(str(local)):
            report["reason"] = f"could not write {local}"
            return usd
        report["reason"] = "prepared"
        prepared_by_source[usd] = str(local)
        return str(local)

    def _rewrite_unresolvable_references(
        self, usd: Any, sdf: Any, root_prim: Any
    ) -> dict[str, Any]:
        """First fallback (confirmed on the GPU, 2026-08-28): the
        5.0 2F-85 part meshes - the ONLY visual and collision geometry the
        asset has - are referenced from omniverse://isaac-dev..., which is
        NXDOMAIN outside NVIDIA, so the gripper composes with no geometry at
        all. The same files should live under the public assets root at the
        same /Isaac/... path, so re-point each such reference at the bucket,
        as an override in this stage's root layer (the remote asset layer is
        read-only). Instanceable prims are de-instanced first: a reference
        that lives on an instance proxy cannot be overridden. A part the
        bucket lacks is left alone, and the bucket directory is listed once so
        the report shows what IS there. Returns the report (applied pairs,
        missing candidates, de-instanced prims, bucket listing). Sim thread."""
        report: dict[str, Any] = {
            "applied": [],
            "missing": [],
            "de_instanced": [],
            "bucket_listing": {},
        }
        try:
            assets_root = self._isaac.get_assets_root_path()
        except RuntimeError:
            assets_root = None
        if not assets_root:
            LOGGER.warning("cannot re-point %s references: no assets root", UNRESOLVABLE_ASSET_HOST)
            report["error"] = "no assets root"
            return report
        # Two passes by PATH, never by held prim object: de-instancing an
        # instance expires every proxy prim under it, and touching an expired
        # proxy from a previously materialized traversal raises.
        stage = root_prim.GetStage()
        instance_paths = [
            str(prim.GetPath())
            for prim in _prim_range(usd, root_prim)
            if not prim.IsInstanceProxy() and prim.IsInstanceable()
        ]
        for path in instance_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsInstanceable():
                prim.SetInstanceable(False)
                report["de_instanced"].append(path)
        rewrite_paths = [
            str(prim.GetPath())
            for prim in _prim_range(usd, root_prim)
            if not prim.IsInstanceProxy()
        ]
        for path in rewrite_paths:
            prim = stage.GetPrimAtPath(path)
            try:
                list_op = prim.GetMetadata("references")
                items = list(list_op.GetAddedOrExplicitItems()) if list_op is not None else []
            except Exception:  # noqa: BLE001 - unreadable reference metadata skips this path, rest of the rewrite continues
                continue
            new_items, pairs, missing = _rewritten_references(
                sdf, items, assets_root, self._usd_exists
            )
            report["missing"].extend(missing)
            if pairs:
                prim.GetReferences().SetReferences(new_items)
                report["applied"].extend(pairs)
        for candidate in report["missing"]:
            directory = candidate.rsplit("/", 1)[0] + "/"
            if directory not in report["bucket_listing"]:
                report["bucket_listing"][directory] = self._list_assets_dir(directory)
        return report

    def _list_assets_dir(self, url: str) -> list[str] | None:
        """Names under a bucket directory via omni.client.list. None when the
        listing is unavailable (no client, or the directory does not exist)."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, entries = client.list(url)
            if result != client.Result.OK:
                return None
            return sorted(str(entry.relative_path) for entry in entries)
        except Exception:  # noqa: BLE001 - any listing failure means unavailable, matching the documented None case
            return None
