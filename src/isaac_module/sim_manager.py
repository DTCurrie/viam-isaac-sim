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
import math
import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from viam.logging import getLogger

from . import FAMILY, NAMESPACE
from .camera_base import CameraHandle, Frame, NoFrameYetError
from .compat import import_isaac, isaac_version
from .encoding import Intrinsics
from .errors import PrimNotFoundError, SimNotBootedError, SimTimeoutError
from .mock_camera import MockCameraHandle
from .spatial import (
    Quat,
    Vec3,
    look_at_quat,
    quat_conj,
    quat_from_euler_deg,
    quat_mul,
    quat_rotate,
    to_vec3,
)

LOGGER = getLogger("viam-isaac-sim")

# Assets shipped on the Isaac Sim nucleus/content server, addressable by a
# short name in component config. Paths are relative to the assets root;
# where isaac 5.0 moved an asset, the 5.0 path is listed first with the 4.x
# path as a fallback - the first candidate that exists is used.
_UR_KINEMATICS = (
    "https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics"
)

# the 6 UR joints in SVA (spatial vector algebra) order - the order the arm
# component's kinematics/motion planning expects, which need not match the
# articulation's PhysX dof order (FINDINGS ARM-1; R-3).
UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

KNOWN_ASSETS: dict[str, dict[str, Any]] = {
    "ur3e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur3e.json",
        "joint_names": UR_JOINT_NAMES,
    },
    "ur5e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur5e.json",
        # verified (FINDINGS XC-1/ARM-10, W7/W9): the usd asset's own base
        # link is rotated 180deg about Z relative to the kinematics frame.
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
    },
    "ur10": {"usd": ["/Isaac/Robots/UniversalRobots/ur10/ur10.usd"], "joint_names": UR_JOINT_NAMES},
    "ur10e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"],
        "joint_names": UR_JOINT_NAMES,
    },
    "ur16e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur16e/ur16e.usd"],
        "joint_names": UR_JOINT_NAMES,
    },
    # ur3e/ur10*/ur16e correction is unchecked; deliberately no entry
    # (identity) until verified.
    "ur20": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur20/ur20.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur20.json",
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
    },
    "franka": {
        "usd": [
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "/Isaac/Robots/Franka/franka.usd",
        ]
    },
    "jetbot": {
        "usd": [
            "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            "/Isaac/Robots/Jetbot/jetbot.usd",
        ],
        "wheel_joints": ["left_wheel_joint", "right_wheel_joint"],
        "wheel_radius": 0.03,
        "wheel_base": 0.1125,
    },
}


@dataclass
class SimConfig:
    mock: bool = False
    headless: bool = True
    livestream: bool = True
    usd_stage: str | None = None
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    boot_timeout: float = 300.0
    # IP the livestream advertises to clients; auto-detected if empty
    livestream_public_ip: str = ""
    # props to spawn into the scene at boot; each entry:
    #   {"type": "cube"|"usd", "name": ..., "position": [x,y,z] (m),
    #    "size": edge_m, "scale": [sx,sy,sz], "color": [r,g,b] 0-1,
    #    "fixed": bool, "usd_path": ...}
    props: list[dict[str, Any]] = field(default_factory=list)
    # kit console verbosity (verbose/info/warning/error). Kit prints thousands
    # of lines at info, and viam-server records the module's stderr as
    # error-level logs, so default to warning.
    kit_log_level: str = "warning"
    # scene lighting (FINDINGS SCN-9 / W30). None = leave the stage's lights
    # alone. Shape: {"dome": {"intensity": 1000, "color": [1, 1, 1]},
    # "sphere_intensity": 30000}; Isaac creates a DomeLight prim and rescales
    # /World/SphereLight, the mock records the config for tests.
    lighting: dict[str, Any] | None = None


# FINDINGS W30 lighting defaults: the DomeLight the module adds, and the prim
# paths in default_environment.usd it adjusts.
# a camera frequency must divide the render rate (IS-3); slack for float error
FREQUENCY_DIVISOR_TOLERANCE = 1e-6
DEFAULT_DOME_INTENSITY = 1000.0
DEFAULT_DOME_COLOR = (1.0, 1.0, 1.0)
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
SPHERE_LIGHT_PRIM_PATH = "/World/SphereLight"


def _as_quat(values: Sequence[float]) -> Quat:
    """Four numbers -> a (w, x, y, z) quaternion tuple (validates arity)."""
    w, x, y, z = (float(v) for v in values)
    return (w, x, y, z)


# create_camera attrs contract defaults (camera_base.py module docstring,
# FINDINGS W18/W19).
DEFAULT_CAMERA_WIDTH = 848
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FOV_DEG = 90.5
DEFAULT_CLIP_NEAR_M = 0.05
DEFAULT_CLIP_FAR_M = 10.0


def _camera_prim_path(name: str, attrs: dict[str, Any]) -> str:
    """Prim path for a to-be-created camera: parented under ``parent_prim``,
    else an explicit ``prim_path``, else ``/World/<name>``."""
    parent = attrs.get("parent_prim")
    if parent:
        return f"{parent.rstrip('/')}/{_prim_name(name)}"
    return attrs.get("prim_path") or f"/World/{_prim_name(name)}"


def _place_camera(cam: Any, attrs: dict[str, Any]) -> None:
    """Pose a just-initialized camera per the create_camera attrs contract
    (camera_base.py module docstring). ``parent_prim`` rides a (possibly
    moving) link; ``local_orientation_wxyz`` - derived from the Viam frame -
    is the source of truth (CAM-10) and is applied in ROS-optical axes so the
    camera's +Z is the frame's forward axis. Absent that, the legacy
    ``local_orientation_rpy_deg`` pose (usd axes, 180 deg about X to flip the
    usd camera's -Z forward) still applies. Free-standing ``target``/
    ``orientation_wxyz`` cameras are unchanged."""
    parent = attrs.get("parent_prim")
    if parent:
        local_position = list(to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.05)))
        if attrs.get("local_orientation_wxyz") is not None:
            quat = list(_as_quat(attrs["local_orientation_wxyz"]))
            cam.set_local_pose(local_position, quat, camera_axes="ros")
        else:
            r, p, y = to_vec3(attrs.get("local_orientation_rpy_deg"), default=(180.0, 0.0, 0.0))
            quat = list(quat_from_euler_deg(r, p, y))
            cam.set_local_pose(local_position, quat, camera_axes="usd")
    elif attrs.get("target") is not None:
        # aim at a target point (world axes: +X forward, +Z up)
        position = to_vec3(attrs.get("position"), default=(3.0, 3.0, 2.5))
        world_quat = look_at_quat(position, to_vec3(attrs.get("target")))
        cam.set_world_pose(list(position), list(world_quat), camera_axes="world")
    elif attrs.get("orientation_wxyz") is not None:
        position = to_vec3(attrs.get("position"))
        world_quat = _as_quat(attrs["orientation_wxyz"])
        cam.set_world_pose(list(position), list(world_quat), camera_axes="world")


def _configure_camera_optics(
    cam: Any, attrs: dict[str, Any], rendering_dt: float = 1.0 / 60.0
) -> None:
    """Focal length from ``fov_deg`` (CAM-4's aperture cancels usd's unit
    convention so this is unit-safe on both Isaac versions), a matching
    vertical aperture so pixels stay square (CAM-4), the clipping range
    (CAM-3 - OpenUSD's unauthored default is a 1 m near clip), the depth
    annotator (CAM-1) and an optional capture rate."""
    width, height = cam.get_resolution()

    # newly created cameras default to a 90.5 degree horizontal FOV; a
    # camera bound to an existing prim (explicit prim_path) keeps that
    # prim's authored FOV unless fov_deg overrides it.
    if not attrs.get("prim_path") or attrs.get("fov_deg"):
        fov = float(attrs.get("fov_deg", DEFAULT_CAMERA_FOV_DEG))
        horizontal_aperture = cam.get_horizontal_aperture()
        cam.set_focal_length(horizontal_aperture / (2.0 * math.tan(math.radians(fov) / 2.0)))

    horizontal_aperture = cam.get_horizontal_aperture()
    cam.set_vertical_aperture(horizontal_aperture * height / width)

    clip_near = float(attrs.get("clip_near", DEFAULT_CLIP_NEAR_M))
    clip_far = float(attrs.get("clip_far", DEFAULT_CLIP_FAR_M))
    cam.set_clipping_range(clip_near, clip_far)
    LOGGER.info(
        "camera %s clipping range %s",
        getattr(cam, "name", "<camera>"),
        cam.get_clipping_range(),
    )

    if attrs.get("depth"):
        cam.add_distance_to_image_plane_to_frame()

    frequency = attrs.get("frequency")
    if frequency:
        frequency = float(frequency)
        ticks_per_capture = (1.0 / rendering_dt) / frequency
        if abs(ticks_per_capture - round(ticks_per_capture)) > FREQUENCY_DIVISOR_TOLERANCE:
            LOGGER.warning(
                "camera %s frequency %s Hz is not an integer divisor of the "
                "render rate (%.4f Hz); the effective capture rate will differ",
                getattr(cam, "name", "<camera>"),
                frequency,
                1.0 / rendering_dt,
            )
        cam.set_frequency(frequency)


def spawn_orientation(attrs: dict[str, Any], meta: dict[str, Any]) -> Quat:
    """The (w,x,y,z) quaternion to spawn an arm's articulation with: the
    configured frame/orientation composed with the known asset's
    base_frame_correction (frame first), if any (FINDINGS XC-1/ARM-10)."""
    q_frame: Quat = (
        _as_quat(attrs["orientation_wxyz"])
        if attrs.get("orientation_wxyz") is not None
        else (1.0, 0.0, 0.0, 0.0)
    )
    correction = meta.get("base_frame_correction")
    if correction is not None:
        return quat_mul(q_frame, _as_quat(correction))
    return q_frame


def pose_in_frame(base_pos: Vec3, base_quat: Quat, pos: Vec3, quat: Quat) -> tuple[Vec3, Quat]:
    """Express a world pose (pos, quat) in the frame defined by
    (base_pos, base_quat) - both (w,x,y,z)."""
    base_quat_conj = quat_conj(base_quat)
    relative_position = quat_rotate(
        base_quat_conj,
        (pos[0] - base_pos[0], pos[1] - base_pos[1], pos[2] - base_pos[2]),
    )
    relative_orientation = quat_mul(base_quat_conj, quat)
    return relative_position, relative_orientation


def compose_pose(
    parent_pos: Vec3, parent_quat: Quat, local_pos: Vec3, local_quat: Quat
) -> tuple[Vec3, Quat]:
    """Inverse of pose_in_frame: express a pose (local_pos, local_quat) given
    in the frame (parent_pos, parent_quat) back in the parent's frame."""
    world_position = (
        parent_pos[0] + quat_rotate(parent_quat, local_pos)[0],
        parent_pos[1] + quat_rotate(parent_quat, local_pos)[1],
        parent_pos[2] + quat_rotate(parent_quat, local_pos)[2],
    )
    world_orientation = quat_mul(parent_quat, local_quat)
    return world_position, world_orientation


def viam_base_frame(root_pos: Vec3, root_quat: Quat, correction: Quat) -> tuple[Vec3, Quat]:
    """Recover Viam's arm frame from the Isaac articulation root's world
    pose: spawn composed root = frame * correction (FINDINGS ARM-10/XC-1),
    so frame = root * correction^-1."""
    return root_pos, quat_mul(root_quat, quat_conj(correction))


def anchor_fixed_joint_frame(
    spawn_pos: Vec3, spawn_quat: Quat, authored_pos: Vec3, authored_quat: Quat
) -> tuple[Vec3, Quat]:
    """Re-express a world-anchored fixed-base joint frame (authored_pos,
    authored_quat) so it matches an articulation spawned at (spawn_pos,
    spawn_quat). The UR assets' base FixedJoint has an empty body0 (= world
    frame) with localPos0/localRot0 authored in world coordinates, so PhysX
    resyncs the root xform to that joint frame on world.reset() and undoes
    any spawn pose passed to SingleArticulation (FINDINGS ARM-9/XC-1)."""
    return (
        (
            spawn_pos[0] + quat_rotate(spawn_quat, authored_pos)[0],
            spawn_pos[1] + quat_rotate(spawn_quat, authored_pos)[1],
            spawn_pos[2] + quat_rotate(spawn_quat, authored_pos)[2],
        ),
        quat_mul(spawn_quat, authored_quat),
    )


class SimManager:
    """Owns the sim thread. Get the process-wide instance via SimManager.get()."""

    _instance: Optional["SimManager"] = None
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
        self._boot_error: BaseException | None = None
        self._stop = threading.Event()
        self._sim_thread_id: int | None = None

        self.cfg: SimConfig | None = None
        self.mock = False
        # Isaac objects are created at boot; typed Any because the isaacsim
        # modules are not importable (or type-checkable) outside Isaac Sim.
        self._sim_app: Any = None
        self.world: Any = None
        self._isaac: Any = None  # lazily-populated namespace of isaac imports
        self._step_callbacks: dict[str, Callable[[float], None]] = {}
        # scene lighting config from the world component (FINDINGS SCN-9/W30);
        # stored so status() and tests can read it even in mock mode.
        self.lighting: dict[str, Any] | None = None
        # hooks fired (in registration order) after every world reset -
        # XC-5, so component handles can re-anchor state that resets undo.
        self._post_reset_hooks: list[Callable[[], None]] = []
        self._post_reset_lock = threading.Lock()
        # component name -> (spawn attrs, handle). viam-server rebuilds
        # resources on config change, but prims can't be re-spawned without
        # restarting kit, so handles are cached per component name.
        self._handles: dict[str, tuple[dict[str, Any], Any]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ensure_booted(self, cfg: SimConfig) -> None:
        """Called by the world component's reconfigure. Boots the sim on the
        sim thread the first time; subsequent calls with a different config
        log that a module restart is required (Kit can't be re-created)."""
        if self._booted.is_set():
            if self.cfg != cfg:
                LOGGER.warning(
                    "isaac sim is already running; changes to world config "
                    "(stage/headless/etc) require restarting the module"
                )
            return
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot previously: {self._boot_error}")

        self.cfg = cfg
        self._boot_requested.set()
        if not self._booted.wait(timeout=cfg.boot_timeout):
            raise SimTimeoutError(f"isaac sim did not boot within {cfg.boot_timeout}s")
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot: {self._boot_error}")

    def request_stop(self) -> None:
        self._stop.set()

    def register_post_reset(self, fn: Callable[[], None]) -> None:
        """Register a hook that fires (in registration order) after every
        world reset - boot, a component spawn, or an explicit reset command -
        on whichever thread performs the reset (the sim thread in practice)."""
        with self._post_reset_lock:
            self._post_reset_hooks.append(fn)

    def _reset_world(self) -> None:
        """The single chokepoint for resetting the isaac world: resets it
        (skipped in mock mode) then runs every registered post-reset hook,
        isolating each hook's failures so one can't block the rest."""
        if not self.mock:
            self.world.reset()
        with self._post_reset_lock:
            hooks = list(self._post_reset_hooks)
        for hook in hooks:
            try:
                hook()
            except Exception:
                LOGGER.exception("post-reset hook failed")

    def main_loop(self) -> None:
        """Run forever on the owning (main) thread: wait for a boot request,
        boot, then step the sim while draining queued tasks."""
        self._sim_thread_id = threading.get_ident()

        while not self._stop.is_set() and not self._boot_requested.wait(timeout=0.1):
            self._drain_tasks()

        if self._stop.is_set():
            return

        try:
            self._boot()
        except BaseException as e:  # SimulationApp failures can be SystemExit
            LOGGER.exception("failed to boot isaac sim")
            self._boot_error = e
            self._booted.set()
            return
        self._booted.set()

        last = time.monotonic()
        while not self._stop.is_set():
            self._drain_tasks()
            now = time.monotonic()
            dt = now - last
            last = now
            if self.mock:
                for cb in list(self._step_callbacks.values()):
                    cb(dt)
                time.sleep(0.01)
            else:
                self.world.step(render=True)

        if self._sim_app is not None:
            try:
                self._sim_app.close()
            except Exception:
                LOGGER.exception("error closing isaac sim")

    def _drain_tasks(self) -> None:
        while True:
            try:
                fn, fut = self._tasks.get_nowait()
            except queue.Empty:
                return
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as e:
                    fut.set_exception(e)

    def run(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Run fn on the sim thread and return its result."""
        if threading.get_ident() == self._sim_thread_id:
            return fn()
        fut: Future = Future()
        self._tasks.put((fn, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise SimTimeoutError(f"sim-thread call timed out after {timeout}s") from exc

    # ------------------------------------------------------------------
    # boot
    # ------------------------------------------------------------------

    def _boot(self) -> None:
        cfg = self.cfg
        assert cfg is not None
        self.lighting = cfg.lighting
        if cfg.mock:
            LOGGER.info("booting in MOCK mode - no isaac sim")
            self.mock = True
            self._reset_world()
            return

        LOGGER.info("booting isaac sim (headless=%s)...", cfg.headless)
        try:
            from isaacsim import SimulationApp  # isaac sim >= 4.5
        except ImportError:
            from omni.isaac.kit import SimulationApp  # older releases

        # quiet kit's console stream; unknown argv entries are forwarded to kit
        level = cfg.kit_log_level.capitalize()
        sys.argv.append(f"--/log/outputStreamLevel={level}")

        self._sim_app = SimulationApp({"headless": cfg.headless})

        try:
            import carb.settings

            carb.settings.get_settings().set("/log/outputStreamLevel", level)
        except Exception:
            pass

        if cfg.livestream and cfg.headless:
            try:
                try:
                    from isaacsim.core.utils.extensions import enable_extension
                except ImportError:
                    from omni.isaac.core.utils.extensions import enable_extension

                ip = cfg.livestream_public_ip or _local_ip()
                self._sim_app.set_setting("/app/livestream/enabled", True)
                if ip:
                    self._sim_app.set_setting("/app/livestream/publicEndpointAddress", ip)
                enable_extension("omni.kit.livestream.webrtc")
                LOGGER.info(
                    "livestream enabled - connect the 'Isaac Sim WebRTC Streaming "
                    "Client' app to %s (TCP 49100 + UDP 47998 must be reachable)",
                    ip or "<this machine's IP>",
                )
            except Exception:
                LOGGER.exception("could not enable livestream; continuing without it")

        self._isaac = import_isaac()

        if cfg.usd_stage:
            LOGGER.info("opening stage %s", cfg.usd_stage)
            self._isaac.open_stage(cfg.usd_stage)

        self.world = self._isaac.World(
            physics_dt=cfg.physics_dt,
            rendering_dt=cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        if not cfg.usd_stage:
            self.world.scene.add_default_ground_plane()
        for prop in cfg.props:
            try:
                self._spawn_prop(prop)
            except Exception:
                LOGGER.exception("failed to spawn prop %s", prop.get("name"))
        if cfg.lighting is not None:
            self._apply_lighting(cfg.lighting)
        self._reset_world()
        LOGGER.info("isaac sim world ready")

    def _apply_lighting(self, lighting: dict[str, Any]) -> None:
        """Configure scene lights per FINDINGS SCN-9/W30. Best-effort: never
        raises, so bad/unavailable lighting config can't block boot."""
        try:
            import omni.usd
            from pxr import Gf, UsdLux

            stage = omni.usd.get_context().get_stage()

            dome = lighting.get("dome")
            if dome is not None:
                dome_light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
                dome_light.CreateIntensityAttr(float(dome.get("intensity", DEFAULT_DOME_INTENSITY)))
                color = dome.get("color", DEFAULT_DOME_COLOR)
                dome_light.CreateColorAttr(Gf.Vec3f(*[float(v) for v in color]))

            sphere_intensity = lighting.get("sphere_intensity")
            if sphere_intensity is not None:
                sphere_prim = stage.GetPrimAtPath(SPHERE_LIGHT_PRIM_PATH)
                if sphere_prim.IsValid():
                    UsdLux.SphereLight(sphere_prim).GetIntensityAttr().Set(float(sphere_intensity))
        except Exception:
            LOGGER.exception("failed to apply scene lighting")

    def _spawn_prop(self, prop: dict[str, Any]) -> None:
        """Add a configured prop to the scene (runs on the sim thread,
        before the initial world.reset)."""
        import numpy as np

        from .spatial import to_vec3

        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        prim_path = f"/World/{name}"
        position = list(to_vec3(prop.get("position")))
        kind = str(prop.get("type", "cube"))

        if kind == "usd":
            usd_path = prop.get("usd_path")
            if not usd_path:
                raise ValueError(f"prop {name}: type 'usd' needs usd_path")
            if self._usd_exists(usd_path) is False:
                raise ValueError(f"prop {name}: usd not found: {usd_path}")
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(position=position)
            return

        if kind != "cube":
            raise ValueError(f"prop {name}: unknown type {kind!r} (cube or usd)")

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            size=float(prop.get("size", 0.05)),
        )
        if prop.get("scale") is not None:
            kwargs["scale"] = np.array([float(v) for v in prop["scale"]])
        if prop.get("color") is not None:
            kwargs["color"] = np.array([float(v) for v in prop["color"]])
        cls = self._isaac.FixedCuboid if prop.get("fixed") else self._isaac.DynamicCuboid
        self.world.scene.add(cls(**kwargs))

    def _require_booted(self) -> None:
        if not self._booted.is_set():
            raise SimNotBootedError(
                "isaac sim world is not running - configure a "
                f"{NAMESPACE}:{FAMILY}:world component and depend on it"
            )
        if self._boot_error is not None:
            raise SimNotBootedError(f"isaac sim failed to boot: {self._boot_error}")

    # ------------------------------------------------------------------
    # world controls (used by the world component's DoCommand)
    # ------------------------------------------------------------------

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
            # OQ-14 / GPU checklist item 6: None in mock or when no probe answers
            "isaac_version": _version_string(isaac_version()),
        }
        if self._booted.is_set() and not self.mock:
            out["playing"] = self.run(lambda: bool(self.world.is_playing()))
            out["sim_time"] = self.run(lambda: float(self.world.current_time))
        return out

    def add_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._require_booted()
        if self.mock:
            return

        def _add():
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            prim = self._isaac.SingleXFormPrim(prim_path)
            prim.set_world_pose(position=list(position))

        self.run(_add, timeout=60.0)

    # ------------------------------------------------------------------
    # component factories
    # ------------------------------------------------------------------

    # attributes that only affect the viam-side model, not the spawned prim
    _RUNTIME_KEYS = frozenset({"world", "move_timeout_sec", "max_linear_mps", "max_angular_rps"})

    def _cached_handle(
        self, kind: str, name: str, attrs: dict[str, Any], factory: Callable[[], Any]
    ) -> Any:
        if name in self._handles:
            old_attrs, handle = self._handles[name]

            def strip(attrs):
                return {k: v for k, v in attrs.items() if k not in self._RUNTIME_KEYS}

            if strip(old_attrs) != strip(attrs):
                LOGGER.warning(
                    "%s %r: spawn config changed but the prim is already in the "
                    "stage; restart the module to apply",
                    kind,
                    name,
                )
            return handle
        handle = factory()
        self._handles[name] = (dict(attrs), handle)
        return handle

    def _usd_exists(self, path: str) -> bool | None:
        """True/False if we can check, None if omni.client is unavailable."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, _ = client.stat(path)
            return result == client.Result.OK
        except Exception:
            return None

    def _resolve_usd(self, attrs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Return (absolute usd path or None, known-asset metadata)."""
        meta: dict[str, Any] = {}
        usd = attrs.get("usd_path")
        asset = attrs.get("asset")
        if asset:
            if asset not in KNOWN_ASSETS:
                raise ValueError(
                    f"unknown asset {asset!r}; known: {sorted(KNOWN_ASSETS)} "
                    "(or set usd_path directly)"
                )
            meta = KNOWN_ASSETS[asset]
            if not usd:
                root = self._isaac.get_assets_root_path()
                if root is None:
                    raise RuntimeError("could not reach the isaac sim assets server")
                candidates = meta["usd"]
                for rel in candidates:
                    if self._usd_exists(root + rel) is not False:
                        usd = root + rel
                        break
                if usd is None:
                    raise ValueError(
                        f"asset {asset!r}: none of {candidates} exist under {root}; "
                        "the asset layout may have changed in this isaac release"
                    )
        # a USD reference to a missing file "succeeds" but leaves an empty
        # prim, which later fails with confusing physics-tensor errors -
        # catch it here instead
        if usd and self._usd_exists(usd) is False:
            raise ValueError(f"usd not found: {usd}")
        return usd, meta

    def create_arm(self, name: str, attrs: dict[str, Any]) -> "ArmHandle":
        self._require_booted()
        if self.mock:

            def factory():
                return MockArmHandle(name, attrs)
        else:

            def factory():
                return self.run(lambda: self._create_arm_isaac(name, attrs), timeout=120.0)

        return self._cached_handle("arm", name, attrs, factory)

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
        authored in world coordinates; PhysX re-syncs the root xform to that
        joint frame on world.reset(), silently undoing the spawn pose passed
        to SingleArticulation (FINDINGS ARM-9/XC-1). Never raises: a failure
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
                "no world-anchored fixed joint found under %s; spawn pose relies on the prim xform",
                prim_path,
            )
            return False
        except Exception:
            LOGGER.exception("failed to re-anchor fixed base joint under %s", prim_path)
            return False

    def _create_arm_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacArmHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        position = to_vec3(attrs.get("position"))
        orientation = spawn_orientation(attrs, meta)
        if usd:
            self._isaac.add_reference_to_stage(usd_path=usd, prim_path=prim_path)
            # Spawn pose goes into USD first (see _place_root_xform) and the
            # world-anchored base joint is moved with it; the wrapper below is
            # constructed WITHOUT a pose on purpose.
            self._place_root_xform(prim_path, position, orientation)
            self._anchor_fixed_base(prim_path, position, orientation)

        art = self._isaac.SingleArticulation(prim_path=prim_path, name=name)
        self.world.scene.add(art)
        self._reset_world()
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

        ee = None
        asset = attrs.get("asset")
        ee_path = attrs.get("end_effector_prim") or (
            f"{prim_path}/wrist_3_link"
            if isinstance(asset, str) and asset.startswith("ur")
            else None
        )
        if ee_path:
            ee = self._isaac.SingleXFormPrim(ee_path)
        correction = (
            _as_quat(meta["base_frame_correction"])
            if meta.get("base_frame_correction") is not None
            else (1.0, 0.0, 0.0, 0.0)
        )
        return IsaacArmHandle(
            self, art, ee, meta.get("joint_names"), base_correction=correction, prim_path=prim_path
        )

    def create_camera(self, name: str, attrs: dict[str, Any]) -> "CameraHandle":
        self._require_booted()
        # CAM-17: wired once per handle (not in the model) so both backends
        # drop their cache / re-arm acquisition after every world.reset().
        # Registered inside factory() - which _cached_handle only calls on
        # first construction - because viam-server re-runs reconfigure ->
        # create_camera on every config change and _cached_handle returns the
        # same handle each time; registering outside factory() would append
        # a duplicate hook per reconfigure. Dispatched dynamically (not a
        # bound-method reference captured now) so tests can monkeypatch
        # handle.post_reset after creation.
        if self.mock:

            def factory():
                handle = MockCameraHandle(name, attrs)
                self.register_post_reset(lambda: handle.post_reset())
                return handle
        else:

            def factory():
                handle = self.run(lambda: self._create_camera_isaac(name, attrs), timeout=120.0)
                self.register_post_reset(lambda: handle.post_reset())
                return handle

        return self._cached_handle("camera", name, attrs, factory)

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
            r, p, y = to_vec3(attrs.get("orientation_rpy_deg"))
            kwargs["orientation"] = list(quat_from_euler_deg(r, p, y))

        cam = self._isaac.Camera(**kwargs)
        # 4.5: get_resolution()/apertures only read back correctly once the
        # render product exists (IS-1), so initialize() must come first.
        cam.initialize()

        _place_camera(cam, attrs)
        rendering_dt = self.cfg.rendering_dt if self.cfg is not None else 1.0 / 60.0
        _configure_camera_optics(cam, attrs, rendering_dt)

        return IsaacCameraHandle(
            self,
            cam,
            depth_enabled=bool(attrs.get("depth")),
            image_format=attrs.get("image_format", "png"),
            frequency=attrs.get("frequency"),
        )

    def _require_prim(self, prim_path: str) -> None:
        """Raise a helpful error if prim_path doesn't exist in the stage."""
        get_prim = getattr(self._isaac, "get_prim_at_path", None)
        if get_prim is None:
            return
        prim = get_prim(prim_path)
        if prim is None or not prim.IsValid():
            parent_path = prim_path.rsplit("/", 1)[0] or "/"
            hint = ""
            parent = get_prim(parent_path)
            if parent is not None and parent.IsValid():
                children = [c.GetName() for c in parent.GetChildren()]
                hint = f"; children of {parent_path}: {children}"
            raise PrimNotFoundError(f"prim not found: {prim_path}{hint}")

    def create_base(self, name: str, attrs: dict[str, Any]) -> "BaseHandle":
        self._require_booted()
        if self.mock:

            def factory():
                return MockBaseHandle(name, attrs)
        else:

            def factory():
                return self.run(lambda: self._create_base_isaac(name, attrs), timeout=120.0)

        return self._cached_handle("base", name, attrs, factory)

    def _create_base_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacBaseHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        wheel_joints = attrs.get("wheel_joints") or meta.get("wheel_joints")
        if not wheel_joints or len(wheel_joints) != 2:
            raise ValueError(
                "base needs wheel_joints: [left_joint_name, right_joint_name] "
                "(known assets like 'jetbot' provide defaults)"
            )
        wheel_radius = float(attrs.get("wheel_radius", meta.get("wheel_radius", 0.05)))
        wheel_base = float(attrs.get("wheel_base", meta.get("wheel_base", 0.3)))
        position = to_vec3(attrs.get("position"))

        base_kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=list(wheel_joints),
            create_robot=usd is not None,
            usd_path=usd,
            position=list(position),
        )
        if attrs.get("orientation_wxyz") is not None:
            base_kwargs["orientation"] = [float(v) for v in attrs["orientation_wxyz"]]
        robot = self._isaac.WheeledRobot(**base_kwargs)
        self.world.scene.add(robot)
        self._reset_world()

        controller = self._isaac.DifferentialController(
            name=f"{name}_controller",
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
        )
        handle = IsaacBaseHandle(self, robot, controller, wheel_radius, wheel_base)
        self.world.add_physics_callback(f"{name}_drive", handle._on_physics_step)
        return handle


def _version_string(version: tuple[int, int, int] | None) -> str | None:
    return None if version is None else ".".join(str(part) for part in version)


def _local_ip() -> str:
    """Best-effort primary local IP (no traffic is actually sent)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


def _prim_name(name: str) -> str:
    """Component names may contain characters USD prim names can't."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


# ======================================================================
# Handles - the interface component models talk to. All public methods are
# safe to call from any thread.
# ======================================================================


def resolve_joint_indices(
    dof_names: Sequence[str], joint_names: Sequence[str] | None
) -> list[int] | None:
    """Map an asset's declared arm joint names onto the articulation's PhysX
    dof order, by name rather than position (FINDINGS ARM-1; R-3: attaching a
    gripper later can add/reorder dofs, so a positional slice would silently
    pick up the wrong joints). Returns None when the asset declares no joint
    names, meaning "all dofs, in PhysX order"."""
    if joint_names is None:
        return None
    indices = []
    missing = []
    for name in joint_names:
        try:
            indices.append(dof_names.index(name))
        except ValueError:
            missing.append(name)
    if missing:
        raise ValueError(
            f"joint(s) not found in articulation: {missing}; actual dof_names: {list(dof_names)}"
        )
    return indices


class ArmHandle:
    def dof_names(self) -> list[str]:
        """Names of the arm's named joints, in the asset's declared order
        (all DOFs, in PhysX order, when the asset declares none)."""
        raise NotImplementedError

    def get_joint_positions(self) -> list[float]:  # radians
        """Positions of the arm's named joints, in the asset's declared
        order (all DOFs when the asset declares none)."""
        raise NotImplementedError

    def set_joint_targets(self, positions: list[float]) -> None:
        """Targets for the arm's named joints, in the asset's declared
        order (all DOFs when the asset declares none)."""
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_end_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """((x,y,z) meters, (w,x,y,z) quaternion) of the end effector, in
        Viam's arm frame - the Isaac root un-rotated by the asset's
        base_frame_correction, if any (FINDINGS ARM-10)."""
        raise NotImplementedError

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        """((x,y,z) meters, (w,x,y,z) quaternion) world pose of an arbitrary
        prim on the stage (FINDINGS XC-1 GPU acceptance)."""
        raise NotImplementedError


class IsaacArmHandle(ArmHandle):
    def __init__(
        self,
        sim: SimManager,
        articulation,
        ee_prim,
        joint_names: Sequence[str] | None = None,
        base_correction: Quat = (1.0, 0.0, 0.0, 0.0),
        prim_path: str = "",
    ) -> None:
        self._sim = sim
        self._art = articulation
        self._ee = ee_prim
        self._joint_names = joint_names
        self._base_correction: Quat = base_correction
        self._prim_path = prim_path
        self._dof_names: list[str] = list(articulation.dof_names)
        LOGGER.info(
            "arm %r articulation dof_names: %s",
            getattr(articulation, "name", ""),
            self._dof_names,
        )
        self._joint_indices: list[int] | None = resolve_joint_indices(self._dof_names, joint_names)

    def dof_names(self) -> list[str]:
        if self._joint_indices is None:
            return list(self._dof_names)
        return [self._dof_names[i] for i in self._joint_indices]

    def get_joint_positions(self) -> list[float]:
        def _get():
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            return [float(v) for v in positions]

        return self._sim.run(_get)

    def set_joint_targets(self, positions: list[float]) -> None:
        import numpy as np

        def _apply():
            action = self._sim._isaac.ArticulationAction(
                joint_positions=np.array(positions, dtype=float),
                joint_indices=self._joint_indices,
            )
            self._art.apply_action(action)

        self._sim.run(_apply)

    def is_moving(self) -> bool:
        def _check():
            vels = self._art.get_joint_velocities(joint_indices=self._joint_indices)
            if vels is None:
                return False
            return bool(max(abs(float(v)) for v in vels) > 1e-3)

        return self._sim.run(_check)

    def stop(self) -> None:
        # hold the current position
        current = self.get_joint_positions()
        self.set_joint_targets(current)

    def get_end_pose(self):
        if self._ee is None:
            raise NotImplementedError(
                "set end_effector_prim in the arm config to report end position"
            )

        def _pose():
            root_pos, root_quat = self._art.get_world_pose()
            pos, quat = self._ee.get_world_pose()
            root_pos_t = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
            root_quat_t = (
                float(root_quat[0]),
                float(root_quat[1]),
                float(root_quat[2]),
                float(root_quat[3]),
            )
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            base_pos, base_quat = viam_base_frame(root_pos_t, root_quat_t, self._base_correction)
            return pose_in_frame(base_pos, base_quat, pos_t, quat_t)

        return self._sim.run(_pose)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        def _pose() -> tuple[Vec3, Quat]:
            self._sim._require_prim(prim_path)
            pos, quat = self._sim._isaac.SingleXFormPrim(prim_path).get_world_pose()
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            return pos_t, quat_t

        return self._sim.run(_pose)


class MockArmHandle(ArmHandle):
    """Joints move linearly toward their targets at a fixed speed. Total dof
    count is mock_dof (default: the number of declared joint names, else 6);
    the arm's named joints are selected by index the same way the Isaac
    handle does (FINDINGS ARM-1; R-3), and any remaining dofs are padding
    that never moves."""

    SPEED = 1.0  # rad/s per joint

    # the mock's end effector, fixed in Viam's arm frame (public,
    # deterministic value; unchanged by spawn pose or base_frame_correction).
    FIXED_LOCAL_EE: tuple[Vec3, Quat] = ((0.3, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))

    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        from .spatial import to_vec3

        self.name = name
        meta = KNOWN_ASSETS.get(str(attrs.get("asset", "")), {})
        joint_names: Sequence[str] | None = meta.get("joint_names")
        default_dof = len(joint_names) if joint_names else 6
        dof = int(attrs.get("mock_dof", default_dof))
        if joint_names:
            names = list(joint_names) + [f"mock_extra_{i}" for i in range(dof - len(joint_names))]
            self._joint_indices: list[int] | None = list(range(len(joint_names)))
        else:
            names = [f"mock_joint_{i}" for i in range(dof)]
            self._joint_indices = None
        self._dof_names = names
        self._lock = threading.Lock()
        self._start = [0.0] * dof
        self._target = [0.0] * dof
        self._t0 = time.monotonic()
        self.spawn_position = to_vec3(attrs.get("position"))
        self.spawn_orientation = spawn_orientation(attrs, meta)
        correction = meta.get("base_frame_correction")
        self._base_correction: Quat = (
            _as_quat(correction)
            if correction is not None
            else (
                1.0,
                0.0,
                0.0,
                0.0,
            )
        )
        self._prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"

    def dof_names(self) -> list[str]:
        return list(self._dof_names)

    def _selected(self) -> list[int]:
        if self._joint_indices is None:
            return list(range(len(self._dof_names)))
        return self._joint_indices

    def _positions_at(self, now: float) -> list[float]:
        out = []
        dt = max(0.0, now - self._t0)
        for s, t in zip(self._start, self._target, strict=True):
            delta = t - s
            travel = self.SPEED * dt
            if abs(delta) <= travel:
                out.append(t)
            else:
                out.append(s + math.copysign(travel, delta))
        return out

    def get_all_joint_positions(self) -> list[float]:
        """Test-only accessor for the full (unselected) dof array."""
        with self._lock:
            return self._positions_at(time.monotonic())

    def get_joint_positions(self) -> list[float]:
        with self._lock:
            all_pos = self._positions_at(time.monotonic())
        return [all_pos[i] for i in self._selected()]

    def set_joint_targets(self, positions: list[float]) -> None:
        with self._lock:
            now = time.monotonic()
            all_pos = self._positions_at(now)
            selected = self._selected()
            if len(positions) != len(selected):
                raise ValueError(f"expected {len(selected)} joint positions, got {len(positions)}")
            self._start = all_pos
            self._target = list(all_pos)
            for i, p in zip(selected, positions, strict=True):
                self._target[i] = p
            self._t0 = now

    def is_moving(self) -> bool:
        with self._lock:
            pos = self._positions_at(time.monotonic())
        selected = self._selected()
        return any(abs(pos[i] - self._target[i]) > 1e-9 for i in selected)

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            self._target = list(self._start)
            self._t0 = now

    def _ee_world_pose(self) -> tuple[Vec3, Quat]:
        """The mock's simulated Isaac root is (spawn_position,
        spawn_orientation) - already composed with base_frame_correction -
        so its end effector's world pose is FIXED_LOCAL_EE expressed in
        Viam's arm frame, then re-composed onto that rotated root."""
        base_pos, base_quat = viam_base_frame(
            self.spawn_position, self.spawn_orientation, self._base_correction
        )
        local_pos, local_quat = self.FIXED_LOCAL_EE
        return compose_pose(base_pos, base_quat, local_pos, local_quat)

    def get_end_pose(self):
        # a fixed, deterministic pose for testing, defined in Viam's arm
        # frame (FINDINGS ARM-10) - it must not change with spawn_position/
        # spawn_orientation/base_frame_correction.
        base_pos, base_quat = viam_base_frame(
            self.spawn_position, self.spawn_orientation, self._base_correction
        )
        ee_pos, ee_quat = self._ee_world_pose()
        return pose_in_frame(base_pos, base_quat, ee_pos, ee_quat)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        ee_prim_path = f"{self._prim_path}/wrist_3_link"
        if prim_path != ee_prim_path:
            raise PrimNotFoundError(f"prim not found: {prim_path}")
        return self._ee_world_pose()


# CAM-2: bounded retry on the caller's thread while the renderer warms up
# after create/reset, sleeping between attempts so the sim thread gets to run.
WARMUP_RETRIES = 30
WARMUP_SLEEP_S = 1.0 / 60.0
WARMUP_MESSAGE = "no frame available yet - is the simulation playing?"


class IsaacCameraHandle(CameraHandle):
    def __init__(
        self,
        sim: SimManager,
        cam: Any,
        *,
        depth_enabled: bool,
        image_format: str,
        frequency: float | None,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sim = sim
        self._cam = cam
        self.depth_enabled = depth_enabled
        self.image_format = image_format
        self.frequency = frequency
        self._now = now or (lambda: float(sim.world.current_time))
        self._sleep = sleep
        self._cached_frame: Frame | None = None

    def _grab(self) -> Frame:
        # runs on the sim thread (via sim.run): one rgb(+depth) read per sim
        # step, cached by sim_time so GetImages + GetPointCloud in the same
        # tick share one grab (CAM-9).
        sim_time = self._now()
        cached = self._cached_frame
        if cached is not None and cached.sim_time == sim_time:
            return cached

        rgba = self._cam.get_rgba()
        if rgba is None or rgba.size == 0:
            raise NoFrameYetError(WARMUP_MESSAGE)
        rgb = rgba[:, :, :3].copy()

        depth = None
        if self.depth_enabled:
            raw_depth = self._cam.get_depth()
            if raw_depth is None:
                raise NoFrameYetError(WARMUP_MESSAGE)
            depth = np.asarray(raw_depth)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth = depth.astype(np.float32)

        frame = Frame(rgb=rgb, depth=depth, sim_time=sim_time)
        self._cached_frame = frame
        return frame

    def get_frame(self) -> Frame:
        last_error: NoFrameYetError | None = None
        for _ in range(WARMUP_RETRIES):
            try:
                return self._sim.run(self._grab)
            except NoFrameYetError as exc:
                last_error = exc
                self._sleep(WARMUP_SLEEP_S)
        raise NoFrameYetError(WARMUP_MESSAGE) from last_error

    def get_intrinsics(self) -> Intrinsics:
        def _read() -> Intrinsics:
            focal_length = self._cam.get_focal_length()
            horizontal_aperture = self._cam.get_horizontal_aperture()
            vertical_aperture = self._cam.get_vertical_aperture()
            width, height = self._cam.get_resolution()
            if not focal_length or not horizontal_aperture or not vertical_aperture:
                raise RuntimeError(
                    "camera intrinsics unavailable: focal length or aperture is 0 "
                    "(has the camera been initialized?)"
                )
            return Intrinsics(
                fx=width * focal_length / horizontal_aperture,
                fy=height * focal_length / vertical_aperture,
                cx=width / 2,
                cy=height / 2,
                width=width,
                height=height,
            )

        return self._sim.run(_read)

    def post_reset(self) -> None:
        def _reset() -> None:
            self._cached_frame = None
            try:
                post_reset = getattr(self._cam, "post_reset", None)
                if post_reset is not None:
                    post_reset()
                else:
                    self._cam.initialize()
            except Exception:
                LOGGER.exception("camera post-reset failed")

        self._sim.run(_reset)


class BaseHandle:
    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError


class IsaacBaseHandle(BaseHandle):
    def __init__(
        self, sim: SimManager, robot, controller, wheel_radius: float, wheel_base: float
    ) -> None:
        self._sim = sim
        self._robot = robot
        self._controller = controller
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def _on_physics_step(self, step_size: float) -> None:
        # runs on the sim thread every physics step
        with self._lock:
            lin, ang = self._cmd
        try:
            self._robot.apply_wheel_actions(self._controller.forward(command=[lin, ang]))
        except Exception:
            LOGGER.exception("error driving base")

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)


class MockBaseHandle(BaseHandle):
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self.wheel_radius = float(attrs.get("wheel_radius", 0.05))
        self.wheel_base = float(attrs.get("wheel_base", 0.3))
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)
