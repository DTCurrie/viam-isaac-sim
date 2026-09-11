from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from viam.logging import getLogger

from ..encoding import Intrinsics, intrinsics_from_fov
from ..prim_paths import prim_name
from ..spatial import _as_quat, look_at_quat, quat_from_euler_deg, to_vec3

if TYPE_CHECKING:
    from ..sim_manager import SimManager

LOGGER = getLogger(__name__)

# create_camera attrs contract defaults (CameraHandle class docstring).
DEFAULT_CAMERA_FOV_DEG = 90.5
DEFAULT_CLIP_NEAR_M = 0.05
DEFAULT_CLIP_FAR_M = 10.0


class Frame(NamedTuple):
    """One synchronized grab. ``depth`` is None when depth is not enabled."""

    rgb: NDArray[np.uint8]  # (H, W, 3) uint8
    depth: NDArray[np.float32] | None  # (H, W) float32 meters. Non-finite means no hit
    sim_time: float  # world.current_time (or the mock's clock) at grab


class NoFrameYetError(RuntimeError):
    """No rendered frame is available, such as during warm-up right after
    create or reset. The model maps this to FAILED_PRECONDITION after a
    bounded retry."""


class CameraHandle:
    """Backend-neutral camera contract shared by the Isaac and mock backends.
    Every public method is safe to call from any thread.

    Attribute contract for ``SimManager.create_camera(name, attrs)``. The model
    (``models/camera.py``) and ``models/component_frame_pose.apply_frame_to_attrs``
    produce these keys, and both backends consume them.

      width (int, 848) · height (int, 480) · fov_deg (float, 90.5, horizontal)
      depth (bool, False)           - attach the depth annotator (wrist cam: true)
      clip_near / clip_far (m)      - 0.05 / 10.0
      image_format ("png"|"jpeg")   - color encoding for GetImages, default "png"
      frequency (float | None)      - capture rate. None means every rendered frame
      prim_path / position / target / orientation_wxyz / orientation_rpy_deg
                                    - free-standing cameras (unchanged)
      parent_prim (str)             - ride a link. The Viam ``frame`` is then the
                                      single source of truth for the mount:
      local_position ([x,y,z] m)    - from frame.translation (mm → m)
      local_orientation_wxyz        - from frame.orientation. Applied with
                                      ``set_local_pose(..., camera_axes="ros")``
                                      so the camera optical axis is the frame's +Z
    """

    depth_enabled: bool = False
    image_format: str = "png"
    frequency: float | None = None

    def get_frame(self) -> Frame:
        """rgb (+ depth) grabbed once per sim step and cached by sim_time, so
        GetImages and GetPointCloud in one tick share one grab. Raises
        NoFrameYetError while the renderer has not produced a frame yet."""
        raise NotImplementedError

    def get_rgb(self) -> NDArray[np.uint8]:
        return self.get_frame().rgb

    def get_depth(self) -> NDArray[np.float32]:
        """(H, W) float32 meters. Raises NoFrameYetError, or RuntimeError when
        depth is not enabled on this camera."""
        depth = self.get_frame().depth
        if depth is None:
            raise RuntimeError("depth is not enabled on this camera (set depth: true)")
        return depth

    def get_intrinsics(self) -> Intrinsics:
        """Pinhole intrinsics, never zero-filled."""
        raise NotImplementedError

    def post_reset(self) -> None:
        """Called by SimManager after every world.reset(). Backends drop
        cached frames and re-arm acquisition here."""
        return None

    def release(self) -> None:
        """Called by SimManager.release_handle when the owning component
        closes. Backends drop annotators and render products here. The
        camera prim itself stays in the stage."""
        return None


def _camera_prim_path(name: str, attrs: dict[str, Any]) -> str:
    """Prim path for a to-be-created camera: parented under ``parent_prim``,
    else an explicit ``prim_path``, else ``/World/<name>``."""
    parent = attrs.get("parent_prim")
    if parent:
        return f"{parent.rstrip('/')}/{prim_name(name)}"
    return attrs.get("prim_path") or f"/World/{prim_name(name)}"


def _place_camera(cam: Any, attrs: dict[str, Any]) -> None:
    """Pose a just-initialized camera per the create_camera attrs contract
    (CameraHandle class docstring). ``parent_prim`` rides a (possibly
    moving) link. ``local_orientation_wxyz`` - derived from the Viam frame -
    is the source of truth and is applied in ROS-optical axes so the
    camera's +Z is the frame's forward axis. Absent that, the legacy
    ``local_orientation_rpy_deg`` pose (usd axes, 180 deg about X to flip the
    usd camera's -Z forward) still applies. Free-standing ``orientation_wxyz``
    is world axes (+X forward) unless ``orientation_axes`` says "ros" - the
    camera model sets that when the quat came from a Viam frame, whose
    convention is ROS-optical (+Z forward), so a frame-configured fixed
    camera aims where the frame system believes it aims (measured: the side
    camera's world-axes read of a ROS quat put the backdrop at 7994 mm)."""
    parent = attrs.get("parent_prim")
    if parent:
        local_position = list(to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.05)))
        if attrs.get("local_orientation_wxyz") is not None:
            quat = list(_as_quat(attrs["local_orientation_wxyz"]))
            cam.set_local_pose(local_position, quat, camera_axes="ros")
        else:
            roll, pitch, yaw = to_vec3(
                attrs.get("local_orientation_rpy_deg"), default=(180.0, 0.0, 0.0)
            )
            quat = list(quat_from_euler_deg(roll, pitch, yaw))
            cam.set_local_pose(local_position, quat, camera_axes="usd")
    elif attrs.get("target") is not None:
        # aim at a target point (world axes: +X forward, +Z up)
        position = to_vec3(attrs.get("position"), default=(3.0, 3.0, 2.5))
        world_quat = look_at_quat(position, to_vec3(attrs.get("target")))
        cam.set_world_pose(list(position), list(world_quat), camera_axes="world")
    elif attrs.get("orientation_wxyz") is not None:
        position = to_vec3(attrs.get("position"))
        world_quat = _as_quat(attrs["orientation_wxyz"])
        camera_axes = str(attrs.get("orientation_axes", "world"))
        cam.set_world_pose(list(position), list(world_quat), camera_axes=camera_axes)


def _configure_camera_optics(cam: Any, attrs: dict[str, Any]) -> None:
    """Focal length from ``fov_deg`` (the aperture cancels usd's unit
    convention so this is unit-safe on both Isaac versions), a matching
    vertical aperture so pixels stay square, the clipping range
    (OpenUSD's unauthored default is a 1 m near clip), and the depth
    annotator."""
    width, height = cam.get_resolution()

    # newly created cameras default to a 90.5 degree horizontal FOV. A
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
        # tick share one grab.
        sim_time = self._now()
        cached = self._cached_frame
        if cached is not None and cached.sim_time == sim_time:
            return cached

        rgba = self._cam.get_rgba()
        if rgba is None or rgba.size == 0:
            raise NoFrameYetError(WARMUP_MESSAGE)
        # annotator_device="cuda" (5.0) returns a warp.array, not an
        # np.ndarray, and warp.array has no .copy() - convert at the
        # boundary before slicing.
        rgba = rgba.numpy() if not isinstance(rgba, np.ndarray) else rgba
        rgb = rgba[:, :, :3].copy()

        depth = None
        if self.depth_enabled:
            raw_depth = self._cam.get_depth()
            if raw_depth is None:
                raise NoFrameYetError(WARMUP_MESSAGE)
            raw_depth = raw_depth.numpy() if not isinstance(raw_depth, np.ndarray) else raw_depth
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
            except Exception:  # post-reset re-init, any failure is logged, not fatal
                LOGGER.exception("camera post-reset failed")

        self._sim.run(_reset)

    def release(self) -> None:
        """Camera.destroy() only exists on Isaac 5.0, guarded by getattr,
        never a version check. Older releases have nothing to release."""

        def _release() -> None:
            destroy = getattr(self._cam, "destroy", None)
            if destroy is not None:
                destroy()

        self._sim.run(_release)


DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FOV_DEG = 90.5

# Red block's column/row span relative to the principal point (cx, cy), in
# pixels. Deliberately off-center in both axes so a sign or axis bug in the
# back-projection changes the analytic centroid.
BLOCK_LEFT_OFFSET_PX = 40
BLOCK_RIGHT_OFFSET_PX = 140
BLOCK_TOP_OFFSET_PX = -20
BLOCK_BOTTOM_OFFSET_PX = 60

RED_BLOCK_RGB = (224, 32, 32)
RED_BLOCK_DEPTH_M = 0.40

FLOOR_FAR_M = 1.20  # depth at the top row
FLOOR_NEAR_M = 0.60  # depth at the bottom row
FLOOR_GRAY_TOP = 40  # rgb gray level at the top row
FLOOR_GRAY_BOTTOM = 200  # rgb gray level at the bottom row

NAN_BAND_FRACTION = 10  # top height // NAN_BAND_FRACTION rows are "no hit"
NAN_BAND_RGB = 20  # dark gray shown for the no-hit band


MIN_BLOCK_SPAN_PX = 2


def _side_block_pixel_bounds(
    k: Intrinsics, column_offset_px: int, size_mm: float, height_mm: float, depth_m: float
) -> tuple[int, int, int, int]:
    """Integer pixel bounds (u0, u1, v0, v1), u1/v1 exclusive, of a side-view block.

    Rises from the principal row ``cy`` (the support line), left edge at ``cx +
    column_offset_px``.
    """
    cx_i, cy_i = int(k.cx), int(k.cy)
    span_u = max(MIN_BLOCK_SPAN_PX, round(size_mm / 1000 * k.fx / depth_m))
    span_v = max(MIN_BLOCK_SPAN_PX, round(height_mm / 1000 * k.fy / depth_m))
    u0 = cx_i + column_offset_px
    v1 = cy_i
    v0 = v1 - span_v
    return u0, u0 + span_u, v0, v1


def _block_pixel_bounds(
    k: Intrinsics, block_size_mm: float | None = None
) -> tuple[int, int, int, int]:
    """Integer pixel bounds (u0, u1, v0, v1), u1/v1 exclusive, of the block.

    Unset ``block_size_mm`` reproduces today's fixed pixel-offset rectangle. Set,
    the block becomes a square anchored at the same top-left corner, sized from
    the metric ``block_size_mm`` at ``RED_BLOCK_DEPTH_M`` through the intrinsics.
    """
    cx_i, cy_i = int(k.cx), int(k.cy)
    u0 = cx_i + BLOCK_LEFT_OFFSET_PX
    v0 = cy_i + BLOCK_TOP_OFFSET_PX
    if block_size_mm is None:
        u1 = cx_i + BLOCK_RIGHT_OFFSET_PX
        v1 = cy_i + BLOCK_BOTTOM_OFFSET_PX
        return u0, u1, v0, v1

    size_m = block_size_mm / 1000
    span_u = max(MIN_BLOCK_SPAN_PX, round(size_m * k.fx / RED_BLOCK_DEPTH_M))
    span_v = max(MIN_BLOCK_SPAN_PX, round(size_m * k.fy / RED_BLOCK_DEPTH_M))
    return u0, u0 + span_u, v0, v0 + span_v


def _block_center_m(
    k: Intrinsics, block_size_mm: float | None = None
) -> tuple[float, float, float]:
    u0, u1, v0, v1 = _block_pixel_bounds(k, block_size_mm)
    u_mean = (u0 + u1 - 1) / 2
    v_mean = (v0 + v1 - 1) / 2
    z = RED_BLOCK_DEPTH_M
    x = (u_mean - k.cx) * z / k.fx
    y = (v_mean - k.cy) * z / k.fy
    return (x, y, z)


class MockCameraHandle(CameraHandle):
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self._width = int(attrs.get("width", DEFAULT_WIDTH))
        self._height = int(attrs.get("height", DEFAULT_HEIGHT))
        fov_deg = float(attrs.get("fov_deg", DEFAULT_FOV_DEG))
        self.depth_enabled = bool(attrs.get("depth", False))
        self.image_format = attrs.get("image_format", "png")
        self.frequency = attrs.get("frequency")
        self.reset_count = 0
        block_size_mm = attrs.get("block_size_mm")
        self.block_size_mm = float(block_size_mm) if block_size_mm is not None else None
        self.view = attrs.get("view", "top")
        self._blocks = [dict(block) for block in attrs.get("blocks", [])]

        self._k = intrinsics_from_fov(self._width, self._height, fov_deg)
        if self.view == "side":
            self._rgb, self._depth = self._build_side_scene()
        else:
            self._rgb, self._depth = self._build_scene()

    def _build_side_scene(self) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        width, height = self._width, self._height
        cy_i = int(self._k.cy)

        row_gray = np.linspace(FLOOR_GRAY_TOP, FLOOR_GRAY_BOTTOM, height)
        row_depth = np.linspace(FLOOR_FAR_M, FLOOR_NEAR_M, height, dtype=np.float32)

        rgb = np.repeat(row_gray[:, None, None], width, axis=1).astype(np.uint8)
        rgb = np.repeat(rgb, 3, axis=2)
        depth = np.repeat(row_depth[:, None], width, axis=1).astype(np.float32)

        # Above the support line, off-block, is NaN "no hit": a far backdrop
        # would otherwise read as a tall object.
        rgb[:cy_i, :, :] = NAN_BAND_RGB
        depth[:cy_i, :] = np.nan

        # Nearest-depth-last-wins: paint farther blocks first so a nearer
        # block occludes a farther one where their rectangles overlap.
        for block in sorted(self._blocks, key=lambda b: -float(b["depth_m"])):
            rgb_tuple = tuple(int(channel) for channel in block["rgb"])
            size_mm = float(block["size_mm"])
            height_mm = float(block["height_mm"])
            column_offset_px = int(block["column_offset_px"])
            depth_m = float(block["depth_m"])
            u0, u1, v0, v1 = _side_block_pixel_bounds(
                self._k, column_offset_px, size_mm, height_mm, depth_m
            )
            rgb[v0:v1, u0:u1, :] = rgb_tuple
            depth[v0:v1, u0:u1] = depth_m

        return rgb, depth

    def _build_scene(self) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        width, height = self._width, self._height

        row_gray = np.linspace(FLOOR_GRAY_TOP, FLOOR_GRAY_BOTTOM, height)
        row_depth = np.linspace(FLOOR_FAR_M, FLOOR_NEAR_M, height, dtype=np.float32)

        rgb = np.repeat(row_gray[:, None, None], width, axis=1).astype(np.uint8)
        rgb = np.repeat(rgb, 3, axis=2)
        depth = np.repeat(row_depth[:, None], width, axis=1).astype(np.float32)

        nan_band_rows = height // NAN_BAND_FRACTION
        rgb[:nan_band_rows, :, :] = NAN_BAND_RGB
        depth[:nan_band_rows, :] = np.nan

        u0, u1, v0, v1 = _block_pixel_bounds(self._k, self.block_size_mm)
        rgb[v0:v1, u0:u1, :] = RED_BLOCK_RGB
        depth[v0:v1, u0:u1] = RED_BLOCK_DEPTH_M

        return rgb, depth

    @property
    def red_block_center_m(self) -> tuple[float, float, float]:
        return _block_center_m(self._k, self.block_size_mm)

    @property
    def tallest_block_height_mm(self) -> float | None:
        if self.view != "side" or not self._blocks:
            return None
        return max(float(block["height_mm"]) for block in self._blocks)

    def get_frame(self) -> Frame:
        sim_time = math.floor(time.monotonic() * 60) / 60
        depth = self._depth if self.depth_enabled else None
        return Frame(rgb=self._rgb, depth=depth, sim_time=sim_time)

    def get_intrinsics(self) -> Intrinsics:
        return self._k

    def post_reset(self) -> None:
        self.reset_count += 1


# Camera-optical-frame center (x right, y down, z forward) of the mock's red
# block for the default 848x480 @ 90.5 deg configuration, meters. A model
# test asserts the PCD red-cluster centroid equals it within 1 mm.
MOCK_RED_BLOCK_CENTER_M: tuple[float, float, float] = MockCameraHandle(
    "_default", {}
).red_block_center_m
