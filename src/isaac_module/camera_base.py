"""Camera handle contract shared by the Isaac and mock backends (phase 2 seam).

Attribute contract for ``SimManager.create_camera(name, attrs)`` — the model
(``models/camera.py``) and ``models/utils.apply_frame_to_attrs`` produce these
keys, both backends consume them (FINDINGS W18–W21, CAM-10):

  width (int, 848) · height (int, 480) · fov_deg (float, 90.5, horizontal)
  depth (bool, False)           - attach the depth annotator (wrist cam: true)
  clip_near / clip_far (m)      - 0.05 / 10.0
  image_format ("png"|"jpeg")   - colour encoding for GetImages, default "png"
  frequency (float | None)      - capture rate; None = every rendered frame
  prim_path / position / target / orientation_wxyz / orientation_rpy_deg
                                - free-standing cameras (unchanged)
  parent_prim (str)             - ride a link; then the Viam ``frame`` is the
                                  single source of truth for the mount:
  local_position ([x,y,z] m)    - from frame.translation (mm → m)
  local_orientation_wxyz        - from frame.orientation; applied with
                                  ``set_local_pose(..., camera_axes="ros")``
                                  so the camera optical axis is the frame's +Z
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .encoding import Intrinsics


class Frame(NamedTuple):
    """One synchronized grab. ``depth`` is None when depth is not enabled."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) float32 metres; non-finite = no hit
    sim_time: float  # world.current_time (or the mock's clock) at grab


class NoFrameYetError(RuntimeError):
    """No rendered frame is available (warm-up after create/reset). The model
    maps this to FAILED_PRECONDITION after a bounded retry (CAM-2, CAM-18)."""


class CameraHandle:
    """Backend-neutral camera. All public methods are safe from any thread."""

    depth_enabled: bool = False
    image_format: str = "png"
    frequency: float | None = None

    def get_frame(self) -> Frame:
        """rgb (+ depth) grabbed once per sim step and cached by sim_time, so
        GetImages + GetPointCloud in one tick share one grab (CAM-9). Raises
        NoFrameYetError while the renderer has not produced a frame (CAM-2)."""
        raise NotImplementedError

    def get_rgb(self) -> np.ndarray:
        return self.get_frame().rgb

    def get_depth(self) -> np.ndarray:
        """(H, W) float32 metres; raises NoFrameYetError / RuntimeError when
        depth is not enabled on this camera (CAM-1)."""
        depth = self.get_frame().depth
        if depth is None:
            raise RuntimeError("depth is not enabled on this camera (set depth: true)")
        return depth

    def get_intrinsics(self) -> Intrinsics:
        """Pinhole intrinsics, never zero-filled (CAM-5)."""
        raise NotImplementedError

    def post_reset(self) -> None:
        """Called by SimManager after every world.reset() (CAM-17 via XC-5).
        Backends drop cached frames / re-arm acquisition here."""
        return None

    def get_attr_summary(self) -> dict[str, Any]:  # diagnostics only
        return {"depth": self.depth_enabled, "image_format": self.image_format}
