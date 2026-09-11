from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from viam.logging import getLogger

from ..errors import PrimNotFoundError
from ..prim_paths import prim_name
from ..spatial import Quat, Vec3, _as_quat, to_vec3

if TYPE_CHECKING:
    from ..sim_manager import SimManager

LOGGER = getLogger(__name__)


class BaseHandle:
    """The interface the base component model talks to. Every public method
    is safe to call from any thread."""

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        """((x,y,z) meters, (w,x,y,z) quaternion) world pose of an arbitrary
        prim on the stage."""
        raise NotImplementedError

    def release(self) -> None:
        """Isaac backends remove their `<name>_drive` physics callback and
        the scene-registry entry (registry_only). The prim stays."""
        return None


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
        except Exception:  # the sim-thread drive step, any failure is logged, not fatal
            LOGGER.exception("error driving base")

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        def _pose() -> tuple[Vec3, Quat]:
            self._sim._require_prim(prim_path)
            pos, quat = self._sim._isaac.SingleXFormPrim(prim_path).get_world_pose()
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            return pos_t, quat_t

        return self._sim.run(_pose)

    def release(self) -> None:
        """Remove the `<name>_drive` physics callback and drop the
        scene-registry entry (registry_only - the prim stays), so a later
        create_base for this name can re-attach."""

        def _release() -> None:
            world = self._sim.world
            name = getattr(self._robot, "name", "")
            callback_name = f"{name}_drive"
            if world.physics_callback_exists(callback_name):
                world.remove_physics_callback(callback_name)
            if world.scene.get_object(name) is not None:
                world.scene.remove_object(name, registry_only=True)

        self._sim.run(_release)


class MockBaseHandle(BaseHandle):
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self.wheel_radius = float(attrs.get("wheel_radius", 0.05))
        self.wheel_base = float(attrs.get("wheel_base", 0.3))
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()
        self._prim_path = attrs.get("prim_path") or f"/World/{prim_name(name)}"
        self.spawn_position: Vec3 = to_vec3(attrs.get("position"))
        self.spawn_orientation: Quat = (
            _as_quat(attrs["orientation_wxyz"])
            if attrs.get("orientation_wxyz") is not None
            else (1.0, 0.0, 0.0, 0.0)
        )

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        if prim_path != self._prim_path:
            raise PrimNotFoundError(f"prim not found: {prim_path}")
        return self.spawn_position, self.spawn_orientation
