"""erh:isaac-sim:arm - a simulated arm.

Attributes:
  world (string, required)   - name of the erh:isaac-sim:world component
  asset (string)             - known robot, e.g. "ur20", "ur10", "franka"
  usd_path (string)          - explicit USD to spawn instead of a known asset
  prim_path (string)         - where to place it (default /World/<name>), or
                               an existing articulation in the stage
  position ([x,y,z] meters)  - spawn position
  end_effector_prim (string) - prim path whose world pose is reported by
                               GetEndPosition
  move_timeout_sec (float)   - max time to wait for a move (default 30)
"""

import asyncio
import math
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm, JointPositions, KinematicsFileFormat, Pose
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..sim_manager import ArmHandle, SimManager
from ..spatial import quat_to_ov
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_TOLERANCE_RAD = math.radians(0.5)


class IsaacArm(Arm, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[ArmHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._move_timeout = 30.0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        arm = cls(config.name)
        arm.reconfigure(config, dependencies)
        return arm

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        return validate_sim_component(config)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._move_timeout = float(attrs.get("move_timeout_sec", 30.0))
        self._attrs = attrs
        self._handle = SimManager.get().create_arm(self.name, attrs)

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    async def get_end_position(self, **kwargs) -> Pose:
        (x, y, z), quat = await asyncio.to_thread(self._h().get_end_pose)
        ox, oy, oz, theta = quat_to_ov(quat)
        return Pose(
            x=x * 1000.0,
            y=y * 1000.0,
            z=z * 1000.0,
            o_x=ox,
            o_y=oy,
            o_z=oz,
            theta=math.degrees(theta),
        )

    async def move_to_position(self, pose: Pose, **kwargs) -> None:
        raise NotImplementedError(
            "IK and motion planning are Viam's job, not the module's: use the "
            "motion service (needs GetKinematics, on the roadmap) or "
            "move_to_joint_positions"
        )

    async def move_to_joint_positions(
        self, positions: JointPositions, **kwargs
    ) -> None:
        targets = [math.radians(v) for v in positions.values]
        handle = self._h()
        await asyncio.to_thread(handle.set_joint_targets, targets)

        deadline = time.monotonic() + self._move_timeout
        while time.monotonic() < deadline:
            current = await asyncio.to_thread(handle.get_joint_positions)
            if len(current) >= len(targets) and all(
                abs(c - t) <= _TOLERANCE_RAD for c, t in zip(current, targets)
            ):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"arm {self.name} did not reach target within {self._move_timeout}s"
        )

    async def get_joint_positions(self, **kwargs) -> JointPositions:
        radians = await asyncio.to_thread(self._h().get_joint_positions)
        return JointPositions(values=[math.degrees(r) for r in radians])

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(self, **kwargs) -> Tuple[KinematicsFileFormat.ValueType, bytes]:
        raise NotImplementedError("kinematics files are not provided yet")

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        return []

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "get_joint_positions_radians":
            return {"values": await asyncio.to_thread(self._h().get_joint_positions)}
        raise ValueError(f"unknown command: {command}")
