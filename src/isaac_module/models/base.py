"""The simulated differential-drive base model."""

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.base import Base
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..length_units import to_meters
from ..sim_manager import BaseHandle, SimManager
from .component_frame_pose import apply_frame_to_attrs, get_attrs
from .sim_component_validation import validate_sim_component

# wheeled_base.go treats an angle or speed within this threshold of zero as
# "nearly 0" (`wheeled_base.go:243`, `wheeled_base.go:248`)
NEAR_ZERO_THRESHOLD = 0.0001


class BaseMoveTimeoutError(ViamGRPCError, TimeoutError):
    """move_straight/spin did not reach their target before the SDK's
    timeout= kwarg cut the wait short of the move's natural duration."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.DEADLINE_EXCEEDED)
        Exception.__init__(self, message)


def _validate_base_attrs(name: str, attrs: dict[str, Any]) -> None:
    for attr in ("width_mm", "wheel_circumference_mm"):
        value = attrs.get(attr)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'{name}: "{attr}" must be a positive number, got {value!r}')


def _with_wheel_geometry_from_mm(attrs: dict[str, Any]) -> dict[str, Any]:
    """When a real wheeled base's width_mm / wheel_circumference_mm are set
    and the meter-valued wheel_base / wheel_radius aren't, derive the latter
    so the sim geometry and GetProperties follow the real config. carry is a
    flat copy (docs/SIMULATION.md), so this is where the mm-to-m conversion
    a flat copy can't express happens instead."""
    derived = dict(attrs)
    width_mm = attrs.get("width_mm")
    if width_mm is not None and "wheel_base" not in attrs:
        derived["wheel_base"] = to_meters(float(width_mm))
    wheel_circumference_mm = attrs.get("wheel_circumference_mm")
    if wheel_circumference_mm is not None and "wheel_radius" not in attrs:
        derived["wheel_radius"] = to_meters(float(wheel_circumference_mm)) / (2 * math.pi)
    return derived


class IsaacBase(Base, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    """viam:isaac-sim-devin:base, a simulated differential-drive base.

    Drives the two wheel joints of a robot spawned in the world's sim. The wire
    carries millimeters and degrees, the sim takes meters and radians, and the
    conversion happens in the verbs below. close() releases the handle and
    leaves the prim in the stage. move_straight/spin always stop the base when
    their wait ends, whether that is the move's natural duration, a dropped
    RPC, or the SDK's timeout= kwarg cutting the wait short."""

    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "base")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: BaseHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._max_linear = 0.5
        self._max_angular = 2.0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        base = cls(config.name)
        base.reconfigure(config, dependencies)
        return base

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        """Attributes:
        world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
        asset (string)               - known robot, e.g. "jetbot" (brings sensible
                                       wheel defaults)
        usd_path (string)            - explicit robot USD to spawn
        prim_path (string)           - where to place it / existing robot prim
        position ([x,y,z] meters)    - spawn position
        wheel_joints ([left, right]) - wheel joint names (required unless the asset
                                       provides them)
        wheel_radius (meters)        - default from asset, else 0.05
        wheel_base (meters)          - default from asset, else 0.3
        width_mm (float, positive)   - real base width; derives wheel_base
                                       (width_mm / 1000) when wheel_base isn't
                                       set
        wheel_circumference_mm (float, positive) - real wheel circumference;
                                       derives wheel_radius when wheel_radius
                                       isn't set
        max_linear_mps (float)       - full-power linear speed, default 0.5
        max_angular_rps (float)      - full-power angular speed (rad/s), default 2.0
        """
        deps, opt_deps = validate_sim_component(config)
        _validate_base_attrs(config.name, get_attrs(config))
        return deps, opt_deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = _with_wheel_geometry_from_mm(apply_frame_to_attrs(config, get_attrs(config)))
        self._max_linear = float(attrs.get("max_linear_mps", 0.5))
        self._max_angular = float(attrs.get("max_angular_rps", 2.0))
        self._attrs = attrs
        self._handle = SimManager.get().create_base(self.name, attrs)

    async def close(self) -> None:
        """Release the handle. The prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> BaseHandle:
        if self._handle is None:
            raise RuntimeError(f"base {self.name} is not attached to the sim")
        return self._handle

    async def move_straight(
        self, distance: int, velocity: float, *, timeout: float | None = None, **kwargs
    ) -> None:
        if abs(velocity) < NEAR_ZERO_THRESHOLD:
            # a zero speed is a no-op stop, not an error (`wheeled_base.go:267`)
            await asyncio.to_thread(self._h().stop)
            return
        meters = distance / 1000.0
        speed = abs(velocity) / 1000.0
        direction = 1.0 if (meters >= 0) == (velocity >= 0) else -1.0
        duration = abs(meters) / speed
        wait_s = duration if timeout is None else min(duration, timeout)
        handle = self._h()
        await asyncio.to_thread(handle.set_velocity, direction * speed, 0.0)
        try:
            await asyncio.sleep(wait_s)
        finally:
            await asyncio.to_thread(handle.stop)
        if timeout is not None and timeout < duration:
            raise BaseMoveTimeoutError(
                f"base {self.name}: move_straight did not reach its target within "
                f"{timeout:.2f}s (needed {duration:.2f}s)"
            )

    async def spin(
        self, angle: float, velocity: float, *, timeout: float | None = None, **kwargs
    ) -> None:
        if abs(angle) < NEAR_ZERO_THRESHOLD:
            # nearly-zero angle is an error, not a no-op (`wheeled_base.go:243`)
            raise ValueError(f"cannot move base {self.name} for an angle that is nearly 0")
        if abs(velocity) < NEAR_ZERO_THRESHOLD:
            # a zero speed is a no-op stop, not an error (`wheeled_base.go:248`)
            await asyncio.to_thread(self._h().stop)
            return
        radians = math.radians(angle)
        speed = math.radians(abs(velocity))
        direction = 1.0 if (radians >= 0) == (velocity >= 0) else -1.0
        duration = abs(radians) / speed
        wait_s = duration if timeout is None else min(duration, timeout)
        handle = self._h()
        await asyncio.to_thread(handle.set_velocity, 0.0, direction * speed)
        try:
            await asyncio.sleep(wait_s)
        finally:
            await asyncio.to_thread(handle.stop)
        if timeout is not None and timeout < duration:
            raise BaseMoveTimeoutError(
                f"base {self.name}: spin did not reach its target within "
                f"{timeout:.2f}s (needed {duration:.2f}s)"
            )

    async def set_power(
        self, linear: Vector3, angular: Vector3, *, timeout: float | None = None, **kwargs
    ) -> None:
        lin = max(-1.0, min(1.0, linear.y)) * self._max_linear
        ang = max(-1.0, min(1.0, angular.z)) * self._max_angular
        await asyncio.to_thread(self._h().set_velocity, lin, ang)

    async def set_velocity(
        self, linear: Vector3, angular: Vector3, *, timeout: float | None = None, **kwargs
    ) -> None:
        # viam: linear mm/s, angular deg/s -> isaac: m/s, rad/s
        await asyncio.to_thread(self._h().set_velocity, linear.y / 1000.0, math.radians(angular.z))

    async def stop(self, *, timeout: float | None = None, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_geometries(self, **kwargs):
        """Empty, so the motion service can build a world state when this base
        has a frame. The chassis is not modeled yet."""
        return []

    async def get_properties(self, **kwargs) -> Base.Properties:
        wheel_radius = getattr(self._h(), "wheel_radius", 0.05)
        wheel_base = getattr(self._h(), "wheel_base", 0.3)
        return Base.Properties(
            width_meters=wheel_base,
            turning_radius_meters=0.0,
            wheel_circumference_meters=2 * math.pi * wheel_radius,
        )
