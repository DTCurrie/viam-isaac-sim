import asyncio
import time
from typing import ClassVar

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.base import BaseMoveTimeoutError, IsaacBase
from isaac_module.sim_manager import SimConfig, SimManager


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _make_base(name: str) -> IsaacBase:
    return IsaacBase.new(_config(name, {"world": "isaac-world", "asset": "jetbot"}), {})


def test_move_straight_zero_velocity_stops_without_raising(world):
    base = _make_base("zero-velocity-base")

    async def scenario():
        await base.move_straight(distance=100, velocity=0)
        assert await base.is_moving() is False

    asyncio.run(scenario())


def test_spin_near_zero_angle_raises(world):
    base = _make_base("near-zero-angle-base")

    async def scenario():
        with pytest.raises(ValueError):
            await base.spin(angle=0, velocity=30)

    asyncio.run(scenario())


def test_spin_zero_velocity_stops_without_raising(world):
    base = _make_base("zero-spin-velocity-base")

    async def scenario():
        await base.spin(angle=90, velocity=0)
        assert await base.is_moving() is False

    asyncio.run(scenario())


def test_move_straight_normal_move_still_moves(world):
    base = _make_base("normal-move-base")

    async def scenario():
        moved = asyncio.create_task(base.move_straight(distance=100, velocity=1000))
        await asyncio.sleep(0.02)
        assert await base.is_moving() is True
        await moved

    asyncio.run(scenario())


def test_move_straight_short_timeout_raises_within_the_deadline(world):
    """distance/velocity here would naturally take 100s - unbounded, from the
    caller's point of view - so a short timeout= must cut it off and raise,
    not block for the full move."""
    base = _make_base("move-straight-timeout-base")

    async def scenario():
        start = time.monotonic()
        with pytest.raises(BaseMoveTimeoutError):
            await base.move_straight(distance=100000, velocity=1, timeout=0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
        assert await base.is_moving() is False  # stopped, not left coasting

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# IS-24: DifferentialController is built without its three speed limits, so
# only the commanded velocity is clamped (models/base.py) and never the
# resulting wheel speed.
# ----------------------------------------------------------------------


class _FakeWheeledRobot:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get("name")


class _FakeDifferentialController:
    last_kwargs: ClassVar[dict] = {}

    def __init__(self, **kwargs) -> None:
        _FakeDifferentialController.last_kwargs = kwargs


class _FakeIsaacNamespace:
    WheeledRobot = _FakeWheeledRobot
    DifferentialController = _FakeDifferentialController


class _FakeWorld:
    def __init__(self) -> None:
        self.scene = self

    def add(self, obj) -> None:
        pass

    def reset(self) -> None:
        pass

    def add_physics_callback(self, name, callback) -> None:
        pass


def _isaac_base_manager() -> SimManager:
    manager = SimManager()
    manager.mock = False
    manager.cfg = SimConfig(mock=False)
    manager._isaac = _FakeIsaacNamespace()
    manager.world = _FakeWorld()
    return manager


def test_differential_controller_receives_the_three_speed_limits():
    manager = _isaac_base_manager()
    attrs = {
        "usd_path": "fake_robot.usd",
        "wheel_joints": ["left_wheel", "right_wheel"],
        "wheel_radius": 0.05,
        "wheel_base": 0.3,
        "max_linear_mps": 0.8,
        "max_angular_rps": 3.0,
    }

    manager._create_base_isaac("speed-limited-base", attrs)

    kwargs = _FakeDifferentialController.last_kwargs
    assert kwargs["max_linear_speed"] == pytest.approx(0.8)
    assert kwargs["max_angular_speed"] == pytest.approx(3.0)
    assert kwargs["max_wheel_speed"] == pytest.approx((0.8 + 3.0 * 0.3 / 2.0) / 0.05)


def test_differential_controller_speed_limits_default_to_the_model_defaults():
    # models/base.py defaults: max_linear_mps 0.5, max_angular_rps 2.0.
    manager = _isaac_base_manager()
    attrs = {
        "usd_path": "fake_robot.usd",
        "wheel_joints": ["left_wheel", "right_wheel"],
        "wheel_radius": 0.05,
        "wheel_base": 0.3,
    }

    manager._create_base_isaac("default-speed-base", attrs)

    kwargs = _FakeDifferentialController.last_kwargs
    assert kwargs["max_linear_speed"] == pytest.approx(0.5)
    assert kwargs["max_angular_speed"] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# width_mm / wheel_circumference_mm: real-base geometry attributes carried
# by a flat copy (docs/SIMULATION.md), derived here into the meter-valued
# wheel_base / wheel_radius before create_base.
# ----------------------------------------------------------------------


def test_validate_config_rejects_non_positive_width_mm(world):
    config = _config("base-bad-width", {"world": "isaac-world", "asset": "jetbot", "width_mm": 0})
    with pytest.raises(ValueError, match="width_mm"):
        IsaacBase.validate_config(config)


def test_validate_config_rejects_non_positive_wheel_circumference_mm(world):
    config = _config(
        "base-bad-circumference",
        {"world": "isaac-world", "asset": "jetbot", "wheel_circumference_mm": -1},
    )
    with pytest.raises(ValueError, match="wheel_circumference_mm"):
        IsaacBase.validate_config(config)


def test_get_properties_derives_wheel_geometry_from_mm_attributes(world):
    base = IsaacBase.new(
        _config(
            "base-derived-geometry",
            {
                "world": "isaac-world",
                "asset": "jetbot",
                "width_mm": 300,
                "wheel_circumference_mm": 314.159265,
            },
        ),
        {},
    )

    async def scenario():
        props = await base.get_properties()
        assert props.width_meters == pytest.approx(0.3)
        assert props.wheel_circumference_meters == pytest.approx(0.314159265)

    asyncio.run(scenario())


def test_explicit_wheel_base_wins_over_width_mm(world):
    base = IsaacBase.new(
        _config(
            "base-explicit-wheel-base-wins",
            {"world": "isaac-world", "asset": "jetbot", "width_mm": 300, "wheel_base": 0.5},
        ),
        {},
    )

    async def scenario():
        props = await base.get_properties()
        assert props.width_meters == pytest.approx(0.5)

    asyncio.run(scenario())
