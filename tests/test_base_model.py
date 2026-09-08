"""Unit tests for IsaacBase's zero-velocity / near-zero-angle contract in mock mode."""

import asyncio

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.base import IsaacBase


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
