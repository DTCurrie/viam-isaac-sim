"""Unit tests for IsaacGripper's Viam-facing contract in mock mode."""

import asyncio
import json

import pytest
from grpclib import Status
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.gripper import IsaacGripper
from isaac_module.sim_manager import SimManager

_ABSTRACT_METHODS = {
    "open",
    "stop",
    "grab",
    "is_moving",
    "is_holding_something",
    "get_kinematics",
    "get_current_inputs",
    "go_to_inputs",
}


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _make_arm(world, name: str) -> None:
    from isaac_module.models.arm import IsaacArm

    IsaacArm.new(_config(name, {"world": "isaac-world", "asset": "ur5e", "mock_dof": 6}), {})


def _make_gripper(world, arm_name: str, name: str, extra: dict | None = None) -> IsaacGripper:
    attrs = {"world": "isaac-world", "arm": arm_name}
    if extra:
        attrs.update(extra)
    return IsaacGripper.new(_config(name, attrs), {})


def test_instantiates_with_exactly_the_eight_abstract_methods():
    assert Gripper.__abstractmethods__ == frozenset(_ABSTRACT_METHODS)
    IsaacGripper("gripper-instantiate-only")


def test_validate_config_requires_arm():
    with pytest.raises(ValueError, match="arm"):
        IsaacGripper.validate_config(_config("gripper-no-arm", {"world": "isaac-world"}))


def test_validate_config_frame_parent_must_be_arm():
    config = ComponentConfig(
        name="gripper-bad-frame",
        attributes=dict_to_struct({"world": "isaac-world", "arm": "my-arm"}),
    )
    config.frame.parent = "not-my-arm"
    with pytest.raises(ValueError, match="frame.parent"):
        IsaacGripper.validate_config(config)


def test_validate_config_valid_returns_deps():
    config = ComponentConfig(
        name="gripper-valid",
        attributes=dict_to_struct({"world": "isaac-world", "arm": "my-arm"}),
    )
    config.frame.parent = "my-arm"
    deps, implicit = IsaacGripper.validate_config(config)
    assert list(deps) == ["isaac-world", "my-arm"]
    assert list(implicit) == []


def test_grab_and_release_with_object(world):
    _make_arm(world, "grab-arm-a")
    gripper = _make_gripper(world, "grab-arm-a", "grab-gripper-a", {"mock_object_width_m": 0.05})

    async def scenario():
        await gripper.open()
        assert await gripper.grab() is True

        status = await gripper.is_holding_something()
        assert status.is_holding_something is True
        assert status.meta["open_deg"] < status.meta["jaw_deg"] < status.meta["closed_deg"]

        await gripper.open()
        status = await gripper.is_holding_something()
        assert status.is_holding_something is False

    asyncio.run(scenario())


def test_grab_with_no_object_returns_false(world):
    _make_arm(world, "grab-arm-b")
    gripper = _make_gripper(world, "grab-arm-b", "grab-gripper-b")

    async def scenario():
        assert await gripper.grab() is False
        status = await gripper.is_holding_something()
        assert status.meta["jaw_deg"] == pytest.approx(status.meta["closed_deg"], abs=0.5)

    asyncio.run(scenario())


def test_go_to_inputs_and_get_current_inputs_round_trip(world):
    _make_arm(world, "inputs-arm")
    gripper = _make_gripper(world, "inputs-arm", "inputs-gripper")

    async def scenario():
        await gripper.go_to_inputs([0.5])
        inputs = await gripper.get_current_inputs()
        assert inputs == pytest.approx([0.5], abs=1e-3)

    asyncio.run(scenario())


def test_go_to_inputs_out_of_range_raises(world):
    _make_arm(world, "inputs-arm-oob")
    gripper = _make_gripper(world, "inputs-arm-oob", "inputs-gripper-oob")

    async def scenario():
        with pytest.raises(Exception) as excinfo:
            await gripper.go_to_inputs([1.5])
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_go_to_inputs_wrong_length_raises(world):
    _make_arm(world, "inputs-arm-len")
    gripper = _make_gripper(world, "inputs-arm-len", "inputs-gripper-len")

    async def scenario():
        with pytest.raises(Exception) as excinfo:
            await gripper.go_to_inputs([0.2, 0.3])
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_get_kinematics_is_one_link_zero_joints(world):
    _make_arm(world, "kinematics-arm")
    gripper = _make_gripper(world, "kinematics-arm", "kinematics-gripper")

    async def scenario():
        fmt, data = await gripper.get_kinematics()
        from viam.proto.common import KinematicsFileFormat

        assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        sva = json.loads(data)
        assert len(sva["links"]) == 1
        assert sva["joints"] == []
        link = sva["links"][0]
        assert link["parent"] == "world"
        geometry = link["geometry"]
        assert (geometry["x"], geometry["y"], geometry["z"]) == (36, 146, 153)
        # flange -> fingertips: centre 57.5 mm behind the TCP, so the box never
        # extends below the pads (a floor-level grasp would read as a collision)
        assert geometry["translation"]["z"] == pytest.approx(-57.5)

    asyncio.run(scenario())


def test_get_geometries_is_one_box(world):
    _make_arm(world, "geometries-arm")
    gripper = _make_gripper(world, "geometries-arm", "geometries-gripper")

    async def scenario():
        geometries = await gripper.get_geometries()
        assert len(geometries) == 1
        geometry = geometries[0]
        box = geometry.box
        assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (36, 146, 153)
        assert geometries[0].center.z == pytest.approx(-57.5)
        assert geometry.center.o_z == 1

    asyncio.run(scenario())


def test_open_blocks_until_the_jaw_settles_at_the_open_limit(world):
    _make_arm(world, "moving-arm")
    gripper = _make_gripper(world, "moving-arm", "moving-gripper")

    async def scenario():
        await gripper.grab()  # closes with nothing to grab -> jaw at closed_rad
        await gripper.open()
        # open() returns only once the jaw has stopped moving, at the open limit
        assert await gripper.is_moving() is False
        status = await gripper.is_holding_something()
        assert status.meta["jaw_deg"] == pytest.approx(status.meta["open_deg"], abs=0.5)

    asyncio.run(scenario())


def test_close_releases_the_handle(world):
    _make_arm(world, "close-arm")
    gripper = _make_gripper(world, "close-arm", "close-gripper")

    async def scenario():
        await gripper.close()
        assert "close-gripper" not in SimManager.get()._handles

    asyncio.run(scenario())
