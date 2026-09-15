import asyncio
import json

import pytest
from grpclib import Status
from viam.components.gripper import Gripper
from viam.errors import MethodNotImplementedError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import KinematicsFileFormat
from viam.utils import dict_to_struct

from isaac_module.models.vacuum import IsaacVacuum
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


def _make_vacuum(world, arm_name: str, name: str, extra: dict | None = None) -> IsaacVacuum:
    # grab_delay_ms defaults to 0 here so tests that aren't about timing stay fast;
    # tests that care about the delay window pass their own grab_delay_ms.
    attrs = {"world": "isaac-world", "arm": arm_name, "grab_delay_ms": 0}
    if extra:
        attrs.update(extra)
    return IsaacVacuum.new(_config(name, attrs), {})


def test_instantiates_with_exactly_the_eight_abstract_methods():
    assert Gripper.__abstractmethods__ == frozenset(_ABSTRACT_METHODS)
    IsaacVacuum("vacuum-instantiate-only")


def test_validate_config_requires_arm():
    with pytest.raises(ValueError, match="arm"):
        IsaacVacuum.validate_config(_config("vacuum-no-arm", {"world": "isaac-world"}))


def test_validate_config_frame_parent_must_be_arm():
    config = ComponentConfig(
        name="vacuum-bad-frame",
        attributes=dict_to_struct({"world": "isaac-world", "arm": "my-arm"}),
    )
    config.frame.parent = "not-my-arm"
    with pytest.raises(ValueError, match=r"frame\.parent"):
        IsaacVacuum.validate_config(config)


def test_validate_config_valid_returns_deps():
    config = ComponentConfig(
        name="vacuum-valid",
        attributes=dict_to_struct({"world": "isaac-world", "arm": "my-arm"}),
    )
    config.frame.parent = "my-arm"
    deps, implicit = IsaacVacuum.validate_config(config)
    assert list(deps) == ["isaac-world", "my-arm"]
    assert list(implicit) == []


def test_grab_with_attach_prop_holds(world):
    _make_arm(world, "grab-arm-a")
    vacuum = _make_vacuum(world, "grab-arm-a", "grab-vacuum-a", {"mock_attach_prop": "some-block"})

    async def scenario():
        await vacuum.open()
        assert await vacuum.grab() is True

        status = await vacuum.is_holding_something()
        assert status.is_holding_something is True
        assert status.meta["engaged"] is True
        assert status.meta["holding"] is True

        await vacuum.open()
        status = await vacuum.is_holding_something()
        assert status.is_holding_something is False
        assert status.meta["engaged"] is False
        assert status.meta["holding"] is False

    asyncio.run(scenario())


def test_grab_with_no_attach_prop_returns_false(world):
    _make_arm(world, "grab-arm-b")
    vacuum = _make_vacuum(world, "grab-arm-b", "grab-vacuum-b")

    async def scenario():
        assert await vacuum.grab() is False
        status = await vacuum.is_holding_something()
        assert status.is_holding_something is False

    asyncio.run(scenario())


def test_stop_and_is_moving_are_false_with_no_grab_delay(world):
    _make_arm(world, "moving-arm")
    vacuum = _make_vacuum(world, "moving-arm", "moving-vacuum")

    async def scenario():
        await vacuum.stop()
        assert await vacuum.is_moving() is False
        await vacuum.grab()
        assert await vacuum.is_moving() is False

    asyncio.run(scenario())


def test_grab_waits_out_grab_delay_ms_and_is_moving_reports_the_window(world):
    _make_arm(world, "delay-arm")
    vacuum = _make_vacuum(
        world,
        "delay-arm",
        "delay-vacuum",
        {"mock_attach_prop": "some-block", "grab_delay_ms": 50},
    )

    async def scenario():
        assert await vacuum.is_moving() is False

        grab_task = asyncio.ensure_future(vacuum.grab())
        await asyncio.sleep(0.01)
        assert await vacuum.is_moving() is True

        assert await grab_task is True
        assert await vacuum.is_moving() is False

    asyncio.run(scenario())


def test_validate_config_rejects_a_negative_grab_delay_ms():
    with pytest.raises(ValueError, match="grab_delay_ms"):
        IsaacVacuum.validate_config(
            _config(
                "vacuum-bad-delay", {"world": "isaac-world", "arm": "my-arm", "grab_delay_ms": -1}
            )
        )


def test_go_to_inputs_and_get_current_inputs_round_trip(world):
    _make_arm(world, "inputs-arm")
    vacuum = _make_vacuum(world, "inputs-arm", "inputs-vacuum", {"mock_attach_prop": "some-block"})

    async def scenario():
        await vacuum.go_to_inputs([1.0])
        assert await vacuum.get_current_inputs() == pytest.approx([1.0])

        await vacuum.go_to_inputs([0.0])
        assert await vacuum.get_current_inputs() == pytest.approx([0.0])

        # exactly at the engage threshold: engages
        await vacuum.go_to_inputs([0.5])
        assert await vacuum.get_current_inputs() == pytest.approx([1.0])

    asyncio.run(scenario())


def test_go_to_inputs_out_of_range_raises(world):
    _make_arm(world, "inputs-arm-oob")
    vacuum = _make_vacuum(world, "inputs-arm-oob", "inputs-vacuum-oob")

    async def scenario():
        with pytest.raises(Exception) as excinfo:
            await vacuum.go_to_inputs([1.5])
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_go_to_inputs_wrong_length_raises(world):
    _make_arm(world, "inputs-arm-len")
    vacuum = _make_vacuum(world, "inputs-arm-len", "inputs-vacuum-len")

    async def scenario():
        with pytest.raises(Exception) as excinfo:
            await vacuum.go_to_inputs([0.2, 0.3])
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_do_command_raises_method_not_implemented(world):
    _make_arm(world, "do-command-arm")
    vacuum = _make_vacuum(world, "do-command-arm", "do-command-vacuum")

    async def scenario():
        with pytest.raises(MethodNotImplementedError) as excinfo:
            await vacuum.do_command({"command": "not-a-real-command"})
        assert excinfo.value.grpc_code == Status.UNIMPLEMENTED

    asyncio.run(scenario())


def test_get_kinematics_and_get_geometries_describe_the_same_box(world):
    _make_arm(world, "kinematics-arm")
    vacuum = _make_vacuum(world, "kinematics-arm", "kinematics-vacuum")

    async def scenario():
        fmt, data = await vacuum.get_kinematics()
        assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        sva = json.loads(data)
        assert len(sva["links"]) == 1
        assert sva["joints"] == []
        link = sva["links"][0]
        assert link["parent"] == "world"
        geometry = link["geometry"]
        assert (geometry["x"], geometry["y"], geometry["z"]) == (80, 80, 196)
        assert geometry["translation"]["z"] == pytest.approx(-98.0)

        geometries = await vacuum.get_geometries()
        assert len(geometries) == 1
        box = geometries[0].box
        assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (80, 80, 196)
        assert geometries[0].center.z == pytest.approx(-98.0)
        assert geometries[0].center.o_z == 1

    asyncio.run(scenario())


def test_close_releases_the_handle(world):
    _make_arm(world, "close-arm")
    vacuum = _make_vacuum(world, "close-arm", "close-vacuum")

    async def scenario():
        await vacuum.close()
        assert "close-vacuum" not in SimManager.get()._handles

    asyncio.run(scenario())


def test_an_engaged_cup_that_caught_nothing_reports_its_command_not_its_catch(world):
    """The parallel-jaw model reports a jaw closed on nothing at its
    commanded position, so a cup that ran with nothing under it has to report
    the same way. With mock_attach_prop unset there is nothing to catch, and
    that is the only case where the commanded state and the catch differ."""
    _make_arm(world, "empty-cup-arm")
    vacuum = _make_vacuum(world, "empty-cup-arm", "empty-cup-vacuum")

    async def scenario():
        assert await vacuum.grab() is False
        assert await vacuum.get_current_inputs() == pytest.approx([1.0])

        status = await vacuum.is_holding_something()
        assert status.is_holding_something is False
        assert status.meta["engaged"] is True
        assert status.meta["holding"] is False

        await vacuum.open()
        assert await vacuum.get_current_inputs() == pytest.approx([0.0])

    asyncio.run(scenario())


def test_the_tool_body_reaches_exactly_as_far_as_the_declared_tcp():
    """The tool is authored as a cuboid hung from the flange by half its own
    length, so its cup face lands at its full length below the flange. That
    has to be the same number the frame and the planner are given as
    tcp_offset_m, or the arm drives a face that is not where the geometry
    ends and stalls against whatever it meets first."""
    from isaac_module.asset_catalog import VACUUM_TOOL

    tool_length_mm = float(VACUUM_TOOL["box_mm"][2])
    assert tool_length_mm == float(VACUUM_TOOL["tcp_offset_m"]) * 1000.0
