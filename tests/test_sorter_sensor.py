"""sorter-sensor: at-most-once emission of new conductor loop records to
data capture, at both bounds - a repeated capture poll raises
NoCaptureToStoreError (never re-emits, never returns the empty map the RDK
rejects), and a genuinely new record is never silently dropped. Interactive
polls are live snapshots that never consume the capture cursor."""

from typing import Any

import pytest
from viam.errors import NoCaptureToStoreError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.utils import dict_to_struct

from isaac_module.models.sorter_sensor import SorterSensor

CONDUCTOR_NAME = "block-sorter"
CAPTURE_EXTRA = {"fromDataManagement": True}


def _config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _valid_attrs(**overrides: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {"conductor": CONDUCTOR_NAME}
    attrs.update(overrides)
    return attrs


class FakeConductor:
    def __init__(self) -> None:
        self.status: dict[str, Any] = {
            "state": "idle",
            "run": None,
            "success_rate": None,
        }

    async def do_command(self, command: dict[str, Any]) -> dict[str, Any]:
        assert command == {"command": "status"}
        return self.status


def _dependencies(conductor: FakeConductor) -> dict[ResourceName, Any]:
    return {ResourceName(name=CONDUCTOR_NAME): conductor}


def _sensor(conductor: FakeConductor) -> SorterSensor:
    sensor = SorterSensor(name="sorter-sensor-1")
    sensor.reconfigure(_config("sorter-sensor-1", _valid_attrs()), _dependencies(conductor))
    return sensor


def _record(record_id: int) -> dict[str, Any]:
    return {"record_id": record_id, "seed": record_id, "placed": 1, "failed": 0}


class TestValidateConfig:
    def test_rejects_missing_conductor(self) -> None:
        with pytest.raises(ValueError, match="conductor"):
            SorterSensor.validate_config(_config("s", {}))

    def test_returns_conductor_as_dependency(self) -> None:
        dependencies, optional = SorterSensor.validate_config(_config("s", _valid_attrs()))
        assert list(dependencies) == [CONDUCTOR_NAME]
        assert list(optional) == []


class TestCapturePolls:
    @pytest.mark.asyncio
    async def test_skips_storage_when_no_loop_records_key(self) -> None:
        conductor = FakeConductor()
        sensor = _sensor(conductor)
        with pytest.raises(NoCaptureToStoreError):
            await sensor.get_readings(extra=CAPTURE_EXTRA)

    @pytest.mark.asyncio
    async def test_skips_storage_when_loop_records_is_empty(self) -> None:
        conductor = FakeConductor()
        conductor.status["loop_records"] = []
        sensor = _sensor(conductor)
        with pytest.raises(NoCaptureToStoreError):
            await sensor.get_readings(extra=CAPTURE_EXTRA)

    @pytest.mark.asyncio
    async def test_first_poll_emits_the_whole_window(self) -> None:
        conductor = FakeConductor()
        conductor.status["loop_records"] = [_record(0), _record(1), _record(2)]
        sensor = _sensor(conductor)

        readings = await sensor.get_readings(extra=CAPTURE_EXTRA)

        assert readings["loops"] == [_record(0), _record(1), _record(2)]
        assert readings["state"] == "idle"
        assert readings["run"] is None
        assert readings["success_rate"] is None

    @pytest.mark.asyncio
    async def test_immediately_repeated_poll_skips_storage(self) -> None:
        conductor = FakeConductor()
        conductor.status["loop_records"] = [_record(0), _record(1), _record(2)]
        sensor = _sensor(conductor)

        await sensor.get_readings(extra=CAPTURE_EXTRA)
        with pytest.raises(NoCaptureToStoreError):
            await sensor.get_readings(extra=CAPTURE_EXTRA)

    @pytest.mark.asyncio
    async def test_new_record_emits_only_that_record(self) -> None:
        conductor = FakeConductor()
        conductor.status["loop_records"] = [_record(0), _record(1), _record(2)]
        sensor = _sensor(conductor)
        await sensor.get_readings(extra=CAPTURE_EXTRA)

        conductor.status["loop_records"] = [_record(0), _record(1), _record(2), _record(3)]
        third = await sensor.get_readings(extra=CAPTURE_EXTRA)

        assert third["loops"] == [_record(3)]

    @pytest.mark.asyncio
    async def test_repeated_poll_after_new_record_skips_storage(self) -> None:
        conductor = FakeConductor()
        conductor.status["loop_records"] = [_record(0), _record(1)]
        sensor = _sensor(conductor)
        await sensor.get_readings(extra=CAPTURE_EXTRA)
        conductor.status["loop_records"] = [_record(0), _record(1), _record(2)]
        await sensor.get_readings(extra=CAPTURE_EXTRA)

        with pytest.raises(NoCaptureToStoreError):
            await sensor.get_readings(extra=CAPTURE_EXTRA)


class TestInteractivePolls:
    @pytest.mark.asyncio
    async def test_never_returns_the_empty_map_the_rdk_rejects(self) -> None:
        conductor = FakeConductor()
        sensor = _sensor(conductor)

        readings = await sensor.get_readings()

        assert readings == {"loops": [], "state": "idle", "run": None, "success_rate": None}

    @pytest.mark.asyncio
    async def test_does_not_consume_the_capture_cursor(self) -> None:
        conductor = FakeConductor()
        conductor.status["loop_records"] = [_record(0), _record(1)]
        sensor = _sensor(conductor)

        interactive = await sensor.get_readings()
        assert interactive["loops"] == [_record(0), _record(1)]

        # the capture poll still sees everything the panel just displayed
        captured = await sensor.get_readings(extra=CAPTURE_EXTRA)
        assert captured["loops"] == [_record(0), _record(1)]
