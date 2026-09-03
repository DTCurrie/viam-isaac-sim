"""viam:isaac-sim-devin:sorter-sensor - proxies a conductor's ``status``
DoCommand for data management, emitting each new loop record at most once.

Attributes:
  conductor (string, required) - name of the viam:isaac-sim-devin:conductor
                                  generic service to poll

Readings, on a data-management capture poll (``extra`` carries
``fromDataManagement``): ``{"loops": [only-the-new LoopRecord dicts],
"state": ..., "run": ..., "success_rate": ...}`` taken verbatim from the
conductor's status, and the high-water mark (the greatest ``record_id``
seen) advances to cover everything just emitted; with nothing new the poll
raises ``NoCaptureToStoreError`` so the data manager stores nothing (the
RDK rejects an empty readings map outright). Any other caller (app panel,
scripts) gets the same shape as a live snapshot - ``loops`` holds whatever
the capture cursor has not consumed yet - and never advances the mark, so
an open status panel cannot eat records out from under data capture. The
mark lives for the module's lifetime and is never reset by
``reconfigure``; a module restart may re-emit the whole window, which
downstream keys on ``record_id`` to dedupe again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from typing_extensions import Self
from viam.components.sensor import Sensor
from viam.errors import NoCaptureToStoreError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, from_dm_from_extra, struct_to_dict

from .. import FAMILY, NAMESPACE


class SorterSensor(Sensor, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "sorter-sensor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._last_emitted_record_id: int = -1

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        conductor_name = attrs.get("conductor")
        if not conductor_name or not isinstance(conductor_name, str):
            raise ValueError(f'{config.name}: set the "conductor" attribute to a resource name')
        return [conductor_name], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        conductor_name = str(attrs["conductor"])
        by_name: dict[str, ResourceBase] = {
            rn.name: resource for rn, resource in dependencies.items()
        }
        if conductor_name not in by_name:
            raise ValueError(
                f"{config.name}: dependency {conductor_name!r} for 'conductor' was not resolved"
            )
        self._conductor = by_name[conductor_name]

    async def get_readings(
        self,
        *,
        extra: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Mapping[str, SensorReading]:
        status = cast("Mapping[str, Any]", await self._conductor.do_command({"command": "status"}))
        loop_records = status.get("loop_records")
        records = list(loop_records) if isinstance(loop_records, Sequence) else []
        new_records = [
            record
            for record in records
            if isinstance(record, Mapping)
            and int(cast(Any, record.get("record_id", -1))) > self._last_emitted_record_id
        ]
        snapshot = cast(
            "Mapping[str, SensorReading]",
            {
                "loops": new_records,
                "state": status.get("state"),
                "run": status.get("run"),
                "success_rate": status.get("success_rate"),
            },
        )
        if not from_dm_from_extra(dict(extra) if extra is not None else None):
            # a live view (app panel, scripts): never consume the capture
            # cursor, never error - the RDK rejects an empty readings map
            return snapshot
        if not new_records:
            raise NoCaptureToStoreError
        self._last_emitted_record_id = max(
            int(cast(Any, record["record_id"])) for record in new_records
        )
        return snapshot
