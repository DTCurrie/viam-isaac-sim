"""The simulated vacuum-cup gripper model."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.gripper import Gripper
from viam.errors import MethodNotImplementedError, ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    KinematicsFileFormat,
    Pose,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import DEFAULT_WORLD_NAME, FAMILY, NAMESPACE
from ..asset_catalog import VACUUM_TOOL
from ..handles.vacuum import DEFAULT_GRAB_DELAY_MS, VacuumGripperHandle
from ..sim_manager import SimManager
from .component_frame_pose import get_attrs

VACUUM_DEFAULT_TCP_OFFSET_M = float(VACUUM_TOOL["tcp_offset_m"])
VACUUM_BOX_MM: tuple[float, float, float] = VACUUM_TOOL["box_mm"]
# The box spans flange -> cup face. In the gripper frame (origin = TCP) its
# center sits half its own length behind the TCP - the same box-center-from-
# TCP arithmetic models/gripper.py uses for GRIPPER_BOX_CENTER_Z_MM.
VACUUM_BOX_CENTER_Z_MM = (VACUUM_BOX_MM[2] / 2000.0 - VACUUM_DEFAULT_TCP_OFFSET_M) * 1000.0

# A vacuum cup is a binary actuator: it is engaged or it is not, so the SDK's
# continuous [0, 1] input range only ever lands on these two points. Anything
# at or above this counts as commanding "engaged".
ENGAGE_INPUT_THRESHOLD = 0.5


def _vacuum_sva(
    link_id: str, box_mm: tuple[float, float, float], box_center_z_mm: float
) -> dict[str, Any]:
    """The vacuum tool's kinematics: one link, no joints, whose geometry is
    the box_mm RectangularPrism spanning flange to cup face - centered
    box_center_z_mm along the tool axis from the TCP, the frame origin."""
    box_x_mm, box_y_mm, box_z_mm = box_mm
    return {
        "name": link_id,
        "kinematic_param_type": "SVA",
        "links": [
            {
                "id": link_id,
                "parent": "world",
                "translation": {"x": 0, "y": 0, "z": 0},
                "orientation": {
                    "type": "ov_degrees",
                    "value": {"x": 0, "y": 0, "z": 1, "th": 0},
                },
                "geometry": {
                    "x": box_x_mm,
                    "y": box_y_mm,
                    "z": box_z_mm,
                    "translation": {"x": 0, "y": 0, "z": box_center_z_mm},
                },
            }
        ],
        "joints": [],
    }


class IsaacVacuum(Gripper, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    """viam:isaac-sim-devin:vacuum, a simulated suction-cup gripper riding an
    arm. It serves the ordinary Viam Gripper API over the vacuum handle, so a
    client can't tell it from the Robotiq gripper model and a machine config
    written for real hardware runs unedited against the sim.

    Frame: as models/gripper.py's IsaacGripper, the vacuum's frame is its
    TCP (the cup's contact face), so the motion service plans the TCP onto
    the block:

        "frame": {"parent": "<arm>", "translation": {"x": 0, "y": 0, "z": <tcp_offset_m * 1000>}}

    Unlike a mounted camera, the frame does NOT place the prim: the tool
    bolts to parent_prim at local_position / local_orientation_rpy_deg, and
    the frame's translation is the TCP the planner uses. validate_config
    requires frame.parent == arm.

    API mapping (viam-sdk Gripper, all eight abstract methods). A real epick
    builds suction pressure over real time, so grab() waits out
    grab_delay_ms before it reports whether the cup caught anything, and
    is_moving() reports True for that window.
      open / stop                      -> the handle
      is_moving()                      -> True while a commanded grab is still
                                          within its grab_delay_ms window, False otherwise
      grab() -> bool                   -> engage the cup, wait grab_delay_ms, return is_holding()
      is_holding_something()           -> HoldingStatus(is_holding, meta={"engaged": bool})
      get_current_inputs() / go_to_inputs([v])  -> one value in [0, 1]: >= 0.5 engages,
                                          < 0.5 releases - the only two points a binary
                                          actuator has on that range
      get_kinematics()                 -> 1-link / 0-joint SVA whose link is the 80 x 80 x 196 mm
                                          vacuum tool box, centered 98 mm behind the TCP (a
                                          gripper whose Kinematics fails is silently dropped
                                          from the frame system)
      get_geometries()                 -> that same single box (rdk keeps only [0])
      close()                          -> SimManager.release_handle
    """

    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "vacuum")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: VacuumGripperHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._kinematics: tuple[KinematicsFileFormat.ValueType, bytes] | None = None
        self._grab_delay_ms = DEFAULT_GRAB_DELAY_MS

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        vacuum = cls(config.name)
        vacuum.reconfigure(config, dependencies)
        return vacuum

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        """Requires world + arm. When a frame is set its parent must be the
        arm (the frame is the TCP in the arm's tool frame). Returns both as
        dependencies so viam-server builds the arm before the vacuum gripper.

        Attributes:
          world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
          arm (string, required)          - name of the viam:isaac-sim-devin:arm it is bolted to
          parent_prim (string)            - link it is bolted to, default the riding arm's
                                            wrist_3_link
          local_position ([x,y,z] m)      - mount pose of the tool's base on parent_prim,
                                            default identity
          local_orientation_rpy_deg       - mount orientation, default identity
          tcp_offset_m (float)            - flange -> cup face along tool +Z, default 0.196
          max_payload_gap_m (float)       - gap from a candidate's top face up to the cup
                                            face that still counts as contact, default 0.01
          grab_delay_ms (int)             - how long a commanded grab takes to build suction,
                                            matching viam:robotiq:simulated-epick-vacuum-gripper,
                                            default 1000. is_moving() reports True for this
                                            window after grab() is commanded.
          mock_attach_prop (string)       - mock only: name of the prop the cup finds under
                                            it (unset = nothing to grab, so grab() returns False)
        """
        attrs = get_attrs(config)
        world = attrs.get("world", DEFAULT_WORLD_NAME)
        if not world or not isinstance(world, str):
            raise ValueError(
                f'{config.name}: "world" defaults to "{DEFAULT_WORLD_NAME}" and, when set, '
                "must be a non-empty string naming your "
                f"{NAMESPACE}:{FAMILY}:world component"
            )
        arm = attrs.get("arm")
        if not arm or not isinstance(arm, str):
            raise ValueError(
                f'{config.name}: set the "arm" attribute to the name of the '
                f"{NAMESPACE}:{FAMILY}:arm component this vacuum gripper is attached to"
            )
        if config.HasField("frame") and config.frame.parent.split(":")[0] != arm:
            raise ValueError(
                f"{config.name}: frame.parent must be the arm {arm!r} (the frame is the "
                f"vacuum gripper's TCP in the arm's tool frame), got {config.frame.parent!r}"
            )
        if "grab_delay_ms" in attrs:
            grab_delay_ms = attrs["grab_delay_ms"]
            if not isinstance(grab_delay_ms, (int, float)) or grab_delay_ms < 0:
                raise ValueError(
                    f'{config.name}: "grab_delay_ms" must be a non-negative number of '
                    f"milliseconds, got {grab_delay_ms!r}"
                )
        return [world, arm], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        self._attrs = attrs
        self._kinematics = None
        self._grab_delay_ms = int(attrs.get("grab_delay_ms", DEFAULT_GRAB_DELAY_MS))
        self._handle = SimManager.get().create_vacuum_gripper(self.name, attrs)

    async def close(self) -> None:
        """Release the handle (hooks, callbacks). The prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> VacuumGripperHandle:
        if self._handle is None:
            raise RuntimeError(f"vacuum gripper {self.name} is not attached to the sim")
        return self._handle

    async def open(self, *, timeout: float | None = None, **kwargs) -> None:
        """Releases the cup. There is no travel to wait out, so this returns
        as soon as the release is commanded."""
        await asyncio.to_thread(self._h().open)

    async def grab(self, *, timeout: float | None = None, **kwargs) -> bool:
        """Engages the cup, waits out grab_delay_ms while suction builds,
        then reports is_holding()."""
        handle = self._h()
        await asyncio.to_thread(handle.grab)
        if self._grab_delay_ms > 0:
            await asyncio.sleep(self._grab_delay_ms / 1000.0)
        return await asyncio.to_thread(handle.is_holding)

    async def is_holding_something(
        self, *, timeout: float | None = None, **kwargs
    ) -> Gripper.HoldingStatus:
        handle = self._h()
        is_holding = await asyncio.to_thread(handle.is_holding)
        is_engaged = await asyncio.to_thread(handle.is_engaged)
        return Gripper.HoldingStatus(
            is_holding_something=is_holding,
            meta={"engaged": is_engaged, "holding": is_holding},
        )

    async def stop(self, *, timeout: float | None = None, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self, *, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            sva = _vacuum_sva(self.name, VACUUM_BOX_MM, VACUUM_BOX_CENTER_Z_MM)
            self._kinematics = (
                KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA,
                json.dumps(sva).encode(),
            )
        return self._kinematics

    async def get_current_inputs(self, *, timeout: float | None = None, **kwargs) -> list[float]:
        """The cup's commanded state, not whether it caught anything - the
        same reading the parallel-jaw model gives, where a jaw closed on
        nothing still reports its commanded position."""
        is_engaged = await asyncio.to_thread(self._h().is_engaged)
        return [1.0 if is_engaged else 0.0]

    async def go_to_inputs(
        self, values: list[float], *, timeout: float | None = None, **kwargs
    ) -> None:
        """values holds exactly one number in [0, 1]. The mechanism is
        binary, so the range is quantised: a value at or above 0.5 engages
        the cup, below 0.5 releases it - there is no partial engagement."""
        if len(values) != 1:
            raise ViamGRPCError(
                f"vacuum gripper {self.name}: go_to_inputs expects exactly one value in "
                f"[0, 1], got {len(values)}",
                Status.INVALID_ARGUMENT,
            )
        value = values[0]
        if not 0.0 <= value <= 1.0:
            raise ViamGRPCError(
                f"vacuum gripper {self.name}: go_to_inputs value must be in [0, 1], got {value}",
                Status.INVALID_ARGUMENT,
            )

        handle = self._h()
        if value >= ENGAGE_INPUT_THRESHOLD:
            await asyncio.to_thread(handle.grab)
        else:
            await asyncio.to_thread(handle.open)

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return [
            Geometry(
                center=Pose(x=0, y=0, z=VACUUM_BOX_CENTER_Z_MM, o_x=0, o_y=0, o_z=1, theta=0),
                box=RectangularPrism(
                    dims_mm=Vector3(x=VACUUM_BOX_MM[0], y=VACUUM_BOX_MM[1], z=VACUUM_BOX_MM[2])
                ),
                label=self.name,
            )
        ]

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        """No sim-only verbs live here. The world's DoCommand answers
        dof_names/tcp_pose so a real driver's do_command never grows a verb
        it can't answer."""
        raise MethodNotImplementedError("do_command")
