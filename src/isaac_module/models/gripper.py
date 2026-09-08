"""viam:isaac-sim-devin:gripper - a simulated parallel-jaw gripper riding an arm.

Attributes:
  world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
  arm (string, required)          - name of the viam:isaac-sim-devin:arm it is bolted to
  asset (string)                  - known gripper asset, default "robotiq_2f_85"
  parent_prim (string)            - link it is bolted to, default <arm prim>/wrist_3_link
  local_position ([x,y,z] m)      - mount pose of the gripper's base_link on parent_prim
  local_orientation_rpy_deg       - (defaults: identity - the 2F-85 base sits on the flange)
  tcp_offset_m (float)            - flange -> tool centre point along tool +Z, default 0.134
                                    = the fingertip pad centre measured on the GPU. The
                                    pads span 115-153 mm and the published 115 mm spec
                                    value is their near edge.
  open_deg / closed_deg (float)   - drive-joint angles for open / fully closed; defaults 0
                                    and the Isaac-release value from compat.caps()
                                    (47 on 5.0, 45 on 4.5)
  grab_timeout_sec (float)        - how long grab() waits for a stall or full closure, default 5
  holding_tolerance_deg (float)   - commanded-vs-measured gap that counts as holding, default 2
  mock_object_width_m (float)     - mock only: width of the object between the jaws
                                    (unset = nothing to grab, so grab() returns False)

Frame - the gripper's frame is its TCP, so the motion service plans the TCP
(not the flange) onto the block:

    "frame": {"parent": "<arm>", "translation": {"x": 0, "y": 0, "z": <tcp_offset_m * 1000>}}

Unlike a mounted camera, the frame does NOT place the prim: base_link bolts to
parent_prim at local_position / local_orientation_rpy_deg, and the frame's
translation is the TCP the planner uses. validate_config requires
frame.parent == arm.

API mapping (viam-sdk 0.80.0 Gripper, all eight abstract methods):
  open / stop / is_moving          -> the handle
  grab() -> bool                   -> close, wait <= grab_timeout_sec for stall-or-closed,
                                      return is_holding()
  is_holding_something()           -> HoldingStatus(is_holding, meta={jaw angles, degrees})
  get_current_inputs() / go_to_inputs([v])  -> one value in [0, 1]: 0 = open, 1 = closed
  get_kinematics()                 -> 1-link / 0-joint SVA whose link is the 36 x 146 x 153 mm
                                      gripper box spanning flange to fingertips, centred 57.5 mm
                                      behind the TCP (a gripper whose Kinematics
                                      fails is silently dropped from the frame system)
  get_geometries()                 -> that same single box (rdk keeps only [0])
  close()                          -> SimManager.release_handle
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.gripper import Gripper
from viam.errors import ViamGRPCError
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
from ..sim_manager import KNOWN_ASSETS, GripperHandle, SimManager, _prim_name
from .utils import get_attrs

DEFAULT_GRIPPER_ASSET = "robotiq_2f_85"
DEFAULT_TCP_OFFSET_M = float(KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["tcp_offset_m"])
DEFAULT_GRAB_TIMEOUT_S = 5.0
GRAB_POLL_INTERVAL_S = 1.0 / 120.0  # matches MockArmHandle.STEP_S; the sim's "physics step"
JAW_CLOSED_TOLERANCE_RAD = 1e-3
JAW_BOX_MM: tuple[float, float, float] = KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["jaw_box_mm"]
# The box spans flange -> fingertips; in the gripper frame (origin = TCP) its
# centre sits reach/2 - tcp behind the TCP (measured: 76.5 - 134 = -57.5 mm).
GRIPPER_BOX_CENTRE_Z_MM = (
    KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["fingertip_reach_m"] / 2.0 - DEFAULT_TCP_OFFSET_M
) * 1000.0


def default_parent_prim(arm_name: str) -> str:
    """The flange link of the arm's prim, matching SimManager's prim naming."""
    return f"/World/{_prim_name(arm_name)}/wrist_3_link"


def _gripper_sva(
    link_id: str, box_mm: tuple[float, float, float], box_centre_z_mm: float
) -> dict[str, Any]:
    """The gripper's kinematics: one link, no joints, whose geometry is the
    box_mm RectangularPrism spanning flange to fingertips - centred
    box_centre_z_mm along the tool axis from the TCP, the frame origin."""
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
                    "translation": {"x": 0, "y": 0, "z": box_centre_z_mm},
                },
            }
        ],
        "joints": [],
    }


class IsaacGripper(Gripper, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "gripper")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: GripperHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._grab_timeout = DEFAULT_GRAB_TIMEOUT_S
        self._tcp_offset_m = DEFAULT_TCP_OFFSET_M
        self._kinematics: tuple[KinematicsFileFormat.ValueType, bytes] | None = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        gripper = cls(config.name)
        gripper.reconfigure(config, dependencies)
        return gripper

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        """Requires world + arm; when a frame is set its parent must be the
        arm (the frame is the TCP in the arm's tool frame). Returns both as
        dependencies so viam-server builds the arm before the gripper."""
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
                f"{NAMESPACE}:{FAMILY}:arm component this gripper is attached to"
            )
        if config.HasField("frame") and config.frame.parent.split(":")[0] != arm:
            raise ValueError(
                f"{config.name}: frame.parent must be the arm {arm!r} (the frame is the "
                f"gripper's TCP in the arm's tool frame), got {config.frame.parent!r}"
            )
        return [world, arm], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        attrs.setdefault("asset", DEFAULT_GRIPPER_ASSET)
        attrs.setdefault("parent_prim", default_parent_prim(str(attrs["arm"])))
        self._grab_timeout = float(attrs.get("grab_timeout_sec", DEFAULT_GRAB_TIMEOUT_S))
        self._tcp_offset_m = float(attrs.get("tcp_offset_m", DEFAULT_TCP_OFFSET_M))
        self._attrs = attrs
        self._kinematics = None
        self._handle = SimManager.get().create_gripper(self.name, attrs)

    async def close(self) -> None:
        """Release the handle (hooks, callbacks). The prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> GripperHandle:
        if self._handle is None:
            raise RuntimeError(f"gripper {self.name} is not attached to the sim")
        return self._handle

    async def open(self, **kwargs) -> None:
        handle = self._h()
        await asyncio.to_thread(handle.open)

        deadline = time.monotonic() + self._grab_timeout
        while time.monotonic() < deadline and await asyncio.to_thread(handle.is_moving):
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

    async def grab(self, **kwargs) -> bool:
        handle = self._h()
        await asyncio.to_thread(handle.close)

        deadline = time.monotonic() + self._grab_timeout
        while time.monotonic() < deadline and await asyncio.to_thread(handle.is_moving):
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

        _, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        while time.monotonic() < deadline:
            if await asyncio.to_thread(handle.is_holding):
                return True
            jaw = await asyncio.to_thread(handle.get_jaw)
            if abs(jaw - closed_rad) <= JAW_CLOSED_TOLERANCE_RAD:
                break
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

        return await asyncio.to_thread(handle.is_holding)

    async def is_holding_something(self, **kwargs) -> Gripper.HoldingStatus:
        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        jaw_rad = await asyncio.to_thread(handle.get_jaw)
        is_holding = await asyncio.to_thread(handle.is_holding)
        span = closed_rad - open_rad
        input_value = (jaw_rad - open_rad) / span if span else 0.0
        return Gripper.HoldingStatus(
            is_holding_something=is_holding,
            meta={
                "jaw_deg": math.degrees(jaw_rad),
                "open_deg": math.degrees(open_rad),
                "closed_deg": math.degrees(closed_rad),
                "input": min(max(input_value, 0.0), 1.0),
            },
        )

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            sva = _gripper_sva(self.name, JAW_BOX_MM, GRIPPER_BOX_CENTRE_Z_MM)
            self._kinematics = (
                KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA,
                json.dumps(sva).encode(),
            )
        return self._kinematics

    async def get_current_inputs(self, **kwargs) -> list[float]:
        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        jaw_rad = await asyncio.to_thread(handle.get_jaw)
        span = closed_rad - open_rad
        value = (jaw_rad - open_rad) / span if span else 0.0
        return [min(max(value, 0.0), 1.0)]

    async def go_to_inputs(self, values: list[float], **kwargs) -> None:
        if len(values) != 1:
            raise ViamGRPCError(
                f"gripper {self.name}: go_to_inputs expects exactly one value in [0, 1], "
                f"got {len(values)}",
                Status.INVALID_ARGUMENT,
            )
        value = values[0]
        if not 0.0 <= value <= 1.0:
            raise ViamGRPCError(
                f"gripper {self.name}: go_to_inputs value must be in [0, 1], got {value}",
                Status.INVALID_ARGUMENT,
            )

        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        await asyncio.to_thread(handle.set_jaw, open_rad + value * (closed_rad - open_rad))

        deadline = time.monotonic() + self._grab_timeout
        while time.monotonic() < deadline and await asyncio.to_thread(handle.is_moving):
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return [
            Geometry(
                center=Pose(x=0, y=0, z=GRIPPER_BOX_CENTRE_Z_MM, o_x=0, o_y=0, o_z=1, theta=0),
                box=RectangularPrism(
                    dims_mm=Vector3(x=JAW_BOX_MM[0], y=JAW_BOX_MM[1], z=JAW_BOX_MM[2])
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
        # No sim-only verbs live here: the world's DoCommand answers
        # dof_names/jaw_deg/tcp_pose so a real driver's do_command never
        # grows a verb it can't answer.
        raise ValueError(f"unknown command: {command}")
