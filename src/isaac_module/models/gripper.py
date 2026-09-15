"""The simulated parallel-jaw gripper model."""

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
from ..sim_manager import KNOWN_ASSETS, JawGripperHandle, SimManager, prim_name
from .component_frame_pose import get_attrs

DEFAULT_GRIPPER_ASSET = "robotiq_2f_85"
DEFAULT_TCP_OFFSET_M = float(KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["tcp_offset_m"])
DEFAULT_GRAB_TIMEOUT_S = 5.0
GRAB_POLL_INTERVAL_S = 1.0 / 120.0  # matches MockArmHandle.STEP_S, the sim's "physics step"
JAW_CLOSED_TOLERANCE_RAD = 1e-3
JAW_BOX_MM: tuple[float, float, float] = KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["jaw_box_mm"]
# The box spans flange -> fingertips. In the gripper frame (origin = TCP) its
# center sits reach/2 - tcp behind the TCP (measured: 76.5 - 134 = -57.5 mm).
GRIPPER_BOX_CENTER_Z_MM = (
    KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["fingertip_reach_m"] / 2.0 - DEFAULT_TCP_OFFSET_M
) * 1000.0


class GripperMoveTimeoutError(ViamGRPCError, TimeoutError):
    """open/grab/go_to_inputs did not settle before the SDK's timeout= kwarg
    cut the wait short of grab_timeout_sec."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.DEADLINE_EXCEEDED)
        Exception.__init__(self, message)


def default_parent_prim(arm_name: str, arm_asset: str | None) -> str:
    """The flange link of the arm's prim, matching SimManager's prim naming.
    Raises when arm_asset declares no ee_prim, since there's then no known
    attachment point to fall back to and a caller must set parent_prim."""
    ee_prim = (
        KNOWN_ASSETS[arm_asset].get("ee_prim") if arm_asset and arm_asset in KNOWN_ASSETS else None
    )
    if not ee_prim:
        raise ValueError(
            f"arm asset {arm_asset!r} has no known end-effector prim; "
            'set "parent_prim" on the gripper explicitly'
        )
    return f"/World/{prim_name(arm_name)}/{ee_prim}"


def _gripper_sva(
    link_id: str, box_mm: tuple[float, float, float], box_center_z_mm: float
) -> dict[str, Any]:
    """The gripper's kinematics: one link, no joints, whose geometry is the
    box_mm RectangularPrism spanning flange to fingertips - centered
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


class IsaacGripper(Gripper, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    """viam:isaac-sim-devin:gripper, a simulated parallel-jaw gripper riding an arm.

    Frame: the gripper's frame is its TCP, so the motion service plans the TCP
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
                                          gripper box spanning flange to fingertips, centered
                                          57.5 mm behind the TCP (a gripper whose Kinematics
                                          fails is silently dropped from the frame system)
      get_geometries()                 -> that same single box (rdk keeps only [0])
      close()                          -> SimManager.release_handle

    open/grab/go_to_inputs wait bounded by grab_timeout_sec, capped by the
    SDK's timeout= kwarg; a dropped RPC during that wait holds the jaw at its
    current position instead of continuing to close unwatched.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "gripper")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: JawGripperHandle | None = None
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
        """Requires world + arm. When a frame is set its parent must be the
        arm (the frame is the TCP in the arm's tool frame). Returns both as
        dependencies so viam-server builds the arm before the gripper.

        Attributes:
          world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
          arm (string, required)          - name of the viam:isaac-sim-devin:arm it is bolted to
          asset (string)                  - known gripper asset, default "robotiq_2f_85"
          parent_prim (string)            - link it is bolted to, default the riding arm's
                                            ee_prim (<arm prim>/wrist_3_link for a known UR
                                            asset). Raises at reconfigure if the arm's asset
                                            declares no ee_prim and parent_prim isn't set
          local_position ([x,y,z] m)      - mount pose of the gripper's base_link on parent_prim
          local_orientation_rpy_deg       - (defaults: identity - the 2F-85 base sits on the flange)
          tcp_offset_m (float)            - flange -> tool center point along tool +Z, default 0.134
                                            = the fingertip pad center measured on the GPU. The
                                            pads span 115-153 mm and the published 115 mm spec
                                            value is their near edge.
          open_deg / closed_deg (float)   - drive-joint angles for open / fully closed. Defaults 0
                                            and the Isaac-release value from compat.caps()
                                            (47 on 5.0, 45 on 4.5)
          grab_timeout_sec (float)        - how long grab() waits for a stall or full closure,
                                            default 5
          holding_tolerance_deg (float)   - commanded-vs-measured gap that counts as holding,
                                            default 2
          mock_object_width_m (float)     - mock only: width of the object between the jaws
                                            (unset = nothing to grab, so grab() returns False)
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
        if "parent_prim" not in attrs:
            arm_name = str(attrs["arm"])
            arm_attrs, _ = SimManager.get().handle_entry(arm_name)
            attrs["parent_prim"] = default_parent_prim(arm_name, arm_attrs.get("asset"))
        self._grab_timeout = float(attrs.get("grab_timeout_sec", DEFAULT_GRAB_TIMEOUT_S))
        self._tcp_offset_m = float(attrs.get("tcp_offset_m", DEFAULT_TCP_OFFSET_M))
        self._attrs = attrs
        self._kinematics = None
        self._handle = SimManager.get().create_gripper(self.name, attrs)

    async def close(self) -> None:
        """Release the handle (hooks, callbacks). The prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> JawGripperHandle:
        if self._handle is None:
            raise RuntimeError(f"gripper {self.name} is not attached to the sim")
        return self._handle

    def _deadline_s(self, timeout: float | None) -> float:
        """grab_timeout_sec, capped by the SDK's timeout= kwarg when given."""
        return self._grab_timeout if timeout is None else min(self._grab_timeout, timeout)

    async def open(self, *, timeout: float | None = None, **kwargs) -> None:
        handle = self._h()
        await asyncio.to_thread(handle.open)

        deadline = time.monotonic() + self._deadline_s(timeout)
        while time.monotonic() < deadline:
            moving, _holding = await asyncio.to_thread(handle.poll_state)
            if not moving:
                return
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

        if timeout is not None and timeout < self._grab_timeout:
            raise GripperMoveTimeoutError(
                f"gripper {self.name}: open did not settle within {timeout:.2f}s"
            )

    async def grab(self, *, timeout: float | None = None, **kwargs) -> bool:
        """Closes the jaw and waits (bounded by grab_timeout_sec, capped by
        the SDK's timeout= kwarg) for a stall or full closure. A dropped RPC
        holds the jaw at its current position instead of continuing to close
        unwatched. When the SDK's timeout= (not grab_timeout_sec on its own)
        is why the wait ends before the jaw settles, raises
        GripperMoveTimeoutError instead of guessing at a result."""
        handle = self._h()
        await asyncio.to_thread(handle.grab)

        deadline = time.monotonic() + self._deadline_s(timeout)
        try:
            while time.monotonic() < deadline:
                moving, _holding = await asyncio.to_thread(handle.poll_state)
                if not moving:
                    break
                await asyncio.sleep(GRAB_POLL_INTERVAL_S)

            _, closed_rad = await asyncio.to_thread(handle.jaw_limits)
            while time.monotonic() < deadline:
                jaw, _moving, holding = await asyncio.to_thread(handle.poll_jaw_state)
                if holding:
                    return True
                if abs(jaw - closed_rad) <= JAW_CLOSED_TOLERANCE_RAD:
                    return False
                await asyncio.sleep(GRAB_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            await asyncio.to_thread(handle.stop)
            raise

        if timeout is not None and timeout < self._grab_timeout:
            raise GripperMoveTimeoutError(
                f"gripper {self.name}: grab did not settle within {timeout:.2f}s"
            )
        return await asyncio.to_thread(handle.is_holding)

    async def is_holding_something(
        self, *, timeout: float | None = None, **kwargs
    ) -> Gripper.HoldingStatus:
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

    async def stop(self, *, timeout: float | None = None, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self, *, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            sva = _gripper_sva(self.name, JAW_BOX_MM, GRIPPER_BOX_CENTER_Z_MM)
            self._kinematics = (
                KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA,
                json.dumps(sva).encode(),
            )
        return self._kinematics

    async def get_current_inputs(self, *, timeout: float | None = None, **kwargs) -> list[float]:
        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        jaw_rad = await asyncio.to_thread(handle.get_jaw)
        span = closed_rad - open_rad
        value = (jaw_rad - open_rad) / span if span else 0.0
        return [min(max(value, 0.0), 1.0)]

    async def go_to_inputs(
        self, values: list[float], *, timeout: float | None = None, **kwargs
    ) -> None:
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

        deadline = time.monotonic() + self._deadline_s(timeout)
        while time.monotonic() < deadline:
            moving, _holding = await asyncio.to_thread(handle.poll_state)
            if not moving:
                return
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

        if timeout is not None and timeout < self._grab_timeout:
            raise GripperMoveTimeoutError(
                f"gripper {self.name}: go_to_inputs did not settle within {timeout:.2f}s"
            )

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return [
            Geometry(
                center=Pose(x=0, y=0, z=GRIPPER_BOX_CENTER_Z_MM, o_x=0, o_y=0, o_z=1, theta=0),
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
        """No sim-only verbs live here. The world's DoCommand answers
        dof_names/jaw_deg/tcp_pose so a real driver's do_command never grows a
        verb it can't answer."""
        raise MethodNotImplementedError("do_command")
