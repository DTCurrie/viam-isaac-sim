"""The simulated vacuum-cup gripper model."""

from __future__ import annotations

import asyncio
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
from ..asset_catalog import CUP_APPROACH_GAP_MM, EPICK
from ..epick import epick_render_solids
from ..handles.vacuum import DEFAULT_GRAB_DELAY_MS, VacuumGripperHandle
from ..sim_manager import SimManager
from ..surface_gripper import (
    DEFAULT_COAXIAL_FORCE_LIMIT_N,
    DEFAULT_CUP_DAMPING_N_S_PER_M,
    DEFAULT_CUP_STIFFNESS_N_PER_M,
    DEFAULT_MAX_GRIP_DISTANCE_MM,
    DEFAULT_RETRY_INTERVAL_S,
    DEFAULT_SHEAR_FORCE_LIMIT_N,
)
from .component_frame_pose import get_attrs

# A vacuum cup is a binary actuator: it is engaged or it is not, so the SDK's
# continuous [0, 1] input range only ever lands on these two points. Anything
# at or above this counts as commanding "engaged".
ENGAGE_INPUT_THRESHOLD = 0.5

# The numeric attributes validate_config checks are non-negative numbers, each
# paired with the attribute's own default.
_NUMERIC_ATTR_DEFAULTS: dict[str, float] = {
    "tcp_offset_m": float(EPICK["tcp_offset_m"]),
    "grab_delay_ms": float(DEFAULT_GRAB_DELAY_MS),
    "coaxial_force_limit_n": DEFAULT_COAXIAL_FORCE_LIMIT_N,
    "shear_force_limit_n": DEFAULT_SHEAR_FORCE_LIMIT_N,
    "max_grip_distance_mm": DEFAULT_MAX_GRIP_DISTANCE_MM,
    "retry_interval_s": DEFAULT_RETRY_INTERVAL_S,
    "cup_stiffness": DEFAULT_CUP_STIFFNESS_N_PER_M,
    "cup_damping": DEFAULT_CUP_DAMPING_N_S_PER_M,
}


class IsaacVacuum(Gripper, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    """viam:isaac-sim-devin:vacuum, the Robotiq EPick riding an arm, held
    through Isaac's surface gripper. It serves the ordinary Viam Gripper API
    over the vacuum handle, so a client can't tell it from the real driver
    and a machine config written for hardware runs unedited against the sim.

    Frame: as models/gripper.py's IsaacGripper, the vacuum's frame is its
    TCP, the plane the four cups share 196 mm from the flange, so the motion
    service plans that plane onto the block:

        "frame": {"parent": "<arm>", "translation": {"x": 0, "y": 0, "z": <tcp_offset_m * 1000>}}

    Unlike a mounted camera, the frame does NOT place the prim: the tool
    bolts to parent_prim at local_position / local_orientation_rpy_deg, and
    the frame's translation is the TCP the planner uses. validate_config
    requires frame.parent == arm.

    API mapping (viam-sdk Gripper, all eight abstract methods). The hold is
    contact-gated, not commanded: closing the gripper raycasts from each cup
    for whatever the approach put under it, and the hold breaks on its own,
    with no command from here, once a carried load exceeds the coaxial or
    shear force limit.
      open()                            -> releases the cups, then waits out the handle's
                                            own release delay (the EPick's vacuum bleed-down,
                                            zero on the mock)
      stop()                            -> releases the cups like the real driver's GTO 0,
                                            with no delay
      is_moving()                       -> True while a commanded grab is still within its
                                            grab_delay_ms window, False otherwise
      grab() -> bool                    -> engage the cups, wait grab_delay_ms, return is_holding()
      is_holding_something()            -> HoldingStatus(is_holding, meta with the gripper's
                                            own status, its gripped object paths, whether it
                                            is holding, whether it was last commanded engaged,
                                            and the coaxial monitor's coaxial_load_n,
                                            peak_coaxial_load_n, released_load_n,
                                            coaxial_monitor)
      get_current_inputs() / go_to_inputs([v])  -> one value in [0, 1]: >= 0.5 engages,
                                            < 0.5 releases - the only two points a binary
                                            actuator has on that range
      get_kinematics()                 -> the EPick's own kinematics document, the same file
                                            its real driver serves
      get_geometries()                 -> the EPick's six render solids as boxes, since
                                            Viam's wire carries no cylinder geometry
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
          tcp_offset_m (float)             - flange -> cup plane along tool +Z, default the
                                            EPick's own 0.196 m reach
          grab_delay_ms (int)              - how long a commanded grab takes to build suction,
                                            default the EPick manual's 150 ms gripping time.
                                            is_moving() reports True for this window
          coaxial_force_limit_n (float)    - the pull on any one cup, read by the module off
                                            its bellows' stretch and averaged over 0.1 s, past
                                            which the module opens the gripper. 0 turns the
                                            check off. Default the cup's holding force at the
                                            manual's 80 % maximum vacuum, 153 N. The plugin's
                                            own one-step coaxial check is off; shear stays with
                                            the plugin (see shear_force_limit_n)
          shear_force_limit_n (float)      - sideways load that breaks the hold, default half
                                            the cups' holding force in friction summed over
                                            the four cups, 306 N
          max_grip_distance_mm (float)     - how far past its clearance the cup's raycast
                                            looks for a payload, default 15, must exceed the
                                            approach gap the clearance is set to
          retry_interval_s (float)         - how long the automatic mode retries for vacuum
                                            before giving up, default the manual's 2.0 s
          cup_stiffness (float, N/m)       - the bellows' spring rate along the cup axis,
                                            default 5000
          cup_damping (float, N s/m)       - the bellows' damping along the cup axis,
                                            default 20, low because the gripper reads the
                                            damper's force at a motion onset as load
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
        for attr_name, default in _NUMERIC_ATTR_DEFAULTS.items():
            if attr_name not in attrs:
                continue
            value = attrs[attr_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f'{config.name}: "{attr_name}" must be a non-negative number, '
                    f"default {default!r}, got {value!r}"
                )
        max_grip_distance_mm = attrs.get("max_grip_distance_mm", DEFAULT_MAX_GRIP_DISTANCE_MM)
        if (
            isinstance(max_grip_distance_mm, (int, float))
            and not isinstance(max_grip_distance_mm, bool)
            and max_grip_distance_mm <= CUP_APPROACH_GAP_MM
        ):
            raise ValueError(
                f'{config.name}: "max_grip_distance_mm" must exceed the {CUP_APPROACH_GAP_MM} mm '
                "approach gap the cup's clearance is set to, or the raycast has no length left "
                f"once it clears the gap, got {max_grip_distance_mm!r}"
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
        """Releases the cups, then waits out the handle's own release delay,
        the EPick's vacuum bleed-down. The mock's release delay is zero."""
        handle = self._h()
        await asyncio.to_thread(handle.open)
        if handle.release_delay_s > 0:
            await asyncio.sleep(handle.release_delay_s)

    async def grab(self, *, timeout: float | None = None, **kwargs) -> bool:
        """Engages the cups, waits out grab_delay_ms while suction builds,
        then reports is_holding()."""
        handle = self._h()
        await asyncio.to_thread(handle.grab)
        if self._grab_delay_ms > 0:
            await asyncio.sleep(self._grab_delay_ms / 1000.0)
        return await asyncio.to_thread(handle.is_holding)

    async def is_holding_something(
        self, *, timeout: float | None = None, **kwargs
    ) -> Gripper.HoldingStatus:
        """meta carries the gripper's own status and gripped object paths,
        whether it is holding, whether it was last commanded engaged, and the
        handle's coaxial monitor reading (coaxial_load_n, peak_coaxial_load_n,
        released_load_n and the monitor's state under coaxial_monitor),
        zero/zero/None/off on the mock, which has no bellows to read."""
        handle = self._h()
        status, gripped_objects = await asyncio.to_thread(handle.gripper_status)
        is_holding = status == "Closed" and len(gripped_objects) > 0
        is_engaged = await asyncio.to_thread(handle.is_engaged)
        hold = await asyncio.to_thread(handle.hold_load)
        return Gripper.HoldingStatus(
            is_holding_something=is_holding,
            meta={
                "engaged": is_engaged,
                "holding": is_holding,
                "status": status,
                "gripped_objects": gripped_objects,
                "coaxial_load_n": hold.coaxial_load_n,
                "peak_coaxial_load_n": hold.peak_coaxial_load_n,
                "released_load_n": hold.released_load_n,
                "coaxial_monitor": hold.monitor,
            },
        )

    async def stop(self, *, timeout: float | None = None, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self, *, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        """Serves the EPick's own vendored kinematics document verbatim, the
        same file its real driver serves from GetKinematics, so a plan built
        against the sim is a plan built against the file the real gripper
        answers with too."""
        if self._kinematics is None:
            self._kinematics = (
                KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA,
                EPICK["kinematics_path"].read_bytes(),
            )
        return self._kinematics

    async def get_current_inputs(self, *, timeout: float | None = None, **kwargs) -> list[float]:
        """The cups' commanded state, not whether they caught anything - the
        same reading the parallel-jaw model gives, where a jaw closed on
        nothing still reports its commanded position."""
        is_engaged = await asyncio.to_thread(self._h().is_engaged)
        return [1.0 if is_engaged else 0.0]

    async def go_to_inputs(
        self, values: list[float], *, timeout: float | None = None, **kwargs
    ) -> None:
        """values holds exactly one number in [0, 1]. The mechanism is
        binary, so the range is quantised: a value at or above 0.5 engages
        the cups, below 0.5 releases them - there is no partial engagement."""
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
                center=Pose(
                    x=solid.center_mm[0],
                    y=solid.center_mm[1],
                    z=solid.center_mm[2],
                    o_x=0,
                    o_y=0,
                    o_z=1,
                    theta=0,
                ),
                box=RectangularPrism(
                    dims_mm=Vector3(x=solid.size_mm[0], y=solid.size_mm[1], z=solid.size_mm[2])
                ),
                label=f"{self.name}:{solid.name}",
            )
            for solid in epick_render_solids()
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
