"""viam:isaac-sim-devin:palletizer - the generic service that runs one pick
and place: lift a box off the world and set it at a configured place pose.
Nothing is measured and nothing is planned yet - later phases replace one
assumption at a time (a detected box pose, a chosen slot from the sequencer,
a multi-box loop), so this stays the thinnest path from a DoCommand to a box
placed somewhere.

The box's pick pose is read from the world at run time rather than assumed:
this service asks for ``prop_geometries`` and grasps wherever ``box_prop``
currently sits. The place pose is configured, since nothing in this phase
yet computes it. Neither pose is packing geometry this service owns.

Attributes:
  world (string, default "isaac-world") - name of the viam:isaac-sim-devin:world component
  arm (string, required)     - name of the arm component (boot ordering only; every
                                motion goes through "motion")
  gripper (string, required) - name of the gripper component (driven only through the
                                Viam Gripper API: open/grab/is_holding_something, so
                                either a jaw or a vacuum gripper model works unchanged)
  motion (string, required)  - name of the motion service ("builtin" works)
  box_prop (string, required) - prop name of the box to pick, wherever it currently
                                 sits in the world
  place_pose_mm (mapping, required) - {"x", "y", "z"} the gripper TCP descends to for
                                       release, in millimetres

DoCommand:
  {"command": "start"} -> runs one pick and place. {"ok": true, "state": "running"}, or
    {"ok": false, "state": "running"} unchanged when already running |
  {"command": "stop"} -> cancels between motions, never mid-motion. {"ok": true} |
  {"command": "status"} -> {"state": "idle|running|stopping|complete|failed",
    "records": [PalletizerRecord.to_dict(), ...]}

The sequence, every motion through the motion service, never through the arm
directly: move above the box at a standoff, descend onto its top face
(straight line), grab() on the gripper - a False return fails the record with
a reason - lift (straight line), move above the place pose, descend to it
(straight line), open(), retreat.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from typing_extensions import Self
from viam.components.arm import Arm
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName, WorldState
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import MotionClient
from viam.utils import ValueTypes, struct_to_dict

from pickcell.movers import RealMover
from pickcell.obstacles import obstacles_from_prop_geometries, support_obstacle, world_state
from pickcell.pipeline import GripperApi, Mover, WorldApi
from pickcell.poses import PRE_GRASP_STANDOFF_MM, _pointing_down

from .. import DEFAULT_WORLD_NAME, FAMILY, NAMESPACE
from ..sort_plan import OUTCOME_FAILED, OUTCOME_PLACED

LOGGER = getLogger(__name__)

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_COMPLETE = "complete"
STATE_FAILED = "failed"

_DEPENDENCY_ATTRS = ("world", "arm", "gripper", "motion")
_PLACE_POSE_AXES = ("x", "y", "z")

# the floor everything else stands on, so a discrete collision check cannot
# step a link through it mid-swing (the same reasoning as
# pickcell.obstacles.support_obstacle's own docstring)
_FLOOR_Z_MM = 0.0

# the cup stops this far above a payload's top face rather than on it. Driving
# a rigid tool onto a rigid box makes contact the arm cannot push through, and
# it stalls short of its commanded pose. A vacuum mechanism constant, not
# packing geometry: it holds regardless of which box or which cell this runs in.
CUP_APPROACH_GAP_MM = 5.0


def _box_top_face_xyz_mm(
    geometries: Sequence[Mapping[str, Any]], box_prop: str
) -> tuple[float, float, float] | None:
    """``box_prop``'s current (x, y, top-face z) in the world, from a
    ``prop_geometries`` reading, or None when the prop is not there."""
    for geometry in geometries:
        if geometry.get("name") != box_prop:
            continue
        pose = geometry["pose_in_world_mm"]
        _dim_x, _dim_y, dim_z = geometry["box_dims_mm"]
        return (float(pose["x"]), float(pose["y"]), float(pose["z"]) + float(dim_z) / 2.0)
    return None


def pick_grasp_pose(top_face_xyz_mm: tuple[float, float, float]) -> Pose:
    """The gripper TCP pose just above the box's top face, pointing straight
    down. CUP_APPROACH_GAP_MM short of the face rather than on it, since a cup
    driven onto a box stalls the arm against it."""
    x, y, top_face_z = top_face_xyz_mm
    return _pointing_down(x, y, top_face_z + CUP_APPROACH_GAP_MM)


def pick_grasp_standoff_pose(top_face_xyz_mm: tuple[float, float, float]) -> Pose:
    """The stationary pose above the box the arm descends from and lifts
    back to, by PRE_GRASP_STANDOFF_MM above its top face."""
    x, y, top_face_z = top_face_xyz_mm
    return _pointing_down(x, y, top_face_z + CUP_APPROACH_GAP_MM + PRE_GRASP_STANDOFF_MM)


def place_release_pose(place_pose_mm: Mapping[str, float]) -> Pose:
    """The gripper TCP pose to release the box at, from the configured
    ``place_pose_mm``."""
    return _pointing_down(place_pose_mm["x"], place_pose_mm["y"], place_pose_mm["z"])


def place_release_standoff_pose(place_pose_mm: Mapping[str, float]) -> Pose:
    """The stationary pose above the place pose the arm descends from and
    retreats to, by PRE_GRASP_STANDOFF_MM above the release height."""
    return _pointing_down(
        place_pose_mm["x"], place_pose_mm["y"], place_pose_mm["z"] + PRE_GRASP_STANDOFF_MM
    )


@dataclass(frozen=True)
class PalletizerRecord:
    """One pick and place's terminal outcome. This service has no colour
    (boxes are cardboard, identified by prop name), so unlike the
    conductor's ``PickRecord`` this carries a ``place_pose_mm`` rather than a
    ``color``."""

    box_prop: str
    place_pose_mm: Mapping[str, float]
    outcome: str  # one of the OUTCOME_* literals from sort_plan
    duration_s: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "box_prop": self.box_prop,
            "place_pose_mm": dict(self.place_pose_mm),
            "outcome": self.outcome,
            "duration_s": self.duration_s,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        return result


def _outcome_record(
    box_prop: str,
    place_pose_mm: Mapping[str, float],
    outcome: str,
    duration_s: float,
    reason: str | None = None,
) -> PalletizerRecord:
    """The one pick's terminal outcome."""
    return PalletizerRecord(
        box_prop=box_prop,
        place_pose_mm=place_pose_mm,
        outcome=outcome,
        duration_s=duration_s,
        reason=reason,
    )


def _valid_place_pose_mm(config_name: str, attrs: Mapping[str, Any]) -> dict[str, float]:
    pose = attrs.get("place_pose_mm")
    if not isinstance(pose, Mapping) or not all(axis in pose for axis in _PLACE_POSE_AXES):
        raise ValueError(
            f'{config_name}: set "place_pose_mm" to a mapping with x, y and z in millimetres'
        )
    return {axis: float(pose[axis]) for axis in _PLACE_POSE_AXES}


class IsaacPalletizer(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the service, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "palletizer")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._state: str = STATE_IDLE
        self._task: asyncio.Task[None] | None = None
        self._cancel_requested = False
        self._records: list[PalletizerRecord] = []

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        palletizer = cls(config.name)
        palletizer.reconfigure(config, dependencies)
        return palletizer

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        dependencies: list[str] = []
        for key in _DEPENDENCY_ATTRS:
            default = DEFAULT_WORLD_NAME if key == "world" else None
            value = attrs.get(key, default)
            if not value or not isinstance(value, str):
                raise ValueError(f'{config.name}: set the "{key}" attribute to a resource name')
            dependencies.append(value)

        box_prop = attrs.get("box_prop")
        if not box_prop or not isinstance(box_prop, str):
            raise ValueError(f'{config.name}: set the "box_prop" attribute to a prop name')

        _valid_place_pose_mm(config.name, attrs)
        return dependencies, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        attrs.setdefault("world", DEFAULT_WORLD_NAME)
        by_name: dict[str, ResourceBase] = {
            rn.name: resource for rn, resource in dependencies.items()
        }

        def dep(key: str) -> ResourceBase:
            resource_name = str(attrs[key])
            if resource_name not in by_name:
                raise ValueError(
                    f"{config.name}: dependency {resource_name!r} for {key!r} was not resolved"
                )
            return by_name[resource_name]

        self._world = cast("WorldApi", dep("world"))
        self._arm = cast(Arm, dep("arm"))
        self._gripper = cast("GripperApi", dep("gripper"))
        self._gripper_name = str(attrs["gripper"])
        self._motion = cast(MotionClient, dep("motion"))
        self._box_prop = str(attrs["box_prop"])
        self._place_pose_mm = _valid_place_pose_mm(config.name, attrs)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Mapping[str, ValueTypes]:
        cmd = str(command.get("command", ""))
        if cmd == "start":
            return self._handle_start()
        if cmd == "stop":
            return self._handle_stop()
        if cmd == "status":
            return cast("Mapping[str, ValueTypes]", self._status_snapshot())
        raise ValueError(f"unknown command {cmd!r}; supported: start, stop, status")

    def _handle_start(self) -> dict[str, ValueTypes]:
        if self._state in (STATE_RUNNING, STATE_STOPPING):
            return {"ok": False, "state": self._state}
        self._state = STATE_RUNNING
        self._records = []
        self._cancel_requested = False
        self._task = asyncio.create_task(self._run())
        return {"ok": True, "state": "running"}

    def _handle_stop(self) -> dict[str, ValueTypes]:
        if self._state == STATE_RUNNING:
            self._state = STATE_STOPPING
        self._cancel_requested = True
        return {"ok": True}

    def _status_snapshot(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "records": [record.to_dict() for record in self._records],
        }

    async def wait_until_done(self) -> None:
        """Test-only join on the background pick-and-place task started by
        ``start``. Never awaited from ``do_command``: production callers poll
        ``status``."""
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        try:
            record = await self._pick_and_place()
            self._records = [record]
            self._state = STATE_IDLE if self._cancel_requested else STATE_COMPLETE
        except Exception:
            self._state = STATE_FAILED
            LOGGER.exception("palletizer pick and place failed")
        finally:
            self._cancel_requested = False

    def _build_mover(self) -> Mover:
        # RealMover's camera_name only matters for look_from, which this
        # service never calls (there is no scan, the box pose is known) -
        # the gripper name is a harmless stand-in rather than a fourth
        # dependency the config would otherwise have to name.
        return RealMover(self._motion, self._gripper_name, self._gripper_name)

    async def _prop_geometries(self) -> Sequence[Mapping[str, Any]]:
        response = await self._world.do_command({"command": "prop_geometries"})
        return cast("Sequence[Mapping[str, Any]]", response.get("geometries", []))

    def _build_world_state(self, geometries: Sequence[Mapping[str, Any]]) -> WorldState:
        obstacles = obstacles_from_prop_geometries(geometries, {self._box_prop})
        return world_state(None, obstacles, support_obstacle(_FLOOR_Z_MM))

    async def _pick_and_place(self) -> PalletizerRecord:
        start = time.monotonic()
        mover = self._build_mover()
        geometries = await self._prop_geometries()
        state = self._build_world_state(geometries)

        top_face_xyz_mm = _box_top_face_xyz_mm(geometries, self._box_prop)
        if top_face_xyz_mm is None:
            return _outcome_record(
                self._box_prop,
                self._place_pose_mm,
                OUTCOME_FAILED,
                time.monotonic() - start,
                reason=f"{self._box_prop!r} has no known geometry in the world",
            )

        async def move(pose: Pose, linear: bool) -> bool:
            """Moves unless a stop landed since the previous motion. Returns
            whether the move happened, so a caller can bail out on False."""
            if self._cancel_requested:
                return False
            await mover.move_to(pose, state, linear=linear)
            return True

        if not await move(pick_grasp_standoff_pose(top_face_xyz_mm), False):
            return self._stopped_record(start)
        if not await move(pick_grasp_pose(top_face_xyz_mm), True):
            return self._stopped_record(start)

        if self._cancel_requested:
            return self._stopped_record(start)
        if not await self._gripper.grab():
            return _outcome_record(
                self._box_prop,
                self._place_pose_mm,
                OUTCOME_FAILED,
                time.monotonic() - start,
                reason="gripper reported no object grasped",
            )

        if not await move(pick_grasp_standoff_pose(top_face_xyz_mm), True):
            return self._stopped_record(start)
        if not await move(place_release_standoff_pose(self._place_pose_mm), False):
            return self._stopped_record(start)
        if not await move(place_release_pose(self._place_pose_mm), True):
            return self._stopped_record(start)

        if self._cancel_requested:
            return self._stopped_record(start)
        await self._gripper.open()
        await mover.move_to(place_release_standoff_pose(self._place_pose_mm), state, linear=True)

        return _outcome_record(
            self._box_prop, self._place_pose_mm, OUTCOME_PLACED, time.monotonic() - start
        )

    def _stopped_record(self, start: float) -> PalletizerRecord:
        return _outcome_record(
            self._box_prop,
            self._place_pose_mm,
            OUTCOME_FAILED,
            time.monotonic() - start,
            reason="stopped before the next motion",
        )
