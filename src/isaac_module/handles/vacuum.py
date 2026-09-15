from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from viam.logging import getLogger

from ..asset_catalog import VACUUM_TOOL
from ..spatial import Quat, Vec3, pose_in_frame, quat_rotate
from .arm import ArmHandle

if TYPE_CHECKING:
    from ..sim_manager import SimManager

from .gripper import GripperHandle

LOGGER = getLogger(__name__)

DEFAULT_MAX_PAYLOAD_GAP_M = 0.01  # a candidate's top face must be within this of the cup face
# matches viam:robotiq:simulated-epick-vacuum-gripper's grab_delay_ms default: how long a
# real epick takes to build suction after a grab is commanded
DEFAULT_GRAB_DELAY_MS = 1000


def select_payload_for_cup(
    cup_position: Vec3,
    cup_side_m: float,
    max_gap_m: float,
    candidates: Iterable[tuple[str, Vec3, tuple[float, float, float]]],
) -> str | None:
    """The name of the candidate the cup at ``cup_position`` (its contact
    face, tcp_offset_m already applied) would take hold of, or None.

    A candidate (name, world position of its center, box dims in metres)
    qualifies when the gap from its top face (position.z + dims.z / 2) up to
    the cup face is within max_gap_m, AND its XY footprint (dims.x by
    dims.y, centered on its position) overlaps the cup's cup_side_m square
    footprint centered under cup_position. The gap may be slightly negative:
    the cup reaches a payload by descending onto it, so contact offsets and
    numerical error routinely put the cup face a fraction of a millimetre
    below the payload's top face, and that penetrating contact is the one a
    real grab has to recognise. Ties go to the smallest absolute gap (a
    slight penetration never outranks clean contact), then alphabetically by
    name - tuple comparison on (abs(gap), name) gives exactly that ordering,
    so the pick never depends on iteration order."""
    cup_x, cup_y, cup_z = cup_position
    half_cup = cup_side_m / 2.0
    best: tuple[float, str] | None = None
    for candidate_name, position, dims in candidates:
        pos_x, pos_y, pos_z = position
        dim_x, dim_y, dim_z = dims
        top_z = pos_z + dim_z / 2.0
        gap = cup_z - top_z
        if gap < -max_gap_m or gap > max_gap_m:
            continue
        overlaps_x = abs(pos_x - cup_x) < (half_cup + dim_x / 2.0)
        overlaps_y = abs(pos_y - cup_y) < (half_cup + dim_y / 2.0)
        if not (overlaps_x and overlaps_y):
            continue
        key = (abs(gap), candidate_name)
        if best is None or key < best:
            best = key
    return best[1] if best is not None else None


def suction_joint_local_frame(
    anchor_pose: tuple[Vec3, Quat], payload_pose: tuple[Vec3, Quat]
) -> tuple[Vec3, Quat]:
    """The ANCHOR-side local frame of a weld that holds the anchor and the
    payload in the poses they are ALREADY in: the payload's pose expressed in
    the anchor's frame.

    A FixedJoint constrains body0's joint frame onto body1's. Leaving both
    local frames at identity therefore asks PhysX to make the payload's origin
    coincide with the anchor's, and it snaps the payload through the cup to get
    there. A cup picks a box up where the box is, so the relative pose at the
    moment of the weld is the one to record.

    The offset goes on the anchor side rather than the payload side because a
    joint's local frame is expressed in its body's own local space, and this
    repo's boxes are cube prims carrying a non-uniform ``scale``. An offset
    written in a scaled body's local space comes back multiplied by that scale,
    which stretches the weld by hundreds of millimetres and leaves the payload
    trailing the tool instead of riding it. An arm link carries no scale."""
    return pose_in_frame(anchor_pose[0], anchor_pose[1], payload_pose[0], payload_pose[1])


def author_suction_joint(
    usd_physics: Any,
    sdf: Any,
    gf: Any,
    stage: Any,
    joint_path: str,
    anchor_prim_path: str,
    payload_prim_path: str,
    local_pos0: Vec3 = (0.0, 0.0, 0.0),
    local_rot0: Quat = (1.0, 0.0, 0.0, 0.0),
) -> None:
    """Author a UsdPhysics.FixedJoint welding ``anchor_prim_path`` to
    ``payload_prim_path``. The payload side stays at identity, so the joint
    frame is the payload's own origin; the anchor side carries the relative
    pose from ``suction_joint_local_frame``. The offset sits on the anchor
    because that body carries no scale, and a joint frame is read in its own
    body's local space."""
    joint = usd_physics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([sdf.Path(anchor_prim_path)])
    joint.CreateBody1Rel().SetTargets([sdf.Path(payload_prim_path)])
    joint.CreateLocalPos0Attr(gf.Vec3f(*(float(v) for v in local_pos0)))
    joint.CreateLocalRot0Attr(
        gf.Quatf(
            float(local_rot0[0]),
            gf.Vec3f(float(local_rot0[1]), float(local_rot0[2]), float(local_rot0[3])),
        )
    )
    joint.CreateLocalPos1Attr(gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(gf.Quatf(1.0, gf.Vec3f(0.0, 0.0, 0.0)))


def remove_suction_joint(stage: Any, joint_path: str) -> bool:
    """Remove the joint prim at joint_path if one is there. Returns whether
    there was one to remove, so a caller can tell a real release from a
    no-op on an already-open cup."""
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        return False
    stage.RemovePrim(joint_path)
    return True


class VacuumGripperHandle(GripperHandle):
    """The suction extension of the core protocol. A cup that ran with nothing
    under it is engaged and holding nothing, which a jaw cannot be, so the two
    states are separate here. A caller that needs the command rather than the
    outcome has to narrow to this type, the same way a jaw caller narrows to
    JawGripperHandle."""

    _grab_delay_s: float = DEFAULT_GRAB_DELAY_MS / 1000.0
    _engaged_at: float | None = None

    def is_engaged(self) -> bool:
        """Whether the cup was last commanded to take hold. True after grab()
        even when nothing was found, False after open()."""
        raise NotImplementedError

    def _start_grab_window(self) -> None:
        """Record that a grab was just commanded, so is_moving() can report
        True until grab_delay_s has passed."""
        self._engaged_at = time.monotonic()

    def _clear_grab_window(self) -> None:
        self._engaged_at = None

    def is_moving(self) -> bool:
        """True while a commanded grab is still within its grab_delay_s
        window, matching the time a real epick takes to build suction."""
        if self._engaged_at is None:
            return False
        return (time.monotonic() - self._engaged_at) < self._grab_delay_s


class IsaacVacuumHandle(VacuumGripperHandle):
    """A suction cup weld its tool prim to whatever is under it with a
    FixedJoint, and lets go by removing that joint. The weld itself is
    authored the moment grab() is called, matching a real epick's suction
    building against whatever the cup already found, but is_moving() still
    reports True for grab_delay_s after that, so a caller watching for the
    grab to settle sees the same window a real epick takes to build
    pressure. Every stage-touching body runs on the sim thread, as in
    IsaacGripperHandle."""

    def __init__(
        self,
        sim: SimManager,
        name: str,
        tool_prim_path: str,
        cup_side_m: float,
        max_payload_gap_m: float,
        cup_face_offset_m: float,
        grab_delay_ms: float = DEFAULT_GRAB_DELAY_MS,
    ) -> None:
        self._sim = sim
        self._name = name
        self._tool_prim_path = tool_prim_path
        self._joint_path = f"{tool_prim_path}/SuctionJoint"
        self._cup_side_m = cup_side_m
        self._max_payload_gap_m = max_payload_gap_m
        # distance from the tool PRIM's origin to the cup face. The tool is a
        # cuboid whose origin is its centre, so this is half its length, not
        # the flange-to-face tcp_offset_m the frame system is given. Using the
        # full offset here puts the believed cup face half a tool-length
        # inside whatever the cup is resting on, and no payload is ever in
        # range.
        self._cup_face_offset_m = cup_face_offset_m
        self._grab_delay_s = float(grab_delay_ms) / 1000.0
        # set by _create_vacuum_gripper_isaac: the link the tool is bolted to
        self.parent_prim_path: str | None = None
        # the payload's prop name grab() last welded, so post_reset can
        # re-author the joint a world reset dropped
        self._held_payload: str | None = None
        # whether the cup was last COMMANDED to take hold, which is not the
        # same as whether it found anything
        self._engaged = False

    def _cup_face_pose(self) -> tuple[Vec3, Quat]:
        """World pose of the cup's contact face: the tool prim's own world
        pose, offset by cup_face_offset_m along its local +Z. Sim thread
        only."""
        pos, quat = self._sim._isaac.SingleXFormPrim(self._tool_prim_path).get_world_pose()
        tool_quat: Quat = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        offset = quat_rotate(tool_quat, (0.0, 0.0, self._cup_face_offset_m))
        face_pos: Vec3 = (
            float(pos[0]) + offset[0],
            float(pos[1]) + offset[1],
            float(pos[2]) + offset[2],
        )
        return face_pos, tool_quat

    def _payload_candidates(self) -> list[tuple[str, Vec3, tuple[float, float, float]]]:
        """Every live, non-fixed prop's (name, world position, box dims) -
        the pool select_payload_for_cup picks from. A fixed prop (a table,
        the pallet deck) is a static collider, and welding the tool to one
        would anchor the arm to the world the same way a static tool prim
        would (see _create_vacuum_gripper_isaac), so it never qualifies as a
        payload. Sim thread only."""
        from ..prop_scatter import prop_box_dims

        candidates: list[tuple[str, Vec3, tuple[float, float, float]]] = []
        for prop_name, spec in self._sim._prop_specs.items():
            if spec.get("fixed"):
                continue
            dims = prop_box_dims(spec)
            pos, _quat = self._sim._isaac.SingleXFormPrim(f"/World/{prop_name}").get_world_pose()
            candidates.append((prop_name, (float(pos[0]), float(pos[1]), float(pos[2])), dims))
        return candidates

    def _weld_anchor_prim_path(self) -> str:
        """The body the payload is welded to: the arm LINK the tool is bolted
        to, not the tool body itself.

        The tool is a free rigid body held to the wrist by its own fixed joint
        rather than a link of the arm's articulation, so welding a payload to
        it would put two maximal-coordinate joints in series off the end of an
        articulation. PhysX resolves that chain with a corrective impulse big
        enough to throw the arm across the cell and leave the box balanced on a
        corner. A link the articulation solver already owns takes the payload
        without a fight, which is how the parallel-jaw grasp attaches too.
        Falls back to the tool when no mount link was recorded."""
        return self.parent_prim_path or self._tool_prim_path

    def _payload_local_frame(self, payload_name: str) -> tuple[Vec3, Quat]:
        """The weld's payload-side local frame for ``payload_name``, read from
        the live poses. Sim thread only."""

        def _world_pose(prim_path: str) -> tuple[Vec3, Quat]:
            pos, quat = self._sim._isaac.SingleXFormPrim(prim_path).get_world_pose()
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )

        return suction_joint_local_frame(
            _world_pose(self._weld_anchor_prim_path()), _world_pose(f"/World/{payload_name}")
        )

    def grab(self) -> None:
        self._engaged = True
        self._start_grab_window()

        def _grab() -> None:
            from pxr import Gf, Sdf, UsdPhysics

            cup_position, _cup_quat = self._cup_face_pose()
            chosen = select_payload_for_cup(
                cup_position, self._cup_side_m, self._max_payload_gap_m, self._payload_candidates()
            )
            if chosen is None:
                self._held_payload = None
                return
            stage = self._sim._isaac.get_prim_at_path(self._tool_prim_path).GetStage()
            local_pos, local_rot = self._payload_local_frame(chosen)
            author_suction_joint(
                UsdPhysics,
                Sdf,
                Gf,
                stage,
                self._joint_path,
                self._weld_anchor_prim_path(),
                f"/World/{chosen}",
                local_pos,
                local_rot,
            )
            self._held_payload = chosen
            LOGGER.info("vacuum %r welded suction joint to %r", self._name, chosen)

        self._sim.run(_grab)

    def open(self) -> None:
        self._engaged = False
        self._clear_grab_window()

        def _open() -> None:
            stage = self._sim._isaac.get_prim_at_path(self._tool_prim_path).GetStage()
            remove_suction_joint(stage, self._joint_path)
            self._held_payload = None

        self._sim.run(_open)

    def stop(self) -> None:
        """A suction cup has no travel to freeze mid-move - engaged or not,
        stop() leaves it exactly where it is."""
        return None

    def is_engaged(self) -> bool:
        return self._engaged

    def is_holding(self) -> bool:
        def _check() -> bool:
            stage = self._sim._isaac.get_prim_at_path(self._tool_prim_path).GetStage()
            return bool(stage.GetPrimAtPath(self._joint_path).IsValid())

        return self._sim.run(_check)

    def poll_state(self) -> tuple[bool, bool]:
        return self.is_moving(), self.is_holding()

    def dof_names(self) -> list[str]:
        """A vacuum tool adds no articulated joint of its own."""
        return []

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        def _poses() -> dict[str, tuple[Vec3, Quat]]:
            out: dict[str, tuple[Vec3, Quat]] = {}
            if self.parent_prim_path:
                pos, quat = self._sim._isaac.SingleXFormPrim(self.parent_prim_path).get_world_pose()
                out["parent"] = (
                    (float(pos[0]), float(pos[1]), float(pos[2])),
                    (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
            pos, quat = self._sim._isaac.SingleXFormPrim(self._tool_prim_path).get_world_pose()
            out["tool"] = (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
            return out

        return self._sim.run(_poses)

    def post_reset(self) -> None:
        """Re-weld the suction joint to the payload grab() last chose - a
        world reset drops the joint prim just like it drops the gripper's
        commanded jaw target."""

        def _redo() -> None:
            if self._held_payload is None:
                return
            from pxr import Gf, Sdf, UsdPhysics

            stage = self._sim._isaac.get_prim_at_path(self._tool_prim_path).GetStage()
            local_pos, local_rot = self._payload_local_frame(self._held_payload)
            author_suction_joint(
                UsdPhysics,
                Sdf,
                Gf,
                stage,
                self._joint_path,
                self._weld_anchor_prim_path(),
                f"/World/{self._held_payload}",
                local_pos,
                local_rot,
            )

        self._sim.run(_redo)

    def release(self) -> None:
        """Drop the post-reset hook create_vacuum_gripper registered under
        this component's name. The prim stays attached to the arm."""
        self._sim.unregister_post_reset(self._name)


class MockVacuumHandle(VacuumGripperHandle):
    """attrs["mock_attach_prop"] names the prop the cup finds under it.
    Unset means nothing is there, so grab() leaves is_holding() False - the
    mock counterpart to a real grab finding no payload within
    max_payload_gap_m. The weld itself is instant, but is_moving() reports
    True for grab_delay_ms after grab() is commanded, matching
    IsaacVacuumHandle."""

    def __init__(self, name: str, attrs: dict[str, Any], arm: ArmHandle) -> None:
        self.name = name
        self._arm = arm
        self._attach_prop: str | None = attrs.get("mock_attach_prop")
        self.tcp_offset_m = float(attrs.get("tcp_offset_m", VACUUM_TOOL["tcp_offset_m"]))
        self._grab_delay_s = float(attrs.get("grab_delay_ms", DEFAULT_GRAB_DELAY_MS)) / 1000.0
        self._holding = False
        self._engaged = False

    def grab(self) -> None:
        self._engaged = True
        self._start_grab_window()
        self._holding = self._attach_prop is not None

    def open(self) -> None:
        self._engaged = False
        self._clear_grab_window()
        self._holding = False

    def stop(self) -> None:
        return None

    def is_engaged(self) -> bool:
        return self._engaged

    def is_holding(self) -> bool:
        return self._holding

    def poll_state(self) -> tuple[bool, bool]:
        return self.is_moving(), self._holding

    def dof_names(self) -> list[str]:
        return []

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        """Synthetic: the mount link at the origin, the tool straddling the
        TCP along +Z, in the spirit of MockGripperHandle.link_world_poses."""
        identity: Quat = (1.0, 0.0, 0.0, 0.0)
        return {
            "parent": ((0.0, 0.0, 0.0), identity),
            "tool": ((0.0, 0.0, self.tcp_offset_m), identity),
        }

    def post_reset(self) -> None:
        return None

    def release(self) -> None:
        return None
