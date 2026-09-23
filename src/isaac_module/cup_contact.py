"""Where a suction cup meets the box under it, from the stage's poses and the
box's own dimensions. Isaac's surface gripper attaches its joints inside PhysX
and leaves nothing about the attached frame on the stage, so the vacuum
handle's coaxial monitor reads the cups' stretch off the held box's face
instead, and reads the gap to that face before a grab to know where the plugin
will leave the face at rest."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .spatial import Quat, Vec3, compose_pose, pose_in_frame, quat_conj, quat_rotate

_IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)
_TOOL_FORWARD: Vec3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class CupContact:
    """The point on the box's face nearest the tool straight under one cup, in
    the tool frame, and whether the cup sits over that face at all rather than
    past its edge."""

    point_tool_m: Vec3
    inside_face: bool

    @property
    def gap_m(self) -> float:
        """The face's distance from the cup plane along the tool's +Z, which is
        the cup axis pointing at the box."""
        return self.point_tool_m[2]


def cup_contact(
    cup_point_tool_m: Vec3, box_pos_tool_m: Vec3, box_quat_tool: Quat, half_dims_m: Vec3
) -> CupContact:
    """The face point under one cup for a box posed (position, orientation) in
    the tool frame with half edge lengths ``half_dims_m`` along its own axes.
    The face is the one that looks at the tool: the box axis most against the
    tool's +Z, on the side the tool is on."""
    forward_box = quat_rotate(quat_conj(box_quat_tool), _TOOL_FORWARD)
    face_axis = max(range(3), key=lambda index: abs(forward_box[index]))
    face_sign = -1.0 if forward_box[face_axis] > 0.0 else 1.0
    cup_box, _ = pose_in_frame(box_pos_tool_m, box_quat_tool, cup_point_tool_m, _IDENTITY)
    face_point_box = list(cup_box)
    face_point_box[face_axis] = face_sign * half_dims_m[face_axis]
    inside = all(
        abs(cup_box[index]) <= half_dims_m[index] for index in range(3) if index != face_axis
    )
    point_tool, _ = compose_pose(
        box_pos_tool_m,
        box_quat_tool,
        (face_point_box[0], face_point_box[1], face_point_box[2]),
        _IDENTITY,
    )
    return CupContact(point_tool_m=point_tool, inside_face=inside)


def cup_gaps_before_grab(
    cup_points_tool_m: Sequence[Vec3],
    box_pos_tool_m: Vec3,
    box_quat_tool: Quat,
    half_dims_m: Vec3,
    max_gap_m: float,
) -> list[float] | None:
    """Every cup's gap to the face under it, in cup order, when every cup sits
    over the face within (0, ``max_gap_m``]; None when any cup is off the face
    or the face is out of reach, so the plugin's rays would not all land on
    this box."""
    gaps: list[float] = []
    for cup_point in cup_points_tool_m:
        contact = cup_contact(cup_point, box_pos_tool_m, box_quat_tool, half_dims_m)
        if not contact.inside_face or not 0.0 < contact.gap_m <= max_gap_m:
            return None
        gaps.append(contact.gap_m)
    return gaps


def rest_offsets_m(gaps_m: Sequence[float], clearance_offset_m: float) -> list[float]:
    """Where the plugin leaves the face at rest below each cup, given the gap
    it saw. It starts its ray ``clearance_offset_m`` past the cup, so the ray
    travels the excess ``t = gap - clearance`` to the face, places the joint's
    object-side anchor ``t`` behind the cup, and PhysX draws the box by ``t``
    the other way: the face settles ``2 t`` below the cup. A face already
    inside the clearance settles at the cup plane."""
    return [2.0 * max(0.0, gap - clearance_offset_m) for gap in gaps_m]
