import math

import pytest

from isaac_module.cup_contact import cup_contact, cup_gaps_before_grab, rest_offsets_m
from isaac_module.spatial import quat_from_axis_angle

HALF = (0.2, 0.15, 0.125)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
CUPS = ((0.08, 0.04, 0.0), (0.08, -0.04, 0.0), (-0.08, 0.04, 0.0), (-0.08, -0.04, 0.0))


def _box_with_face_at(gap_m, quat=IDENTITY, x=0.0, y=0.0):
    """A box posed in the tool frame with its near face `gap_m` past the cup
    plane: with the box's frame aligned to the tool's, the tool looks along
    the box's +Z, so the near face is the box's -Z face."""
    return ((x, y, gap_m + HALF[2]), quat)


def test_flat_box_under_a_cup_reads_the_gap_to_its_near_face():
    pos, quat = _box_with_face_at(0.0057)

    contact = cup_contact(CUPS[0], pos, quat, HALF)

    assert contact.inside_face is True
    assert contact.gap_m == pytest.approx(0.0057)
    assert contact.point_tool_m[:2] == pytest.approx(CUPS[0][:2])


def test_the_face_that_looks_at_the_tool_is_picked_from_the_box_orientation():
    # box rolled a half turn about X: its +Z face now looks at the tool
    flipped = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)
    pos = (0.0, 0.0, 0.004 + HALF[2])

    contact = cup_contact(CUPS[0], pos, flipped, HALF)

    assert contact.inside_face is True
    assert contact.gap_m == pytest.approx(0.004, abs=1e-9)


def test_a_cup_past_the_face_edge_is_outside():
    pos, quat = _box_with_face_at(0.005, x=0.15)

    contact = cup_contact(CUPS[2], pos, quat, HALF)

    assert contact.inside_face is False


def test_a_tilted_box_reads_a_larger_gap_under_its_far_cups():
    tilt = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(10.0))
    pos, quat = _box_with_face_at(0.005, quat=tilt)

    near = cup_contact(CUPS[2], pos, quat, HALF).gap_m
    far = cup_contact(CUPS[0], pos, quat, HALF).gap_m

    assert far != pytest.approx(near)
    assert abs(far - near) == pytest.approx(0.16 * math.sin(math.radians(10.0)), rel=0.05)


def test_gaps_before_grab_list_every_cup_in_order():
    pos, quat = _box_with_face_at(0.0057)

    assert cup_gaps_before_grab(CUPS, pos, quat, HALF, 0.015) == pytest.approx([0.0057] * 4)


@pytest.mark.parametrize(
    "pose",
    [
        _box_with_face_at(0.040),  # out of the grip distance
        _box_with_face_at(-0.002),  # face already past the cup plane
        _box_with_face_at(0.005, x=0.15),  # two cups off the face's edge
    ],
)
def test_gaps_before_grab_are_none_when_a_ray_would_miss(pose):
    pos, quat = pose

    assert cup_gaps_before_grab(CUPS, pos, quat, HALF, 0.015) is None


def test_rest_offsets_double_the_excess_over_the_clearance_and_clamp_at_the_plane():
    assert rest_offsets_m([0.0057, 0.005, 0.0043], 0.005) == pytest.approx([0.0014, 0.0, 0.0])
