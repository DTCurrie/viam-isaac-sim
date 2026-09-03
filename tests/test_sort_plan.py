"""sort_plan is pure planning: nearest-first sequencing and pad-slot
assignment, both proven against hand-computed positions and
cell_layout's own constants."""

import pytest

from isaac_module import cell_layout, sort_plan


def test_nearest_first_walks_to_each_closest_remaining_block_in_turn():
    # start at (0, 0): b (1mm) is nearest first, then a (9mm from b) is
    # nearer than c or d, then c (90mm from a) is nearer than d, then d
    a = sort_plan.WorkItem(name="a", color="red", x_mm=10.0, y_mm=0.0, size_mm=60.0)
    b = sort_plan.WorkItem(name="b", color="red", x_mm=1.0, y_mm=0.0, size_mm=60.0)
    c = sort_plan.WorkItem(name="c", color="red", x_mm=100.0, y_mm=0.0, size_mm=60.0)
    d = sort_plan.WorkItem(name="d", color="red", x_mm=100.0, y_mm=50.0, size_mm=60.0)

    ordered = sort_plan.nearest_first([a, b, c, d], from_xy_mm=(0.0, 0.0))

    assert [item.name for item in ordered] == ["b", "a", "c", "d"]


def test_nearest_first_breaks_ties_by_name():
    # both blocks sit exactly 10 mm from the start point
    east = sort_plan.WorkItem(name="east", color="red", x_mm=10.0, y_mm=0.0, size_mm=60.0)
    west = sort_plan.WorkItem(name="west", color="red", x_mm=-10.0, y_mm=0.0, size_mm=60.0)

    ordered = sort_plan.nearest_first([east, west], from_xy_mm=(0.0, 0.0))

    assert [item.name for item in ordered] == ["east", "west"]


def test_nearest_first_of_an_empty_list_is_empty():
    assert sort_plan.nearest_first([], from_xy_mm=(0.0, 0.0)) == []


def test_slot_tracker_hands_out_the_three_offsets_in_order_then_raises():
    tracker = sort_plan.SlotTracker()
    slots = [tracker.next_slot("red") for _ in range(cell_layout.POOL_BLOCKS_PER_COLOR)]

    assert slots == list(cell_layout.PAD_SLOT_OFFSETS_MM)
    assert len(set(slots)) == cell_layout.POOL_BLOCKS_PER_COLOR

    with pytest.raises(ValueError, match="red"):
        tracker.next_slot("red")


def test_slot_tracker_cursors_are_independent_per_color():
    tracker = sort_plan.SlotTracker()
    tracker.next_slot("red")
    tracker.next_slot("red")

    # green's cursor is untouched by red's two draws
    assert tracker.next_slot("green") == cell_layout.PAD_SLOT_OFFSETS_MM[0]


def test_slot_tracker_rejects_an_unknown_color():
    tracker = sort_plan.SlotTracker()
    with pytest.raises(ValueError):
        tracker.next_slot("magenta")


def test_slot_tracker_release_makes_a_drawn_offset_reusable():
    tracker = sort_plan.SlotTracker()
    first = tracker.next_slot("red")

    tracker.release("red", first)
    reused = tracker.next_slot("red")

    assert reused == first


def test_slot_tracker_release_of_a_non_outstanding_offset_raises():
    tracker = sort_plan.SlotTracker()
    with pytest.raises(ValueError, match="red"):
        tracker.release("red", cell_layout.PAD_SLOT_OFFSETS_MM[0])


def test_slot_tracker_release_of_an_already_released_offset_raises():
    tracker = sort_plan.SlotTracker()
    offset = tracker.next_slot("red")
    tracker.release("red", offset)

    with pytest.raises(ValueError, match="red"):
        tracker.release("red", offset)


def test_slot_tracker_draw_release_draw_cycle_never_exhausts_the_pool():
    tracker = sort_plan.SlotTracker()
    slots = [tracker.next_slot("red") for _ in range(cell_layout.POOL_BLOCKS_PER_COLOR)]
    for slot in slots:
        tracker.release("red", slot)

    replayed = [tracker.next_slot("red") for _ in range(cell_layout.POOL_BLOCKS_PER_COLOR)]

    assert set(replayed) == set(slots)


def test_place_target_mm_adds_the_offset_to_the_pad_centre():
    tracker = sort_plan.SlotTracker()
    centre_x, centre_y = cell_layout.PAD_CENTRES_MM["blue"]
    offset = cell_layout.PAD_SLOT_OFFSETS_MM[1]

    target = tracker.place_target_mm("blue", offset)

    assert target == (centre_x + offset[0], centre_y + offset[1])


def test_place_target_mm_rejects_an_unknown_color():
    tracker = sort_plan.SlotTracker()
    with pytest.raises(ValueError):
        tracker.place_target_mm("magenta", (0.0, 0.0))


def test_clearance_ordered_sorts_a_crowded_pair_last():
    # a (1000,0) is isolated, d (21,0) is next-clearest (nearest other is c
    # at 16mm), and the crowded pair b (0,0)/c (5,0) - 5mm apart - sorts
    # last; b before c since nearest_first from (0,0) visits b then c first
    a = sort_plan.WorkItem(name="a", color="red", x_mm=1000.0, y_mm=0.0, size_mm=60.0)
    b = sort_plan.WorkItem(name="b", color="red", x_mm=0.0, y_mm=0.0, size_mm=60.0)
    c = sort_plan.WorkItem(name="c", color="red", x_mm=5.0, y_mm=0.0, size_mm=60.0)
    d = sort_plan.WorkItem(name="d", color="red", x_mm=21.0, y_mm=0.0, size_mm=60.0)

    ordered = sort_plan.clearance_ordered([a, b, c, d], from_xy_mm=(0.0, 0.0))

    assert [item.name for item in ordered] == ["a", "d", "b", "c"]


def test_clearance_ordered_recompute_brings_the_crowder_neighbour_forward():
    # once b (the crowder that pinned c's clearance to 5mm) is gone - as it
    # would be after an earlier pass placed it - c's clearance against the
    # same remaining pool jumps to 16mm (nearest other is now d), moving it
    # ahead of d instead of trailing it as it did with b still present
    a = sort_plan.WorkItem(name="a", color="red", x_mm=1000.0, y_mm=0.0, size_mm=60.0)
    c = sort_plan.WorkItem(name="c", color="red", x_mm=5.0, y_mm=0.0, size_mm=60.0)
    d = sort_plan.WorkItem(name="d", color="red", x_mm=21.0, y_mm=0.0, size_mm=60.0)

    ordered = sort_plan.clearance_ordered([a, c, d], from_xy_mm=(0.0, 0.0))

    assert [item.name for item in ordered] == ["a", "c", "d"]


def test_clearance_ordered_of_an_empty_list_is_empty():
    assert sort_plan.clearance_ordered([], from_xy_mm=(0.0, 0.0)) == []


def test_outcome_literals_match_the_status_vocabulary():
    assert sort_plan.OUTCOME_PLACED == "placed"
    assert sort_plan.OUTCOME_SKIPPED_OVERSIZE == "skipped_oversize"
    assert sort_plan.OUTCOME_FAILED == "failed"
