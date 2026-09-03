"""The conductor's pure planning contract: work-list sequencing and pad-slot
assignment, with no viam imports so the module is importable by both the
conductor model and its tests without side effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaac_module import cell_layout

OUTCOME_PLACED = "placed"
OUTCOME_SKIPPED_OVERSIZE = "skipped_oversize"
OUTCOME_FAILED = "failed"


@dataclass(frozen=True)
class WorkItem:
    """One block the conductor still has to pick, sized and posed from the
    source-zone scan."""

    name: str
    color: str
    x_mm: float
    y_mm: float
    size_mm: float


def nearest_first(items: list[WorkItem], from_xy_mm: tuple[float, float]) -> list[WorkItem]:
    """A greedy nearest-neighbour walk: each next block is the one nearest
    (planar euclidean) to the previously chosen block's position, starting
    from ``from_xy_mm``. Ties break by name for a deterministic order."""
    remaining = list(items)
    ordered: list[WorkItem] = []
    current_x, current_y = from_xy_mm
    while remaining:
        nearest = min(
            remaining,
            key=lambda item: (math.hypot(item.x_mm - current_x, item.y_mm - current_y), item.name),
        )
        ordered.append(nearest)
        remaining.remove(nearest)
        current_x, current_y = nearest.x_mm, nearest.y_mm
    return ordered


def clearance_ordered(items: list[WorkItem], from_xy_mm: tuple[float, float]) -> list[WorkItem]:
    """A greedy clearance-first walk: each next block is whichever remaining
    block has the largest planar clearance (distance to its nearest OTHER
    remaining block), so isolated blocks are attempted before crowded ones -
    a crowded block's descent corridor can clip a still-present neighbour
    (phase-4 GPU evidence), so it should wait until fewer neighbours remain.
    Ties break by ``nearest_first``'s walk order from ``from_xy_mm``.

    Implementation choice (documented per the phase-4 brief): clearance is
    recomputed from scratch against the shrinking remaining set after every
    pick, rather than incrementally updated, since re-sorting a scattered
    work list of this size is cheap and the simpler code is easier to trust.
    """
    walk_rank = {item.name: rank for rank, item in enumerate(nearest_first(items, from_xy_mm))}

    def clearance_mm(item: WorkItem, remaining: list[WorkItem]) -> float:
        distances = [
            math.hypot(item.x_mm - other.x_mm, item.y_mm - other.y_mm)
            for other in remaining
            if other is not item
        ]
        return min(distances) if distances else math.inf

    remaining = list(items)
    ordered: list[WorkItem] = []
    while remaining:
        chosen = max(
            remaining, key=lambda item: (clearance_mm(item, remaining), -walk_rank[item.name])
        )
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


class SlotTracker:
    """Per-color cursor over ``cell_layout.PAD_SLOT_OFFSETS_MM``, handing out
    each color's next free spread slot. A drawn slot can be handed back with
    ``release`` (an attempt that never placed must not lose it), and a
    released slot is reused before any new one is drawn."""

    def __init__(self) -> None:
        self._next_index: dict[str, int] = {color: 0 for color in cell_layout.BLOCK_COLORS}
        self._released: dict[str, list[tuple[float, float]]] = {
            color: [] for color in cell_layout.BLOCK_COLORS
        }
        self._held: dict[str, set[tuple[float, float]]] = {
            color: set() for color in cell_layout.BLOCK_COLORS
        }

    def next_slot(self, color: str) -> tuple[float, float]:
        """The next free pad-slot offset for ``color``: a released offset
        first (LIFO), otherwise the next never-drawn offset, advancing its
        cursor.

        Raises ``ValueError`` naming the color once its slots are exhausted.
        """
        if color not in self._next_index:
            raise ValueError(f"unknown color: {color}")
        if self._released[color]:
            offset = self._released[color].pop()
        else:
            index = self._next_index[color]
            if index >= len(cell_layout.PAD_SLOT_OFFSETS_MM):
                raise ValueError(f"no free pad slots left for color: {color}")
            self._next_index[color] = index + 1
            offset = cell_layout.PAD_SLOT_OFFSETS_MM[index]
        self._held[color].add(offset)
        return offset

    def release(self, color: str, offset: tuple[float, float]) -> None:
        """Returns ``offset`` to ``color``'s pool for reuse by a later draw.

        Raises ``ValueError`` if ``offset`` was not currently drawn for
        ``color`` (a double release, or one that never happened).
        """
        if color not in self._held or offset not in self._held[color]:
            raise ValueError(f"offset {offset} is not outstanding for color: {color}")
        self._held[color].remove(offset)
        self._released[color].append(offset)

    def place_target_mm(self, color: str, offset: tuple[float, float]) -> tuple[float, float]:
        """The absolute pad-slot centre for ``color``'s pad plus ``offset``."""
        if color not in cell_layout.PAD_CENTRES_MM:
            raise ValueError(f"unknown color: {color}")
        centre_x, centre_y = cell_layout.PAD_CENTRES_MM[color]
        offset_x, offset_y = offset
        return (centre_x + offset_x, centre_y + offset_y)
