"""Stack stability as pure functions over pose samples.

Nothing here touches Isaac. The palletizer feeds it the readings it takes
while a box comes to rest, and the GPU checklist feeds it the poses of a
whole stack at settle and again after a hold with the arm parked. Both get
back a verdict that a test can pin, so every physics change is measured
against the same rule rather than argued from a viewport.

Two questions, two rules:

- **Settled.** A body has settled once every reading over a trailing window
  of ``window_s`` stays within ``tolerance_mm`` of the reading that opened
  the window. Two consecutive reads agreeing is not enough: on 2026-09-16
  a box read mid-fall agreed with itself for one poll and the cup closed on
  air. ``settle_time_s`` reports when that window OPENED, which is the
  settle time itself, not the moment the policy noticed it.
- **Held.** A stack held when every box is within ``placement_mm`` of the
  sequencer's target at settle, still within ``drift_mm`` of its settled
  pose after the hold, and within ``tilt_deg`` of the target's orientation
  at both readings. One box past any bound fails the stack, and the verdict
  names the box and the bound.

Poses are the ``pose_in_world_mm`` mappings the world's ``prop_geometries``
verb answers with: ``x``, ``y``, ``z`` in millimetres, ``o_x``, ``o_y``,
``o_z`` an orientation vector and ``theta`` in degrees. A box's top face
and its centre share an orientation, so which one a caller passes only has
to be the same on both sides of a comparison.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PoseMM = Mapping[str, float]

# (seconds since the first sample, the pose read then, or None when the prop
# was not in the reading at all)
Sample = tuple[float, PoseMM | None]


@dataclass(frozen=True)
class SettlePolicy:
    """When a body counts as settled and how long to wait for it.

    ``tolerance_mm`` is the most a reading may differ from the one that
    opened the window and still count as still. ``window_s`` is how long
    readings must stay inside it. ``poll_s`` is the gap between readings,
    and ``timeout_s`` is the most a caller waits before giving up and
    using the last reading it has."""

    tolerance_mm: float = 1.0
    window_s: float = 0.25
    poll_s: float = 0.05
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.tolerance_mm <= 0:
            raise ValueError(f"settle policy: tolerance_mm must be > 0 (got {self.tolerance_mm!r})")
        if self.poll_s <= 0:
            raise ValueError(f"settle policy: poll_s must be > 0 (got {self.poll_s!r})")
        if self.window_s < self.poll_s:
            raise ValueError(
                f"settle policy: window_s must be >= poll_s "
                f"(got window_s={self.window_s!r}, poll_s={self.poll_s!r})"
            )
        if self.timeout_s < self.window_s:
            raise ValueError(
                f"settle policy: timeout_s must be >= window_s "
                f"(got timeout_s={self.timeout_s!r}, window_s={self.window_s!r})"
            )


DEFAULT_SETTLE_POLICY = SettlePolicy()


@dataclass(frozen=True)
class StackTolerances:
    """The three bounds a placed box is judged against.

    ``placement_mm``: the sim's boxes sit 3 mm inside their 200 x 150 slot on
    every side (the overlay's scale, chosen 2026-09-24), so a top face more
    than 3 mm from its target has crossed into a neighbour's slot.
    ``drift_mm``: the palletizer's own settle tolerance. A box that moved
    more than its settle threshold over the hold did not hold.
    ``tilt_deg``: the checklist's release tilt tolerance, the most a level
    box shows on the cups."""

    placement_mm: float = 3.0
    drift_mm: float = 1.0
    tilt_deg: float = 2.0

    def __post_init__(self) -> None:
        for key in ("placement_mm", "drift_mm", "tilt_deg"):
            value = getattr(self, key)
            if value <= 0:
                raise ValueError(f"stack tolerances: {key} must be > 0 (got {value!r})")


DEFAULT_STACK_TOLERANCES = StackTolerances()


@dataclass(frozen=True)
class BoxPlacement:
    """One box's readings, in the sequencer's own top-face convention.

    ``target_mm`` is the sequencer's target. ``at_settle_mm`` is where the
    palletizer measured the box once it stopped (None when it was never
    placed). ``after_hold_mm`` is the same box read again after the hold
    with the arm parked (None when no hold was read, or the prop was gone).
    ``settle_s`` is the palletizer's measured settle time for the box."""

    name: str
    seq: int
    layer: int
    target_mm: PoseMM
    at_settle_mm: PoseMM | None
    after_hold_mm: PoseMM | None = None
    settle_s: float | None = None


@dataclass(frozen=True)
class BoxVerdict:
    """One box against the tolerances. ``error_mm`` is at settle against the
    target, ``drift_mm`` is after the hold against at settle (None when the
    verdict was asked for without a hold), ``tilt_deg`` is the larger of the
    two readings' tilt from the target. A bound crossed puts a reason in
    ``reasons`` and clears ``ok``."""

    name: str
    seq: int
    layer: int
    error_mm: float
    drift_mm: float | None
    tilt_deg: float
    ok: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StackVerdict:
    """The stack as a whole. ``held`` is every box ok. ``failures`` is one
    line per failed box, ``<name>: <reason>; <reason>``. The worst values
    are over every box, so the headline reads the same whether one box or
    all of them crept."""

    held: bool
    boxes: tuple[BoxVerdict, ...]
    worst_error_mm: float
    worst_drift_mm: float | None
    worst_tilt_deg: float
    failures: tuple[str, ...]


def pose_distance_mm(pose_a: PoseMM, pose_b: PoseMM) -> float:
    """Straight-line distance between two poses' positions."""
    return math.sqrt(
        sum((float(pose_a[axis]) - float(pose_b[axis])) ** 2 for axis in ("x", "y", "z"))
    )


def pose_tilt_deg(pose_a: PoseMM, pose_b: PoseMM) -> float:
    """The angle between two poses' orientation vectors, in degrees. A box
    lying on its side reads 90 against one standing up. ``theta``, the
    spin about that vector, is not a tilt and is ignored here."""
    ax, ay, az = (float(pose_a[k]) for k in ("o_x", "o_y", "o_z"))
    bx, by, bz = (float(pose_b[k]) for k in ("o_x", "o_y", "o_z"))
    norm_a = math.sqrt(ax * ax + ay * ay + az * az)
    norm_b = math.sqrt(bx * bx + by * by + bz * bz)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("pose_tilt_deg: an orientation vector is all zero")
    cosine = (ax * bx + ay * by + az * bz) / (norm_a * norm_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def settle_time_s(
    samples: Sequence[Sample], policy: SettlePolicy = DEFAULT_SETTLE_POLICY
) -> float | None:
    """Seconds from the first sample to the sample that opened the first
    window in which every reading stayed within ``policy.tolerance_mm`` of
    that opening reading for at least ``policy.window_s``.

    None when no such window completes inside ``samples``: the body was
    still moving, or the samples do not span a window. A None pose breaks
    any window it falls in, since a prop that vanished has not settled."""
    for opener_index, (opened_at, opener) in enumerate(samples):
        if opener is None:
            continue
        window_closed = False
        for sampled_at, pose in samples[opener_index + 1 :]:
            if pose is None or pose_distance_mm(opener, pose) > policy.tolerance_mm:
                break
            if sampled_at - opened_at >= policy.window_s:
                window_closed = True
                break
        if window_closed:
            return opened_at - samples[0][0]
    return None


def trailing_window_settled(
    samples: Sequence[Sample], policy: SettlePolicy = DEFAULT_SETTLE_POLICY
) -> bool:
    """Whether the newest samples show a settled body: the readings from the
    newest back to the first one at least ``policy.window_s`` older all lie
    within ``policy.tolerance_mm`` of that older reading, and such a reading
    exists. The live loop's stop condition, so a caller polls, appends and
    asks this after every reading."""
    if not samples:
        return False
    newest_at, newest = samples[-1]
    if newest is None:
        return False
    for opened_at, opener in reversed(samples):
        if opener is None:
            return False
        if pose_distance_mm(opener, newest) > policy.tolerance_mm:
            return False
        if newest_at - opened_at >= policy.window_s:
            return all(
                pose is not None and pose_distance_mm(opener, pose) <= policy.tolerance_mm
                for sampled_at, pose in samples
                if opened_at <= sampled_at <= newest_at
            )
    return False


def hold_excursion_mm(samples: Sequence[Sample]) -> float:
    """The farthest any sample strayed from the first, over a hold. A box
    that wobbled and came back still shows it here where a two-point drift
    would not. inf when any sample lost the prop, 0.0 for one sample, and
    a ValueError for none."""
    if not samples:
        raise ValueError("hold_excursion_mm: no samples")
    _first_at, first = samples[0]
    if first is None:
        return math.inf
    worst = 0.0
    for _sampled_at, pose in samples[1:]:
        if pose is None:
            return math.inf
        worst = max(worst, pose_distance_mm(first, pose))
    return worst


def _box_verdict(placement: BoxPlacement, tolerances: StackTolerances, *, hold: bool) -> BoxVerdict:
    """One placement judged against ``tolerances``. ``hold`` selects whether
    ``after_hold_mm`` is judged too, exactly as :func:`stack_verdict` documents."""
    reasons: list[str] = []

    if placement.at_settle_mm is None:
        error_mm = math.inf
        reasons.append("never placed")
    else:
        error_mm = pose_distance_mm(placement.target_mm, placement.at_settle_mm)
        if error_mm > tolerances.placement_mm:
            reasons.append(f"error {error_mm:.2f} mm > {tolerances.placement_mm} mm")

    if hold:
        if placement.at_settle_mm is None:
            drift_mm: float | None = math.inf
        elif placement.after_hold_mm is None:
            drift_mm = math.inf
            reasons.append("gone after the hold")
        else:
            drift_mm = pose_distance_mm(placement.at_settle_mm, placement.after_hold_mm)
            if drift_mm > tolerances.drift_mm:
                reasons.append(f"drift {drift_mm:.2f} mm > {tolerances.drift_mm} mm")
    else:
        drift_mm = None

    tilt_readings = []
    if placement.at_settle_mm is not None:
        tilt_readings.append(pose_tilt_deg(placement.target_mm, placement.at_settle_mm))
    if placement.after_hold_mm is not None:
        tilt_readings.append(pose_tilt_deg(placement.target_mm, placement.after_hold_mm))
    tilt_deg = max(tilt_readings) if tilt_readings else 0.0
    if tilt_deg > tolerances.tilt_deg:
        reasons.append(f"tilt {tilt_deg:.1f} deg > {tolerances.tilt_deg} deg")

    return BoxVerdict(
        name=placement.name,
        seq=placement.seq,
        layer=placement.layer,
        error_mm=error_mm,
        drift_mm=drift_mm,
        tilt_deg=tilt_deg,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def stack_verdict(
    placements: Sequence[BoxPlacement],
    tolerances: StackTolerances = DEFAULT_STACK_TOLERANCES,
    *,
    hold: bool,
    layer: int | None = None,
) -> StackVerdict:
    """Judges ``placements`` against ``tolerances``.

    With ``hold`` True every box is also judged on its ``after_hold_mm``:
    a None there is a box that vanished during the hold, which fails it.
    With ``hold`` False the drift is not judged and reads None. ``layer``
    restricts the verdict to that layer's boxes, so the second layer can be
    held to the same standard as the first on its own. An empty selection
    is a ValueError, since a stack of nothing did not hold."""
    selected = [p for p in placements if layer is None or p.layer == layer]
    if not selected:
        raise ValueError(f"stack_verdict: no placements selected (layer={layer!r})")
    selected.sort(key=lambda placement: placement.seq)

    boxes = [_box_verdict(placement, tolerances, hold=hold) for placement in selected]

    worst_drift_mm = (
        max(box.drift_mm for box in boxes if box.drift_mm is not None) if hold else None
    )
    return StackVerdict(
        held=all(box.ok for box in boxes),
        boxes=tuple(boxes),
        worst_error_mm=max(box.error_mm for box in boxes),
        worst_drift_mm=worst_drift_mm,
        worst_tilt_deg=max(box.tilt_deg for box in boxes),
        failures=tuple(f"{box.name}: {'; '.join(box.reasons)}" for box in boxes if not box.ok),
    )


def format_stack_table(verdict: StackVerdict) -> str:
    """The verdict as a fixed-width table, one row per box in seq order:
    seq, layer, name, error mm, drift mm (or a dash), tilt deg, ok, and the
    reasons. A caller prints it under the checklist item's heading."""
    header = (
        f"{'seq':>3}  {'layer':>5}  {'name':<20}  {'error_mm':>9}  "
        f"{'drift_mm':>9}  {'tilt_deg':>9}  {'ok':>3}  reasons"
    )
    rows = [header]
    for box in sorted(verdict.boxes, key=lambda entry: entry.seq):
        error_str = "inf" if math.isinf(box.error_mm) else f"{box.error_mm:.2f}"
        if box.drift_mm is None:
            drift_str = "-"
        elif math.isinf(box.drift_mm):
            drift_str = "inf"
        else:
            drift_str = f"{box.drift_mm:.2f}"
        rows.append(
            f"{box.seq:>3}  {box.layer:>5}  {box.name:<20}  {error_str:>9}  "
            f"{drift_str:>9}  {box.tilt_deg:>9.1f}  {'yes' if box.ok else 'no':>3}  "
            f"{'; '.join(box.reasons)}"
        )
    return "\n".join(rows)
