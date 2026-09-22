"""Talks to ``viam:pack-sequencer:sequencer``, the service that owns the
pack order, the placement cursor and the placed-set for this cell.

Our palletizer service does not decide where a box goes. It asks this
client, moves the arm there, and reports what happened. Everything below
is either a shape the sequencer publishes or a translation of one into
what the motion service wants, so the packing arithmetic lives in exactly
one place and that place is not this repo.

Sourcing
--------
Every verb, key and unit below is transcribed from
``viam-labs/pack-sequencer`` at ``0.3.0`` - ``module.go`` on the
repository's ``main`` branch, whose ``VERSION`` file reads ``0.3.0`` and
whose published ``0.3.0`` binary carries exactly the verb set the source
dispatches. Not reconstructed, not inferred from a README: the response
maps below are the literal maps the Go handlers return.

The wire convention is ``{"<verb>": <argument>}``, not the
``{"command": "<verb>"}`` shape this repo's own DoCommand verbs use -
the same convention ``viam:workcell-components`` uses, and for the same
reason ``workcell_client._verb`` exists. A verb that takes no argument
is sent as ``{"<verb>": true}``. A verb that takes one is sent with the
argument in that slot, and ``DoCommand`` dispatches on the first key it
recognises.

Units are millimetres and degrees throughout, matching
``viam.proto.common.Pose``. No conversion is needed in either direction,
which is why nothing here carries an ``_m`` suffix.

Frames
------
The sequencer publishes each slot twice: ``pose_in_pallet`` is relative
to the pallet component's own frame, and the ``*_in_world`` poses are
pre-composed with the pallet's world pose by the sequencer itself. We
consume the world poses, so this cell never holds a second copy of the
pallet origin - the drift this repo has already been bitten by once.

``place_start_in_world`` is the pose the arm descends FROM and
``place_end_in_world`` is where it releases. The offset between them is
the sequencer's own diagonal approach (``approach_offset_in_pallet``),
tilted so a descending box does not plow through an already-placed
neighbour, and it is clamped by the sequencer to at least
``box_height_mm + 10``. That standoff is the sequencer's to choose, so
the palletizer must not add ``PRE_GRASP_STANDOFF_MM`` on top of it at
the place end. The pick end keeps our own standoff, since the sequencer
knows nothing about the pick station.

A placement's ``z`` is the height of the box's TOP face above the pallet
deck (``(layer + 1) * box_height_mm``), not the box's centre. A vacuum
cup grips the top face, so this maps directly onto a gripper TCP pose
with no half-box correction - the same convention
``models.palletizer.pick_grasp_pose`` already uses at the pick end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Imported for its import-time side effect, not for a name. The SDK registers
# the world_state_store API inside this package's __init__, so a module process
# that never imports it cannot build a client for a dependency of that kind.
# viam-server answers `No world_state_store with name "pack-sequencer" found in
# the registry` and the palletizer never constructs. Nothing else here imports
# the package, so this line is what makes the sequencer dependency resolvable.
import viam.services.worldstatestore  # noqa: F401
from viam.proto.common import Pose
from viam.resource.base import ResourceBase

# The registry model this client speaks to. The palletizer service names
# an instance of it in its "sequencer" attribute.
SEQUENCER_MODEL = "viam:pack-sequencer:sequencer"

# Pose6D.ToMap() in github.com/viam-labs/viamkit/geom, which every pose
# on this wire goes through. Named here so a parse can assert the shape
# it was handed rather than silently reading zeros off a typo.
POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")

# next_box's "box_dimensions_mm" is NOT the same shape as get_box_dims'
# reply. The former is {"width", "length", "height"}, the latter is
# {"box_width_mm", "box_length_mm", "box_height_mm"}. Both are
# millimetres, and conflating them reads three zeros.
BOX_DIMS_KEYS = ("width", "length", "height")


def verb(name: str, argument: Any = True) -> dict[str, Any]:
    """One DoCommand payload: ``{name: argument}``.

    ``argument`` defaults to ``True`` for the verbs that take none
    (``get_pack_order``, ``next_box``, ``get_progress``, ``reset_cursor``,
    ``get_attributes``, ``get_box_dims``, ``get_pallet_home``).
    """
    return {name: argument}


def parse_pose(pose: Mapping[str, Any]) -> Pose:
    """One ``Pose6D.ToMap()`` reply as a ``viam.proto.common.Pose``.

    Millimetres and an orientation vector in degrees, straight across -
    the two types carry the same seven numbers under the same names.

    Raises ``ValueError`` naming the missing keys when ``pose`` is not a
    complete ``POSE_KEYS`` mapping, rather than defaulting them to zero:
    an orientation read as all zeros is not a pose, it is a plan that
    drives the wrist somewhere nobody asked for.
    """
    missing = [key for key in POSE_KEYS if key not in pose]
    if missing:
        raise ValueError(f"pose reply is missing keys: {', '.join(missing)}")
    return Pose(
        x=float(pose["x"]),
        y=float(pose["y"]),
        z=float(pose["z"]),
        o_x=float(pose["o_x"]),
        o_y=float(pose["o_y"]),
        o_z=float(pose["o_z"]),
        theta=float(pose["theta"]),
    )


def _parse_box_dimensions(dims: Mapping[str, Any]) -> BoxDimensions:
    """``next_box``'s ``box_dimensions_mm`` (``BOX_DIMS_KEYS``) as a
    ``BoxDimensions``."""
    missing = [key for key in BOX_DIMS_KEYS if key not in dims]
    if missing:
        raise ValueError(f"box_dimensions_mm reply is missing keys: {', '.join(missing)}")
    return BoxDimensions(
        width_mm=float(dims["width"]),
        length_mm=float(dims["length"]),
        height_mm=float(dims["height"]),
    )


@dataclass(frozen=True)
class BoxDimensions:
    """One box's footprint and height in millimetres."""

    width_mm: float
    length_mm: float
    height_mm: float


@dataclass(frozen=True)
class NextBox:
    """``next_box``'s reply: the slot the sequencer wants filled next,
    or the completion tally when there is none left.

    ``is_complete`` True means every other pose field is absent from the
    reply, so they are None here. A caller checks ``is_complete`` before
    reading ``place_end_in_world``.

    The cursor does NOT advance on ``next_box``. It advances on a
    successful ``report_placement``, and stays put on a failed one so the
    same seq comes back for a retry, which is what makes the retry path
    the sequencer's business rather than ours.
    """

    is_complete: bool
    total: int
    placed: int
    failed: int
    skipped: int
    remaining: int
    seq: int | None = None
    col: int | None = None
    row: int | None = None
    layer: int | None = None
    pose_in_pallet: Pose | None = None
    approach_offset_in_pallet: tuple[float, float, float] | None = None
    place_start_in_world: Pose | None = None
    place_end_in_world: Pose | None = None
    box_dimensions_mm: BoxDimensions | None = None


@dataclass(frozen=True)
class Placement:
    """One entry of ``get_pack_order``'s ``placements`` list: a slot in
    the full plan, whether or not it has been filled."""

    seq: int
    col: int
    row: int
    layer: int
    pose_in_pallet: Pose
    pose_in_world: Pose
    approach_offset_in_pallet: tuple[float, float, float]
    box_dimensions_mm: BoxDimensions
    label: str = ""


@dataclass(frozen=True)
class PackOrder:
    """``get_pack_order``'s reply: the whole plan, plus the pallet it was
    computed against.

    ``overflow`` is how many boxes ``quantity`` asked for beyond
    ``capacity``. ``warnings`` carries the sequencer's own complaints,
    such as a ``rotate_alternate_layers`` config on a box that is not
    2:1. Both are reported rather than raised: a short pack order is a
    result, not an error.
    """

    placements: Sequence[Placement]
    cols: int
    rows: int
    layers: int
    capacity: int
    quantity: int
    overflow: int
    mode: str
    warnings: Sequence[str]
    pallet_pose: Pose
    pallet_width_mm: float
    pallet_length_mm: float
    pallet_thickness_mm: float


@dataclass(frozen=True)
class PlacementReport:
    """``report_placement``'s reply: the cursor after recording one
    outcome.

    ``next_seq`` is where ``next_box`` will resume. On a success it has
    walked past the reported seq and any already-done seqs after it; on a
    failure it is unchanged, so the same box comes back.
    """

    acknowledged: bool
    next_seq: int
    placed: int
    failed: int
    skipped: int
    remaining: int
    complete: bool
    last_error: str = ""


@dataclass(frozen=True)
class SkipResult:
    """``skip_box``'s reply. ``skipped`` is the seq that was skipped, not
    a count - the count is ``report_placement``'s field of the same name,
    which is the kind of collision worth naming once here rather than
    rediscovering at a call site."""

    skipped: int
    next_seq: int
    placed: int
    remaining: int


@dataclass(frozen=True)
class Progress:
    """``get_progress``'s reply: which seqs are done, skipped and failed,
    for a caller that wants the sets rather than the tallies."""

    next_seq: int
    done_seqs: Sequence[int]
    skipped_seqs: Sequence[int]
    failed_seqs: Sequence[int]
    placed_count: int
    failed_count: int
    skipped_count: int
    total: int
    complete: bool


def parse_next_box(reply: Mapping[str, Any]) -> NextBox:
    """``next_box``'s reply as a ``NextBox``.

    A reply whose ``is_complete`` is True carries only the tallies, so
    every pose field stays None. Anything else must carry the full slot,
    and a missing pose raises rather than defaulting.
    """
    is_complete = bool(reply["is_complete"])
    total = int(reply["total"])
    placed = int(reply["placed"])
    failed = int(reply["failed"])
    skipped = int(reply["skipped"])
    remaining = int(reply["remaining"])
    if is_complete:
        return NextBox(
            is_complete=is_complete,
            total=total,
            placed=placed,
            failed=failed,
            skipped=skipped,
            remaining=remaining,
        )

    offset = reply["approach_offset_in_pallet"]
    return NextBox(
        is_complete=is_complete,
        total=total,
        placed=placed,
        failed=failed,
        skipped=skipped,
        remaining=remaining,
        seq=int(reply["seq"]),
        col=int(reply["col"]),
        row=int(reply["row"]),
        layer=int(reply["layer"]),
        pose_in_pallet=parse_pose(reply["pose_in_pallet"]),
        approach_offset_in_pallet=(float(offset["x"]), float(offset["y"]), float(offset["z"])),
        place_start_in_world=parse_pose(reply["place_start_in_world"]),
        place_end_in_world=parse_pose(reply["place_end_in_world"]),
        box_dimensions_mm=_parse_box_dimensions(reply["box_dimensions_mm"]),
    )


def _parse_placement(entry: Mapping[str, Any]) -> Placement:
    """One ``get_pack_order`` ``placements`` entry as a ``Placement``."""
    offset = entry["approach_offset_in_pallet"]
    return Placement(
        seq=int(entry["seq"]),
        col=int(entry["col"]),
        row=int(entry["row"]),
        layer=int(entry["layer"]),
        pose_in_pallet=parse_pose(entry["pose_in_pallet"]),
        pose_in_world=parse_pose(entry["pose_in_world"]),
        approach_offset_in_pallet=(float(offset["x"]), float(offset["y"]), float(offset["z"])),
        box_dimensions_mm=BoxDimensions(
            width_mm=float(entry["width_mm"]),
            length_mm=float(entry["length_mm"]),
            height_mm=float(entry["height_mm"]),
        ),
        label=str(entry.get("label", "")),
    )


def parse_pack_order(reply: Mapping[str, Any]) -> PackOrder:
    """``get_pack_order``'s reply as a ``PackOrder``."""
    return PackOrder(
        placements=[_parse_placement(entry) for entry in reply["placements"]],
        cols=int(reply["cols"]),
        rows=int(reply["rows"]),
        layers=int(reply["layers"]),
        capacity=int(reply["capacity"]),
        quantity=int(reply["quantity"]),
        overflow=int(reply["overflow"]),
        mode=str(reply["mode"]),
        warnings=list(reply["warnings"]),
        pallet_pose=parse_pose(reply["pallet_pose"]),
        pallet_width_mm=float(reply["pallet_width_mm"]),
        pallet_length_mm=float(reply["pallet_length_mm"]),
        pallet_thickness_mm=float(reply["pallet_thickness_mm"]),
    )


def parse_placement_report(reply: Mapping[str, Any]) -> PlacementReport:
    """``report_placement``'s reply as a ``PlacementReport``."""
    return PlacementReport(
        acknowledged=bool(reply["acknowledged"]),
        next_seq=int(reply["next_seq"]),
        placed=int(reply["placed"]),
        failed=int(reply["failed"]),
        skipped=int(reply["skipped"]),
        remaining=int(reply["remaining"]),
        complete=bool(reply["complete"]),
        last_error=str(reply.get("last_error", "")),
    )


def parse_skip_result(reply: Mapping[str, Any]) -> SkipResult:
    """``skip_box``'s reply as a ``SkipResult``."""
    return SkipResult(
        skipped=int(reply["skipped"]),
        next_seq=int(reply["next_seq"]),
        placed=int(reply["placed"]),
        remaining=int(reply["remaining"]),
    )


def parse_progress(reply: Mapping[str, Any]) -> Progress:
    """``get_progress``'s reply as a ``Progress``."""
    return Progress(
        next_seq=int(reply["next_seq"]),
        done_seqs=[int(seq) for seq in reply["done_seqs"]],
        skipped_seqs=[int(seq) for seq in reply["skipped_seqs"]],
        failed_seqs=[int(seq) for seq in reply["failed_seqs"]],
        placed_count=int(reply["placed_count"]),
        failed_count=int(reply["failed_count"]),
        skipped_count=int(reply["skipped_count"]),
        total=int(reply["total"]),
        complete=bool(reply["complete"]),
    )


class SequencerClient:
    """The sequencer's DoCommand surface, as typed methods.

    Wraps whatever ``ResourceBase`` the module resolved for the
    configured ``sequencer`` name. The installed SDK's
    ``WorldStateStoreClient`` serves ``do_command``, so a
    ``rdk:service:world_state_store`` dependency needs no special
    handling here beyond being called through it.

    This class holds no state of its own. The cursor, the placed-set and
    the pack order all live in the sequencer, which is the point: a
    restart of our module does not lose a half-built pallet.
    """

    def __init__(self, resource: ResourceBase) -> None:
        self._resource = resource

    async def next_box(self) -> NextBox:
        """The slot to fill next."""
        reply = await self._resource.do_command(verb("next_box"))
        return parse_next_box(reply)

    async def pack_order(self) -> PackOrder:
        """The whole plan."""
        reply = await self._resource.do_command(verb("get_pack_order"))
        return parse_pack_order(reply)

    async def report_placement(
        self, seq: int, *, success: bool, error: str = ""
    ) -> PlacementReport:
        """Records one outcome. ``error`` is stored by the sequencer for
        later inspection and is ignored on a success."""
        reply = await self._resource.do_command(
            verb("report_placement", {"seq": seq, "success": success, "error": error})
        )
        return parse_placement_report(reply)

    async def skip_box(self, seq: int, *, reason: str = "") -> SkipResult:
        """Retires one seq without placing it, so the run moves on."""
        reply = await self._resource.do_command(verb("skip_box", {"seq": seq, "reason": reason}))
        return parse_skip_result(reply)

    async def reset_cursor(self) -> None:
        """Puts the pack cursor back to seq 1 and forgets every outcome, so
        the next `next_box` is the first slot again. The reply (`reset`,
        `next_seq`) says nothing a caller acts on, so it is not parsed."""
        await self._resource.do_command(verb("reset_cursor"))

    async def progress(self) -> Progress:
        """The done, skipped and failed sets."""
        reply = await self._resource.do_command(verb("get_progress"))
        return parse_progress(reply)

    async def set_box_transform(
        self, seq: int, pose: Pose, *, parent: str = "", color: Mapping[str, Any] | None = None
    ) -> str:
        """Publishes where the box ACTUALLY ended up, and returns the
        UUID the sequencer minted for it.

        This is the verb the whole plan exists to exercise. In the demo a
        placed box is drawn where the plan said it would go; here we
        measure it in the simulator after the physics has settled and
        publish that instead, so the difference between the two is
        visible rather than assumed.

        ``parent`` defaults to the sequencer's own ``observer_frame``
        when empty. A fresh UUID is minted per call, so repeated calls
        for one seq are how a box is watched settling rather than an
        error.
        """
        argument: dict[str, Any] = {
            "seq": seq,
            "pose": {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "o_x": pose.o_x,
                "o_y": pose.o_y,
                "o_z": pose.o_z,
                "theta": pose.theta,
            },
        }
        if parent:
            argument["parent"] = parent
        if color is not None:
            argument["color"] = dict(color)
        reply = await self._resource.do_command(verb("set_box_transform", argument))
        return str(reply["uuid"])

    async def clear_box_transform(self, seq: int) -> None:
        """Drops the measured pose for one seq. A seq that was reported
        placed falls back to the sequencer's canonical on-pallet
        rendering; one that was not disappears from the viewer."""
        await self._resource.do_command(verb("clear_box_transform", {"seq": seq}))
