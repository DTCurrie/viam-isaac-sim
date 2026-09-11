"""The prop scatter engine: deterministic random placement and sizing for
world component props, and the pool-block scatter_cell/clear_cell helpers,
whose prop names, region and park grid come from the caller's payload.
"""

from __future__ import annotations

import math
import random
from collections.abc import Container, Mapping, Sequence
from typing import Any, NamedTuple

from .spatial import Quat, Vec3, _as_quat, quat_from_euler_deg

DEFAULT_MIN_SEPARATION_M = 0.15
# scatter_cell's own floor: DEFAULT_MIN_SEPARATION_M is infeasible for a
# dense 18-block pool inside the shipped cell's scatter region (0.15m fails
# every seed, 0.12m holds 0 failures across 3000), comfortably above the
# footprint-based edge clearance (0.07m for two 60mm blocks) with margin
POOL_SCATTER_MIN_SEPARATION_M = 0.12
# sized props (dynamic-blocks): pairwise face gap beyond the two
# footprints, and how far above the support face a placed prop rests
PROP_EDGE_CLEARANCE_M = 0.01
PROP_REST_EPSILON_M = 0.0005
RANDOMIZE_MAX_ATTEMPTS = 100  # per prop, within one layout attempt
# a dense-but-feasible request can strand the LAST prop no matter how many
# single-prop draws it gets (observed at seed 6) - redraw the whole layout
RANDOMIZE_LAYOUT_RESTARTS = 50


class PropGeometry(NamedTuple):
    """One prop's oriented box at its CURRENT world pose.

    ``box_dims_m`` is the full edge length per axis: exact for cube props
    (size x scale). A usd prop reports its optional ``box_dims`` config
    attr, else (0, 0, 0) = unknown.
    """

    name: str
    box_dims_m: tuple[float, float, float]
    position_m: Vec3
    orientation_wxyz: Quat
    color: tuple[float, float, float] | None
    fixed: bool


class RandomizeResult(NamedTuple):
    """What randomize_props changed: each named prop's new center position
    and its full box dims after the call (freshly drawn where a size range
    covered it, its current dims otherwise)."""

    positions_m: dict[str, Vec3]
    dims_m: dict[str, tuple[float, float, float]]


class ScatterCellResult(NamedTuple):
    """What scatter_cell drew and placed: the drawn roster only (undrawn
    pool blocks are reported by name in ``parked``, not by pose/size)."""

    seed: int
    counts: dict[str, int]
    positions_m: dict[str, Vec3]
    sizes_m: dict[str, tuple[float, float, float]]
    parked: list[str]


class ClearCellResult(NamedTuple):
    """clear_cell's result: every pool block, now parked."""

    parked: list[str]


def _pool_block_names(names_by_color: Mapping[str, Sequence[str]]) -> list[str]:
    """All pool block prop names, colors in ``names_by_color`` iteration
    order, each color's names in list order."""
    return [name for names in names_by_color.values() for name in names]


def _scattered_block_names(
    names_by_color: Mapping[str, Sequence[str]], counts: dict[str, int]
) -> list[str]:
    """The drawn roster's names for a scatter_cell count draw, in
    names_by_color/pool-index order (the order sizes then positions draw
    in, and the order they place in)."""
    return [name for color, names in names_by_color.items() for name in names[: counts[color]]]


def _validate_pool_counts_override(
    names_by_color: Mapping[str, Sequence[str]], counts: dict[str, int]
) -> None:
    unknown = set(counts) - set(names_by_color)
    if unknown:
        raise ValueError(f"scatter_cell: counts names not in names_by_color: {sorted(unknown)}")
    for color, value in counts.items():
        pool_size = len(names_by_color[color])
        if not (0 <= value <= pool_size):
            raise ValueError(f"scatter_cell: counts[{color!r}]={value} out of [0, {pool_size}]")


def _draw_pool_counts(
    rng: random.Random,
    names_by_color: Mapping[str, Sequence[str]],
    counts: dict[str, int] | None,
) -> dict[str, int]:
    """Per-color drawn count, joining the seeded stream ahead of sizes and
    positions (scatter_cell contract): one ``rng.randint(1,
    len(names_by_color[color]))`` per color in ``names_by_color`` order,
    with no draw consumed for a color given in ``counts``."""
    if counts:
        _validate_pool_counts_override(names_by_color, counts)
    drawn: dict[str, int] = {}
    for color, names in names_by_color.items():
        if counts is not None and color in counts:
            drawn[color] = counts[color]
        else:
            drawn[color] = rng.randint(1, len(names))
    return drawn


def _require_pool_blocks(
    available: Container[str], names_by_color: Mapping[str, Sequence[str]], verb: str
) -> list[str]:
    """Every pool block name in ``names_by_color``, or ValueError naming
    whichever are missing from ``available``."""
    pool_names = _pool_block_names(names_by_color)
    missing = [name for name in pool_names if name not in available]
    if missing:
        raise ValueError(f"{verb}: missing pool blocks: {sorted(missing)}")
    return pool_names


def _require_park_positions(
    pool_names: list[str], park_positions_m: Mapping[str, tuple[float, float]], verb: str
) -> None:
    """Every ``pool_names`` entry has a park spot in ``park_positions_m``,
    or ValueError naming whichever are missing."""
    missing = [name for name in pool_names if name not in park_positions_m]
    if missing:
        raise ValueError(f"{verb}: missing park_positions_m: {sorted(missing)}")


def _park_pose_m(dims: tuple[float, float, float], park_xy: tuple[float, float]) -> Vec3:
    """A pool block's park pose: the seam's floor (x, y), z = half its
    CURRENT height + the contact-offset convention."""
    x, y = park_xy
    return (x, y, dims[2] / 2.0 + PROP_REST_EPSILON_M)


def prop_spawn_orientation(prop: dict[str, Any]) -> Quat:
    """The (w,x,y,z) quaternion a prop spawns with.

    ``orientation_wxyz`` wins over ``orientation_rpy_deg`` (extrinsic
    x-y-z euler, degrees). Identity when neither is set.
    """
    if prop.get("orientation_wxyz") is not None:
        return _as_quat(prop["orientation_wxyz"])
    rpy = prop.get("orientation_rpy_deg")
    if rpy is not None:
        roll, pitch, yaw = (float(v) for v in rpy)
        return quat_from_euler_deg(roll, pitch, yaw)
    return (1.0, 0.0, 0.0, 0.0)


def prop_box_dims(prop: dict[str, Any]) -> tuple[float, float, float]:
    """Full edge lengths (m) of a prop's box: cube = size x scale
    (defaults 0.05 and [1, 1, 1]). usd = its ``box_dims`` attr else zeros
    (unknown, and the README obstacle recipe tells users to set it)."""
    if str(prop.get("type", "cube")) == "usd":
        dims = prop.get("box_dims")
        if dims is None:
            return (0.0, 0.0, 0.0)
        return (float(dims[0]), float(dims[1]), float(dims[2]))
    size = float(prop.get("size", 0.05))
    scale = prop.get("scale") or (1.0, 1.0, 1.0)
    return (size * float(scale[0]), size * float(scale[1]), size * float(scale[2]))


def _prop_footprint_m(dims: tuple[float, float, float]) -> float:
    """A prop's placement footprint (sized props): the larger of
    its x/y edges, used for edge-aware separation."""
    return max(dims[0], dims[1])


def _place_props(
    dims_by_name: dict[str, tuple[float, float, float]],
    region: tuple[Vec3, Vec3],
    rng: random.Random,
    min_separation_m: float,
) -> dict[str, Vec3]:
    """The draw loop behind ``sample_prop_positions``, taking an
    already-seeded ``rng`` so a caller can consume size draws from the
    same stream first (sized props, dynamic-blocks)."""
    (x0, y0, z0), (x1, y1, z1) = region
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    face_z = (float(z0) + float(z1)) / 2.0
    for name, dims in dims_by_name.items():
        half_x, half_y = dims[0] / 2.0, dims[1] / 2.0
        if lo_x + half_x > hi_x - half_x or lo_y + half_y > hi_y - half_y:
            raise ValueError(f"randomize_props: region cannot hold {name!r}'s footprint")
    for _restart in range(RANDOMIZE_LAYOUT_RESTARTS):
        placed: dict[str, Vec3] = {}
        footprints: dict[str, float] = {}
        for name, dims in dims_by_name.items():
            half_x, half_y = dims[0] / 2.0, dims[1] / 2.0
            footprint = _prop_footprint_m(dims)
            for _ in range(RANDOMIZE_MAX_ATTEMPTS):
                x = rng.uniform(lo_x + half_x, hi_x - half_x)
                y = rng.uniform(lo_y + half_y, hi_y - half_y)
                if all(
                    math.hypot(x - px, y - py)
                    >= max(
                        min_separation_m,
                        (footprint + footprints[pname]) / 2.0 + PROP_EDGE_CLEARANCE_M,
                    )
                    for pname, (px, py, _pz) in placed.items()
                ):
                    placed[name] = (x, y, face_z + dims[2] / 2.0 + PROP_REST_EPSILON_M)
                    footprints[name] = footprint
                    break
            else:
                break  # this layout stranded ``name``: redraw everything
        else:
            return placed
    raise ValueError(
        f"randomize_props: no layout for {sorted(dims_by_name)} after "
        f"{RANDOMIZE_LAYOUT_RESTARTS} layout attempts. Widen the region, "
        "drop props, or lower min_separation"
    )


def sample_prop_positions(
    dims_by_name: dict[str, tuple[float, float, float]],
    region: tuple[Vec3, Vec3],
    seed: int,
    min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
) -> dict[str, Vec3]:
    """Deterministic prop placement on a table's top face.

    ``region`` is ((x0, y0, z), (x1, y1, z)) in meters, world frame: the
    rectangle of the top face the props' footprints must stay inside, at
    the face's height z (the two z values are averaged). Each prop lands
    with its footprint (center +/- dims/2 in x and y) inside the
    rectangle, its center at least max(``min_separation_m``, the two
    props' edge-aware gap) from every other placed center in the x/y
    plane (sized props: edge-aware = (footprint_a +
    footprint_b) / 2 + PROP_EDGE_CLEARANCE_M, footprint = max x/y dim),
    and its center z at face z + dims_z / 2 + PROP_REST_EPSILON_M so it
    rests just above the face.

    Same inputs -> the same placements on every call: draws come from one
    ``random.Random(seed)`` stream and props place in ``dims_by_name``
    insertion order. A layout that strands a prop (no clear spot within
    RANDOMIZE_MAX_ATTEMPTS draws) is redrawn wholesale, up to
    RANDOMIZE_LAYOUT_RESTARTS times, so a dense-but-feasible request still
    converges. Raises ValueError when the region cannot hold a footprint
    or no layout fits.
    """
    return _place_props(dims_by_name, region, random.Random(seed), min_separation_m)


def _validate_size_range_names(
    names: list[str], size_range_m: dict[str, tuple[float, float]] | None
) -> None:
    if not size_range_m:
        return
    unknown = set(size_range_m) - set(names)
    if unknown:
        raise ValueError(f"randomize_props: size_range_m names not in names: {sorted(unknown)}")


def _require_cube_prop(name: str, spec: dict[str, Any]) -> None:
    if str(spec.get("type", "cube")) != "cube":
        raise ValueError(f"randomize_props: size_range_m on non-cube prop {name!r}")


def _draw_sizes_and_positions(
    names: list[str],
    dims_by_name: dict[str, tuple[float, float, float]],
    region: tuple[Vec3, Vec3],
    seed: int,
    min_separation_m: float,
    size_range_m: dict[str, tuple[float, float]] | None,
    rng: random.Random | None = None,
) -> tuple[dict[str, Vec3], dict[str, tuple[float, float, float]]]:
    """One ``random.Random(seed)`` stream: sizes first (``names`` order,
    only props with a range), then positions (WorldHandle.randomize_props
    contract). Returns the placements and every named prop's post-draw
    dims.

    ``rng`` lets a caller join an already-seeded stream ahead of this call
    (scatter_cell's counts-then-sizes-then-positions draw). Omitted, a
    fresh ``random.Random(seed)`` is used (randomize_props' own contract,
    unchanged)."""
    rng = rng if rng is not None else random.Random(seed)
    drawn_dims = dict(dims_by_name)
    if size_range_m:
        for name in names:
            size_range = size_range_m.get(name)
            if size_range is None:
                continue
            lo, hi = size_range
            drawn_edge = rng.uniform(lo, hi)
            drawn_dims[name] = (drawn_edge, drawn_edge, drawn_edge)
    placed = _place_props({name: drawn_dims[name] for name in names}, region, rng, min_separation_m)
    return placed, drawn_dims
