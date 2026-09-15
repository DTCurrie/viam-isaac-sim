"""Writes the ``--regions`` file ``gpu_checklist_photoreal.py``'s hue item reads.

The hue item samples one pixel box per block colour. Finding those boxes by
looking at a frame is what left phase 1 without a regions file at all, so this
derives them instead: each block's world pose comes from the world's
``prop_geometries``, the camera's own intrinsics and a four-probe
camera-to-world affine turn that pose into pixels, and the box lands on the
block's top face.

Every box is then sampled with the camera's ``sample_color`` DoCommand and
checked against that block's configured colour. A box the arm or a shadow
covers fails the saturation floor and the next candidate block is tried, so a
projection that lands off the block fails here rather than silently producing
a wrong hue on the checklist.

Usage (the wrist camera the colour detectors classify, from the conductor's
own census look poses)::

    PYTHONPATH=src python examples/make_hue_regions.py --camera wrist-cam --look

Options. ``--out`` is the regions path, default ``regions.json``. ``--camera``
takes a camera name, or ``auto`` to try each in turn. ``--look`` drives the
camera to the census look poses first, trying each until one sees every
colour. ``--scatter-seed N`` runs ``scatter_cell`` at that seed first, which
puts at least one block of every colour on the source table.

Reads ``VIAM_API_KEY``, ``VIAM_API_KEY_ID`` and ``VIAM_MACHINE_ADDRESS`` from
the environment. Exit code 0 when the regions file is written, 1 when no view
confirmed a block of every colour.
"""

from __future__ import annotations

import argparse
import asyncio
import colorsys
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

BLOCK_NAME = re.compile(r"^block_([a-z]+)_\d+$")
CAMERA_CANDIDATES = ("wrist-cam", "side-cam", "scene-cam")
DEFAULT_OUT_PATH = "regions.json"
DEFAULT_WORLD_NAME = "isaac-world"
DEFAULT_ARM_NAME = "pick-arm"
# past this much accumulated rotation on one joint, the next motion plan
# through it is refused outright, so the winding is cleared first
MAX_JOINT_WIND_DEG = 200.0
# the transform probe offset, big enough that the pose round trip's own
# rounding does not dominate the measured axis
PROBE_OFFSET_MM = 100.0
# the sampled box is this fraction of the block's projected top face, so the
# patch stays clear of the edges and their rounded shading
BOX_FRACTION = 0.4
MIN_BOX_PX = 4
# a block projecting smaller than this is too far away to sample cleanly
MIN_PROJECTED_PX = 10.0
# how far a sample may sit from the block's configured hue before the box is
# treated as landing somewhere other than that block
HUE_TOLERANCE_DEG = 45.0
# a painted block face renders well above the arm's blue-gray silhouette
# (cell_layout.ARM_SILHOUETTE_MAX_SATURATION is 0.36). Below this the patch is
# the arm, a shadow or the table, and its hue carries no information
MIN_BLOCK_SATURATION = 0.25

BEHIND_CAMERA = "behind the camera"
TOO_SMALL = "too small"
OUTSIDE_FRAME = "outside the frame"


@dataclass(frozen=True)
class Intrinsics:
    """A camera's pinhole parameters, as ``get_properties`` reports them."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Candidate:
    """One block that projects into frame, and the box to sample on it."""

    name: str
    box: tuple[int, int, int, int]
    colour_rgb: tuple[float, float, float]
    projected_px: float
    centre_offset_px: float


@dataclass(frozen=True)
class Measurement:
    """A sampled candidate: what the renderer actually put on those pixels."""

    candidate: Candidate
    srgb_hex: str
    mean_rgb: tuple[float, float, float]
    hue_deg: float
    saturation: float


def hue_degrees(rgb: Sequence[float]) -> float:
    """Hue in degrees of an ``[r, g, b]`` triple, either 0-1 or 0-255."""
    scale = 255.0 if max(rgb) > 1.0 else 1.0
    red, green, blue = (float(v) / scale for v in rgb)
    return colorsys.rgb_to_hsv(red, green, blue)[0] * 360.0


def saturation(rgb: Sequence[float]) -> float:
    """HSV saturation of an ``[r, g, b]`` triple, either 0-1 or 0-255."""
    scale = 255.0 if max(rgb) > 1.0 else 1.0
    red, green, blue = (float(v) / scale for v in rgb)
    return colorsys.rgb_to_hsv(red, green, blue)[1]


def hue_gap_degrees(one_deg: float, other_deg: float) -> float:
    """The shorter way round the hue circle between two hues."""
    gap = abs(one_deg - other_deg) % 360.0
    return min(gap, 360.0 - gap)


def project(
    point_world_mm: Sequence[float],
    rotation: np.ndarray,
    translation_mm: np.ndarray,
    intrinsics: Intrinsics,
) -> tuple[float, float, float]:
    """``(u px, v px, depth mm)`` of a world point. A point at or behind the
    image plane returns NaN pixels, since it has no projection."""
    point_camera = rotation.T @ (np.asarray(point_world_mm, dtype=float) - translation_mm)
    x_mm, y_mm, depth_mm = (float(v) for v in point_camera)
    if depth_mm <= 0:
        return (float("nan"), float("nan"), depth_mm)
    return (
        intrinsics.fx * x_mm / depth_mm + intrinsics.cx,
        intrinsics.fy * y_mm / depth_mm + intrinsics.cy,
        depth_mm,
    )


def box_around(
    u_px: float, v_px: float, projected_px: float, intrinsics: Intrinsics
) -> tuple[int, int, int, int] | None:
    """The sample box centred on ``(u, v)``, or None when it would run off the
    frame. ``sample_color`` needs ``0 <= x0 < x1 <= width``, so a box the edge
    would clip is no box at all."""
    half = max(MIN_BOX_PX, projected_px * BOX_FRACTION) / 2.0
    x0, x1 = round(u_px - half), round(u_px + half)
    y0, y1 = round(v_px - half), round(v_px + half)
    if x0 < 0 or y0 < 0 or x1 > intrinsics.width or y1 > intrinsics.height:
        return None
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def rank_candidates(
    geometries: Sequence[Mapping[str, Any]],
    rotation: np.ndarray,
    translation_mm: np.ndarray,
    intrinsics: Intrinsics,
) -> tuple[list[Candidate], dict[str, int]]:
    """Pure. Every geometry that projects into frame, biggest and most central
    first, plus a count of why the others were dropped. Biggest first because a
    larger projection samples more block and less edge."""
    candidates: list[Candidate] = []
    rejected = {BEHIND_CAMERA: 0, TOO_SMALL: 0, OUTSIDE_FRAME: 0}
    for geometry in geometries:
        pose = geometry["pose_in_world_mm"]
        dims_mm = [float(d) for d in geometry["box_dims_mm"]]
        top_face_mm = (
            float(pose["x"]),
            float(pose["y"]),
            float(pose["z"]) + dims_mm[2] / 2.0,
        )
        u_px, v_px, depth_mm = project(top_face_mm, rotation, translation_mm, intrinsics)
        if not np.isfinite(u_px):
            rejected[BEHIND_CAMERA] += 1
            continue
        projected_px = intrinsics.fx * dims_mm[0] / depth_mm
        if projected_px < MIN_PROJECTED_PX:
            rejected[TOO_SMALL] += 1
            continue
        box = box_around(u_px, v_px, projected_px, intrinsics)
        if box is None:
            rejected[OUTSIDE_FRAME] += 1
            continue
        colour = geometry["color"]
        candidates.append(
            Candidate(
                name=str(geometry["name"]),
                box=box,
                colour_rgb=(float(colour[0]), float(colour[1]), float(colour[2])),
                projected_px=projected_px,
                centre_offset_px=abs(u_px - intrinsics.cx) + abs(v_px - intrinsics.cy),
            )
        )
    candidates.sort(key=lambda c: (-c.projected_px, c.centre_offset_px))
    return candidates, rejected


def sample_verdict(measurement: Measurement) -> str:
    """Pure. ``"ok"`` when the sample reads as its own block, else why not."""
    expected_deg = hue_degrees(measurement.candidate.colour_rgb)
    gap_deg = hue_gap_degrees(expected_deg, measurement.hue_deg)
    if measurement.saturation < MIN_BLOCK_SATURATION:
        return f"REJECT washed out (saturation {measurement.saturation:.2f})"
    if gap_deg > HUE_TOLERANCE_DEG:
        return f"REJECT hue {gap_deg:.0f} deg from the configured colour"
    return "ok"


def rendered_hue_block(measurements: Mapping[str, Measurement], colours: Sequence[str]) -> str:
    """The measurements as a ``cell_layout.RENDERED_BLOCK_HUE_DEG`` literal."""
    lines = ["RENDERED_BLOCK_HUE_DEG = {"]
    lines += [f'    "{colour}": {measurements[colour].hue_deg:.1f},' for colour in colours]
    lines.append("}")
    return "\n".join(lines)


def blocks_by_colour(
    geometries: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Pure. The pool blocks among ``geometries``, keyed by their colour."""
    out: dict[str, list[Mapping[str, Any]]] = {}
    for geometry in geometries:
        match = BLOCK_NAME.match(str(geometry["name"]))
        if match:
            out.setdefault(match.group(1), []).append(geometry)
    return out


def census_look_points_mm() -> tuple[tuple[float, float], ...]:
    """The conductor's own three census look points, so the hues are sampled
    from the vantages the colour detectors actually run from."""
    from isaac_module.models.conductor import _census_look_points_mm

    return _census_look_points_mm()


async def camera_to_world(robot: Any, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(rotation 3x3, translation mm)``: the four-probe affine
    ``pickcell.scanners.camera_world_transform_mm`` builds, so this agrees with
    what the pick pipeline already trusts on this machine."""
    from viam.proto.common import Pose, PoseInFrame

    async def probe(x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
        pose_in_frame = PoseInFrame(
            reference_frame=camera_name,
            pose=Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        )
        pose = (await robot.transform_pose(pose_in_frame, "world")).pose
        return np.array([pose.x, pose.y, pose.z])

    origin = await probe(0.0, 0.0, 0.0)
    axes = [
        await probe(PROBE_OFFSET_MM, 0.0, 0.0),
        await probe(0.0, PROBE_OFFSET_MM, 0.0),
        await probe(0.0, 0.0, PROBE_OFFSET_MM),
    ]
    rotation = np.column_stack([(axis - origin) / PROBE_OFFSET_MM for axis in axes])
    return rotation, origin


async def intrinsics_of(camera: Any) -> Intrinsics:
    parameters = (await camera.get_properties()).intrinsic_parameters
    return Intrinsics(
        width=parameters.width_px,
        height=parameters.height_px,
        fx=parameters.focal_x_px,
        fy=parameters.focal_y_px,
        cx=parameters.center_x_px,
        cy=parameters.center_y_px,
    )


async def measure_colours(
    robot: Any, camera_name: str, blocks: Mapping[str, list[Mapping[str, Any]]]
) -> dict[str, Measurement]:
    """Sample each colour's candidates in order, keeping the first that reads
    as that block. Prints one line per sample, accepted or not."""
    from viam.components.camera import Camera

    camera = Camera.from_robot(robot, camera_name)
    intrinsics = await intrinsics_of(camera)
    rotation, translation_mm = await camera_to_world(robot, camera_name)

    confirmed: dict[str, Measurement] = {}
    for colour, geometries in sorted(blocks.items()):
        candidates, rejected = rank_candidates(geometries, rotation, translation_mm, intrinsics)
        dropped = {reason: count for reason, count in rejected.items() if count}
        if not candidates:
            print(f"  {colour:7s} no candidate projects into frame, dropped {dropped}")
            continue
        for candidate in candidates:
            sample = await camera.do_command(
                {"command": "sample_color", "region": list(candidate.box)}
            )
            sampled = cast("Sequence[float]", sample["mean_rgb"])
            mean_rgb = tuple(float(v) for v in sampled)
            measurement = Measurement(
                candidate=candidate,
                srgb_hex=str(sample["srgb_hex"]),
                mean_rgb=(mean_rgb[0], mean_rgb[1], mean_rgb[2]),
                hue_deg=hue_degrees(mean_rgb),
                saturation=saturation(mean_rgb),
            )
            verdict = sample_verdict(measurement)
            print(
                f"  {colour:7s} {candidate.name:16s} box={list(candidate.box)} "
                f"{measurement.srgb_hex} hue={measurement.hue_deg:5.1f} "
                f"(configured {hue_degrees(candidate.colour_rgb):5.1f}) "
                f"saturation={measurement.saturation:.2f}  {verdict}"
            )
            if verdict == "ok":
                confirmed[colour] = measurement
                break
    return confirmed


def unwound_degrees(angle_deg: float) -> float:
    """The same orientation expressed in [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def wound_joint_indices(
    positions_deg: Sequence[float], limit_deg: float = MAX_JOINT_WIND_DEG
) -> list[int]:
    """Which joints carry more than ``limit_deg`` of accumulated rotation.

    Planning to the same pose repeatedly can add a full turn to a wrist each
    time, and once one passes its own limit every later plan is refused with
    "joint ... out of range". A sorting loop run against an arm in that state
    fails picks for reasons that look like detection failures, so this tool
    clears the winding rather than leaving it for the next run."""
    return [index for index, angle in enumerate(positions_deg) if abs(angle) > limit_deg]


async def unwind_arm(
    robot: Any, arm_name: str = DEFAULT_ARM_NAME, limit_deg: float = MAX_JOINT_WIND_DEG
) -> None:
    """Rewrite any wound joint as its equivalent angle in [-180, 180). The
    arm's pose is unchanged apart from the turns being taken out."""
    from viam.components.arm import Arm
    from viam.proto.component.arm import JointPositions

    arm = Arm.from_robot(robot, arm_name)
    positions_deg = list((await arm.get_joint_positions()).values)
    wound = wound_joint_indices(positions_deg, limit_deg)
    if not wound:
        return
    unwound = [
        unwound_degrees(angle) if index in wound else angle
        for index, angle in enumerate(positions_deg)
    ]
    await arm.move_to_joint_positions(JointPositions(values=unwound))
    print(
        f"unwound {len(wound)} joint(s): "
        + ", ".join(f"{positions_deg[i]:.1f} -> {unwound[i]:.1f} deg" for i in wound)
    )


async def look_from_census_point(robot: Any, camera_name: str, point_index: int) -> None:
    """Move ``camera_name`` to one of the conductor's census look poses: that
    point at scan height, optical axis straight down. Same motion call and
    same world state every sorting loop already makes."""
    from viam.components.generic import Generic
    from viam.proto.common import PoseInFrame
    from viam.services.motion import MotionClient

    from isaac_module import cell_layout
    from pickcell.obstacles import obstacles_from_prop_geometries, support_obstacle, world_state
    from pickcell.poses import SCAN_HEIGHT_ABOVE_SUPPORT_MM, look_pose_from

    world = Generic.from_robot(robot, DEFAULT_WORLD_NAME)
    response = await world.do_command({"command": "prop_geometries"})
    state = world_state(
        None,
        obstacles_from_prop_geometries(
            cast("Sequence[Mapping[str, Any]]", response.get("geometries", [])), set()
        ),
        support_obstacle(cell_layout.TABLE_TOP_Z_MM),
    )
    look_x_mm, look_y_mm = census_look_points_mm()[point_index]
    look_z_mm = cell_layout.TABLE_TOP_Z_MM + SCAN_HEIGHT_ABOVE_SUPPORT_MM
    # the resource list is a snapshot taken at connect, and the builtin
    # services are absent from it until a refresh (scanners.py refreshes for
    # the same reason)
    await robot.refresh()
    motion = MotionClient.from_robot(robot, "builtin")
    moved = await motion.move(
        component_name=camera_name,
        destination=PoseInFrame(
            reference_frame="world",
            pose=look_pose_from(f"{look_x_mm},{look_y_mm},{look_z_mm}"),
        ),
        world_state=state,
    )
    if not moved:
        raise RuntimeError(f"motion move of {camera_name!r} to the census look pose failed")
    print(
        f"{camera_name} at census look point {point_index + 1} of "
        f"{len(census_look_points_mm())}: "
        f"({look_x_mm:.0f}, {look_y_mm:.0f}, {look_z_mm:.0f}) mm, pointing down"
    )


async def scatter_pool(world: Any, seed: int) -> None:
    """``scatter_cell`` at ``seed``: it draws at least one block of every
    colour onto the source table. Reversible with the cell's ``clear_cell``."""
    from isaac_module import cell_layout

    x_lo, x_hi = cell_layout.SCATTER_ZONE_X_MM
    y_lo, y_hi = cell_layout.SCATTER_ZONE_Y_MM
    table_top_z = cell_layout.TABLE_TOP_Z_MM
    result = await world.do_command(
        {
            "command": "scatter_cell",
            "names_by_color": {
                colour: [
                    cell_layout.pool_block_name(colour, index)
                    for index in range(1, cell_layout.POOL_BLOCKS_PER_COLOR + 1)
                ]
                for colour in cell_layout.BLOCK_COLORS
            },
            "region": [[x_lo, y_lo, table_top_z], [x_hi, y_hi, table_top_z]],
            "park_positions_mm": {
                name: list(xy) for name, xy in cell_layout.park_positions_mm().items()
            },
            "seed": seed,
        }
    )
    print(f"scatter_cell seed={seed}: {result.get('counts', {})}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="path to write the regions JSON to")
    parser.add_argument(
        "--camera",
        default="auto",
        help=f"a camera name, or auto to try {', '.join(CAMERA_CANDIDATES)} in turn",
    )
    parser.add_argument(
        "--look",
        action="store_true",
        help="drive the camera to the conductor's census look poses first, trying each until "
        "one sees a block of every colour",
    )
    parser.add_argument(
        "--scatter-seed",
        type=int,
        default=None,
        help="scatter_cell at this seed first, so a block of every colour is on the source table",
    )
    parser.add_argument("--arm", default=DEFAULT_ARM_NAME, help="the arm --look moves and unwinds")
    parser.add_argument("--address", help="machine address (else VIAM_MACHINE_ADDRESS)")
    parser.add_argument("--api-key", help="API key (else VIAM_API_KEY)")
    parser.add_argument("--api-key-id", help="API key id (else VIAM_API_KEY_ID)")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    import os

    from viam.components.generic import Generic
    from viam.robot.client import RobotClient

    from isaac_module.cell_layout import BLOCK_COLORS

    options = RobotClient.Options.with_api_key(
        api_key=args.api_key or os.environ["VIAM_API_KEY"],
        api_key_id=args.api_key_id or os.environ["VIAM_API_KEY_ID"],
    )
    address = args.address or os.environ["VIAM_MACHINE_ADDRESS"]
    robot = await RobotClient.at_address(address, options)
    try:
        world = Generic.from_robot(robot, DEFAULT_WORLD_NAME)
        if args.scatter_seed is not None:
            await scatter_pool(world, args.scatter_seed)
        response = await world.do_command({"command": "prop_geometries"})
        geometries = cast("Sequence[Mapping[str, Any]]", response["geometries"])
        blocks = blocks_by_colour(geometries)
        print(f"pool blocks: { ({colour: len(v) for colour, v in sorted(blocks.items())}) }")

        measurements: dict[str, Measurement] = {}
        camera_used: str | None = None
        if args.look:
            camera_used = "wrist-cam" if args.camera == "auto" else args.camera
            await unwind_arm(robot, args.arm)
            try:
                for point_index in range(len(census_look_points_mm())):
                    await look_from_census_point(robot, camera_used, point_index)
                    measurements = await measure_colours(robot, camera_used, blocks)
                    missing = [c for c in BLOCK_COLORS if c not in measurements]
                    if not missing:
                        break
                    print(f"  still missing {missing}, trying the next census look point")
            finally:
                # never hand the next sorting loop a wound-up arm, whatever
                # happened above
                await unwind_arm(robot, args.arm)
        else:
            names = CAMERA_CANDIDATES if args.camera == "auto" else (args.camera,)
            for camera_name in names:
                print(f"{camera_name}:")
                measurements = await measure_colours(robot, camera_name, blocks)
                camera_used = camera_name
                if not [c for c in BLOCK_COLORS if c not in measurements]:
                    break

        missing = [c for c in BLOCK_COLORS if c not in measurements]
        if missing:
            print(
                f"FAILED: no view confirmed a block of every colour (missing {missing}). "
                "Scatter the pool, or look from another pose, and re-run."
            )
            return 1

        print("\n" + rendered_hue_block(measurements, BLOCK_COLORS))
        print(
            "detector hexes: " + ", ".join(f"{c}={measurements[c].srgb_hex}" for c in BLOCK_COLORS)
        )
        print(
            "lowest measured saturation: "
            f"{min(measurements[c].saturation for c in BLOCK_COLORS):.2f}"
        )
        with open(args.out, "w") as regions_file:
            json.dump(
                {colour: list(measurements[colour].candidate.box) for colour in BLOCK_COLORS},
                regions_file,
                indent=2,
            )
            regions_file.write("\n")
        print(f"wrote {args.out} for --camera {camera_used}")
        return 0
    finally:
        await robot.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure as a clean exit code
        print(f"FAILED: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
