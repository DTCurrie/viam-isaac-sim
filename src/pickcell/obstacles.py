"""Motion-planner obstacles: the support slab, the pick-area keep-out,
prop-geometry obstacles, ``WorldState``, the held-block transform and the
table recipe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
    WorldState,
)

SUPPORT_OBSTACLE_SIDE_MM = 3000.0
# thick: a 20 mm slab let a discrete collision check step a link straight
# through the floor mid-swing (GPU run 7)
SUPPORT_OBSTACLE_THICKNESS_MM = 200.0


def support_obstacle(support_z_mm: float) -> Geometry:
    """The surface the block rests on, as a thin slab whose top is at
    support_z_mm. Without it the planner happily swings the fingertips through
    the floor on a joint-space descent (GPU run 18)."""
    return Geometry(
        center=Pose(
            x=0.0,
            y=0.0,
            z=support_z_mm - SUPPORT_OBSTACLE_THICKNESS_MM / 2.0,
            o_x=0.0,
            o_y=0.0,
            o_z=1.0,
            theta=0.0,
        ),
        box=RectangularPrism(
            dims_mm=Vector3(
                x=SUPPORT_OBSTACLE_SIDE_MM,
                y=SUPPORT_OBSTACLE_SIDE_MM,
                z=SUPPORT_OBSTACLE_THICKNESS_MM,
            )
        ),
        label="support",
    )


def obstacles_from_prop_geometries(
    geometries: Sequence[Mapping[str, Any]], exclude: set[str]
) -> list[Geometry]:
    """Sim obstacle boxes from the world's ``prop_geometries`` DoCommand
    result: one box per entry, skipping ``exclude`` (the block about to be
    grasped) and any entry whose dims are all zero (an unknown-size usd prop
    the module could not infer a box for)."""
    obstacles: list[Geometry] = []
    for geometry in geometries:
        name = geometry["name"]
        if name in exclude:
            continue
        dim_x, dim_y, dim_z = geometry["box_dims_mm"]
        if dim_x == 0.0 and dim_y == 0.0 and dim_z == 0.0:
            continue
        pose_mm = geometry["pose_in_world_mm"]
        obstacles.append(
            Geometry(
                center=Pose(
                    x=pose_mm["x"],
                    y=pose_mm["y"],
                    z=pose_mm["z"],
                    o_x=pose_mm["o_x"],
                    o_y=pose_mm["o_y"],
                    o_z=pose_mm["o_z"],
                    theta=pose_mm["theta"],
                ),
                box=RectangularPrism(dims_mm=Vector3(x=dim_x, y=dim_y, z=dim_z)),
                label=name,
            )
        )
    return obstacles


def table_recipe_unless_served(
    table: Geometry | None, sim_obstacles: Sequence[Geometry]
) -> Geometry | None:
    """The --table recipe box, or None when the live scene already serves a
    geometry with the same label - the motion service rejects two WorldState
    geometries sharing a name, and the P5 cell's table arrives live via
    ``prop_geometries``."""
    if table is None:
        return None
    if any(obstacle.label == table.label for obstacle in sim_obstacles):
        return None
    return table


# extra size on the held block's planning box: absorbs the ~4 mm believed-vs-
# physical TCP gap (ARM-10) plus tracking error, so a grazing plan cannot
# become a touch
HELD_BLOCK_PADDING_MM = 20.0

# the pick-area keep-out (GPU run 12): a no-fly box over the scatter region
# lets the carry plan FREELY (fast joint-space motion) instead of crawling
# along a constrained linear line - the planner simply may not enter the
# airspace where blocks live. Height above the support = block tops (60) +
# held-cube hang + margin; the region's own z is the support it stands on.
KEEPOUT_HEIGHT_MM = 130.0
KEEPOUT_MARGIN_MM = 50.0
# TCP height above the support whose held padded cube bottom (TCP - 9 offset
# - 40 half-box) clears the keep-out ceiling with ~20 mm to spare
CARRY_CLEAR_ABOVE_SUPPORT_MM = 200.0


def pick_area_keepout(
    region_mm: tuple[Sequence[float], Sequence[float]],
    height_mm: float = KEEPOUT_HEIGHT_MM,
    label: str = "pick_area_keepout",
) -> Geometry:
    """The carry-phase no-fly box over a work region: from the region's
    own z (the support surface) up ``height_mm`` (KEEPOUT_HEIGHT_MM by
    default; phase 4 passes the measured-tallest-derived height), grown by
    KEEPOUT_MARGIN_MM sideways. ``label`` names the box in planner
    violation messages (the place-pad keep-out passes its own)."""
    (x0, y0, z0), (x1, y1, _z1) = region_mm
    lo_x = min(x0, x1) - KEEPOUT_MARGIN_MM
    hi_x = max(x0, x1) + KEEPOUT_MARGIN_MM
    lo_y = min(y0, y1) - KEEPOUT_MARGIN_MM
    hi_y = max(y0, y1) + KEEPOUT_MARGIN_MM
    return Geometry(
        center=Pose(
            x=(lo_x + hi_x) / 2.0,
            y=(lo_y + hi_y) / 2.0,
            z=z0 + height_mm / 2.0,
            o_x=0.0,
            o_y=0.0,
            o_z=1.0,
            theta=0.0,
        ),
        box=RectangularPrism(dims_mm=Vector3(x=hi_x - lo_x, y=hi_y - lo_y, z=height_mm)),
        label=label,
    )


def pad_top_centre_mm(
    geometries: Sequence[Mapping[str, Any]], pad_name: str
) -> tuple[float, float, float] | None:
    """The place pad's top-face centre (x, y, top z) in mm from a
    ``prop_geometries`` result, or None when the scene has no such prop."""
    for geometry in geometries:
        if geometry["name"] == pad_name:
            pose = geometry["pose_in_world_mm"]
            return (pose["x"], pose["y"], pose["z"] + geometry["box_dims_mm"][2] / 2.0)
    return None


# the sorting cell's scatter zone on the source table; the arm base sits at
# the world origin, a ur20 reaches ~1750 mm and the zone stays inside 80% of
# that (1400 mm)
#
# NOTE: unlike the other cell-default constants (DEFAULT_LOOK_XY_MM,
# TABLE_CENTER_MM/DIMS_MM), these live here rather than the script: the
# pipeline's own randomize fallback (_sim_obstacles) calls reachable_region_mm
# directly, so the function and its constants must be importable from the
# same module PickPipeline lives beside. examples/pick_red_block.py re-exports
# both names so the phase-1 seam cross-check against cell_layout still holds.
REACHABLE_REGION_X_MM = (-1350.0, -700.0)
REACHABLE_REGION_Y_MM = (-300.0, 300.0)


def reachable_region_mm(face_z_mm: float = 0.0) -> tuple[list[float], list[float]]:
    """The scatter rectangle a randomized block stays pickable in: inside the
    ur20's reach envelope, resting on ``face_z_mm``. The table-footprint
    recipe (randomize_region_mm) suits the source table's own footprint
    instead when its far corners fall beyond reach (GPU run 4: a block at
    (846, 242) = 880 mm radius could not be looked at or picked)."""
    (x0, x1), (y0, y1) = REACHABLE_REGION_X_MM, REACHABLE_REGION_Y_MM
    return ([x0, y0, face_z_mm], [x1, y1, face_z_mm])


def world_state(
    table: Geometry | None,
    other_blocks: Sequence[Geometry] = (),
    support: Geometry | None = None,
    transforms: Sequence[Transform] = (),
) -> WorldState:
    """A WorldState whose obstacles are the support surface, the table (when
    the scene has one) and any distractor blocks (SCN-5), all in the world
    frame. ``transforms`` carries the held block while grasping (DEC-14), so
    the planner treats it as geometry attached to the gripper."""
    return WorldState(
        obstacles=[
            GeometriesInFrame(
                reference_frame="world",
                geometries=[g for g in (support, table, *other_blocks) if g is not None],
            )
        ],
        transforms=list(transforms),
    )


def held_block_transform(
    block_name: str, block_size_mm: float, gripper_name: str, centre_below_tcp_mm: float = 0.0
) -> Transform:
    """DEC-14(a): once grasped, the block's pose is reported to the motion
    service as a Transform parented to the gripper (the gripper's own
    GetGeometries cannot carry it - see DEC-14's rationale).
    ``centre_below_tcp_mm`` hangs the box below the gripper frame by the
    grasp offset, so the planner carries it at its real height (GPU run 9:
    an unmodelled held cube clipped a distractor the arm itself cleared)."""
    origin = Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    held_centre = Pose(x=0.0, y=0.0, z=-centre_below_tcp_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    return Transform(
        reference_frame=block_name,
        pose_in_observer_frame=PoseInFrame(reference_frame=gripper_name, pose=held_centre),
        physical_object=Geometry(
            center=origin,
            box=RectangularPrism(
                dims_mm=Vector3(x=block_size_mm, y=block_size_mm, z=block_size_mm)
            ),
            label=block_name,
        ),
    )
