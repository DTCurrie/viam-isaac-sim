"""The sorting cell's geometry, the one source of truth the fragment and the
pick client copy their literals from. All values are world-frame millimetres
with the arm base at the origin; tests/test_cell_layout.py proves every
scatter corner and pad centre sits inside the arm's safe reach."""

UR20_REACH_MM = 1750.0
# zones stay inside this fraction of reach (PLAN.md §Approach)
REACH_SAFETY_FRACTION = 0.8
MAX_PLANAR_REACH_MM = UR20_REACH_MM * REACH_SAFETY_FRACTION

ARM_BASE_XY_MM = (0.0, 0.0)
ARM_BASE_Z_MM = 750.0

# each table is 1.2 x 0.8 m with its top at 750 mm; three in a row along x,
# flush edge to edge, the arm at the centre of the middle one
TABLE_DIMS_MM = (1200.0, 800.0, 750.0)
TABLE_TOP_Z_MM = 750.0
TABLE_CENTRES_MM: dict[str, tuple[float, float]] = {
    "table_source": (-1200.0, 0.0),
    "table_arm": (0.0, 0.0),
    "table_place": (1200.0, 0.0),
}

BLOCK_COLORS = ("red", "green", "blue", "yellow", "purple", "orange")
POOL_BLOCKS_PER_COLOR = 3
BLOCK_SPAWN_SIZE_MM = 60.0

# scatter zone: on the source table, inset from its edges, far corner inside
# MAX_PLANAR_REACH_MM (hypot(1350, 300) = 1383)
SCATTER_ZONE_X_MM = (-1350.0, -700.0)
SCATTER_ZONE_Y_MM = (-300.0, 300.0)

PAD_SIDE_MM = 300.0
PAD_THICKNESS_MM = 10.0
PAD_TOP_Z_MM = TABLE_TOP_Z_MM + PAD_THICKNESS_MM
# the cell's configured size-range ceiling (PLAN §Approach: pad-zone carry
# heights and slot spacing assume no block exceeds it)
MAX_BLOCK_SIZE_MM = 80.0

# GPU-measured RENDERED block hues (wrist camera, two 2026-09-03 calibration
# frames): the cell's lighting lifts dark channels ~+90..110/255 and clips
# bright ones, so rendered hue shifts off the nominal block color - color
# detectors must target THESE hues, not the nominals (red's proven #EA8D8D
# always did). Rendered hue also DRIFTS ~±5 deg between frames, so yellow
# (49-58 observed) and orange (45-50 observed) cannot be separated by hue:
# their detector bands deliberately overlap, and the conductor's proximity
# dedup + prim-color routing absorb the conflation. Re-probe after any
# lighting change.
RENDERED_BLOCK_HUE_DEG = {
    "red": 0.0,
    "green": 136.5,
    "blue": 228.0,
    "yellow": 53.0,
    "purple": 294.1,
    "orange": 48.6,
}
# rendered saturation floor that separates blocks from the arm's blue-gray
# silhouette (arm <= 0.36 measured, blue blocks >= 0.42): the blue detector's
# saturation_cutoff_pct sits between them
ARM_SILHOUETTE_MAX_SATURATION = 0.36
# spread slots on one pad: up to POOL_BLOCKS_PER_COLOR blocks per color, each
# slot's block fully on the pad and clear of its neighbours at MAX_BLOCK_SIZE
PAD_SLOT_OFFSETS_MM: tuple[tuple[float, float], ...] = ((-85.0, -70.0), (85.0, -70.0), (0.0, 75.0))
# 3 columns x 2 rows on the place table; the near column overhangs the flush
# middle table by 30 mm (same height, seamless), the far centre sits at
# hypot(1380, 180) = 1392, inside MAX_PLANAR_REACH_MM
PAD_CENTRES_MM: dict[str, tuple[float, float]] = {
    "red": (720.0, -180.0),
    "green": (1050.0, -180.0),
    "blue": (1380.0, -180.0),
    "yellow": (720.0, 180.0),
    "purple": (1050.0, 180.0),
    "orange": (1380.0, 180.0),
}

# the place-pad zone: the axis-aligned envelope of all six pads, the region
# the conductor keeps the carry out of once blocks stand on the pads
PLACE_ZONE_X_MM = (
    min(x for x, _y in PAD_CENTRES_MM.values()) - PAD_SIDE_MM / 2.0,
    max(x for x, _y in PAD_CENTRES_MM.values()) + PAD_SIDE_MM / 2.0,
)
PLACE_ZONE_Y_MM = (
    min(y for _x, y in PAD_CENTRES_MM.values()) - PAD_SIDE_MM / 2.0,
    max(y for _x, y in PAD_CENTRES_MM.values()) + PAD_SIDE_MM / 2.0,
)

# park grid on the floor behind the cell (-y of the tables): one column per
# color, one row per pool index, never inside any zone the arm works
PARK_COLUMN_X_MM = (-1250.0, -750.0, -250.0, 250.0, 750.0, 1250.0)
PARK_ROW_Y_MM = (-1000.0, -1200.0, -1400.0)

# side camera: at the source table's far (-x) end looking back toward the
# arm, 150 mm above the table top, aimed at the scatter-zone centre (the
# fragment's frame orientation must parallel the lens->centre ray,
# tests/test_fragment.py). Moved from the +y edge 2026-09-03 (user request):
# facing the arm keeps the scatter zone unobstructed in its view.
SIDE_CAMERA_POSITION_MM = (-1750.0, 0.0, 900.0)
SCATTER_CENTRE_MM = (
    (SCATTER_ZONE_X_MM[0] + SCATTER_ZONE_X_MM[1]) / 2.0,
    (SCATTER_ZONE_Y_MM[0] + SCATTER_ZONE_Y_MM[1]) / 2.0,
    TABLE_TOP_Z_MM,
)

# scene overview: pulled back far enough to frame all three tables
SCENE_CAMERA_POSITION_MM = (3200.0, 3200.0, 2600.0)
SCENE_CAMERA_TARGET_M = (0.0, 0.0, 0.3)


def pool_block_name(color: str, index: int) -> str:
    """The fragment prop name of pool block ``index`` (1-based) of ``color``."""
    return f"block_{color}_{index}"


def pad_name(color: str) -> str:
    """The fragment prop name of ``color``'s place pad."""
    return f"place_pad_{color}"


def scatter_corners_mm() -> list[tuple[float, float]]:
    """The scatter zone's four table-top corners."""
    return [(x, y) for x in SCATTER_ZONE_X_MM for y in SCATTER_ZONE_Y_MM]


def park_positions_mm() -> dict[str, tuple[float, float]]:
    """Every pool block's parked floor position: block name -> (x, y),
    column by color, row by pool index."""
    return {
        pool_block_name(color, index): (PARK_COLUMN_X_MM[column], PARK_ROW_Y_MM[index - 1])
        for column, color in enumerate(BLOCK_COLORS)
        for index in range(1, POOL_BLOCKS_PER_COLOR + 1)
    }


_MM_PER_M = 1000.0


def scatter_region_m() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The scatter zone as a metre-space region on the table top, the shape
    the sim manager's placement code consumes."""
    return (
        (
            SCATTER_ZONE_X_MM[0] / _MM_PER_M,
            SCATTER_ZONE_Y_MM[0] / _MM_PER_M,
            TABLE_TOP_Z_MM / _MM_PER_M,
        ),
        (
            SCATTER_ZONE_X_MM[1] / _MM_PER_M,
            SCATTER_ZONE_Y_MM[1] / _MM_PER_M,
            TABLE_TOP_Z_MM / _MM_PER_M,
        ),
    )


def park_positions_m() -> dict[str, tuple[float, float]]:
    """park_positions_mm in metres: block name -> floor (x, y)."""
    return {name: (x / _MM_PER_M, y / _MM_PER_M) for name, (x, y) in park_positions_mm().items()}
