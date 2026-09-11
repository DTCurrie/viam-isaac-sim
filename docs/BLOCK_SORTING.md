# The block-sorting cell

The `isaac-sim-block-sorting` fragment (source in
`fragments/isaac-sim-block-sorting.json`) is a ready-made three-table sorting
cell (`cell_layout.TABLE_CENTRES_MM`). A UR20 arm (`pick-arm`) stands on
`table_arm` at the world origin. `table_source` carries the scatter zone that
pool blocks are drawn into. `table_place` carries six 300 mm color-coded place
pads, one `place_pad_<color>` for each of `cell_layout.BLOCK_COLORS`: `red`,
`green`, `blue`, `yellow`, `purple`, `orange`. Each pad holds up to
`cell_layout.POOL_BLOCKS_PER_COLOR` (3) blocks in spread slots, and nothing
stacks. An 18-block pool (`block_<color>_1`, `_2`, `_3`, three per color) parks
off-cell on the floor at boot (`cell_layout.park_positions_mm`) until a
`scatter_cell` draws some of it onto `table_source`.

`pick-grip` is the gripper. `wrist-cam` rides the arm's flange, `scene-cam`
watches the whole three-table workspace, and a fixed `side-cam` looks back
toward the arm, aimed at the scatter-zone center. The side camera is mounted on
the `side_cam_body`/`side_cam_mount` props at the source table's far end. The
`builtin` motion service and six `<color>-detector`/`<color>-segmenter`
vision-service pairs (one per block color) do the finding. `block-sorter` (the
`viam:isaac-sim-devin:conductor` service) drives the whole sort via DoCommand,
and `block-sorter-sensor` proxies its telemetry for data management. Add the
fragment to any machine that meets the requirements in the
[README](../README.md) and the world spawns everything at boot.

Props are configured on the world with the `props` attribute, as cubes or USD
references, fixed or dynamic. See the fragment for the shape of it.

The fragment ships nineteen `$variable`s, each with a `default_value` equal
to the numbers below, so a machine that sets nothing boots the exact cell:

| variable | binds | default |
|---|---|---|
| `table-height-m` | all three tables' `scale[2]` | `0.75` |
| `block-color-red` | `block_red_*.color` | `[0.9, 0.1, 0.1]` |
| `block-color-green` | `block_green_*.color` | `[0.05, 0.65, 0.1]` |
| `block-color-blue` | `block_blue_*.color` | `[0.05, 0.1, 0.9]` |
| `block-color-yellow` | `block_yellow_*.color` | `[0.9, 0.75, 0.05]` |
| `block-color-purple` | `block_purple_*.color` | `[0.55, 0.1, 0.75]` |
| `block-color-orange` | `block_orange_*.color` | `[1.0, 0.4, 0.05]` |
| `detect-color` | `red-detector`'s `detect_color` | `"#EA8D8D"` |
| `hue-tolerance-pct` | `red-detector`'s `hue_tolerance_pct` | `0.05` |
| `detect-color-green` | `green-detector`'s `detect_color` | `"#6AE28B"` |
| `detect-color-blue` | `blue-detector`'s `detect_color` | `"#869EEE"` |
| `detect-color-yellow` | `yellow-detector`'s `detect_color` | `"#EEDE64"` |
| `detect-color-purple` | `purple-detector`'s `detect_color` | `"#E399EB"` |
| `detect-color-orange` | `orange-detector`'s `detect_color` | `"#F0D76B"` |
| `hue-tolerance-pct-green` | `green-detector`'s `hue_tolerance_pct` | `0.05` |
| `hue-tolerance-pct-blue` | `blue-detector`'s `hue_tolerance_pct` | `0.05` |
| `hue-tolerance-pct-yellow` | `yellow-detector`'s `hue_tolerance_pct` | `0.05` |
| `hue-tolerance-pct-purple` | `purple-detector`'s `hue_tolerance_pct` | `0.05` |
| `hue-tolerance-pct-orange` | `orange-detector`'s `hue_tolerance_pct` | `0.05` |

`table-height-m` only substitutes each table's `scale[2]`. It has no
arithmetic, so overriding it desyncs every other number derived from the table
height across all three tables. Those numbers are each table's own
`position[2]` (`h / 2`), the six place pads' and eighteen pool blocks' z (all
resting on or above a table top), `pick-arm`'s frame z (`h` in mm), and
`isaac-world`'s own `frame.geometry` box if one is configured. Override the
table height only via `fragment_mods` `$set` overrides on those other fields
too, kept in sync by hand.

## The single-pick client

`examples/pick_red_block.py` drives one block of this cell end to end. It
detects the red block with `red-detector`/`red-segmenter` on `wrist-cam`,
picks it, and places it on `place_pad_red`. It is a regression tool from before
the conductor existed, not the way to operate the cell day to day. It
exercises the pick-verify-carry-place pipeline against the `red` block and
`place_pad_red` alone, without the conductor's scatter, census, or loop
management.

Clone this repository on your dev machine, then create the venv it uses:

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

Find your connection details in the web UI's 'connect' tab (select 'Python'),
or the 'code sample' tab, which shows a runnable connection snippet with your
machine's address and API key filled in. Run the script against the fragment
with:

```sh
PYTHONPATH=src python examples/pick_red_block.py \
  --address <machine-address> --api-key <key> --api-key-id <key-id>
```

`examples/pick_red_block.py --help` lists every flag with its default. Two
matter on a first run. `--support-z-mm` (default `750`, the three-table cell's
table top) already tells it the block rests on a table, not the floor.
`--randomize-seed <n>` scatters blocks deterministically before the pick, see
"Randomizing props" below.

The table is already a planner obstacle, since the world serves every prop
geometry live, so `--table` (the recipe box for scenes that serve none) is
unnecessary and is dropped automatically when a live `table` box is present.
To drive the whole 18-block pool at once, use the conductor's
`scatter_cell`/`start` DoCommand instead of this single-block script, see
"Sorting with the conductor" below.

`--block-size-mm` is optional. Omit it, the default, and the script measures
the target's size itself, from the focused detection's point cloud. It reads
the footprint (x/y extents) and the height (top face minus the support),
cross-checks them against each other, and prints
`MEASURED_BLOCK_JSON={"footprint_mm": [x, y], "height_mm": h, "size_mm": s,
"scan_pose_mm": {...}}`. A view where the three readings disagree is treated as
degenerate and re-scanned rather than grasped on. A measured size wider than
the gripper's 75 mm jaw (the 2F-85's 85 mm opening minus finger clearance)
refuses the grasp cleanly instead of attempting a doomed pick. Pass
`--block-size-mm <mm>` to skip measurement and use a fixed size end to end.

Combine `--randomize-seed <n>` with `--randomize-size-mm <lo>,<hi>` to also
redraw each scattered block's size, see "Randomizing props" below. The
script warns, without failing, if the measured size falls outside that range.

On success the script prints `PLACED_BLOCK_JSON=` with `"placed_on_pad": true`,
meaning the arm found the block, picked it up, and set it down on the pad. If
the block measures over 75 mm, the script refuses the grasp and leaves the arm
parked, which is correct behavior for an oversize block, not a failure.

## Measuring the tallest block (dynamic carry heights)

With sizes randomized, a fixed constant no longer bounds how high the arm
must lift the held block to clear the scattered ones, so the script measures
the tallest object in the scatter region before picking. The primary sensor
is `side-cam`, a fixed camera looking across the table. It is frame-oriented,
so no arm motion is needed, and occlusion-proof for the max height, except
when a nearer block's silhouette covers a farther, taller one. Each candidate
measurement runs four trust checks. There must be enough in-region points
above the support plane, a size-range-plausible result, enough points near the
measured top (no lone stray point), and all four footprint quadrants covered.
Only a measurement that passes all four is used. When the side scan fails
trust (or `--tallest-camera` disables it), the wrist camera sweeps
region-corner vantages as a fallback. If that also fails, the
`--randomize-size-mm` range's upper bound is used as a conservative ceiling,
with a logged warning. GPU runs validated the side scan within 0.8 mm of the
drawn (ground-truth) size and the wrist sweep within 0.1 mm.

When `--randomize-size-mm` is on, the measured tallest height replaces the
legacy fixed 130 mm keep-out ceiling and 200 mm carry-hop height (both
60 mm-block constants) with heights derived from the measurement. Without
`--randomize-size-mm`, the script keeps using those fixed constants.
`--tallest-camera` (default `side-cam`) names the camera component to use as
the primary sensor. Pass an empty string to disable it and go straight to
the wrist sweep.

The script prints `MEASURED_TALLEST_JSON={"tallest_mm": 81.06, "source":
"side", "trusted": true, "reasons": [], "points": 1842, "scan_poses_mm": [],
"keepout_height_mm": 148.3, "carry_clear_above_support_mm": 231.6,
"drawn_tallest_mm": 81.06, "drawn_delta_mm": 0.0}`. `source` is `"side"`,
`"wrist_sweep"`, or `"fallback"`. `drawn_tallest_mm` and `drawn_delta_mm` are
log-only evidence from the randomize response's ground-truth sizes, null
without a sizes-bearing randomize response. The pipeline never reads sim
ground truth to make a decision, only the client-side measurement does.

## Props and obstacles

A table is a `fixed` cube prop on the world. The README's "Units and
conventions" section covers how `size`, `scale` and `position` work:

```json
{
  "props": [
    {"name": "table", "type": "cube", "fixed": true, "size": 1.0,
     "scale": [1.2, 0.8, 0.75], "position": [0.60, 0.00, 0.375]}
  ]
}
```

`props` are visual/physical geometry in the sim. Whether the motion service
plans around them depends on which of three routes you take:

1. **Nothing extra, `isaac-world`'s own `GetGeometries`.** The world
   component serves every prop's current box live via `GetGeometries` (world
   frame, millimeters, `label` = the prop's `name`), skipping any prop whose
   box is all zero. An unknown-size `"usd"` prop with no `box_dims` set is the
   one case, see the README's "world attributes". A motion service that plans
   with the frame system picks these up with no client code, and only when
   `isaac-world` itself has a `frame` (`{"parent": "world"}`, as the fragment
   configures). When the module added its own ground plane (no `usd_stage`),
   the served geometries also include a 10 x 10 m `floor` box whose top face is
   at z = 0, so plans keep the arm out of the floor. Use `{"command":
   "ignore_props", "names": [...]}` on the world to exclude the prop being
   grasped, e.g. so the block doesn't obstruct its own pick.
2. **A client-built `WorldState`.** A client that assembles its own
   `WorldState` fetches `{"command": "prop_geometries"}` (see the world's
   `DoCommand` list in the README) and converts the returned boxes into
   `Geometry` obstacles itself. `obstacles_from_prop_geometries` in
   `examples/pick_red_block.py` is the recipe: one `RectangularPrism` per
   entry, skipping the excluded name(s) and any all-zero box. `table_obstacle`
   in the same file is the worked example for the fragment's 1.2 x 0.8 m table
   top, centered at (600, 0, 370) mm.
3. **A static `frame.geometry` on `isaac-world`.** For fixed furniture that
   never moves, add its box as `frame.geometry` on the world component
   itself, in millimeters and centered on the frame origin. This route stays
   valid alongside the other two. Here the table box is 1200 x 800 x 740 mm
   at translation (600, 0, 370), 10 mm below the real 750 mm surface so the
   arm isn't blocked from resting on it:

```json
{
  "name": "isaac-world",
  "api": "rdk:component:generic",
  "model": "viam:isaac-sim-devin:world",
  "frame": {
    "parent": "world",
    "geometry": {
      "type": "box",
      "x": 1200,
      "y": 800,
      "z": 740,
      "translation": { "x": 600, "y": 0, "z": 370 }
    }
  },
  "attributes": { "props": [ /* ... */ ] }
}
```

## Randomizing props

`{"command": "randomize_props", ...}` scatters named props to random
positions inside a region, deterministically: the same `seed` always
produces the same layout. Positions are kept at least `min_separation` mm
apart, or the two props' edge-to-edge gap if that is larger for props with
known sizes (default `min_separation` 150 mm). A worked example scattering
three of the pool's 18 blocks inside the source table's scatter zone
(`cell_layout.SCATTER_ZONE_X_MM` (-1350, -700), `SCATTER_ZONE_Y_MM`
(-300, 300) mm, table top at z = 750 mm):

```json
{
  "command": "randomize_props",
  "names": ["block_red_1", "block_green_1", "block_blue_1"],
  "region": [[-1350, -300, 750], [-700, 300, 750]],
  "seed": 42
}
```

-> `{"positions": {"block_red_1": [x, y, 780.5], "block_green_1": [x, y, 780.5],
"block_blue_1": [x, y, 780.5]}, "sizes_mm": {"block_red_1": [x, y, z], ...}}`
(center z = face z + half the 60 mm block + a 0.5 mm rest gap) with the same
`x`/`y` values every time `seed: 42` is passed for this region
and these names. `sizes_mm` is always present: each named prop's current
box dims, in millimeters.

Adding `"size_range_mm": [30, 90]` (or `{"block_red_1": [30, 90]}` to target
specific props) redraws a fresh size for each ranged cube prop before
placing it. That is one uniform draw per prop, in millimeters, applied to all
three axes, from the same seeded stream as the positions, so `seed: 42`
reproduces both the sizes and the positions. The prop's `size` config
attribute stays the spawn baseline: rescaling is always relative to it, so
repeated `randomize_props` calls with different ranges never compound.

A sized call also resets the world before the teleports, the same way
`spawn_prop` does. Rescaling a live rigid body invalidates PhysX's cooked
state, so every prop snaps to its spawn pose and the post-reset hooks fire,
and then the named props teleport to their sampled positions. A call
without `size_range_mm` never resets.

## Pooled scatter

`scatter_cell` draws a fresh sorting problem from a pool of blocks without
spawning or deleting anything. The world component knows no cell, so the
caller supplies the cell's numbers: the pool as `names_by_color`, the
scatter `region` as two `[x, y, z]` corners in mm, and the park grid as
`park_positions_mm`. The conductor builds them from `cell_layout` (18
blocks, `BLOCK_COLORS` x 3). A per-color count is drawn first, 1-3 per
color by default, from the same seeded stream as the sizes and positions.
That many blocks per color then place inside the region with the existing
separation rules, and every undrawn block parks out of the way. The same
`seed` always draws the same counts, sizes, and positions.

```json
{
  "command": "scatter_cell",
  "seed": 42,
  "size_range_mm": [50, 70],
  "names_by_color": {"red": ["block_red_1", "block_red_2", "block_red_3"], "green": ["..."]},
  "region": [[-1350, -300, 750], [-700, 300, 750]],
  "park_positions_mm": {"block_red_1": [x, y], "block_red_2": [x, y]}
}
```

-> `{"seed": 42, "counts": {"red": 2, "green": 1, ...}, "positions":
{"block_red_1": [x, y, z], ...}, "sizes_mm": {"block_red_1": [x, y, z],
...}, "parked": ["block_red_3", ...]}`. Pass `"counts": {"red": 0}` to
force a color out of the draw entirely (it parks in full), or any other
count in `[0, 3]` to override that color's draw without consuming a
random number for it. As with `randomize_props`, the response is
log-only evidence for a caller (e.g. a test harness) that wants to know
what got drawn. It is never control input for the arm or the motion
service.

`{"command": "clear_cell", "names_by_color": {...}, "park_positions_mm":
{...}}` re-parks every pool block and returns `{"parked": [names]}`, the
same log-only shape.

## Arm mount recipe

Mount an arm on the table by frame-placing it at the table's top height,
centered on the table (the fragment centers `pick-arm` on `table_arm` this
way):

```json
{
  "name": "pick-arm",
  "api": "rdk:component:arm",
  "model": "viam:isaac-sim-devin:arm",
  "frame": { "parent": "world", "translation": { "x": 0, "y": 0, "z": 750 } },
  "attributes": { "world": "isaac-world", "asset": "ur20" }
}
```

The Isaac articulation root is a fixed joint to the world, so no mount joint
needs authoring. A center mount sits 600 mm from the table's long edges and
400 mm from its short edges (`cell_layout.TABLE_DIMS_MM`), clear of the
collider on every side. UR assets carry a built-in base-frame correction so
this frame placement and Viam's kinematics agree with the simulated pose,
see the README's "arm attributes".

## conductor attributes

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name (`viam:isaac-sim-devin:world`) |
| `arm` | required | name of the arm component (boot ordering only, every motion goes through `motion`) |
| `gripper` | required | name of the gripper component |
| `camera` | required | name of the wrist camera |
| `side_camera` | required | name of the fixed side camera |
| `motion` | required | name of the motion service (`"builtin"` works) |
| `detectors` | required | `{color: segmenter vision-service name}` for exactly `cell_layout.BLOCK_COLORS` (`red`, `green`, `blue`, `yellow`, `purple`, `orange`) |
| `size_range_mm` | `[50, 80]` | `[lo, hi]`, and `hi` must be `<=` `cell_layout.MAX_BLOCK_SIZE_MM` (`80`) |

## sorter-sensor attributes

| attribute | default | notes |
|---|---|---|
| `conductor` | required | name of the `viam:isaac-sim-devin:conductor` generic service to poll |

## Sorting with the conductor (DoCommand)

The conductor sorts one scatter end to end, with no python script in the
loop: an optional scatter, one source-zone census to build the work list,
then nearest-first pick-verify-carry-place of every block onto its color's
pad.

| command | request | response |
|---|---|---|
| start | `{"command": "start", "seed"?: int, "counts"?: {color: int}, "loops"?: int, "continuous"?: bool}` | `{"ok": true, "state": "running"}`, or `{"ok": false, "state": "running"}` unchanged if a run is already in progress |
| stop | `{"command": "stop"}` | `{"ok": true}` |
| status | `{"command": "status"}` | `{"state": "idle\|running\|stopping\|complete\|failed", "remaining": [names], "current": name\|null, "outcomes": {name: {"outcome": "placed\|skipped_oversize\|failed", "prim"?: str, "reason"?: str, "attempts"?: int}}, "seed": int\|null, "pass": int, "run": {"loops_requested": int\|0-for-continuous\|null, "continuous": bool, "loop": int, "loops_completed": int, "loops_errored": int, "base_seed": int\|null, "placed": int, "failed": int, "skipped_oversize": int}, "success_rate": float\|null, "loop_records": [LoopRecord.to_dict(), ...]}` |

A `start` with neither `"loops"` nor `"continuous"` set is single-shot. With
`"seed"`, it scatters via the world's `scatter_cell` first, and without, it
sorts the standing scatter, and it is recorded in telemetry as one loop.
`"loops": N` (`N >= 1`) runs N loops, and `"loops": 0` or `"continuous": true`
runs until `stop`. Loop mode always resets each loop, including the first,
with `clear_cell` then `scatter_cell` at a per-loop seed derived from
`"seed"`, or from a time-derived base seed when `"seed"` is absent. The seed
advances by one (`run_log.loop_seed`) for each successive loop. A second
`start` while a run is already in progress is a no-op that leaves the
running run untouched.

`stop` cancels between motions and between passes (never mid-motion) and
at loop boundaries. It lands the run in state `idle` with `remaining`
preserved as it stood at cancellation.

Sorting is multi-pass: one pass is a census, then resolving detections to
scattered prims, then a clearance-ordered attempt loop (isolated blocks
first). A pass ends the sort when everything attempted that pass placed or
was skipped (no failures), when the pass placed nothing at all, or after
`MAX_PASSES` (5) passes, a hard cap against runaway re-censusing. A failed
grasp is retried once (`MAX_ATTEMPTS_PER_BLOCK` = 2 attempts total) before
the block is terminally `failed` for that loop. Oversize blocks (measured
size over the gripper's 75 mm jaw limit) are recorded as `skipped_oversize`
and never retried, since oversize never shrinks between passes.

In loop/continuous mode, a loop killed by a transient failure (e.g. a
dropped gRPC stream to viam-server) is recorded with an `"error"` field and
skipped, and the run continues with the next loop's seed. Only
`MAX_CONSECUTIVE_LOOP_ERRORS` (3) such loops in a row fail the run. In
single-shot mode any exception fails the run.

Vision alone decides what to pick and where. `prop_geometries` and
the `scatter_cell` response are the sim's own ground truth, and the
conductor consults them only for bookkeeping, building planner obstacles
and, after the scan, resolving each detection to its real prim. Everything
the conductor reports in `status` is log-only evidence, never control input
for the run.

## Telemetry

`status`'s `run` block carries the whole run's rollup. Its fields are
`loops_requested` (the requested count, `0` for continuous, `null` for
single-shot), `continuous`, `loop` (the loop currently running, or the one
that finished last), `loops_completed`, `loops_errored`, `base_seed`, and the running
totals `placed`/`failed`/`skipped_oversize`. `success_rate`
(`run_log.success_rate`) is `placed / (placed + failed)` (`skipped_oversize`
excluded from both terms), or `null` before any terminal placed/failed
outcome exists.

`loop_records` holds the last `LOOP_RECORD_WINDOW` (50) completed loops,
oldest first (`run_log.RollingLog`), bounding memory in continuous mode.
Each `LoopRecord` carries:

* `record_id`, monotonic for the module's lifetime, never reused
* `loop`, the 1-based loop number within its run
* `seed`, the loop's derived seed, or `null`
* `duration_s`, wall time for the whole loop
* `passes`, census passes run
* `placed` / `skipped_oversize` / `failed`, counts derived from `picks`
* `picks`, the loop's `PickRecord`s
* `rss_mb`, resident set size of the module process in MiB, sampled when
  the loop's record is cut, and `null` when the platform offers no reading
* `error`, present only when the loop died on a transient exception
  instead of completing (its `picks` are then empty, and it counted toward
  `loops_errored`, not `loops_completed`)

Each `PickRecord` is one block's terminal outcome within one loop. It carries
`name` (the resolved prim name), `color`, `outcome`
(`placed`/`skipped_oversize`/`failed`), `attempts` (1..`MAX_ATTEMPTS_PER_BLOCK`),
`duration_s` (wall time across all attempts of that block), and `reason`
(present only when set).

The sorter sensor's `get_readings` proxies the conductor's `status`
verbatim, but what it returns depends on the caller. On a data-management
capture poll (`extra` carries `fromDataManagement`), it emits only the
`LoopRecord`s newer than the greatest `record_id` it has already emitted,
and advances that high-water mark to cover everything returned in that call. If
nothing is new, it raises `NoCaptureToStoreError` so the data manager stores
nothing (an empty readings map is rejected outright). Any other caller, an
app panel or a script, gets the same shape as a live snapshot, where `loops`
holds whatever the capture cursor has not consumed yet. That call never
advances the mark, so an open status panel cannot eat records out from
under data capture. The mark lives for the module's lifetime and is never
reset by `reconfigure`. A module restart may re-emit the whole window,
which downstream dedupes again on `record_id`.
