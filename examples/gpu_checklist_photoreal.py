"""GPU checklist for the photoreal-workcell HDRI environment (phase 1) and
fixture-table mesh (phase 2) work. Six items per phase, verified against a
running world on the GPU machine, or against the module's in-process mock
with ``--mock`` for the parts that only need a duck-typed world/camera pair.
Pass ``--phase 1`` or ``--phase 2`` (default) to pick which set runs.

Phase-1 checklist items (`.claude/plans/photoreal-workcell/phase-1-hdri-environment.md`):

1. boot with the bundled 1k HDRI via `module://` - the scene camera's
   `get_image` shows the backdrop, horizon level (DEC-W6), no grid texture
   on the floor
2. `render.viewport_grid: false` removes the overlay in the WebRTC client
3. a `data://` path to the user's 4k EXR loads, and a bogus path logs one
   exception and boots to the plain floor with the untextured dome
4. re-measure the six rendered block hues with the wrist camera's
   `sample_color` DoCommand (CAM-15), one block of each colour parked in
   view: update `cell_layout.RENDERED_BLOCK_HUE_DEG` and the saturation
   floor, refresh the six `detect-color*` defaults in the fragment, and
   record old and new values here; `tests/test_fragment.py`'s hue tests go
   green on the new numbers
5. `{"command": "start", "loops": 1, "seed": 20}` on `block-sorter` ends
   `complete` with `placed` at the baseline (12 of 13 on 2026-09-09)
6. cost: cold `ready` time (finalizer window), warm `ready` time, and the
   10 s step rate from `examples/gpu_checklist_photoreal.py`, each against
   the grid+flat baseline measured first on the same machine. Record, do
   not tune yet.

Phase-2 checklist items (`.claude/plans/photoreal-workcell/phase-2-table-mesh.md`):

1. the converted table USD under `data://` loads as `/World/table_source_visual`,
   `/World/table_arm_visual`, `/World/table_place_visual`; each top face coincides
   with its collider top within 2 mm (`prim_pose` plus a bounding-box query,
   record all three)
2. cubes rest on the colliders exactly as before (drop a block over a hole,
   it rests on the collider top)
3. `prop_geometries` and the world's `frame.geometry` are unchanged from
   phase 1 (diff the JSON)
4. the seed-20 sorting loop ends `complete` at the baseline, and this is the
   first loop at `rendering_dt` 1/30 (phase 1 ruling), so its `placed` and
   duration are the 30 Hz baseline row
5. wrist-cam depth image of a table top: the mesh must not appear in depth
   where the collider is (visual prims are still ray-traced; if a mesh top
   sits above its collider by more than the 0.5 mm rest epsilon, the
   segmenter's support height moves, record the offset)
6. cost: cold and warm `ready` times and the 10 s step rate vs phase 1 (BVH
   cost of three instances; note whether the three references share one
   asset load)

Usage (real machine)::

    python examples/gpu_checklist_photoreal.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id> --world isaac-world

Usage (in-process mock, no GPU, no running machine)::

    PYTHONPATH=src python examples/gpu_checklist_photoreal.py --mock

Prints one heading plus the raw observations per item. The pure/async
helpers at the top take a duck-typed world so they are unit-tested on a
laptop (see tests/test_gpu_checklist_photoreal.py). Phase 1's item 4 needs
`--regions <path>` (a JSON file mapping each cell_layout.BLOCK_COLORS colour
to a [x0, y0, x1, y1] pixel box) on a real machine, and is skipped without
it; `--mock` always runs it against a placeholder region. Item 5's
`block-sorter` name is a placeholder a later phase wires to the real cell
layout. Phase 2's item 3 needs `--geometry-baseline <path>` (a JSON dump of
a prior `prop_geometries` result) to diff against; without one it prints the
current geometry count and a `--dump-geometries <path>` hint for next time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from gpu_checklist_world import WorldApi, run_step_rate_measurement

CHECKLIST_ITEMS: tuple[str, ...] = (
    "1. boot with the bundled 1k HDRI via `module://` - the scene camera's `get_image` shows "
    "the backdrop, horizon level (DEC-W6), no grid texture on the floor",
    "2. `render.viewport_grid: false` removes the overlay in the WebRTC client",
    "3. a `data://` path to the user's 4k EXR loads, and a bogus path logs one exception and "
    "boots to the plain floor with the untextured dome",
    "4. re-measure the six rendered block hues with the wrist camera's `sample_color` "
    "DoCommand (CAM-15), one block of each colour parked in view: update "
    "`cell_layout.RENDERED_BLOCK_HUE_DEG` and the saturation floor, refresh the six "
    "`detect-color*` defaults in the fragment, and record old and new values here; "
    "`tests/test_fragment.py`'s hue tests go green on the new numbers",
    '5. `{"command": "start", "loops": 1, "seed": 20}` on `block-sorter` ends `complete` with '
    "`placed` at the baseline (12 of 13 on 2026-09-09)",
    "6. cost: cold `ready` time (finalizer window), warm `ready` time, and the 10 s step rate "
    "from `examples/gpu_checklist_photoreal.py`, each against the grid+flat baseline measured "
    "first on the same machine. Record, do not tune yet.",
)

PHASE_2_ITEMS: tuple[str, ...] = (
    "1. the converted table USD under `data://` loads as "
    "`/World/table_source_visual`, `/World/table_arm_visual`, "
    "`/World/table_place_visual`; each top face coincides with its collider top "
    "within 2 mm (`prim_pose` plus a bounding-box query, record all three)",
    "2. cubes rest on the colliders exactly as before (drop a block over a hole, "
    "it rests on the collider top)",
    "3. `prop_geometries` and the world's `frame.geometry` are unchanged from "
    "phase 1 (diff the JSON)",
    "4. the seed-20 sorting loop ends `complete` at the baseline, and this is the "
    "first loop at `rendering_dt` 1/30 (phase 1 ruling), so its `placed` and "
    "duration are the 30 Hz baseline row",
    "5. wrist-cam depth image of a table top: the mesh must not appear in depth "
    "where the collider is (visual prims are still ray-traced; if a mesh top "
    "sits above its collider by more than the 0.5 mm rest epsilon, the "
    "segmenter's support height moves, record the offset)",
    "6. cost: cold and warm `ready` times and the 10 s step rate vs phase 1 (BVH "
    "cost of three instances; note whether the three references share one asset "
    "load)",
)

DEFAULT_STEP_RATE_WINDOW_S = 10.0
DEFAULT_READY_POLL_S = 0.5
DEFAULT_READY_TIMEOUT_S = 600.0
DEFAULT_BLOCK_SORTER_NAME = "block-sorter"
# --mock's placeholder pixel rectangle for every colour, since the mock scene
# has no six distinct blocks. Real machines pass --regions instead
DEFAULT_BLOCK_REGION_PX = (10, 200, 40, 230)
MM_PER_M = 1000.0
# phase-2 item 1's tolerance for a mesh top face vs its collider top
TOP_FACE_TOLERANCE_MM = 2.0
# phase-2 item 3's tolerance for a prop_geometries value being unchanged
GEOMETRY_TOLERANCE_MM = 0.5


class CameraApi(Protocol):
    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]: ...


# pure helpers


def hue_degrees(rgb: Sequence[float]) -> float:
    """0-360 hue of an 0-255 RGB triple, red at 0, matching
    tests/test_fragment.py::_hue_degrees's colorsys convention."""
    import colorsys

    r, g, b = (channel / 255.0 for channel in rgb)
    hue, _saturation, _value = colorsys.rgb_to_hsv(r, g, b)
    return hue * 360.0


def saturation(rgb: Sequence[float]) -> float:
    """0-1 HSV saturation of an 0-255 RGB triple."""
    import colorsys

    r, g, b = (channel / 255.0 for channel in rgb)
    _hue, sat, _value = colorsys.rgb_to_hsv(r, g, b)
    return sat


def rendered_hue_block(samples: Mapping[str, Sequence[float]]) -> str:
    """Renders a ``RENDERED_BLOCK_HUE_DEG = {...}`` block in
    cell_layout.py's exact format, ready to paste, one decimal per hue,
    keys in ``cell_layout.BLOCK_COLORS`` order."""
    from isaac_module.cell_layout import BLOCK_COLORS

    lines = ["RENDERED_BLOCK_HUE_DEG = {"]
    for colour in BLOCK_COLORS:
        lines.append(f'    "{colour}": {hue_degrees(samples[colour]):.1f},')
    lines.append("}")
    return "\n".join(lines)


def ready_time_s(samples: Sequence[tuple[float, Mapping[str, Any]]]) -> float | None:
    """Seconds from the first sample to the first sample where
    ``status["ready"]`` is true, or None if no sample ever was."""
    if not samples:
        return None
    started_at = samples[0][0]
    for sampled_at, status in samples:
        if status.get("ready"):
            return sampled_at - started_at
    return None


def visual_top_offsets_mm(
    visual_props: Sequence[Mapping[str, Any]], geometries: Sequence[Mapping[str, Any]]
) -> dict[str, float | None]:
    """For each visual prop with a ``collider``, the offset in mm of its
    ``bounds_m["max"][2]`` above the named collider's top face
    (``pose_in_world_mm["z"] + box_dims_mm[2] / 2``). ``None`` when
    ``bounds_m`` is unmeasured (the mock) or the collider name is not among
    ``geometries``."""
    geometries_by_name = {str(geometry["name"]): geometry for geometry in geometries}
    offsets: dict[str, float | None] = {}
    for visual in visual_props:
        collider_name = visual.get("collider")
        if collider_name is None:
            continue
        name = str(visual["name"])
        geometry = geometries_by_name.get(str(collider_name))
        bounds_m = visual.get("bounds_m")
        if geometry is None or bounds_m is None:
            offsets[name] = None
            continue
        collider_top_mm = geometry["pose_in_world_mm"]["z"] + geometry["box_dims_mm"][2] / 2
        mesh_top_mm = bounds_m["max"][2] * MM_PER_M
        offsets[name] = mesh_top_mm - collider_top_mm
    return offsets


def geometry_diff(
    baseline: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]
) -> list[str]:
    """One line per prop whose ``box_dims_mm`` or ``pose_in_world_mm``
    differs from the baseline by more than ``GEOMETRY_TOLERANCE_MM`` on any
    component, plus one line per name present on only one side. Empty when
    the two are the same within tolerance."""
    baseline_by_name = {str(geometry["name"]): geometry for geometry in baseline}
    current_by_name = {str(geometry["name"]): geometry for geometry in current}

    lines: list[str] = []
    for name in sorted(set(baseline_by_name) - set(current_by_name)):
        lines.append(f"{name}: only in baseline")
    for name in sorted(set(current_by_name) - set(baseline_by_name)):
        lines.append(f"{name}: only in current")

    for name in sorted(set(baseline_by_name) & set(current_by_name)):
        before = baseline_by_name[name]
        after = current_by_name[name]
        dims_delta = _max_abs_delta(before["box_dims_mm"], after["box_dims_mm"])
        pose_delta = _max_abs_pose_delta(before["pose_in_world_mm"], after["pose_in_world_mm"])
        if dims_delta > GEOMETRY_TOLERANCE_MM or pose_delta > GEOMETRY_TOLERANCE_MM:
            lines.append(
                f"{name}: box_dims_mm {before['box_dims_mm']} -> {after['box_dims_mm']}, "
                f"pose_in_world_mm {before['pose_in_world_mm']} -> {after['pose_in_world_mm']}"
            )
    return lines


def _max_abs_delta(before: Sequence[float], after: Sequence[float]) -> float:
    return max(abs(b - a) for b, a in zip(before, after, strict=True))


def _max_abs_pose_delta(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    return max(abs(before[key] - after[key]) for key in before if key in after)


# async drivers, not unit-tested (need a live/mock world+camera pair)


def load_regions(text: str) -> dict[str, tuple[int, int, int, int]]:
    """Parses a ``--regions`` JSON file: a mapping of every
    ``cell_layout.BLOCK_COLORS`` colour to a ``[x0, y0, x1, y1]`` pixel box.
    Raises ``ValueError`` naming any missing colour, extra colour, or box
    that is not four non-negative ints with x0 < x1 and y0 < y1."""
    from isaac_module.cell_layout import BLOCK_COLORS

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("regions file must be a JSON object mapping colour to [x0, y0, x1, y1]")

    missing = [colour for colour in BLOCK_COLORS if colour not in data]
    if missing:
        raise ValueError(f"regions file is missing colour(s): {', '.join(missing)}")
    extra = [colour for colour in data if colour not in BLOCK_COLORS]
    if extra:
        raise ValueError(f"regions file has unknown colour(s): {', '.join(extra)}")

    return {colour: _validated_region_box(colour, data[colour]) for colour in BLOCK_COLORS}


def _validated_region_box(colour: str, box: Any) -> tuple[int, int, int, int]:
    is_four_ints = (
        isinstance(box, Sequence)
        and not isinstance(box, (str, bytes))
        and len(box) == 4
        and all(isinstance(v, int) and not isinstance(v, bool) for v in box)
    )
    if not is_four_ints:
        raise ValueError(f"regions[{colour!r}] must be four non-negative ints [x0, y0, x1, y1]")
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
        raise ValueError(f"regions[{colour!r}] must be four non-negative ints [x0, y0, x1, y1]")
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"regions[{colour!r}] must have x0 < x1 and y0 < y1, got {box}")
    return (x0, y0, x1, y1)


async def sample_block_hues(
    world: WorldApi, camera: CameraApi, blocks: Mapping[str, tuple[int, int, int, int]]
) -> dict[str, list[float]]:
    """Samples ``sample_color`` per colour's pixel region, prints the
    fragment-ready hex plus hue/saturation for each, then the
    ``RENDERED_BLOCK_HUE_DEG`` paste block."""
    means: dict[str, list[float]] = {}
    for colour, region in blocks.items():
        result = await camera.do_command({"command": "sample_color", "region": list(region)})
        mean_rgb = [float(v) for v in result["mean_rgb"]]
        means[colour] = mean_rgb
        print(
            f"  {colour}: srgb_hex={result['srgb_hex']} mean_rgb={mean_rgb} "
            f"hue_deg={hue_degrees(mean_rgb):.1f} saturation={saturation(mean_rgb):.2f}"
        )
    print(rendered_hue_block(means))
    return means


async def sample_ready_time(
    world: WorldApi,
    poll_s: float = DEFAULT_READY_POLL_S,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
) -> dict[str, Any]:
    """Polls ``status`` through the world's ``do_command`` until ``ready``
    is true or ``timeout_s`` elapses, and reports the elapsed ready time."""
    samples: list[tuple[float, Mapping[str, Any]]] = []
    started_at = time.monotonic()
    while True:
        status = await world.do_command({"command": "status"})
        samples.append((time.monotonic(), status))
        if status.get("ready") or time.monotonic() - started_at > timeout_s:
            break
        await asyncio.sleep(poll_s)
    return {"ready_time_s": ready_time_s(samples), "sample_count": len(samples)}


async def _run_confirmation_item(world: WorldApi, index: int) -> None:
    """Items 1-3 and 5 need a human to look at an image or a scene readout.
    Prints the item's instructions plus the status/prop_geometries a human
    confirms against."""
    print(CHECKLIST_ITEMS[index])
    status = await world.do_command({"command": "status"})
    print(f"  status: {json.dumps(status, default=str, sort_keys=True)}")
    geometries = await world.do_command({"command": "prop_geometries"})
    print(f"  prop_geometries: {json.dumps(geometries, default=str, sort_keys=True)}")


async def _run_hue_item(
    world: WorldApi, camera: CameraApi, regions: Mapping[str, tuple[int, int, int, int]] | None
) -> None:
    print(CHECKLIST_ITEMS[3])
    if regions is None:
        print(
            "  skipped: no --regions file given (pass --regions <path> with a pixel box "
            "per cell_layout.BLOCK_COLORS colour on --camera)"
        )
        return
    await sample_block_hues(world, camera, regions)


async def _run_cost_item(world: WorldApi, heading: str = CHECKLIST_ITEMS[5]) -> None:
    print(heading)
    ready = await sample_ready_time(world)
    print(f"  ready: {json.dumps(ready, default=str, sort_keys=True)}")
    step_rate = await run_step_rate_measurement(world, window_s=DEFAULT_STEP_RATE_WINDOW_S)
    print(f"  step_rate: {json.dumps(step_rate, default=str, sort_keys=True)}")


async def _run_phase_1(
    world: WorldApi, camera: CameraApi, regions: Mapping[str, tuple[int, int, int, int]] | None
) -> None:
    await _run_confirmation_item(world, 0)
    await _run_confirmation_item(world, 1)
    await _run_confirmation_item(world, 2)
    await _run_hue_item(world, camera, regions)
    await _run_confirmation_item(world, 4)
    await _run_cost_item(world)


async def _run_phase_2_offsets_item(world: WorldApi, heading: str) -> dict[str, float | None]:
    """Item 1's/5's shared work: status's ``visual_props`` against
    ``prop_geometries``, printing each visual's mesh dims, scale and top-face
    offset. Returns the offsets so item 5 can restate them."""
    print(heading)
    status = await world.do_command({"command": "status"})
    visual_props = status.get("visual_props") or []
    if "visual_props" not in status:
        print("  status has no visual_props key: the module predates visual props, reload it")
    elif not visual_props:
        print(
            "  no visual props spawned: none configured, or each failed at boot (grep the "
            "module log for `failed to spawn prop` and `usd not found`)"
        )
    geometries_result = await world.do_command({"command": "prop_geometries"})
    geometries = geometries_result.get("geometries") or []
    offsets = visual_top_offsets_mm(visual_props, geometries)
    for visual in visual_props:
        name = str(visual["name"])
        print(f"  {name}: mesh_dims_m={visual.get('mesh_dims_m')} scale={visual.get('scale')}")
        offset = offsets.get(name)
        if offset is None:
            print(f"  {name}: top-face offset unmeasured")
        else:
            verdict = "PASS" if abs(offset) <= TOP_FACE_TOLERANCE_MM else "FAIL"
            print(f"  {name}: top-face offset {offset:.3f} mm ({verdict})")
    return offsets


async def _run_phase_2_geometry_diff_item(
    world: WorldApi, baseline_path: str | None, dump_path: str | None
) -> None:
    print(PHASE_2_ITEMS[2])
    geometries_result = await world.do_command({"command": "prop_geometries"})
    geometries = geometries_result.get("geometries") or []
    if dump_path:
        with open(dump_path, "w") as dump_file:
            json.dump(geometries_result, dump_file, default=str, sort_keys=True, indent=2)
        print(f"  wrote {len(geometries)} geometries to {dump_path}")
    if not baseline_path:
        print(
            f"  no --geometry-baseline given: {len(geometries)} geometries in the current "
            "prop_geometries; pass --dump-geometries <path> now, then --geometry-baseline "
            "<path> on the next run to diff against it"
        )
        return
    with open(baseline_path) as baseline_file:
        baseline_result = json.load(baseline_file)
    baseline = baseline_result.get("geometries") or []
    diff = geometry_diff(baseline, geometries)
    if not diff:
        print("  PASS: prop_geometries unchanged from the baseline")
        return
    print("  FAIL: prop_geometries differs from the baseline")
    for line in diff:
        print(f"    {line}")


async def _run_phase_2(
    world: WorldApi,
    camera: CameraApi,
    args: argparse.Namespace,
) -> None:
    offsets = await _run_phase_2_offsets_item(world, PHASE_2_ITEMS[0])

    print(PHASE_2_ITEMS[1])
    print("  human: drop a block over a hole and confirm it rests on the collider top")

    await _run_phase_2_geometry_diff_item(world, args.geometry_baseline, args.dump_geometries)

    print(PHASE_2_ITEMS[3])
    print(
        f'  human: run {{"command": "start", "loops": 1, "seed": 20}} on '
        f"`{DEFAULT_BLOCK_SORTER_NAME}` and record `placed` and duration as the 30 Hz baseline row"
    )

    print(PHASE_2_ITEMS[4])
    print(
        "  human: compare a wrist-cam depth image of a table top against the offsets above; "
        "a mesh top more than 0.5 mm above its collider moves the segmenter's support height"
    )
    for name, offset in offsets.items():
        print(f"  {name}: top-face offset {offset}")

    await _run_cost_item(
        world,
        heading=(
            f"{PHASE_2_ITEMS[5]}\n  human: read the Kit log for whether the three "
            "table_source_visual/table_arm_visual/table_place_visual references share one "
            "asset load"
        ),
    )


async def _run(
    world: WorldApi,
    camera: CameraApi,
    args: argparse.Namespace,
    regions: Mapping[str, tuple[int, int, int, int]] | None,
) -> None:
    if args.phase == 1:
        await _run_phase_1(world, camera, regions)
    else:
        await _run_phase_2(world, camera, args)


def _ensure_mock_sim_booted() -> Any:
    import threading

    from isaac_module.sim_manager import SimConfig, SimManager

    manager = SimManager.get()
    if not manager._booted.is_set():
        sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
        sim_thread.start()
        manager.ensure_booted(SimConfig(mock=True))
    return manager


async def _run_real(args: argparse.Namespace) -> None:
    from viam.components.camera import Camera
    from viam.components.generic import Generic
    from viam.robot.client import RobotClient

    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    regions = None
    if args.regions:
        with open(args.regions) as regions_file:
            regions = load_regions(regions_file.read())

    robot = await RobotClient.at_address(args.address, opts)
    try:
        world = Generic.from_robot(robot, args.world)
        camera = Camera.from_robot(robot, args.camera)
        await _run(world, camera, args, regions)
    finally:
        await robot.close()


async def _run_mock(args: argparse.Namespace) -> None:
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from isaac_module.models.world import IsaacWorld

    def config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
        return ComponentConfig(name=name, attributes=dict_to_struct(attrs))

    _ensure_mock_sim_booted()
    world = IsaacWorld.new(config("gpu-checklist-photoreal-mock-world", {"mock": True}), {})

    from isaac_module.cell_layout import BLOCK_COLORS
    from isaac_module.models.camera import IsaacCamera

    camera = IsaacCamera.new(
        config(
            "gpu-checklist-photoreal-mock-cam",
            {"world": "gpu-checklist-photoreal-mock-world", "width": 320, "height": 240},
        ),
        {},
    )
    regions = {colour: DEFAULT_BLOCK_REGION_PX for colour in BLOCK_COLORS}
    await _run(world, camera, args, regions)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mock", action="store_true", help="run in-process against the module's mock backend"
    )
    parser.add_argument("--address", help="machine address (required unless --mock)")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--world", default="isaac-world", help="the isaac-sim world component name")
    parser.add_argument(
        "--camera",
        default="wrist-cam",
        help="the camera item 4's sample_color hue re-measurement uses",
    )
    parser.add_argument(
        "--regions",
        help="path to a JSON file mapping each cell_layout.BLOCK_COLORS colour to a "
        "[x0, y0, x1, y1] pixel box on --camera, for phase 1 item 4's hue re-measurement. "
        "Without it on a real machine, item 4 is skipped",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=2,
        help="which phase's checklist items to run (default: 2)",
    )
    parser.add_argument(
        "--geometry-baseline",
        help="path to a JSON file with a prior prop_geometries result, for phase 2 item 3's "
        "diff. Without it, item 3 prints the current geometry count instead of diffing",
    )
    parser.add_argument(
        "--dump-geometries",
        help="path to write the current prop_geometries result, for use as a later run's "
        "--geometry-baseline",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.mock and not args.address:
        print("FAILED: --address is required unless --mock is set")
        return 1
    try:
        if args.mock:
            asyncio.run(_run_mock(args))
        else:
            asyncio.run(_run_real(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure as a clean exit code
        print(f"FAILED: {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
