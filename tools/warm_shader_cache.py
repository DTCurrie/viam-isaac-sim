#!/usr/bin/env python3
"""Warm Isaac Sim's shader cache with the module's own world, without viam-server.

Usage: `warm_shader_cache.py [--fragment PATH] [--frames N] [--mock]`

Kit compiles the RTX ray-tracing pipeline on the first render after a scene
change and caches the result on disk, per GPU and driver. On a cold cache
those first `world.step` calls take minutes, and every queued
`SimManager.run()` call times out meanwhile, so the module reads as broken.
The image build runs this tool once, as the user viam-server runs the module
as, so the cache is warm on every VM created from the image.

The scene comes from a fragment, `fragments/isaac-sim-block-sorting.json` by
default, so the compiled pipelines match the shipped cell: the world's props
and lighting, then the arm, gripper and camera components in fragment order,
with the attributes each model's reconfigure would pass to
`SimManager.create_*`. Fragment `$variable` objects resolve to their defaults
(`isaac_module.config_resolver.resolve_fragment_variables`). The world config
comes from `isaac_module.models.world.sim_config_from_attrs`, then is forced
to `headless=True`, `livestream=False`, `wait_for_finalizer=False` (the tool
has no finalizer component and its own retry stands in for the scene gate)
and a boot timeout of `WARMUP_BOOT_TIMEOUT_S`, since a cold boot can outlast
the module's default.

After the last create the tool observes `--frames` completed sim steps through
the task queue, one `SimManager.run()` round trip per step, so it returns only
once the renderer has finished the compiles the scene provoked. A `create_*`
call or round trip that raises `SimTimeoutError` is retried until it answers
or `CREATE_RETRY_TIMEOUT_S` passes, since on a cold cache that timeout is the
expected path, not a failure.

Exit 0 when every component was created and the frames were observed,
`EXIT_ERROR` with the error logged otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from isaac_module.config_resolver import resolve_fragment_variables  # noqa: E402
from isaac_module.errors import SimTimeoutError  # noqa: E402
from isaac_module.models.gripper import default_parent_prim  # noqa: E402
from isaac_module.models.world import sim_config_from_attrs  # noqa: E402
from isaac_module.sim_manager import SimConfig, SimManager  # noqa: E402

DEFAULT_FRAGMENT = REPO / "fragments" / "isaac-sim-block-sorting.json"
DEFAULT_FRAMES = 300
# A cold boot compiles the base pipelines before the world reports ready. The
# module's 110 s default is sized for viam-server's resource timeout, which
# does not apply here.
WARMUP_BOOT_TIMEOUT_S = 3600.0
# Upper bound on retrying one create, or one queue round trip, through a slow
# step. Below it a SimTimeoutError is the expected cold-cache path.
CREATE_RETRY_TIMEOUT_S = 1800.0
EXIT_ERROR = 1

ComponentKind = Literal["arm", "gripper", "camera"]
# Fragment `model` strings end in one of these; anything else (the world, the
# sorter sensor, non-module components) is not warmed.
KIND_BY_MODEL_SUFFIX: dict[str, ComponentKind] = {
    ":arm": "arm",
    ":gripper": "gripper",
    ":camera": "camera",
}
WORLD_MODEL_SUFFIX = ":world"

LOGGER = logging.getLogger("warm_shader_cache")


@dataclass(frozen=True)
class WarmupComponent:
    """One `SimManager.create_<kind>(name, attrs)` call, in fragment order.

    `attrs` is what the matching model's reconfigure would pass: the fragment
    attributes with variables resolved, plus the gripper's default
    `parent_prim` derived from its arm's asset
    (`isaac_module.models.gripper.default_parent_prim`) when the fragment
    leaves it unset.
    """

    kind: ComponentKind
    name: str
    attrs: dict[str, Any]


@dataclass(frozen=True)
class WarmupPlan:
    """Everything the run will do, computed before Isaac boots."""

    sim_config: SimConfig
    components: tuple[WarmupComponent, ...]
    frames: int


def _kind_for_model(model: str) -> ComponentKind | None:
    for suffix, kind in KIND_BY_MODEL_SUFFIX.items():
        if model.endswith(suffix):
            return kind
    return None


def _gripper_attrs(
    name: str, attrs: dict[str, Any], by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """`attrs` with `parent_prim` defaulted from the gripper's arm, matching
    `IsaacGripper.reconfigure`. Raises ValueError when the arm named in
    `attrs["arm"]` is not one of the fragment's own components."""
    if "parent_prim" in attrs:
        return attrs
    arm_name = str(attrs.get("arm", ""))
    arm_component = by_name.get(arm_name)
    if arm_component is None:
        raise ValueError(f"gripper {name!r}: arm {arm_name!r} is not a component in the fragment")
    arm_attrs = arm_component.get("attributes") or {}
    attrs["parent_prim"] = default_parent_prim(arm_name, arm_attrs.get("asset"))
    return attrs


def plan_from_fragment(
    fragment: dict[str, Any], *, frames: int = DEFAULT_FRAMES, mock: bool = False
) -> WarmupPlan:
    """Pure. Resolve the fragment's variables, build the world's SimConfig
    with the warm-up overrides (`headless`, `livestream`, `boot_timeout`,
    `mock`), and list the arm, gripper and camera components in fragment
    order. Raises ValueError when the fragment has no world component or no
    component to warm."""
    resolved, _unresolved = resolve_fragment_variables(fragment)
    components: list[dict[str, Any]] = resolved.get("components", [])
    by_name = {component["name"]: component for component in components}

    world_component = next(
        (c for c in components if str(c.get("model", "")).endswith(WORLD_MODEL_SUFFIX)), None
    )
    if world_component is None:
        raise ValueError(f'fragment has no component whose model ends with "{WORLD_MODEL_SUFFIX}"')
    sim_config = dataclasses.replace(
        sim_config_from_attrs(world_component.get("attributes") or {}),
        headless=True,
        livestream=False,
        wait_for_finalizer=False,
        boot_timeout=WARMUP_BOOT_TIMEOUT_S,
        mock=mock,
    )

    warmup_components: list[WarmupComponent] = []
    for component in components:
        kind = _kind_for_model(str(component.get("model", "")))
        if kind is None:
            continue
        name = component["name"]
        attrs = dict(component.get("attributes") or {})
        if kind == "gripper":
            attrs = _gripper_attrs(name, attrs, by_name)
        warmup_components.append(WarmupComponent(kind=kind, name=name, attrs=attrs))

    if not warmup_components:
        raise ValueError("fragment has no arm, gripper or camera component to warm")

    return WarmupPlan(sim_config=sim_config, components=tuple(warmup_components), frames=frames)


def run_plan(plan: WarmupPlan, sim: SimManager) -> None:
    """Drive `plan` against `sim` from a worker thread while `sim.main_loop()`
    runs on the main thread: `ensure_booted`, one create per component through
    `retry_through_slow_steps`, then `plan.frames` queue round trips of
    `sim.run(lambda: None)`, each also through the retry. Logs every phase
    with its wall time. Raises on the first failure. Always ends by calling
    `sim.request_stop()`."""
    try:
        start = time.monotonic()
        retry_through_slow_steps(lambda: sim.ensure_booted(plan.sim_config), "boot")
        LOGGER.info("booted in %.1fs", time.monotonic() - start)

        for component in plan.components:
            create = getattr(sim, f"create_{component.kind}")
            start = time.monotonic()
            retry_through_slow_steps(
                partial(create, component.name, component.attrs),
                f"create {component.kind} {component.name}",
            )
            LOGGER.info(
                "created %s %s in %.1fs", component.kind, component.name, time.monotonic() - start
            )

        start = time.monotonic()
        for i in range(plan.frames):
            retry_through_slow_steps(lambda: sim.run(lambda: None), f"frame {i + 1}/{plan.frames}")
        LOGGER.info("observed %d frames in %.1fs", plan.frames, time.monotonic() - start)
    finally:
        sim.request_stop()


def retry_through_slow_steps(call: Callable[[], Any], what: str) -> Any:
    """Call `call()`, retrying `SimTimeoutError` until it answers or
    `CREATE_RETRY_TIMEOUT_S` has passed since the first attempt, logging each
    retry with the elapsed time and `what`. Any other exception propagates,
    and so does the last `SimTimeoutError` once the bound is reached."""
    start = time.monotonic()
    while True:
        try:
            return call()
        except SimTimeoutError:
            elapsed = time.monotonic() - start
            if elapsed > CREATE_RETRY_TIMEOUT_S:
                raise
            LOGGER.warning("%s timed out, retrying (%.1fs elapsed)", what, elapsed)


MAIN_LOOP_WORKER_JOIN_TIMEOUT_S = 60.0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    sim_factory: Callable[[], SimManager] = SimManager.get,
) -> int:
    """Parse args, load the fragment, build the plan, start `run_plan` on a
    worker thread and run `sim.main_loop()` on this thread until the worker
    stops it. `sim_factory` exists for tests, which pass `SimManager` for a
    fresh instance instead of the process singleton the test session already
    booted."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    try:
        fragment = json.loads(args.fragment.read_text())
        plan = plan_from_fragment(fragment, frames=args.frames, mock=args.mock)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        LOGGER.error("failed to build the warm-up plan: %s", exc)
        return EXIT_ERROR

    sim = sim_factory()
    worker_errors: list[BaseException] = []

    def _run() -> None:
        try:
            run_plan(plan, sim)
        except BaseException as exc:  # noqa: BLE001 - reported on the main thread below, not swallowed
            worker_errors.append(exc)

    worker = threading.Thread(target=_run, name="warm-shader-cache", daemon=True)
    worker.start()
    sim.main_loop()
    worker.join(timeout=MAIN_LOOP_WORKER_JOIN_TIMEOUT_S)

    if worker_errors:
        LOGGER.error("warm-up failed: %s", worker_errors[0])
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
