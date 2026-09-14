from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from viam.logging import getLogger

from ..assets import resolve_asset
from ..prim_paths import prim_name
from ..prop_scatter import (
    DEFAULT_MIN_SEPARATION_M,
    POOL_SCATTER_MIN_SEPARATION_M,
    ClearCellResult,
    PropGeometry,
    RandomizeResult,
    ScatterCellResult,
    _draw_pool_counts,
    _draw_sizes_and_positions,
    _park_pose_m,
    _require_cube_prop,
    _require_park_positions,
    _require_pool_blocks,
    _scattered_block_names,
    _validate_size_range_names,
    prop_box_dims,
    prop_spawn_orientation,
)
from ..spatial import Quat, Vec3, _as_quat, to_vec3
from ..visual_props import fit_collider_name, is_visual_prop, visual_prop_error, visual_prop_record

if TYPE_CHECKING:
    from ..sim_manager import SimManager

LOGGER = getLogger(__name__)


def _require_not_visual(sim: SimManager, verb: str, names: Sequence[str]) -> None:
    """Raise on the first ``names`` entry that is a visual prop. Every
    non-geometry prop verb calls this before any other lookup."""
    for name in names:
        if prim_name(name) in sim._visual_props:
            raise ValueError(visual_prop_error(verb, name))


class WorldHandle:
    """The seam every world-component verb drives the sim through.
    models/world.py talks to this interface only. SimManager
    stays an implementation detail behind it.

    Poses cross this seam in meters and (w,x,y,z) world-frame
    quaternions. Prims added via ``add_usd`` are stage furniture, not
    registered props: they never appear in ``prop_geometries`` and no
    pose verb can move them (spawn a ``type: "usd"`` prop for that).
    """

    def status(self) -> dict[str, Any]:
        raise NotImplementedError

    def play(self) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def reset(self, soft: bool = False) -> None:
        """Full reset (soft=False): the chokepoint - the sim
        world resets (every prop returns to its configured spawn pose)
        and every post-reset hook replays (gain snapshots, camera
        re-inits). Soft reset (soft=True): pose-only - every registered
        prop returns to its spawn pose with velocities zeroed.
        Articulations, hooks and gains are left alone."""
        raise NotImplementedError

    def add_usd(
        self,
        usd_path: str,
        prim_path: str,
        position_m: Vec3,
        orientation_wxyz: Quat | None = None,
    ) -> None:
        """Reference a USD file into the stage at ``prim_path``
        (orientation applies here too)."""
        raise NotImplementedError

    def prop_geometries(self) -> list[PropGeometry]:
        """One entry per registered prop (configured or runtime-spawned),
        at its current world pose."""
        raise NotImplementedError

    def set_prop_pose(
        self, name: str, position_m: Vec3, orientation_wxyz: Quat | None = None
    ) -> None:
        """Teleport prop ``name`` (orientation kept when None) and zero
        its velocities. Never rewrites the prop's spawn/default state: a
        later reset() still restores the configured pose (mock gate).
        Unknown name -> ValueError."""
        raise NotImplementedError

    def randomize_props(
        self,
        names: list[str],
        region: tuple[Vec3, Vec3],
        seed: int,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        size_range_m: dict[str, tuple[float, float]] | None = None,
    ) -> RandomizeResult:
        """Place ``names`` (in list order) deterministically and teleport
        each one there with set_prop_pose semantics, optionally redrawing
        sizes first.

        ``size_range_m`` maps a prop name to (lo, hi) full edge length in
        meters: that prop's new size draws as one uniform(lo, hi) scalar
        applied to all three axes. Keys must name cube props in ``names``
        (ValueError otherwise). Sizes and positions come from one
        ``random.Random(seed)`` stream - sizes first, in ``names`` order,
        then positions - so a seed reproduces both, and a call with no
        ranges consumes no size draws (existing seeded layouts are
        unchanged). Rescaling is absolute against the spawn size
        (scale = drawn / spawn edge), so repeated draws never accumulate,
        and the new dims persist through reset (reset restores poses,
        never sizes). prop_geometries serves the post-draw dims afterward.

        A sized call (any range given) replays spawn_prop's full-reset
        pattern before the teleports: rescaling a live rigid body
        invalidates PhysX's tensor views, so the world resets (every prop
        snaps to its spawn pose, post-reset hooks fire) and then the named
        props teleport to their sampled positions. A call with no ranges
        never resets.

        Placement uses the post-draw dims: the centers of props a and b
        stay at least max(min_separation_m, (footprint_a + footprint_b)/2
        + PROP_EDGE_CLEARANCE_M) apart in x/y, where a prop's footprint is
        the max of its x/y dims, and each prop rests at
        face z + dims_z / 2 + PROP_REST_EPSILON_M.
        Unknown name -> ValueError."""
        raise NotImplementedError

    def scatter_cell(
        self,
        names_by_color: dict[str, list[str]],
        region: tuple[Vec3, Vec3],
        park_positions_m: dict[str, tuple[float, float]],
        seed: int,
        size_range_m: tuple[float, float] | None = None,
        counts: dict[str, int] | None = None,
    ) -> ScatterCellResult:
        """Draw a fresh sorting problem from the parked pool blocks named
        in ``names_by_color`` (color -> its ordered pool names, e.g. 3 per
        color for the shipped cell).

        One ``random.Random(seed)`` stream, in order: a per-color count
        (``rng.randint(1, len(names_by_color[color]))`` in
        ``names_by_color`` order, skipped for a color given in
        ``counts``), then sizes, then positions. Drawn blocks per color are
        that color's first N names (``names_by_color[color][:n]``) and
        place inside ``region`` with the same separation machinery as
        randomize_props, at POOL_SCATTER_MIN_SEPARATION_M
        (DEFAULT_MIN_SEPARATION_M is infeasible for a dense 18-block pool
        in the shipped cell's region). ``size_range_m`` (lo, hi) applies to
        every drawn block, or None to keep current sizes. Undrawn blocks
        re-park at ``park_positions_m`` with z = half their current height
        + the contact-offset convention.

        ``counts`` values must be in [0, len(names_by_color[color])], keys
        in ``names_by_color`` (ValueError naming the offender). Any
        missing pool prim or missing park position -> ValueError listing
        the missing names. Only the named pool prims move or rescale."""
        raise NotImplementedError

    def clear_cell(
        self,
        names_by_color: dict[str, list[str]],
        park_positions_m: dict[str, tuple[float, float]],
    ) -> ClearCellResult:
        """Re-park every pool block named in ``names_by_color`` at
        ``park_positions_m``, poses only - no rescale, since rescaling is
        absolute against spawn size and the next scatter_cell's draw is
        unaffected either way."""
        raise NotImplementedError

    def spawn_prop(self, prop: dict[str, Any]) -> None:
        """Add a prop at runtime: same schema as the world's
        ``props`` config attr (validated at the Viam edge, name
        uniqueness re-checked here -> ValueError). On the real sim this
        replays the component-spawn path - spawn on the sim thread, then
        a full world reset (hooks fire, earlier teleports snap back to
        spawn poses) - so spawn before randomizing. The prop registers
        like a configured one: it appears in prop_geometries and
        survives reset."""
        raise NotImplementedError


class MockWorldHandle(WorldHandle):
    """Plain-python scene registry: every scene behavior above
    is testable without Isaac Sim. Registry entries keep the spawn attrs
    and both spawn and current poses."""

    def __init__(self, sim: SimManager, props: Sequence[dict[str, Any]]) -> None:
        self._sim = sim
        self._registry: dict[str, dict[str, Any]] = {}
        # non-visual props register first so a visual's fit.collider always
        # finds its cube already in the registry
        ordered_props = [p for p in props if not is_visual_prop(p)]
        ordered_props += [p for p in props if is_visual_prop(p)]
        for prop in ordered_props:
            self._register(prop)

    def _register(self, prop: dict[str, Any]) -> None:
        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        if is_visual_prop(prop):
            self._register_visual(prop)
            return
        name = prim_name(str(prop["name"]))
        if name in self._registry or name in self._sim._visual_props:
            raise ValueError(f"prop {name!r} already exists")
        position = to_vec3(prop.get("position"))
        orientation = prop_spawn_orientation(prop)
        self._registry[name] = {
            "spawn": dict(prop),
            "spawn_position": position,
            "spawn_orientation": orientation,
            "position": position,
            "orientation": orientation,
        }

    def _register_visual(self, prop: dict[str, Any]) -> None:
        name = prim_name(str(prop["name"]))
        if name in self._registry or name in self._sim._visual_props:
            raise ValueError(f"prop {name!r} already exists")
        if not prop.get("usd_path"):
            LOGGER.info("visual prop %s skipped: usd_path is empty, nothing to reference", name)
            return
        collider_name = fit_collider_name(prop)
        collider_dims_m: Vec3 | None = None
        if collider_name is not None:
            entry = self._registry.get(prim_name(collider_name))
            if entry is None or str(entry["spawn"].get("type", "cube")) != "cube":
                raise ValueError(
                    f"prop {name}: fit.collider {collider_name!r} must name an existing cube prop"
                )
            collider_dims_m = prop_box_dims(entry["spawn"])
        self._sim._visual_props[name] = visual_prop_record(
            prop,
            name=name,
            resolved_path=resolve_asset(str(prop["usd_path"])),
            position=to_vec3(prop.get("position")),
            orientation=prop_spawn_orientation(prop),
            collider_dims_m=collider_dims_m,
            mesh_dims_m=None,
            bounds_m=None,
        )

    def registry(self) -> dict[str, dict[str, Any]]:
        """The live registry, keyed by prim name (tests read it)."""
        return self._registry

    def _entry(self, name: str) -> dict[str, Any]:
        entry = self._registry.get(prim_name(name))
        if entry is None:
            raise ValueError(f"unknown prop {name!r}, have {sorted(self._registry)}")
        return entry

    def status(self) -> dict[str, Any]:
        return self._sim.status()

    def play(self) -> None:
        self._sim.play()

    def pause(self) -> None:
        self._sim.pause()

    def reset(self, soft: bool = False) -> None:
        for entry in self._registry.values():
            entry["position"] = entry["spawn_position"]
            entry["orientation"] = entry["spawn_orientation"]
        if not soft:
            self._sim.reset()

    def add_usd(
        self,
        usd_path: str,
        prim_path: str,
        position_m: Vec3,
        orientation_wxyz: Quat | None = None,
    ) -> None:
        self._sim.add_usd_reference(usd_path, prim_path, position_m)

    def prop_geometries(self) -> list[PropGeometry]:
        out: list[PropGeometry] = []
        for name, entry in self._registry.items():
            spawn = entry["spawn"]
            color = spawn.get("color")
            out.append(
                PropGeometry(
                    name=name,
                    box_dims_m=prop_box_dims(spawn),
                    position_m=entry["position"],
                    orientation_wxyz=entry["orientation"],
                    color=(float(color[0]), float(color[1]), float(color[2]))
                    if color is not None
                    else None,
                    fixed=bool(spawn.get("fixed", False)),
                )
            )
        return out

    def set_prop_pose(
        self, name: str, position_m: Vec3, orientation_wxyz: Quat | None = None
    ) -> None:
        _require_not_visual(self._sim, "set_prop_pose", [name])
        entry = self._entry(name)
        entry["position"] = to_vec3(position_m)
        if orientation_wxyz is not None:
            entry["orientation"] = _as_quat(orientation_wxyz)

    def randomize_props(
        self,
        names: list[str],
        region: tuple[Vec3, Vec3],
        seed: int,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        size_range_m: dict[str, tuple[float, float]] | None = None,
    ) -> RandomizeResult:
        _require_not_visual(self._sim, "randomize_props", names)
        _validate_size_range_names(names, size_range_m)
        entries = {name: self._entry(name) for name in names}
        if size_range_m:
            for name in size_range_m:
                _require_cube_prop(name, entries[name]["spawn"])
        dims = {name: prop_box_dims(entries[name]["spawn"]) for name in names}
        placed, drawn_dims = _draw_sizes_and_positions(
            names, dims, region, seed, min_separation_m, size_range_m
        )
        self._rescale_and_reset(drawn_dims, size_range_m)
        for name in names:
            self.set_prop_pose(name, placed[name])
        dims_m = {name: prop_box_dims(entries[name]["spawn"]) for name in names}
        return RandomizeResult(positions_m=placed, dims_m=dims_m)

    def _rescale_and_reset(
        self,
        drawn_dims: dict[str, tuple[float, float, float]],
        size_range_m: dict[str, tuple[float, float]] | None,
    ) -> None:
        """Absolute rescale against spawn size (never compounds), then
        parity with IsaacWorldHandle: a sized call replays spawn_prop's
        full-reset pattern (all props snap to spawn poses, hooks fire)
        before any teleport. A no-range call is a no-op."""
        for name in size_range_m or ():
            drawn_edge = drawn_dims[name][0]
            self._registry[name]["spawn"] = {
                **self._registry[name]["spawn"],
                "size": drawn_edge,
                "scale": (1.0, 1.0, 1.0),
            }
        if size_range_m:
            for entry in self._registry.values():
                entry["position"] = entry["spawn_position"]
                entry["orientation"] = entry["spawn_orientation"]
            self._sim._reset_world()

    def scatter_cell(
        self,
        names_by_color: dict[str, list[str]],
        region: tuple[Vec3, Vec3],
        park_positions_m: dict[str, tuple[float, float]],
        seed: int,
        size_range_m: tuple[float, float] | None = None,
        counts: dict[str, int] | None = None,
    ) -> ScatterCellResult:
        _require_not_visual(
            self._sim, "scatter_cell", [name for names in names_by_color.values() for name in names]
        )
        pool_names = _require_pool_blocks(self._registry, names_by_color, "scatter_cell")
        _require_park_positions(pool_names, park_positions_m, "scatter_cell")
        rng = random.Random(seed)
        drawn_counts = _draw_pool_counts(rng, names_by_color, counts)
        scattered = _scattered_block_names(names_by_color, drawn_counts)
        parked = [name for name in pool_names if name not in scattered]
        entries = {name: self._registry[name] for name in pool_names}
        dims = {name: prop_box_dims(entries[name]["spawn"]) for name in scattered}
        size_range_by_name = {name: size_range_m for name in scattered} if size_range_m else None
        placed, drawn_dims = _draw_sizes_and_positions(
            scattered,
            dims,
            region,
            seed,
            POOL_SCATTER_MIN_SEPARATION_M,
            size_range_by_name,
            rng=rng,
        )
        self._rescale_and_reset(drawn_dims, size_range_by_name)
        for name in scattered:
            self.set_prop_pose(name, placed[name])
        for name in parked:
            dims_now = prop_box_dims(entries[name]["spawn"])
            self.set_prop_pose(name, _park_pose_m(dims_now, park_positions_m[name]))
        sizes_m = {name: drawn_dims[name] for name in scattered}
        return ScatterCellResult(
            seed=seed, counts=drawn_counts, positions_m=placed, sizes_m=sizes_m, parked=parked
        )

    def clear_cell(
        self,
        names_by_color: dict[str, list[str]],
        park_positions_m: dict[str, tuple[float, float]],
    ) -> ClearCellResult:
        _require_not_visual(
            self._sim, "clear_cell", [name for names in names_by_color.values() for name in names]
        )
        pool_names = _require_pool_blocks(self._registry, names_by_color, "clear_cell")
        _require_park_positions(pool_names, park_positions_m, "clear_cell")
        for name in pool_names:
            dims_now = prop_box_dims(self._registry[name]["spawn"])
            self.set_prop_pose(name, _park_pose_m(dims_now, park_positions_m[name]))
        return ClearCellResult(parked=pool_names)

    def spawn_prop(self, prop: dict[str, Any]) -> None:
        self._register(prop)


def _zero_prop_velocity(sim: SimManager, prim_path: str) -> None:
    """Best-effort velocity zeroing after a teleport. Fixed props
    have no rigid-body API to zero (and never move on their own), so a
    missing or incompatible API is swallowed rather than blocking the
    teleport itself."""
    try:
        from isaacsim.core.prims import SingleRigidPrim as RigidPrim

        rigid = RigidPrim(prim_path)
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
    except Exception:  # noqa: BLE001 - a missing/incompatible rigid-body API can't block the teleport itself
        LOGGER.warning("could not zero velocity for %s after teleport", prim_path)


class IsaacWorldHandle(WorldHandle):
    """Drives the real sim by wrapping SimManager. Scene
    mutations run on the sim thread via SimManager.run."""

    def __init__(self, sim: SimManager) -> None:
        self._sim = sim

    def status(self) -> dict[str, Any]:
        return self._sim.status()

    def play(self) -> None:
        self._sim.play()

    def pause(self) -> None:
        self._sim.pause()

    def reset(self, soft: bool = False) -> None:
        if not soft:
            self._sim.reset()
            return

        def _restore() -> None:
            for name, spec in self._sim._prop_specs.items():
                self._teleport(name, spec["position"], spec.get("spawn_orientation"))

        self._sim.run(_restore)

    def add_usd(
        self,
        usd_path: str,
        prim_path: str,
        position_m: Vec3,
        orientation_wxyz: Quat | None = None,
    ) -> None:
        self._sim.add_usd_reference(usd_path, prim_path, position_m, orientation_wxyz)

    def prop_geometries(self) -> list[PropGeometry]:
        def _read() -> list[PropGeometry]:
            out: list[PropGeometry] = []
            for name, spec in self._sim._prop_specs.items():
                pos, quat = self._sim._isaac.SingleXFormPrim(f"/World/{name}").get_world_pose()
                color = spec.get("color")
                out.append(
                    PropGeometry(
                        name=name,
                        box_dims_m=prop_box_dims(spec),
                        position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                        orientation_wxyz=_as_quat(quat),
                        color=(float(color[0]), float(color[1]), float(color[2]))
                        if color is not None
                        else None,
                        fixed=bool(spec.get("fixed", False)),
                    )
                )
            return out

        return self._sim.run(_read)

    def _prop_spec(self, name: str) -> dict[str, Any]:
        spec = self._sim._prop_specs.get(prim_name(name))
        if spec is None:
            raise ValueError(f"unknown prop {name!r}, have {sorted(self._sim._prop_specs)}")
        return spec

    def _teleport(self, name: str, position_m: Vec3, orientation_wxyz: Quat | None) -> None:
        """Runs on the sim thread: teleport + velocity zeroing. Never touches
        ``_prop_specs`` (mock gate: a later reset must still restore the
        configured spawn pose). Cube props teleport through their scene
        object - the API PhysX tracks. A raw-XForm teleport of a live rigid
        body desyncs it and the prop can tumble. usd props are
        not scene-registered, so they keep the raw-XForm fallback."""
        prim_path = f"/World/{name}"
        kwargs: dict[str, Any] = {"position": list(position_m)}
        if orientation_wxyz is not None:
            kwargs["orientation"] = list(orientation_wxyz)
        scene_object = self._sim.world.scene.get_object(name)
        if scene_object is not None:
            scene_object.set_world_pose(**kwargs)
            for setter_name in ("set_linear_velocity", "set_angular_velocity"):
                setter = getattr(scene_object, setter_name, None)
                if setter is None:
                    continue
                try:
                    setter(np.zeros(3))
                except Exception:  # noqa: BLE001 - fixed props have no rigid-body velocity
                    LOGGER.warning(
                        "could not zero %s for %s, fixed props have no rigid-body velocity",
                        setter_name,
                        name,
                    )
            return
        self._sim._isaac.SingleXFormPrim(prim_path).set_world_pose(**kwargs)
        _zero_prop_velocity(self._sim, prim_path)

    def set_prop_pose(
        self, name: str, position_m: Vec3, orientation_wxyz: Quat | None = None
    ) -> None:
        _require_not_visual(self._sim, "set_prop_pose", [name])
        self._prop_spec(name)  # ValueError on unknown name
        spawned_prim_name = prim_name(name)
        self._sim.run(lambda: self._teleport(spawned_prim_name, position_m, orientation_wxyz))

    def randomize_props(
        self,
        names: list[str],
        region: tuple[Vec3, Vec3],
        seed: int,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        size_range_m: dict[str, tuple[float, float]] | None = None,
    ) -> RandomizeResult:
        _require_not_visual(self._sim, "randomize_props", names)
        _validate_size_range_names(names, size_range_m)
        specs = {name: self._prop_spec(name) for name in names}
        if size_range_m:
            for name in size_range_m:
                _require_cube_prop(name, specs[name])
        dims = {name: prop_box_dims(specs[name]) for name in names}
        placed, drawn_dims = _draw_sizes_and_positions(
            names, dims, region, seed, min_separation_m, size_range_m
        )
        self._rescale_and_reset(specs, drawn_dims, size_range_m)
        for name, position in placed.items():
            self.set_prop_pose(name, position)
        dims_m = {name: prop_box_dims(specs[name]) for name in names}
        return RandomizeResult(positions_m=placed, dims_m=dims_m)

    def _rescale_and_reset(
        self,
        specs: dict[str, dict[str, Any]],
        drawn_dims: dict[str, tuple[float, float, float]],
        size_range_m: dict[str, tuple[float, float]] | None,
    ) -> None:
        """A no-op with no ranges. Otherwise runs on the sim thread. Stop
        BEFORE writing the scale: the stop inside reset restores the
        stage's pre-play state, so a scale authored mid-play is reverted
        (GPU: drawn 44.7 mm, block stayed 60 mm) - and a live-play write
        also invalidates PhysX's tensor view (GPU: "Failed to get rigid
        body transforms from backend"). Then spawn_prop's pattern: full
        reset re-cooks the colliders at the new scale and refires the
        post-reset hooks (arm gains, camera re-inits) before any teleport
        touches a pose."""
        if not size_range_m:
            return

        def _rescale_and_rebuild() -> None:
            self._sim.world.stop()
            self._rescale_props(specs, drawn_dims, size_range_m)
            self._sim._reset_world()

        self._sim.run(_rescale_and_rebuild)

    def scatter_cell(
        self,
        names_by_color: dict[str, list[str]],
        region: tuple[Vec3, Vec3],
        park_positions_m: dict[str, tuple[float, float]],
        seed: int,
        size_range_m: tuple[float, float] | None = None,
        counts: dict[str, int] | None = None,
    ) -> ScatterCellResult:
        _require_not_visual(
            self._sim, "scatter_cell", [name for names in names_by_color.values() for name in names]
        )
        pool_names = _require_pool_blocks(self._sim._prop_specs, names_by_color, "scatter_cell")
        _require_park_positions(pool_names, park_positions_m, "scatter_cell")
        rng = random.Random(seed)
        drawn_counts = _draw_pool_counts(rng, names_by_color, counts)
        scattered = _scattered_block_names(names_by_color, drawn_counts)
        parked = [name for name in pool_names if name not in scattered]
        specs = {name: self._prop_spec(name) for name in pool_names}
        dims = {name: prop_box_dims(specs[name]) for name in scattered}
        size_range_by_name = {name: size_range_m for name in scattered} if size_range_m else None
        placed, drawn_dims = _draw_sizes_and_positions(
            scattered,
            dims,
            region,
            seed,
            POOL_SCATTER_MIN_SEPARATION_M,
            size_range_by_name,
            rng=rng,
        )
        self._rescale_and_reset(specs, drawn_dims, size_range_by_name)
        for name, position in placed.items():
            self.set_prop_pose(name, position)
        for name in parked:
            dims_now = prop_box_dims(specs[name])
            self.set_prop_pose(name, _park_pose_m(dims_now, park_positions_m[name]))
        sizes_m = {name: drawn_dims[name] for name in scattered}
        return ScatterCellResult(
            seed=seed, counts=drawn_counts, positions_m=placed, sizes_m=sizes_m, parked=parked
        )

    def clear_cell(
        self,
        names_by_color: dict[str, list[str]],
        park_positions_m: dict[str, tuple[float, float]],
    ) -> ClearCellResult:
        _require_not_visual(
            self._sim, "clear_cell", [name for names in names_by_color.values() for name in names]
        )
        pool_names = _require_pool_blocks(self._sim._prop_specs, names_by_color, "clear_cell")
        _require_park_positions(pool_names, park_positions_m, "clear_cell")
        specs = {name: self._prop_spec(name) for name in pool_names}
        for name in pool_names:
            dims_now = prop_box_dims(specs[name])
            self.set_prop_pose(name, _park_pose_m(dims_now, park_positions_m[name]))
        return ClearCellResult(parked=pool_names)

    def _rescale_props(
        self,
        specs: dict[str, dict[str, Any]],
        drawn_dims: dict[str, tuple[float, float, float]],
        size_range_m: dict[str, tuple[float, float]],
    ) -> None:
        """Runs on the sim thread: absolute rescale against the spawn
        size (never the previous draw), so repeated randomize calls don't
        compound (dynamic-blocks)."""
        for name in size_range_m:
            spec = specs[name]
            spawn_size = float(spec.get("size", 0.05))
            scale_factor = drawn_dims[name][0] / spawn_size
            scale = (scale_factor, scale_factor, scale_factor)
            scene_object = self._sim.world.scene.get_object(name)
            if scene_object is not None:
                set_local_scale = getattr(scene_object, "set_local_scale", None)
                if set_local_scale is not None:
                    set_local_scale(np.array(scale))
            spec["scale"] = scale

    def spawn_prop(self, prop: dict[str, Any]) -> None:
        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = prim_name(str(prop["name"]))
        if name in self._sim._prop_specs:
            raise ValueError(f"prop {name!r} already exists")

        def _spawn() -> None:
            self._sim._spawn_prop(prop)
            self._sim._reset_world()

        self._sim.run(_spawn)
