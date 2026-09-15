# viam-isaac-sim

A [Viam](https://www.viam.com) module for controlling and simulating robots in
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim).

You configure the core `world` component and it boots the simulator. You add
the components you want to drive, an arm with `"asset": "ur20"` for example,
and each one spawns the right prim in the stage and answers the normal Viam
APIs. You control the robots and read the cameras exactly as you would on real
hardware, and you watch the scene through Viam camera components or the
built-in WebRTC livestream.

Start here: [`getting-started.md`](getting-started.md) takes a new machine from
an empty stage to a sorting cell that runs itself.

## Models

| Model | Viam API | What it does |
|---|---|---|
| `viam:isaac-sim-devin:world` | `rdk:component:generic` | Boots Isaac Sim, opens the USD stage, runs the sim loop. Configure exactly one. |
| `viam:isaac-sim-devin:scene-finalizer` | `rdk:component:generic` | Built last (its `depends_on` names every sim component) and tells the world the scene is populated, so a cold start reads as initializing instead of timing out. |
| `viam:isaac-sim-devin:arm` | `rdk:component:arm` | Spawns (or attaches to) an articulation, whether a UR arm, a Franka, or any USD, and exposes joint control. |
| `viam:isaac-sim-devin:camera` | `rdk:component:camera` | Creates (or attaches to) a camera prim and serves RGB, depth (`image/vnd.viam.dep`) and point clouds (`pointcloud/pcd`). |
| `viam:isaac-sim-devin:base` | `rdk:component:base` | Spawns a differential-drive robot (e.g. jetbot) and drives it. |
| `viam:isaac-sim-devin:gripper` | `rdk:component:gripper` | Bolts a parallel-jaw gripper (e.g. Robotiq 2F-85) onto an arm's link and drives it open/closed. |
| `viam:isaac-sim-devin:vacuum` | `rdk:component:gripper` | Bolts a suction tool onto an arm's link and takes hold by welding whatever is under the cup to it. |
| `viam:isaac-sim-devin:conductor` | `rdk:service:generic` | Sorts a scattered pool of colored blocks onto per-color pads end to end via DoCommand (`start`/`stop`/`status`), single-shot, N loops, or continuous. |
| `viam:isaac-sim-devin:sorter-sensor` | `rdk:component:sensor` | Proxies a conductor's `status` for data management, emitting each new loop record at most once. |
| `viam:isaac-sim-devin:palletizer` | `rdk:service:generic` | Picks a box off the pick station and places it on the pallet via DoCommand (`start`/`stop`/`status`). |

The conductor and the palletizer are services, not components, so they belong in a
config's `services` array. Every other model above is a component.

Known assets (usable via the `asset` attribute): `ur3e`, `ur5e`, `ur10`,
`ur10e`, `ur16e`, `ur20`, `franka`, `jetbot`. Anything else can be loaded with
`usd_path`, or attach to prims already in your stage with `prim_path`.

## Machine requirements and automatic setup

On a standard Ubuntu 22.04/24.04 x86_64 machine, the module sets itself up.
When it is first installed, viam-server runs `first_run.sh`. That script
installs the system libraries kit needs (vulkan/GL) and the right python (via
deadsnakes on 24.04). It installs an NVIDIA driver if none is present, then
Isaac Sim itself, pip-installed into a venv under the module's data directory
(4.5.0 on 22.04, 5.0.0 on 24.04). The driver it installs is the validated 580 branch,
and newer is not better here, see the driver bullet below. `run.sh` finds that
install automatically. The EULA is accepted via environment variable.

Notes on the automatic setup:

* The Isaac Sim download is 10GB+. If it exceeds viam-server's default
  first-run timeout, set `"first_run_timeout": "2h0m0s"` on the module entry
  in your machine config.
* If the script had to install the NVIDIA driver, **reboot** before
  configuring the world component.
* Already have Isaac Sim? Set `ISAAC_SIM_PATH` (dir containing `python.sh`)
  or `ISAAC_PYTHON` in the module's environment variables and the script
  skips everything.

What the machine must already be or have, since the script can't do these for
you:

* Ubuntu 22.04 or 24.04 on x86_64 with an RTX-capable NVIDIA GPU (8GB+ VRAM
  minimum, RTX 4080+/L40 recommended), 32GB+ RAM, ~60GB free disk. See
  NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
* An NVIDIA driver from a branch Isaac Sim validates against, and that means
  **the 580 branch**. Newer branches (590/595+) are known to crash Isaac's RTX
  renderer on startup (`librtx.scenedb.plugin.so`) and break CUDA init
  (`cuDeviceGetUuid` Warp errors), see
  [isaac-sim/IsaacSim#537](https://github.com/isaac-sim/IsaacSim/issues/537)
  and [isaac-sim/IsaacSim#643](https://github.com/isaac-sim/IsaacSim/issues/643).
  If you're on 595+: `sudo apt-get install -y nvidia-driver-580 && sudo reboot`.
* viam-server installed, machine online in the Viam app, running as root (or
  a user with passwordless sudo) for the apt/driver steps.
* Network access to pypi.nvidia.com, pypi.org, and NVIDIA's asset servers.
* Open ports for the livestream viewer if you want it: TCP 49100 (signaling)
  plus UDP 47998 (media). Both are hardcoded in NVIDIA's streaming client. On
  GCP, `provisioning/create-sim-machine.sh --open-livestream` opens them for
  you (see [`provisioning/README.md`](provisioning/README.md)).

No GPU and no Isaac Sim? `"mock": true` on the world runs the module anywhere
for development, see "Development without Isaac Sim" below.

[`provisioning/README.md`](provisioning/README.md) builds a GCP image with the
driver, python and Isaac Sim already baked in, and creates a sim machine from
it in one command.

## Install and first run

1. Create the machine. Use the `Add machine` button on your
   [fleet page](https://app.viam.com/fleet/), then install viam-server on the
   GPU host with the credentials the app hands you.
2. Add the world. In the machine's `CONFIGURE` tab, install the
   `isaac-sim-world-devin` fragment, which carries the `viam:isaac-sim-devin`
   registry module entry and one `isaac-world` component. The module is
   private, so the machine must be in an org that can see it.
3. Save, then watch the `LOGS` tab until `isaac-world` comes up. First boot
   pays for the Isaac Sim install described above.
4. Open the `isaac-world` livestream in the `CONTROL` tab. An empty stage,
   with no tables and no arm, is a working module.

[`getting-started.md`](getting-started.md) is the same path in full, through to
the block-sorting cell. Add components to the world by hand from
"Minimal machine config" below.

## Development without Isaac Sim (mock mode)

Set `"mock": true` on the world component and the module runs anywhere python
does. Arms integrate joint targets over time, cameras produce synthetic frames,
bases accept velocity commands, and a gripper's `grab()` succeeds only when its
`mock_object_width_m` attribute is set, that being the width of the object
between the jaws. Unset means nothing to grab. This is what the test suite
uses. Try it end to end with `PYTHONPATH=src python examples/pick_red_block.py
--mock`.

Set up a dev venv and run the checks with `make`:

```sh
uv venv --python 3.11 .venv && uv pip install -r requirements-dev.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make fmt-check lint typecheck test
```

Tests run under the `pyproject.toml` config (`pythonpath = ["src"]`). The
`gpu` marker is skipped by default, so `make test` only runs mock-mode tests
and needs no Isaac Sim install. CI runs the same suite on both Python 3.10
and 3.11.

## Minimal machine config

One world and one arm is the smallest useful machine:

```json
{
  "components": [
    {
      "name": "isaac-world",
      "api": "rdk:component:generic",
      "model": "viam:isaac-sim-devin:world",
      "attributes": {
        "headless": true,
        "livestream": true
      }
    },
    {
      "name": "my-ur20",
      "api": "rdk:component:arm",
      "model": "viam:isaac-sim-devin:arm",
      "frame": { "parent": "world" },
      "attributes": {
        "world": "isaac-world",
        "asset": "ur20"
      }
    }
  ]
}
```

A camera and a base are configured the same way:

```json
{
  "components": [
    {
      "name": "overhead-cam",
      "api": "rdk:component:camera",
      "model": "viam:isaac-sim-devin:camera",
      "frame": {
        "parent": "world",
        "translation": { "x": 2000, "y": 2000, "z": 2000 }
      },
      "attributes": {
        "world": "isaac-world",
        "target": [0, 0, 0.5],
        "width": 1280,
        "height": 720
      }
    },
    {
      "name": "my-jetbot",
      "api": "rdk:component:base",
      "model": "viam:isaac-sim-devin:base",
      "frame": {
        "parent": "world",
        "translation": { "x": 1000, "y": 0, "z": 100 }
      },
      "attributes": {
        "world": "isaac-world",
        "asset": "jetbot"
      }
    }
  ]
}
```

Every non-world component must set `"world"` to the world component's name.
That attribute is also returned as an implicit dependency from each model's
validate, so viam-server starts the world first and no `depends_on` is needed.

Components are **placed with the standard frame config** (translations in mm,
any orientation representation). The spawn pose in Isaac and viam's frame
system then agree, so things like the motion service see components where
they are. The `position` (meters) / `orientation_rpy_deg` attributes
still work as a fallback when no frame is set. A camera `target` attribute
overrides orientation to aim at a point.

**Frames:** a spawned component's `frame.parent` must be `"world"`. The
spawn path does not resolve an arbitrary frame chain, only the world's own
origin.

The one exception is a component that also sets `parent_prim`, such as a wrist
camera riding an arm link. It must instead name the component that owns that
prim as its `frame.parent`. With `parent_prim "/World/pick_arm/wrist_3_link"`,
`frame.parent` must be `"pick-arm"` or a sub-frame like `"pick-arm:ee_link"`.

For such a mounted component, `frame.parent "world"` or no `frame` at all is
rejected with a validation error. The frame is the single source of truth for
the mount, and its translation and orientation become the camera's local pose
on the link.

## Component reference

One section per model, listing every attribute it reads, plus the shared units
and conventions. The conductor's and sorter-sensor's attributes are in
[`docs/BLOCK_SORTING.md`](docs/BLOCK_SORTING.md) with the rest of the demo cell.

### world attributes

| attribute | default | notes |
|---|---|---|
| `mock` | `false` | run without Isaac Sim installed (development/testing) |
| `headless` | `true` | no local GUI window |
| `livestream` | `true` | viewer for the Isaac Sim WebRTC Streaming Client, on TCP 49100 and UDP 47998 |
| `livestream_public_ip` | _unset_ | IP advertised to streaming clients when the sim machine has multiple interfaces |
| `usd_stage` | _empty stage + ground plane_ | USD file or omniverse:// URL to open. When set without `lighting`, the module logs a warning that the stage must provide its own floor and lights (it adds neither to a user stage) |
| `physics_dt` / `rendering_dt` | `1/60` | step sizes in seconds. The block-sorting cell uses `1/120` for `physics_dt` (>= 80 steps/s is the floor for a 2F-85 grasp), rendering `1/30` (measured 2026-09-14: at `1/60` the matte HDRI floor halved the sim's real-time factor, at `1/30` it runs at 0.54x, faster than the old grid floor at `1/60`) |
| `boot_timeout_sec` | `110` | stays under viam-server's 2 minute resource configuration timeout (`VIAM_RESOURCE_CONFIGURATION_TIMEOUT`) |
| `wait_for_finalizer` | `false` | defer world steps until a `scene-finalizer` component runs, see below |
| `kit_log_level` | `"warning"` | kit console verbosity |
| `props` | `[]` | objects spawned into the scene at boot, see below |
| `lighting` | _unset_ | dome and sphere lights applied at boot, see below |
| `ground` | _unset_ | the floor the module adds when it owns the stage, default is today's grid environment, see below |
| `render` | _unset_ | render-cost levers applied at boot, see below |

`lighting` takes `{"dome": {"intensity": 1000, "color": [1, 1, 1]},
"sphere_intensity": 30000}`. Both keys are optional, and leaving it unset
leaves the stage's lights alone. The default stage has a single
100 000-intensity sphere light, so a dome light is useful to even out color
for detection. It is applied at boot only, and a change takes effect after a
part restart. `dome` also takes `texture` (a path, URL, `module://` or
`data://` HDRI, see "Asset paths" below), `texture_format` (a UsdLux dome
format, default `"latlong"`, one of `"automatic"`, `"latlong"`,
`"mirroredBall"`, `"angular"`, `"cubeMapVerticalCross"`), and `rotation_deg`
(yaw about Z). Kit orients the dome to the Z-up stage on its own, so no tilt
is authored. The bundled 1k HDRI lights the scene well but reads soft as a
backdrop through a 60 degree camera. Use a 4k or larger file under `data://`
for a sharp backdrop.

`ground` takes `{"kind": "grid" | "plane" | "none", "color": [r, g, b],
"size": m, "friction": f, "restitution": r, "matte": bool, "material": ...}`.
`kind` defaults to `"grid"`, today's default environment. `"plane"` adds a
plain ground plane, with plane-only keys `color` (default `[0.5, 0.5, 0.5]`),
`size` (default `100` m), `friction` (default `0.5`), `restitution` (default
`0`), `matte` (default `false`) and `material` (the same shape as a prop's
`material`, see "Bundled material sets" above). An explicit `ground.color`
acts as the tint for a named set, and has no visible effect under
`matte: true`. `"none"` adds no floor at all, so a block
knocked off the table falls forever. `ground` is ignored with a warning when
`usd_stage` is set, since the module adds a floor only to the stage it owns.
When `matte` is true, the plane is invisible to the camera but still catches
shadows, so the HDRI's own floor shows through under the tables while the
collider stays. This reads right from a fixed camera such as `scene-cam`,
since the dome sits at infinity, but slides under a moving one.

`render` takes `{"motion_bvh": bool, "disable_viewport_updates": bool,
"viewport_grid": bool}`, best-effort. All keys are optional, and leaving it
unset leaves the renderer's defaults alone. `disable_viewport_updates: true`
requires `livestream: false`, since the livestream needs viewport updates,
and is refused otherwise. `viewport_grid: false` hides the viewport grid
overlay, applied after launch.

**Asset paths:** a texture or USD path in world config may use one of two
module schemes in addition to whatever Isaac's own resolver accepts.
`module://<rel>` resolves under the module's bundled `assets/` directory,
shipped inside `module.tar.gz`. `data://<rel>` resolves under
`$VIAM_MODULE_DATA/assets/<rel>` (default `/opt/viam-isaac-sim` when
`VIAM_MODULE_DATA` is unset). Anything else is handed to Isaac's resolver
unchanged. The bundled HDRI, `assets/hdri/empty_warehouse_01_1k.hdr`, is CC0
(Poly Haven). The 1k file is for lighting, and a sharp backdrop wants a 4k
or larger file under `data://`, for example
`"lighting": {"dome": {"texture": "data://hdri/empty_warehouse_01_8k.hdr"}}`.

`wait_for_finalizer: true` holds the world at the boot pose, draining its task
queue but not stepping, until a `scene-finalizer` component is built (see
"scene-finalizer" below). The renderer compiles shaders during the first
`world.step` calls after the scene changes, and on a cold shader cache that
can take minutes per step. Without the gate, a queued call made during one of
those steps times out, so viam-server's resource builds fail and retry for
minutes. With the gate, those slow steps run once, after every sim component
is built, and a caller meets `UNAVAILABLE` instead of a timeout. A
`scene-finalizer` naming this world and every sim component:

```json
{
  "name": "cell-ready",
  "api": "rdk:component:generic",
  "model": "viam:isaac-sim-devin:scene-finalizer",
  "depends_on": ["isaac-world", "pick-arm", "pick-grip", "scene-cam"]
}
```

While the world is initializing, every sim resource's operational calls fail
with gRPC `UNAVAILABLE` and the message `Isaac Sim is initializing; retry
shortly`, so a client retries instead of timing out. The world's `status`
DoCommand still answers, reporting `"ready": false` until the finalizer runs
and three world steps complete.

Each entry in `props` is an object:

| key | value |
|---|---|
| `name` | string, snake_cased for the prim path |
| `type` | `"cube"`, `"usd"` or `"visual"` |
| `position` | `[x,y,z]` meters, the prop's **center** |
| `size` | meters, the cube's base edge length, > 0 |
| `scale` | `[sx,sy,sz]`, multiplies `size` per axis |
| `color` | `[r,g,b]`, each in `[0, 1]` |
| `material` | `"cube"` only. Either a bundled set name (see "Bundled material sets" below) or an object with `albedo`, `normal`, `roughness`, `metallic` (texture paths, the "Asset paths" schemes apply), `tint` (`[r,g,b]` each in `[0, 1]`, exclusive with `color`) and `texture_scale` (`[u,v]`, each > 0). A named set uses the prop's `color` as its tint |
| `fixed` | bool, static rather than dynamic and physics-driven |
| `usd_path` | required when `type` is `"usd"` or `"visual"`. A `"visual"` path may use `module://` or `data://` (see "Asset paths" below). An empty string on a `"visual"` skips the prop, so a fragment variable can default to no asset |
| `fit` | `"visual"` only, exclusive with `scale`: `"true"` keeps the asset's authored size, `{"collider": "<cube prop name>"}` scales the asset's bounds onto that cube's `size x scale` box, per axis |
| `orientation_rpy_deg` | `[r,p,y]` degrees, the prop's initial orientation |
| `orientation_wxyz` | `[w,x,y,z]`, not all zero, the same thing the other way |
| `box_dims` | `[x,y,z]` meters, each > 0, the obstacle box for a `"usd"` prop whose geometry this module can't infer from the asset |
| `mass` | kg, > 0 |
| `friction` | unitless, static = dynamic, >= 0, combine mode `max` |
| `restitution` | unitless, in `[0, 1]` |
| `contact_offset` | m, >= 0 |
| `rest_offset` | m, >= 0, <= `contact_offset` when both are set |

At most one of `orientation_rpy_deg` and `orientation_wxyz` may be set.
Without `box_dims`, an unknown-size `"usd"` prop's `get_geometries` /
`prop_geometries` box is all zero and it is skipped as an obstacle, see "Props
and obstacles" in [`docs/BLOCK_SORTING.md`](docs/BLOCK_SORTING.md). The five physics
keys apply only when set. Otherwise Isaac's authored defaults are left alone.
The shipped block-sorting cell sets `mass: 0.05, friction: 0.7, restitution: 0,
contact_offset: 0.005` on the block and `friction: 0.7, restitution: 0` on the
(fixed, so massless) place pad.

A `"visual"` prop is a referenced USD given a pose and a scale, with no
collider and no rigid body. It never appears in `prop_geometries`,
`get_geometries`, `randomize_props`, `set_prop_pose`, `scatter_cell` or
`clear_cell`. Each of those verbs rejects its name with a `ValueError`. The
converter that produces a visual asset (`tools/convert_mesh.py`, see
[`tools/README.md`](tools/README.md)) puts the asset's origin at the centre
of its top face, so `position` is the collider's top-centre, for example
`[-1.2, 0, 0.75]` for `table_source`. A fitted collider is hidden from the
renderer (USD visibility `invisible`) so only the mesh shows. Its physics is
unchanged, and a visual that fails to spawn leaves its collider visible.
`fit: {"collider": "..."}` resolves
against the other entries in the same `props` list at boot, so a visual prop
spawned later through `spawn_prop` must use `scale` or `fit: "true"` instead.
The world's `status` DoCommand lists visual props under `visual_props`, one
row per prop with `name`, `usd_path`, `resolved_path`, `position`, `scale`,
`collider_dims_m`, `mesh_dims_m` and `bounds_m`. The last two are `null` in
the mock, which has no stage to measure.

#### Bundled material sets

| set | maps | default texture_scale | look |
|---|---|---|---|
| `painted_wood` | `normal`, `roughness` | `[0.2, 0.2]` | satin painted wood grain in the prop's colour |
| `painted_mat` | `normal`, `roughness` | `[0.3, 0.3]` | matte rubber mat in the prop's colour |
| `concrete_floor` | `albedo`, `normal`, `roughness` | `[1.0, 1.0]` | light smooth concrete for a `ground` with `matte: false` |

Every set is CC0 from ambientCG at 1k under `assets/materials/<set>/`, with a
`LICENSE.md`. The two tinted sets carry no albedo map, so the six hue
detectors see one flat hue per face. A local map path that is not on disk
logs a warning, and the prop renders its flat colour. The binding sits on the
prim, so a material survives `randomize_props`, `scatter_cell`, `clear_cell`
and `reset`. `texture_scale` stretches with a rescaled prim. The
`texture_scale` defaults are first guesses that the GPU pass tunes.

The GrabCAD table asset the converter is built for is not committed to this
repository. Read GrabCAD's terms before committing or publishing a converted
copy in the module. The converted USD used on the GPU VM lives under
`data://` instead.

`props` validation rules (`ValueError` on config, surfaced as
`INVALID_ARGUMENT`):
* names must be unique once snake_cased (the same normalisation used for prim
  paths)
* `type` must be `"cube"`, `"usd"` or `"visual"`
* `usd_path` is required when `type` is `"usd"` or `"visual"`
* a `"visual"` prop rejects `size`, `color`, `fixed`, `box_dims`, `mass`,
  `friction`, `restitution`, `contact_offset` and `rest_offset`, since it has
  no collider and no rigid body
* `scale` and `fit` are exclusive
* `fit` must be `"true"` or `{"collider": "<name>"}` naming a `"cube"` prop in
  the same `props` list
* `fit` is only valid on a `"visual"` prop
* `position`, `scale`, and `color` must each be 3-number sequences
* `color` values must be in `[0, 1]`
* `size` must be a positive number
* at most one of `orientation_rpy_deg` / `orientation_wxyz` may be set, and
  `orientation_wxyz` must not be all zero
* `box_dims` values must be positive
* `material` is `"cube"` only
* a named `material` must be a bundled set (see "Bundled material sets" above)
* the object form of `material` needs at least one map or a `tint`
* `color` and `material.tint` are exclusive
* `material.tint` values must be in `[0, 1]`
* `material.texture_scale` values must be positive

The world also supports `DoCommand`:

* `{"command": "status" | "play" | "pause"}`
* `{"command": "reset", "soft"?: bool (default false)}`
* `{"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
  "position": [x, y, z] meters, "orientation_rpy_deg"?: [r, p, y] degrees}`,
  which drops an extra USD reference into the scene
* `{"command": "prop_geometries"}` -> `{"geometries": [{"name",
  "box_dims_mm": [x,y,z], "pose_in_world_mm": {"x", "y", "z", "o_x", "o_y",
  "o_z", "theta"} (theta in degrees), "color": [r,g,b] or `None`, "fixed":
  bool}]}`, every prop's current box and pose, in millimeters, for a client
  that builds its own `WorldState`
* `{"command": "spawn_prop", "prop": {...same schema as the `props` config
  attribute...}}`, which spawns a prop at runtime
* `{"command": "set_prop_pose", "name": "...", "position": [x,y,z] mm,
  "orientation_rpy_deg"?: [r,p,y] degrees}`, which moves an existing prop
* `{"command": "randomize_props", "names": [...], "region": [[x0,y0,z],
  [x1,y1,z]] mm, "seed": int, "min_separation"?: mm (default 150),
  "size_range_mm"?: [lo, hi] (applies to every named prop) or
  {name: [lo, hi]} (keys must be a subset of `names`), cube props only,
  0 < lo <= hi}` -> `{"positions": {name: [x,y,z] mm}, "sizes_mm":
  {name: [x,y,z] mm}}`, which scatters the named props inside the region,
  optionally redrawing each ranged prop's size first
* `{"command": "ignore_props", "names": [...]}` -> `{"ignored": [...]}`. An
  empty list clears the exclusion. Excludes the named props from
  `GetGeometries` (e.g. the block currently being grasped)
* `{"command": "scatter_cell", "seed": int, "names_by_color": {color:
  [names]}, "region": [[x0,y0,z0], [x1,y1,z1]] mm, "park_positions_mm":
  {name: [x,y]}, "size_range_mm"?: [lo, hi] (applies to every drawn block,
  0 < lo <= hi, omit to keep current sizes), "counts"?: {color: int}
  (overrides the default 1-3 per-color draw for that color)}` -> `{"seed",
  "counts": {color: n}, "positions": {name: [x,y,z] mm}, "sizes_mm": {name:
  [x,y,z] mm}, "parked": [names]}`, which draws a fresh sorting problem
  from the named pool. The world knows no cell, so the caller (the
  conductor, from `cell_layout`) supplies the pool, region and park grid
* `{"command": "clear_cell", "names_by_color": {...}, "park_positions_mm":
  {...}}` -> `{"parked": [names]}`, which re-parks every pool block
* `{"command": "joint_state", "name": "<arm component>"}` -> `{"joints":
  [{"name", "named", "position_deg", "velocity_deg_s", "target_deg"}]}`,
  the named arm's per-joint state
* `{"command": "dof_names", "name": "<arm or gripper component>", "all"?:
  bool}` -> `{"dof_names": [...]}`, the named component's DOF names.
  `"all": true` (arms only) returns every DOF of the articulation, including
  anything attached under it, e.g. a gripper
* `{"command": "prim_pose", "name": "<arm or base component>",
  "prim_path"?: "..."}` -> `{"prim_path", "position_mm", "quaternion_wxyz",
  "orientation_vector"}`, the world pose of a prim under the named component.
  The default prim is the arm's end-effector prim, or the base's own root
  prim
* `{"command": "tcp_pose", "name": "<gripper component>"}` -> measured link
  poses plus a diagnostic comparing the configured `tcp_offset_m` against
  the geometry the sim actually measures. This is calibration tooling for
  placing a gripper's TCP
* `{"command": "jaw_deg", "name": "<gripper component>"}` -> `{"jaw_deg",
  "open_deg", "closed_deg"}`, the named gripper's current, open, and closed
  jaw angles

The `randomize_props`, `scatter_cell` and `clear_cell` verbs are worked
through with examples in [`docs/BLOCK_SORTING.md`](docs/BLOCK_SORTING.md).

### Units and conventions

Viam's frame system (component `frame` config, `GetEndPosition`, camera
`target`, etc.) uses **millimeters and degrees**, per the standard Viam
convention. The world's `props` attribute, by contrast, is Isaac-native:
**meters**, and `position` is always the prop's **center**, not a corner.
For a cube prop the rendered extent along each axis is `size × scale[axis]`
(so `size` is a base edge length and `scale` stretches it per axis). Isaac
Sim is Z-up, matching Viam's frame convention.

Worked example, a table as a `fixed` cube prop:

```json
{"type": "cube", "fixed": true, "size": 1.0, "scale": [1.2, 0.8, 0.75],
 "position": [0.60, 0.00, 0.375]}
```

The table top's height above the world origin is:

```
z_top = position.z + size * scale.z / 2
      = 0.375 + 1.0 * 0.75 / 2
      = 0.75 m
```

and its top face spans x ∈ [0.00, 1.20], y ∈ [-0.40, +0.40] (the cube is
centered at `position`, so each face sits `size * scale[axis] / 2` from the
center along that axis). Anything you place *on* the table, a block or a
mount frame, belongs at `z_top + <that thing's own half-height>`, e.g. a
block of `size` 0.05 sits with its center at `z_top + 0.025`.

### arm attributes

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name |
| `asset` | _one of asset/usd_path/prim_path required_ | known asset, e.g. `"ur20"` |
| `usd_path` | _one of asset/usd_path/prim_path required_ | arbitrary USD file or omniverse:// URL |
| `prim_path` | _one of asset/usd_path/prim_path required_ | attach to an articulation already in the stage |
| `position` | _unset_ | `[x,y,z]` meters, fallback when no `frame` is set |
| `orientation_wxyz` | _unset (identity)_ | `[w,x,y,z]`, fallback orientation when no `frame` is set, and the field a `frame` config is converted into |
| `end_effector_prim` | `<arm prim>/wrist_3_link` for UR assets, else unset | prim path whose pose is reported by `GetEndPosition`, converted to Viam's orientation-vector convention |
| `move_timeout_sec` | `30` | deadline for a move to converge |
| `max_vel_degs_per_sec` | _unset (drive's own limit)_ | default velocity cap for a move that carries no `MoveOptions` cap of its own, e.g. a real UR's `speed_degs_per_sec` |
| `kinematics_url` | _unset_ | SVA `.json` or `.urdf` (http(s):// or file://) for assets `GetKinematics` doesn't fetch automatically |

**`GetEndPosition` reports the end effector's pose in the arm base frame.**
This matches how a real arm driver reports its end position, and lets Viam's
frame system (via the component's `frame` config) compose it into world
frame itself.

`MoveToJointPositions` and `GetJointPositions` are implemented. `MoveToPosition`
solves inverse kinematics against the served kinematics file (SVA or URDF,
the same bytes `GetKinematics` returns) from the arm's current joints, so
the nearest solution wins. It then drives the joint path, and settle, stall
and timeout behave exactly as a joint move does. A target already within
1 mm and 0.06 degrees of the current pose is not moved to, matching the real
UR driver. There is no collision awareness, the same as a real driver's
direct move. Collision-free planning stays the motion service's job. The Viam
API defines `MoveToPosition` as a Cartesian straight line, and driving a joint
path instead is a documented deviation, recorded in
[`docs/PARITY.md`](docs/PARITY.md).

`GetGeometries` returns `[]`: rdk derives arm link geometry from
`GetKinematics` (the SVA already carries the link capsules) and never calls
`Geometries` for an arm that serves kinematics.

`GetKinematics` works: for `ur3e`/`ur5e`/`ur7e`/`ur20` the official viam SVA
kinematics files ship inside the module archive, so no fetch is needed. For
anything else set `kinematics_url` to an SVA `.json` or `.urdf` (http(s)://
or file://), cached in the module data dir after the first load. Either way
the file loads on `reconfigure`, on a background thread, so the first
`GetKinematics` or `MoveToPosition` after a machine comes up never blocks on
the load. With kinematics served, the motion service can plan for the
simulated arm.

UR assets (`ur3e`/`ur5e`/`ur7e`/`ur20`) get a built-in
**base-frame correction** applied at spawn so the arm's frame in Isaac lines
up with the kinematics Viam's motion service uses. Without it the sim and
Viam's idea of the arm's pose would silently disagree.

**`IsMoving`** is true while any named joint's `|velocity| > VEL_EPS_RAD_S`
OR any `|commanded - measured| > SETTLE_TOL_RAD`, so a stalled arm that never
reached its target keeps reporting `True`.

**Move errors**: a target outside the SVA's declared joint limits, or a
joint count that doesn't match the arm's DOF count, raises `INVALID_ARGUMENT`.
The arm stalling (settling short of its target, e.g. blocked by an obstacle)
raises `ABORTED`. The move deadline (`move_timeout_sec`, capped by the SDK's
`timeout=`) passing while still converging raises `DEADLINE_EXCEEDED`. For a
multi-waypoint trajectory, an intermediate waypoint that times out only
warns and continues (it uses a loose tolerance and short deadline so the arm
flows through it). An intermediate waypoint that stalls still raises
`ABORTED` (an obstacle blocking the path won't clear itself). The final
waypoint settles tight against the full move deadline. For `MoveToPosition`,
a pose no joint solution reaches raises `INVALID_ARGUMENT`, and a solution
that lands outside the SVA's declared joint limits raises `INVALID_ARGUMENT`
as above. An arm with no kinematics file configured (no `kinematics_url` and
no known asset kinematics) raises `FAILED_PRECONDITION`.

**`MoveOptions`**: `max_vel_degs_per_sec_joints` (per-joint velocity limits,
the min across joints) wins over the scalar `max_vel_degs_per_sec` when set.
When `MoveOptions` carries neither, the configured `max_vel_degs_per_sec`
attribute applies, unset meaning the drive's own limit. `MoveToJointPositions`
takes no `MoveOptions` at all, so it always uses the configured attribute.
The acceleration fields and `max_tcp_speed` are logged once and not honored.

**DoCommand** answers no sim-only verb. The world component's `DoCommand`
carries the arm's joint state, DOF names, and prim pose diagnostics, keyed
by this arm's component name (see "world attributes" above).

### gripper attributes

`world` (default `isaac-world`), `arm` (required, name of the `viam:isaac-sim-devin:arm`
component this gripper is bolted to).

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name |
| `arm` | _required_ | name of the arm it is bolted to |
| `asset` | `"robotiq_2f_85"` | known gripper asset |
| `parent_prim` | `<arm prim>/wrist_3_link` | link it is bolted to |
| `local_position` | _unset (identity)_ | `[x,y,z]` meters, mount pose of the gripper's `base_link` on `parent_prim` |
| `local_orientation_rpy_deg` | _unset (identity)_ | the 2F-85 base sits flush on the flange |
| `tcp_offset_m` | `0.134` | flange -> tool center point along tool +Z: the fingertip pad center as measured in Isaac (the pads span 115-153 mm) |
| `open_deg` | `0` | drive-joint angle when fully open |
| `closed_deg` | `47` on Isaac 5.0, `45` on 4.5 | drive-joint angle when fully closed (the Isaac-release value from `compat.caps()`) |
| `grab_timeout_sec` | `5` | how long `grab()` waits for a stall or full closure |
| `holding_tolerance_deg` | `2` | commanded-vs-measured gap that counts as holding |
| `mock_object_width_m` | _unset_ | mock only: width of the object between the jaws (unset = nothing to grab, so `grab()` returns `false`) |

**Frame**: unlike a mounted camera, the gripper's frame does not place its
prim. `base_link` bolts to `parent_prim` at `local_position` /
`local_orientation_rpy_deg`, and the frame's translation is the TCP the
motion service plans against (not the flange). `frame.parent` must be the
arm, and the translation is the TCP offset along the arm's tool axis, e.g.:

```json
{"frame": {"parent": "pick-arm", "translation": {"x": 0, "y": 0, "z": 134}}}
```

**Tool axis (confirmed on the GPU):** the arm's tool/approach axis is the
link frame's **+Z**, so gripper and wrist-camera `frame.translation` offsets
off an arm link both go along +Z.

**API mapping** (viam-sdk `Gripper`, all eight abstract methods): `stop` /
`is_moving` drive the handle directly. `open()` commands the jaw open and
blocks until it settles or the grab deadline passes, the same as `grab()`.
`grab()` closes the jaw, waits up to `grab_timeout_sec` for a stall or full
closure, and returns `is_holding_something()`. Both are stall-short-of-closure
checks, not a force sensor. `get_current_inputs()` / `go_to_inputs([v])` use a single
value in `[0, 1]`: `0` = open, `1` = closed. `GetKinematics` returns a
1-link/0-joint SVA whose link is the 36 × 146 × 153 mm gripper box (flange to
fingertips, center 57.5 mm behind the TCP). `GetGeometries` returns that same
single box.

**Asset loading (startup time):** Isaac 5.0's 2F-85 layer references its 11
part meshes from `omniverse://isaac-dev...`, a host that does not resolve
outside NVIDIA. Each such reference stalls the stage about 12 s while it
composes, 131 s per module start, past viam-server's 2-minute resource
timeout, so the gripper only came up on the retry. The module now opens the
gripper's layer on its own first, which resolves none of its sub-references.
It rewrites those references onto the public assets root and references the
rewritten copy, saved under `$VIAM_MODULE_DATA/viam-isaac-sim-assets/`
(the system temp dir when viam-server sets no data dir). Later starts reuse
the copy, so the gripper attaches in about a second. The module log line
`gripper '<name>' asset layer prepared|cached|...` says which path was taken.
Delete the directory to redo the preparation.


### vacuum attributes

`world` (default `isaac-world`), `arm` (required, name of the `viam:isaac-sim-devin:arm`
component this tool is bolted to). The tool is geometry this module authors rather than an
asset on the content server, so there is no `asset` attribute.

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name |
| `arm` | _required_ | name of the arm it is bolted to |
| `parent_prim` | `<arm prim>/wrist_3_link` | link it is bolted to |
| `local_position` | _unset (identity)_ | `[x,y,z]` meters, mount pose of the tool on `parent_prim` |
| `local_orientation_rpy_deg` | _unset (identity)_ | the tool sits flush on the flange |
| `tcp_offset_m` | `0.196` | flange -> cup face along tool +Z, the Robotiq EPick's own reach |
| `grab_delay_ms` | `1000` | how long `grab()` waits after engaging before it reports a hold, standing in for the pump cycle a real cup needs. `is_moving()` is true for that window |
| `max_payload_gap_m` | `0.01` | how far a candidate's top face may sit from the cup face and still count as contact, in either direction, since a descending cup often presses slightly into its payload |
| `mock_attach_prop` | _unset_ | mock only: name of the prop the cup finds under it (unset = nothing to grab, so `grab()` returns `false`) |

**Frame**: like the parallel-jaw gripper, the vacuum's frame is its TCP, which is the cup
face, so the motion service plans the cup onto the payload rather than the flange. Set
`frame.parent` to the arm.

**Holding versus engaged**: a cup commanded to take hold with nothing under it is engaged and
holding nothing, which a jaw cannot be. `get_current_inputs` reports the command, matching the
parallel-jaw model, where a jaw closed on nothing still reads at its commanded position.
`is_holding_something` reports the catch, and its `meta` carries both.

### camera attributes

`world` (default `isaac-world`), and either `prim_path` of an existing camera in your stage
or `position` plus `target` (aim-at point) or `orientation_rpy_deg` to create
one.

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name |
| `prim_path` | _unset_ | attach to an existing camera prim instead of spawning one |
| `position` | _unset_ | `[x,y,z]` meters, fallback when no `frame` is set |
| `target` | _unset_ | aim-at point, overrides orientation to point at it |
| `orientation_rpy_deg` | _unset_ | fallback orientation when no `frame` is set |
| `orientation_wxyz` | _unset (identity)_ | `[w,x,y,z]`, alternate fallback orientation when no `frame` is set and no `target`, and the field a `frame` config is converted into |
| `local_position` | `[0, 0, 0.05]` | mount position on `parent_prim` when no `frame` is set (legacy) |
| `local_orientation_rpy_deg` | `[180, 0, 0]` | mount orientation on `parent_prim` when no `frame` is set (legacy) |
| `width` | `848` | image width in pixels |
| `height` | `480` | image height in pixels |
| `fov_deg` | `90.5` | horizontal field of view |
| `depth` | `false` | attach the depth annotator (wrist cam: `true`) |
| `clip_near` | `0.05` | meters, near clip plane |
| `clip_far` | `10.0` | meters, far clip plane |
| `image_format` | `"png"` | color encoding for `GetImages` (`"png"` or `"jpeg"`) |
| `block_size_mm` | _unset_ | mock backend only: metric edge of the fabricated red block, unset keeps the fixed pixel-offset block |
| `view` | `"top"` | mock backend only: `"side"` fabricates a profile scene of the `blocks` list rising from the support line at the principal row |
| `blocks` | _unset_ | mock backend only, side view: list of `{rgb, size_mm, height_mm, column_offset_px, depth_m}` fabricated blocks |
| `frequency` | _rejected at validation_ | Isaac Sim's `Camera.set_frequency` validates the value against `/app/runLoops/main/rateLimitFrequency`, not this module's `rendering_dt`, and the render product renders every frame regardless of what is set |
| `parent_prim` | _unset_ | ride a link (e.g. a wrist camera) instead of spawning free-standing, see below |
| `annotator_device` | _unset_ | GPU-resident annotator data path, e.g. `"cuda"`, applied on Isaac Sim 5.0+, logged and ignored on 4.5 |
| `orientation_axes` | `"world"` | convention of `orientation_wxyz` for a free-standing camera, see below |

`parent_prim` requires a `frame` whose `parent` is the component that owns
that prim, see "Frames" above. The frame's translation and orientation become
the camera's local pose, applied in ROS-optical axes (camera +Z = frame +Z).
The legacy `local_position`/`local_orientation_rpy_deg` attributes only apply
when no `frame` is set.

`orientation_axes` is `"world"` (+X forward) for the legacy
`orientation_wxyz` attribute, and `"ros"` (+Z forward) when the quaternion
comes from a Viam `frame` orientation, which the module sets automatically.
Don't set it by hand.

**What the camera serves:** `GetImages` returns `[NamedImage("color", png|jpeg),
NamedImage("depth", image/vnd.viam.dep)]`, color first, honoring
`filter_source_names` (unknown names are dropped, so fewer images come back).
`GetPointCloud` (depth cameras only) returns binary `pointcloud/pcd`, in
meters, in the camera-optical frame (+X right, +Y down, +Z forward). Invalid
points are dropped. An encoded cloud over 32 MiB raises, the same
message-size guard the RealSense module applies. `GetProperties` reports `supports_pcd` (true iff `depth`
is set), real pinhole intrinsics (`fx = fy = W/(2·tan(hfov/2))`, e.g. 420.3 px
at 848x480 @ 90.5°), `mime_types`, and `frame_rate`. `DoCommand
{"command": "sample_color", "region": [x0, y0, x1, y1]}` returns
`{"srgb_hex": "#RRGGBB", "mean_rgb": [r, g, b]}`, useful for picking
`detect_color` for Viam's `color_detector`. While the renderer has not
produced a frame yet (immediately after create/reset), calls fail with gRPC
`FAILED_PRECONDITION`, so retry.

A wrist camera riding an arm link:

```json
{
  "name": "wrist-cam",
  "api": "rdk:component:camera",
  "model": "viam:isaac-sim-devin:camera",
  "frame": {
    "parent": "pick-arm",
    "translation": { "x": 0, "y": 0, "z": 60 },
    "orientation": { "type": "ov_degrees", "value": { "x": 0, "y": 0, "z": 1, "th": 0 } }
  },
  "attributes": {
    "world": "isaac-world",
    "parent_prim": "/World/pick_arm/wrist_3_link",
    "depth": true
  }
}
```

### base attributes

`world` (default `isaac-world`), `asset` (e.g. `jetbot`, which brings wheel defaults) or
`usd_path`/`prim_path` plus `wheel_joints: [left, right]`, `wheel_radius`,
`wheel_base`. `max_linear_mps` / `max_angular_rps` scale `SetPower`.

A real wheeled base's geometry carries over as `width_mm` and
`wheel_circumference_mm` (both positive numbers), the same units a real
`rdk:builtin:wheeled` config uses. When set, and `wheel_base` /
`wheel_radius` aren't also set, they derive `wheel_base` (`width_mm / 1000`)
and `wheel_radius` (`wheel_circumference_mm / 1000 / (2 * pi)`) before the
base is created, so the sim geometry and `GetProperties` follow the real
config instead of the asset's own defaults.

`MoveStraight` with a velocity nearly zero stops the base and returns, the
same no-op-stop semantics as rdk's wheeled base. `Spin` with an angle
nearly zero raises, and `Spin` with a velocity nearly zero stops the base
and returns, again matching the wheeled base. Both hold their velocity
until the computed duration has passed in sim time, not wall time, so a sim
stepping below real time still covers the commanded distance. A `timeout`
still cuts the wait on the wall clock.

### scene-finalizer

This model has no attributes. Its `depends_on` must name the world component
and every sim arm, gripper, camera and base in the machine, or `validate_config`
rejects the config. viam-server then builds it last, and building it (or
reconfiguring it) tells the world the scene is populated and it is safe to
start stepping, see `wait_for_finalizer` under "world attributes" above.
`tools/simulate_config.py` emits this component automatically, naming the
world plus every swapped sim component, and both `fragments/isaac-sim-block-sorting.json`
and `fragments/isaac-sim-world.json` carry one.

## Lifecycle (close and reconfigure)

Closing a component (`close()`) releases its handle and post-reset hooks.
The underlying prim stays in the stage (Kit cannot un-spawn it), so a later
`create_*` for the same name re-attaches to it.

Changing a **spawn** attribute on an already-attached component is rejected
with a `ValueError` that says to restart the module to apply it. Those
attributes are `asset`, `usd_path`, `prim_path`, `position`, `orientation`,
the frame pose, `parent_prim`, camera optics, and the like. The component is
not left silently running a stale prim.

**Runtime attributes** re-apply live without a restart: `world`,
`move_timeout_sec`, `max_linear_mps`, `max_angular_rps`, `arm`,
`tcp_offset_m`, `open_deg`, `closed_deg`, `grab_timeout_sec`,
`holding_tolerance_deg`, `mock_object_width_m`.

## Viewing the simulator

* **Through Viam (recommended)**: add an `viam:isaac-sim-devin:camera` component with
  `position` + `target` (see the example config) and watch it in the Viam app
  like any other camera. The control tab, data capture and the SDKs all work.
* **Full interactive viewport**: install NVIDIA's
  [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
  and connect it to the sim machine's IP. Use the plain IP with no port, since
  TCP 49100 and UDP 47998 are hardcoded and must be reachable. The module
  advertises the GCP metadata server's external IP when it can reach one, and
  falls back to the local interface address otherwise. Set
  `livestream_public_ip` on the world to override either.
* **Local GUI**: set `"headless": false` on the world (needs a display on the
  sim machine).

## The block-sorting cell

The `isaac-sim-block-sorting` fragment boots a three-table sorting cell: a
UR20 arm with a Robotiq gripper, three cameras, six vision-service pairs, and
an 18-block pool that scatters onto the source table. The `block-sorter`
conductor service sorts the whole scatter onto per-color pads over DoCommand,
single-shot, N loops, or continuous, with no script in the loop.

[`docs/BLOCK_SORTING.md`](docs/BLOCK_SORTING.md) is the cell's reference. It carries
the fragment variables, the conductor's and sorter-sensor's attributes, the
scatter and telemetry verbs, and the single-pick client.
[`getting-started.md`](getting-started.md) is the walkthrough that drives it.

## Simulation contract

Simulation is hardware. Each sim model in this module implements the Viam API
a real driver serves and stands in for that driver's model in a sim machine's
config, and a swap changes `model` and `attributes` only. The full contract is
[`docs/SIMULATION.md`](docs/SIMULATION.md), and
[`docs/PARITY.md`](docs/PARITY.md) is the ledger that tracks parity method by
method.

`tools/simulate_config.py` does the swap. It reads a real machine's config and
writes the config of its sim machine:

```sh
.venv/bin/python tools/simulate_config.py examples/configs/real-ur5e-cell.json \
  --out examples/configs/sim-ur5e-cell.json
```

It prints `swapped:`, `passed through:` and `placeholders:` lines on stderr,
naming the components in each group, so you can see what it did without
reading the whole output config.

## How it works

Isaac Sim's Python API only runs inside Isaac Sim's own interpreter, and
Omniverse Kit wants to own the thread it runs on. So:

* `run.sh` launches the module with Isaac Sim's bundled python (found via
  `$ISAAC_SIM_PATH/python.sh` or `$ISAAC_PYTHON`), installing `viam-sdk` into
  it on first run.
* The **main thread** runs the simulation loop (`SimulationApp` boot, stepping,
  and a task queue). The **Viam module server** runs on a side thread, and all
  component calls are marshalled onto the sim thread.
* All models live in one module process and share the sim through a singleton,
  so an arm, a camera or a base names its world component and gets attached.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) maps the source tree: which
package owns what, how a request travels, and where a new model or handle
goes.

