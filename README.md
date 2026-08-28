# viam-isaac-sim

A [Viam](https://www.viam.com) module for controlling and simulating robots in
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim).

What it is/does
====
* Lets you use Viam to control robots in NVIDIA Isaac Sim
* Models
  * Core simulator model (`world`) that configures the world and runs Isaac Sim
  * A model for each arm, camera, base or other component you want to control/simulate from Viam
* User does:
  * creates the core sim/world component in Viam
  * adds their components, e.g. an arm with `"asset": "ur20"`
  * the world model starts the sim and the component models spawn the right prims
  * views the simulator via the built-in WebRTC livestream or through Viam camera components
  * controls robots and sees cameras through the normal Viam APIs

## Models

| Model | Viam API | What it does |
|---|---|---|
| `dtcurrie:isaac-sim:world` | `generic` | Boots Isaac Sim, opens the USD stage, runs the sim loop. Configure exactly one. |
| `dtcurrie:isaac-sim:arm` | `arm` | Spawns (or attaches to) an articulation - UR arms, Franka, or any USD - and exposes joint control. |
| `dtcurrie:isaac-sim:camera` | `camera` | Creates (or attaches to) a camera prim and serves its RGB frames. |
| `dtcurrie:isaac-sim:base` | `base` | Spawns a differential-drive robot (e.g. jetbot) and drives it. |

Known assets (usable via the `asset` attribute): `ur3e`, `ur5e`, `ur10`,
`ur10e`, `ur16e`, `ur20`, `franka`, `jetbot`. Anything else can be loaded with
`usd_path`, or attach to prims already in your stage with `prim_path`.

## Example machine config

```json
{
  "components": [
    {
      "name": "sim-world",
      "model": "dtcurrie:isaac-sim:world",
      "type": "generic",
      "attributes": {
        "headless": true,
        "livestream": true
      }
    },
    {
      "name": "my-ur20",
      "model": "dtcurrie:isaac-sim:arm",
      "type": "arm",
      "frame": { "parent": "world" },
      "attributes": {
        "world": "sim-world",
        "asset": "ur20"
      }
    },
    {
      "name": "overhead-cam",
      "model": "dtcurrie:isaac-sim:camera",
      "type": "camera",
      "frame": {
        "parent": "world",
        "translation": { "x": 2000, "y": 2000, "z": 2000 }
      },
      "attributes": {
        "world": "sim-world",
        "target": [0, 0, 0.5],
        "width": 1280,
        "height": 720
      }
    },
    {
      "name": "my-jetbot",
      "model": "dtcurrie:isaac-sim:base",
      "type": "base",
      "frame": {
        "parent": "world",
        "translation": { "x": 1000, "y": 0, "z": 100 }
      },
      "attributes": {
        "world": "sim-world",
        "asset": "jetbot"
      }
    }
  ]
}
```

Every non-world component must set `"world"` to the world component's name.
That attribute is also returned as an implicit dependency from each model's
validate, so viam-server starts the world first - no `depends_on` needed.

Components are **placed with the standard frame config** (translations in mm,
any orientation representation) - the spawn pose in Isaac and viam's frame
system then agree, so things like the motion service see components where
they actually are. The `position` (meters) / `orientation_rpy_deg` attributes
still work as a fallback when no frame is set; a camera `target` attribute
overrides orientation to aim at a point.

**Frames:** a spawned component's `frame.parent` must be `"world"` - the
spawn path does not resolve an arbitrary frame chain, only the world's own
origin. The one exception is a component that also sets `parent_prim` (e.g.
a wrist camera riding an arm link): it may name that arm (or other)
component as its frame parent instead, since it is attaching to a prim
inside the sim rather than asking the spawn path to place it.

### world attributes

| attribute | default | notes |
|---|---|---|
| `mock` | `false` | run without Isaac Sim installed (development/testing) |
| `headless` | `true` | no local GUI window |
| `livestream` | `true` | WebRTC viewer at `http://<host>:8211/streaming/webrtc-client` |
| `livestream_public_ip` | _unset_ | IP advertised to streaming clients when the sim machine has multiple interfaces |
| `usd_stage` | _empty stage + ground plane_ | USD file or omniverse:// URL to open |
| `physics_dt` / `rendering_dt` | `1/60` | step sizes in seconds |
| `boot_timeout_sec` | `300` | Isaac Sim can take a while on first boot |
| `kit_log_level` | `"warning"` | kit console verbosity |
| `props` | `[]` | objects spawned into the scene at boot; see below |

Each entry in `props` is an object: `name` (string, snake_cased for the prim
path), `type` (`"cube"` or `"usd"`), `position` ([x,y,z] meters, the prop's
**centre**), `size` (meters, the cube's base edge length, > 0), `scale`
([sx,sy,sz], multiplies `size` per axis), `color` ([r,g,b] each in `[0, 1]`),
`fixed` (bool - static vs. dynamic/physics-driven), and `usd_path` (required
when `type` is `"usd"`).

`props` validation rules (`ValueError` on config, surfaced as
`INVALID_ARGUMENT`):
* names must be unique once snake_cased (the same normalisation used for prim
  paths)
* `type` must be `"cube"` or `"usd"`
* `usd_path` is required when `type` is `"usd"`
* `position`, `scale`, and `color` must each be 3-number sequences
* `color` values must be in `[0, 1]`
* `size` must be a positive number

The world also supports `DoCommand`: `{"command": "status" | "play" | "pause" |
"reset"}` and `{"command": "add_usd", "usd_path": "...", "prim_path":
"/World/thing", "position": [x, y, z]}` to drop extra props into the scene.

### Units and conventions

Viam's frame system (component `frame` config, `GetEndPosition`, camera
`target`, etc.) uses **millimetres and degrees**, per the standard Viam
convention. The world's `props` attribute, by contrast, is Isaac-native:
**metres**, and `position` is always the prop's **centre**, not a corner.
For a cube prop the rendered extent along each axis is `size × scale[axis]`
(so `size` is a base edge length and `scale` stretches it per axis). Isaac
Sim is Z-up, matching Viam's frame convention.

Worked example - a table as a `fixed` cube prop:

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
centred at `position`, so each face sits `size * scale[axis] / 2` from the
centre along that axis). Anything you place *on* the table - a block, a
mount frame - belongs at `z_top + <that thing's own half-height>`, e.g. a
block of `size` 0.05 sits with its centre at `z_top + 0.025`.

### arm attributes

`world` (required), one of `asset` / `usd_path` / `prim_path`, plus optional
`position` ([x,y,z] meters), `end_effector_prim` (prim path whose pose is
reported by `GetEndPosition`, converted to Viam's orientation-vector
convention; defaults to `<arm prim>/wrist_3_link` for UR assets), and
`move_timeout_sec`.

**`GetEndPosition` reports the end effector's pose in the arm base frame**
(a breaking change this release - it previously reported world frame). This
matches how a real arm driver reports its end position, and lets Viam's
frame system (via the component's `frame` config) compose it into world
frame itself.

`MoveToJointPositions` / `GetJointPositions` work today. IK and motion
planning are deliberately left to Viam (the motion service), not Isaac - the
module's job is just to expose the simulated arm. `MoveToPosition` is
**unimplemented by decision** and returns `UNIMPLEMENTED` - use the motion
service instead.

`GetGeometries` returns `[]` by decision: rdk derives arm link geometry from
`GetKinematics` (the SVA already carries the link capsules) and never calls
`Geometries` for an arm that serves kinematics.

`GetKinematics` works: for `ur3e`/`ur5e`/`ur20` the official viam SVA
kinematics files are fetched automatically (and cached in the module data
dir); for anything else set `kinematics_url` to an SVA `.json` or `.urdf`
(http(s):// or file://). With kinematics served, the motion service can plan
for the simulated arm.

UR assets (`ur3e`/`ur5e`/`ur10`/`ur10e`/`ur16e`/`ur20`) get a built-in
**base-frame correction** applied at spawn so the arm's frame in Isaac lines
up with the kinematics Viam's motion service uses - without it the sim and
Viam's idea of the arm's pose would silently disagree.

### camera attributes

`world` (required), and either `prim_path` of an existing camera in your stage
or `position` plus `target` (aim-at point) or `orientation_rpy_deg` to create
one. `width`/`height` default to 640x480.

### base attributes

`world` (required), `asset` (e.g. `jetbot`, which brings wheel defaults) or
`usd_path`/`prim_path` plus `wheel_joints: [left, right]`, `wheel_radius`,
`wheel_base`. `max_linear_mps` / `max_angular_rps` scale `SetPower`.

## Pick-and-place fragment

The `isaac-sim-pick-and-place` fragment (source in
`fragments/pick-and-place.json`) is a ready-made scene: a UR20 (`pick-arm`)
at the origin, a red 6cm cube to pick up, a flat blue pad to place it on, and
a `scene-cam` watching the workspace. Add the fragment to any machine that
meets the requirements above and the world spawns everything at boot.

Props are configured on the world with the `props` attribute (cubes or USD
references, fixed or dynamic) - see the fragment for the shape of it.

The `isaac-sim-pick-and-place` fragment in the registry is the original
upstream public one; this fork's fragment ships with P5 - until then use
`viam module reload-local` (local module) with the JSON in
`fragments/pick-and-place.json`.

### Table recipe

A table is just a `fixed` cube prop on the world (see "Units and
conventions" above for how `size`/`scale`/`position` work):

```json
{
  "props": [
    {"name": "table", "type": "cube", "fixed": true, "size": 1.0,
     "scale": [1.2, 0.8, 0.75], "position": [0.60, 0.00, 0.375]}
  ]
}
```

`props` are visual/physical geometry only - the motion service doesn't see
them automatically. To make the table an obstacle the motion service plans
around, also add its box as `frame.geometry` on the world component itself,
in millimetres and centred on the frame origin (here 1200 × 800 × 740 mm at
translation (600, 0, 370) - 10 mm below the real 750 mm surface so the arm
isn't blocked from resting on it):

```json
{
  "name": "sim-world",
  "model": "dtcurrie:isaac-sim:world",
  "type": "generic",
  "frame": {
    "parent": "world",
    "translation": { "x": 600, "y": 0, "z": 370 },
    "geometry": { "type": "box", "x": 1200, "y": 800, "z": 740 }
  },
  "attributes": { "props": [ /* ... */ ] }
}
```

### Arm mount recipe

Mount an arm on the table by frame-placing it at the table's top height,
inset from the edge:

```json
{
  "name": "pick-arm",
  "model": "dtcurrie:isaac-sim:arm",
  "type": "arm",
  "frame": { "parent": "world", "translation": { "x": 150, "y": -250, "z": 750 } },
  "attributes": { "world": "sim-world", "asset": "ur5e" }
}
```

The Isaac articulation root is a fixed joint to the world - no mount joint
needs authoring. Keep the base at least 70 mm inside the table's edge so its
collider clears the table. UR assets carry a built-in base-frame correction
so this frame placement and Viam's kinematics agree with the simulated pose
(see "arm attributes" above).

## Viewing the simulator

* **Through Viam (recommended)**: add an `dtcurrie:isaac-sim:camera` component with
  `position` + `target` (see the example config) and watch it in the Viam app
  like any other camera - control tab, data capture, SDKs, everything works.
* **Full interactive viewport**: install NVIDIA's
  [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
  and connect it to the sim machine's IP (plain IP, no port - TCP 49100 and
  UDP 47998 are hardcoded and must be reachable). If the machine has multiple
  interfaces, set `livestream_public_ip` on the world.
* **Local GUI**: set `"headless": false` on the world (needs a display on the
  sim machine).

## How it works

Isaac Sim's Python API only runs inside Isaac Sim's own interpreter, and
Omniverse Kit wants to own the thread it runs on. So:

* `run.sh` launches the module with Isaac Sim's bundled python (found via
  `$ISAAC_SIM_PATH/python.sh` or `$ISAAC_PYTHON`), installing `viam-sdk` into
  it on first run.
* The **main thread** runs the simulation loop (`SimulationApp` boot, stepping,
  and a task queue). The **Viam module server** runs on a side thread; all
  component calls are marshalled onto the sim thread.
* All models live in one module process and share the sim through a singleton,
  so arms/cameras/bases just name their world component and get attached.

## Machine requirements & automatic setup

On a standard Ubuntu 22.04/24.04 x86_64 machine, the module sets itself up:
when first installed, viam-server runs `first_run.sh`, which installs the
system libraries kit needs (vulkan/GL), the right python (via deadsnakes on
24.04), an NVIDIA driver if none is present (the validated 580 branch - newer
is not better here, see below), and Isaac Sim itself
(pip-installed into a venv under the module's data directory - 4.5.0 on
22.04, 5.0.0 on 24.04). `run.sh` finds that install automatically; the EULA
is accepted via environment variable.

Notes on the automatic setup:

* The Isaac Sim download is 10GB+. If it exceeds viam-server's default
  first-run timeout, set `"first_run_timeout": "2h0m0s"` on the module entry
  in your machine config.
* If the script had to install the NVIDIA driver, **reboot** before
  configuring the world component.
* Already have Isaac Sim? Set `ISAAC_SIM_PATH` (dir containing `python.sh`)
  or `ISAAC_PYTHON` in the module's environment variables and the script
  skips everything.

What the machine must already be/have (the script can't do these for you):

* Ubuntu 22.04 or 24.04 on x86_64 with an RTX-capable NVIDIA GPU (8GB+ VRAM
  minimum, RTX 4080+/L40 recommended), 32GB+ RAM, ~60GB free disk. See
  NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
* An NVIDIA driver from a branch Isaac Sim validates against - **use the 580
  branch**. Newer branches (590/595+) are known to crash Isaac's RTX renderer
  on startup (`librtx.scenedb.plugin.so`) and break CUDA init
  (`cuDeviceGetUuid` Warp errors); see
  [isaac-sim/IsaacSim#537](https://github.com/isaac-sim/IsaacSim/issues/537).
  If you're on 595+: `sudo apt-get install -y nvidia-driver-580 && sudo reboot`.
* viam-server installed, machine online in the Viam app, running as root (or
  a user with passwordless sudo) for the apt/driver steps.
* Network access to pypi.nvidia.com, pypi.org, and NVIDIA's asset servers.
* Open ports for the livestream viewer if you want it: TCP 49100 (signaling)
  plus UDP 47998 (media) - both hardcoded in NVIDIA's streaming client.

No GPU/Isaac at all? `"mock": true` on the world runs the module anywhere for
development.

## Development without Isaac Sim (mock mode)

Set `"mock": true` on the world component and the module runs anywhere python
does - arms integrate joint targets over time, cameras produce synthetic
frames, bases accept velocity commands. This is what the test suite uses.

Set up a dev venv and run the checks with `make`:

```sh
uv venv --python 3.11 .venv && uv pip install -r requirements-dev.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make fmt-check lint typecheck test
```

Tests run under the `pyproject.toml` config (`pythonpath = ["src"]`); the
`gpu` marker is skipped by default, so `make test` only runs mock-mode tests
and needs no Isaac Sim install. CI runs the same suite on both Python 3.10
and 3.11.

## Status / roadmap

- [x] world boot, stage loading, livestream, play/pause/reset, add_usd
- [x] arm joint control (UR family, Franka, arbitrary USD articulations)
- [x] RGB cameras
- [x] differential-drive bases
- [x] cloud builds / registry publishing (tag a release)
- [x] serve kinematics files (`GetKinematics`) so Viam's motion service can do
      IK and planning for simulated arms (all motion stays in Viam, not Isaac)
- [ ] depth / point clouds from cameras
- [ ] gripper support
