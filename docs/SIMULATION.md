# Simulation contract

A user has a machine configured with the real drivers for hardware they do not have yet: a UR arm
on `viam:universal-robots:ur5e`, a RealSense camera on `viam:camera:realsense`, motion and vision
services running on top. They want to run that exact config, unedited, against a simulator, on a
GPU machine created for the purpose. This document states the contract that machine keeps with
everything above it: viam-server, motion, vision, the frame system, data management, and every
client script. Read it before you add a resource to this module, or before you build a tool that
reads or writes `simulates.json`.

## The pattern

Simulation is hardware. This module reaches that goal two ways.

Substitution: this module's models (arm, camera, gripper, base) implement the same Viam component
API a real driver serves, backed by Isaac physics and rendering, and stand in for the real driver's
model in a derived config (see "The switch"). It works for every asset this module ships and needs no
driver to change. Parity is its bar: a client cannot tell a sim resource from a real one
by behavior. [`docs/PARITY.md`](PARITY.md) tracks that method by method and is the ledger. One
documented deviation stands. The arm API defines `MoveToPosition` as a straight line in
Cartesian space, and the sim solves IK and drives a joint-space path between the same two
endpoints.

Integration: a real driver accepts a `world` attribute and sends its own commands to the world
service over a simulation API instead of to hardware, reusing all of its own logic. That API does
not exist yet. What keeps the door open: the world component is the one resource that knows it is
a sim (see "The sim-only resources"), the models reach it only through the handle interface in
`src/isaac_module/handles/`, and that interface is the surface a simulation API would be cut from,
with these models as its first clients.

Everything above the component API, motion, vision, the frame system, data management, clients,
the app, stays untouched under either pattern. It answers gRPC calls with the same shapes whether
the resource underneath is real or simulated, and never learns which.

## The switch

The sim machine is a derived machine, not an edited one. A resolver reads the user's real config
and writes the config of a second machine: every hardware resource whose model has a row in
`simulates.json` swapped to that row's sim model, with `world` set to `isaac-world`, the row's
attribute template applied, and the attributes the row's `carry` map names copied over from the
real entry. The module entry and an `isaac-world` component are added. `name`, `api`, `frame`,
`depends_on`, and every service stay byte-identical, so a client script pointed at the sim machine
runs unchanged. The real config is never touched.

The example below is the real `pick-arm` entry from a user's config, beside the entry the
resolver writes for it in the sim machine's config.

```json
{
  "name": "pick-arm",
  "api": "rdk:component:arm",
  "model": "viam:universal-robots:ur20",
  "frame": { "parent": "world", "translation": { "x": 0, "y": 0, "z": 750 } },
  "attributes": { "host": "10.1.10.84" }
}
```

```json
{
  "name": "pick-arm",
  "api": "rdk:component:arm",
  "model": "viam:isaac-sim-devin:arm",
  "frame": { "parent": "world", "translation": { "x": 0, "y": 0, "z": 750 } },
  "attributes": { "world": "isaac-world", "asset": "ur20" }
}
```

The resolver writes `world` explicitly even though `isaac-world` is every sim model's default, so
a resolved config never depends on a default a later reader might not know.

## Source of truth

Isaac is to this simulated machine what the physical world is to a real one. Every read goes
through to the sim at the moment of the call: joint positions, end pose, images, point clouds, and
prop geometries all come from Isaac, never from a value Viam cached on an earlier call. Every
write is a command that Isaac's physics executes, and the driver reports settle, stall, and
timeout the way a real controller would, not the way a script that assumes success would.

Frame config is the single source of spawn poses. A resource's `frame` in the machine config is
where it appears in the scene, and the world's `GetGeometries` serves live prop poses, so there is
nothing to keep in sync. One copy of the state exists, in the sim, and the driver is the only thing
that touches it.

Source of truth is not the same as oracle access. The world component exposes reads and authoring
verbs, `prop_geometries`, `set_prop_pose`, `scatter_cell`, that a test, a planner obstacle, or a
scenario setup script can use. Those calls are for building and asserting on a scenario, never for
the control path a real machine would also run. A pipeline that reads a block's pose from the
world instead of from the camera has stopped predicting what the real machine would do, because
the real machine has no such shortcut. The pick pipeline already keeps this rule: it uses vision to
find a block and a point cloud to measure it, the same as it would against a camera bolted to a
real arm. That is the rule for every pipeline.

## The sim-only resources

The `world` component is a resource in this module with no real-world counterpart, and the
only thing a fragment adds beyond components that could also point at real hardware. It owns
booting Isaac Sim, opening the USD stage, holding props and the scene, serving scenario verbs, and
running the livestream.

Nothing else in this module ever grows a verb a real component could not have. A sim arm answers
the same gRPC calls a real arm answers and nothing more. `scatter_cell` and `clear_cell` on the
world are scene verbs that take their prop names, region and park grid from the command, so they
serve any scene. They set no precedent for adding scenario logic to a component that represents a
piece of hardware.

The `scene-finalizer` component is the second resource with no real counterpart. A module cannot
see the machine's full component list, so it has no way to know on its own when every arm,
gripper, camera and base it will drive has finished spawning. The config tells it instead: the
finalizer's `depends_on` names the world and every sim component in the scene, viam-server builds
it last, and building it is the signal that the scene is populated. Until that happens, and for a
few steps after, every sim resource answers `UNAVAILABLE` with `Isaac Sim is initializing; retry
shortly`. This is part of the contract, not a startup wrinkle a client works around: the status
clients and the app already treat `UNAVAILABLE` as transient and retry, and a resource's `stop`
verb still works while the world initializes, so an operator can always stop a machine. See
`README.md`'s "world attributes" and "scene-finalizer" sections for the config shape.

## Granularity

Machine-level simulation is the target: every hardware resource on a machine swapped, running on
the machine created for it, because the module process runs wherever the world it drives runs
(`src/main.py`'s docstring: Isaac Sim owns the process's main thread). The sim machine created by
the resolver's output is, for that reason, already the sim host.

Per-resource simulation, hardware in the loop, applies the same resolver to a subset of a
machine's resources with the sim machine standing in as a remote of the real machine, so
co-location still holds for whichever resources move there. It stays possible under this contract.

Integration is what removes the co-location constraint for drivers, once a driver reaches the
world over the simulation API instead of running inside the module that hosts the world.

The default is machine-level simulation, on a machine created for it.

## Substitution table

The table below mirrors `simulates.json` at the repo root, row for row, in the file's order. It is
the human-readable form of the same lookup: for a given real model and API, which sim model
replaces it, what attribute template the resolver builds, and which real attributes it carries
over unchanged.

| Real model | API | Sim model | Template | Carry | Verified |
|---|---|---|---|---|---|
| `viam:universal-robots:ur3e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur3e"}` | `speed_degs_per_sec` → `max_vel_degs_per_sec` | yes |
| `viam:universal-robots:ur5e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur5e"}` | `speed_degs_per_sec` → `max_vel_degs_per_sec` | yes |
| `viam:universal-robots:ur7e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur7e"}` | `speed_degs_per_sec` → `max_vel_degs_per_sec` | yes |
| `viam:universal-robots:ur20` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur20"}` | `speed_degs_per_sec` → `max_vel_degs_per_sec` | yes |
| `viam:franka:panda` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "franka"}` | `{}` | no |
| `viam:robotiq:2f-grippers` | `rdk:component:gripper` | `viam:isaac-sim-devin:gripper` | `{"asset": "robotiq_2f_85", "arm": "$frame.parent"}` | `{}` | yes |
| `viam:robotiq:epick` | `rdk:component:gripper` | `viam:isaac-sim-devin:vacuum` | `{"arm": "$frame.parent"}` | `{}` | yes |
| `rdk:builtin:wheeled` | `rdk:component:base` | `viam:isaac-sim-devin:base` | `{"asset": "jetbot"}` | `width_mm` → `width_mm`, `wheel_circumference_mm` → `wheel_circumference_mm` | yes |
| `viam:camera:realsense` | `rdk:component:camera` | `viam:isaac-sim-devin:camera` | `{}` | `width_px` → `width`, `height_px` → `height` | yes |
| `*` (catch-all) | `*` | `rdk:builtin:fake` | `{}` | `{}` | yes |

The module ships as `viam:isaac-sim-devin`, private in the `viam` org. The `-devin` suffix marks
experimental code not yet intended for public use, and `simulates.json` carries that id. Dropping
the suffix is a separate decision, not part of this contract.

A row's `verified: false` means the real model string follows its module's naming pattern but was
not confirmed against that module's published manifest or model list on the date recorded in
`simulates.json`. Ship it, but flag it, rather than guessing silently. A template value starting
with `$` copies a field off the real resource's own config instead of a literal. The gripper row's
`"arm": "$frame.parent"` is the only one: the real gripper entry carries which arm it is
bolted to only as its frame's parent, so the resolver reads that value from the real entry and
writes it into the sim gripper's attributes.

`carry` is a flat copy from a real attribute name to the sim attribute it becomes, never a
transform. The camera row carries a RealSense's `width_px` and `height_px` unchanged into the sim
camera's `width` and `height`. The four UR rows carry `speed_degs_per_sec` into the sim arm's
`max_vel_degs_per_sec`, so a real speed limit tuned in degrees per second still applies after the
swap. The wheeled row carries `width_mm` and `wheel_circumference_mm` into the sim base's
attributes of the same names, which derive the sim's wheel geometry in meters, since a flat copy
cannot do the mm-to-m conversion itself. A carry only fires when the real entry has the attribute,
and every other row's `carry` is empty because nothing on that real driver needs to survive the
swap this way.

Every row names one asset, and that name is what the resolver swaps in for substitution.

## Scope

Producing a sim machine from a real config is a tool in this repo, `tools/simulate_config.py`, not
something the module runs on its own. An app-side flow or a viam-server flag could reproduce what
the resolver does. The simulation API protos that "The pattern" describes for integration, each
driver's work to adopt them, GPU host provisioning beyond `provisioning/`, and the network transport
a remote-sim path would need are all outside this module.

Four things here are stable for that other work to rely on. The model strings
`viam:isaac-sim-devin:*`. `simulates.json`, the machine-readable substitution table, with
[`PARITY.md`](PARITY.md) tracking API parity method by method. The `isaac-sim-world-devin` fragment,
the one place a config gains `isaac-world`. And the resolver, the reference behavior for how a real
config becomes a sim machine's config.

## Row notes

**`viam:franka:panda` is flagged `verified: false` for a capability gap.** The model triple is
confirmed against `viam-modules/viam-franka-arm`'s `meta.json`, but that module ships no kinematics
file the `franka` asset could carry, so a resolved Franka arm cannot answer `GetKinematics`,
`GetEndPosition` or `MoveToPosition`. The row ships anyway, flagged, rather than dropping simulation
of a supported arm.

**`ur7e` has no Isaac Sim 5.0 USD of its own.** The assets root's `UniversalRobots` folder lists
`ur10`, `ur10e`, `ur16e`, `ur20`, `ur3`, `ur30`, `ur3e`, `ur5` and `ur5e`. The `ur7e` asset spawns
the `ur5e` mesh as a geometry stand-in, paired with the real `ur7e` kinematics from
`viam-modules/universal-robots`, so `MoveToPosition` and joint limits are the `ur7e`'s even though
the arm looks like a `ur5e`.

**A wheeled base's motors have no sim row.** The `wheeled` row swaps the base, but a jetbot is a
wheeled base built over two motors, and no motor row exists. The motors become placeholder
`rdk:builtin:fake` motors through the catch-all row, visible in the config as the thing to remove
or replace.

**The catch-all row turns unknown hardware into a placeholder.** A hardware component with no row
keeps its own api and is pointed at `rdk:builtin:fake`, keeping its name, frame and `depends_on`
and carrying no attributes, so the sim machine's config loads and the gap is visible in the app.
`pose_tracker` is the one api that changes, to `rdk:component:generic`, since no fake
`pose_tracker` model exists. `--allow-unmatched` keeps the real entry instead, for anyone who wants
the failure to show at construction. A real driver on `rdk:component:generic` is not treated as
hardware, because the world component itself is generic: it keeps its real model, and a user who
wants it simulated writes that row by hand.

**The RealSense row keeps the frame and resolution and drops the real optics.** The sim camera
keeps the real camera's frame and carried resolution, and a `depth` sensor in the RealSense
`sensors` list (or an absent list, the driver's default) becomes `depth: true`, but the lens takes
the sim model's defaults, so the field of view differs.

**`$frame.parent` is the only template reference form.** It serves the one row that needs it, the
gripper's arm, and a second reference form waits for a second template that needs one.

## Trying it

Run the resolver on the committed example with
`.venv/bin/python tools/simulate_config.py examples/configs/real-ur5e-cell.json --out examples/configs/sim-ur5e-cell.json`.
The UR arm, RealSense camera and Robotiq gripper entries swap to their sim models, `world` and the
module entry appear, and the carried camera resolution lands under the sim names. Every other
entry, including `builtin` motion and the `color_detector` vision service, comes out byte-identical
to the input, because the resolver never touches services or the fields "The switch" names.

`examples/arm_camera_smoke.py` exercises exactly the resources the resolver swaps. It reads joint
positions and end position off the arm, pulls images off the camera, drives one motion `Move`, and
opens and grabs with the gripper, then prints a JSON report. It names no world and no sim verb, so
the same script, unchanged, is the acceptance artifact the real machine runs once the hardware
arrives.

Turning the resolved config into a running machine is the sim machine image and create-sim-machine
flow: it takes this tool's output and boots the machine the smoke script above connects to.
