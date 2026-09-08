# Simulation contract

A user has a machine configured with the real drivers for hardware they do not have yet: a UR arm
on `viam:universal-robots:ur5e`, a RealSense camera on `viam:camera:realsense`, motion and vision
services running on top. They want to run that exact config, unedited, against a simulator, on a
GPU machine created for the purpose. This document states the contract that machine keeps with
everything above it: viam-server, motion, vision, the frame system, data management, and every
client script. Read it before you add a resource to this module, or before you build a tool that
reads or writes `simulates.json`.

## The pattern

Simulation is hardware, and this module reaches that goal two ways. One ships now.

Substitution: this module's models (arm, camera, gripper, base) implement the same Viam component
API a real driver serves, backed by Isaac physics and rendering, and stand in for the real driver's
model in a derived config (see "The switch"). It works today for every asset this module ships and
needs no driver to change. Parity is its bar: a client cannot tell a sim resource from a real one
by behavior. [`docs/PARITY.md`](PARITY.md) tracks that method by method and is the ledger.

Integration: a real driver accepts a `world` attribute and sends its own commands to the world
service over a simulation API instead of to hardware, reusing all of its own logic. That API does
not exist yet, and no driver in this repo builds it. What this plan keeps ready for it: the world
component is the one resource that knows it is a sim (see "The one sim-only resource"), our models
reach it only through the handle interface in `sim_manager.py` (the `create_arm`, `create_camera`,
and similar methods), and that interface is the surface the simulation API gets cut from, with our
models as its first clients.

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

An earlier draft of this contract added a per-resource field that named which world a resource
should run against and resolved it in place, on the same machine. That field is dropped, because
the sim machine is a separate machine, produced whole by the resolver rather than edited resource
by resource.

The example below is the real `pick-arm` entry a user's config carries today, beside the entry the
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
real arm. This contract makes that a rule for every future pipeline, not just an accident of how
the pick pipeline happens to be written.

If twin mode is ever built, it is the one inversion of this rule: the real hardware becomes the
truth and the sim follows it. See "Twin mode" below.

## The one sim-only resource

The `world` component is the only resource in this module with no real-world counterpart, and the
only thing a fragment adds beyond components that could also point at real hardware. It owns
booting Isaac Sim, opening the USD stage, holding props and the scene, serving scenario verbs, and
running the livestream.

Nothing else in this module ever grows a verb a real component could not have. A sim arm answers
the same gRPC calls a real arm answers and nothing more. `scatter_cell` and `clear_cell` on the
world are scenario verbs scoped to the shipped block-sorting cell, not general scene tools, and
they are recorded debt: they will move out of the world component or generalize later, but they do
not set a precedent for adding scenario logic to a component that represents a piece of hardware.

## Granularity

Machine-level simulation is the target: every hardware resource on a machine swapped, running on
the machine created for it, because the module process runs wherever the world it drives runs
(`src/main.py`'s docstring: Isaac Sim owns the process's main thread). The sim machine created by
the resolver's output is, for that reason, already the sim host.

Per-resource simulation, hardware in the loop, applies the same resolver to a subset of a
machine's resources with the sim machine standing in as a remote of the real machine, so
co-location still holds for whichever resources move there. It stays possible under this contract.
This plan does not build it.

Integration is what removes the co-location constraint for drivers, once a driver reaches the
world over the simulation API instead of running inside the module that hosts the world.

Recommendation: machine-level simulation, on a machine created for it.

## Twin mode (deferred)

A sim component that mirrors a real resource which keeps running, so a user can compare real
hardware and its simulated twin side by side, is a known future ask. It is not in this plan. The
design and the constraints this contract keeps open for it live in
[deferred-twin-mode.md](../.claude/plans/sim-platform/deferred-twin-mode.md): a sim component may
depend on another component of the same API by name, the same asset table that serves
substitution also serves a twin, and the `world` pointer stays a plain component name in both
modes.

## Substitution table

The table below mirrors `simulates.json` at the repo root, row for row, in the file's order. It is
the human-readable form of the same lookup: for a given real model and API, which sim model
replaces it, what attribute template the resolver builds, and which real attributes it carries
over unchanged.

| Real model | API | Sim model | Template | Carry | Verified |
|---|---|---|---|---|---|
| `viam:universal-robots:ur3e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur3e"}` | `{}` | yes |
| `viam:universal-robots:ur5e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur5e"}` | `{}` | yes |
| `viam:universal-robots:ur10` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur10"}` | `{}` | no |
| `viam:universal-robots:ur10e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur10e"}` | `{}` | no |
| `viam:universal-robots:ur16e` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur16e"}` | `{}` | no |
| `viam:universal-robots:ur20` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "ur20"}` | `{}` | yes |
| `viam:franka:panda` | `rdk:component:arm` | `viam:isaac-sim-devin:arm` | `{"asset": "franka"}` | `{}` | yes |
| `viam:robotiq:2f-grippers` | `rdk:component:gripper` | `viam:isaac-sim-devin:gripper` | `{"asset": "robotiq_2f_85", "arm": "$frame.parent"}` | `{}` | yes |
| `rdk:builtin:wheeled` | `rdk:component:base` | `viam:isaac-sim-devin:base` | `{"asset": "jetbot"}` | `{}` | yes |
| `viam:camera:realsense` | `rdk:component:camera` | `viam:isaac-sim-devin:camera` | `{}` | `width_px` → `width`, `height_px` → `height` | yes |
| `*` (catch-all) | `*` | `rdk:builtin:fake` | `{}` | `{}` | yes |

The module ships as `viam:isaac-sim-devin`, private in the `viam` org. The `-devin` suffix marks
experimental code not yet intended for public use, and `simulates.json` carries that id. Dropping
the suffix is a separate decision, not part of this contract.

A row's `verified: false` means the real model string follows its module's naming pattern but was
not confirmed against that module's published manifest or model list on the date recorded in
`simulates.json`. Ship it, but flag it, rather than guessing silently. A template value starting
with `$` copies a field off the real resource's own config instead of a literal. The gripper row's
`"arm": "$frame.parent"` is the only one today: the real gripper entry carries which arm it is
bolted to only as its frame's parent, so the resolver reads that value from the real entry and
writes it into the sim gripper's attributes.

`carry` is a flat copy from a real attribute name to the sim attribute it becomes, never a
transform. The camera row is the one row that uses it: a RealSense's `width_px` and `height_px`
carry unchanged into the sim camera's `width` and `height`. A carry only fires when the real entry
has the attribute, and every other row's `carry` is empty because nothing on that real driver needs
to survive the swap this way.

Every row serves two futures with one asset name. Today it tells the resolver what to swap in for
substitution. If twin mode is built, the same asset tells a twin component what to spawn to mirror
the real resource, so a row never needs to be forked between the two modes.

## Owned elsewhere

Honoring a real config by producing its sim twin is not this module's job to run automatically. It
can be an app-side button, or a viam-server flag, that reproduces what the resolver in this repo
does, and this contract does not decide which. Also owned elsewhere: the simulation API protos
that "The pattern" describes for integration, and every driver's work to adopt them, provisioning
the GPU host beyond what this repo's create-sim-machine script covers, and the network transport a
remote-sim path would need.

What that other work can rely on from this module: the model strings `viam:isaac-sim-devin:*` for
as long as the module carries the suffix, `simulates.json` as the machine-readable substitution
table, API parity tracked in `docs/PARITY.md` once a later phase
writes it, the `isaac-sim-world-devin` fragment as the one place a config gains `isaac-world`, and the
resolver in this repo as the reference behavior for how a real config becomes a sim machine's
config.

## Open questions

**Where the simulation API protos for integration will live.** Default: drafted from the handle
interface in `sim_manager.py` in this repo, once the first driver wants to integrate rather than
substitute.

**How two worlds on one machine are named.** Default: one world component per module process.
Another simulator's module names its own world component under its own convention, and
`isaac-world` is this one's.

**The `ur10`, `ur10e`, and `ur16e` real model strings are unverified.** They follow the naming
pattern of `viam-modules/universal-robots`, but that module's manifest lists only `ur3e`, `ur5e`,
`ur7e`, and `ur20`. Default: ship the rows anyway, flagged `verified: false` in `simulates.json`,
rather than omitting arms this module can otherwise simulate.

**A wheeled base's motors have no sim row.** `simulates.json`'s `wheeled` row swaps the base, but a
jetbot is a wheeled base built over two motors, and no motor row exists to swap alongside it.
Default: the motors become placeholder generic components through the catch-all row, visible in the
config as the thing to remove or replace with `fake` motors.

**The catch-all row turns unknown hardware into a placeholder.** A hardware component with no row
becomes a generic component on `rdk:builtin:fake`, keeping its name, frame and `depends_on` and
carrying no attributes, so the sim machine's config loads and the gap is visible in the app rather
than failing the resolver. It is the one row where the API changes. Default: placeholder, with
`--allow-unmatched` keeping the real entry for anyone who wants the failure to show at
construction instead.

**The RealSense row keeps the frame and resolution and drops the real optics.** The resolved sim
camera keeps the real camera's frame and carried resolution so it appears in the same place at the
same size, and a `depth` sensor in the RealSense `sensors` list becomes `depth: true`, but the lens
takes the sim model's defaults. Default: sim optics, correct placement and resolution, mismatched
field of view.

**The `$frame.parent` reference syntax in templates is the only reference form today.** It is
enough for the one row that needs it, the gripper's arm. Default: keep it as the only reference
form until a second template needs a different field, rather than designing a general reference
syntax against one use.

**Whether the conductor service also defaults its world attribute to `isaac-world`.** The
conductor is not a Viam component API, it is this module's own orchestration service, but it
points at a world the same way every other model does. Default: yes, for symmetry with every other
model's default.

An earlier draft of this contract left open whether the sim machine and the real machine were the
same machine or two separate ones. That question is settled: the sim machine is a second machine,
produced by the resolver and never the machine the real config was written for.

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
