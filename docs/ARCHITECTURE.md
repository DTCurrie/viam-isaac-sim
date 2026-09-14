# Architecture

This module lets a Viam machine drive robots inside NVIDIA Isaac Sim. A `world` component boots the
simulator and opens a USD stage. `arm`, `camera`, `gripper` and `base` components spawn prims in that
stage and answer the ordinary Viam component APIs against them, so a client cannot tell a simulated
arm from a real one. The design goal is stated in [SIMULATION.md](SIMULATION.md): take a machine
config written for real hardware, run it unedited against a simulator, and change nothing above the
component API.

This document describes the code as it is, then the extension points a second sim example plugs
into. Read it before you add a model or move a file.

## The module map

`src/` is 14,476 lines across 49 Python files in two packages, plus the entry point.

| path | lines | what it owns |
| --- | --- | --- |
| `src/main.py` | 114 | The process layout. Isaac Sim gets the main thread, the Viam module gRPC server gets a daemon thread with a fixed 8-worker executor. Signals stop the module before the sim. |
| `src/isaac_module/sim_manager.py` | 2192 | `SimConfig` and `SimManager`: the Kit lifecycle, the boot sequence, the sim thread and its task queue, the component factories and the handle cache. |
| `src/isaac_module/handles/world.py` | 677 | `WorldHandle`, its Isaac backend and its mock: scene verbs, prop poses, geometries. |
| `src/isaac_module/handles/arm.py` | 751 | `ArmHandle`, Isaac and mock: joint drive, settle detection, the velocity cap, prim poses. |
| `src/isaac_module/handles/gripper.py` | 536 | `GripperHandle`, Isaac and mock: jaw drive, stall and hold detection, the attach sequence. |
| `src/isaac_module/handles/camera.py` | 475 | `CameraHandle`, Isaac and mock: frames, intrinsics, the warm-up retry, the deterministic mock scene. |
| `src/isaac_module/handles/base.py` | 151 | `BaseHandle`, Isaac and mock: wheel velocities and the base's root pose. |
| `src/isaac_module/models/world.py` | 341 | The `isaac-world` generic component: boot config, `close()`, `do_command` dispatch through `asyncio.to_thread`. |
| `src/isaac_module/models/world_commands.py` | 363 | One function per world verb behind `COMMAND_HANDLERS`, including the payload-driven `scatter_cell` and `clear_cell`. |
| `src/isaac_module/models/world_config_validation.py` | 424 | The world's attribute validators, including `kit_log_level` and the identity-frame rule. |
| `src/isaac_module/models/scene_finalizer.py` | 53 | The `scene-finalizer` generic component: validates `depends_on` names at least one component, and calls `finalize_scene()` on build and reconfigure. |
| `src/isaac_module/models/arm.py` | 650 | The `arm` component: joint moves, IK against the served kinematics, `max_vel_degs_per_sec`, typed gRPC errors, hold-on-cancel. |
| `src/isaac_module/models/camera.py` | 339 | The `camera` component: images, point clouds, encoding. |
| `src/isaac_module/models/gripper.py` | 378 | The `gripper` component: open, grab, jaw state, deadlines. |
| `src/isaac_module/models/base.py` | 242 | The `base` component: differential drive, `width_mm` and `wheel_circumference_mm`. |
| `src/isaac_module/models/component_frame_pose.py` | 81 | Frame-to-spawn-pose conversion. |
| `src/isaac_module/models/sim_component_validation.py` | 100 | The shared spawned-component validation, including the `<arm>:<link>` frame-parent rule. |
| `src/isaac_module/models/conductor.py` | 1098 | The block-sorting service. Not a driver, a client of the sim models. |
| `src/isaac_module/models/sorter_sensor.py` | 112 | A sensor that proxies the conductor's loop records. |
| `src/isaac_module/config_resolver.py` | 478 | Reads a real machine config and writes the sim machine's config, reporting swaps, placeholders, pruned modules and unresolved variables. |
| `src/isaac_module/asset_catalog.py` | 120 | `KNOWN_ASSETS`: USD candidates, joint names, `ee_prim`, packaged kinematics paths. |
| `src/isaac_module/kinematics_files/` | | The UR3e, UR5e, UR7e and UR20 SVA files shipped with the module. |
| `src/isaac_module/usd_assets.py` | 270 | The USD asset repair helpers for the unresolvable Robotiq references. |
| `src/isaac_module/prop_scatter.py` | 306 | The prop scatter engine and its result records. |
| `src/isaac_module/prim_paths.py` | 37 | Prim naming and the default end-effector and base prim paths. |
| `src/isaac_module/assets.py` | 59 | `module://` and `data://` asset path schemes for textures, HDRIs and USD files named in world config. |
| `src/isaac_module/visual_props.py` | 151 | Visual-only props: the `visual` kind's constants, the fit-scale arithmetic and the record shape `status.visual_props` lists. |
| `src/isaac_module/component_diagnostics.py` | 144 | The bodies of the world's per-component introspection verbs. |
| `src/isaac_module/compat.py` | 241 | The one place Isaac Sim is imported, the `IsaacAPI` protocol and the 5.0 capability row. |
| `src/isaac_module/spatial.py` | 237 | Quaternion, orientation-vector and pose composition math. |
| `src/isaac_module/length_units.py` | 11 | The one millimeter-to-meter conversion. |
| `src/isaac_module/encoding.py` | 203 | Image and point cloud encoding. |
| `src/isaac_module/kinematics.py` | 416 | SVA and URDF chains, forward and inverse. |
| `src/isaac_module/physics.py` | 168 | Solver iteration counts and prop physics defaults. |
| `src/isaac_module/errors.py` | 79 | The `SimError` hierarchy, each class carrying its gRPC status. |
| `src/isaac_module/sdk_patches.py` | 58 | One monkeypatch that adds the `MoveThroughJointPositions` handler the Python SDK 0.80 does not serve. |
| `src/isaac_module/cell_layout.py` | 156 | The sorting cell's geometry. |
| `src/isaac_module/sort_plan.py` | 133 | The sorting order. |
| `src/isaac_module/run_log.py` | 171 | Sorting loop records. |
| `src/pickcell/` | 1944 | The pick pipeline, a pure Viam-client library with no `isaac_module` import. |

The block-sorting demo, `models/conductor.py`, `models/sorter_sensor.py`, `cell_layout.py`,
`sort_plan.py`, `run_log.py` and `src/pickcell/`, is a layer above the sim models. It reaches the
sim only through the Viam API and the world's `DoCommand`, and nothing in the sim core imports it.

Around `src/` sit five more directories. `tests/` is 19,095 lines across 69 files and runs entirely
against mocks. `examples/` is 3,469 lines: five GPU checklist scripts, one smoke script, and the
single-pick client. `tools/` is 1,211 lines: the config resolver's CLI, the machine creator, the
RealSense mesh generator, the shader cache warmer, and `convert_mesh.py`, which turns a STEP or
OBJ mesh into a visual USD in metres, Z-up, with its origin at the top-face centre.
`fragments/` holds the two shipped Viam fragments. `provisioning/` holds the GCP image and machine
scripts. `simulates.json` at the repo root is the machine-readable substitution table.

## How a request travels

Take `GetJointPositions` on a simulated arm.

1. viam-server sends the RPC to the module process over its socket. The Viam SDK's `Module` server
   is running on a daemon thread started in `src/main.py`.
2. The SDK dispatches to `IsaacArm.get_joint_positions` in `src/isaac_module/models/arm.py`. The
   model holds an `ArmHandle` it got from `SimManager.create_arm` during `reconfigure`.
3. The model calls `await asyncio.to_thread(handle.joint_positions)`. This hop matters. The handle
   call blocks, and blocking it on the module's event loop would freeze every other component.
4. `ArmHandle.joint_positions` in `handles/arm.py` calls `SimManager.run`. If the caller is
   already on the sim thread, `run` calls `fn` inline, gate or no gate, since nothing on that
   thread can queue behind a slow step. Otherwise `run` checks the scene gate: while the world is
   waiting on a `scene-finalizer` and is not yet ready, it refuses with `UNAVAILABLE` unless the
   caller is a component factory or a stop verb, both of which pass
   `allow_during_initialization=True`. Past the gate it puts `(fn, Future)` on a `queue.Queue`
   and waits on the future with a 30 second default timeout.
5. The sim thread is `SimManager.main_loop`. Once per step it drains the
   queue in `_drain_tasks`, runs each callable, and sets its future. While a configured
   `scene-finalizer` has not yet been built, the loop keeps draining the queue but does not step
   the world. Once the finalizer runs, the loop steps three times before reporting the world
   ready. Otherwise it calls `world.step(render=True)` every cycle.
6. `fn` reads the Isaac articulation and returns radians. The model converts to degrees and builds
   the Viam response.

Every read follows this path at the moment of the call. Nothing is cached on the Viam side. Every
write is a command Isaac's physics executes, and the handle reports settle, stall or timeout the way
a real controller would. That is the source-of-truth rule in [SIMULATION.md](SIMULATION.md), and the
queue is what makes it safe: there is one thread that touches Kit, and `run` is the only door.

## The layers and the seams between them

Four layers, and the seams between them are the parts to protect.

**The Viam model layer** is `src/isaac_module/models/*.py`. A model parses its config, resolves its
dependencies, converts units, and delegates. It never imports `omni` or `pxr`. It reaches the sim
only through `SimManager.create_arm`, `create_camera`, `create_gripper`, `create_base`,
`world_handle` and `handle_entry`. Viam speaks millimeters and degrees, Isaac speaks meters and
radians, and every conversion lives here.

**The handle seam** is `WorldHandle`, `ArmHandle`, `GripperHandle`, `BaseHandle` and `CameraHandle`,
each with an Isaac implementation and a mock one. Every public method is safe to call from any
thread, because every one of them is a `SimManager.run` wrapper. Handles never construct a Viam type
and models never touch a prim. This is the interface [SIMULATION.md](SIMULATION.md) reserves as the
surface a future simulation API gets cut from, with these models as its first clients. It is also
what makes the mock gate possible: 923 tests run the whole module with no GPU because every handle
has a mock twin. `docs/PARITY.md` tracks the model layer method by method against the real drivers.

**The sim lifecycle** is `SimManager`, a process singleton. Kit must own the process main thread,
because `SimulationApp.__init__` installs a SIGINT handler and `signal.signal` raises off the main
thread. `src/main.py` honors that: the sim loop runs on the main thread and the module server runs
beside it. `ensure_booted` is called from the world component's
`reconfigure` and blocks until the sim is up. `_reset_world` is the single chokepoint for a
world reset, and it runs the post-reset hooks that re-apply PD gains and solver iterations, which
`World.reset()` discards.

**The config-time layer** is `config_resolver.py` and `simulates.json`. The resolver reads a real
machine config and writes a second machine's config. Every hardware resource whose model has a row
gets swapped to that row's sim model, with `world` set to `isaac-world`, the row's attribute
template applied, and the attributes the row's `carry` map names copied over. `name`, `api`, `frame`,
`depends_on` and every service come out byte-identical, which is what lets a client script point at
either machine unchanged. `tests/test_simulate_config.py` pins that byte-for-byte against
`examples/configs/real-ur5e-cell.json` and `examples/configs/sim-ur5e-cell.json`.

**The demo cell** sits above all four. `models/conductor.py` is a service, not a driver. It holds
the sort loop, depends on motion and vision services and two cameras, and drives the arm through
`src/pickcell/`, which is a pure Viam-client library that imports nothing from `isaac_module`.
`tests/test_pickcell_imports.py` enforces that boundary in a subprocess.

The one rule that keeps the seams honest is that the `world` component is the only resource that
knows it is a sim. Scene verbs and per-component introspection verbs live on the world's
`DoCommand`, keyed by component name, and reach the sim through `SimManager.handle_entry`. A sim arm
never grows a verb a real arm could not answer. `models/arm.py` and `models/gripper.py` raise on
every `DoCommand`, and `tests/test_arm_model.py` gates it.

`cell_layout`, the demo's geometry module, is imported only by the demo's own files. The pooled
`scatter_cell` and `clear_cell` verbs take their prop names, region and park grid from the
`DoCommand` payload, and the conductor sends them.

## Extension points

The four things a second sim example is most likely to need, and where each one lands.

**A mobile base on a floor needs no module code.** Write a fragment and a config. The world adds a
default ground plane when no `usd_stage` is set. The `jetbot` asset is already in `KNOWN_ASSETS`
with its wheel radius and wheel base. The base's frame parent is `world`, so the spawn-pose
validation in `models/sim_component_validation.py` accepts it. This is the cheapest new example by a wide margin.

**A new arm family needs three edits.** Add the asset to `KNOWN_ASSETS` in
`src/isaac_module/asset_catalog.py` with its USD candidates, `joint_names`, `ee_prim`, a
`kinematics` path (ship the SVA under `kinematics_files/`) and any `base_frame_correction`. Add a
row to `simulates.json` mapping the real driver's model string to `viam:isaac-sim-devin:arm` with
`{"asset": "<name>"}` and a `carry` for the real speed attribute. Mirror that row into the
substitution table in [SIMULATION.md](SIMULATION.md#substitution-table), which a test compares row
for row. `ee_prim` is read in one place, `prim_paths.default_ee_prim_path`, and a camera or gripper
mounted on an arm whose asset declares none is rejected at resolve time instead of spawning at a
prim that does not exist. The `ur7e` row shows the shape: its own kinematics, the UR5e mesh as a
geometry stand-in until Isaac ships a UR7e USD.

**A new gripper needs code, and less of it than it looks.** `SimManager.create_gripper` dispatches
on `KNOWN_ASSETS[asset]["kind"]`, so a second parallel-jaw gripper is an asset row with a different
`drive_joint` and a different `tcp_offset_m`. A suction or three-finger gripper is a new `kind` and
a new branch.

**A second world asset is a config change.** Set `usd_stage` on the world component and the module
opens that stage instead of adding a ground plane. Props are declared in the world's `props`
attribute and spawned at boot. If the new scene needs a scatter or reset verb, add it to
`_SUPPORTED_COMMANDS` in `models/world.py`, write its function in `models/world_commands.py` and
give it a handler on `WorldHandle` with both an Isaac and a mock implementation, so the mock gate
still covers it.

Whatever you add, three invariants hold. A component model never grows a verb a real driver could
not answer. Every handle gets a mock twin, or the test suite stops being able to run without a GPU.
And the frame config, not an attribute, is where a spawned prim's pose comes from.

