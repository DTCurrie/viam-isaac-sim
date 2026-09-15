# The palletizing cell

This cell is Viam's own palletizer workcell, booted in Isaac Sim. The demo behind the Viam 101
lessons already runs it, on an `rdk:builtin:simulated` arm and a gripper that reports a hold on a
timer. Nothing in that stack simulates contact, so nothing in it can answer whether a pallet
stands up. Swapping three resources is what makes that question answerable.

Two files describe the cell. `fragments/isaac-sim-palletizing.json` is the upstream workcell
fragment, vendored so an upstream edit cannot surprise a run, with its provenance under a
top-level `_upstream` key. `examples/configs/sim-palletizer-cell.json` is the sim overlay. Together
they are the machine config.

The overlay swaps the arm for `viam:isaac-sim-devin:arm`, the gripper for
`viam:isaac-sim-devin:vacuum`, and adds a world component to boot the simulator. The fragment's
scenery, its sensors and `pack-sequencer` stay exactly as they are.

## The geometry is the workcell's, not ours

Every pose comes from the vendored fragment. A UR5e stands on a 150 mm `robot-pedestal` at the
world origin. The `pick-station` sits at (400, -650, 200) mm and the `pallet` at (200, 500, 200) mm,
where a pallet's frame origin is its bounding-box centroid, so its deck top is half its
`thickness_mm` above that. Boxes are 150 x 200 x 100 mm, from `pack-sequencer`'s own
`box_length_mm`, `box_width_mm` and `box_height_mm`.

The earlier version of this cell invented all of those numbers. Adopting the workcell's means
there is nothing here to defend or to keep in sync.

## How scenery becomes physics

`viam:workcell-components` describes each component two ways, and between them the cell is covered.

`GetGeometries` is the `resource.Shaped` path, implemented by `pallet`, `pick-station` and
`safety-fence` and nothing else. Each returns a coarse typed box, which is what the frame system
plans against and what PhysX wants for a collider.

`get_visuals` is a DoCommand verb every component serves. It decomposes a component into
primitives, so a pedestal is a box base, a capsule column and a box flange. That is render detail,
not collision detail.

So the rule is: collider from `GetGeometries` where the component offers one, render from
`get_visuals` for everything, and a derived collider for a component with no `GetGeometries` that
the arm or a box still touches. `robot-pedestal` is the known one, derived from its `height_mm` and
`diameter_mm`.

`isaac_module.workcell_scenery` holds those decisions as pure functions.
`isaac_module.workcell_client` fetches the payloads, and `SimManager.materialise_components` turns
them into prims. The scene finalizer drives all of it at boot, discovering which of its generic
dependencies are workcell scenery by probing each one with `get_schema`.

A render-only primitive carries `"collision": False` and reaches the stage with no collider and no
rigid body, so a fence or a floor decal is something the planner routes around rather than
something a box can bounce off.

## The box is ours

In the demo a box is a transform the sequencer publishes and the pick station draws. Here it is a
rigid body that can slide, tilt and fall. Every millimetre of difference between where the plan put
it and where it ended up is something the demo cannot produce.

The end effector is a `viam:isaac-sim-devin:vacuum` wearing the Robotiq EPick's dimensions, a
196 mm tool. A suction cup takes hold by welding whatever is under it to its tool prim with a
`UsdPhysics.FixedJoint`, and lets go by removing that joint. It holds a box by the box's TOP face,
so the box hangs its full height below the TCP. Every place height in this cell follows from that.

`grab()` waits out `grab_delay_ms` before reporting a hold, which stands in for the pump cycle a
real cup needs, and `is_moving()` is true for that window. The module defaults it to 1000 ms and
the overlay sets 250 ms to match the demo machine.

Boxes are cardboard and are not identified by colour, so this cell runs no colour detectors and the
world's albedo textures are free to make cardboard look like cardboard.

## palletizer attributes

`box-palletizer` (the `viam:isaac-sim-devin:palletizer` service) drives one pick and place via
DoCommand. The `builtin` motion service plans every motion, and the service never drives the arm
directly.

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name (`viam:isaac-sim-devin:world`) |
| `arm` | required | name of the arm component (boot ordering only, every motion goes through `motion`) |
| `gripper` | required | name of the gripper component, either gripper model |
| `motion` | required | name of the motion service (`"builtin"` works) |
| `box_prop` | required | prop name of the box to pick |
| `place_pose_mm` | required | `{"x", "y", "z"}` the gripper TCP descends to before releasing. The TCP, not the box centre |

The pick pose is not configured. It is read live from the box prop's own geometry, so the service
carries no packing arithmetic of its own.

## DoCommand

- `{"command": "start"}` runs one pick and place. `{"ok": true, "state": "running"}`, or
  `{"ok": false, "state": "running"}` unchanged when a run is already going.
- `{"command": "stop"}` cancels between motions, never mid-motion. `{"ok": true}`.
- `{"command": "status"}` reports the run state and its records.

The gripper is driven only through the Viam Gripper API (`open`, `grab`,
`is_holding_something`), so the same service drives either gripper model without knowing which
mechanism is under it.

Note that the workcell's own components take a different DoCommand shape, `{"<verb>": true}` rather
than `{"command": "<verb>"}`. That difference is load bearing: it is why probing this module's own
world component with `{"get_schema": true}` finds no handler and excludes it from scenery.
