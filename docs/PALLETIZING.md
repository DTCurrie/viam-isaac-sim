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

Two declarations reach the stage, and they come from two different places.

The collider is the component's `frame.geometry` in the vendored fragment, read back through the
machine's own frame system (`isaac_module.frame_system`) and spawned as a fixed cube named
`frame_<component>`, with a hyphen in the component name becoming an underscore. The motion
service plans against exactly that box, so one declaration serves the planner and the physics. A
component with no `frame.geometry` (`scan-tunnel`, `hmi-cabinet`, `stack-light`, `caution-tape`)
has no collider, and that is a declaration too: the arch would be sealed by one box.

The collider's pose is the frame's pose composed with the geometry's, orientation included.
`fence-left` and `fence-right` turn their frames 90 degrees about z, and until 2026-09-22 the
collider took the translation and dropped the turn, so two 1200 mm panels stood across the cell
along x, in front of the back fences, while the planner had them along y.

The render is the component's `get_visuals` DoCommand reply. Every `viam:workcell-components`
component builds its primitives in the world frame, then re-expresses each one relative to an
anchor primitive it puts first in the reply, a `frame` labelled `<component>/group` sitting at the
component's frame pose. `isaac_module.workcell_client.group_anchor` reads that anchor, and
`SimManager.materialise_components` composes each box and capsule onto it before spawning a
render-only cube named `<component>-<label>`. Spheres and arrows are not spawned.

The anchor is read from the reply rather than from `get_attributes.pose` because the two are not
the same pose for every model. `pick-station` reports its bottom-left-top corner as its pose,
centre plus half its width and length back and half its thickness up, and its primitives are
anchored on the centre. Composing onto the corner drew the whole station 585 mm from its own
collider.

`GetGeometries` exists on the Go types for `pallet`, `pick-station` and `safety-fence`, and is
not served over the generic API: every component answers UNIMPLEMENTED for it on the wire. The
client still asks, logs the refusal, and takes nothing from it. `robot-pedestal` also gets a
collider derived from its own `height_mm` and `diameter_mm`, which duplicates its `frame.geometry`
box exactly, and retiring that duplicate is open.

A render-only primitive carries `"collision": False` and reaches the stage with no collider and no
rigid body, so a fence screen or a floor decal is something the planner routes around, through the
frame system, rather than something a box can bounce off.

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

## The sequencer owns the pack

`box-palletizer` (the `viam:isaac-sim-devin:palletizer` service) does not decide where a box goes.
It asks `pack-sequencer`, an instance of `viam:pack-sequencer:sequencer` pinned at `0.3.0`, for the
next slot, moves the arm there through the `builtin` motion service, reports what happened, and
publishes the box's settled pose back once physics has finished with it. The sequencer owns the pack
order, the placement cursor and every place target. This service carries no packing arithmetic of
its own, and it never drives the arm directly.

Every motion goes through the motion service. The sequencer's `place_start_in_world` is the pose the
arm descends from, and the offset between it and `place_end_in_world` already carries the
sequencer's own approach clearance, so the service adds no standoff of its own at the place end.
The pick end keeps its own standoff, since the sequencer knows nothing about the pick station.

The cup does not descend all the way to `place_end_in_world`. That pose is the cup at the box's top
face with the box on its slot, and the cup never holds a box at its top face: it takes hold
`CUP_APPROACH_GAP_MM` above it, the gap that keeps a rigid tool from being driven into a rigid box,
and the weld freezes that gap. A descent to `place_end` itself puts the box that far into the deck,
and on 2026-09-22 the arm stalled a few tenths of a degree short of it with the box already
resting on the pallet. So the cup releases at `place_release_pose`: `place_end` raised by the grasp
gap and by `PLACE_RELEASE_CLEARANCE_MM`, which leaves the box's bottom that clearance above the
deck. The clearance is sized for the arm's tracking at the end of a descent, not for the geometry:
in one of its configurations the arm arrives a degree over at the shoulder and four short at the
wrist, which puts a hanging box's corner 16 mm lower than commanded, and a clearance smaller than
that lands the corner on the deck's edge before the cup reaches its pose.

A straight-line leg that the planner refuses falls back to a free move. A straight-line leg that
planned and then stalled on the arm does not: the arm is blocked by contact, and the failure is
reported as the stall it was. The one exception is the place descent. A descent that stalls with
the cup within `PLACE_STALL_TOLERANCE_MM` of its release pose has been stopped by the deck or by a
neighbouring box, which is what it was descending towards, and the box is released where it is.
The sequencer's first slot sits flush with two edges of the pallet, and a box that arrives a few
millimetres wide catches the deck's edge before the cup reaches its target.

From the lift to the place descent the box is on the cup, and the planner is told so twice. Each
of those legs carries a `Transform` parented to the gripper frame with the box's own dimensions,
hanging the grasp gap plus half a box height along the gripper's z, which is the tool axis and
points down at a grasp. And each of them is level: a free path keeps the tool's orientation within
`CARRY_ORIENTATION_TOLERANCE_DEG` of where it started. The transform alone was not enough. On
2026-09-22 the planner, box attached, still joined two pointing-down poses whose wrist 2 solutions
differed by 180 degrees with one segment that turned the box over the top of the arm and into the
forearm. Which solution the arm is in when a carry begins is the planner's coin flip, so the carry
forbids the turn rather than hoping the coin lands right.

## Keep-outs, not obstacles

Neither the box being picked nor the pallet can be an obstacle in its own right. The cup's job is
to reach both, and a plan that treats either as solid cannot descend onto it. Leaving them out of
the obstacle set on every leg is what the service did first, and the GPU runs of 2026-09-16 showed
what that costs: the swing to the pick standoff routed a link straight through the box and knocked
it off the station before the descent had begun.

So a no-fly box stands in their place, the same `pickcell.obstacles.pick_area_keepout` the
colour-sorting cell uses. The pick keep-out covers the box, grown sideways and stopping short of
the standoff so the pose the arm descends from stays reachable. The place keep-out covers the whole
place support, from its deck up to just below the pose the arm approaches it from, since what it
protects is the boxes already stacked there.

Each leg opens the zone it is working in and keeps the other closed.

| leg | pick zone | place zone |
|---|---|---|
| approach to the pick standoff | closed | closed |
| descent onto the box | open | closed |
| lift off the pick | open | closed |
| cross to the place approach | closed | closed |
| descent onto the place | closed | open |
| retreat off the place | closed | open |

## palletizer attributes

| attribute | default | notes |
|---|---|---|
| `world` | `isaac-world` | name of the world component, defaults to this module's world name (`viam:isaac-sim-devin:world`) |
| `arm` | required | name of the arm component (boot ordering only, every motion goes through `motion`) |
| `gripper` | required | name of the gripper component, driven only through the Viam Gripper API (`open`, `grab`, `is_holding_something`), so either gripper model works unchanged |
| `motion` | required | name of the motion service (`"builtin"` works) |
| `sequencer` | required | name of the `viam:pack-sequencer:sequencer` service |
| `box_props` | required, non-empty list of strings | prop names in pick order. `box_props[i]` fills the sequencer's seq `i + 1` |
| `place_support_prop` | `frame_pallet` | the prop whose airspace the arm keeps out of except while placing onto it. A `frame.geometry` collider spawns as `frame_<component>`, and a prim name cannot hold a hyphen, so the `pallet` component's collider is `frame_pallet`. Named here rather than hardcoded, since which component carries the place support is the cell's business |
| `obstacle_source` | `world_state_store` | where the motion service's obstacles come from. `world_state_store` leaves obstacle assembly to the frame system and the store the motion service already consults. The only other accepted value, `prop_geometries`, keeps this service's own hand-built `WorldState`, the same obstacle helper the colour-sorting cell's conductor already uses, so it stays available as a verified fallback if the store path does not hold up |

## One box on the pick station at a time

Eight boxes exist as rigid bodies from the start of a run, `infeed_box_1` through `infeed_box_8`,
one sitting on the pick station and the other seven parked clear of the cell. The pick station is
1100 mm long and each box is 150 mm along the axis it queues on, so eight of them touching end to
end would not fit.

`box_props[0]`'s pose on the first `prop_geometries` read of a run is captured as the infeed pose,
read from the sim rather than from a configured pick pose. Every later `box_props[i]` is re-posed
onto that same pose, through the world's `set_prop_pose` verb, before its own pick. That re-pose is
what a conveyor does: nothing else in this cell moves a box onto the pick station, so replacing the
picked box with the next one in line is the closest single-prop model of a belt delivering it. The
pallet stack itself is never re-posed. Every box placed on it is a rigid body that slides, tilts and
settles like the rest of this cell.

A seq that fails twice in a row is skipped, through the sequencer's own `skip_box` verb, rather than
retried a third time, since the sequencer's cursor already gives one retry for free by staying put
on the first failure.

## The verbs this service drives

| verb | argument | what it returns |
|---|---|---|
| `next_box` | none | the slot to fill next, or the completion tally when there is none left |
| `report_placement` | `seq`, `success`, `error` | the cursor after recording one outcome |
| `skip_box` | `seq`, `reason` | retires one seq without placing it, so the run moves on |
| `set_box_transform` | `seq`, the measured pose | the settled pose published back to the sequencer, so the viewer shows what physics did rather than what the plan intended |

## DoCommand

- `{"command": "start"}` runs the whole pack. `{"ok": true, "state": "running"}`, or
  `{"ok": false, "state": "running"}` unchanged when a run is already going.
- `{"command": "stop"}` cancels between motions, never mid-motion. `{"ok": true}`.
- `{"command": "status"}` reports the run state and its records.

Note that the workcell's own components, and the sequencer, take a different DoCommand shape,
`{"<verb>": <argument>}` rather than `{"command": "<verb>"}`. That difference is load bearing: it is
why probing this module's own world component with `{"get_schema": true}` finds no handler and
excludes it from scenery.
