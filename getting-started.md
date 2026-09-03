# Getting started guide

This guide assumes that you're on linux with the Isaac simulator already installed.

These instructions were tested with:
- nvidia-open driver 580.178.04
- dual RTX 5060 Ti
- isaac sim 5.0.0-rc45 installed at /isaac-sim
- ubuntu 24.04.4

## Viam setup

1. Go to app.viam.com and follow the account creation flow, or sign in if you already have an account
1. Go to your [fleet page](https://app.viam.com/fleet/) and use the 'Add machine' button on the right to create a machine we'll use to host the simulation
1. If you're on your personal dev computer, you probably don't want to install the viam daemon. Instead, pick a folder on your computer to download the viam-server binary:
	- `~/viam-isaac` is a good default
	- inside the new folder, download viam-server with `wget https://storage.googleapis.com/packages.viam.com/apps/viam-server/viam-server-stable-$(uname -m) && mv viam-server-stable-$(uname -m) viam-server && chmod +x viam-server && ./viam-server -version`
1. Grab your credentials: back in the web UI, go to the status dropdown on the top menu bar. It's likely in the blue 'awaiting setup' state. Open it, hit the 'Machine cloud credentials' button, then paste the credentials into a `viam.json` file in your `~/viam-isaac` folder.
1. Boot viam: in your terminal, run `ISAAC_SIM_PATH=/isaac-sim ./viam-server -config viam.json`. As it comes up, in the web UI, you should see the status dropdown turn to a green 'Online' state.

## Start Isaac in Viam

In the 'configure' tab of the web UI, hit the '+' button or tap 'A', then tap 'B' for blocks, then find the `isaac-sim-pick-and-place` fragment from the `viam-dev` org (it pulls in the private `viam:isaac-sim-devin` registry module — the machine must be in `viam-dev` to see it) and install it. Click 'Save' in the top right.

The fragment card has a 'Variables' section with nineteen entries. Leave every one of them unset for your first run. They group as:

- `table-height-m` (default `0.75`): the table top height, in metres, that every other measurement in the cell is built from.
- Six `block-color-<color>` entries, one per `red`, `green`, `blue`, `yellow`, `purple`, `orange`: the RGB triple each pool block is spawned with.
- `detect-color` (default `#EA8D8D`) and `hue-tolerance-pct` (default `0.05`): the red detector's target hue and tolerance, kept unsuffixed for backward compatibility.
- Five `detect-color-<color>` entries (green, blue, yellow, purple, orange): the other five detectors' target hues, each with its own default.
- Five `hue-tolerance-pct-<color>` entries (green, blue, yellow, purple, orange): each defaulting to `0.05`.

The defaults boot the exact shipped cell: eighteen pool blocks (three of each color), six place pads, and a `ur20` arm with a gripper.

Switch to the 'logs' tab to watch the installed components start up. On the test machine this takes on the order of tens of seconds. You'll see an 'event=complete' event from the rdk.activity logger when this is done.

Now switch to the 'control' tab to interact with the cameras and arm. The cell has three cameras: `wrist-cam` (rides the arm's flange), `scene-cam` (a fixed overview), and `side-cam` (a fixed camera at the source table's far end, used to measure the tallest scattered block). Open the `scene-cam` livestream.

You should see three tables in a row. On the right sits the empty source table. In the centre sits the table with the `ur20` arm and its gripper. On the left sits the place table, with six colored pads laid out on it. In the arm's own frame, the source table sits on the arm's negative-x side and the place table on its positive-x side, so you can still orient from a different camera. The eighteen pool blocks aren't on any table yet. They're parked off-cell, on the floor behind the tables, one column per color and one row per pool index, and they stay parked until the first loop scatters them onto the source table. If the tables or arm are missing, something failed during boot. Check the logs tab before continuing.

If something goes wrong, the place to debug is the logs tab. To cut down on noise, find the components list on the left-side menu bar and click the `viam_isaac-sim-devin` module (or whatever you named the local module) to filter down the output.

## Put the cell in pick-and-place mode

Everything from here on happens in the 'control' tab, with no script and no local Python. Find the `block-sorter` generic service in the components list on the left, and open its DoCommand panel.

Paste this into the DoCommand input and hit send:

```json
{"command": "start", "loops": 1, "seed": 20}
```

You should get back:

```json
{"ok": true, "state": "running"}
```

`"loops": 1` runs exactly one scatter-and-sort loop, then stops on its own. Raise it to sort several loops back to back (`"loops": 3` runs three), or set `"continuous": true` (equivalently, `"loops": 0`) to run until you stop it yourself. `"seed": 20` fixes the scatter so the layout is reproducible. Leave `"seed"` out entirely to get a time-derived base seed instead, which is still reported back to you (as `status.run.base_seed`), so you can replay any run later by passing that number as `"seed"`.

If you send another `{"command": "start", ...}` while a run is already in progress, it's a no-op: you get `{"ok": false, "state": "running"}` and the run in progress is untouched.

While it's running (or after it finishes), send:

```json
{"command": "status"}
```

to see where the run stands. See "Reading the results" below for what to look at in the reply.

To stop a run early, send:

```json
{"command": "stop"}
```

which replies `{"ok": true}`. The run doesn't stop instantly: it finishes whatever motion or pass is in flight, then stops at the next safe boundary (between motions, between passes, or between loops) and parks the arm. `status` will read `"state": "idle"` once it has actually stopped.

## What a loop looks like

Each loop runs the same sequence, whether it's loop 1 of 1 or loop 40 of a continuous soak:

1. The arm moves to a park pose. This happens once before the very first loop's census, not on every loop.
2. The cell resets: `clear_cell` then `scatter_cell` at that loop's seed. In the `scene-cam` livestream you'll see every block visibly jump as it's re-parked and then re-scattered onto the source table. That's expected, not a fault.
3. A three-pose census scans the source table to find every block and its color.
4. The conductor picks nearest-first: pick, verify the grasp, carry, and place each block on its color's pad. This happens in passes. A block that's too crowded to reach safely waits for a neighboring block to clear first, and the conductor re-censuses and tries again, up to five passes.
5. Each block ends the loop as `placed`, `skipped_oversize` (measured over the 75 mm jaw limit), or `failed` (a grasp that failed is retried once before it's given up on).
6. A loop record is cut, the seed advances for the next loop, and either the next loop starts at step 1 (minus the park, which only happens once per run) or the run reports `"state": "complete"`.

On the test machine, a GPU run of three loops (`{"loops": 3, "seed": 20}`) placed 30 blocks, skipped 4 as oversize, and failed 1, for a `success_rate` of 0.97. A separate seedless continuous soak ran about 31 minutes across three loops before being stopped, so budget on the order of ten minutes per loop as a rough guide, not a guarantee — actual wall time depends on your machine and how crowded the scatter is. During that soak, one loop was lost partway through to a dropped connection between the module and viam-server, and the conductor recorded that loop with an `"error"` field and moved straight on to the next loop's seed rather than failing the whole run. Sending `{"command": "stop"}` mid-loop during that same soak landed the run in `"state": "idle"` cleanly, with whatever blocks hadn't been picked yet left where they were.

## Reading the results

When you send `{"command": "status"}`, the fields worth looking at first are:

- `"state"`: `idle`, `running`, `stopping`, `complete`, or `failed`.
- `"run"`: cumulative counts for the whole run so far, including `"placed"`, `"failed"`, `"skipped_oversize"`, the current loop number in `"loop"`, `"loops_completed"`, `"loops_errored"`, and `"base_seed"` (the seed to pass to replay this exact run).
- `"success_rate"`: `placed / (placed + failed)` across the run, `null` until at least one block has been placed or failed. Oversize skips don't count against it.
- `"loop_records"`: the most recent loop records, oldest first. Look at the newest one for that loop's `"duration_s"`, `"passes"`, and `"rss_mb"` (the module process's resident memory in MiB, sampled when that loop's record was cut, useful for spotting a memory leak across a long soak).

`"skipped_oversize"` isn't a bug: the default scatter draws each block's size from a `[50, 80]` mm range, but the gripper's jaw tops out at 75 mm, so roughly a sixth of blocks are, by design, too big to grasp safely and get skipped instead of attempted. If you want every block in the scatter to be sortable, override the `block-sorter` service's `size_range_mm` attribute to `[50, 74]` (via a fragment mod in the machine's JSON config) so nothing scattered can exceed the jaw limit.

The `block-sorter-sensor` component in the control tab mirrors this same status as sensor readings, which is what lets you wire up data capture on it: each loop record is captured at most once, deduplicated so a data-management poll never stores the same loop twice.

If a `loop_records` entry carries an `"error"` field, that loop didn't finish sorting. It was lost to a transient failure, for example a dropped connection to viam-server mid-motion, but the run itself kept going with the next loop's seed rather than stopping. Only three such loops in a row will fail the whole run.

## Optional: the single-pick client

`examples/pick_red_block.py` is a regression tool from before the conductor existed, not the way to operate the cell day to day — it exercises the pick-verify-carry-place pipeline against just the `red` block and `place_pad_red`, without the conductor's scatter, census, or loop management.

Clone this repository on your dev machine, then create the venv it uses:

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

Find your connection details in the web UI's 'connect' tab (select 'Python'), or the 'code sample' tab, which shows a runnable connection snippet with your machine's address and API key filled in.

Then run:

```
.venv/bin/python examples/pick_red_block.py --address <machine-address> --api-key <key> --api-key-id <key-id> --support-z-mm 750
```

`--support-z-mm 750` tells the script the block rests 750 mm up, on the table top. Useful flags if you want to poke at it further: `--block` (default `block_red_1`) picks a different pool block by name, `--place-pad` (default `place_pad_red`) targets a different pad, `--no-place` releases at the lift pose instead of placing, and `--randomize-seed <n>` with `--randomize-size-mm <lo,hi>` re-scatter and re-size the named blocks before the pick through the world's `randomize_props` verb, which is a different mechanism from the conductor's pooled `scatter_cell` reset.

On success, the script prints `PLACED_BLOCK_JSON=` with `"placed_on_pad": true`, meaning the arm found the block, picked it up, and set it down on the pad. If the block measures over 75 mm, the script refuses the grasp cleanly and leaves the arm parked instead of attempting a doomed pick, which is correct behavior for an oversize block, not a failure.
