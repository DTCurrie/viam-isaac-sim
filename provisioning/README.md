# Provisioning a sim machine

Redeploy: not applicable, this provisions a new host.

Three ways to get a running sim machine on Isaac Sim, in the order a reader
should try them: build the image once, then either drive it by hand or with
`create-sim-machine.sh`.

## 1. Building the image

`build-image.sh` turns a bare GCP GPU instance into the `viam-isaac-sim`
image: the NVIDIA driver, Python 3.11, Isaac Sim 5.0.0 baked into the venv
`first_run.sh` looks for, and viam-agent installed and ready for a machine's
credentials.

1. Confirm `gcloud` is authenticated and pointed at the right project:
   `gcloud auth list` and `gcloud config get-value project`.
2. Run the build:

   ```
   provisioning/build-image.sh --project YOUR_PROJECT
   ```

   Flags: `--zone` (default `us-central1-a`), `--machine-type` (default
   `g2-standard-8`), `--image-family` (default `viam-isaac-sim`), `--dry-run`
   (prints every `gcloud` and remote command it would run, without running
   any).
3. Expect it to create a builder instance, install the driver and Isaac Sim,
   reboot once to confirm the install, install viam-agent, then stop the
   builder, snapshot it into an image named `viam-isaac-sim-<YYYYMMDD>` in the
   `viam-isaac-sim` family, and delete the builder. Isaac Sim's pip install is
   the slow step, and the whole run takes on the order of 20-30 minutes,
   most of it the reboot and the Isaac Sim download.
4. Confirm the image landed: `gcloud compute images list --filter="family=viam-isaac-sim"`.

## 2. Creating a sim machine by hand

Use this path on a cloud other than GCP, or when you want to see every step.

1. Create the machine. Target: [app.viam.com](https://app.viam.com), the
   `Add machine` button on your [fleet page](https://app.viam.com/fleet/).
   Expected result: a new machine page in the `AWAITING SETUP` state.
2. Copy its credentials. Target: the machine's status dropdown in the top
   menu bar, `Machine cloud credentials`. Expected result: a `viam.json`
   document with the part's cloud address and secret.
3. Launch an instance from the image and deliver the credentials:

   ```
   gcloud compute instances create YOUR_INSTANCE_NAME \
     --image-family viam-isaac-sim \
     --accelerator type=nvidia-l4,count=1 \
     --maintenance-policy TERMINATE \
     --metadata-from-file=viam-json=/path/to/viam.json \
     --metadata=startup-script='curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/viam-json" > /etc/viam.json && systemctl restart viam-agent'
   ```

   Target: your GCP project. Expected result: a running instance with
   `/etc/viam.json` in place and viam-agent restarted against it.
4. Wait for it to come online. Target: the machine's status dropdown.
   Expected result: the state turns to `ONLINE`, on the order of a minute
   after `viam-agent` restarts.
5. Add the sim world. Target: the `CONFIGURE` tab, the `+` button, then
   search for the `isaac-sim-world-devin` fragment by name and install it. Save.
   Expected result: the `LOGS` tab shows the `isaac-world` component coming
   up. Alternatively, paste the output of `tools/simulate_config.py` in place
   of the fragment if you are starting from a real machine's config, since
   its output already carries the module entry and `isaac-world` and adding
   the fragment on top would duplicate the world.
6. Open the livestream. Target: the `CONTROL` tab, the `isaac-world`
   component's livestream panel. Expected result: an empty Isaac Sim stage
   with no tables and no arm.

## 3. Creating a sim machine with the script

`provisioning/create-sim-machine.sh` wraps the steps above into one command,
built by a sibling slice of this phase.

```
provisioning/create-sim-machine.sh <real-config.json> --name NAME --location-id ID \
  [--project P] [--zone Z] [--dry-run]
```

Target: the Viam org identified by your `VIAM_API_KEY` / `VIAM_API_KEY_ID`,
plus the GCP project named by `--project` (or the active `gcloud` project).
Expected result: a machine named `NAME` in location `ID`, an instance
launched from the `viam-isaac-sim` image with that machine's credentials, and
the resolved config (from `tools/simulate_config.py`) pushed once the part
reports online. `--dry-run` prints every remote call without making any.

## Pins

- Ubuntu 24.04, image family `ubuntu-2404-lts-amd64` in project
  `ubuntu-os-cloud`: `build-image.sh`'s source image, matching the version
  `first_run.sh` targets on the `24.04` branch of its case statement.
- NVIDIA driver 580: the branch `first_run.sh` installs. Later branches
  (590/595) crash the RTX renderer (isaac-sim/IsaacSim#537, #643).
- Python 3.11: the interpreter `first_run.sh` picks for Ubuntu 24.04.
- isaacsim 5.0.0: the pip package version `first_run.sh` installs from
  `https://pypi.nvidia.com`.
- `g2-standard-8` (one NVIDIA L4): `build-image.sh`'s default machine type,
  matching `tools/create_sim_machine.py`'s `DEFAULT_MACHINE_TYPE`.

These pins were not read off the reference VM directly, since `gcloud` was
logged out when this doc was written. The reference VM inventory (user-run)
corrects them if it finds a mismatch.
