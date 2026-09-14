# tools

- `create_sim_machine.py` — creates a sim machine for a real machine's config end to end: resolves
  the config, creates the Viam machine, launches a GCP GPU instance from the module's image, and
  pushes the resolved config once the part is online.
- `generate_realsense_mesh.py` — authors the low-poly RealSense D435 body mesh embedded in the
  fragment, a binary STL in millimetres centred on the camera frame origin.
- `simulate_config.py` — resolves a real machine's config into its sim twin and writes it as JSON.
- `warm_shader_cache.py` — boots the module's own world and drives a few sim steps so Isaac Sim's
  RTX shader cache is warm before the image ships, without a running viam-server.
- `convert_mesh.py` — converts a STEP or OBJ mesh into a visual-only USD in metres, Z-up, with its
  origin at the top-face centre. See the fixture-table recipe below.

## Fixture-table conversion

The block-sorting cell's three tables are visual meshes referencing one converted USD, fit to each
table's existing collider. The source model is the GrabCAD fixture table:
`https://grabcad.com/library/modular-welding-fixture-table-o28-hole-grid-15-mm-steel-top-1`.

GrabCAD's library terms govern redistribution of that model. Read them before a converted copy is
committed to this repo or published as part of the module. Until that read happens, the converted
USD stays outside the repo, next to the downloaded STEP, and is copied to a VM by hand.

### 1. List the part labels

Run `--list-parts` on the STEP to see every part label and its face count, so you can choose a
`--keep` regex:

```
.venv/bin/python tools/convert_mesh.py \
  "/path/to/SWT-160x100x090-00 Schweisstisch.STEP" \
  --list-parts
```

### 2. Convert

`convert_mesh.py` writes the USD with `pxr` (`usd-core` from PyPI), which is not part of this
repo's `.venv`. Run the conversion under a Python that has `usd-core` installed, for example a
scratch virtualenv:

```
python3 -m venv /tmp/usdenv
/tmp/usdenv/bin/pip install usd-core

/tmp/usdenv/bin/python tools/convert_mesh.py \
  "/path/to/SWT-160x100x090-00 Schweisstisch.STEP" \
  -o "/path/to/converted/SWT-160x100x090-00.usd" \
  --obj-out "/path/to/converted/SWT-160x100x090-00.obj" \
  --keep '^(K0742(?!.*004)|SWT-(?!160x100x090-00)(?!.*Rack)(?!.*Scharnier))' \
  --units mm --up y --origin top-center
```

`--keep` matches part labels, not group names in the intermediate OBJ. This regex keeps the
table's own leaf bodies (`SWT-160x100-01 Tischplatte*`, the steel top plate, `SWT-090-01
Fuss_Standard<Wie bearbeitet>*`, the four legs, `SWT-160-02 Versteifungsrippe*`, the reinforcement
ribs, and `K0742_008016X2000 Gelenkfuss*`, the adjustable leveling feet that give the table its
nominal 900 mm height) and drops everything else: the `SWT-160x100x090-00 Schweisstisch` assembly
root, the `SWT-090-02 Fuss Rack_Standard<Wie bearbeitet>*` and `SWT-000-01 Scharnier Rack` parts
that carry the table's own prefix but belong to the accessory rack, the `SWR-*` rack bodies, the
`SWZ-*` stops and angles, the `DIN`/`ISO`/`hex`/`prevailing` fasteners, and the datum planes. Add
`--linear-deflection-mm 1.0` if the default 0.5 mm tessellation produces more than a million
triangles.

Check `per_part_bounds_m` in the JSON report before trusting this regex on a different STEP
export. On this table five parts carry the `K0742_008016X2000 Gelenkfuss*` label. Four sit under
the table's legs and bottom out at exactly -0.9 m. The fifth, `Gelenkfuss004`, sits at the
accessory rack's foot position and reaches -0.915 m, a labelling defect in the source model, so
the regex's `(?!.*004)` drops it. The result is 49 parts, 143,776 triangles, bounds
`[-0.8, -0.5, -0.9]` to `[0.8, 0.5, 0.0]`, a 1.6 x 1.0 x 0.9 m table with its top face at z 0.

The command prints one JSON report on stdout: `bounds_m`, `dims_m`, `triangles`, `groups`,
`per_part_bounds_m` (each kept part's own bounds, for spotting a mislabelled or contaminating
part), `output`, `bytes`, `path_used`. Record it.

### 3. Copy to the VM

Copy the USD under `$VIAM_MODULE_DATA/assets/table/` on the GPU machine, so a `usd_path` of
`data://table/SWT-160x100x090-00.usd` resolves through `isaac_module.assets.resolve_asset`.
Under viam-server that directory is
`/root/.viam/module-data/<part-id>/viam_isaac-sim-devin/`, not run.sh's `/opt/viam-isaac-sim`
default, which applies only when nothing sets the variable. A wrong target shows in the module
log as `usd not found: <resolved path>` and the world boots without the prop.

### 4. Reference it as a visual prop

```json
{
  "name": "table_source_visual",
  "type": "visual",
  "usd_path": "data://table/SWT-160x100x090-00.usd",
  "fit": {"collider": "table_source"},
  "position": [-1.2, 0, 0.75]
}
```

`position` is the collider's own top-centre, since the mesh's origin is already at its top-face
centre. `fit.collider` scales the mesh to the named cube prop's `size × scale` at boot.
