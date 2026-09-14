#!/usr/bin/env python3
"""Convert a STEP or OBJ mesh into a visual-only USD.

Usage: `convert_mesh.py <input.step|.obj> -o <out.usd> [--keep REGEX]
[--units mm|m] [--up y|z] [--origin top-center|none] [--steel REGEX]
[--freecad PATH] [--linear-deflection-mm N] [--angular-deflection-rad N]
[--list-parts] [--obj-out PATH]`

A STEP input is first tessellated to OBJ with FreeCAD (`--freecad`, default
`/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd`), keeping only
the part labels matching `--keep`. An OBJ input skips that step. `--list-parts`
runs only the FreeCAD labelling pass and prints every part label with its
face count, so a caller can choose a `--keep` regex before converting.

The mesh is then normalised to metres, Z-up, with its origin at the top-face
centre, and written as a USD with one mesh per source part group and two
`UsdPreviewSurface` materials, `Steel` for groups matching `--steel` and
`DarkFrame` for the rest. Writing the USD requires `pxr` (`usd-core` on PyPI)
in the running interpreter, not in FreeCAD's.

Prints one JSON report to stdout on success: `bounds_m`, `dims_m`,
`triangles`, `groups`, `per_part_bounds_m`, `output`, `bytes`, `path_used`.
Exit 0 on success, non-zero if `pxr` is missing or FreeCAD fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_LINEAR_DEFLECTION_MM = 0.5
DEFAULT_ANGULAR_DEFLECTION_RAD = 0.3
DEFAULT_FREECAD_BIN = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"
DEFAULT_KEEP = ".*"
DEFAULT_STEEL = "Tischplatte"
EXIT_ERROR = 1


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    groups: dict[str, list[int]] = field(default_factory=dict)


def parse_obj(text: str) -> Mesh:
    """Parse Wavefront OBJ text into a `Mesh`.

    Handles `v` vertex lines, `f` face lines with bare or `a/b/c` vertex
    references (1-based, fan-triangulated for polygons past a triangle), and
    `o`/`g` lines that start a new named group. Every other line is ignored.
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    groups: dict[str, list[int]] = {}
    current_group = "default"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        keyword = parts[0]

        if keyword == "v":
            x, y, z = (float(value) for value in parts[1:4])
            vertices.append((x, y, z))
        elif keyword in ("o", "g"):
            current_group = parts[1] if len(parts) > 1 else "default"
            groups.setdefault(current_group, [])
        elif keyword == "f":
            indices = [int(token.split("/")[0]) - 1 for token in parts[1:]]
            for a, b, c in _fan_triangulate(indices):
                groups.setdefault(current_group, []).append(len(faces))
                faces.append((a, b, c))

    return Mesh(vertices=vertices, faces=faces, groups=groups)


def _fan_triangulate(indices: list[int]) -> list[tuple[int, int, int]]:
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def filter_groups(mesh: Mesh, keep: re.Pattern[str]) -> Mesh:
    """Keep only faces in groups matching `keep`, dropping unreferenced vertices and reindexing."""
    kept_groups = {name: faces for name, faces in mesh.groups.items() if keep.search(name)}
    kept_face_indices = sorted({index for faces in kept_groups.values() for index in faces})

    used_vertex_indices = sorted(
        {vertex for face_index in kept_face_indices for vertex in mesh.faces[face_index]}
    )
    old_to_new_vertex = {old: new for new, old in enumerate(used_vertex_indices)}
    new_vertices = [mesh.vertices[old] for old in used_vertex_indices]

    old_to_new_face = {old: new for new, old in enumerate(kept_face_indices)}
    new_faces: list[tuple[int, int, int]] = []
    for old in kept_face_indices:
        a, b, c = mesh.faces[old]
        new_faces.append((old_to_new_vertex[a], old_to_new_vertex[b], old_to_new_vertex[c]))

    new_groups = {
        name: [old_to_new_face[old] for old in faces if old in old_to_new_face]
        for name, faces in kept_groups.items()
    }
    return Mesh(vertices=new_vertices, faces=new_faces, groups=new_groups)


def merge_meshes(named: Sequence[tuple[str, Mesh]]) -> Mesh:
    """Concatenate meshes into one, each `(name, mesh)` pair becoming its own group."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    groups: dict[str, list[int]] = {}

    for name, part in named:
        vertex_offset = len(vertices)
        vertices.extend(part.vertices)
        face_indices_for_group = groups.setdefault(name, [])
        for a, b, c in part.faces:
            face_indices_for_group.append(len(faces))
            faces.append((a + vertex_offset, b + vertex_offset, c + vertex_offset))

    return Mesh(vertices=vertices, faces=faces, groups=groups)


def per_part_bounds(
    mesh: Mesh,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return each group's own axis-aligned bounds, for diagnosing a group outside expectations."""
    result: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for name, face_indices in mesh.groups.items():
        if not face_indices:
            continue
        vertex_indices = {vertex for index in face_indices for vertex in mesh.faces[index]}
        xs = [mesh.vertices[vertex][0] for vertex in vertex_indices]
        ys = [mesh.vertices[vertex][1] for vertex in vertex_indices]
        zs = [mesh.vertices[vertex][2] for vertex in vertex_indices]
        result[name] = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    return result


def to_metres(mesh: Mesh, units: Literal["mm", "m"]) -> Mesh:
    """Rescale vertex coordinates from `units` to metres."""
    if units == "m":
        return Mesh(vertices=list(mesh.vertices), faces=list(mesh.faces), groups=dict(mesh.groups))
    scale = 1.0 / 1000.0
    vertices = [(x * scale, y * scale, z * scale) for x, y, z in mesh.vertices]
    return Mesh(vertices=vertices, faces=list(mesh.faces), groups=dict(mesh.groups))


def y_up_to_z_up(mesh: Mesh) -> Mesh:
    """Convert Y-up coordinates to Z-up: `(x, y, z) -> (x, -z, y)`."""
    vertices = [(x, -z, y) for x, y, z in mesh.vertices]
    return Mesh(vertices=vertices, faces=list(mesh.faces), groups=dict(mesh.groups))


def bounds(mesh: Mesh) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the axis-aligned bounding box as `(min, max)` corners."""
    xs = [v[0] for v in mesh.vertices]
    ys = [v[1] for v in mesh.vertices]
    zs = [v[2] for v in mesh.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def shift_origin_top_center(mesh: Mesh) -> Mesh:
    """Translate so the AABB's x/y centre and its z maximum land on the origin."""
    (min_x, min_y, _min_z), (max_x, max_y, max_z) = bounds(mesh)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    vertices = [(x - center_x, y - center_y, z - max_z) for x, y, z in mesh.vertices]
    return Mesh(vertices=vertices, faces=list(mesh.faces), groups=dict(mesh.groups))


def triangle_count(mesh: Mesh) -> int:
    return len(mesh.faces)


def _safe_filename(label: str) -> str:
    """Replace characters unsafe in a filename with `_`, so a label round-trips through disk."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", label)


def freecad_script(
    step_path: Path,
    obj_dir: Path,
    keep: str,
    linear_deflection_mm: float = DEFAULT_LINEAR_DEFLECTION_MM,
    angular_deflection_rad: float = DEFAULT_ANGULAR_DEFLECTION_RAD,
) -> str:
    """Return FreeCAD console Python that tessellates matching labels into one OBJ per part.

    Imports `step_path`, meshes every object whose `Shape` has faces and whose
    `Label` matches `keep`, and exports each as its own `<index>-<safe
    label>.obj` under `obj_dir` rather than one merged OBJ, since
    `Mesh.export` of a feature list writes a single mesh with no group
    boundaries. Prints a JSON line `{"kept": [...], "skipped": [...]}`, with
    `kept` in the same order the files were written.
    """
    return f"""
import json
import os
import re

import FreeCAD
import Import
import Mesh
import MeshPart

KEEP = re.compile({keep!r})
LINEAR_DEFLECTION_MM = {linear_deflection_mm!r}
ANGULAR_DEFLECTION_RAD = {angular_deflection_rad!r}
OBJ_DIR = {str(obj_dir)!r}

os.makedirs(OBJ_DIR, exist_ok=True)

doc = FreeCAD.newDocument("convert_mesh")
Import.open({str(step_path)!r}, doc.Name)
doc = FreeCAD.getDocument(doc.Name)

kept = []
skipped = []

for obj in doc.Objects:
    shape = getattr(obj, "Shape", None)
    if shape is None or not shape.Faces:
        continue
    if not KEEP.search(obj.Label):
        skipped.append(obj.Label)
        continue
    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=LINEAR_DEFLECTION_MM,
        AngularDeflection=ANGULAR_DEFLECTION_RAD,
        Relative=False,
    )
    feature = doc.addObject("Mesh::Feature", obj.Label)
    feature.Mesh = mesh
    feature.Label = obj.Label
    safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", obj.Label)
    index = len(kept)
    Mesh.export([feature], os.path.join(OBJ_DIR, "%d-%s.obj" % (index, safe_label)))
    kept.append(obj.Label)

print(json.dumps({{"kept": kept, "skipped": skipped}}))
"""


def freecad_list_parts_script(step_path: Path) -> str:
    """Return FreeCAD console Python that prints every part `Label` with its face count."""
    return f"""
import json

import FreeCAD
import Import

doc = FreeCAD.newDocument("convert_mesh_list")
Import.open({str(step_path)!r}, doc.Name)
doc = FreeCAD.getDocument(doc.Name)

parts = []
for obj in doc.Objects:
    shape = getattr(obj, "Shape", None)
    if shape is None or not shape.Faces:
        continue
    parts.append({{"label": obj.Label, "faces": len(shape.Faces)}})

print(json.dumps(parts))
"""


def _extract_json_line(stdout: str) -> str:
    """Return the last line of `stdout` that parses as JSON.

    FreeCAD prints its own version banner to stdout after the script's own
    output finishes flushing, so the JSON report is not reliably the last
    line of the full stream.
    """
    for line in reversed(stdout.strip().splitlines()):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        return line
    raise ValueError("no JSON line found in FreeCAD output")


def run_freecad(freecad_bin: str, script_source: str, workdir: Path) -> str:
    """Write `script_source` under `workdir` and run it with `freecad_bin`, returning stdout."""
    script_path = workdir / "convert_mesh_freecad_script.py"
    script_path.write_text(script_source)
    result = subprocess.run(
        [freecad_bin, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def write_usd(mesh: Mesh, out_path: Path, steel: re.Pattern[str]) -> None:
    """Write `mesh` as a visual USD at `out_path`.

    One `UsdGeom.Mesh` per group under a default-prim `/Table` root, bound to
    a `Steel` or `DarkFrame` `UsdPreviewSurface` under `/Table/Looks`
    depending on whether the group name matches `steel`. Normals are left
    unauthored, since the renderer computes flat per-face normals from the
    unshared triangle topology `filter_groups` already reindexed.
    """
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    except ImportError:
        print("pxr is not installed, run: python -m pip install usd-core", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/Table")
    stage.SetDefaultPrim(root.GetPrim())

    looks_path = Sdf.Path("/Table/Looks")
    steel_material = _define_preview_material(
        stage, looks_path.AppendChild("Steel"), (0.55, 0.56, 0.58), 0.9, 0.35
    )
    dark_frame_material = _define_preview_material(
        stage, looks_path.AppendChild("DarkFrame"), (0.08, 0.08, 0.09), 0.6, 0.55
    )

    min_corner, max_corner = bounds(mesh)

    for group_name, face_indices in mesh.groups.items():
        if not face_indices:
            continue
        group_mesh = filter_groups(mesh, re.compile(f"^{re.escape(group_name)}$"))
        prim_path = looks_path.GetParentPath().AppendChild(_sanitize_prim_name(group_name))
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in group_mesh.vertices])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(group_mesh.faces))
        usd_mesh.CreateFaceVertexIndicesAttr([index for face in group_mesh.faces for index in face])
        usd_mesh.CreateExtentAttr([Gf.Vec3f(*min_corner), Gf.Vec3f(*max_corner)])
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        material = steel_material if steel.search(group_name) else dark_frame_material
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()


def _define_preview_material(
    stage: Any,
    path: Any,
    diffuse_color: tuple[float, float, float],
    metallic: float,
    roughness: float,
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _sanitize_prim_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if sanitized[:1].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized or "part"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    parser.add_argument("--keep", default=DEFAULT_KEEP)
    parser.add_argument("--units", choices=("mm", "m"), default="mm")
    parser.add_argument("--up", choices=("y", "z"), default="y")
    parser.add_argument("--origin", choices=("top-center", "none"), default="top-center")
    parser.add_argument("--steel", default=DEFAULT_STEEL)
    parser.add_argument("--freecad", default=DEFAULT_FREECAD_BIN)
    parser.add_argument("--linear-deflection-mm", type=float, default=DEFAULT_LINEAR_DEFLECTION_MM)
    parser.add_argument(
        "--angular-deflection-rad", type=float, default=DEFAULT_ANGULAR_DEFLECTION_RAD
    )
    parser.add_argument("--list-parts", action="store_true")
    parser.add_argument("--obj-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_parts:
        with tempfile.TemporaryDirectory() as workdir:
            script = freecad_list_parts_script(args.input)
            stdout = run_freecad(args.freecad, script, Path(workdir))
        print(_extract_json_line(stdout))
        return 0

    is_step_input = args.input.suffix.lower() in (".step", ".stp")
    if is_step_input:
        obj_dir = (
            args.obj_out.with_name(args.obj_out.stem) if args.obj_out else Path(tempfile.mkdtemp())
        )
        obj_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as workdir:
            script = freecad_script(
                args.input,
                obj_dir,
                args.keep,
                linear_deflection_mm=args.linear_deflection_mm,
                angular_deflection_rad=args.angular_deflection_rad,
            )
            stdout = run_freecad(args.freecad, script, Path(workdir))
        kept_labels = json.loads(_extract_json_line(stdout))["kept"]
        named_meshes = [
            (label, parse_obj((obj_dir / f"{index}-{_safe_filename(label)}.obj").read_text()))
            for index, label in enumerate(kept_labels)
        ]
        mesh = merge_meshes(named_meshes)
    else:
        mesh = filter_groups(parse_obj(args.input.read_text()), re.compile(args.keep))

    mesh = to_metres(mesh, args.units)
    if args.up == "y":
        mesh = y_up_to_z_up(mesh)
    if args.origin == "top-center":
        mesh = shift_origin_top_center(mesh)

    if args.out is None:
        print("--out/-o is required", file=sys.stderr)
        return EXIT_ERROR

    write_usd(mesh, args.out, re.compile(args.steel))

    min_corner, max_corner = bounds(mesh)
    dims = tuple(max_corner[axis] - min_corner[axis] for axis in range(3))
    report = {
        "bounds_m": [list(min_corner), list(max_corner)],
        "dims_m": list(dims),
        "triangles": triangle_count(mesh),
        "groups": {name: len(faces) for name, faces in mesh.groups.items()},
        "per_part_bounds_m": {
            name: [list(part_min), list(part_max)]
            for name, (part_min, part_max) in per_part_bounds(mesh).items()
        },
        "output": str(args.out),
        "bytes": args.out.stat().st_size,
        "path_used": "freecad+usd-core",
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
