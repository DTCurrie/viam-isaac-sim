import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import convert_mesh  # noqa: E402

# A 2 x 1 x 0.5 mm box named SWT-top, plus a small 0.1 mm cube fastener
# named DIN-912-M8, in Y-up millimetres like the real STEP export.
SYNTHETIC_OBJ = """
o SWT-top
v 0 0 0
v 2 0 0
v 2 1 0
v 0 1 0
v 0 0 0.5
v 2 0 0.5
v 2 1 0.5
v 0 1 0.5
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
o DIN-912-M8
v 0 0 0
v 0.1 0 0
v 0.1 0.1 0
f 9 10 11
"""


def test_parse_obj_counts_vertices_and_faces() -> None:
    mesh = convert_mesh.parse_obj(SYNTHETIC_OBJ)

    assert len(mesh.vertices) == 11
    # three quads fan-triangulate to 2 triangles each, plus one triangle
    assert len(mesh.faces) == 7
    assert set(mesh.groups) == {"SWT-top", "DIN-912-M8"}
    assert len(mesh.groups["SWT-top"]) == 6
    assert len(mesh.groups["DIN-912-M8"]) == 1


def test_filter_groups_drops_din_and_reindexes() -> None:
    mesh = convert_mesh.parse_obj(SYNTHETIC_OBJ)

    filtered = convert_mesh.filter_groups(mesh, re.compile("^SWT"))

    assert set(filtered.groups) == {"SWT-top"}
    assert len(filtered.vertices) == 8
    assert len(filtered.faces) == 6
    for face in filtered.faces:
        for vertex_index in face:
            assert 0 <= vertex_index < len(filtered.vertices)


def test_to_metres_divides_by_a_thousand() -> None:
    mesh = convert_mesh.Mesh(vertices=[(1000.0, 500.0, 2000.0)], faces=[], groups={})

    metres = convert_mesh.to_metres(mesh, "mm")

    assert metres.vertices[0] == (1.0, 0.5, 2.0)


def test_to_metres_is_identity_for_metres() -> None:
    mesh = convert_mesh.Mesh(vertices=[(1.0, 0.5, 2.0)], faces=[], groups={})

    metres = convert_mesh.to_metres(mesh, "m")

    assert metres.vertices[0] == (1.0, 0.5, 2.0)


def test_y_up_to_z_up_maps_a_point() -> None:
    mesh = convert_mesh.Mesh(vertices=[(1.0, 2.0, 3.0)], faces=[], groups={})

    z_up = convert_mesh.y_up_to_z_up(mesh)

    assert z_up.vertices[0] == (1.0, -3.0, 2.0)


def test_shift_origin_top_center_centres_and_zeroes_top() -> None:
    mesh = convert_mesh.Mesh(
        vertices=[(0.0, 0.0, 0.0), (2.0, 1.0, 0.5)],
        faces=[],
        groups={},
    )

    shifted = convert_mesh.shift_origin_top_center(mesh)

    min_corner, max_corner = convert_mesh.bounds(shifted)
    assert min_corner[0] == -1.0
    assert max_corner[0] == 1.0
    assert min_corner[1] == -0.5
    assert max_corner[1] == 0.5
    assert max_corner[2] == 0.0
    assert min_corner[2] == -0.5


def test_bounds_and_triangle_count() -> None:
    mesh = convert_mesh.parse_obj(SYNTHETIC_OBJ)

    min_corner, max_corner = convert_mesh.bounds(mesh)

    assert min_corner == (0.0, 0.0, 0.0)
    assert max_corner == (2.0, 1.0, 0.5)
    assert convert_mesh.triangle_count(mesh) == 7


def test_merge_meshes_sums_vertices_faces_and_names_groups() -> None:
    first = convert_mesh.Mesh(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        groups={},
    )
    second = convert_mesh.Mesh(
        vertices=[(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (5.0, 6.0, 5.0), (6.0, 6.0, 5.0)],
        faces=[(0, 1, 2), (1, 3, 2)],
        groups={},
    )

    merged = convert_mesh.merge_meshes([("plate", first), ("legs", second)])

    assert len(merged.vertices) == 3 + 4
    assert len(merged.faces) == 1 + 2
    assert set(merged.groups) == {"plate", "legs"}
    assert merged.groups["plate"] == [0]
    assert merged.groups["legs"] == [1, 2]
    # the second mesh's faces point at vertices offset past the first mesh's
    for face_index in merged.groups["legs"]:
        for vertex_index in merged.faces[face_index]:
            assert vertex_index >= 3


def test_per_part_bounds_reports_each_groups_own_extent() -> None:
    mesh = convert_mesh.parse_obj(SYNTHETIC_OBJ)

    part_bounds = convert_mesh.per_part_bounds(mesh)

    assert set(part_bounds) == {"SWT-top", "DIN-912-M8"}
    swt_min, swt_max = part_bounds["SWT-top"]
    assert swt_min == (0.0, 0.0, 0.0)
    assert swt_max == (2.0, 1.0, 0.5)
    din_min, din_max = part_bounds["DIN-912-M8"]
    assert din_min == (0.0, 0.0, 0.0)
    assert din_max == (0.1, 0.1, 0.0)


def test_freecad_script_contains_deflection_and_keep_regex() -> None:
    script = convert_mesh.freecad_script(
        Path("/tmp/table.step"),
        Path("/tmp/table.obj"),
        "SWT-160x100x090-00",
        linear_deflection_mm=0.5,
        angular_deflection_rad=0.3,
    )

    assert "0.5" in script
    assert "0.3" in script
    assert "SWT-160x100x090-00" in script
    assert "MeshPart.meshFromShape" in script


def test_parse_args_defaults() -> None:
    args = convert_mesh._parse_args(["table.step", "-o", "table.usd"])

    assert args.units == "mm"
    assert args.up == "y"
    assert args.origin == "top-center"
    assert args.keep == convert_mesh.DEFAULT_KEEP
    assert args.steel == convert_mesh.DEFAULT_STEEL
    assert args.freecad == convert_mesh.DEFAULT_FREECAD_BIN
    assert args.linear_deflection_mm == convert_mesh.DEFAULT_LINEAR_DEFLECTION_MM
    assert args.angular_deflection_rad == convert_mesh.DEFAULT_ANGULAR_DEFLECTION_RAD
    assert args.list_parts is False
