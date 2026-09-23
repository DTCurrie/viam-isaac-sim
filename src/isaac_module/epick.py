"""The Robotiq EPick's body: the six solids the real driver models plus the
USD authoring of the same shape, both derived from ``EPICK``.

Every length here is in the gripper frame the driver's kinematics use:
millimetres, z = 0 at the TCP (the suction plane), and -z running back toward
the arm flange at z = -196 mm. The render solids and their USD authoring work
in metres on the stage, so every millimetre figure from ``EPICK`` is divided
by 1000 before it reaches a stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .asset_catalog import EPICK
from .prim_paths import prim_name
from .spatial import Quat, Vec3, compose_pose

_MM_PER_M = 1000.0


@dataclass(frozen=True)
class EpickSolid:
    """One of the six solids the EPick is built from, in millimetres."""

    name: str
    shape: Literal["box", "cylinder"]
    center_mm: Vec3
    size_mm: Vec3
    radius_mm: float | None = None
    length_mm: float | None = None


def _cup_offsets_mm() -> tuple[Vec3, ...]:
    return tuple((offset_x, offset_y, 0.0) for offset_x, offset_y in EPICK["cups"]["offsets_mm"])


def epick_collision_solids() -> tuple[EpickSolid, ...]:
    """The six collision boxes the real driver's planner sees, matching
    ``epick_model.json`` link for link."""
    body = EPICK["body"]
    plate = EPICK["plate"]
    cups = EPICK["cups"]
    solids = [
        EpickSolid(
            name="body",
            shape="box",
            center_mm=(0.0, 0.0, body["collision_center_z_mm"]),
            size_mm=body["collision_mm"],
        ),
        EpickSolid(
            name="plate",
            shape="box",
            center_mm=(0.0, 0.0, plate["center_z_mm"]),
            size_mm=plate["size_mm"],
        ),
    ]
    for name, (offset_x, offset_y) in zip(cups["names"], cups["offsets_mm"], strict=True):
        solids.append(
            EpickSolid(
                name=name,
                shape="box",
                center_mm=(offset_x, offset_y, cups["collision_center_z_mm"]),
                size_mm=cups["collision_mm"],
            )
        )
    return tuple(solids)


def epick_render_solids() -> tuple[EpickSolid, ...]:
    """The six solids as drawn: round parts as cylinders, the plate as a
    box, matching the real CAD's render/collision split."""
    body = EPICK["body"]
    plate = EPICK["plate"]
    cups = EPICK["cups"]
    solids = [
        EpickSolid(
            name="body",
            shape="cylinder",
            center_mm=(0.0, 0.0, body["visual_center_z_mm"]),
            size_mm=(body["radius_mm"] * 2.0, body["radius_mm"] * 2.0, body["visual_length_mm"]),
            radius_mm=body["radius_mm"],
            length_mm=body["visual_length_mm"],
        ),
        EpickSolid(
            name="plate",
            shape="box",
            center_mm=(0.0, 0.0, plate["center_z_mm"]),
            size_mm=plate["size_mm"],
        ),
    ]
    for name, (offset_x, offset_y) in zip(cups["names"], cups["offsets_mm"], strict=True):
        solids.append(
            EpickSolid(
                name=name,
                shape="cylinder",
                center_mm=(offset_x, offset_y, cups["visual_center_z_mm"]),
                size_mm=(
                    cups["radius_mm"] * 2.0,
                    cups["radius_mm"] * 2.0,
                    cups["visual_length_mm"],
                ),
                radius_mm=cups["radius_mm"],
                length_mm=cups["visual_length_mm"],
            )
        )
    return tuple(solids)


def attachment_points_tool_m() -> tuple[Vec3, ...]:
    """One attachment point per cup, in the tool frame, metres. z is 0
    because the suction plane is the TCP itself."""
    return tuple(
        (offset_x / _MM_PER_M, offset_y / _MM_PER_M, 0.0)
        for offset_x, offset_y in EPICK["cups"]["offsets_mm"]
    )


def _solid_volume_mm3(solid: EpickSolid) -> float:
    if solid.shape == "cylinder":
        radius = solid.radius_mm if solid.radius_mm is not None else solid.size_mm[0] / 2.0
        length = solid.length_mm if solid.length_mm is not None else solid.size_mm[2]
        return 3.141592653589793 * radius * radius * length
    x, y, z = solid.size_mm
    return x * y * z


def collider_masses_kg(total_kg: float = float(EPICK["mass_kg"])) -> dict[str, float]:
    """The gripper's total mass split over the six colliders, in proportion
    to each collider's volume."""
    solids = epick_collision_solids()
    volumes = {solid.name: _solid_volume_mm3(solid) for solid in solids}
    total_volume = sum(volumes.values())
    return {name: total_kg * volume / total_volume for name, volume in volumes.items()}


def tool_pose_in_link(
    mount_position_m: Vec3,
    mount_orientation_wxyz: Quat,
    tcp_offset_m: float = float(EPICK["tcp_offset_m"]),
) -> tuple[Vec3, Quat]:
    """The tool frame (TCP at its origin, +Z the cup axis) in the arm
    link's frame: the mount pose composed with a translation of
    ``tcp_offset_m`` along the mount's own +Z."""
    return compose_pose(
        mount_position_m, mount_orientation_wxyz, (0.0, 0.0, tcp_offset_m), (1.0, 0.0, 0.0, 0.0)
    )


def _author_render_cylinder(
    usd_geom: Any, gf: Any, stage: Any, path: str, solid: EpickSolid
) -> Any:
    if solid.radius_mm is None or solid.length_mm is None:
        raise ValueError(f"{solid.name} is not a cylinder: it has no radius or length")
    cylinder = usd_geom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(solid.radius_mm / _MM_PER_M)
    cylinder.CreateHeightAttr(solid.length_mm / _MM_PER_M)
    cylinder.CreateAxisAttr("Z")
    xformable = usd_geom.Xformable(cylinder)
    xformable.ClearXformOpOrder()
    cx, cy, cz = (v / _MM_PER_M for v in solid.center_mm)
    xformable.AddTranslateOp().Set(gf.Vec3d(cx, cy, cz))
    return cylinder.GetPrim()


def _author_box(usd_geom: Any, gf: Any, stage: Any, path: str, solid: EpickSolid) -> Any:
    cube = usd_geom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = usd_geom.Xformable(cube)
    xformable.ClearXformOpOrder()
    cx, cy, cz = (v / _MM_PER_M for v in solid.center_mm)
    xformable.AddTranslateOp().Set(gf.Vec3d(cx, cy, cz))
    sx, sy, sz = (v / _MM_PER_M for v in solid.size_mm)
    xformable.AddScaleOp().Set(gf.Vec3f(sx, sy, sz))
    return cube.GetPrim()


def render_prim_path(body_path: str, solid_name: str) -> str:
    """Where a render solid is authored under the body. The solid names are
    the kinematics document's link ids, and USD refuses a hyphen in a prim
    name, so the segment is the name made prim-safe."""
    return f"{body_path}/render/{prim_name(solid_name)}"


def collision_prim_path(body_path: str, solid_name: str) -> str:
    return f"{body_path}/collision/{prim_name(solid_name)}"


def epick_body_prim_paths(body_path: str) -> list[str]:
    """Every prim author_epick_body puts on the stage, in authoring order:
    the body itself, the six render solids, the six collision boxes. A
    caller checks the last one to know whether a body is complete, since an
    authoring that failed part way leaves the Xform behind."""
    paths = [body_path]
    paths += [render_prim_path(body_path, solid.name) for solid in epick_render_solids()]
    paths += [collision_prim_path(body_path, solid.name) for solid in epick_collision_solids()]
    return paths


def author_epick_body(
    usd_geom: Any,
    usd_physics: Any,
    gf: Any,
    stage: Any,
    body_path: str,
    tool_pose_in_link: tuple[Vec3, Quat],
) -> list[str]:
    """Author the EPick as plain USD geometry under ``body_path``: an
    Xform placed at the tool pose (so its own origin is the TCP), visible
    render solids under ``render/`` and invisible, massed collision boxes
    under ``collision/``. Returns every authored prim path."""
    position_m, orientation_wxyz = tool_pose_in_link
    xform = usd_geom.Xform.Define(stage, body_path)
    xformable = usd_geom.Xformable(xform)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(gf.Vec3d(*(float(v) for v in position_m)))
    w, x, y, z = (float(v) for v in orientation_wxyz)
    xformable.AddOrientOp().Set(gf.Quatf(w, gf.Vec3f(x, y, z)))

    paths: list[str] = [body_path]

    for solid in epick_render_solids():
        path = render_prim_path(body_path, solid.name)
        if solid.shape == "cylinder":
            _author_render_cylinder(usd_geom, gf, stage, path, solid)
        else:
            _author_box(usd_geom, gf, stage, path, solid)
        paths.append(path)

    masses = collider_masses_kg()
    for solid in epick_collision_solids():
        path = collision_prim_path(body_path, solid.name)
        prim = _author_box(usd_geom, gf, stage, path, solid)
        usd_physics.CollisionAPI.Apply(prim)
        usd_physics.MassAPI.Apply(prim).CreateMassAttr(masses[solid.name])
        usd_geom.Imageable(prim).MakeInvisible()
        paths.append(path)

    return paths
