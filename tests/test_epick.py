"""The EPick's six solids and their USD authoring, driven off ``EPICK`` and
checked against ``epick_model.json``, the model the real driver's planner
sees."""

from __future__ import annotations

import json
import math
from pathlib import Path

from isaac_module.asset_catalog import EPICK
from isaac_module.epick import (
    attachment_points_tool_m,
    author_epick_body,
    collider_masses_kg,
    epick_collision_solids,
    epick_render_solids,
    tool_pose_in_link,
)
from isaac_module.prim_paths import prim_name
from isaac_module.spatial import quat_from_axis_angle, quat_rotate

KINEMATICS_PATH = (
    Path(__file__).parent.parent / "src/isaac_module/kinematics_files/epick_model.json"
)


# -- fakes, the shape of the pxr calls the authoring makes --


class _Attr:
    def __init__(self, value=None) -> None:
        self.value = value

    def Set(self, value):
        self.value = value


class _Prim:
    def __init__(self, path: str) -> None:
        self.path = path
        self.applied: list[str] = []
        self.mass: float | None = None
        self.visible = True


class _Mass:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def CreateMassAttr(self, value):
        self.prim.mass = value


class _Imageable:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def MakeInvisible(self):
        self.prim.visible = False


class _UsdPhysics:
    class CollisionAPI:
        @staticmethod
        def Apply(prim):
            prim.applied.append("CollisionAPI")

    class MassAPI:
        @staticmethod
        def Apply(prim):
            prim.applied.append("MassAPI")
            return _Mass(prim)


class _XformOps:
    def __init__(self) -> None:
        self.ops: list[tuple[str, tuple]] = []


class _Cube:
    def __init__(self, prim: _Prim, ops: _XformOps) -> None:
        self.prim = prim
        self.ops = ops
        self.size = None

    def CreateSizeAttr(self, value):
        self.size = value

    def GetPrim(self):
        return self.prim


class _Cylinder:
    def __init__(self, prim: _Prim, ops: _XformOps) -> None:
        self.prim = prim
        self.ops = ops
        self.radius = None
        self.height = None
        self.axis = None

    def CreateRadiusAttr(self, value):
        self.radius = value

    def CreateHeightAttr(self, value):
        self.height = value

    def CreateAxisAttr(self, value):
        self.axis = value

    def GetPrim(self):
        return self.prim


class _Xform:
    def __init__(self, prim: _Prim, ops: _XformOps) -> None:
        self.prim = prim
        self.ops = ops

    def GetPrim(self):
        return self.prim


class _Xformable:
    def __init__(self, thing) -> None:
        self.ops = thing.ops

    def ClearXformOpOrder(self):
        self.ops.ops.clear()

    def _op(self, kind):
        ops = self.ops

        class Op:
            def Set(self, value):
                ops.ops.append((kind, value))

        return Op()

    def AddTranslateOp(self):
        return self._op("translate")

    def AddOrientOp(self):
        return self._op("orient")

    def AddScaleOp(self):
        return self._op("scale")


class _Stage:
    def __init__(self) -> None:
        self.prims: dict[str, _Prim] = {}
        self.ops: dict[str, _XformOps] = {}

    def prim(self, path: str) -> tuple[_Prim, _XformOps]:
        if path not in self.prims:
            self.prims[path] = _Prim(path)
            self.ops[path] = _XformOps()
        return self.prims[path], self.ops[path]


class _UsdGeom:
    class Xform:
        @staticmethod
        def Define(stage, path):
            prim, ops = stage.prim(path)
            return _Xform(prim, ops)

    class Cube:
        @staticmethod
        def Define(stage, path):
            prim, ops = stage.prim(path)
            return _Cube(prim, ops)

    class Cylinder:
        @staticmethod
        def Define(stage, path):
            prim, ops = stage.prim(path)
            return _Cylinder(prim, ops)

    Xformable = _Xformable
    Imageable = _Imageable


class _Gf:
    class Vec3f(tuple):
        def __new__(cls, x, y, z):
            return super().__new__(cls, (x, y, z))

    class Vec3d(tuple):
        def __new__(cls, x, y, z):
            return super().__new__(cls, (x, y, z))

    class Quatf(tuple):
        def __new__(cls, w, xyz):
            return super().__new__(cls, (w, *xyz))


# -- collision solids match epick_model.json --


def test_collision_solids_match_kinematics_file():
    solids = epick_collision_solids()
    assert len(solids) == 6

    with open(KINEMATICS_PATH) as handle:
        model = json.load(handle)
    links_by_id = {link["id"]: link for link in model["links"]}
    assert len(links_by_id) == 6

    for solid in solids:
        link = links_by_id[solid.name]
        geometry = link["geometry"]
        expected_size = (geometry["x"], geometry["y"], geometry["z"])
        expected_center = (
            geometry["translation"]["x"],
            geometry["translation"]["y"],
            geometry["translation"]["z"],
        )
        for got, want in zip(solid.size_mm, expected_size, strict=True):
            assert abs(got - want) < 1e-9
        for got, want in zip(solid.center_mm, expected_center, strict=True):
            assert abs(got - want) < 1e-9


# -- render solids --


def test_render_solids_body_and_plate():
    solids = epick_render_solids()
    assert len(solids) == 6

    body = next(s for s in solids if s.name == "body")
    assert abs(body.radius_mm - 35.5) < 1e-9
    assert abs(body.length_mm - 129.0) < 1e-9
    assert abs(body.center_mm[2] - (-134.5)) < 1e-9


def test_render_and_collision_cups_at_expected_offsets_and_ends():
    render_by_name = {s.name: s for s in epick_render_solids()}
    collision_by_name = {s.name: s for s in epick_collision_solids()}

    offsets = EPICK["cups"]["offsets_mm"]
    names = EPICK["cups"]["names"]
    assert len(names) == 4

    for name, (offset_x, offset_y) in zip(names, offsets, strict=True):
        render = render_by_name[name]
        assert abs(render.radius_mm - 24.5) < 1e-9
        assert abs(render.length_mm - 60.0) < 1e-9
        assert abs(render.center_mm[2] - (-40.0)) < 1e-9
        assert abs(render.center_mm[0] - offset_x) < 1e-9
        assert abs(render.center_mm[1] - offset_y) < 1e-9
        render_end_z = render.center_mm[2] + render.length_mm / 2.0
        assert abs(render_end_z - (-10.0)) < 1e-9

        collision = collision_by_name[name]
        collision_end_z = collision.center_mm[2] + collision.size_mm[2] / 2.0
        assert abs(collision_end_z - (-26.0)) < 1e-9


# -- attachment points --


def test_attachment_points_tool_m_exact():
    points = attachment_points_tool_m()
    expected = (
        (0.07975, 0.04065, 0.0),
        (0.07975, -0.04065, 0.0),
        (-0.07975, 0.04065, 0.0),
        (-0.07975, -0.04065, 0.0),
    )
    assert len(points) == 4
    for got_point, want_point in zip(points, expected, strict=True):
        for got, want in zip(got_point, want_point, strict=True):
            assert abs(got - want) < 1e-12


# -- collider mass split --


def test_collider_masses_kg_sums_and_body_largest():
    masses = collider_masses_kg()
    assert set(masses.keys()) == {
        "body",
        "plate",
        "cup-xp-yp",
        "cup-xp-yn",
        "cup-xn-yp",
        "cup-xn-yn",
    }
    assert abs(sum(masses.values()) - 0.706) < 1e-9
    for share in masses.values():
        assert share > 0.0
    assert masses["body"] == max(masses.values())


# -- tool pose in link --


def test_tool_pose_in_link_identity_mount():
    position, orientation = tool_pose_in_link((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    assert abs(position[0] - 0.0) < 1e-9
    assert abs(position[1] - 0.0) < 1e-9
    assert abs(position[2] - 0.196) < 1e-9
    assert abs(orientation[0] - 1.0) < 1e-9
    assert abs(orientation[1] - 0.0) < 1e-9
    assert abs(orientation[2] - 0.0) < 1e-9
    assert abs(orientation[3] - 0.0) < 1e-9


def test_tool_pose_in_link_rotated_mount():
    mount_quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi / 2.0)
    position, _orientation = tool_pose_in_link((0.0, 0.0, 0.0), mount_quat)
    expected = quat_rotate(mount_quat, (0.0, 0.0, 0.196))
    for got, want in zip(position, expected, strict=True):
        assert abs(got - want) < 1e-9


# -- authoring on fakes --


def test_author_epick_body_places_xform_at_tool_pose():
    stage = _Stage()
    tool_pose = ((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
    author_epick_body(_UsdGeom, _UsdPhysics, _Gf, stage, "/World/Arm/EPick", tool_pose)

    body_ops = stage.ops["/World/Arm/EPick"].ops
    translate = next(v for k, v in body_ops if k == "translate")
    orient = next(v for k, v in body_ops if k == "orient")
    for got, want in zip(translate, tool_pose[0], strict=True):
        assert abs(got - want) < 1e-9
    for got, want in zip(orient, tool_pose[1], strict=True):
        assert abs(got - want) < 1e-9


def test_author_epick_body_render_prims_have_no_collision_api():
    stage = _Stage()
    tool_pose = ((0.0, 0.0, 0.196), (1.0, 0.0, 0.0, 0.0))
    paths = author_epick_body(_UsdGeom, _UsdPhysics, _Gf, stage, "/World/Arm/EPick", tool_pose)

    render_paths = [p for p in paths if p.startswith("/World/Arm/EPick/render/")]
    assert len(render_paths) == 6
    for path in render_paths:
        prim = stage.prims[path]
        assert "CollisionAPI" not in prim.applied


def test_author_epick_body_collision_prims_massed_and_invisible():
    stage = _Stage()
    tool_pose = ((0.0, 0.0, 0.196), (1.0, 0.0, 0.0, 0.0))
    paths = author_epick_body(_UsdGeom, _UsdPhysics, _Gf, stage, "/World/Arm/EPick", tool_pose)

    collision_paths = [p for p in paths if p.startswith("/World/Arm/EPick/collision/")]
    assert len(collision_paths) == 6

    masses = collider_masses_kg()
    for path in collision_paths:
        segment = path.rsplit("/", 1)[-1]
        name = next(
            solid.name for solid in epick_collision_solids() if prim_name(solid.name) == segment
        )
        prim = stage.prims[path]
        assert "CollisionAPI" in prim.applied
        assert "MassAPI" in prim.applied
        assert abs(prim.mass - masses[name]) < 1e-9
        assert prim.visible is False


def test_author_epick_body_collision_body_translate_and_scale():
    stage = _Stage()
    tool_pose = ((0.0, 0.0, 0.196), (1.0, 0.0, 0.0, 0.0))
    author_epick_body(_UsdGeom, _UsdPhysics, _Gf, stage, "/World/Arm/EPick", tool_pose)

    body_ops = stage.ops["/World/Arm/EPick/collision/body"].ops
    translate = next(v for k, v in body_ops if k == "translate")
    scale = next(v for k, v in body_ops if k == "scale")

    expected_translate = (0.0, 0.0, -0.133)
    expected_scale = (0.071, 0.071, 0.126)
    for got, want in zip(translate, expected_translate, strict=True):
        assert abs(got - want) < 1e-9
    for got, want in zip(scale, expected_scale, strict=True):
        assert abs(got - want) < 1e-9


def test_every_authored_prim_path_is_usd_safe():
    """The solid names are the kinematics document's link ids, which carry
    hyphens, and USD refuses a hyphen in a prim name: the first GPU build of
    the gripper on 2026-09-22 died on `cup-xp-yp`. Every segment has to be a
    legal identifier, and the path list has to be the one the authoring
    actually produces."""
    import re

    from isaac_module.epick import epick_body_prim_paths

    paths = epick_body_prim_paths("/World/arm_1/wrist_3_link/EPick")
    assert len(paths) == 13
    for path in paths:
        for segment in path.strip("/").split("/"):
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", segment), path
    assert "/World/arm_1/wrist_3_link/EPick/collision/cup_xn_yn" == paths[-1]
