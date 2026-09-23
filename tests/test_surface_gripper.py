"""Isaac's surface gripper authored through fakes, and the smoke command's
arguments checked before anything reaches a stage."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from isaac_module.asset_catalog import CUP_APPROACH_GAP_MM
from isaac_module.spatial import quat_rotate
from isaac_module.surface_gripper import (
    ATTR_CLEARANCE_OFFSET,
    ATTR_COAXIAL_FORCE_LIMIT,
    ATTR_FORWARD_AXIS,
    ATTR_MAX_GRIP_DISTANCE,
    ATTR_RETRY_INTERVAL,
    ATTR_SHEAR_FORCE_LIMIT,
    DEFAULT_MAX_GRIP_DISTANCE_MM,
    DEFAULT_SHEAR_FORCE_LIMIT_N,
    LOCKED_AXES,
    PLUGIN_COAXIAL_CHECK_OFF_N,
    REL_ATTACHMENT_POINTS,
    SMOKE_STEPS,
    AttachmentRig,
    CupCompliance,
    GripperLimits,
    LoadWindow,
    SmokeSpec,
    SurfaceGripperSmokeRig,
    anchor_relative_offsets,
    attachment_joint_paths,
    author_attachment_joint,
    author_attachment_rig,
    author_ghost_anchor,
    author_surface_gripper,
    cup_pull_load_n,
    cup_vacuum_force_n,
    parse_smoke_spec,
    status_name,
)

# -- fakes, the shape of the pxr calls the authoring makes --


class _Attr:
    def __init__(self, value=None) -> None:
        self.value = value

    def Set(self, value):
        self.value = value

    def Get(self):
        return self.value

    def __bool__(self) -> bool:
        return True


class _Rel:
    def __init__(self) -> None:
        self.targets: list = []

    def SetTargets(self, targets):
        self.targets = list(targets)


class _Prim:
    def __init__(self, path: str) -> None:
        self.path = path
        self.attrs: dict[str, _Attr] = {}
        self.rels: dict[str, _Rel] = {}
        self.applied: list[str] = []
        self.drives: dict[str, dict[str, object]] = {}

    def GetAttribute(self, name):
        return self.attrs.get(name)

    def CreateAttribute(self, name, type_name):
        attr = _Attr()
        attr.type_name = type_name  # type: ignore[attr-defined]
        self.attrs[name] = attr
        return attr

    def GetRelationship(self, name):
        return self.rels.setdefault(name, _Rel())

    def IsValid(self) -> bool:
        return True


class _Joint:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim
        self.body0 = _Rel()
        self.body1 = _Rel()
        self.limits: dict[str, tuple[_Attr, _Attr]] = {}

    def CreateBody0Rel(self):
        return self.body0

    def CreateBody1Rel(self):
        return self.body1

    def _attr(self, name):
        return self.prim.attrs.setdefault(name, _Attr())

    def CreateLocalPos0Attr(self):
        return self._attr("local_pos0")

    def CreateLocalRot0Attr(self):
        return self._attr("local_rot0")

    def CreateLocalPos1Attr(self):
        return self._attr("local_pos1")

    def CreateLocalRot1Attr(self):
        return self._attr("local_rot1")

    def CreateExcludeFromArticulationAttr(self):
        return self._attr("exclude_from_articulation")

    def GetPrim(self):
        return self.prim


class _Limit:
    def __init__(self, prim: _Prim, axis: str) -> None:
        self.low = _Attr()
        self.high = _Attr()
        prim.attrs[f"limit:{axis}:low"] = self.low
        prim.attrs[f"limit:{axis}:high"] = self.high

    def CreateLowAttr(self):
        return self.low

    def CreateHighAttr(self):
        return self.high


class _Drive:
    def __init__(self, prim: _Prim, token: str) -> None:
        self.prim = prim
        self.token = token

    def _record(self, key: str, value):
        self.prim.drives.setdefault(self.token, {})[key] = value

    def CreateTypeAttr(self, value):
        self._record("type", value)

    def CreateStiffnessAttr(self, value):
        self._record("stiffness", value)

    def CreateDampingAttr(self, value):
        self._record("damping", value)

    def CreateTargetPositionAttr(self, value):
        self._record("target_position", value)


class _Body:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def CreateKinematicEnabledAttr(self, value):
        self.prim.attrs["kinematic"] = _Attr(value)


class _Mass:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def CreateMassAttr(self, value):
        self.prim.attrs["mass"] = _Attr(value)

    def CreateDiagonalInertiaAttr(self, value):
        self.prim.attrs["inertia"] = _Attr(value)


class _PhysxBody:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def CreateDisableGravityAttr(self, value):
        self.prim.attrs["disable_gravity"] = _Attr(value)


class _SoftLimit:
    def __init__(self, prim: _Prim, axis: str) -> None:
        self.stiffness = _Attr()
        self.damping = _Attr()
        prim.attrs[f"physxLimit:{axis}:stiffness"] = self.stiffness
        prim.attrs[f"physxLimit:{axis}:damping"] = self.damping

    def CreateStiffnessAttr(self, value):
        self.stiffness.Set(value)
        return self.stiffness

    def CreateDampingAttr(self, value):
        self.damping.Set(value)
        return self.damping


class _PhysxSchema:
    class PhysxRigidBodyAPI:
        @staticmethod
        def Apply(prim):
            prim.applied.append("PhysxRigidBodyAPI")
            return _PhysxBody(prim)

    class PhysxLimitAPI:
        @staticmethod
        def Apply(prim, axis):
            return _SoftLimit(prim, axis)


class _Stage:
    def __init__(self) -> None:
        self.prims: dict[str, _Prim] = {}
        self.defined: list[tuple[str, str]] = []
        self.joints: dict[str, _Joint] = {}
        self.removed: list[str] = []

    def prim(self, path: str) -> _Prim:
        return self.prims.setdefault(path, _Prim(path))

    def DefinePrim(self, path, type_name):
        self.defined.append((path, type_name))
        return self.prim(path)

    def GetPrimAtPath(self, path):
        prim = self.prims.get(path)
        return prim if prim is not None else SimpleNamespace(IsValid=lambda: False)

    def RemovePrim(self, path):
        self.removed.append(path)
        for known in [p for p in self.prims if p == path or p.startswith(path + "/")]:
            del self.prims[known]


class _UsdPhysics:
    class Joint:
        @staticmethod
        def Define(stage, path):
            joint = _Joint(stage.prim(path))
            stage.joints[path] = joint
            return joint

    class LimitAPI:
        @staticmethod
        def Apply(prim, axis):
            return _Limit(prim, axis)

    class RigidBodyAPI:
        @staticmethod
        def Apply(prim):
            prim.applied.append("RigidBodyAPI")
            return _Body(prim)

    class MassAPI:
        @staticmethod
        def Apply(prim):
            prim.applied.append("MassAPI")
            return _Mass(prim)

    class DriveAPI:
        @staticmethod
        def Apply(prim, token):
            return _Drive(prim, token)


class _Xform:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim
        self.ops: list[tuple[str, object]] = []

    def GetPrim(self):
        return self.prim


class _Xformable:
    def __init__(self, xform: _Xform) -> None:
        self.xform = xform

    def ClearXformOpOrder(self):
        self.xform.ops.clear()

    def _op(self, kind):
        xform = self.xform

        class Op:
            def Set(self, value):
                xform.ops.append((kind, value))
                xform.prim.attrs[f"xform:{kind}"] = _Attr(value)

        return Op()

    def AddTranslateOp(self):
        return self._op("translate")

    def AddOrientOp(self):
        return self._op("orient")


class _UsdGeom:
    class Xform:
        @staticmethod
        def Define(stage, path):
            return _Xform(stage.prim(path))

    Xformable = _Xformable


class _Sdf:
    class Path(str):
        pass

    ValueTypeNames = SimpleNamespace(Token="token", Float="float")


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


class _RobotSchema:
    def __init__(self) -> None:
        self.attachment_api_on: list[str] = []
        self.grippers: list[str] = []

    def ApplyAttachmentPointAPI(self, prim):
        self.attachment_api_on.append(prim.path)

    def CreateSurfaceGripper(self, stage, path):
        self.grippers.append(path)
        return stage.prim(path)


# -- the smoke command's arguments --


def test_parse_smoke_spec_reads_points_and_defaults():
    spec = parse_smoke_spec({"name": "gripper-1", "points_m": [[0, 0, 0.196]]})
    assert spec == SmokeSpec(gripper="gripper-1", points_m=((0.0, 0.0, 0.196),))
    assert spec.max_grip_distance_m == 0.02
    assert spec.retry_interval_s == 1.0


def test_parse_smoke_spec_takes_every_limit():
    spec = parse_smoke_spec(
        {
            "name": "gripper-1",
            "points_m": [[0.07975, 0.04065, 0.196], [-0.07975, -0.04065, 0.196]],
            "clearance_offset_m": 0.001,
            "max_grip_distance_m": 0.03,
            "coaxial_force_limit_n": 177,
            "shear_force_limit_n": 88.5,
            "retry_interval_s": 2,
        }
    )
    assert len(spec.points_m) == 2
    assert spec.coaxial_force_limit_n == 177.0
    assert spec.shear_force_limit_n == 88.5
    assert spec.retry_interval_s == 2.0


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ({"points_m": [[0, 0, 0]]}, "requires 'name'"),
        ({"name": "g"}, "requires 'points_m'"),
        ({"name": "g", "points_m": []}, "requires 'points_m'"),
        ({"name": "g", "points_m": [[0, 0]]}, "every point must be"),
        ({"name": "g", "points_m": [[0, 0, "z"]]}, "every point must be"),
        ({"name": "g", "points_m": [[0, 0, 0]], "max_grip_distance_m": 0}, "max_grip_distance_m"),
        (
            {"name": "g", "points_m": [[0, 0, 0]], "clearance_offset_m": 0.02},
            "clearance .* must be below",
        ),
        ({"name": "g", "points_m": [[0, 0, 0]], "retry_interval_s": -1}, "retry_interval_s"),
    ],
)
def test_parse_smoke_spec_refuses_bad_arguments(command, message):
    with pytest.raises(ValueError, match=message):
        parse_smoke_spec(command)


def test_attachment_layout_helpers():
    assert attachment_joint_paths("/World/S", 2) == [
        "/World/S/AttachmentPoint_0",
        "/World/S/AttachmentPoint_1",
    ]
    points = [(0.08, 0.04, 0.196), (-0.08, 0.04, 0.196), (0.08, -0.04, 0.196)]
    offsets = anchor_relative_offsets(points)
    assert offsets[0] == (0.0, 0.0, 0.0)
    assert offsets[1] == pytest.approx((-0.16, 0.0, 0.0))
    assert offsets[2] == pytest.approx((0.0, -0.08, 0.0))


# -- authoring, through the fakes --


def test_attachment_joint_is_a_locked_excluded_d6_with_the_attachment_api():
    stage = _Stage()
    schema = _RobotSchema()
    author_attachment_joint(
        _UsdPhysics,
        _Sdf,
        _Gf,
        schema,
        stage,
        "/World/S/AttachmentPoint_0",
        "/World/arm/wrist_3_link",
        "/World/S/Anchor",
        (0.0, 0.0, 0.196),
        (0.0, 0.0, 0.0),
        "Z",
        0.003,
    )
    prim = stage.prims["/World/S/AttachmentPoint_0"]
    assert prim.attrs["local_pos0"].value == (0.0, 0.0, 0.196)
    assert prim.attrs["local_rot0"].value == (1.0, 0.0, 0.0, 0.0)
    assert prim.attrs["local_pos1"].value == (0.0, 0.0, 0.0)
    assert prim.attrs["exclude_from_articulation"].value is True
    for axis in LOCKED_AXES:
        assert prim.attrs[f"limit:{axis}:low"].value == 1.0
        assert prim.attrs[f"limit:{axis}:high"].value == -1.0
    assert schema.attachment_api_on == ["/World/S/AttachmentPoint_0"]
    assert prim.attrs[ATTR_FORWARD_AXIS].value == "Z"
    assert prim.attrs[ATTR_CLEARANCE_OFFSET].value == 0.003


def test_attachment_joint_bodies_are_the_link_and_the_anchor():
    stage = _Stage()
    author_attachment_joint(
        _UsdPhysics,
        _Sdf,
        _Gf,
        _RobotSchema(),
        stage,
        "/World/S/AttachmentPoint_0",
        "/World/arm/wrist_3_link",
        "/World/S/Anchor",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    joint = stage.joints["/World/S/AttachmentPoint_0"]
    assert joint.body0.targets == ["/World/arm/wrist_3_link"]
    assert joint.body1.targets == ["/World/S/Anchor"]


def test_attachment_joint_rotation_lands_on_the_joint():
    stage = _Stage()
    rotated = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)
    author_attachment_joint(
        _UsdPhysics,
        _Sdf,
        _Gf,
        _RobotSchema(),
        stage,
        "/World/S/AttachmentPoint_0",
        "/World/arm/wrist_3_link",
        "/World/S/Anchor",
        (0.0, 0.0, 0.196),
        (0.0, 0.0, 0.0),
        local_rot0=rotated,
    )
    prim = stage.prims["/World/S/AttachmentPoint_0"]
    assert prim.attrs["local_rot0"].value == pytest.approx(rotated)
    assert prim.attrs["local_rot1"].value == (1.0, 0.0, 0.0, 0.0)


def test_compliant_joint_frees_the_forward_axis_and_its_lateral_rotations():
    stage = _Stage()
    compliance = CupCompliance()
    author_attachment_joint(
        _UsdPhysics,
        _Sdf,
        _Gf,
        _RobotSchema(),
        stage,
        "/World/S/AttachmentPoint_0",
        "/World/arm/wrist_3_link",
        "/World/S/Anchor",
        (0.0, 0.0, 0.196),
        (0.0, 0.0, 0.0),
        forward_axis="Z",
        compliance=compliance,
        physx_schema=_PhysxSchema,
    )
    prim = stage.prims["/World/S/AttachmentPoint_0"]

    for axis in ("transX", "transY", "rotZ"):
        assert prim.attrs[f"limit:{axis}:low"].value == 1.0
        assert prim.attrs[f"limit:{axis}:high"].value == -1.0

    # the spring is a soft limit, whose force PhysX reports to the plugin, and
    # never a drive, whose force it does not
    assert prim.attrs["limit:transZ:low"].value == pytest.approx(-compliance.dead_band_m)
    assert prim.attrs["limit:transZ:high"].value == pytest.approx(compliance.dead_band_m)
    assert prim.attrs["physxLimit:transZ:stiffness"].value == compliance.stiffness_n_per_m
    assert prim.attrs["physxLimit:transZ:damping"].value == compliance.damping_n_s_per_m
    assert prim.drives == {}
    for axis in ("rotX", "rotY"):
        assert prim.attrs[f"limit:{axis}:low"].value == pytest.approx(-compliance.lateral_limit_deg)
        assert prim.attrs[f"limit:{axis}:high"].value == pytest.approx(compliance.lateral_limit_deg)


def test_compliant_joint_refuses_to_author_a_spring_the_plugin_cannot_read():
    with pytest.raises(ValueError, match="soft limit"):
        author_attachment_joint(
            _UsdPhysics,
            _Sdf,
            _Gf,
            _RobotSchema(),
            _Stage(),
            "/World/S/AttachmentPoint_0",
            "/World/arm/wrist_3_link",
            "/World/S/Anchor",
            (0.0, 0.0, 0.196),
            (0.0, 0.0, 0.0),
            forward_axis="Z",
            compliance=CupCompliance(),
            physx_schema=None,
        )


def test_surface_gripper_prim_points_at_its_joints_and_carries_the_limits():
    stage = _Stage()
    schema = _RobotSchema()
    prim = author_surface_gripper(
        schema,
        _Sdf,
        stage,
        "/World/S/SurfaceGripper",
        ["/World/S/AttachmentPoint_0", "/World/S/AttachmentPoint_1"],
        0.02,
        177.0,
        88.5,
        2.0,
    )
    assert schema.grippers == ["/World/S/SurfaceGripper"]
    assert prim.rels[REL_ATTACHMENT_POINTS].targets == [
        "/World/S/AttachmentPoint_0",
        "/World/S/AttachmentPoint_1",
    ]
    assert prim.attrs[ATTR_MAX_GRIP_DISTANCE].value == 0.02
    assert prim.attrs[ATTR_COAXIAL_FORCE_LIMIT].value == 177.0
    assert prim.attrs[ATTR_SHEAR_FORCE_LIMIT].value == 88.5
    assert prim.attrs[ATTR_RETRY_INTERVAL].value == 2.0


def test_ghost_anchor_is_a_light_shapeless_body_that_gravity_ignores():
    stage = _Stage()
    author_ghost_anchor(
        _UsdGeom,
        _UsdPhysics,
        _PhysxSchema,
        _Gf,
        stage,
        "/World/S/Anchor",
        (0.1, 0.2, 0.3),
        (1.0, 0.0, 0.0, 0.0),
    )
    prim = stage.prims["/World/S/Anchor"]
    assert prim.applied == ["RigidBodyAPI", "MassAPI", "PhysxRigidBodyAPI"]
    assert "kinematic" not in prim.attrs
    assert prim.attrs["mass"].value == 0.1
    assert prim.attrs["disable_gravity"].value is True


def test_ghost_anchor_without_the_physx_schema_still_authors_a_body():
    stage = _Stage()
    author_ghost_anchor(
        _UsdGeom,
        _UsdPhysics,
        None,
        _Gf,
        stage,
        "/World/S/Anchor",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert stage.prims["/World/S/Anchor"].applied == ["RigidBodyAPI", "MassAPI"]


def test_author_attachment_rig_places_the_anchor_and_joints_from_a_moved_link():
    stage = _Stage()
    schema = _RobotSchema()
    modules = {
        "UsdGeom": _UsdGeom,
        "UsdPhysics": _UsdPhysics,
        "PhysxSchema": _PhysxSchema,
        "Gf": _Gf,
        "Sdf": _Sdf,
        "robot_schema": schema,
    }
    body0_pos = (1.0, 2.0, 3.0)
    # 90 degrees about Z, so the link's world pose is not the identity
    body0_quat = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)
    tool_pos = (0.0, 0.0, 0.196)
    tool_quat = (1.0, 0.0, 0.0, 0.0)
    points = [(0.08, 0.04, 0.0), (-0.08, 0.04, 0.0)]

    rig = author_attachment_rig(
        modules,
        stage,
        scope_path="/World/EPickRig",
        body0_path="/World/arm/wrist_3_link",
        body0_world_pose=(body0_pos, body0_quat),
        tool_pose_in_body0=(tool_pos, tool_quat),
        points_tool_m=points,
        clearance_offset_m=0.005,
        limits=GripperLimits(),
        compliance=None,
    )

    assert isinstance(rig, AttachmentRig)
    assert rig.scope_path == "/World/EPickRig"
    assert rig.anchor_path == "/World/EPickRig/Anchor"
    assert rig.joint_paths == tuple(attachment_joint_paths("/World/EPickRig", 2))
    assert rig.gripper_path == "/World/EPickRig/SurfaceGripper"
    assert ("/World/EPickRig", "Scope") in stage.defined

    first_in_body0 = (
        tool_pos[0] + points[0][0],
        tool_pos[1] + points[0][1],
        tool_pos[2] + points[0][2],
    )
    expected_anchor_pos = tuple(
        body0_pos[i] + quat_rotate(body0_quat, first_in_body0)[i] for i in range(3)
    )
    anchor_prim = stage.prims[rig.anchor_path]
    assert anchor_prim.attrs["xform:translate"].value == pytest.approx(expected_anchor_pos)
    assert anchor_prim.attrs["xform:orient"].value == pytest.approx(body0_quat)

    joint_1_prim = stage.prims[rig.joint_paths[1]]
    expected_local_pos0 = (
        tool_pos[0] + points[1][0],
        tool_pos[1] + points[1][1],
        tool_pos[2] + points[1][2],
    )
    expected_local_pos1 = (
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    )
    assert joint_1_prim.attrs["local_pos0"].value == pytest.approx(expected_local_pos0)
    assert joint_1_prim.attrs["local_pos1"].value == pytest.approx(expected_local_pos1)

    gripper_prim = stage.prims[rig.gripper_path]
    assert gripper_prim.attrs[ATTR_MAX_GRIP_DISTANCE].value == GripperLimits().max_grip_distance_m
    # the plugin's own coaxial check is authored off: the module reads the
    # bellows' stretch itself and averages it over a window (handles/vacuum.py)
    assert gripper_prim.attrs[ATTR_COAXIAL_FORCE_LIMIT].value == PLUGIN_COAXIAL_CHECK_OFF_N
    assert gripper_prim.attrs[ATTR_SHEAR_FORCE_LIMIT].value == GripperLimits().shear_force_limit_n
    assert gripper_prim.attrs[ATTR_RETRY_INTERVAL].value == GripperLimits().retry_interval_s


def test_rated_coaxial_force_and_gripper_limits_defaults():
    expected_n = math.pi * 0.0245**2 * 80.0 * 1.013 * 1000.0
    assert cup_vacuum_force_n() == pytest.approx(expected_n, abs=1e-9)
    assert 150.0 < cup_vacuum_force_n() < 155.0
    assert DEFAULT_SHEAR_FORCE_LIMIT_N == pytest.approx(cup_vacuum_force_n() * 0.5 * 4)
    assert 300.0 < DEFAULT_SHEAR_FORCE_LIMIT_N < 310.0
    assert DEFAULT_MAX_GRIP_DISTANCE_MM > CUP_APPROACH_GAP_MM

    limits = GripperLimits()
    assert limits.max_grip_distance_m == pytest.approx(DEFAULT_MAX_GRIP_DISTANCE_MM / 1000.0)
    assert limits.coaxial_force_limit_n == pytest.approx(cup_vacuum_force_n(), abs=1e-9)
    assert limits.shear_force_limit_n == pytest.approx(DEFAULT_SHEAR_FORCE_LIMIT_N)


def test_cup_pull_load_n_zero_inside_dead_band_and_in_compression():
    compliance = CupCompliance()
    assert cup_pull_load_n(0.0, compliance) == 0.0
    assert cup_pull_load_n(compliance.dead_band_m, compliance) == 0.0
    assert cup_pull_load_n(-0.01, compliance) == 0.0


def test_cup_pull_load_n_past_the_dead_band_is_stiffness_times_stretch():
    compliance = CupCompliance()
    extension_m = compliance.dead_band_m + 0.002
    assert cup_pull_load_n(extension_m, compliance) == pytest.approx(
        compliance.stiffness_n_per_m * 0.002
    )


def test_load_window_steps_from_window_over_dt_with_a_floor_of_one():
    assert LoadWindow(0.1, 1 / 120).steps == 12
    assert LoadWindow(0.1, 10.0).steps == 1


def test_load_window_mean_is_zero_until_filled_then_rolls():
    window = LoadWindow(0.1, 1 / 120)
    for _ in range(window.steps - 1):
        assert window.push(10.0) == 0.0
    assert window.filled is False

    assert window.push(10.0) == pytest.approx(10.0)
    assert window.filled is True

    # the window rolls: the oldest sample drops off as a new one arrives
    for _ in range(window.steps - 1):
        window.push(0.0)
    assert window.push(0.0) == pytest.approx(0.0)


def test_load_window_reset_clears_the_samples():
    window = LoadWindow(0.1, 1 / 120)
    for _ in range(window.steps):
        window.push(10.0)
    assert window.filled is True

    window.reset()

    assert window.filled is False
    assert window.mean_n == 0.0


def test_status_name_normalises_every_shape_the_interface_uses():
    assert status_name("Closed") == "Closed"
    assert status_name(SimpleNamespace(name="Closing")) == "Closing"
    assert status_name(0) == "Open"
    assert status_name(2) == "Closed"
    assert status_name(7) == "7"


# -- the rig's own guards, no Isaac needed --


def test_rig_refuses_steps_before_authoring_and_unknown_steps():
    rig = SurfaceGripperSmokeRig(sim=None)
    with pytest.raises(ValueError, match="nothing authored yet"):
        rig.status()
    with pytest.raises(ValueError, match="not wired yet"):
        rig._spec = SmokeSpec("g", ((0.0, 0.0, 0.0),))
        rig.close()
    with pytest.raises(ValueError, match="unknown step"):
        rig.step("weld")
    assert rig.cleanup() == {"ok": True, "removed": []}
    assert SMOKE_STEPS == ("author", "wire", "restart", "close", "open", "status", "cleanup")


def test_only_the_shear_carrying_joint_locks_the_lateral_axes():
    """Four cups each locking the box sideways and against turning are
    over-constrained, and the plugin read PhysX's internal fight as load.
    One cup carries the shear, the others are springs along the cup axis."""
    stage = _Stage()
    compliance = CupCompliance()
    for index in (0, 1):
        author_attachment_joint(
            _UsdPhysics,
            _Sdf,
            _Gf,
            _RobotSchema(),
            stage,
            f"/World/S/AttachmentPoint_{index}",
            "/World/arm/wrist_3_link",
            "/World/S/Anchor",
            (0.0, 0.0, 0.196),
            (0.0, 0.0, 0.0),
            forward_axis="Z",
            compliance=compliance,
            carries_shear=index == 0,
            physx_schema=_PhysxSchema,
        )
    shear_joint = stage.prims["/World/S/AttachmentPoint_0"]
    spring_joint = stage.prims["/World/S/AttachmentPoint_1"]
    for axis in ("transX", "transY", "rotZ"):
        assert shear_joint.attrs[f"limit:{axis}:low"].value == 1.0
        assert f"limit:{axis}:low" not in spring_joint.attrs
    for prim in (shear_joint, spring_joint):
        assert prim.attrs["limit:transZ:low"].value == pytest.approx(-compliance.dead_band_m)
        assert prim.attrs["physxLimit:transZ:stiffness"].value == compliance.stiffness_n_per_m
        for axis in ("rotX", "rotY"):
            assert prim.attrs[f"limit:{axis}:high"].value == pytest.approx(
                compliance.lateral_limit_deg
            )
