"""SimManager.materialise_components: workcell_scenery's prop dicts turned
into stage prims, on the fake-Isaac path (no GPU, no real Isaac import).

Mirrors the fake-namespace pattern test_world_handle.py already uses for
_spawn_prop: a plain-python stand-in for isaacsim's SingleXFormPrim,
DynamicCuboid/FixedCuboid and the pxr schemas, so the real Isaac cube and
mesh paths run under test with no GPU and no Isaac import.
"""

from __future__ import annotations

import threading
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from isaac_module.epick import epick_body_prim_paths
from isaac_module.handles.arm import IsaacArmHandle
from isaac_module.handles.vacuum import DEFAULT_GRAB_DELAY_MS
from isaac_module.sim_manager import ComponentScenery, SimManager
from isaac_module.surface_gripper import (
    DEFAULT_COAXIAL_FORCE_LIMIT_N,
    DEFAULT_CUP_DAMPING_N_S_PER_M,
    DEFAULT_CUP_STIFFNESS_N_PER_M,
    DEFAULT_MAX_GRIP_DISTANCE_MM,
    DEFAULT_RETRY_INTERVAL_S,
    DEFAULT_SHEAR_FORCE_LIMIT_N,
    AttachmentRig,
)


class _FakeXForm:
    """Shared pose store keyed by prim_path, standing in for
    SingleXFormPrim and the Dynamic/FixedCuboid constructors."""

    _STORE: ClassVar[dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def __init__(self, prim_path: str, name: str = "", position=None, orientation=None, **kwargs):
        self.prim_path = prim_path
        self.name = name
        if position is not None or orientation is not None:
            self.set_world_pose(position=position, orientation=orientation)
        elif prim_path not in self._STORE:
            self._STORE[prim_path] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    def set_world_pose(self, position=None, orientation=None) -> None:
        pos, quat = self._STORE.get(self.prim_path, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        if position is not None:
            pos = np.array([float(v) for v in position])
        if orientation is not None:
            quat = np.array([float(v) for v in orientation])
        self._STORE[self.prim_path] = (pos, quat)

    def get_world_pose(self):
        return self._STORE[self.prim_path]

    def set_local_scale(self, scale) -> None:
        pass


class _FakeScene:
    def __init__(self) -> None:
        self._objects: dict[str, object] = {}

    def add(self, obj) -> None:
        name = getattr(obj, "name", None)
        if name:
            self._objects[name] = obj

    def get_object(self, name):
        return self._objects.get(name)


class _FakeWorld:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self._callbacks: dict[str, Any] = {}

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def add_physics_callback(self, name: str, fn: Any) -> None:
        self._callbacks[name] = fn

    def remove_physics_callback(self, name: str) -> None:
        self._callbacks.pop(name, None)

    def physics_callback_exists(self, name: str) -> bool:
        return name in self._callbacks


class _FakePrim:
    def __init__(self) -> None:
        self.removed_apis: list[str] = []

    def RemoveAPI(self, schema: str) -> None:
        self.removed_apis.append(schema)


class _FakeUsdPhysics:
    CollisionAPI = "CollisionAPI"
    RigidBodyAPI = "RigidBodyAPI"


class _FakeIsaacNamespace:
    DynamicCuboid = _FakeXForm
    FixedCuboid = _FakeXForm
    PhysicsMaterial = None
    PhysxSchema = None
    UsdPhysics = _FakeUsdPhysics

    def __init__(self) -> None:
        self._prims: dict[str, _FakePrim] = {}

    def SingleXFormPrim(self, prim_path: str):
        return _FakeXForm(prim_path)

    @staticmethod
    def add_reference_to_stage(usd_path: str, prim_path: str) -> None:
        pass

    def get_prim_at_path(self, prim_path: str) -> _FakePrim:
        return self._prims.setdefault(prim_path, _FakePrim())


def _manager() -> SimManager:
    _FakeXForm._STORE.clear()
    manager = SimManager()
    manager.mock = False
    manager.world = _FakeWorld()
    manager._isaac = _FakeIsaacNamespace()
    manager._sim_thread_id = threading.get_ident()
    return manager


def _recording_spawn(manager: SimManager) -> list[str]:
    """Wrap _spawn_prop so a test can see spawn order without depending on
    which registry (or none) a prop ends up in."""
    order: list[str] = []
    original = manager._spawn_prop

    def _spawn(prop: dict[str, Any]) -> None:
        order.append(str(prop["name"]))
        original(prop)

    manager._spawn_prop = _spawn
    return order


def _box_visual(label: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict[str, Any]:
    return {
        "type": "box",
        "label": label,
        "pose": {"x": x * 1000.0, "y": y * 1000.0, "z": z * 1000.0, "o_z": 1.0, "theta": 0.0},
        "dims_mm": {"x": 100.0, "y": 100.0, "z": 100.0},
    }


# ----------------------------------------------------------------------
# a render prop never becomes a collider
# ----------------------------------------------------------------------


def test_render_prop_has_no_collider_and_is_absent_from_prop_specs():
    manager = _manager()
    scenery = {
        "cabinet": ComponentScenery(
            model="hmi-cabinet",
            visuals={"visuals": [_box_visual("body")]},
            geometries=[],
            attrs={},
        )
    }

    manager.materialise_components(scenery)

    (name,) = manager._isaac._prims  # one render cube spawned, one prim touched
    assert name == "/World/cabinet_body"
    prim = manager._isaac._prims[name]
    assert prim.removed_apis == ["CollisionAPI", "RigidBodyAPI"]
    assert "cabinet-body" not in manager._prop_specs
    assert "cabinet-body" not in manager._visual_props


# ----------------------------------------------------------------------
# a collider spawns before the render props of the same component
# ----------------------------------------------------------------------


def test_collider_spawns_before_render_props_of_the_same_component():
    manager = _manager()
    order = _recording_spawn(manager)
    scenery = {
        "fence-left": ComponentScenery(
            model="safety-fence",
            visuals={"visuals": [_box_visual("post-a"), _box_visual("post-b")]},
            geometries=[{"label": "fence-body", "box_dims_mm": {"x": 1200, "y": 100, "z": 1200}}],
            attrs={},
        )
    }

    manager.materialise_components(scenery)

    assert order == ["fence-left-fence-body", "fence-left-post-a", "fence-left-post-b"]
    assert "fence_left_fence_body" in manager._prop_specs
    assert "fence_left_post_a" not in manager._prop_specs


# ----------------------------------------------------------------------
# a component's frame composes onto its primitives' local poses
# ----------------------------------------------------------------------


def test_component_frame_composes_onto_primitive_local_pose():
    manager = _manager()
    scenery = {
        "robot-pedestal": ComponentScenery(
            model="robot-pedestal",
            visuals={"visuals": []},
            geometries=[],
            attrs={"height_mm": 300.0, "diameter_mm": 220.0},
            frame_position_m=(1.0, 2.0, 0.5),
            frame_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    }

    manager.materialise_components(scenery)

    prim_path = "/World/robot_pedestal_robot_pedestal_collider"
    position, _orientation = _FakeXForm._STORE[prim_path]
    # local pose is (0, 0, height_m / 2) = (0, 0, 0.15); identity frame
    # orientation, so the frame's translation adds straight through.
    assert position == pytest.approx([1.0, 2.0, 0.65])


# ----------------------------------------------------------------------
# _create_vacuum_gripper_isaac authors the EPick body, the attachment rig
# and a world reset, then hands the handle the gripper interface
# ----------------------------------------------------------------------


class _FakeArticulation:
    dof_names: ClassVar[list[str]] = []


class _FakeStage:
    pass


class _FakePrimHandle:
    """get_prim_at_path's stand-in: IsValid() says whether this path was
    already authored, the way a re-attach after release_handle finds one."""

    def __init__(self, valid: bool = False) -> None:
        self._valid = valid

    def GetStage(self):
        return _FakeStage()

    def IsValid(self) -> bool:
        return self._valid


def _fake_get_prim_at_path(existing: set[str]):
    def get_prim_at_path(path: str) -> _FakePrimHandle:
        return _FakePrimHandle(valid=path in existing)

    return get_prim_at_path


@pytest.fixture
def fake_surface_gripper(monkeypatch):
    """Stands in for compat.import_surface_gripper and the two authoring
    functions sim_manager calls, so _create_vacuum_gripper_isaac runs with
    no real Isaac import and every authored prim recorded for assertions."""
    sentinel_iface = object()
    fake_modules: dict[str, Any] = {
        "report": {},
        "surface_gripper": types.SimpleNamespace(
            acquire_surface_gripper_interface=lambda: sentinel_iface
        ),
        "robot_schema": types.SimpleNamespace(),
        "Gf": types.SimpleNamespace(),
        "Sdf": types.SimpleNamespace(),
        "UsdGeom": types.SimpleNamespace(),
        "UsdPhysics": types.SimpleNamespace(),
        "PhysxSchema": None,
    }
    monkeypatch.setattr("isaac_module.sim_manager.import_surface_gripper", lambda: fake_modules)

    epick_calls: list[dict[str, Any]] = []

    def fake_author_epick_body(usd_geom, usd_physics, gf, stage, body_path, tool_pose):
        epick_calls.append({"body_path": body_path, "tool_pose": tool_pose})
        return [body_path]

    monkeypatch.setattr("isaac_module.sim_manager.author_epick_body", fake_author_epick_body)

    rig_calls: list[dict[str, Any]] = []

    def fake_author_attachment_rig(modules, stage, **kwargs):
        rig_calls.append(kwargs)
        scope = kwargs["scope_path"]
        joint_count = len(kwargs["points_tool_m"])
        return AttachmentRig(
            scope_path=scope,
            anchor_path=f"{scope}/Anchor",
            joint_paths=tuple(f"{scope}/AttachmentPoint_{i}" for i in range(joint_count)),
            gripper_path=f"{scope}/SurfaceGripper",
        )

    monkeypatch.setattr(
        "isaac_module.sim_manager.author_attachment_rig", fake_author_attachment_rig
    )

    return types.SimpleNamespace(
        interface=sentinel_iface, epick_calls=epick_calls, rig_calls=rig_calls
    )


def _reset_counter(manager: SimManager) -> list[None]:
    calls: list[None] = []
    manager._reset_world = lambda: calls.append(None)
    return calls


def test_vacuum_gripper_isaac_authors_body_and_rig_with_defaults(fake_surface_gripper):
    manager = _manager()
    manager._isaac.get_prim_at_path = _fake_get_prim_at_path(set())
    reset_calls = _reset_counter(manager)
    arm = IsaacArmHandle(manager, _FakeArticulation(), None, prim_path="/World/Arm")

    handle = manager._create_vacuum_gripper_isaac("vacuum-1", {"arm": "arm-1"}, arm)

    (epick_call,) = fake_surface_gripper.epick_calls
    assert epick_call["body_path"] == "/World/Arm/wrist_3_link/EPick"
    position, orientation = epick_call["tool_pose"]
    assert position == pytest.approx((0.0, 0.0, 0.196))
    assert orientation == pytest.approx((1.0, 0.0, 0.0, 0.0))

    (rig_call,) = fake_surface_gripper.rig_calls
    assert rig_call["scope_path"] == "/World/vacuum_1_gripper"
    assert rig_call["clearance_offset_m"] == pytest.approx(0.005)
    assert len(rig_call["points_tool_m"]) == 4
    limits = rig_call["limits"]
    assert limits.max_grip_distance_m == pytest.approx(DEFAULT_MAX_GRIP_DISTANCE_MM / 1000.0)
    assert limits.coaxial_force_limit_n == pytest.approx(DEFAULT_COAXIAL_FORCE_LIMIT_N)
    assert limits.shear_force_limit_n == pytest.approx(DEFAULT_SHEAR_FORCE_LIMIT_N)
    assert limits.retry_interval_s == pytest.approx(DEFAULT_RETRY_INTERVAL_S)
    compliance = rig_call["compliance"]
    assert compliance.stiffness_n_per_m == pytest.approx(DEFAULT_CUP_STIFFNESS_N_PER_M)
    assert compliance.damping_n_s_per_m == pytest.approx(DEFAULT_CUP_DAMPING_N_S_PER_M)

    assert len(reset_calls) == 1

    assert handle.parent_prim_path == "/World/Arm/wrist_3_link"
    assert handle.tool_prim_path == "/World/Arm/wrist_3_link/EPick"
    assert handle._gripper_prim_path == "/World/vacuum_1_gripper/SurfaceGripper"
    assert handle._grab_delay_s == pytest.approx(DEFAULT_GRAB_DELAY_MS / 1000.0)
    assert handle._interface is fake_surface_gripper.interface
    assert manager.world.physics_callback_exists(handle.coaxial_callback_name)
    assert manager.world._callbacks[handle.coaxial_callback_name] == handle._on_physics_step


def test_vacuum_gripper_isaac_forwards_configured_attrs(fake_surface_gripper):
    manager = _manager()
    manager._isaac.get_prim_at_path = _fake_get_prim_at_path(set())
    reset_calls = _reset_counter(manager)
    arm = IsaacArmHandle(manager, _FakeArticulation(), None, prim_path="/World/Arm")

    handle = manager._create_vacuum_gripper_isaac(
        "vacuum-1",
        {
            "arm": "arm-1",
            "grab_delay_ms": 250,
            "max_grip_distance_mm": 20,
            "coaxial_force_limit_n": 60,
            "cup_stiffness": 3000,
        },
        arm,
    )

    (rig_call,) = fake_surface_gripper.rig_calls
    limits = rig_call["limits"]
    assert limits.max_grip_distance_m == pytest.approx(0.02)
    assert limits.coaxial_force_limit_n == pytest.approx(60.0)
    compliance = rig_call["compliance"]
    assert compliance.stiffness_n_per_m == pytest.approx(3000.0)

    assert len(reset_calls) == 1
    assert handle._grab_delay_s == pytest.approx(0.25)


def test_vacuum_gripper_isaac_reattach_authors_and_resets_nothing(fake_surface_gripper):
    manager = _manager()
    manager._isaac.get_prim_at_path = _fake_get_prim_at_path(
        {
            "/World/Arm/wrist_3_link/EPick",
            epick_body_prim_paths("/World/Arm/wrist_3_link/EPick")[-1],
            "/World/vacuum_1_gripper",
        }
    )
    reset_calls = _reset_counter(manager)
    arm = IsaacArmHandle(manager, _FakeArticulation(), None, prim_path="/World/Arm")

    handle = manager._create_vacuum_gripper_isaac("vacuum-1", {"arm": "arm-1"}, arm)

    assert fake_surface_gripper.epick_calls == []
    assert fake_surface_gripper.rig_calls == []
    assert reset_calls == []
    assert handle._gripper_prim_path == "/World/vacuum_1_gripper/SurfaceGripper"
