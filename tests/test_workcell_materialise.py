"""SimManager.materialise_components: workcell_scenery's prop dicts turned
into stage prims, on the fake-Isaac path (no GPU, no real Isaac import).

Mirrors the fake-namespace pattern test_world_handle.py already uses for
_spawn_prop: a plain-python stand-in for isaacsim's SingleXFormPrim,
DynamicCuboid/FixedCuboid and the pxr schemas, so the real Isaac cube and
mesh paths run under test with no GPU and no Isaac import.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from isaac_module.handles.arm import IsaacArmHandle
from isaac_module.handles.vacuum import DEFAULT_GRAB_DELAY_MS
from isaac_module.sim_manager import ComponentScenery, SimManager


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

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        pass


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
# grab_delay_ms reaches the Isaac vacuum handle
# ----------------------------------------------------------------------


class _FakeArticulation:
    dof_names: ClassVar[list[str]] = []


class _FakeXformableOp:
    def Set(self, value) -> None:
        pass


class _FakeXformable:
    def __init__(self, prim) -> None:
        pass

    def ClearXformOpOrder(self) -> None:
        pass

    def AddTranslateOp(self):
        return _FakeXformableOp()

    def AddOrientOp(self):
        return _FakeXformableOp()

    def AddScaleOp(self):
        return _FakeXformableOp()


class _FakeCubeGeom:
    def CreateSizeAttr(self, value) -> None:
        pass


class _FakeCubeCls:
    @staticmethod
    def Define(stage, path):
        return _FakeCubeGeom()


class _FakeVecOrQuat:
    def __init__(self, *args) -> None:
        self.args = args


class _FakeStage:
    pass


class _FakePrimHandle:
    def GetStage(self):
        return _FakeStage()


@pytest.fixture
def fake_pxr(monkeypatch):
    fake_gf = types.SimpleNamespace(
        Vec3d=_FakeVecOrQuat, Vec3f=_FakeVecOrQuat, Quatf=_FakeVecOrQuat
    )
    fake_usdgeom = types.SimpleNamespace(Cube=_FakeCubeCls, Xformable=_FakeXformable)
    fake_module = types.ModuleType("pxr")
    fake_module.Gf = fake_gf
    fake_module.UsdGeom = fake_usdgeom
    monkeypatch.setitem(sys.modules, "pxr", fake_module)


def test_vacuum_gripper_isaac_forwards_configured_grab_delay_ms(fake_pxr):
    manager = _manager()
    manager._isaac.get_prim_at_path = lambda path: _FakePrimHandle()
    arm = IsaacArmHandle(manager, _FakeArticulation(), None, prim_path="/World/Arm")

    handle = manager._create_vacuum_gripper_isaac(
        "vacuum-1", {"arm": "arm-1", "grab_delay_ms": 250}, arm
    )

    assert handle._grab_delay_s == pytest.approx(0.25)


def test_vacuum_gripper_isaac_defaults_grab_delay_ms(fake_pxr):
    manager = _manager()
    manager._isaac.get_prim_at_path = lambda path: _FakePrimHandle()
    arm = IsaacArmHandle(manager, _FakeArticulation(), None, prim_path="/World/Arm")

    handle = manager._create_vacuum_gripper_isaac("vacuum-1", {"arm": "arm-1"}, arm)

    assert handle._grab_delay_s == pytest.approx(DEFAULT_GRAB_DELAY_MS / 1000.0)
