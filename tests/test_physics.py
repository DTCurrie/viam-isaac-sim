from typing import Any

import pytest

from isaac_module.physics import (
    FRICTION_COMBINE_MODE,
    PICK_CELL_BLOCK_PHYSICS,
    apply_prop_physics,
    prop_physics,
)

# ---------------------------------------------------------------------------
# prop_physics
# ---------------------------------------------------------------------------


def test_prop_physics_returns_only_named_keys_as_floats():
    result = prop_physics({"name": "block", "mass": 1, "size": 0.1})
    assert result == {"mass": 1.0}
    assert isinstance(result["mass"], float)


def test_prop_physics_empty_when_no_keys_set():
    assert prop_physics({"name": "block"}) == {}


def test_prop_physics_pick_cell_block_values():
    prop = {"name": "block", **PICK_CELL_BLOCK_PHYSICS}
    assert prop_physics(prop) == PICK_CELL_BLOCK_PHYSICS


def test_prop_physics_rejects_nonpositive_mass():
    with pytest.raises(ValueError, match="mass"):
        prop_physics({"mass": 0})
    with pytest.raises(ValueError, match="mass"):
        prop_physics({"mass": -1})


def test_prop_physics_rejects_negative_friction():
    with pytest.raises(ValueError, match="friction"):
        prop_physics({"friction": -0.1})


def test_prop_physics_rejects_restitution_out_of_range():
    with pytest.raises(ValueError, match="restitution"):
        prop_physics({"restitution": 1.5})
    with pytest.raises(ValueError, match="restitution"):
        prop_physics({"restitution": -0.1})


def test_prop_physics_rejects_negative_contact_offset():
    with pytest.raises(ValueError, match="contact_offset"):
        prop_physics({"contact_offset": -0.001})


def test_prop_physics_rejects_negative_rest_offset():
    with pytest.raises(ValueError, match="rest_offset"):
        prop_physics({"rest_offset": -0.001})


def test_prop_physics_rejects_rest_offset_above_contact_offset():
    with pytest.raises(ValueError, match="rest_offset"):
        prop_physics({"rest_offset": 0.01, "contact_offset": 0.005})


# ---------------------------------------------------------------------------
# apply_prop_physics
# ---------------------------------------------------------------------------


class _Raiser:
    """Raises on any attribute access, to prove a no-op function never
    touches its arguments."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected attribute access: {name!r}")


class _FakeMaterial:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.prim = "PRIM_SENTINEL"


class _FakePhysxMaterialAPI:
    def __init__(self, prim: Any):
        self.prim = prim
        self.combine_mode: Any = None

    def CreateFrictionCombineModeAttr(self):  # noqa: N802 - matches pxr API naming
        return self

    def Set(self, value):  # noqa: N802 - matches pxr API naming
        self.combine_mode = value


class _FakePhysxSchemaAPI:
    def __init__(self):
        self.applied: list[_FakePhysxMaterialAPI] = []

    def Apply(self, prim):  # noqa: N802 - matches pxr API naming
        api = _FakePhysxMaterialAPI(prim)
        self.applied.append(api)
        return api


class _FakePhysxSchema:
    def __init__(self):
        self.PhysxMaterialAPI = _FakePhysxSchemaAPI()


class _FakeIsaac:
    def __init__(self):
        self.physx_schema = _FakePhysxSchema()
        self.PhysxSchema = self.physx_schema
        self.materials: list[_FakeMaterial] = []

    def PhysicsMaterial(self, **kwargs):
        material = _FakeMaterial(**kwargs)
        self.materials.append(material)
        return material


class _FakeSceneObject:
    def __init__(self, has_set_mass: bool = True):
        self.applied_material: Any = None
        self.contact_offset: float | None = None
        self.rest_offset: float | None = None
        self.mass: float | None = None
        if has_set_mass:
            self.set_mass = self._set_mass  # type: ignore[assignment]

    def apply_physics_material(self, material: Any) -> None:
        self.applied_material = material

    def set_contact_offset(self, value: float) -> None:
        self.contact_offset = value

    def set_rest_offset(self, value: float) -> None:
        self.rest_offset = value

    def _set_mass(self, value: float) -> None:
        self.mass = value


class _FakeScene:
    def __init__(self, objects: dict[str, Any]):
        self._objects = objects

    def get_object(self, name: str) -> Any:
        return self._objects.get(name)


class _FakeWorld:
    def __init__(self, objects: dict[str, Any]):
        self.scene = _FakeScene(objects)


def test_apply_prop_physics_no_keys_touches_nothing():
    apply_prop_physics(_Raiser(), _Raiser(), "/World/block", {"name": "block", "size": 0.1})


def test_apply_prop_physics_pick_cell_block():
    isaac = _FakeIsaac()
    obj = _FakeSceneObject()
    world = _FakeWorld({"block": obj})
    prop = {"name": "block", **PICK_CELL_BLOCK_PHYSICS}

    apply_prop_physics(isaac, world, "/World/block", prop)

    assert len(isaac.materials) == 1
    material = isaac.materials[0]
    assert material.kwargs["static_friction"] == 0.7
    assert material.kwargs["dynamic_friction"] == 0.7
    assert material.kwargs["restitution"] == 0.0

    combine_modes = [api.combine_mode for api in isaac.physx_schema.PhysxMaterialAPI.applied]
    assert combine_modes == [FRICTION_COMBINE_MODE]
    assert FRICTION_COMBINE_MODE == "max"

    assert obj.applied_material is material
    assert obj.contact_offset == 0.005
    assert obj.mass == 0.05


def test_apply_prop_physics_usd_prop_not_scene_registered_is_a_noop():
    isaac = _FakeIsaac()
    world = _FakeWorld({})  # get_object returns None for a usd prop

    apply_prop_physics(isaac, world, "/World/table", {"name": "table", "mass": 1.0})

    assert isaac.materials == []


def test_apply_prop_physics_skips_mass_when_object_has_no_set_mass():
    isaac = _FakeIsaac()
    obj = _FakeSceneObject(has_set_mass=False)
    world = _FakeWorld({"wall": obj})

    apply_prop_physics(isaac, world, "/World/wall", {"name": "wall", "mass": 1.0})

    assert obj.mass is None
