"""Prop and grasp physics (FINDINGS SCN-6, ARM-16, W27, W28; R-22, R-23).

Per-prop physics keys on the world component's ``props`` entries:

  mass (kg) - friction (static = dynamic) - restitution - contact_offset (m) - rest_offset (m)

map to a PhysicsMaterial (frictionCombineMode "max") plus rigid-body /
collision attributes on the spawned prim. ``apply_prop_physics`` runs on the
sim thread from ``SimManager._spawn_prop``, after the prim exists and before
the initial ``world.reset()``.

Explicit beats implicit: a prop that names none of these keys keeps Isaac's
authored defaults (friction 0.2/1.0, mass 0.02 kg, contact_offset 0.1 m),
which R-22/R-23 flag as wrong for a 50 mm block - so the shipped pick cell
sets them in the fragment, using the named constants below.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any

LOGGER = getLogger("viam-isaac-sim")

PROP_PHYSICS_KEYS = ("mass", "friction", "restitution", "contact_offset", "rest_offset")

# W27: block/table material for the pick cell.
PICK_CELL_BLOCK_PHYSICS: dict[str, float] = {
    "mass": 0.05,
    "friction": 0.7,
    "restitution": 0.0,
    "contact_offset": 0.005,
}
FRICTION_COMBINE_MODE = "max"

# W28 / DEC-9: step rates for the pick cell (doc floor for a 2F-85 grasp is
# >= 80 physics steps/s); rendering stays at 1/60 so camera cadence is
# unchanged. Consumed by the shipped fragment and its test (SCN-10 / ARM-7).
PICK_CELL_PHYSICS_DT = 1.0 / 120.0
PICK_CELL_RENDERING_DT = 1.0 / 60.0

# W28: the 2F-85 asset asks for 64 solver position iterations; the UR asset
# authors 32. Re-applied to the arm after every reset (ARM-15/ARM-16 via the
# XC-5 post-reset hook in sim_manager).
ARM_SOLVER_POSITION_ITERATIONS = 64


def prop_physics(prop: dict[str, Any]) -> dict[str, float]:
    """The subset of PROP_PHYSICS_KEYS ``prop`` sets, as floats.

    Raises ValueError for a value outside its physical range (negative mass or
    offsets, friction < 0, restitution outside [0, 1], rest_offset above
    contact_offset)."""
    values = {key: float(prop[key]) for key in PROP_PHYSICS_KEYS if key in prop}

    if "mass" in values and values["mass"] <= 0:
        raise ValueError(f"prop physics: 'mass' must be positive (got {values['mass']!r})")
    if "friction" in values and values["friction"] < 0:
        raise ValueError(f"prop physics: 'friction' must be >= 0 (got {values['friction']!r})")
    if "restitution" in values and not (0 <= values["restitution"] <= 1):
        raise ValueError(
            f"prop physics: 'restitution' must be in [0, 1] (got {values['restitution']!r})"
        )
    if "contact_offset" in values and values["contact_offset"] < 0:
        raise ValueError(
            f"prop physics: 'contact_offset' must be >= 0 (got {values['contact_offset']!r})"
        )
    if "rest_offset" in values and values["rest_offset"] < 0:
        raise ValueError(
            f"prop physics: 'rest_offset' must be >= 0 (got {values['rest_offset']!r})"
        )
    if "rest_offset" in values and "contact_offset" in values:
        if values["rest_offset"] > values["contact_offset"]:
            raise ValueError(
                "prop physics: 'rest_offset' must not be greater than 'contact_offset' "
                f"(got rest_offset={values['rest_offset']!r}, "
                f"contact_offset={values['contact_offset']!r})"
            )

    return values


def _default_material_kwargs(obj: Any, friction: float | None) -> dict[str, float]:
    """Default static/dynamic friction for a PhysicsMaterial when only one of
    friction/restitution is set on the prop: reuse the object's existing
    material if it has one, else fall back to Isaac's authored friction
    (0.2 static, 1.0 dynamic - see module docstring)."""
    if friction is not None:
        return {"static_friction": friction, "dynamic_friction": friction}
    get_material = getattr(obj, "get_applied_physics_material", None)
    existing = get_material() if callable(get_material) else None
    if existing is not None:
        static = getattr(existing, "static_friction", None)
        dynamic = getattr(existing, "dynamic_friction", None)
        return {
            k: v
            for k, v in {"static_friction": static, "dynamic_friction": dynamic}.items()
            if v is not None
        }
    return {}


def apply_prop_physics(isaac: Any, world: Any, prim_path: str, prop: dict[str, Any]) -> None:
    """Author ``prop``'s physics onto ``prim_path`` (sim thread only).

    No-op when the prop sets none of PROP_PHYSICS_KEYS. ``isaac`` is the
    namespace from ``compat.import_isaac``; pxr schemas are imported lazily
    here (they only exist inside Kit)."""
    values = prop_physics(prop)
    if not values:
        return

    prim_name = prim_path.rstrip("/").rsplit("/", 1)[-1]
    obj = world.scene.get_object(prim_name)
    if obj is None:
        LOGGER.warning(
            "prop %s: physics keys are not applied to usd props (no scene object for %s)",
            prop.get("name"),
            prim_path,
        )
        return

    friction = values.get("friction")
    restitution = values.get("restitution")
    if friction is not None or restitution is not None:
        material_kwargs = _default_material_kwargs(obj, friction)
        if restitution is not None:
            material_kwargs["restitution"] = restitution
        material = isaac.PhysicsMaterial(
            prim_path=f"{prim_path}/physics_material",
            name=f"{prim_name}_physics_material",
            **material_kwargs,
        )
        if isaac.PhysxSchema is not None:
            physx_material_api = isaac.PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
            physx_material_api.CreateFrictionCombineModeAttr().Set(FRICTION_COMBINE_MODE)
        apply_material = getattr(obj, "apply_physics_material", None)
        if callable(apply_material):
            apply_material(material)
        else:
            LOGGER.warning("prop %s: object has no apply_physics_material", prop.get("name"))

    contact_offset = values.get("contact_offset")
    if contact_offset is not None:
        set_contact_offset = getattr(obj, "set_contact_offset", None)
        if callable(set_contact_offset):
            set_contact_offset(contact_offset)
        else:
            LOGGER.warning("prop %s: object has no set_contact_offset", prop.get("name"))

    rest_offset = values.get("rest_offset")
    if rest_offset is not None:
        set_rest_offset = getattr(obj, "set_rest_offset", None)
        if callable(set_rest_offset):
            set_rest_offset(rest_offset)
        else:
            LOGGER.warning("prop %s: object has no set_rest_offset", prop.get("name"))

    mass = values.get("mass")
    if mass is not None:
        set_mass = getattr(obj, "set_mass", None)
        if callable(set_mass):
            set_mass(mass)
        else:
            LOGGER.warning("prop %s: object has no set_mass (e.g. a FixedCuboid)", prop.get("name"))
