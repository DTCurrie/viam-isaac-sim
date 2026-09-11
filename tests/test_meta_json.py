import json
from pathlib import Path

import pytest

from isaac_module import FAMILY, NAMESPACE
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.conductor import IsaacConductor
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.scene_finalizer import IsaacSceneFinalizer
from isaac_module.models.sorter_sensor import SorterSensor
from isaac_module.models.world import IsaacWorld

META_PATH = Path(__file__).resolve().parent.parent / "meta.json"
REGISTERED_MODEL_CLASSES = (
    IsaacWorld,
    IsaacSceneFinalizer,
    IsaacArm,
    IsaacCamera,
    IsaacBase,
    IsaacGripper,
    IsaacConductor,
    SorterSensor,
)


# The models array and its short_description field are a contested surface (recorded
# 2026-09-09 by the Viam audit). The published module.schema.json at
# https://dl.viam.dev/module.schema.json names the per-model field description, not
# short_description. The module reference at docs.viam.com documents short_description
# and marks the models array itself deprecated. This fixture pins the shape this module
# actually uploads, not a surface Viam has settled on.
@pytest.fixture(scope="module")
def manifest_models() -> dict[str, str]:
    meta = json.loads(META_PATH.read_text())
    return {entry["model"]: entry["api"] for entry in meta["models"]}


def test_manifest_lists_exactly_the_registered_models(manifest_models: dict[str, str]) -> None:
    registered = {str(cls.MODEL) for cls in REGISTERED_MODEL_CLASSES}
    assert set(manifest_models) == registered


# The registry rejects an upload whose short_description exceeds this (observed
# 2026-09-04: "model description for viam:isaac-sim-devin:conductor exceeds
# maximum length of 100 characters"). Lower bound: an empty description ships a
# blank registry card.
MAX_SHORT_DESCRIPTION_CHARS = 100


def test_each_short_description_fits_the_registry_limit() -> None:
    meta = json.loads(META_PATH.read_text())
    for entry in meta["models"]:
        length = len(entry["short_description"])
        assert 0 < length <= MAX_SHORT_DESCRIPTION_CHARS, (entry["model"], length)


@pytest.mark.parametrize("cls", REGISTERED_MODEL_CLASSES, ids=lambda c: c.MODEL.name)
def test_each_manifest_entry_serves_the_models_api(manifest_models: dict[str, str], cls) -> None:
    assert str(cls.MODEL).startswith(f"{NAMESPACE}:{FAMILY}:")
    assert manifest_models[str(cls.MODEL)] == str(cls.API)
