"""workcell_client: fetching one configured component's scenery over the
Viam API, on fakes only, no GPU and no Isaac import."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from viam.proto.common import Geometry, RectangularPrism, ResourceName, Vector3

from isaac_module.workcell_client import (
    bare_model_name,
    gather_component_scenery,
    generic_dependencies,
    materialise_workcell,
)

LOGGER = logging.getLogger("test-workcell-client")

_SCHEMA = {"schema": [{"key": "width_mm", "type": "number"}]}


class _FakeWorkcellComponent:
    """A ``viam:workcell-components`` generic component: answers
    get_schema, get_status, get_attributes, get_visuals and (for a shaped
    model) GetGeometries the way the real module does."""

    def __init__(
        self,
        *,
        model: str,
        pose: dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
        visuals: dict[str, Any] | None = None,
        geometries: list[Geometry] | None = None,
        fails: bool = False,
    ) -> None:
        self._model = model
        self._pose = pose or {}
        self._attrs = attrs or {}
        self._visuals = visuals if visuals is not None else {"visuals": []}
        self._geometries = geometries or []
        self._fails = fails
        self.geometries_called = False

    async def do_command(self, command: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if "get_schema" in command:
            return _SCHEMA
        if self._fails:
            raise RuntimeError("component unreachable")
        if "get_status" in command:
            return {"model": self._model}
        if "get_attributes" in command:
            return {**self._attrs, "pose": self._pose}
        if "get_visuals" in command:
            return self._visuals
        raise AssertionError(f"unexpected DoCommand: {command}")

    async def get_geometries(self, **kwargs: Any) -> list[Geometry]:
        self.geometries_called = True
        if self._fails:
            raise RuntimeError("component unreachable")
        return self._geometries


class _FakeNonWorkcellComponent:
    """A sibling dependency (e.g. this cell's own world) that answers
    DoCommand but has no get_schema verb, the way IsaacWorld's own
    do_command raises on an unrecognised command key."""

    async def do_command(self, command: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("unknown command")


def _geometry(label: str, dims_mm: tuple[float, float, float]) -> Geometry:
    return Geometry(
        box=RectangularPrism(dims_mm=Vector3(x=dims_mm[0], y=dims_mm[1], z=dims_mm[2])),
        label=label,
    )


def test_bare_model_name_strips_the_namespace_and_family():
    assert bare_model_name("viam:workcell-components:pallet") == "pallet"
    assert bare_model_name("viam:workcell-components:pick-station") == "pick-station"


def test_bare_model_name_leaves_a_bare_name_unchanged():
    assert bare_model_name("pallet") == "pallet"


@pytest.mark.asyncio
async def test_shaped_model_gets_both_get_visuals_and_get_geometries():
    resource = _FakeWorkcellComponent(
        model="viam:workcell-components:pallet",
        visuals={
            "visuals": [{"type": "box", "label": "deck", "dims_mm": {"x": 1, "y": 1, "z": 1}}]
        },
        geometries=[_geometry("pallet-collider", (500.0, 350.0, 100.0))],
    )

    scenery = await gather_component_scenery("pallet", resource, logger=LOGGER)

    assert resource.geometries_called
    assert scenery is not None
    assert scenery.geometries == [
        {"label": "pallet-collider", "box_dims_mm": {"x": 500.0, "y": 350.0, "z": 100.0}}
    ]
    assert scenery.visuals == {
        "visuals": [{"type": "box", "label": "deck", "dims_mm": {"x": 1, "y": 1, "z": 1}}]
    }


@pytest.mark.asyncio
async def test_unshaped_model_gets_only_get_visuals():
    resource = _FakeWorkcellComponent(model="viam:workcell-components:robot-pedestal")

    scenery = await gather_component_scenery("robot-pedestal", resource, logger=LOGGER)

    assert not resource.geometries_called
    assert scenery is not None
    assert scenery.geometries == []


@pytest.mark.asyncio
async def test_a_component_with_no_get_schema_is_not_treated_as_workcell_scenery():
    resource = _FakeNonWorkcellComponent()

    scenery = await gather_component_scenery("isaac-world", resource, logger=LOGGER)

    assert scenery is None


@pytest.mark.asyncio
async def test_a_component_that_fails_past_the_schema_check_is_skipped_not_raised():
    resource = _FakeWorkcellComponent(model="viam:workcell-components:pallet", fails=True)

    scenery = await gather_component_scenery("pallet", resource, logger=LOGGER)

    assert scenery is None


def _group_frame(name: str, pose: dict[str, float]) -> dict[str, Any]:
    """The anchor primitive every workcell component puts first in its
    get_visuals reply, as visuals_group.go's groupUnderFrame emits it."""
    return {"type": "frame", "label": f"{name}/group", "parent_frame": "world", "pose": pose}


def test_the_group_frame_primitive_is_the_components_frame_position_and_orientation():
    resource = _FakeWorkcellComponent(
        model="viam:workcell-components:safety-fence",
        visuals={
            "visuals": [
                _group_frame(
                    "fence-left",
                    {"x": -1150.0, "y": -350.0, "z": 0.0, "o_z": 1.0, "theta": 90.0},
                ),
                {"type": "box", "label": "fence-left/screen", "dims_mm": {"x": 1, "y": 1, "z": 1}},
            ]
        },
    )

    scenery = materialise_workcell({"fence-left": resource}, logger=LOGGER)

    assert scenery["fence-left"].frame_position_m == pytest.approx((-1.15, -0.35, 0.0))
    orientation = scenery["fence-left"].frame_orientation_wxyz
    assert orientation[1] == pytest.approx(0.0, abs=1e-9)
    assert orientation[2] == pytest.approx(0.0, abs=1e-9)
    assert orientation[0] == pytest.approx(orientation[3])


def test_get_attributes_pose_is_not_the_anchor():
    # pick-station reports its bottom-left-top CORNER as get_attributes.pose,
    # centre plus (-200, -550, +20) for the vendored fragment's station, while
    # its primitives hang off the centre. Anchoring on the corner drew the
    # whole station 585 mm from its own collider.
    resource = _FakeWorkcellComponent(
        model="viam:workcell-components:pick-station",
        pose={"x": 200.0, "y": -1200.0, "z": 220.0, "o_z": 1.0},
        visuals={
            "visuals": [
                _group_frame("pick-station", {"x": 400.0, "y": -650.0, "z": 200.0, "o_z": 1.0})
            ]
        },
    )

    scenery = materialise_workcell({"pick-station": resource}, logger=LOGGER)

    assert scenery["pick-station"].frame_position_m == pytest.approx((0.4, -0.65, 0.2))
    assert scenery["pick-station"].frame_orientation_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_a_reply_with_no_group_frame_is_already_in_the_world_frame():
    resource = _FakeWorkcellComponent(
        model="viam:workcell-components:pallet",
        pose={"x": 200.0, "y": 500.0, "z": 200.0},
    )

    scenery = materialise_workcell({"pallet": resource}, logger=LOGGER)

    assert scenery["pallet"].frame_position_m == (0.0, 0.0, 0.0)
    assert scenery["pallet"].frame_orientation_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_a_group_frame_with_no_orientation_is_unrotated():
    resource = _FakeWorkcellComponent(
        model="viam:workcell-components:pallet",
        visuals={"visuals": [_group_frame("pallet", {"x": 200.0, "y": 500.0, "z": 200.0})]},
    )

    scenery = materialise_workcell({"pallet": resource}, logger=LOGGER)

    assert scenery["pallet"].frame_position_m == pytest.approx((0.2, 0.5, 0.2))
    assert scenery["pallet"].frame_orientation_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_get_attributes_pose_key_is_not_leaked_into_component_attrs():
    resource = _FakeWorkcellComponent(
        model="viam:workcell-components:robot-pedestal",
        attrs={"height_mm": 150, "diameter_mm": 220},
        pose={"x": 0.0, "y": 0.0, "z": 0.0},
    )

    scenery = materialise_workcell({"robot-pedestal": resource}, logger=LOGGER)

    assert scenery["robot-pedestal"].attrs == {"height_mm": 150, "diameter_mm": 220}


def test_materialise_workcell_skips_a_failing_component_and_keeps_the_rest():
    ok_resource = _FakeWorkcellComponent(
        model="viam:workcell-components:robot-pedestal",
        attrs={"height_mm": 150, "diameter_mm": 220},
    )
    failing_resource = _FakeWorkcellComponent(
        model="viam:workcell-components:hmi-cabinet", fails=True
    )
    resources = {"robot-pedestal": ok_resource, "hmi-cabinet": failing_resource}

    scenery = materialise_workcell(resources, logger=LOGGER)

    assert list(scenery) == ["robot-pedestal"]


def test_generic_dependencies_filters_to_rdk_component_generic():
    generic_name = ResourceName(namespace="rdk", type="component", subtype="generic", name="pallet")
    arm_name = ResourceName(namespace="rdk", type="component", subtype="arm", name="arm-1")
    generic_resource = _FakeWorkcellComponent(model="viam:workcell-components:pallet")
    arm_resource = _FakeNonWorkcellComponent()

    filtered = generic_dependencies({generic_name: generic_resource, arm_name: arm_resource})

    assert list(filtered) == ["pallet"]
    assert filtered["pallet"] is generic_resource
