from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types

import pytest

from isaac_module import compat


def _isaac_is_installed() -> bool:
    return (
        importlib.util.find_spec("isaacsim") is not None
        or importlib.util.find_spec("omni") is not None
    )


@pytest.mark.skipif(
    _isaac_is_installed(),
    reason="Isaac is importable in this environment; isaac_version() would not be None",
)
def test_isaac_version_is_none_without_isaac() -> None:
    assert compat.isaac_version() is None


def test_caps_by_release_has_only_the_5_0_row() -> None:
    assert set(compat.CAPS_BY_RELEASE) == {(5, 0)}


@pytest.mark.parametrize(
    "version",
    [None, (4, 5, 0), (5, 0, 0), (4, 6, 1), (6, 0, 0), (4, 0, 0)],
)
def test_caps_resolves_to_the_5_0_row_for_any_version(
    version: tuple[int, int, int] | None,
) -> None:
    assert compat.caps(version) == compat.CAPS_BY_RELEASE[(5, 0)]


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch, dotted_path: str, module: types.ModuleType
) -> None:
    monkeypatch.setitem(sys.modules, dotted_path, module)


def test_probe_isaacsim_core_version_parses_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "isaacsim", types.ModuleType("isaacsim"))
    _install_fake_module(monkeypatch, "isaacsim.core", types.ModuleType("isaacsim.core"))

    fake_version_module = types.ModuleType("isaacsim.core.version")
    fake_version_module.get_version = lambda: (  # type: ignore[attr-defined]
        "5.0.0",
        "5",
        "0",
        "0",
        "",
        "107.3",
    )
    _install_fake_module(monkeypatch, "isaacsim.core.version", fake_version_module)

    assert compat.isaac_version() == (5, 0, 0)


def test_probe_that_raises_falls_through_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "isaacsim", types.ModuleType("isaacsim"))
    _install_fake_module(monkeypatch, "isaacsim.core", types.ModuleType("isaacsim.core"))

    def _raise() -> None:
        raise RuntimeError("boom")

    fake_version_module = types.ModuleType("isaacsim.core.version")
    fake_version_module.get_version = _raise  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "isaacsim.core.version", fake_version_module)

    def _raise_metadata_lookup(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise_metadata_lookup)

    assert compat.isaac_version() is None


def test_import_isaac_moved_into_sim_manager() -> None:
    from isaac_module.sim_manager import import_isaac

    assert import_isaac is compat.import_isaac


def _fake_isaac_module(monkeypatch: pytest.MonkeyPatch, dotted_path: str, **attrs: object) -> None:
    """Install ``dotted_path`` in ``sys.modules``, filling in any missing
    parent packages, so ``import_isaac``'s ``from a.b.c import X`` lines
    resolve against fakes instead of the real Isaac Sim packages."""
    parts = dotted_path.split(".")
    for depth in range(1, len(parts)):
        parent = ".".join(parts[:depth])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))

    module = types.ModuleType(dotted_path)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, dotted_path, module)


def test_import_isaac_result_satisfies_isaac_api_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """import_isaac()'s namespace carries every attribute IsaacAPI declares,
    proving the enumerable surface stays in sync with the protocol."""

    class _Stub:
        pass

    _fake_isaac_module(monkeypatch, "isaacsim.core.api", World=_Stub)
    _fake_isaac_module(
        monkeypatch, "isaacsim.core.utils.stage", add_reference_to_stage=_Stub, open_stage=_Stub
    )
    _fake_isaac_module(monkeypatch, "isaacsim.storage.native", get_assets_root_path=_Stub)
    _fake_isaac_module(
        monkeypatch, "isaacsim.core.prims", SingleArticulation=_Stub, SingleXFormPrim=_Stub
    )
    _fake_isaac_module(monkeypatch, "isaacsim.core.utils.types", ArticulationAction=_Stub)
    _fake_isaac_module(
        monkeypatch, "isaacsim.core.api.objects", DynamicCuboid=_Stub, FixedCuboid=_Stub
    )
    _fake_isaac_module(monkeypatch, "isaacsim.sensors.camera", Camera=_Stub)
    _fake_isaac_module(
        monkeypatch,
        "isaacsim.robot.wheeled_robots.controllers.differential_controller",
        DifferentialController=_Stub,
    )
    _fake_isaac_module(monkeypatch, "isaacsim.robot.wheeled_robots.robots", WheeledRobot=_Stub)

    ns = compat.import_isaac()

    for name in compat.IsaacAPI.__annotations__:
        assert hasattr(ns, name), f"import_isaac() result is missing {name!r}"


def test_gripper_caps_follow_the_2f85_asset() -> None:
    """The 2F-85 finger_joint closes at 47 deg on 5.0 - version splits live
    here, never in models/gripper.py."""
    row = compat.CAPS_BY_RELEASE[(5, 0)]
    assert row.gripper_closed_deg == 47.0
    assert compat.caps((5, 0, 0)).gripper_dof_count == 6


def test_gripper_open_angle_is_zero() -> None:
    """With the articulation fixes in place the 2F-85 reaches 0.4 deg at an
    open target of 0 (a 7.76 deg rest was an artifact of the broken
    wrapper); the 0 target stands."""
    assert compat.CAPS_BY_RELEASE[(5, 0)].gripper_open_deg == 0.0


def test_camera_supports_annotator_device() -> None:
    """Camera(annotator_device=...)/get_*(device=...) is a GPU-resident data
    path (CHANGELOG 0.4.0)."""
    assert compat.CAPS_BY_RELEASE[(5, 0)].camera_supports_annotator_device is True


def _fake_surface_gripper_modules(
    monkeypatch: pytest.MonkeyPatch, *, robot_schema_module: str, with_physx_schema: bool
) -> None:
    _fake_isaac_module(
        monkeypatch,
        "isaacsim.core.utils.extensions",
        enable_extension=lambda name: True,
    )
    _fake_isaac_module(monkeypatch, "isaacsim.robot.surface_gripper._surface_gripper")
    if robot_schema_module == "usd.schema.isaac":
        _fake_isaac_module(monkeypatch, "usd.schema.isaac.robot_schema")
    elif robot_schema_module == "isaacsim.robot.schema":
        _fake_isaac_module(monkeypatch, "isaacsim.robot.schema.robot_schema")

    pxr = types.ModuleType("pxr")
    pxr.Gf = _Stub()  # type: ignore[attr-defined]
    pxr.Sdf = _Stub()  # type: ignore[attr-defined]
    pxr.UsdGeom = _Stub()  # type: ignore[attr-defined]
    pxr.UsdPhysics = _Stub()  # type: ignore[attr-defined]
    if with_physx_schema:
        pxr.PhysxSchema = _Stub()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pxr", pxr)


class _Stub:
    pass


def test_import_surface_gripper_reports_the_modules_it_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat.import_surface_gripper.cache_clear()
    _fake_surface_gripper_modules(
        monkeypatch, robot_schema_module="usd.schema.isaac", with_physx_schema=False
    )

    result = compat.import_surface_gripper()

    for key in (
        "report",
        "surface_gripper",
        "robot_schema",
        "Gf",
        "Sdf",
        "UsdGeom",
        "UsdPhysics",
        "PhysxSchema",
    ):
        assert key in result, f"import_surface_gripper() result is missing {key!r}"
    assert result["report"]["robot_schema_module"] == "usd.schema.isaac"
    assert result["PhysxSchema"] is None
    compat.import_surface_gripper.cache_clear()


def test_import_surface_gripper_falls_back_to_isaacsim_robot_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat.import_surface_gripper.cache_clear()
    _fake_surface_gripper_modules(
        monkeypatch, robot_schema_module="isaacsim.robot.schema", with_physx_schema=True
    )

    result = compat.import_surface_gripper()

    assert result["report"]["robot_schema_module"] == "isaacsim.robot.schema"
    assert result["PhysxSchema"] is not None
    compat.import_surface_gripper.cache_clear()


def test_import_surface_gripper_raises_naming_both_failures_when_neither_schema_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat.import_surface_gripper.cache_clear()
    _fake_surface_gripper_modules(monkeypatch, robot_schema_module="", with_physx_schema=False)
    monkeypatch.delitem(sys.modules, "usd.schema.isaac.robot_schema", raising=False)
    monkeypatch.delitem(sys.modules, "isaacsim.robot.schema.robot_schema", raising=False)

    with pytest.raises(ImportError, match=r"usd\.schema\.isaac.*isaacsim\.robot\.schema"):
        compat.import_surface_gripper()
    compat.import_surface_gripper.cache_clear()
    assert compat.caps((5, 0, 0)).camera_supports_annotator_device is True
