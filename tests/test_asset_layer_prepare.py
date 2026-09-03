"""R-4 / OQ-4 applied BEFORE composition (GPU 2026-09-03: the 2F-85's isaac-dev
part references stalled the stage 131 s per module start): preparing a local
copy of an asset layer with those references moved onto the assets root
(SimManager._prepared_asset_layer and its pure helpers), driven with fakes."""

from pathlib import Path

import pytest

from isaac_module.sim_manager import (
    ASSET_CACHE_DIRNAME,
    ASSET_LAYER_CACHE_VERSION,
    SimManager,
    _anchor_asset_path,
    _bucket_candidate,
    _prepared_asset_path,
    asset_cache_dir,
)

ASSETS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
GRIPPER = f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85.usd"
PART = "/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_basestep_JFX.usd"
ISAAC_DEV_PART = f"omniverse://isaac-dev.ov.nvidia.com{PART}"
PUBLIC_PART = f"{ASSETS_ROOT}{PART}"


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_bucket_candidate_maps_the_unresolvable_host_onto_the_assets_root():
    assert _bucket_candidate(ISAAC_DEV_PART, ASSETS_ROOT + "/") == PUBLIC_PART
    assert _bucket_candidate(PUBLIC_PART, ASSETS_ROOT) is None
    assert _bucket_candidate("omniverse://isaac-dev.ov.nvidia.com/other/x.usd", ASSETS_ROOT) is None


@pytest.mark.parametrize(
    ("asset_path", "expected"),
    [
        ("./parts/x.usd", f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/2F-85/parts/x.usd"),
        ("../Materials/m.usd", f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/Materials/m.usd"),
        ("OmniPBR.mdl", "OmniPBR.mdl"),  # search path: the MDL resolver owns it
        ("nvidia/support_definitions.mdl", "nvidia/support_definitions.mdl"),
        ("/abs/x.usd", "/abs/x.usd"),
        (PUBLIC_PART, PUBLIC_PART),
        ("", ""),
    ],
)
def test_anchor_asset_path_resolves_only_explicitly_relative_paths(asset_path, expected):
    assert _anchor_asset_path(asset_path, GRIPPER) == expected


def test_anchor_asset_path_never_pops_into_the_url_host_or_past_root():
    assert _anchor_asset_path("../../../../../x.usd", "https://h/a/b.usd") == "https://h/x.usd"
    assert _anchor_asset_path("../../../x.usd", "/a/b.usd") == "/x.usd"


def _report() -> dict:
    return {"applied": [], "missing": [], "anchored": []}


def test_prepared_asset_path_rewrites_anchors_and_drops_missing_parts():
    report = _report()
    assert _prepared_asset_path(ISAAC_DEV_PART, GRIPPER, ASSETS_ROOT, lambda _p: True, report) == (
        PUBLIC_PART
    )
    assert _prepared_asset_path("./m.usd", GRIPPER, ASSETS_ROOT, lambda _p: True, report).endswith(
        "/2F-85/m.usd"
    )
    # a provably-missing bucket part is dropped ("" removes the reference):
    # the dead-host path can never load and costs a ~12 s stall each
    assert (
        _prepared_asset_path(ISAAC_DEV_PART, GRIPPER, ASSETS_ROOT, lambda _p: False, report) == ""
    )
    assert _prepared_asset_path(PUBLIC_PART, GRIPPER, ASSETS_ROOT, lambda _p: True, report) == (
        PUBLIC_PART
    )
    assert report["applied"] == [(ISAAC_DEV_PART, PUBLIC_PART)]
    assert report["missing"] == [PUBLIC_PART]
    assert len(report["anchored"]) == 1


def test_asset_cache_dir_prefers_the_module_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    assert asset_cache_dir() == tmp_path / ASSET_CACHE_DIRNAME
    monkeypatch.delenv("VIAM_MODULE_DATA")
    assert asset_cache_dir().name == ASSET_CACHE_DIRNAME
    assert asset_cache_dir().parent != tmp_path


# ---------------------------------------------------------------------------
# the layer preparation, with fake Sdf / UsdUtils
# ---------------------------------------------------------------------------


class FakeLayer:
    def __init__(self, paths) -> None:
        self.paths = list(paths)
        self.exported_to: str | None = None

    def GetCompositionAssetDependencies(self):
        return list(self.paths)

    def TransferContent(self, other: "FakeLayer") -> None:
        self.paths = list(other.paths)

    def Export(self, path: str) -> bool:
        self.exported_to = path
        return True


class FakeSdf:
    opened: list[str] = []
    layers: dict[str, FakeLayer] = {}
    anonymous: list[FakeLayer] = []

    class Layer:
        @staticmethod
        def FindOrOpen(path: str):
            FakeSdf.opened.append(path)
            return FakeSdf.layers.get(path)

        @staticmethod
        def CreateAnonymous(_tag: str) -> FakeLayer:
            layer = FakeLayer([])
            FakeSdf.anonymous.append(layer)
            return layer


class FakeUsdUtils:
    @staticmethod
    def ModifyAssetPaths(layer: FakeLayer, fn) -> None:
        layer.paths = [fn(path) for path in layer.paths]


class FakeClient:
    """omni.client stand-in whose stat says every path is missing."""

    class Result:
        OK = "ok"

    @staticmethod
    def stat(_path: str):
        return "not-found", None


class FakeIsaac:
    client = None  # _usd_exists -> None -> "could not check", so the rewrite is tried

    @staticmethod
    def get_assets_root_path():
        return ASSETS_ROOT


@pytest.fixture
def manager(monkeypatch, tmp_path) -> SimManager:
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    FakeSdf.opened = []
    FakeSdf.layers = {}
    FakeSdf.anonymous = []
    sim = SimManager()
    sim._isaac = FakeIsaac()
    return sim


def test_layer_on_the_unresolvable_host_is_copied_rewritten_and_cached(manager, tmp_path):
    FakeSdf.layers[GRIPPER] = FakeLayer([ISAAC_DEV_PART, "./Materials/m.usd", "OmniPBR.mdl"])

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)

    local = Path(path)
    assert local.parent == tmp_path / ASSET_CACHE_DIRNAME
    assert local.parent.is_dir()
    assert local.name.startswith("Robotiq_2F_85-")
    assert local.name.endswith(f"-v{ASSET_LAYER_CACHE_VERSION}.usd")
    assert report["reason"] == "prepared"
    assert report["source"] == GRIPPER
    assert report["applied"] == [(ISAAC_DEV_PART, PUBLIC_PART)]
    assert report["missing"] == []
    assert report["anchored"] == [
        ("./Materials/m.usd", f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/2F-85/Materials/m.usd")
    ]
    (copy,) = FakeSdf.anonymous
    assert copy.exported_to == str(local)
    assert copy.paths == [
        PUBLIC_PART,
        f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/2F-85/Materials/m.usd",
        "OmniPBR.mdl",
    ]


def test_cached_copy_is_reused_without_opening_the_remote_layer(manager):
    FakeSdf.layers[GRIPPER] = FakeLayer([ISAAC_DEV_PART])
    first, _ = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)
    Path(first).touch()  # the fake Export wrote nothing; a real one writes the file
    FakeSdf.opened = []

    second, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)

    assert second == first
    assert report["reason"] == "cached"
    assert FakeSdf.opened == []
    assert len(FakeSdf.anonymous) == 1


def test_layer_without_unresolvable_references_is_referenced_as_is(manager, tmp_path):
    FakeSdf.layers[GRIPPER] = FakeLayer([PUBLIC_PART, "./Materials/m.usd"])

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)

    assert path == GRIPPER
    assert "no isaac-dev references" in report["reason"]
    assert FakeSdf.anonymous == []
    assert not (tmp_path / ASSET_CACHE_DIRNAME).exists()


def test_part_the_bucket_lacks_is_dropped_and_reported(manager):
    manager._isaac.client = FakeClient()
    FakeSdf.layers[GRIPPER] = FakeLayer([ISAAC_DEV_PART])

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)

    assert path != GRIPPER
    assert report["applied"] == []
    assert report["missing"] == [PUBLIC_PART]
    # "" removes the reference under the real UsdUtils.ModifyAssetPaths
    assert FakeSdf.anonymous[0].paths == [""]


def test_unavailable_usd_utils_or_unopenable_layer_falls_back_to_the_source(manager):
    path, report = manager._prepared_asset_layer(FakeSdf, None, GRIPPER)
    assert (path, report["reason"]) == (GRIPPER, "pxr.UsdUtils unavailable")

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)
    assert (path, report["reason"]) == (GRIPPER, "layer did not open")
    assert report["unopened"] == [GRIPPER]
    assert FakeSdf.anonymous == []


def test_no_assets_root_falls_back_to_the_source(manager):
    FakeSdf.layers[GRIPPER] = FakeLayer([ISAAC_DEV_PART])
    manager._isaac.get_assets_root_path = lambda: None

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER)

    assert (path, report["reason"]) == (GRIPPER, "no assets root")


# ---------------------------------------------------------------------------
# recursion: references hiding below the root layer (GPU 2026-09-04 - the
# 2F-85 "edit" asset's isaac-dev parts sit in its payload, so the old
# single-layer check said "no references" and every reload re-paid the stall)
# ---------------------------------------------------------------------------

GRIPPER_EDIT = f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
PAYLOAD_AS_WRITTEN = "./payloads/Robotiq_2F_85_base.usda"
PAYLOAD = f"{ASSETS_ROOT}/Isaac/Robots/Robotiq/2F-85/payloads/Robotiq_2F_85_base.usda"


def test_isaac_dev_references_in_the_payload_layer_are_found_and_rewritten(manager, tmp_path):
    FakeSdf.layers[GRIPPER_EDIT] = FakeLayer([PAYLOAD_AS_WRITTEN])
    FakeSdf.layers[PAYLOAD] = FakeLayer([ISAAC_DEV_PART, "OmniPBR.mdl"])

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER_EDIT)

    assert report["reason"] == "prepared"
    root_copy, payload_copy = None, None
    for layer in FakeSdf.anonymous:
        if PUBLIC_PART in layer.paths:
            payload_copy = layer
        else:
            root_copy = layer
    assert payload_copy is not None and payload_copy.exported_to is not None
    assert payload_copy.paths == [PUBLIC_PART, "OmniPBR.mdl"]
    # the root's payload reference now points at the prepared local copy
    assert root_copy is not None
    assert root_copy.paths == [payload_copy.exported_to]
    assert root_copy.exported_to == path
    assert Path(path).parent == tmp_path / ASSET_CACHE_DIRNAME
    assert report["relinked"] == [(PAYLOAD_AS_WRITTEN, payload_copy.exported_to)]
    assert report["applied"] == [(ISAAC_DEV_PART, PUBLIC_PART)]


def test_clean_composition_closure_returns_the_source_unchanged(manager):
    FakeSdf.layers[GRIPPER_EDIT] = FakeLayer([PAYLOAD_AS_WRITTEN])
    FakeSdf.layers[PAYLOAD] = FakeLayer([PUBLIC_PART, "OmniPBR.mdl"])

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, GRIPPER_EDIT)

    assert path == GRIPPER_EDIT
    assert report["reason"] == "no isaac-dev references in the closure"
    assert FakeSdf.anonymous == []


def test_mutually_referencing_layers_terminate(manager):
    a = f"{ASSETS_ROOT}/Isaac/a.usd"
    b = f"{ASSETS_ROOT}/Isaac/b.usd"
    FakeSdf.layers[a] = FakeLayer([b])
    FakeSdf.layers[b] = FakeLayer([a])

    path, report = manager._prepared_asset_layer(FakeSdf, FakeUsdUtils, a)

    assert path == a
    assert FakeSdf.anonymous == []
