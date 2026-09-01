"""R-4 / OQ-4 fallback 1: re-pointing the 2F-85's `isaac-dev` references at the
public assets root (sim_manager._rewritten_references), driven with fakes."""

from isaac_module.sim_manager import ASSETS_PATH_MARKER, _rewritten_references

ASSETS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/"
PART = "/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_fingertipsstep_JFD.usd"
ISAAC_DEV = f"omniverse://isaac-dev.ov.nvidia.com{PART}"
ALREADY_PUBLIC = f"{ASSETS_ROOT.rstrip('/')}/Isaac/Robots/Robotiq/2F-85/parts/base.usd"


class FakeReference:
    def __init__(self, asset_path: str, prim_path: str = "", layer_offset: str = "offset") -> None:
        self.assetPath = asset_path
        self.primPath = prim_path
        self.layerOffset = layer_offset


class FakeSdf:
    @staticmethod
    def Reference(asset_path, prim_path, layer_offset):
        return FakeReference(asset_path, prim_path, layer_offset)


def _exists_all(_path: str) -> bool:
    return True


def test_isaac_dev_reference_is_rewritten_onto_the_assets_root_keeping_prim_path():
    items = [FakeReference(ISAAC_DEV, "/Mesh")]
    new_items, pairs, missing = _rewritten_references(FakeSdf, items, ASSETS_ROOT, _exists_all)
    assert pairs == [(ISAAC_DEV, ASSETS_ROOT.rstrip("/") + PART)]
    assert new_items[0].assetPath == ASSETS_ROOT.rstrip("/") + PART
    assert new_items[0].primPath == "/Mesh"
    assert new_items[0].layerOffset == "offset"
    assert ASSETS_PATH_MARKER in new_items[0].assetPath


def test_public_reference_is_left_untouched():
    items = [FakeReference(ALREADY_PUBLIC)]
    new_items, pairs, missing = _rewritten_references(FakeSdf, items, ASSETS_ROOT, _exists_all)
    assert pairs == []
    assert new_items[0] is items[0]


def test_missing_bucket_part_is_kept_not_rewritten():
    items = [FakeReference(ISAAC_DEV)]
    new_items, pairs, missing = _rewritten_references(FakeSdf, items, ASSETS_ROOT, lambda _p: False)
    assert pairs == []
    assert new_items[0] is items[0]
    assert missing == [ASSETS_ROOT.rstrip("/") + PART]


def test_unknown_existence_is_tried():
    items = [FakeReference(ISAAC_DEV)]
    _new_items, pairs, missing = _rewritten_references(FakeSdf, items, ASSETS_ROOT, lambda _p: None)
    assert len(pairs) == 1
    assert missing == []


def test_mixed_list_preserves_order_and_count():
    items = [FakeReference(ALREADY_PUBLIC), FakeReference(ISAAC_DEV), FakeReference(ALREADY_PUBLIC)]
    new_items, pairs, missing = _rewritten_references(FakeSdf, items, ASSETS_ROOT, _exists_all)
    assert len(new_items) == 3
    assert new_items[0] is items[0] and new_items[2] is items[2]
    assert len(pairs) == 1
