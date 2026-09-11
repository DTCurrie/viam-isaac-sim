import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import warm_shader_cache as wsc  # noqa: E402

from isaac_module.errors import SimTimeoutError  # noqa: E402
from isaac_module.sim_manager import SimManager  # noqa: E402

FRAGMENT_PATH = REPO_ROOT / "fragments" / "isaac-sim-block-sorting.json"


def _fragment() -> dict[str, Any]:
    return json.loads(FRAGMENT_PATH.read_text())


def _component(plan: wsc.WarmupPlan, name: str) -> wsc.WarmupComponent:
    return next(c for c in plan.components if c.name == name)


class TestPlanFromFragment:
    def test_sim_config_overrides(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        assert plan.sim_config.headless is True
        assert plan.sim_config.livestream is False
        assert plan.sim_config.wait_for_finalizer is False
        assert plan.sim_config.boot_timeout == wsc.WARMUP_BOOT_TIMEOUT_S
        assert plan.sim_config.mock is False

    def test_mock_flag_forwarded(self) -> None:
        plan = wsc.plan_from_fragment(_fragment(), mock=True)
        assert plan.sim_config.mock is True

    def test_props_count_and_lighting(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        assert len(plan.sim_config.props) == 29
        assert plan.sim_config.lighting is not None
        assert plan.sim_config.lighting["dome"]["intensity"] == 1000

    def test_variables_resolved(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        assert "$variable" not in json.dumps(plan.sim_config.props)

    def test_component_order_and_kinds(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        assert [c.name for c in plan.components] == [
            "pick-arm",
            "pick-grip",
            "scene-cam",
            "side-cam",
            "wrist-cam",
        ]
        assert [c.kind for c in plan.components] == [
            "arm",
            "gripper",
            "camera",
            "camera",
            "camera",
        ]

    def test_gripper_default_parent_prim(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        grip = _component(plan, "pick-grip")
        assert grip.attrs["parent_prim"] == "/World/pick_arm/wrist_3_link"

    def test_scene_cam_dimensions(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        scene_cam = _component(plan, "scene-cam")
        assert scene_cam.attrs["width"] == 1280
        assert scene_cam.attrs["height"] == 720

    def test_side_cam_dimensions_and_depth(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        side_cam = _component(plan, "side-cam")
        assert side_cam.attrs["depth"] is True
        assert side_cam.attrs["width"] == 848

    def test_wrist_cam_keeps_own_parent_prim(self) -> None:
        plan = wsc.plan_from_fragment(_fragment())
        wrist_cam = _component(plan, "wrist-cam")
        assert wrist_cam.attrs["parent_prim"] == "/World/pick_arm/wrist_3_link"

    def test_no_world_component_raises(self) -> None:
        fragment = _fragment()
        fragment["components"] = [
            c for c in fragment["components"] if not c.get("model", "").endswith(":world")
        ]
        with pytest.raises(ValueError, match=r"no component whose model ends with"):
            wsc.plan_from_fragment(fragment)

    def test_only_world_present_raises(self) -> None:
        fragment = _fragment()
        world_component = next(
            c for c in fragment["components"] if c.get("model", "").endswith(":world")
        )
        fragment["components"] = [world_component]
        with pytest.raises(ValueError, match=r"no arm, gripper or camera component"):
            wsc.plan_from_fragment(fragment)


class TestRetryThroughSlowSteps:
    def test_retries_sim_timeout_then_succeeds(self) -> None:
        attempts = {"count": 0}

        def call() -> int:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise SimTimeoutError("still cold")
            return 7

        assert wsc.retry_through_slow_steps(call, "warm-up") == 7
        assert attempts["count"] == 3

    def test_other_exception_propagates_immediately(self) -> None:
        attempts = {"count": 0}

        def call() -> int:
            attempts["count"] += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match=r"boom"):
            wsc.retry_through_slow_steps(call, "warm-up")
        assert attempts["count"] == 1

    def test_reraises_after_timeout_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wsc, "CREATE_RETRY_TIMEOUT_S", 0.0)

        def call() -> None:
            raise SimTimeoutError("always cold")

        with pytest.raises(SimTimeoutError):
            wsc.retry_through_slow_steps(call, "warm-up")


class TestMainEndToEnd:
    def test_main_succeeds_against_default_fragment(self) -> None:
        exit_code = wsc.main(["--mock", "--frames", "20"], sim_factory=SimManager)
        assert exit_code == 0

    def test_main_returns_error_for_missing_fragment(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        exit_code = wsc.main(["--mock", "--fragment", str(missing)], sim_factory=SimManager)
        assert exit_code == wsc.EXIT_ERROR
