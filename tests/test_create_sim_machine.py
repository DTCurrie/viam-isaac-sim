import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import create_sim_machine as csm  # noqa: E402

REAL_CONFIG_PATH = REPO_ROOT / "examples" / "configs" / "real-ur5e-cell.json"


def _args(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        name="sim-1",
        location_id="loc",
        project=None,
        zone=csm.DEFAULT_ZONE,
        machine_type=csm.DEFAULT_MACHINE_TYPE,
        image_family=csm.DEFAULT_IMAGE_FAMILY,
        allow_unmatched=False,
        dry_run=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_plan_gcloud_args_carry_launch_flags() -> None:
    args = _args(
        name="sim-1", zone="us-west1-a", machine_type="g2-standard-4", image_family="viam-isaac-sim"
    )
    plan = csm.build_plan(args, {"components": []}, "part-1", "s3cr3t")

    assert "us-west1-a" in plan.gcloud_args
    assert "g2-standard-4" in plan.gcloud_args
    assert "viam-isaac-sim" in plan.gcloud_args
    assert "type=nvidia-l4,count=1" in plan.gcloud_args
    assert any(a.startswith("startup-script=") for a in plan.gcloud_args)
    assert "sim-1" in plan.gcloud_args
    assert "--project" not in plan.gcloud_args


def test_build_plan_credentials_carry_part_id_secret_and_app_address() -> None:
    args = _args()
    plan = csm.build_plan(args, {"components": []}, "part-42", "top-secret")

    document = plan.credentials["document"]
    assert document["cloud"]["id"] == "part-42"
    assert document["cloud"]["secret"] == "top-secret"
    assert document["cloud"]["app_address"] == csm.APP_ADDRESS
    assert "top-secret" in plan.credentials["startup_script"]


def test_build_plan_omits_project_flags_without_a_project() -> None:
    args = _args(project=None)
    plan = csm.build_plan(args, {"components": []}, "part-1", "secret")

    assert "--project" not in plan.gcloud_args
    assert "--image-project" not in plan.gcloud_args


def test_build_plan_includes_both_project_flags_with_a_project() -> None:
    args = _args(project="my-project")
    plan = csm.build_plan(args, {"components": []}, "part-1", "secret")

    assert plan.gcloud_args.count("--project") == 1
    assert plan.gcloud_args.count("--image-project") == 1
    assert "my-project" in plan.gcloud_args


def test_dry_run_exits_zero_prints_no_secret_and_never_imports_sdk() -> None:
    sys.modules.pop("viam.app.viam_client", None)

    exit_code = csm.main(
        [str(REAL_CONFIG_PATH), "--name", "sim-1", "--location-id", "loc", "--dry-run"]
    )

    assert exit_code == 0
    assert "viam.app.viam_client" not in sys.modules


def test_dry_run_prints_resolved_config_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = csm.main(
        [str(REAL_CONFIG_PATH), "--name", "sim-1", "--location-id", "loc", "--dry-run"]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "pick-arm" in output
    assert "pick-grip" in output
    assert "wrist-cam" in output
    assert "top-secret" not in output
    assert "<secret>" in output


def test_dry_run_with_motor_config_reports_a_placeholder(tmp_path: Path, capsys) -> None:
    config = {
        "components": [
            {
                "name": "conveyor",
                "api": "rdk:component:motor",
                "model": "acme:conveyor:v1",
                "attributes": {},
            }
        ]
    }
    config_path = tmp_path / "with-motor.json"
    config_path.write_text(json.dumps(config))

    exit_code = csm.main([str(config_path), "--name", "sim-1", "--location-id", "loc", "--dry-run"])

    assert exit_code == 0
    assert "placeholders: conveyor" in capsys.readouterr().out


@dataclass
class _FakePart:
    id: str
    name: str
    secret: str
    main_part: bool
    last_access: Any


class _FakeAppClient:
    def __init__(self, parts_sequence: list[list[_FakePart]]) -> None:
        self._parts_sequence = parts_sequence
        self.calls: list[str] = []
        self.update_calls: list[tuple[str, str, str | None]] = []

    async def new_robot(self, name: str, location_id: str) -> str:
        self.calls.append("new_robot")
        return "robot-1"

    async def get_robot_parts(self, robot_id: str) -> list[_FakePart]:
        self.calls.append("get_robot_parts")
        if len(self._parts_sequence) > 1:
            return self._parts_sequence.pop(0)
        return self._parts_sequence[0]

    async def update_robot_part(
        self,
        part_id: str,
        name: str,
        robot_config: dict[str, Any] | None = None,
        last_known_update: Any = None,
        robot_config_json: str | None = None,
    ) -> None:
        self.calls.append("update_robot_part")
        self.update_calls.append((part_id, name, robot_config_json))


class _FakeViamClient:
    def __init__(self, app_client: _FakeAppClient) -> None:
        self.app_client = app_client
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_sdk(monkeypatch: pytest.MonkeyPatch, fake_app_client: _FakeAppClient) -> None:
    fake_viam_client = _FakeViamClient(fake_app_client)

    async def fake_create_from_dial_options(
        dial_options: Any, app_url: Any = None
    ) -> _FakeViamClient:
        return fake_viam_client

    fake_viam_module = SimpleNamespace(
        ViamClient=SimpleNamespace(create_from_dial_options=fake_create_from_dial_options)
    )
    fake_dial_module = SimpleNamespace(
        DialOptions=SimpleNamespace(with_api_key=lambda key, key_id: object())
    )
    monkeypatch.setitem(sys.modules, "viam.app.viam_client", fake_viam_module)
    monkeypatch.setitem(sys.modules, "viam.rpc.dial", fake_dial_module)
    monkeypatch.setenv("VIAM_API_KEY", "key")
    monkeypatch.setenv("VIAM_API_KEY_ID", "key-id")


def test_real_run_calls_sdk_then_gcloud_then_polls_then_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import datetime as dt

    offline_part = _FakePart(
        id="part-1", name="main", secret="s3cr3t", main_part=True, last_access=None
    )
    online_part = _FakePart(
        id="part-1",
        name="main",
        secret="s3cr3t",
        main_part=True,
        last_access=dt.datetime.utcnow(),
    )
    fake_app_client = _FakeAppClient([[offline_part], [offline_part], [online_part]])
    _patch_sdk(monkeypatch, fake_app_client)

    subprocess_calls: list[Any] = []

    def fake_run(args: Any, check: bool = False) -> Any:
        subprocess_calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(csm.subprocess, "run", fake_run)
    monkeypatch.setattr(csm.time, "monotonic", _CountingClock())
    monkeypatch.setattr(csm, "ONLINE_POLL_S", 5.0)

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(csm.asyncio, "sleep", fake_sleep)

    exit_code = csm.main([str(REAL_CONFIG_PATH), "--name", "sim-1", "--location-id", "loc"])

    assert exit_code == 0
    assert subprocess_calls, "gcloud was never invoked"
    assert fake_app_client.calls[0] == "new_robot"
    assert fake_app_client.calls[1] == "get_robot_parts"
    assert "update_robot_part" in fake_app_client.calls
    assert fake_app_client.calls.index("update_robot_part") == len(fake_app_client.calls) - 1
    assert fake_app_client.update_calls[0][0] == "part-1"


class _CountingClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 1.0
        return self._value


def test_real_run_exits_three_when_part_never_comes_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offline_part = _FakePart(
        id="part-1", name="main", secret="s3cr3t", main_part=True, last_access=None
    )
    fake_app_client = _FakeAppClient([[offline_part]])
    _patch_sdk(monkeypatch, fake_app_client)

    monkeypatch.setattr(
        csm.subprocess, "run", lambda args, check=False: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(csm, "ONLINE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(csm, "ONLINE_POLL_S", 0.01)

    exit_code = csm.main([str(REAL_CONFIG_PATH), "--name", "sim-1", "--location-id", "loc"])

    assert exit_code == 3


def test_wrapper_script_has_valid_syntax() -> None:
    script = REPO_ROOT / "provisioning" / "create-sim-machine.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
