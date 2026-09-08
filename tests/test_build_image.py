"""Dry-run checks for `provisioning/build-image.sh`.

No real `gcloud` call is ever made: `--dry-run` prints every command it would
run instead of executing it, and these tests only ever exercise that mode.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "provisioning" / "build-image.sh"


def run_dry() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--project", "test-project"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_dry_run_exits_zero() -> None:
    result = run_dry()
    assert result.returncode == 0, result.stderr


def test_dry_run_prints_expected_commands_in_order() -> None:
    result = run_dry()
    output = result.stdout

    create_instance_idx = output.find("instances create")
    assert create_instance_idx != -1
    create_instance_line_end = output.find("\n", create_instance_idx)
    create_instance_line = output[create_instance_idx:create_instance_line_end]
    assert "--accelerator" in create_instance_line
    assert "nvidia-l4" in create_instance_line
    assert "ubuntu-2404-lts-amd64" in create_instance_line

    first_run_idx = output.find("bash first_run.sh", create_instance_idx)
    assert first_run_idx != -1
    first_run_line_start = output.rfind("\n", 0, first_run_idx)
    first_run_line_end = output.find("\n", first_run_idx)
    first_run_line = output[first_run_line_start:first_run_line_end]
    assert "VIAM_MODULE_DATA=/opt/viam-isaac-sim" in first_run_line

    agent_idx = output.find("preinstall.sh", first_run_idx)
    assert agent_idx != -1

    images_create_idx = output.find("images create", agent_idx)
    assert images_create_idx != -1
    images_create_line_end = output.find("\n", images_create_idx)
    images_create_line = output[images_create_idx:images_create_line_end]
    assert "--family viam-isaac-sim" in images_create_line

    delete_idx = output.find("instances delete", images_create_idx)
    assert delete_idx != -1

    assert output.count("images create") == 1
    assert output.count("instances delete") == 1
