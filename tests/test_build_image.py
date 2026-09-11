import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "provisioning" / "build-image.sh"
CREATE_SIM_MACHINE_SCRIPT = REPO / "provisioning" / "create-sim-machine.sh"


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

    scp_section = output[create_instance_line_end:first_run_idx]
    assert "src" in scp_section
    assert "warm_shader_cache.py" in scp_section
    assert "isaac-sim-block-sorting.json" in scp_section

    second_first_run_idx = output.find("bash first_run.sh", first_run_line_end)
    assert second_first_run_idx != -1

    warmup_idx = output.find("warm_shader_cache.py", second_first_run_idx)
    assert warmup_idx != -1
    warmup_line_start = output.rfind("\n", 0, warmup_idx)
    warmup_line_end = output.find("\n", warmup_idx)
    warmup_line = output[warmup_line_start:warmup_line_end]
    assert "sudo" in warmup_line
    assert "VIAM_MODULE_DATA=/opt/viam-isaac-sim" in warmup_line
    assert "isaac-venv/bin/python" in warmup_line
    assert "--fragment fragments/isaac-sim-block-sorting.json" in warmup_line

    ssh_lines_with_warmup = [
        line for line in output.splitlines() if "ssh" in line and "warm_shader_cache.py" in line
    ]
    assert len(ssh_lines_with_warmup) == 1

    agent_idx = output.find("preinstall.sh", warmup_idx)
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


# ----------------------------------------------------------------------
# create-sim-machine.sh: --open-livestream opens TCP 49100 + UDP 47998
# ----------------------------------------------------------------------


def test_create_sim_machine_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(CREATE_SIM_MACHINE_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_create_sim_machine_has_open_livestream_flag() -> None:
    text = CREATE_SIM_MACHINE_SCRIPT.read_text()
    assert "--open-livestream" in text


def test_create_sim_machine_open_livestream_opens_the_livestream_ports() -> None:
    text = CREATE_SIM_MACHINE_SCRIPT.read_text()
    assert "tcp:49100,udp:47998" in text


def test_create_sim_machine_open_livestream_creates_the_firewall_rule_idempotently() -> None:
    text = CREATE_SIM_MACHINE_SCRIPT.read_text()
    assert "firewall-rules describe" in text
    assert "firewall-rules create" in text


def test_create_sim_machine_open_livestream_tags_the_instance() -> None:
    text = CREATE_SIM_MACHINE_SCRIPT.read_text()
    assert "add-tags" in text
    assert "--target-tags" in text
