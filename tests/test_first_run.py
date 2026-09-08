"""Tests for first_run.sh: the guard branches and the FIRST_RUN_DRY_RUN mode.

Every case runs `bash first_run.sh` as a subprocess with VIAM_MODULE_DATA pointed at a
temp dir, mirroring how run.sh's marker file is produced in production. The linux-only
install branches are exercised on every host, including macOS, via a `uname` shim placed
first on PATH that reports Linux/x86_64 and otherwise defers to the real `uname`.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRST_RUN_SH = REPO_ROOT / "first_run.sh"

IS_LINUX_X86_64 = platform.system() == "Linux" and platform.machine() == "x86_64"


def _run(env: dict, path: str | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **env}
    if path is not None:
        full_env["PATH"] = path
    return subprocess.run(
        ["bash", str(FIRST_RUN_SH)],
        capture_output=True,
        text=True,
        env=full_env,
        check=False,
        cwd=REPO_ROOT,
    )


def _uname_shim_path(bin_dir: Path) -> str:
    """Write a `uname` stub reporting Linux/x86_64 into `bin_dir`.

    Returns a PATH with that directory first."""
    real_uname = shutil.which("uname")
    assert real_uname is not None, "uname must be on PATH to build the shim"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "uname"
    stub.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  -s) echo "Linux" ;;\n'
        '  -m) echo "x86_64" ;;\n'
        f'  *) exec "{real_uname}" "$@" ;;\n'
        "esac\n"
    )
    stub.chmod(0o755)
    return f"{bin_dir}:{os.environ['PATH']}"


def test_non_linux_or_non_x86_64_guard(tmp_path: Path) -> None:
    result = _run({"VIAM_MODULE_DATA": str(tmp_path)})

    if IS_LINUX_X86_64:
        assert result.returncode == 0
        assert "not linux/x86_64" not in result.stdout
    else:
        assert result.returncode == 0
        assert "not linux/x86_64" in result.stdout


def test_fast_path_marks_already_installed_venv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = _uname_shim_path(tmp_path / "bin")

    venv_bin = data_dir / "isaac-venv" / "bin"
    venv_bin.mkdir(parents=True)
    stub_python = venv_bin / "python"
    stub_python.write_text("#!/bin/sh\nexit 0\n")
    stub_python.chmod(0o755)

    result = _run({"VIAM_MODULE_DATA": str(data_dir)}, path=path)

    assert result.returncode == 0
    assert "already installed" in result.stdout
    marker = data_dir / "isaac_python"
    assert marker.read_text().strip() == str(stub_python)


def test_dry_run_logs_planned_steps_without_installing(tmp_path: Path) -> None:
    path = _uname_shim_path(tmp_path / "bin")

    result = _run({"VIAM_MODULE_DATA": str(tmp_path), "FIRST_RUN_DRY_RUN": "1"}, path=path)

    assert result.returncode == 0
    assert "apt" in result.stdout
    assert "580" in result.stdout
    assert "isaacsim==" in result.stdout
    assert str(tmp_path / "isaac-venv") in result.stdout
    assert str(tmp_path / "isaac_python") in result.stdout
    assert not (tmp_path / "isaac_python").exists()
    assert os.listdir(tmp_path) == ["bin"]


def test_explicit_isaac_python_skips_install(tmp_path: Path) -> None:
    path = _uname_shim_path(tmp_path / "bin")

    result = _run({"VIAM_MODULE_DATA": str(tmp_path), "ISAAC_PYTHON": "/anything"}, path=path)

    assert result.returncode == 0
    assert "already configured" in result.stdout


def test_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(FIRST_RUN_SH)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
