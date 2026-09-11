from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRST_RUN_SH = REPO_ROOT / "first_run.sh"
RUN_SH = REPO_ROOT / "run.sh"
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"

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


# -- run.sh: the requirements.txt hash marker gates the pip install --------


def _stub_python(bin_dir: Path, call_log: Path) -> Path:
    """A `python` stub standing in for pip and for `src/main.py` alike:
    every invocation's args are appended to call_log, then it exits 0."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "python"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{call_log}"\nexit 0\n')
    stub.chmod(0o755)
    return stub


def _run_run_sh(env: dict) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **env}
    return subprocess.run(
        ["bash", str(RUN_SH)],
        capture_output=True,
        text=True,
        env=full_env,
        check=False,
        cwd=REPO_ROOT,
    )


def test_run_sh_installs_once_then_skips_on_an_unchanged_requirements_hash(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    call_log = tmp_path / "calls.log"
    stub = _stub_python(tmp_path / "bin", call_log)
    env = {"VIAM_MODULE_DATA": str(data_dir), "ISAAC_PYTHON": str(stub)}

    first = _run_run_sh(env)
    assert first.returncode == 0, first.stderr
    assert call_log.read_text().count(" install ") == 1

    second = _run_run_sh(env)
    assert second.returncode == 0, second.stderr
    # marker matches requirements.txt's hash: no second install
    assert call_log.read_text().count(" install ") == 1
    assert "skipping pip install" in second.stderr

    marker = data_dir / "requirements.sha256"
    assert marker.read_text().strip() == hashlib.sha256(REQUIREMENTS_TXT.read_bytes()).hexdigest()


def test_run_sh_reinstalls_when_the_marker_is_stale(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "requirements.sha256").write_text("stale-hash-not-matching-anything")
    call_log = tmp_path / "calls.log"
    stub = _stub_python(tmp_path / "bin", call_log)

    result = _run_run_sh({"VIAM_MODULE_DATA": str(data_dir), "ISAAC_PYTHON": str(stub)})

    assert result.returncode == 0, result.stderr
    assert call_log.read_text().count(" install ") == 1
    marker = data_dir / "requirements.sha256"
    assert marker.read_text().strip() == hashlib.sha256(REQUIREMENTS_TXT.read_bytes()).hexdigest()


def test_run_sh_falls_back_to_the_module_directory_when_module_data_is_unset(
    tmp_path: Path,
) -> None:
    call_log = tmp_path / "calls.log"
    stub = _stub_python(tmp_path / "bin", call_log)
    marker = REPO_ROOT / "requirements.sha256"
    assert not marker.exists(), "a stray marker in the repo would make this test a false pass"

    full_env = {**os.environ, "ISAAC_PYTHON": str(stub)}
    full_env.pop("VIAM_MODULE_DATA", None)

    try:
        result = subprocess.run(
            ["bash", str(RUN_SH)],
            capture_output=True,
            text=True,
            env=full_env,
            check=False,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert call_log.read_text().count(" install ") == 1
        assert marker.exists()
    finally:
        marker.unlink(missing_ok=True)


def test_run_sh_requirements_hash_marker_and_command_are_pinned() -> None:
    source = RUN_SH.read_text()
    assert "requirements.sha256" in source
    assert "sha256sum requirements.txt" in source


def test_run_sh_syntax_is_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUN_SH)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
