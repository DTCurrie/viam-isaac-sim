"""Import-boundary test for the ``pickcell`` package (phase 3 seam): every
submodule imports cleanly, and none of them pulls in a client-SDK-only or
mock-only dependency the module process cannot carry (isaac_module,
viam.robot.client) or the CLI-only argparse."""

import os
import subprocess
import sys
from pathlib import Path

_PICKCELL_SUBMODULES = (
    "pickcell.poses",
    "pickcell.measurement",
    "pickcell.obstacles",
    "pickcell.detector",
    "pickcell.movers",
    "pickcell.scanners",
    "pickcell.pipeline",
)

_FORBIDDEN_MODULES = ("isaac_module", "viam.robot.client", "argparse")

_CHECK_SCRIPT = f"""
import sys
for name in {_PICKCELL_SUBMODULES!r}:
    __import__(name)
forbidden = [name for name in {_FORBIDDEN_MODULES!r} if name in sys.modules]
assert not forbidden, f"forbidden modules imported: {{forbidden}}"
print("OK")
"""


def test_pickcell_submodules_import_without_forbidden_dependencies():
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
