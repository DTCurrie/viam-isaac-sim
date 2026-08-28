import asyncio
import importlib.metadata
import re
from pathlib import Path

from isaac_module.models.arm import IsaacArm

_REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"

_EXPECTED_LINES = [
    "viam-sdk>=0.80.0,<0.81",
    "numpy==1.26.0",
    "Pillow==11.2.1",
]


def test_arm_instantiates_without_abstract_method_error():
    arm = IsaacArm("x")
    assert isinstance(arm, IsaacArm)


def test_get_3d_models_returns_empty_mapping():
    arm = IsaacArm("x")
    assert asyncio.run(arm.get_3d_models()) == {}


def test_requirements_pins_exact():
    lines = _REQUIREMENTS_PATH.read_text().splitlines()
    for expected in _EXPECTED_LINES:
        assert expected in lines, f"missing pin: {expected!r} in {lines}"


def test_viam_sdk_version_satisfies_floor():
    version = importlib.metadata.version("viam-sdk")
    try:
        from packaging.version import Version

        installed = Version(version)
        assert installed >= Version("0.80.0")
        assert installed < Version("0.81")
    except ImportError:
        match = re.match(r"(\d+)\.(\d+)", version)
        assert match, f"unparseable version: {version!r}"
        major, minor = int(match.group(1)), int(match.group(2))
        assert (major, minor) >= (0, 80)
        assert (major, minor) < (0, 81)
