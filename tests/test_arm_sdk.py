import asyncio
import importlib.metadata
import re
from pathlib import Path
from unittest.mock import AsyncMock

from viam.proto.component.arm import (
    MoveOptions,
    MoveThroughJointPositionsRequest,
    MoveThroughJointPositionsResponse,
)

from isaac_module import sdk_patches
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


class _FakeStream:
    def __init__(self, request):
        self._request = request
        self.deadline = None
        self.metadata = {}
        self.sent = None

    async def recv_message(self):
        return self._request

    async def send_message(self, message):
        self.sent = message


class _FakeService:
    def __init__(self, arm):
        self._arm = arm

    def get_resource(self, name):
        return self._arm


def test_move_through_joint_positions_forwards_options():
    arm = AsyncMock()
    service = _FakeService(arm)
    request = MoveThroughJointPositionsRequest(name="test-arm")
    request.options.max_vel_degs_per_sec = 5.0
    stream = _FakeStream(request)

    asyncio.run(sdk_patches._move_through_joint_positions(service, stream))

    arm.move_through_joint_positions.assert_awaited_once()
    _, kwargs = arm.move_through_joint_positions.call_args
    assert kwargs["options"] == MoveOptions(max_vel_degs_per_sec=5.0)
    assert isinstance(stream.sent, MoveThroughJointPositionsResponse)


def test_move_through_joint_positions_forwards_none_options_when_unset():
    arm = AsyncMock()
    service = _FakeService(arm)
    request = MoveThroughJointPositionsRequest(name="test-arm")
    stream = _FakeStream(request)

    asyncio.run(sdk_patches._move_through_joint_positions(service, stream))

    arm.move_through_joint_positions.assert_awaited_once()
    _, kwargs = arm.move_through_joint_positions.call_args
    assert kwargs["options"] is None


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
