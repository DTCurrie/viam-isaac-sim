"""Pure-translation tests for ``sequencer_client``: every parse against a
recorded fixture, and ``SequencerClient``'s payload shape and reply
translation against a fake resource."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from viam.proto.common import Pose

from isaac_module.sequencer_client import (
    BoxDimensions,
    SequencerClient,
    parse_next_box,
    parse_pack_order,
    parse_placement_report,
    parse_pose,
    parse_progress,
    parse_skip_result,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pack-sequencer-responses.json"
FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text())


class FakeResource:
    """A ``ResourceBase`` stand-in that answers from the fixture and
    records every payload it was sent, so a test can assert the wire
    shape as well as the parsed reply."""

    def __init__(self, replies: dict[str, Any]) -> None:
        self._replies = replies
        self.sent: list[dict[str, Any]] = []

    async def do_command(self, command: dict[str, Any]) -> Any:
        self.sent.append(command)
        (verb,) = command.keys()
        return self._replies[verb]


def test_parse_pose_reads_all_seven_fields() -> None:
    pose = parse_pose(FIXTURE["get_pack_order"]["pallet_pose"])
    assert isinstance(pose, Pose)
    assert pose.x == 1000.0
    assert pose.y == 500.0
    assert pose.z == 0.0
    assert pose.o_z == 1.0
    assert pose.theta == 0.0


def test_parse_pose_raises_on_missing_keys() -> None:
    with pytest.raises(ValueError, match="o_z"):
        parse_pose({"x": 1, "y": 2, "z": 3, "o_x": 0, "o_y": 0, "theta": 0})


def test_parse_next_box_seq1() -> None:
    next_box = parse_next_box(FIXTURE["next_box_seq1"])
    assert next_box.is_complete is False
    assert next_box.seq == 1
    assert next_box.col == 0
    assert next_box.row == 0
    assert next_box.layer == 0
    assert next_box.total == 8
    assert next_box.remaining == 8
    assert next_box.box_dimensions_mm == BoxDimensions(
        width_mm=200.0, length_mm=150.0, height_mm=100.0
    )
    assert next_box.approach_offset_in_pallet is not None
    assert next_box.place_end_in_world is not None
    assert next_box.place_start_in_world is not None
    # place_end is the release pose, place_start already carries the
    # sequencer's own diagonal approach standoff on top of it.
    assert next_box.place_end_in_world.z == 100.0
    assert next_box.place_start_in_world.z == 210.0


def test_parse_next_box_complete_carries_only_tallies() -> None:
    next_box = parse_next_box(FIXTURE["next_box_complete"])
    assert next_box.is_complete is True
    assert next_box.placed == 7
    assert next_box.skipped == 1
    assert next_box.seq is None
    assert next_box.pose_in_pallet is None
    assert next_box.place_end_in_world is None
    assert next_box.box_dimensions_mm is None


def test_parse_pack_order_translates_to_place_targets_in_seq_order() -> None:
    pack_order = parse_pack_order(FIXTURE["get_pack_order"])
    assert pack_order.capacity == 8
    assert pack_order.quantity == 8
    assert pack_order.overflow == 0
    assert pack_order.cols == 2
    assert pack_order.rows == 2
    assert pack_order.layers == 2
    assert pack_order.mode == "column"
    assert len(pack_order.placements) == 8

    seqs_in_order = [placement.seq for placement in pack_order.placements]
    assert seqs_in_order == list(range(1, 9))

    first = pack_order.placements[0]
    assert first.pose_in_world.x == 1150.0
    assert first.pose_in_world.y == 600.0
    assert first.pose_in_world.z == 100.0
    assert first.box_dimensions_mm == BoxDimensions(
        width_mm=200.0, length_mm=150.0, height_mm=100.0
    )
    assert first.approach_offset_in_pallet == pytest.approx((29.4744, 0.0, 110.0))

    last = pack_order.placements[-1]
    assert last.pose_in_world.z == 200.0


def test_parse_placement_report_success_and_failure() -> None:
    success = parse_placement_report(FIXTURE["report_placement_success"])
    assert success.acknowledged is True
    assert success.next_seq == 2
    assert success.last_error == ""

    failure = parse_placement_report(FIXTURE["report_placement_failure"])
    assert failure.acknowledged is True
    assert failure.next_seq == 1
    assert failure.last_error == "gripper reported no vacuum seal"


def test_parse_skip_result() -> None:
    result = parse_skip_result(FIXTURE["skip_box"])
    assert result.skipped == 3
    assert result.next_seq == 4
    assert result.placed == 2
    assert result.remaining == 5


def test_parse_progress() -> None:
    progress = parse_progress(FIXTURE["get_progress"])
    assert progress.next_seq == 4
    assert progress.done_seqs == [1, 2]
    assert progress.skipped_seqs == [3]
    assert progress.failed_seqs == []
    assert progress.complete is False


@pytest.mark.asyncio
async def test_client_next_box_sends_bare_verb() -> None:
    resource = FakeResource({"next_box": FIXTURE["next_box_seq1"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    next_box = await client.next_box()

    assert resource.sent == [{"next_box": True}]
    assert next_box.seq == 1


@pytest.mark.asyncio
async def test_client_pack_order_sends_bare_verb() -> None:
    resource = FakeResource({"get_pack_order": FIXTURE["get_pack_order"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    pack_order = await client.pack_order()

    assert resource.sent == [{"get_pack_order": True}]
    assert len(pack_order.placements) == 8


@pytest.mark.asyncio
async def test_client_report_placement_success_advances_cursor() -> None:
    resource = FakeResource({"report_placement": FIXTURE["report_placement_success"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    report = await client.report_placement(1, success=True)

    assert resource.sent == [{"report_placement": {"seq": 1, "success": True, "error": ""}}]
    assert report.next_seq == 2
    assert report.placed == 1


@pytest.mark.asyncio
async def test_client_report_placement_failure_holds_cursor() -> None:
    resource = FakeResource({"report_placement": FIXTURE["report_placement_failure"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    report = await client.report_placement(1, success=False, error="no vacuum seal")

    assert resource.sent == [
        {"report_placement": {"seq": 1, "success": False, "error": "no vacuum seal"}}
    ]
    # The cursor stays put on a failure, so the same seq comes back for a retry.
    assert report.next_seq == 1
    assert report.placed == 0


@pytest.mark.asyncio
async def test_client_reset_cursor_sends_the_bare_verb() -> None:
    resource = FakeResource({"reset_cursor": {"reset": True, "next_seq": 1}})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    await client.reset_cursor()

    assert resource.sent == [{"reset_cursor": True}]


async def test_client_skip_box_sends_seq_and_reason() -> None:
    resource = FakeResource({"skip_box": FIXTURE["skip_box"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    result = await client.skip_box(3, reason="gripper missed twice")

    assert resource.sent == [{"skip_box": {"seq": 3, "reason": "gripper missed twice"}}]
    assert result.skipped == 3
    assert result.next_seq == 4


@pytest.mark.asyncio
async def test_client_progress_sends_bare_verb() -> None:
    resource = FakeResource({"get_progress": FIXTURE["get_progress"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    progress = await client.progress()

    assert resource.sent == [{"get_progress": True}]
    assert progress.total == 8


@pytest.mark.asyncio
async def test_client_set_box_transform_sends_nested_pose_and_returns_uuid() -> None:
    resource = FakeResource({"set_box_transform": FIXTURE["set_box_transform"]})
    client = SequencerClient(resource)  # type: ignore[arg-type]
    pose = Pose(x=150.0, y=100.0, z=100.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)

    uuid = await client.set_box_transform(1, pose, parent="world")

    assert resource.sent == [
        {
            "set_box_transform": {
                "seq": 1,
                "pose": {
                    "x": 150.0,
                    "y": 100.0,
                    "z": 100.0,
                    "o_x": 0.0,
                    "o_y": 0.0,
                    "o_z": 1.0,
                    "theta": 0.0,
                },
                "parent": "world",
            }
        }
    ]
    assert uuid == "box-1-v1"


@pytest.mark.asyncio
async def test_client_clear_box_transform_sends_seq() -> None:
    resource = FakeResource({"clear_box_transform": {"acknowledged": True, "seq": 1}})
    client = SequencerClient(resource)  # type: ignore[arg-type]

    await client.clear_box_transform(1)

    assert resource.sent == [{"clear_box_transform": {"seq": 1}}]


def test_importing_this_module_registers_the_world_state_store_api():
    # The palletizer names a `rdk:service:world_state_store` dependency. The
    # SDK registers that API from `viam.services.worldstatestore`'s __init__,
    # so a module process that never imports it cannot build a client for the
    # dependency: viam-server answers `No world_state_store with name
    # "pack-sequencer" found in the registry` and nothing downstream
    # constructs. Cost a GPU run to find and one import to fix.
    #
    # Run in a subprocess importing ONLY this module. Doing the check in
    # process would import the SDK package here and pass whether or not
    # sequencer_client imports it, which is no test at all.
    source = (
        "import isaac_module.sequencer_client\n"
        "import sys\n"
        "assert 'viam.services.worldstatestore' in sys.modules, 'API never registered'\n"
        "from viam.resource.registry import Registry\n"
        "from viam.services.worldstatestore import WorldStateStore\n"
        "assert Registry.lookup_api(WorldStateStore.API) is not None\n"
    )
    repo_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0, result.stderr
