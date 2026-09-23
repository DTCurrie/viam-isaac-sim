import asyncio
import importlib.util
import sys
from pathlib import Path

# examples/ is not a package: load the script the way the other example tests do
_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "palletizer_demo.py"
_spec = importlib.util.spec_from_file_location("palletizer_demo", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
demo = importlib.util.module_from_spec(_spec)
# dataclasses resolve the script's postponed annotations through sys.modules
sys.modules[_spec.name] = demo
_spec.loader.exec_module(demo)


def test_parse_args_defaults_to_the_cells_pick_spot_and_resets():
    args = demo.parse_args(["--address", "host:8080"])

    assert args.pick_xyz_mm == (400.0, -300.0, 270.0)
    assert args.reset is True
    assert args.stop is False
    assert args.box_prop == "infeed_box_1"
    assert args.palletizer == "box-palletizer"


def test_no_reset_and_stop_flags():
    args = demo.parse_args(["--address", "host:8080", "--no-reset", "--stop"])

    assert args.reset is False
    assert args.stop is True


def test_record_line_names_the_box_outcome_landing_error_and_duration():
    line = demo.record_line(
        {
            "box_prop": "infeed_box_1",
            "outcome": "placed",
            "duration_s": 20.6,
            "target_pose_mm": {"x": 50.0, "y": 400.0},
            "measured_pose_mm": {"x": 50.03, "y": 400.04},
        }
    )

    assert line == "infeed_box_1: placed, landed 0.1 mm from its slot in 20.6 s"


class _FakePalletizer:
    """Replays a sequence of status replies, one per poll."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.polls = 0

    async def do_command(self, command):
        assert command == {"command": "status"}
        self.polls += 1
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


def test_watch_pack_prints_each_new_record_once_and_returns_the_final_state(capsys):
    placed = {"box_prop": "infeed_box_1", "outcome": "placed", "seq": 1}
    service = _FakePalletizer(
        [
            {"state": "running", "records": []},
            {"state": "running", "records": [placed]},
            {"state": "complete", "records": [placed]},
        ]
    )

    final = asyncio.run(demo.watch_pack(service, timeout_s=10.0, poll_s=0.0))

    out = capsys.readouterr().out
    assert final == "complete"
    assert out.count("infeed_box_1: placed") == 1
    assert "state: running" in out and "state: complete" in out


def test_watch_pack_reports_a_failed_packs_reason():
    service = _FakePalletizer([{"state": "failed", "records": [], "reason": "arm stalled"}])

    final = asyncio.run(demo.watch_pack(service, timeout_s=10.0, poll_s=0.0))

    assert final == "failed"
