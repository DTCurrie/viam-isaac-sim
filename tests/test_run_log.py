import isaac_module.run_log as run_log
from isaac_module.run_log import (
    LoopRecord,
    PickRecord,
    RollingLog,
    current_rss_mb,
    loop_seed,
    success_rate,
)


def make_pick(outcome: str, name: str = "block_0", reason: str | None = None) -> PickRecord:
    return PickRecord(
        name=name,
        color="red",
        outcome=outcome,
        attempts=1,
        duration_s=1.5,
        reason=reason,
    )


def test_pick_record_to_dict_omits_reason_when_none() -> None:
    pick = make_pick("placed")

    result = pick.to_dict()

    assert result == {
        "name": "block_0",
        "color": "red",
        "outcome": "placed",
        "attempts": 1,
        "duration_s": 1.5,
    }
    assert "reason" not in result


def test_pick_record_to_dict_includes_reason_when_set() -> None:
    pick = make_pick("failed", reason="grasp slipped")

    result = pick.to_dict()

    assert result["reason"] == "grasp slipped"


def test_loop_record_derived_counts_over_mixed_picks() -> None:
    loop = LoopRecord(
        record_id=1,
        loop=1,
        seed=42,
        duration_s=10.0,
        passes=2,
        picks=(
            make_pick("placed", name="a"),
            make_pick("placed", name="b"),
            make_pick("skipped_oversize", name="c"),
            make_pick("failed", name="d"),
        ),
    )

    assert loop.placed == 2
    assert loop.skipped_oversize == 1
    assert loop.failed == 1


def test_loop_record_to_dict_includes_all_fields_and_picks() -> None:
    picks = (make_pick("placed", name="a"), make_pick("failed", name="b"))
    loop = LoopRecord(record_id=7, loop=3, seed=99, duration_s=5.0, passes=1, picks=picks)

    result = loop.to_dict()

    assert result["record_id"] == 7
    assert result["loop"] == 3
    assert result["seed"] == 99
    assert result["duration_s"] == 5.0
    assert result["passes"] == 1
    assert result["placed"] == 1
    assert result["skipped_oversize"] == 0
    assert result["failed"] == 1
    assert result["picks"] == [pick.to_dict() for pick in picks]


def make_loop(record_id: int) -> LoopRecord:
    return LoopRecord(record_id=record_id, loop=record_id, seed=None, duration_s=0.0, passes=0)


def test_rolling_log_keeps_exactly_window_at_window_appends() -> None:
    log = RollingLog(window=3)
    for i in range(3):
        log.append(make_loop(i))

    records = log.records()

    assert len(records) == 3
    assert [r.record_id for r in records] == [0, 1, 2]


def test_rolling_log_drops_oldest_past_window() -> None:
    log = RollingLog(window=3)
    for i in range(4):
        log.append(make_loop(i))

    records = log.records()

    assert len(records) == 3
    assert [r.record_id for r in records] == [1, 2, 3]


def test_loop_seed_determinism_and_advance() -> None:
    assert loop_seed(100, 0) == 100
    assert loop_seed(100, 1) == 101
    assert loop_seed(100, 5) == 105


def test_success_rate_none_at_zero_zero() -> None:
    assert success_rate(0, 0) is None


def test_success_rate_all_placed() -> None:
    assert success_rate(5, 0) == 1.0


def test_success_rate_mixed_excludes_oversize() -> None:
    assert success_rate(3, 1) == 0.75


def test_success_rate_all_failed() -> None:
    assert success_rate(0, 4) == 0.0


def test_current_rss_mb_reads_a_real_value_on_this_host() -> None:
    """A CPython process with the viam SDK imported resides in tens of MiB at
    minimum, and no test host has anywhere near 100 GiB (100_000 MiB)
    resident, so both bounds catch a broken reading without being flaky."""
    rss = current_rss_mb()

    assert rss is not None
    assert 10 < rss < 100_000


def test_current_rss_mb_returns_none_when_the_platform_reading_fails(monkeypatch) -> None:
    monkeypatch.setattr(run_log.sys, "platform", "linux")

    def broken_read_statm() -> str:
        raise OSError("no such file")

    monkeypatch.setattr(run_log, "_read_statm", broken_read_statm)

    assert current_rss_mb() is None


def test_loop_record_to_dict_round_trips_rss_mb_and_defaults_to_none() -> None:
    with_rss = LoopRecord(
        record_id=1, loop=1, seed=1, duration_s=1.0, passes=1, picks=(), rss_mb=42.5
    )
    assert with_rss.to_dict()["rss_mb"] == 42.5

    without_rss = LoopRecord(record_id=2, loop=2, seed=2, duration_s=1.0, passes=1, picks=())
    assert without_rss.to_dict()["rss_mb"] is None


def test_loop_record_error_field_round_trips_only_when_set():
    from isaac_module.run_log import LoopRecord

    errored = LoopRecord(
        record_id=1, loop=1, seed=5, duration_s=1.0, passes=1, picks=(), error="RuntimeError: x"
    )
    assert errored.to_dict()["error"] == "RuntimeError: x"
    clean = LoopRecord(record_id=2, loop=2, seed=6, duration_s=1.0, passes=1, picks=())
    assert "error" not in clean.to_dict()
