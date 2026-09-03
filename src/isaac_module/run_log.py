"""Pure loop/telemetry records for the conductor's multi-loop runs (phase 5).

No viam imports. The shapes here are the status contract: the conductor
serializes these via ``to_dict`` into its ``status`` DoCommand, and the
sorter sensor forwards them verbatim, deduplicating on ``record_id``.

Policy constants shared by the conductor and its tests:

- ``MAX_ATTEMPTS_PER_BLOCK`` = 2: a failed grasp is retried exactly once
  (attempt 2), then the block is terminally ``failed`` for that loop.
- ``LOOP_RECORD_WINDOW`` = 50: ``RollingLog`` keeps the last 50 loop
  records and drops older ones, bounding memory in continuous mode.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

_BYTES_PER_KIB = 1024
_BYTES_PER_MIB = 1024 * 1024


def _read_statm() -> str:
    """Reads ``/proc/self/statm``, isolated so tests can inject a failure."""
    with open("/proc/self/statm") as statm_file:
        return statm_file.read()


def current_rss_mb() -> float | None:
    """Resident set size of this process in MiB, or ``None`` if unreadable.

    On Linux (where the module runs) this is the current RSS, read as the
    resident-page count from ``/proc/self/statm``. Elsewhere it falls back to
    ``resource.getrusage``, whose ``ru_maxrss`` is the process's PEAK RSS, in
    KiB on Linux but bytes on macOS. A soak reads the Linux value.
    """
    if sys.platform == "linux":
        try:
            resident_pages = int(_read_statm().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / _BYTES_PER_MIB
        except (OSError, ValueError, IndexError):
            return None
    try:
        import resource

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return max_rss / _BYTES_PER_MIB
        return max_rss / _BYTES_PER_KIB
    except (OSError, ValueError, ImportError):
        return None


MAX_ATTEMPTS_PER_BLOCK = 2
LOOP_RECORD_WINDOW = 50
# a loop killed by a transient failure (a dropped gRPC stream, GPU run 7) is
# recorded and skipped; this many in a row means the cell is actually down
MAX_CONSECUTIVE_LOOP_ERRORS = 3

OUTCOME_PLACED = "placed"
OUTCOME_SKIPPED_OVERSIZE = "skipped_oversize"
OUTCOME_FAILED = "failed"


@dataclass(frozen=True)
class PickRecord:
    """One block's terminal outcome within one loop."""

    name: str  # resolved prim name
    color: str
    outcome: str  # one of the OUTCOME_* literals
    attempts: int  # 1..MAX_ATTEMPTS_PER_BLOCK
    duration_s: float  # wall time across all attempts of this block
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "color": self.color,
            "outcome": self.outcome,
            "attempts": self.attempts,
            "duration_s": self.duration_s,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class LoopRecord:
    """One completed loop: its seed, wall time, and per-pick records."""

    record_id: int  # monotonic for the module's lifetime, never reused
    loop: int  # 1-based loop number within its run
    seed: int | None
    duration_s: float
    passes: int
    picks: tuple[PickRecord, ...] = field(default=())
    # set when the loop died on an exception instead of completing; its picks
    # are empty and the loop counted toward loops_errored, not loops_completed
    error: str | None = None
    # resident set size of the module process in MiB, sampled when the loop's
    # record is cut (phase-6 soak self-certification); None when the platform
    # offers no reading
    rss_mb: float | None = None

    @property
    def placed(self) -> int:
        return sum(1 for pick in self.picks if pick.outcome == OUTCOME_PLACED)

    @property
    def skipped_oversize(self) -> int:
        return sum(1 for pick in self.picks if pick.outcome == OUTCOME_SKIPPED_OVERSIZE)

    @property
    def failed(self) -> int:
        return sum(1 for pick in self.picks if pick.outcome == OUTCOME_FAILED)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "record_id": self.record_id,
            "loop": self.loop,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "passes": self.passes,
            "placed": self.placed,
            "skipped_oversize": self.skipped_oversize,
            "failed": self.failed,
            "picks": [pick.to_dict() for pick in self.picks],
            "rss_mb": self.rss_mb,
        }
        if self.error is not None:
            result["error"] = self.error
        return result


class RollingLog:
    """The last ``LOOP_RECORD_WINDOW`` loop records, oldest first.

    Lives for the module's lifetime (never cleared on ``start``), so a
    poller never misses a short run that completed between polls.
    """

    def __init__(self, window: int = LOOP_RECORD_WINDOW) -> None:
        self._window = window
        self._records: list[LoopRecord] = []

    def append(self, record: LoopRecord) -> None:
        self._records.append(record)
        if len(self._records) > self._window:
            self._records = self._records[-self._window :]

    def records(self) -> list[LoopRecord]:
        return list(self._records)


def loop_seed(base_seed: int, loop_index: int) -> int:
    """Seed for 0-based ``loop_index`` of a run scattered from ``base_seed``."""
    return base_seed + loop_index


def success_rate(placed: int, failed: int) -> float | None:
    """Running success rate, oversize skips excluded; None before any
    terminal placed/failed outcome."""
    total = placed + failed
    if total == 0:
        return None
    return placed / total
