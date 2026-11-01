"""Which slow-tier runs count (.github/scripts/slow_runs.py).

Contract (the owner, 2026-11-01): a run counts by its shard jobs, never by
its own conclusion. A run whose shards were skipped concludes "success" but
tested nothing: it is neither green for the release gate nor the last run
for the daily skip. The job lists are those of real runs on main: the daily
runs of 2026-10-31 and 11-01 skipped their shards, the one of 10-30 had a
shard fail, the one of 10-29 passed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))
import slow_runs


def job(name, conclusion):
    return {"name": name, "conclusion": conclusion}


SKIPPED = [
    job("changed", "success"),
    job("report", "skipped"),
    job("Slow (shard ${{ matrix.shard }})", "skipped"),
    job("bisect", "skipped"),
]
RED = [job("changed", "success")] + [
    job(f"Slow (shard {shard})", "failure" if shard == 5 else "success")
    for shard in (1, 4, 3, 6, 5, 2)
]
GREEN = [job("changed", "success")] + [
    job(f"Slow (shard {shard})", "success") for shard in (2, 3, 1, 6, 4, 5)
]


def test_a_run_counts_by_its_shards_not_its_conclusion() -> None:
    assert slow_runs.outcome(SKIPPED) == "not run"
    assert slow_runs.outcome(RED) == "failed"
    assert slow_runs.outcome(GREEN) == "passed"


def test_a_shard_cancelled_or_timed_out_is_not_a_pass() -> None:
    cancelled = GREEN[:-1] + [job("Slow (shard 5)", "cancelled")]
    timed_out = GREEN[:-1] + [job("Slow (shard 5)", "timed_out")]
    assert slow_runs.outcome(cancelled) == "failed"
    assert slow_runs.outcome(timed_out) == "failed"


def test_skipped_runs_hide_neither_the_last_red_nor_the_last_green() -> None:
    # Main's history from 2026-11-01 back: two skipped runs, a red, a green.
    runs = [
        ("eefc88c", "not run"),
        ("eefc88c", "not run"),
        ("eefc88c", "failed"),
        ("923220e", "passed"),
    ]
    assert slow_runs.last_tested(runs) == ("eefc88c", "923220e")
    assert slow_runs.last_tested([("eefc88c", "not run")]) == ("", "")
