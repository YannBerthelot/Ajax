"""``benchmarks/suite/publish.py``, which writes the agent benchmark's tables
from its records.

Contract: a cell is the return at the last evaluation of each seed's latest
attempt, shown as mean ± sample standard deviation, and only once every seed
of the run reached the step budget; seeds run in parts are gathered. The
benchmark page carries exactly what the records give, and the performance
page what the timing pass gives. The aggregate scores follow rliable
(Agarwal et al., NeurIPS 2021): DeepMind Control returns over 1000, MinAtar
returns over the game's best mean; the interquartile mean drops a quarter of
the runs (rounded down) at each end; the interval is a 95% percentile
bootstrap that draws seeds within each task; the profile is the fraction of
runs strictly above each threshold.
"""

import json

import numpy as np
import publish
import pytest


def record(seeds, attempt, start, end, returns, budget=4):
    return {
        "seeds": seeds,
        "attempt": attempt,
        "start": start,
        "end": end,
        "budget": budget,
        "eval_steps": list(range(start + 2, end + 1, 2)),
        "returns": returns,
    }


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_published_tables_match_the_records():
    """Catches records added or changed without rerunning publish.py, and
    hand edits to the generated tables."""
    runs = publish.read_runs()
    page = publish.PAGE.read_text()
    assert (
        publish.replace_block(page, "benchmark-tables", publish.page_block(runs))
        == page
    )
    performance = publish.PERFORMANCE_PAGE.read_text()
    timing = publish.timing_block(runs, publish.read_timings())
    assert publish.replace_block(performance, "agent-timing", timing) == performance


def test_score_uses_the_latest_attempt_and_gathers_seeds_run_in_parts(tmp_path):
    """Catches scoring an abandoned first attempt, or dropping the seeds
    that ran in a second file."""
    write(
        tmp_path / "SAC" / "seeds-0-0.jsonl",
        [
            record([0], 0, 0, 2, [[100.0]]),  # abandoned at step 2 of 4
            record([0], 1, 0, 2, [[0.0]]),
            record([0], 1, 2, 4, [[1.0]]),
        ],
    )
    write(tmp_path / "SAC" / "seeds-1-1.jsonl", [record([1], 0, 0, 4, [[5.0, 3.0]])])

    run = publish.read_run("SAC", tmp_path)

    assert run.final.tolist() == [1.0, 3.0]
    # mean 2, sample standard deviation sqrt(((1-2)^2 + (3-2)^2) / 1) = 1.41
    assert publish.table_body(["T"], ["SAC"], {("T", "SAC"): run}, bold=True) == (
        "| T | 2.0 ± 1.4 |"
    )


def test_unfinished_run_has_no_cell(tmp_path):
    """Catches showing a run in progress, whose last evaluation is not the
    end of training."""
    write(
        tmp_path / "SAC" / "seeds-0-9.jsonl", [record([0, 1], 0, 0, 2, [[1.0], [2.0]])]
    )

    assert publish.read_run("SAC", tmp_path) is None


def test_bold_marks_the_best_mean_of_each_row(tmp_path):
    """Catches bolding the wrong agent, e.g. the lowest or the last."""
    write(
        tmp_path / "SAC" / "seeds-0-1.jsonl",
        [record([0, 1], 0, 0, 4, [[0.0, 2.0], [0.0, 4.0]])],
    )
    write(
        tmp_path / "TD3" / "seeds-0-1.jsonl",
        [record([0, 1], 0, 0, 4, [[0.0, 5.0], [0.0, 7.0]])],
    )
    cell = {("T", a): publish.read_run(a, tmp_path) for a in ("SAC", "TD3")}

    assert publish.table_body(["T"], ["SAC", "TD3"], cell, bold=True) == (
        "| T | 3.0 ± 1.4 | **6.0 ± 1.4** |"
    )


def final_run(agent, task, suite, finals):
    return publish.Run(agent, task, suite, np.array([1]), np.array([finals]).T)


def iqm_interval(per_task):
    resampled = publish.stratified_resamples(per_task)
    return publish.interval(publish.interquartile_mean(resampled))


def test_interquartile_mean_drops_the_lowest_and_highest_quarters():
    """Catches a plain mean, or a cut of the wrong size."""
    # Sorted: -50 1 | 2 3 4 5 | 6 100, two cut at each end: (2 + 3 + 4 + 5) / 4.
    assert publish.interquartile_mean(np.array([5, 1, 100, 3, 2, 4, -50, 6.0])) == 3.5
    # Five values: 5 // 4 = 1 dropped at each end, (2 + 4 + 8) / 3.
    middle = publish.interquartile_mean(np.array([16, 1, 8, 2, 4.0]))
    assert middle == pytest.approx(14 / 3)


def test_interval_holds_the_point_estimate():
    """Catches an interval drawn around another statistic."""
    per_task = [np.array([0.1, 0.4, 0.5, 0.9]), np.array([0.2, 0.3, 0.6, 0.8])]
    # Pooled: 0.1 0.2 | 0.3 0.4 0.5 0.6 | 0.8 0.9, so (0.3 + 0.4 + 0.5 + 0.6) / 4.
    point = publish.interquartile_mean(np.concatenate(per_task))
    low, high = iqm_interval(per_task)
    assert point == pytest.approx(0.45)
    assert low < point < high


def test_a_single_value_has_a_zero_width_interval():
    low, high = iqm_interval([np.array([0.7])])
    assert low == high == 0.7


def test_resamples_draw_seeds_within_each_task():
    """Catches resampling the pooled runs: two tasks of constant scores keep
    four zeros and four ones in every draw, (0 + 0 + 1 + 1) / 4 each."""
    low, high = iqm_interval([np.zeros(4), np.ones(4)])
    assert low == high == 0.5


def test_scores_share_one_scale_per_suite():
    """Catches scaling DeepMind Control by its best agent, or MinAtar by its
    best seed, its last agent or the mean of the agents."""
    runs = [
        final_run("SAC", "CheetahRun", "DeepMind Control", [500.0, 700.0]),
        final_run("PQN", "Breakout", "MinAtar", [40.0, 40.0]),  # mean 40, the best
        final_run("DQN", "Breakout", "MinAtar", [10.0, 50.0]),  # mean 30
    ]
    assert publish.score_scales(runs) == {
        ("DeepMind Control", "CheetahRun"): 1000.0,
        ("MinAtar", "Breakout"): 40.0,
    }


def test_profile_counts_the_runs_strictly_above_each_threshold():
    """Catches counting the runs at the threshold, or those below it."""
    fractions = publish.performance_profile(
        np.array([0.1, 0.5, 0.9]), np.array([0.0, 0.5, 0.95])
    )
    np.testing.assert_allclose(fractions, [1.0, 1 / 3, 0.0])


def timing_record(machine, timed_steps, exec_s, compile_s):
    manifest = {"commit": "abcdef0", "machine": {"cpu": "A CPU"}, "device": "B"}
    record = {"agent": "SAC", "env": "CheetahRun", "machine": machine}
    record.update({"n_seeds": 10, "timed_steps": timed_steps, "exec_s": exec_s})
    record.update({"compile_s": compile_s, "manifest": manifest})
    return record


def test_timing_table_gives_each_machine_and_the_gpu_gain(tmp_path):
    """Catches the gain inverted, or the machines' columns swapped."""
    # 4,000 steps in 100 s on the CPU, 24,000 on the GPU: 40 and 240 per second.
    write(tmp_path / "cpu.jsonl", [timing_record("cpu", 4000, 100.0, 19.0)])
    write(tmp_path / "gpu.jsonl", [timing_record("gpu", 24000, 100.0, 63.0)])
    runs = [final_run("SAC", "CheetahRun", "DeepMind Control", [1.0])]

    table = publish.timing_block(runs, publish.read_timings(tmp_path))

    assert "| SAC | CheetahRun | 40 | 240 | 6 times | 19 / 63 |" in table.splitlines()


def test_timing_table_shows_a_machine_not_timed_yet_as_pending(tmp_path):
    """Catches a missing GPU record read as a speed, or the row dropped; a
    speed below one keeps two digits (16 steps in 250 s)."""
    write(tmp_path / "cpu.jsonl", [timing_record("cpu", 16, 250.0, 19.0)])
    runs = [final_run("SAC", "CheetahRun", "DeepMind Control", [1.0])]

    table = publish.timing_block(runs, publish.read_timings(tmp_path))

    row = f"| SAC | CheetahRun | 0.064 | pending | {publish.DASH} | 19 / pending |"
    assert row in table.splitlines()
