"""The M9 acceptance criteria (``benchmarks/world_models/wm_acceptance.py``).

Pure NumPy: window means, the per-task and aggregate rules of the
paper-protocol runs, the verdict, the multi-task rule; their constants are
pinned (changing one is a new protocol, ``docs/world_models/VALIDATION.md``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import wm_acceptance as acc


def test_the_criteria_constants_are_the_documented_ones():
    assert acc.TOLERANCE == 50.0
    assert acc.TASK_SHARE == 0.8
    assert acc.MIN_SEEDS == 3
    assert acc.MT_FRACTION == 0.5
    assert acc.MT_SOURCE_FLOOR == 100.0
    dreamer = [w.bounds(500_000) for w in acc.WINDOWS["DreamerV3"]]
    assert dreamer == [(200_000, 300_000), (400_000, 500_000)]
    assert [w.bounds(4_000_000) for w in acc.WINDOWS["TDMPC2"]] == [
        (500_000, 1_000_000),
        (3_000_000, 4_000_000),
    ]
    # The final window follows the task's budget (humanoid: 14M).
    assert acc.WINDOWS["TDMPC2"][1].bounds(14_000_000) == (13_000_000, 14_000_000)


def test_window_mean_is_left_open_right_closed_and_weighted():
    x = [100, 200, 300, 400]
    y = [1.0, 2.0, 3.0, 4.0]
    assert acc.window_mean(x, y, 100, 300) == 2.5  # 200 and 300, not 100
    assert acc.window_mean(x, y, 0, 400, weights=[0, 1, 1, 2]) == pytest.approx(
        (2 + 3 + 8) / 4
    )
    assert math.isnan(acc.window_mean(x, y, 400, 500))
    assert math.isnan(acc.window_mean(x, y, 0, 400, weights=[0, 0, 0, 0]))
    assert math.isnan(acc.window_mean([], [], 0, 1))


def test_a_task_passes_at_the_lowest_reference_seed_minus_the_tolerance():
    ref = [600.0, 800.0, 900.0]
    assert acc.judge_task("t", "w", [550.0, 550.0, 550.0], ref).status == "pass"
    assert acc.judge_task("t", "w", [549.0, 550.0, 550.0], ref).status == "fail"
    above = acc.judge_task("t", "w", [990.0, 995.0, 1000.0], ref)
    assert above.status == "pass" and above.bar == 550.0
    assert above.ajax_mean == pytest.approx(995.0)
    assert above.reference_mean == pytest.approx(2300 / 3)


def test_a_task_without_enough_data_is_not_judged():
    ref = [500.0, 600.0, 700.0]
    assert acc.judge_task("t", "w", [], ref).status == "no run"
    assert acc.judge_task("t", "w", [900.0] * 3, ref, reached=False).status == (
        "not reached"
    )
    assert acc.judge_task("t", "w", [900.0, np.nan, 900.0], ref).status == (
        "not reached"
    )
    assert acc.judge_task("t", "w", [900.0, 900.0], ref).status == "too few seeds"
    # A run off its protocol is never judged, whatever its scores.
    off = acc.judge_task("t", "w", [900.0] * 3, ref, on_protocol=False)
    assert off.status == "off protocol"
    assert acc.judge_window("w", [off]).status == acc.INCOMPLETE
    with pytest.raises(ValueError, match="no point"):
        acc.judge_task("t", "w", [900.0] * 3, [])
    with pytest.raises(ValueError, match="no point"):
        acc.judge_task("t", "w", [900.0] * 3, [np.nan])


def _results(ajax_means, ref_means):
    return [
        acc.judge_task(f"t{i}", "w", [a] * 3, [r] * 3)
        for i, (a, r) in enumerate(zip(ajax_means, ref_means))
    ]


def test_the_aggregate_needs_median_mean_and_share():
    ref = [800.0] * 10
    # Every task within tolerance: pass.
    verdict = acc.judge_window("w", _results([760.0] * 10, ref))
    assert verdict.status == acc.PASS and verdict.share_passed == 1.0
    assert (verdict.ajax_median, verdict.reference_median) == (760.0, 800.0)
    # 8 of 10 pass (exactly the share), the median and mean hold: pass.
    eight = [800.0] * 8 + [0.0] * 2
    verdict = acc.judge_window("w", _results(eight, [800.0] * 8 + [100.0] * 2))
    assert verdict.share_passed == 0.8 and verdict.status == acc.PASS
    # 7 of 10: fail, although the median holds.
    seven = [800.0] * 7 + [0.0] * 3
    verdict = acc.judge_window("w", _results(seven, [800.0] * 7 + [100.0] * 3))
    assert verdict.status == acc.FAIL
    # Every task passes its own bar, but the mean falls short of the
    # reference's: high-variance references (low minimum) hide a shortfall.
    wide = [
        acc.judge_task(f"t{i}", "w", [300.0] * 3, [300.0, 900.0, 900.0])
        for i in range(5)
    ]
    verdict = acc.judge_window("w", wide)
    assert all(r.status == "pass" for r in wide) and verdict.status == acc.FAIL


def _judged(cases):
    """``[(ajax seed mean, reference seeds)]`` -> task results (3 seeds)."""
    return [
        acc.judge_task(f"t{i}", "w", [a] * 3, ref) for i, (a, ref) in enumerate(cases)
    ]


def test_the_median_clause_alone_can_fail_the_aggregate():
    """Every task passes and the mean holds (820 >= 560 - 50), but the
    median does not (700 < 800 - 50); the reference's median and mean
    differ, so neither stands in for the other."""
    results = _judged(
        [(700.0, [700.0, 850.0, 850.0])] * 3 + [(1000.0, [200.0] * 3)] * 2
    )
    verdict = acc.judge_window("w", results)
    assert all(r.status == "pass" for r in results)
    assert (verdict.reference_median, verdict.reference_mean) == (800.0, 560.0)
    assert (verdict.ajax_median, verdict.ajax_mean) == (700.0, 820.0)
    assert verdict.status == acc.FAIL


def test_the_mean_clause_alone_can_fail_the_aggregate():
    """Every task passes (bar 250) and the median holds (800 >= 750), but
    the mean does not (600 < 750)."""
    ref = [300.0, 1000.0, 1100.0]
    results = _judged([(a, ref) for a in (800.0, 800.0, 800.0, 300.0, 300.0)])
    verdict = acc.judge_window("w", results)
    assert all(r.status == "pass" for r in results)
    assert (verdict.ajax_median, verdict.ajax_mean) == (800.0, 600.0)
    assert verdict.status == acc.FAIL


def test_the_aggregate_bars_are_inclusive():
    # The Ajax median exactly at the reference's minus the tolerance (750),
    # the mean above it.
    ref = [800.0] * 3
    results = _judged([(a, ref) for a in (750.0, 750.0, 750.0, 1000.0, 1000.0)])
    verdict = acc.judge_window("w", results)
    assert (verdict.ajax_median, verdict.ajax_mean) == (750.0, 850.0)
    assert verdict.status == acc.PASS
    # The Ajax mean exactly at the reference's minus the tolerance (750),
    # the median above it.
    ref = [500.0, 800.0, 1100.0]
    results = _judged([(a, ref) for a in (900.0, 900.0, 900.0, 525.0, 525.0)])
    verdict = acc.judge_window("w", results)
    assert (verdict.ajax_median, verdict.ajax_mean) == (900.0, 750.0)
    assert verdict.status == acc.PASS


def test_the_aggregate_is_incomplete_while_a_task_is_not_judged():
    results = _results([800.0] * 4, [800.0] * 4)
    results.append(acc.judge_task("late", "w", [800.0] * 3, [800.0] * 3, reached=False))
    verdict = acc.judge_window("w", results)
    assert verdict.status == acc.INCOMPLETE
    # The judged tasks' aggregates are still reported.
    assert verdict.ajax_median == 800.0 and verdict.share_passed == 1.0
    nothing = acc.judge_window("w", [acc.judge_task("t", "w", [], [1.0])])
    assert nothing.status == acc.INCOMPLETE and math.isnan(nothing.ajax_mean)
    with pytest.raises(ValueError):
        acc.judge_window("w", [])


def test_overall_verdict():
    assert acc.overall([acc.PASS, acc.PASS]) == acc.PASS
    assert acc.overall([acc.PASS, acc.INCOMPLETE]) == acc.INCOMPLETE
    assert acc.overall([acc.INCOMPLETE, acc.FAIL]) == acc.FAIL
    assert acc.overall([]) == acc.INCOMPLETE


def test_the_multitask_fraction_floor_and_share():
    source = {f"t{i}": [800.0, 900.0] for i in range(10)}
    offline = {f"t{i}": [425.0] for i in range(10)}  # 0.5 x 850 exactly
    results, verdict = acc.judge_multitask(source, offline)
    assert verdict == acc.PASS and all(r.status == "pass" for r in results)
    assert results[0].ratio == pytest.approx(0.5)
    # Two of ten below the fraction: 80% pass, still PASS; three: FAIL.
    offline["t1"] = offline["t2"] = [424.0]
    assert acc.judge_multitask(source, offline)[1] == acc.PASS
    assert acc.judge_multitask(source, {**offline, "t3": [424.0]})[1] == acc.FAIL
    # A source that did not learn is reported, not judged.
    source["t1"] = source["t2"] = [50.0, 99.0]
    results, verdict = acc.judge_multitask(source, offline)
    assert [r.status for r in results[:3]] == ["pass", "not judged", "not judged"]
    assert verdict == acc.PASS
    # A missing or NaN task makes it incomplete; no judged task too.
    assert acc.judge_multitask(source, {**offline, "t3": []})[1] == acc.INCOMPLETE
    assert acc.judge_multitask(source, {**offline, "t3": [np.nan]})[1] == (
        acc.INCOMPLETE
    )
    del offline["t4"]
    results, verdict = acc.judge_multitask(source, offline)
    assert verdict == acc.INCOMPLETE and results[4].status == "missing"
    assert acc.judge_multitask({"a": [10.0]}, {"a": [1.0]})[1] == acc.INCOMPLETE
    assert math.isnan(acc.MultiTaskResult("a", 0.0, 1.0, "not judged").ratio)


def test_the_multitask_share_is_over_the_judged_tasks():
    """10 tasks, 2 not judged, 7 of the 8 judged pass: 7/8 >= 80% (7/10
    would not)."""
    source = {f"t{i}": [800.0] * 3 for i in range(10)}
    source["t8"] = source["t9"] = [50.0] * 3
    offline = {f"t{i}": [500.0] for i in range(10)}
    offline["t0"] = [100.0]
    results, verdict = acc.judge_multitask(source, offline)
    assert [r.status for r in results].count("pass") == 7
    assert verdict == acc.PASS


def test_the_multitask_source_statistic_is_the_seed_mean_and_the_floor_judged():
    # Sources [100, 800, 900]: mean 600 (bar 300), median 800 (bar 400).
    results, verdict = acc.judge_multitask({"a": [100.0, 800.0, 900.0]}, {"a": [350.0]})
    assert results[0].source == 600.0 and results[0].status == "pass"
    assert verdict == acc.PASS
    # A source mean of exactly the floor is judged.
    results, _ = acc.judge_multitask({"a": [100.0, 100.0]}, {"a": [0.0]})
    assert results[0].status == "fail"


def test_the_multitask_verdict_needs_most_tasks_judged():
    """19 tasks whose sources mostly did not learn: one judged task cannot
    validate the mechanisms (INCOMPLETE, not PASS). 80% must be judged:
    16 of 19 (15.2 rounded up), 8 of 10."""

    def verdict(n_tasks, n_learned):
        source = {f"t{i}": [900.0 if i < n_learned else 60.0] for i in range(n_tasks)}
        offline = {f"t{i}": [500.0 if i < n_learned else 0.0] for i in range(n_tasks)}
        return acc.judge_multitask(source, offline)[1]

    assert verdict(19, 1) == acc.INCOMPLETE
    assert verdict(19, 15) == acc.INCOMPLETE
    assert verdict(19, 16) == acc.PASS
    assert verdict(10, 7) == acc.INCOMPLETE
    assert verdict(10, 8) == acc.PASS
