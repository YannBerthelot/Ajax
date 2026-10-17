"""Compare the recorded Ajax arms with their reference: a Markdown table.

usage: python refcmp_report.py [results_dir]

Per arm, over its seeds: ``solved`` counts the seeds at or above the spec's
``solved`` return at the final evaluation; ``held`` is the mean share of
the last third of the evaluations at or above it (how steadily a seed stays
solved); ``final`` is the mean final return. p-values are two-sided against
the reference: Fisher's exact test on ``solved``, and a permutation
Mann-Whitney test on the per-seed ``held`` (ties are the rule there, so not
the normal approximation). The verdict reads the smaller p-value, doubled
for the two tests (Bonferroni), at 5%; with fewer than 16 seeds an adjusted
p-value under 0.2 asks for 16 before concluding. When an arm is scored on
its final policy only, ``held`` is ``solved`` again and Fisher's test alone
counts.
"""

import json
import pathlib
import sys

import numpy as np
from refcmp_specs import COMPARISONS
from scipy.stats import PermutationMethod, fisher_exact, mannwhitneyu

ALPHA, UNCLEAR, FULL_SEEDS = 0.05, 0.2, 16


def load(results_dir: pathlib.Path) -> dict:
    """``{(agent, arm): curves (seeds, evaluations)}``. A seed recorded more
    than once keeps its latest record, so a re-run replaces the old one."""
    runs: dict = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            per_seed = runs.setdefault((r["agent"], r["arm"]), {})
            per_seed.update(zip(r["seeds"], r["curve"], strict=False))
    return {
        key: np.asarray([rows[s] for s in sorted(rows)], dtype=float)
        for key, rows in runs.items()
    }


def summarise(curves: np.ndarray, solved: float) -> dict:
    tail = curves[:, -max(1, curves.shape[1] // 3) :]
    return {
        "n": len(curves),
        "tail": tail.shape[1],
        "solved": int((curves[:, -1] >= solved).sum()),
        "held": (tail >= solved).mean(axis=1),
        "final": float(curves[:, -1].mean()),
    }


def verdict(arm: dict, ref: dict) -> tuple:
    p_solved = fisher_exact(
        [
            [arm["solved"], arm["n"] - arm["solved"]],
            [ref["solved"], ref["n"] - ref["solved"]],
        ]
    )[1]
    tests, p_held = [p_solved], float("nan")
    # One evaluation in the tail makes held the solved count again; equal
    # shares on every seed leave Mann-Whitney undefined (and p = 1).
    if arm["tail"] > 1:
        tied = np.ptp(np.concatenate([arm["held"], ref["held"]])) == 0
        p_held = (
            1.0
            if tied
            else mannwhitneyu(
                arm["held"],
                ref["held"],
                method=PermutationMethod(n_resamples=9_999, random_state=0),
            ).pvalue
        )
        tests.append(p_held)
    p = min(1.0, len(tests) * min(tests))
    if p < ALPHA:
        better = arm["held"].mean() > ref["held"].mean()
        word = "Ajax better" if better else "gap"
    elif p < UNCLEAR and min(arm["n"], ref["n"]) < FULL_SEEDS:
        word = f"unclear: run {FULL_SEEDS} seeds"
    else:
        word = "match"
    return p_solved, p_held, word


def report(results_dir: pathlib.Path) -> str:
    runs = load(results_dir)
    rows = [
        "| Agent | Arm | Seeds | Solved | Held | Final | p(solved) | p(held) | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, spec in COMPARISONS.items():
        if (name, "reference") not in runs:
            continue
        ref = summarise(runs[(name, "reference")], spec.solved)
        rows.append(
            f"| {name} | {spec.reference} | {ref['n']} | {ref['solved']}/{ref['n']} "
            f"| {ref['held'].mean():.2f} | {ref['final']:.1f} | | | |"
        )
        for arm_name in ("matched", "defaults"):
            if (name, arm_name) not in runs:
                continue
            arm = summarise(runs[(name, arm_name)], spec.solved)
            p_solved, p_held, word = verdict(arm, ref)
            held = "-" if np.isnan(p_held) else f"{p_held:.2g}"
            rows.append(
                f"| {name} | Ajax {arm_name} | {arm['n']} | {arm['solved']}/{arm['n']} "
                f"| {arm['held'].mean():.2f} | {arm['final']:.1f} | {p_solved:.2g} "
                f"| {held} | {word} |"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    print(report(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"))
