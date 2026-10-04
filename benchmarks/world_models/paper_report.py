"""Judge paper-protocol runs against the published curves: table, plot, verdict.

Reads the run directories ``paper_protocol.py`` wrote (``<runs>/<run>/
{run.json, curve.jsonl}``) and the committed references
(``references/*.json``), applies the acceptance criteria of
``wm_acceptance.py`` (fixed before any run; ``docs/world_models/
VALIDATION.md``) and writes a markdown report and, per agent, a plot of
every task's curves (Ajax seed mean and range against the reference's).
Runs still in progress are judged on the windows they have reached; the
verdict is then ``INCOMPLETE``. A run whose specification is not its
registry entry's (``paper_protocol.paper_runs``; seeds: the protocol's or
a prefix of them) is "off protocol": reported, never judged. Each run's
provenance (commits, a tree with changes, JAX versions, devices) is
listed, runs spanning several flagged. The exit code is 1 when a verdict
is ``FAIL``.

The references are dm_control curves and the runs use mujoco_playground's
MJX ports (physics, observations and sometimes rewards differ): the
comparison is approximate, and the report says so.

Usage::

    python benchmarks/world_models/paper_report.py --runs runs/ \\
        --out runs/report.md --plot-dir runs/
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import warnings
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from paper_protocol import paper_runs, smoke_runs
from wm_acceptance import (
    FAIL,
    MIN_SEEDS,
    TASK_SHARE,
    TOLERANCE,
    WINDOWS,
    TaskResult,
    WindowVerdict,
    judge_task,
    judge_window,
    overall,
    window_mean,
)
from wm_runs import (
    CURVE_FILE,
    RUN_FILE,
    RunSpec,
    load_reference,
    provenance_summary,
    read_records,
    spec_differences,
)

REFERENCES = {"DreamerV3": "dreamerv3_dmc_proprio", "TDMPC2": "tdmpc2_dmc"}
CAVEAT = (
    "The runs use mujoco_playground's MJX ports of the DMC tasks, whose"
    " physics, observations and sometimes rewards differ from dm_control's,"
    " on which the reference curves were measured: the comparison is"
    " approximate."
)
#: Categorical slots 1 and 2 of the dataviz reference palette.
AJAX_COLOR, REFERENCE_COLOR = "#2a78d6", "#eb6834"


def off_protocol(name: str, spec: dict[str, Any]) -> list[str]:
    """Why the run ``name`` (its directory) with the stored specification
    ``spec`` is not its registry entry (``[]``: it is). The seeds may be the
    protocol's or a prefix of them (``--seeds``; a verdict needs
    :data:`MIN_SEEDS`), never others: no choosing seeds after the fact.
    Smoke runs are checked against the smoke registry."""
    expected: Optional[RunSpec] = (
        smoke_runs() if spec.get("smoke") else paper_runs()
    ).get(name)
    if expected is None:
        return [f"no protocol run named {name!r}"]
    protocol = dataclasses.asdict(expected)
    stored = dict(spec)
    seeds = list(stored.pop("seeds", []))
    reasons = spec_differences({**protocol, "seeds": None}, {**stored, "seeds": None})
    if not seeds or seeds != list(expected.seeds)[: len(seeds)]:
        reasons.append("seeds")
    return reasons


def read_runs(runs_dir: str) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(agent, reference task): {"name", "spec", "records", "dir",
    "off_protocol"}}`` of a runs directory (:func:`off_protocol`)."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for run_file in sorted(glob.glob(os.path.join(runs_dir, "*", RUN_FILE))):
        directory = os.path.dirname(run_file)
        name = os.path.basename(directory)
        with open(run_file) as f:
            spec = json.load(f)["spec"]["run"]
        key = (spec["agent"], spec["reference_task"])
        if key in runs:
            raise ValueError(f"two runs of {key} in {runs_dir}")
        runs[key] = {
            "name": name,
            "spec": spec,
            "records": read_records(os.path.join(directory, CURVE_FILE)),
            "dir": directory,
            "off_protocol": off_protocol(name, spec),
        }
    return runs


def reference_budget(agent: str, reference: dict, task: str) -> int:
    """The env steps of the reference protocol for ``task``."""
    if agent == "DreamerV3":
        return int(reference["protocol"]["env_steps"])
    return max(max(seed["x"]) for seed in reference["tasks"][task])


def ajax_window_scores(records: Sequence[dict], lo: int, hi: int) -> np.ndarray:
    """Per-seed window means of a run's records (weighted by episodes)."""
    if not records:
        return np.zeros(0)
    x = [r["env_frames"] for r in records]
    values = np.asarray([r["value"] for r in records], np.float64)  # [n, S]
    weights = np.asarray([r["episodes"] for r in records], np.float64)
    return np.asarray(
        [
            window_mean(x, values[:, s], lo, hi, weights[:, s])
            for s in range(values.shape[1])
        ]
    )


def judge_agent(
    agent: str, reference: dict, runs: dict[tuple[str, str], dict]
) -> list[WindowVerdict]:
    """Every window of ``agent`` over its protocol tasks (``wm_acceptance``)."""
    verdicts = []
    for window in WINDOWS[agent]:
        results: list[TaskResult] = []
        for task in reference["playground"]:
            lo, hi = window.bounds(reference_budget(agent, reference, task))
            ref = [
                window_mean(seed["x"], seed["y"], lo, hi)
                for seed in reference["tasks"][task]
            ]
            ref = [r for r in ref if not np.isnan(r)]
            run = runs.get((agent, task))
            records = run["records"] if run else []
            reached = bool(records) and records[-1]["env_frames"] >= hi
            results.append(
                judge_task(
                    task,
                    window.label,
                    ajax_window_scores(records, lo, hi),
                    ref,
                    reached=reached,
                    on_protocol=not (run and run["off_protocol"]),
                )
            )
        verdicts.append(judge_window(window.label, results))
    return verdicts


def published_points(agent: str, reference: dict) -> list[str]:
    """The reference's own task mean and median at a few env steps."""
    points = (250_000, 490_000) if agent == "DreamerV3" else (1_000_000, 4_000_000)
    out = []
    for x in points:
        means = []
        for task in reference["playground"]:
            ys = [
                s["y"][s["x"].index(x)] for s in reference["tasks"][task] if x in s["x"]
            ]
            if ys:
                means.append(float(np.mean(ys)))
        out.append(
            f"{_steps(x)} env steps: task mean {np.mean(means):.1f}, median"
            f" {np.median(means):.1f} ({len(means)} tasks)"
        )
    return out


def _steps(x: float) -> str:
    """Env steps as ``250K`` / ``1M`` / ``1.5M``."""
    if x == 0:
        return "0"
    return f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}K"


def _fmt(value: float) -> str:
    return "-" if np.isnan(value) else f"{value:.0f}"


def _seed_range(values: np.ndarray) -> str:
    if values.size == 0 or np.isnan(values).all():
        return "-"
    return f"{np.nanmin(values):.0f}-{np.nanmax(values):.0f}"


def markdown(
    agent: str,
    reference: dict,
    runs: dict[tuple[str, str], dict],
    verdicts: list[WindowVerdict],
) -> str:
    source = reference["source"]
    smoke = any(run["spec"].get("smoke") for (a, _), run in runs.items() if a == agent)
    lines = [
        f"## {agent}: {overall([v.status for v in verdicts])}"
        + (" (smoke runs: plumbing only)" if smoke else ""),
        "",
        f"Reference: {source['repository']} `{source['file']}` at"
        f" `{source['commit'][:7]}` ({source['license']}); x = {reference['units']['x']};"
        f" y = {reference['units']['y']}.",
        "Published values: " + "; ".join(published_points(agent, reference)) + ".",
        "",
        "| window | status | Ajax median | ref median | Ajax mean | ref mean |"
        " tasks passed |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        judged = sum(r.status in ("pass", "fail") for r in v.tasks)
        lines.append(
            f"| {v.window} | {v.status} | {_fmt(v.ajax_median)} |"
            f" {_fmt(v.reference_median)} | {_fmt(v.ajax_mean)} |"
            f" {_fmt(v.reference_mean)} |"
            f" {sum(r.status == 'pass' for r in v.tasks)}/{judged} |"
        )
    header = "| task | env | seeds |" + "".join(
        f" {v.window}: Ajax (range) | {v.window}: ref (range) | {v.window} |"
        for v in verdicts
    )
    lines += ["", header, "|---|---|---|" + "---|---|---|" * len(verdicts)]
    for i, task in enumerate(reference["playground"]):
        run = runs.get((agent, task))
        seeds = len(run["spec"]["seeds"]) if run else 0
        row = f"| {task} | {reference['playground'][task]} | {seeds} |"
        for v in verdicts:
            r = v.tasks[i]
            row += (
                f" {_fmt(r.ajax_mean)} ({_seed_range(r.ajax)}) |"
                f" {r.reference_mean:.0f} ({_seed_range(r.reference)}) | {r.status} |"
            )
        lines.append(row)
    lines += ["", *runs_table(agent, runs)]
    return "\n".join(lines) + "\n"


def runs_table(agent: str, runs: dict[tuple[str, str], dict]) -> list[str]:
    """Per run of ``agent``: protocol check and provenance
    (:func:`~wm_runs.provenance_summary`), flagging off-protocol runs and
    runs whose records span several commits, JAX versions or devices or
    come from a tree with changes."""
    lines = [
        "| run | protocol | commits | tree changes | jax | devices |",
        "|---|---|---|---|---|---|",
    ]
    for (a, _), run in sorted(runs.items()):
        if a != agent:
            continue
        reasons = run["off_protocol"]
        protocol = "**off protocol**: " + ", ".join(reasons) if reasons else "ok"
        lines.append(
            f"| {run['name']} | {protocol} |{provenance_cells(run['records'])}"
        )
    return lines


def provenance_cells(records: Sequence[dict]) -> str:
    """Table cells ``commits | tree changes | jax | devices |`` of a run's
    records; a column with several values is flagged."""
    p = provenance_summary(records)

    def cell(values: list, short: bool = False) -> str:
        text = ", ".join(str(v)[:7] if short else str(v) for v in values)
        return text + (" **(several)**" if len(values) > 1 else "")

    return (
        f" {cell(p['git_sha'], short=True)} | {'**yes**' if p['dirty'] else 'no'} |"
        f" {cell(p['jax'])} | {cell(p['device'])} |"
    )


def plot(agent: str, reference: dict, runs: dict, path: str) -> None:
    """One panel per task: the reference's and Ajax's seed means (lines) and
    seed ranges (bands) against env steps, the criteria windows shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    tasks = list(reference["playground"])
    cols = 6
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.0 * cols, 2.3 * rows), squeeze=False, sharey=True
    )
    for ax, task in zip(axes.flat, tasks):
        budget = reference_budget(agent, reference, task)
        for window in WINDOWS[agent]:
            lo, hi = window.bounds(budget)
            ax.axvspan(lo, hi, color="#000000", alpha=0.05, linewidth=0)
        grid = sorted({x for s in reference["tasks"][task] for x in s["x"]})
        ys = np.asarray(
            [
                [s["y"][s["x"].index(x)] if x in s["x"] else np.nan for x in grid]
                for s in reference["tasks"][task]
            ]
        )
        _curve(ax, grid, ys, REFERENCE_COLOR)
        run = runs.get((agent, task))
        if run and run["records"]:
            x = [r["env_frames"] for r in run["records"]]
            values = np.asarray([r["value"] for r in run["records"]], np.float64).T
            _curve(ax, x, values, AJAX_COLOR, marker="o")
        ax.set_title(task, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _steps(v)))
        ax.grid(color="#e0e0e0", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=REFERENCE_COLOR, linewidth=1.5),
        Line2D([], [], color=AJAX_COLOR, linewidth=1.5, marker="o", markersize=3),
    ]
    labels = ["reference (dm_control)", "Ajax (playground)"]
    fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(
        f"{agent}: return vs env steps (seed mean and range; shaded: criteria"
        " windows). Playground ports differ from dm_control.",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _curve(
    ax: Any, x: Sequence[float], ys: np.ndarray, color: str, marker: str = ""
) -> None:
    with warnings.catch_warnings():  # all-NaN columns (a seed's missing point)
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(ys, axis=0)
        lo, hi = np.nanmin(ys, axis=0), np.nanmax(ys, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.5, marker=marker, markersize=3)


def report(runs_dir: str, plot_dir: Optional[str] = None) -> tuple[str, list[str]]:
    """The markdown report of ``runs_dir`` and each agent's verdict; plots
    to ``plot_dir`` when given."""
    runs = read_runs(runs_dir)
    parts = [
        "# World-model validation: paper protocols",
        "",
        CAVEAT,
        "",
        f"Criteria (`wm_acceptance.py`, fixed before any run): per task and"
        f" window, Ajax seed mean >= lowest reference seed - {TOLERANCE:g};"
        f" per window, Ajax median and mean of the task means >= the"
        f" reference's - {TOLERANCE:g} and >= {TASK_SHARE:.0%} of the tasks"
        f" passing; >= {MIN_SEEDS} seeds per task. A run off its protocol"
        f" (specification other than its registry entry's) is not judged.",
        "",
    ]
    statuses = []
    for agent, name in REFERENCES.items():
        reference = load_reference(name)
        verdicts = judge_agent(agent, reference, runs)
        statuses.append(overall([v.status for v in verdicts]))
        parts.append(markdown(agent, reference, runs, verdicts))
        if plot_dir and any(a == agent for a, _ in runs):
            os.makedirs(plot_dir, exist_ok=True)
            plot(agent, reference, runs, os.path.join(plot_dir, f"{agent}_curves.png"))
    return "\n".join(parts), statuses


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True, help="paper_protocol.py --out")
    parser.add_argument("--out", default=None, help="write the markdown here")
    parser.add_argument("--plot-dir", default=None, help="write the plots here")
    args = parser.parse_args(argv)
    text, statuses = report(args.runs, args.plot_dir)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 1 if FAIL in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
