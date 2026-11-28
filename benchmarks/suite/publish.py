"""Write the agent benchmark's tables and figures from its records.

    uv run python benchmarks/suite/publish.py          # tables and figures
    uv run python benchmarks/suite/publish.py --tables # tables only

The tables go between markers in docs/reproduction/benchmark.md; the figures
go to docs/reproduction/benchmark-figures/. A cell appears once every seed of
its run reached the step budget, so a suite's tab shows up with its first
finished run. The timing pass's records (results/timing-0.6, with a folder per
number of seeds beside the 10-seed pass) give the performance page's speed
and gain tables, the gain-by-seeds figure and the time axis of the
score-against-time figures. tests/benchmarks/test_publish.py checks the tables
still match the records.
"""

import argparse
import dataclasses
import json
import pathlib
import re
import sys
from typing import Any, Optional

import numpy as np
from agent_suite import BENCHMARKS

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "suite" / "results"
PAGE = ROOT / "docs" / "reproduction" / "benchmark.md"
FIGURES = ROOT / "docs" / "reproduction" / "benchmark-figures"
TIMING = RESULTS / "timing-0.6"
PERFORMANCE_PAGE = ROOT / "docs" / "guide" / "performance.md"
RECORDS_URL = "https://github.com/YannBerthelot/Ajax/tree/main/benchmarks/suite/results"
#: The figures' eight hues; agents past the eighth reuse one, dashed.
HUES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
#: Bootstrap resamples per interval, rliable's default for its profiles.
RESAMPLES = 2000

#: Suite name -> short name for its figure files, in tab order.
SUITES = {
    "DeepMind Control": "control",
    "Gymnasium MuJoCo": "mujoco",
    "MinAtar": "minatar",
}
#: A missing cell.
DASH = "\u2013"
DISPLAY_NAMES = {"RVISAC": "RVI-SAC", "TDMPC2": "TD-MPC2"}


@dataclasses.dataclass(frozen=True)
class Run:
    """One agent on one task: each seed's evaluation steps and returns."""

    agent: str
    task: str
    suite: str
    steps: np.ndarray  # (evaluations,)
    returns: np.ndarray  # (seeds, evaluations)

    @property
    def final(self) -> np.ndarray:
        return self.returns[:, -1]


def suite_and_task(env_id: str) -> tuple[str, str]:
    if env_id.endswith("-MinAtar"):
        return "MinAtar", env_id.removesuffix("-MinAtar")
    if env_id.startswith("mujoco:"):
        return "Gymnasium MuJoCo", env_id.removeprefix("mujoco:")
    return "DeepMind Control", env_id


def finished_seeds(path: pathlib.Path) -> dict[int, tuple[list, list]]:
    """Each seed's evaluation steps and returns in one records file, from its
    latest attempt (an earlier one was abandoned), if that reached the
    budget; a file whose run never finished gives none."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records:
        return {}
    attempt = records[-1]["attempt"]
    records = [record for record in records if record["attempt"] == attempt]
    if records[-1]["end"] != records[-1]["budget"]:
        return {}
    steps = [step for record in records for step in record["eval_steps"]]
    return {
        seed: (steps, [value for record in records for value in record["returns"][i]])
        for i, seed in enumerate(records[-1]["seeds"])
    }


def read_run(label: str, results: pathlib.Path = RESULTS) -> Optional[Run]:
    """The label's run, once all its seeds finished; seeds split over several
    files (run in parts) are gathered."""
    seeds: dict[int, tuple[list, list]] = {}
    for path in sorted((results / label).glob("*.jsonl")):
        seeds.update(finished_seeds(path))
    if not seeds:
        return None
    steps = {tuple(steps) for steps, _ in seeds.values()}
    if len(steps) != 1:
        raise ValueError(f"{label}: seeds were evaluated at different steps")
    suite, task = suite_and_task(BENCHMARKS[label].env_id)
    agent = label.split("-")[0]
    returns = np.array([seeds[seed][1] for seed in sorted(seeds)])
    return Run(agent, task, suite, np.array(steps.pop()), returns)


def read_runs(results: pathlib.Path = RESULTS) -> list[Run]:
    """Every finished run, in the order BENCHMARKS lists them."""
    runs = [
        read_run(label, results) for label in BENCHMARKS if (results / label).is_dir()
    ]
    return [run for run in runs if run is not None]


def read_timings(directory: pathlib.Path = TIMING) -> dict[tuple, dict]:
    """The timing pass's latest record of each agent on each machine, keyed
    by (agent, suite, machine)."""
    timings = {}
    for machine in ("cpu", "gpu"):
        path = directory / f"{machine}.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        for record in [json.loads(line) for line in lines if line]:
            suite, _ = suite_and_task(record["env"])
            timings[(record["agent"].split("-")[0], suite, machine)] = record
    return timings


def steps_per_second(record: dict) -> float:
    """A timing record's steps per second per seed, from its unrounded
    parts: the record's own rate keeps one decimal, too few below one."""
    return record["timed_steps"] / record["exec_s"]


def score_scales(runs: list[Run]) -> dict[tuple[str, str], float]:
    """What each (suite, task)'s returns are divided by to share one scale:
    1000 on DeepMind Control, its tasks' highest return; elsewhere the
    task's best mean final return in the benchmark, as the heatmap scales
    it."""
    scales: dict[tuple[str, str], float] = {}
    for run in runs:
        key = (run.suite, run.task)
        if run.suite == "DeepMind Control":
            scales[key] = 1000.0
        else:
            scales[key] = max(scales.get(key, 0.0), float(run.final.mean()))
    return {key: scale if scale > 0 else 1.0 for key, scale in scales.items()}


def interquartile_mean(scores: np.ndarray) -> np.ndarray:
    """The mean of the middle half of scores along the last axis, a quarter
    of them (rounded down) dropped at each end: scipy's trim_mean(0.25),
    which rliable's IQM is (Agarwal et al., NeurIPS 2021)."""
    ordered_scores = np.sort(scores, axis=-1)
    count = ordered_scores.shape[-1]
    return ordered_scores[..., count // 4 : count - count // 4].mean(axis=-1)


def stratified_resamples(per_task: list[np.ndarray]) -> np.ndarray:
    """Bootstrap draws of the runs, (RESAMPLES, runs, ...): each task's runs
    (its seeds, the first axis) are drawn with replacement among
    themselves, as rliable's stratified bootstrap. The generator is fixed,
    so a rerun draws the same figures."""
    generator = np.random.default_rng(0)
    draws = [
        scores[generator.integers(0, len(scores), (RESAMPLES, len(scores)))]
        for scores in per_task
    ]
    return np.concatenate(draws, axis=1)


def interval(estimates: np.ndarray) -> np.ndarray:
    """The 95% percentile interval of bootstrap estimates (the first axis)."""
    return np.percentile(estimates, [2.5, 97.5], axis=0)


def performance_profile(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """The fraction of runs (the last axis) that score above each
    threshold: rliable's performance profile."""
    return (scores[..., None] > thresholds).mean(axis=-2)


def ordered(values) -> list:
    return list(dict.fromkeys(values))


def name(agent: str) -> str:
    return DISPLAY_NAMES.get(agent, agent)


def table_body(rows: list[str], columns: list[str], cell: dict, bold: bool) -> str:
    """A Markdown table's rows; cell[(row, column)] is a run, or missing for
    a dash. Bold marks each row's best mean."""
    lines = []
    for row in rows:
        runs = {column: cell.get((row, column)) for column in columns}
        means = [run.final.mean() for run in runs.values() if run is not None]
        best = max(means) if bold and len(means) > 1 else None
        cells = []
        for run in runs.values():
            if run is None:
                cells.append(DASH)
                continue
            text = f"{run.final.mean():.1f} ± {run.final.std(ddof=1):.1f}"
            cells.append(f"**{text}**" if run.final.mean() == best else text)
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_agents(runs: list[Run]) -> tuple[list[str], list[str]]:
    """A suite's agents: those that ran at least half its tasks, then the
    others, which get a table of their own as they would leave the main one
    mostly dashes."""
    tasks = ordered(run.task for run in runs)
    agents = ordered(run.agent for run in runs)
    count = {agent: sum(run.agent == agent for run in runs) for agent in agents}
    main = [agent for agent in agents if 2 * count[agent] >= len(tasks)]
    return main, [agent for agent in agents if agent not in main]


def suite_tables(runs: list[Run]) -> str:
    """The suite's tables, tasks down and agents across."""
    main, more = split_agents(runs)
    cell = {(run.task, run.agent): run for run in runs}
    corner = "Game" if runs[0].suite == "MinAtar" else "Task"
    parts = [markdown_table(corner, ordered(run.task for run in runs), main, cell)]
    if more:
        more_tasks = ordered(run.task for run in runs if run.agent in more)
        parts.append(
            "The agents that ran under half of these tasks:\n\n"
            + markdown_table(corner, more_tasks, more, cell)
        )
    return "\n\n".join(parts)


def markdown_table(corner: str, rows: list, agents: list, cell: dict) -> str:
    head = f"| {corner} | " + " | ".join(name(agent) for agent in agents) + " |"
    rule = "|---" * (len(agents) + 1) + "|"
    keyed = {(row, name(agent)): run for (row, agent), run in cell.items()}
    body = table_body(rows, [name(agent) for agent in agents], keyed, bold=True)
    return "\n".join([head, rule, body])


def page_block(runs: list[Run]) -> str:
    """The page's tabs: one per suite with a finished run, its tables, then its
    heatmap and learning curves."""
    tabs = []
    for suite, short in SUITES.items():
        members = [run for run in runs if run.suite == suite]
        if not members:
            continue
        body = "\n\n".join(
            [
                suite_tables(members),
                f"![{suite}: final returns, each task scaled to its best mean]"
                f"(benchmark-figures/{short}-heatmap.svg)",
                f"![{suite}: learning curves, mean ± standard deviation over"
                f" seeds](benchmark-figures/{short}-curves.svg)",
            ]
        )
        indented = "\n".join(f"    {line}" if line else "" for line in body.split("\n"))
        tabs.append(f'=== "{suite}"\n\n{indented}')
    return "\n\n".join(tabs)


def number(value: float) -> str:
    """A speed or a gain as the table shows it: whole from 10 up, else two
    significant digits (0.064, 3.9), never in scientific notation."""
    return f"{value:,.0f}" if value >= 10 else f"{value:.2g}"


def timed_machines(timings: dict) -> list[str]:
    """The machines the timing pass has records from, the CPU first."""
    timed = {machine for _, _, machine in timings}
    return [machine for machine in ("cpu", "gpu") if machine in timed]


def timing_block(runs: list[Run], timings: dict) -> str:
    """The performance page's table: each agent's steps per second and
    compile seconds on each machine the timing pass ran on, in the benchmark
    tables' order, with the GPU's gain once both ran; then the machines and
    the commits."""
    records = list(timings.values())
    machines = timed_machines(timings)
    counts = ordered(record["n_seeds"] for record in records)
    seeds = "/".join(str(count) for count in counts)
    cpus = ordered(record["manifest"]["machine"]["cpu"] for record in records)
    gpus = ordered(
        record["manifest"]["device"] for record in records if record["machine"] == "gpu"
    )
    commits = ordered(f"`{record['manifest']['commit'][:7]}`" for record in records)
    titles = {"cpu": "Four CPU cores", "gpu": f"One {' or '.join(gpus)}"}
    head = ["Agent", "Task", *(titles[machine] for machine in machines)]
    head += ["Gain", "Compile s, CPU / GPU"] if len(machines) == 2 else ["Compile s"]
    lines = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
    for suite in SUITES:
        members = [run for run in runs if run.suite == suite]
        for agent in [agent for group in split_agents(members) for agent in group]:
            timed = [timings.get((agent, suite, machine)) for machine in machines]
            found = [record for record in timed if record is not None]
            if not found:
                continue
            speeds, compiles = [DASH] * len(timed), [DASH] * len(timed)
            for column, record in enumerate(timed):
                if record is not None:
                    speeds[column] = number(steps_per_second(record))
                    compiles[column] = f"{record['compile_s']:.0f}"
            if len(machines) == 2:
                cpu, gpu = timed
                gain = DASH
                if cpu is not None and gpu is not None:
                    ratio = steps_per_second(gpu) / steps_per_second(cpu)
                    gain = f"{number(ratio)} times"
                speeds.append(gain)
            task = suite_and_task(found[0]["env"])[1]
            cells = [name(agent), task, *speeds, " / ".join(compiles)]
            lines.append("| " + " | ".join(cells) + " |")
    if not records:
        return "\n".join(lines)
    where = f"The CPU column ran on four cores of an {' and '.join(cpus)}"
    if gpus:
        where += f", the GPU column on one {' and '.join(gpus)} of the same machine"
    commit = "commits" if len(commits) > 1 else "commit"
    lines.append(
        "\nEnvironment steps per second per seed, higher is faster, with"
        f" {seeds} seeds trained together: the run makes {seeds} times as many"
        " in all. Each is timed on a segment of training once the agent's"
        " updates run, without evaluations; the compile seconds are those of"
        f" the run's first call. {where}, at {commit} {' and '.join(commits)}"
        f" ([records]({RECORDS_URL}/timing-0.6))."
    )
    return "\n".join(lines)


def seed_passes(directory: pathlib.Path = TIMING) -> dict[int, dict]:
    """Each timing pass's records by its number of seeds: the 10-seed pass
    beside time_agents.py, the others in folders named for their seeds."""
    passes = {}
    for folder in [directory, *sorted(directory.glob("*-seed*"))]:
        timings = read_timings(folder)
        if timings:
            passes[next(iter(timings.values()))["n_seeds"]] = timings
    return dict(sorted(passes.items()))


def gains(passes: dict[int, dict]) -> dict[tuple, tuple[str, dict]]:
    """For each (agent, suite) that both machines timed: its timing task, and
    the GPU's gain over four CPU cores for each number of seeds."""
    found: dict = {}
    for seeds, timings in passes.items():
        for (agent, suite, machine), gpu in timings.items():
            cpu = timings.get((agent, suite, "cpu"))
            if machine != "gpu" or cpu is None:
                continue
            task = suite_and_task(gpu["env"])[1]
            by_seeds = found.setdefault((agent, suite), (task, {}))[1]
            by_seeds[seeds] = steps_per_second(gpu) / steps_per_second(cpu)
    return found


def gain_block(runs: list[Run], by_agent: dict) -> str:
    """The performance page's gains: how many times as fast one GPU trains
    each agent as four CPU cores, for each number of seeds, in the benchmark
    tables' order."""
    counts = sorted({seeds for _, per_seeds in by_agent.values() for seeds in per_seeds})
    head = ["Agent", "Task", *(f"{n} seed{'' if n == 1 else 's'}" for n in counts)]
    lines = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * len(counts)]
    for suite in SUITES:
        members = [run for run in runs if run.suite == suite]
        for agent in [agent for group in split_agents(members) for agent in group]:
            if (agent, suite) not in by_agent:
                continue
            task, per_seeds = by_agent[(agent, suite)]
            cells = [number(per_seeds[n]) if n in per_seeds else DASH for n in counts]
            lines.append("| " + " | ".join([name(agent), task, *cells]) + " |")
    return "\n".join(lines)


def replace_block(text: str, marker: str, block: str) -> str:
    pattern = re.compile(
        rf"(<!-- {marker}:start -->\n).*?(<!-- {marker}:end -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"no {marker} markers")
    return pattern.sub(lambda match: match[1] + block + "\n" + match[2], text)


def pyplot() -> Any:
    """matplotlib's pyplot, drawing SVG whose text stays text and whose ids
    stay the same from one rerun to the next."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "svg.hashsalt": "ajax",
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def agent_styles(runs: list[Run]) -> dict[str, tuple[str, str]]:
    """Each agent's colour and line, the same on every figure."""
    agents = ordered(run.agent for run in runs)
    return {
        agent: (HUES[i % len(HUES)], "-" if i < len(HUES) else "--")
        for i, agent in enumerate(agents)
    }


def plot_with_band(axis: Any, style: dict, agent: str, x, y, band) -> None:
    """An agent's line in its colour and dashes, over a light band."""
    colour, line = style[agent]
    axis.plot(x, y, color=colour, linestyle=line, linewidth=1.5, label=name(agent))
    axis.fill_between(x, band[0], band[1], color=colour, alpha=0.12, linewidth=0)


def write_figures(runs: list[Run], directory: pathlib.Path = FIGURES) -> None:
    """Each suite's heatmap and learning curves, as SVG."""
    plt = pyplot()
    style = agent_styles(runs)
    directory.mkdir(parents=True, exist_ok=True)
    for suite, short in SUITES.items():
        members = [run for run in runs if run.suite == suite]
        if not members:
            continue
        tasks = ordered(run.task for run in members)
        suite_agents = [agent for group in split_agents(members) for agent in group]
        cell = {(run.task, run.agent): run for run in members}

        grid = np.full((len(tasks), len(suite_agents)), np.nan)
        for (task, agent), run in cell.items():
            grid[tasks.index(task), suite_agents.index(agent)] = run.final.mean()
        best = np.nanmax(grid, axis=1, keepdims=True)
        scaled = grid / np.where(best > 0, best, 1)
        figure, axes = plt.subplots(
            figsize=(1.0 + 0.75 * len(suite_agents), 0.8 + 0.35 * len(tasks))
        )
        axes.imshow(np.clip(scaled, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), value in np.ndenumerate(grid):
            text = DASH if np.isnan(value) else f"{value:.0f}"
            dark = not np.isnan(value) and scaled[i, j] > 0.6
            axes.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else "#1a1a19",
            )
        axes.set_xticks(
            range(len(suite_agents)),
            [name(a) for a in suite_agents],
            rotation=45,
            ha="right",
        )
        axes.set_yticks(range(len(tasks)), tasks)
        axes.tick_params(length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_title(
            "Final return; colour relative to the task's best mean", loc="left"
        )
        figure.tight_layout()
        figure.savefig(directory / f"{short}-heatmap.svg", metadata={"Date": None})
        plt.close(figure)

        columns = min(4, len(tasks))
        rows = -(-len(tasks) // columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(2.6 * columns, 2.1 * rows + 0.6), squeeze=False
        )
        for axis, task in zip(axes.flat, tasks, strict=False):
            for agent in suite_agents:
                run = cell.get((task, agent))
                if run is None:
                    continue
                mean, spread = run.returns.mean(0), run.returns.std(0, ddof=1)
                band = (mean - spread, mean + spread)
                plot_with_band(axis, style, agent, run.steps, mean, band)
            axis.set_title(task, loc="left")
            axis.grid(alpha=0.25)
            axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        for axis in list(axes.flat)[len(tasks) :]:
            axis.set_visible(False)
        handles = {}
        for axis in axes.flat:
            handles.update(
                zip(*reversed(axis.get_legend_handles_labels()), strict=False)
            )
        figure.legend(
            handles.values(),
            handles.keys(),
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=False,
        )
        figure.supxlabel("Environment steps", y=0.06 + 0.02 / rows)
        figure.tight_layout(rect=(0, 0.1 / rows + 0.05, 1, 1))
        figure.savefig(directory / f"{short}-curves.svg", metadata={"Date": None})
        plt.close(figure)


def draw_scores(plt: Any, style: dict, scores: dict, path: pathlib.Path) -> None:
    """The interquartile mean of each agent's normalised final returns over
    its runs, with its 95% interval, beside the performance profile;
    scores[agent] holds each task's normalised returns, (seeds, evaluations)."""
    finals = {agent: [returns[:, -1] for returns in scores[agent]] for agent in scores}
    highest = max(float(np.concatenate(final).max()) for final in finals.values())
    top = max(1.0, highest)
    thresholds = np.linspace(0, top, 101)
    figure, (means, profiles) = plt.subplots(
        1, 2, figsize=(10.5, 1.5 + 0.3 * len(finals)), width_ratios=(1, 1.3)
    )
    for row, (agent, final) in enumerate(finals.items()):
        colour, _ = style[agent]
        pooled, resampled = np.concatenate(final), stratified_resamples(final)
        low, high = interval(interquartile_mean(resampled))
        means.plot([low, high], [row, row], color=colour, linewidth=4, alpha=0.4)
        means.plot(interquartile_mean(pooled), row, "o", color=colour)
        band = interval(performance_profile(resampled, thresholds))
        profile = performance_profile(pooled, thresholds)
        plot_with_band(profiles, style, agent, thresholds, profile, band)
    means.set_yticks(range(len(finals)), [name(agent) for agent in finals])
    means.set_ylim(len(finals) - 0.5, -0.5)
    means.set_xlim(0, top)
    means.set_title("Interquartile mean, 95% interval", loc="left")
    profiles.set_title("Performance profile", loc="left")
    profiles.set_ylabel("Fraction of runs above")
    profiles.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    for axis in (means, profiles):
        axis.set_xlabel("Normalised final return")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)


def draw_time(
    plt: Any, style: dict, panels: list, hours: dict, label: str, path: pathlib.Path
) -> None:
    """The interquartile mean through training against the hours each
    evaluation's step takes; panels: (title, {agent: each task's normalised
    returns}), hours: each timed agent's evaluations, in hours. Agents not
    timed yet are named in a corner."""
    figure, axes = plt.subplots(
        1, len(panels), figsize=(max(6.4, 4.7 * len(panels)), 3.9), squeeze=False
    )
    handles = {}
    for axis, (title, scores) in zip(axes[0], panels, strict=True):
        pending = [name(agent) for agent in scores if agent not in hours]
        if pending:
            names = "" if len(pending) == len(scores) else ": " + ", ".join(pending)
            note = f"Not timed yet{names}"
            axis.text(0.98, 0.04, note, transform=axis.transAxes, ha="right")
        if len(pending) == len(scores):
            axis.set_xlim(0.01, 100)  # no curve sets the log axis's range
        for agent, per_task in scores.items():
            if agent in hours:
                resampled = np.swapaxes(stratified_resamples(per_task), 1, 2)
                band = interval(interquartile_mean(resampled))
                curve = interquartile_mean(np.concatenate(per_task).T)
                plot_with_band(axis, style, agent, hours[agent], curve, band)
        axis.set_xscale("log")
        axis.set_title(title, loc="left")
        axis.grid(alpha=0.25)
        handles.update(zip(*reversed(axis.get_legend_handles_labels()), strict=True))
    axes[0][0].set_ylabel("Interquartile mean, normalised return")
    figure.legend(
        handles.values(), handles.keys(), loc="lower center", ncol=6, frameon=False
    )
    # The legend's rows take about 0.065 of the height each, the label above.
    label_height = 0.065 * -(-len(handles) // 6) + 0.01
    figure.supxlabel(label, y=label_height)
    figure.tight_layout(rect=(0, label_height + 0.05, 1, 1))
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)


def figure_hours(members: list[Run], timings: dict) -> tuple[dict, str]:
    """Each timed agent's evaluations in hours at the timing pass's speed,
    on one GPU once the pass ran there and on four CPU cores until then,
    and the axis label naming that machine and the timed task."""
    machine = (timed_machines(timings) or ["cpu"])[-1]
    hours, timed = {}, []
    for run in members:
        timing = timings.get((run.agent, run.suite, machine))
        if timing is not None:
            hours[run.agent] = run.steps / steps_per_second(timing) / 3600
            timed.append(timing)
    where = "four CPU cores"
    if machine == "gpu":
        where = f"one {timed[0]['manifest']['device']}" if timed else "one GPU"
    task = suite_and_task(timed[0]["env"])[1] if timed else members[0].task
    return hours, f"Hours on {where}, from the timing pass on {task}"


def write_aggregate_figures(
    runs: list[Run], timings: dict, directory: pathlib.Path = FIGURES
) -> None:
    """Each suite's scores over its tasks as rliable draws them (Agarwal et
    al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021),
    for the agents that ran every task: the interquartile mean of the
    normalised final returns over the runs (a task and a seed each) with a
    95% stratified bootstrap interval, beside the performance profile. Then
    that mean through training, against the hours to reach each evaluation
    at the timing pass's speed on the suite's first task (figure_hours);
    agents that ran only that task get a panel of it."""
    plt = pyplot()
    style, scales = agent_styles(runs), score_scales(runs)
    directory.mkdir(parents=True, exist_ok=True)
    for suite, short in SUITES.items():
        members = [run for run in runs if run.suite == suite]
        if not members:
            continue
        tasks = ordered(run.task for run in members)
        agents = [agent for group in split_agents(members) for agent in group]
        returns: dict = {agent: {} for agent in agents}
        for run in members:
            returns[run.agent][run.task] = run.returns / scales[(suite, run.task)]
        complete = {
            agent: list(returns[agent].values())
            for agent in agents
            if len(returns[agent]) == len(tasks)
        }
        if not complete:
            continue
        draw_scores(plt, style, complete, directory / f"{short}-scores.svg")
        noun = "games" if suite == "MinAtar" else "tasks"
        panels = [(f"{len(tasks)} {noun}", complete)]
        if len(complete) < len(agents):
            first_task = {
                agent: [returns[agent][tasks[0]]]
                for agent in agents
                if tasks[0] in returns[agent]
            }
            panels.append((tasks[0], first_task))
        hours, label = figure_hours(members, timings)
        draw_time(plt, style, panels, hours, label, directory / f"{short}-time.svg")


def write_gain_figure(
    runs: list[Run], by_agent: dict, directory: pathlib.Path = FIGURES
) -> None:
    """The GPU's gain over four CPU cores against the number of seeds trained
    together, a panel per suite's timing task, a line per agent; at 1 the two
    machines run as fast."""
    plt = pyplot()
    style = agent_styles(runs)
    suites = [suite for suite in SUITES if any(key[1] == suite for key in by_agent)]
    if not suites:
        return
    figure, axes = plt.subplots(
        1, len(suites), figsize=(4.7 * len(suites), 3.9), squeeze=False
    )
    for axis, suite in zip(axes[0], suites, strict=True):
        counts = set()
        for (agent, agent_suite), (task, per_seeds) in by_agent.items():
            if agent_suite != suite:
                continue
            seeds = sorted(per_seeds)
            counts.update(seeds)
            colour, line = style[agent]
            axis.plot(seeds, [per_seeds[n] for n in seeds], color=colour,
                      linestyle=line, marker="o", markersize=3, label=name(agent))
            axis.set_title(task, loc="left")
        axis.axhline(1, color="grey", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sorted(counts), [str(n) for n in sorted(counts)])
        axis.set_xlabel("Seeds trained together")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("Times as fast on one GPU as on four CPU cores")
    figure.tight_layout()
    figure.savefig(directory / "gain-by-seeds.svg", metadata={"Date": None})
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", action="store_true", help="skip the figures")
    args = parser.parse_args(argv)
    runs, timings = read_runs(), read_timings()
    PAGE.write_text(
        replace_block(PAGE.read_text(), "benchmark-tables", page_block(runs))
    )
    passes = seed_passes()
    performance = replace_block(
        PERFORMANCE_PAGE.read_text(), "agent-timing", timing_block(runs, timings)
    )
    performance = replace_block(
        performance, "agent-gain-by-seeds", gain_block(runs, gains(passes))
    )
    PERFORMANCE_PAGE.write_text(performance)
    if not args.tables:
        write_figures(runs)
        write_aggregate_figures(runs, timings)
        write_gain_figure(runs, gains(passes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
