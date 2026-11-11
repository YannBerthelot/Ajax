"""Write the agent benchmark's tables and figures from its records.

    uv run python benchmarks/suite/publish.py          # tables and figures
    uv run python benchmarks/suite/publish.py --tables # tables only

The tables go between markers in docs/reproduction/benchmark.md and in
README.md; the figures go to docs/reproduction/benchmark-figures/. A cell appears
once every seed of its run reached the step budget, so a suite's tab shows
up with its first finished run. tests/benchmarks/test_publish.py checks the
tables still match the records.
"""

import argparse
import dataclasses
import json
import pathlib
import re
import sys
from typing import Optional

import numpy as np
from agent_suite import BENCHMARKS

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "suite" / "results"
PAGE = ROOT / "docs" / "reproduction" / "benchmark.md"
README = ROOT / "README.md"
FIGURES = ROOT / "docs" / "reproduction" / "benchmark-figures"

#: Suite name -> short name for its figure files, in tab order.
SUITES = {
    "DeepMind Control": "control",
    "Gymnasium MuJoCo": "mujoco",
    "MinAtar": "minatar",
}
#: The README's columns, each shown once a run on it has finished.
HEADLINE_TASKS = {
    "DeepMind Control": ("CheetahRun", "WalkerWalk"),
    "Gymnasium MuJoCo": ("HalfCheetah-v5", "Hopper-v5"),
    "MinAtar": ("Breakout", "SpaceInvaders"),
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


def readme_block(runs: list[Run]) -> str:
    """The README's summary: agents down, a few headline tasks across, means
    rounded to whole numbers."""
    agents = ordered(run.agent for run in runs)
    seeds = sorted({len(run.final) for run in runs})
    suites = [suite for suite in SUITES if any(run.suite == suite for run in runs)]
    lines = [
        f"{len(agents)} agents on {len(ordered((r.suite, r.task) for r in runs))}"
        f" {', '.join(suites[:-1])} and {suites[-1]} tasks, with"
        f" {'/'.join(map(str, seeds))} seeds each and default settings apart"
        " from the number of environments. These are mean returns at the end"
        " of training: about a million steps on control tasks, and about five"
        " million on MinAtar. DeepMind Control tasks score up to 1000."
    ]
    cell = {(name(run.agent), run.task): run for run in runs}
    for group in (("DeepMind Control", "Gymnasium MuJoCo"), ("MinAtar",)):
        tasks = [
            task
            for suite in group
            for task in HEADLINE_TASKS[suite]
            if any(run.task == task and run.suite == suite for run in runs)
        ]
        rows = ordered(name(run.agent) for run in runs if run.task in tasks)
        if not tasks:
            continue
        head = "| Agent | " + " | ".join(tasks) + " |"
        body = []
        for row in rows:
            values = [cell.get((row, task)) for task in tasks]
            body.append(
                f"| {row} | "
                + " | ".join(
                    DASH if run is None else f"{run.final.mean():.0f}" for run in values
                )
                + " |"
            )
        lines.append("\n".join([head, "|---" * (len(tasks) + 1) + "|", *body]))
    lines.append(
        "[Agent benchmark](docs/reproduction/benchmark.md) has every task, the"
        " spread over seeds, learning curves and what to keep in mind when"
        " reading them."
    )
    return "\n\n".join(lines)


def replace_block(text: str, marker: str, block: str) -> str:
    pattern = re.compile(
        rf"(<!-- {marker}:start -->\n).*?(<!-- {marker}:end -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"no {marker} markers")
    return pattern.sub(lambda match: match[1] + block + "\n" + match[2], text)


def write_figures(runs: list[Run], directory: pathlib.Path = FIGURES) -> None:
    """Each suite's heatmap and learning curves, as SVG. Colours are a fixed
    eight-hue palette; agents past the eighth reuse a hue with a dashed line,
    so each agent keeps its look on every panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hues = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ]
    agents = ordered(run.agent for run in runs)
    style = {
        agent: (hues[i % len(hues)], "-" if i < len(hues) else "--")
        for i, agent in enumerate(agents)
    }
    plt.rcParams.update(
        {
            "font.size": 9,
            "svg.hashsalt": "ajax",
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
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
                colour, line = style[agent]
                mean, spread = run.returns.mean(0), run.returns.std(0, ddof=1)
                axis.plot(
                    run.steps,
                    mean,
                    color=colour,
                    linestyle=line,
                    linewidth=1.5,
                    label=name(agent),
                )
                axis.fill_between(
                    run.steps,
                    mean - spread,
                    mean + spread,
                    color=colour,
                    alpha=0.12,
                    linewidth=0,
                )
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", action="store_true", help="skip the figures")
    args = parser.parse_args(argv)
    runs = read_runs()
    PAGE.write_text(
        replace_block(PAGE.read_text(), "benchmark-tables", page_block(runs))
    )
    README.write_text(
        replace_block(README.read_text(), "benchmark-summary", readme_block(runs))
    )
    if not args.tables:
        write_figures(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
