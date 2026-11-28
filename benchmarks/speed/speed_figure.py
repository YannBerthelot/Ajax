"""Draws the speed comparison's figure and writes its tables from the records.

usage: uv run python benchmarks/speed/speed_figure.py

Each cell shows the latest record of its algorithm, task, library, seed count
and device in results/speed.jsonl. A library that runs one process per seed
shows ten seeds at the faster of its ten processes at once and its one-seed
speed, which its seeds make one after another. The headline of a family is
Ajax's speed over the fastest other library's, with ten seeds on four CPU
cores, from its lowest to its highest cell. The figure goes to
docs/guide/performance-figures/; the table and headlines go between markers
in docs/guide/performance.md, and the headline sentence into the README.
tests/benchmarks/test_speed.py checks both still match the records.
"""

import json
import pathlib
import re

import speed_specs as specs

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECORDS = HERE / "results" / "speed.jsonl"
PAGE = ROOT / "docs" / "guide" / "performance.md"
README = ROOT / "README.md"
FIGURE = ROOT / "docs" / "guide" / "performance-figures" / "speed-by-family.svg"
RECORDS_URL = "https://github.com/YannBerthelot/Ajax/blob/main/benchmarks/speed/results/speed.jsonl"
PAGE_URL = "https://yannberthelot.github.io/Ajax/latest/guide/performance/#against-other-libraries"
#: Ajax in the benchmark figures' blue, every other library in grey: each
#: row names its library, so colour only singles Ajax out (seven hues cannot
#: all be told apart, side by side, by colour-blind readers).
AJAX, OTHER, MUTED = "#2a78d6", "#5f5e5a", "#8a8984"
DEVICES = {"cpu": "Four CPU cores", "gpu": "One NVIDIA L40S"}


def read(path: pathlib.Path = RECORDS) -> dict:
    """(algorithm, task, seeds, device) -> library -> (steps per second, how)."""
    latest = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        r = json.loads(line)
        latest[(r["algorithm"], r["task"], r["library"], r["seeds"], r["device"])] = r
    speeds: dict = {}
    for (algorithm, task, library, seeds, device), r in latest.items():
        speed, how = r["env_steps_per_s"], r["how"]
        one = latest.get((algorithm, task, library, 1, device))
        if seeds == 10 and how != "program" and one and one["env_steps_per_s"] > speed:
            speed, how = one["env_steps_per_s"], "one after another"
        speeds.setdefault((algorithm, task, seeds, device), {})[library] = (speed, how)
    for (algorithm, task, seeds, device), row in list(speeds.items()):
        # Brax's seeds run one after another on a GPU: its one-seed speed.
        if device == "gpu" and seeds == 1 and "Brax" in row:
            ten = speeds.setdefault((algorithm, task, 10, device), {})
            ten.setdefault("Brax", (row["Brax"][0], "one after another"))
    return speeds


def rows() -> list[tuple[str, str, str, int]]:
    """(family, algorithm, task, seeds), in the figure's order."""
    return [
        (family, algorithm, task, seeds)
        for family, algorithms in specs.FAMILIES.items()
        for algorithm in algorithms
        for task in specs.TASKS[algorithm]
        for seeds in (1, 10)
    ]


def factor(row: dict):
    """Ajax's speed over the fastest other library's, and that library."""
    others = {lib: v[0] for lib, v in row.items() if lib != "Ajax"}
    if "Ajax" not in row or not others:
        return None, None
    best = max(others, key=others.get)
    return row["Ajax"][0] / others[best], best


def number(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def extremes(speeds: dict, device: str) -> dict:
    """Family -> its lowest and highest ten-seed (factor, library, algorithm,
    task)."""
    out: dict = {}
    for family, algorithm, task, seeds in rows():
        ratio, best = factor(speeds.get((algorithm, task, seeds, device), {}))
        if seeds == 10 and ratio is not None:
            cell = (ratio, best, algorithm, task)
            low, high = out.get(family, (cell, cell))
            out[family] = (min(low, cell), max(high, cell))
    return out


def headlines(speeds: dict, device: str = "cpu") -> dict:
    """Family -> (lowest, highest) factor over its cells at ten seeds."""
    return {f: (low[0], high[0]) for f, (low, high) in extremes(speeds, device).items()}


def span(low: float, high: float) -> str:
    if f"{low:.1f}" == f"{high:.1f}":
        return f"{low:.1f}"
    return f"{low:.1f} to {high:.1f}"


def table(speeds: dict) -> str:
    lines = []
    intro = {
        "cpu": "With 10 seeds on four CPU cores, Ajax trains at",
        "gpu": "With 10 seeds on one NVIDIA L40S, against the JAX libraries, at",
    }
    for device in DEVICES:
        found = extremes(speeds, device)
        if found:
            lines += [intro[device], ""]
        for family, (low, high) in found.items():
            ends = [
                f"{r:.1f} times {lib}'s speed ({a}, {t})"
                for r, lib, a, t in (low, high)
            ]
            text = ends[0] if low == high else f"{ends[0]} to {ends[1]}"
            lines.append(f"- **{family}**: {text}")
        if found:
            lines.append("")
    lines += [
        "| Algorithm | Task | Seeds | Device | Ajax | Fastest other | Ajax's factor |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for device in DEVICES:
        for _, algorithm, task, seeds in rows():
            row = speeds.get((algorithm, task, seeds, device), {})
            ratio, best = factor(row)
            if ratio is None:
                continue
            other = f"{best}, {number(row[best][0])}"
            if row[best][1] == "one after another":
                other += " (one after another)"
            lines.append(
                f"| {algorithm} | {task} | {seeds} | {device.upper()} | "
                f"{number(row['Ajax'][0])} | {other} | {ratio:.2f} |"
            )
    lines.append(
        "\nEnvironment steps per second over every seed together, from a"
        f" stretch of training once the updates run ([records]({RECORDS_URL}))."
    )
    return "\n".join(lines)


#: The settings table's rows: a setting and how to show its value.
LABELS = {
    "n_envs": "Environments per seed",
    "n_steps": "Rollout steps per environment",
    "n_epochs": "Epochs per rollout",
    "n_minibatches": "Minibatches per epoch",
    "batch_size": "Batch size",
    "buffer_size": "Replay buffer",
    "updates_per_step": "Updates per environment step",
    "train_frequency": "Environment steps per update",
    "target_update": "Environment steps per target copy",
    "learning_starts": "Updates start after",
    "hidden": "Hidden layers",
    "activation": "Activation",
    "layer_norm": "Layer norm",
    "learning_rate": "Learning rate",
    "gamma": "Discount",
    "tau": "Target smoothing",
    "policy_delay": "Policy delay",
    "target_noise": "Target noise (clip)",
    "exploration_noise": "Exploration noise",
    "epsilon": "Epsilon (final, decay share)",
    "gae_lambda": "GAE lambda",
    "q_lambda": "Q(lambda)",
    "clip_range": "Clip range",
    "ent_coef": "Entropy coefficient",
    "max_grad_norm": "Gradient norm clip",
    "normalize": "Observation and reward normalisation",
}


def shown(key: str, s: dict) -> str:
    value = s.get(key)
    if value is None:
        return ""
    if key == "hidden":
        return "-".join(map(str, value))
    if key == "epsilon":
        return f"{value[0]:g} to {value[1]:g} ({value[2]:.0%})"
    if key == "target_noise":
        return f"{value:g} ({s['target_noise_clip']:g})"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return (
        f"{value:,}"
        if isinstance(value, int)
        else f"{value:g}"
        if isinstance(value, float)
        else str(value)
    )


def settings_block() -> str:
    """Every algorithm's settings, and what a library keeps of its own."""
    columns = [(a, t) for a in specs.TASKS for t in specs.TASKS[a]]
    columns = [
        c for c in columns if not (c[0] in ("SAC", "TD3") and c[1] != "Pendulum-v1")
    ]
    names = [f"{a}, {t}" if a == "PPO" else a for a, t in columns]
    lines = [
        "| Setting | " + " | ".join(names) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for key, label in LABELS.items():
        cells = [shown(key, specs.settings(a, t)) for a, t in columns]
        if any(cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines += [
        f"- {library}'s {a}: {text}." for (a, library), text in specs.EXCEPTIONS.items()
    ]
    return "\n".join(lines)


def sentence(speeds: dict) -> str:
    """The README's line: each family's range of factors, per device."""
    parts = []
    where = {
        "cpu": "on four CPU cores, against the fastest other library",
        "gpu": "On one NVIDIA L40S, against the JAX libraries",
    }
    for device in DEVICES:
        heads = headlines(speeds, device)
        each = [f"{family} at {span(*heads[family])} times" for family in heads]
        if each:
            parts.append(
                f"{where[device]}, Ajax trained " + ", ".join(each) + " its speed."
            )
    if not parts:
        return ""
    return "With 10 seeds " + " ".join(parts) + f" ([Figure and settings]({PAGE_URL}).)"


def draw(speeds: dict, path: pathlib.Path = FIGURE) -> None:
    """One row per library and task, its one-seed speed (hollow) joined to
    its ten-seed speed (filled), on four CPU cores and on the GPU; each
    task's ten-seed factor at the right of Ajax's row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "svg.hashsalt": "ajax",
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    groups = {
        f: [(a, t) for a in algorithms for t in specs.TASKS[a]]
        for f, algorithms in specs.FAMILIES.items()
    }
    heights = [sum(len(specs.LIBRARIES[a]) + 1 for a, _ in g) for g in groups.values()]
    figure, axes = plt.subplots(
        len(groups),
        2,
        figsize=(10, 0.6 + 0.165 * sum(heights)),
        gridspec_kw={"height_ratios": heights},
        sharex="col",
    )
    for column, device in enumerate(DEVICES):
        for (family, group), axis in zip(groups.items(), axes[:, column], strict=True):
            labels, y = [], 0  # (text, colour, weight) per row
            for algorithm, task in group:
                labels.append((f"{algorithm}, {task}", "#0b0b0b", "bold"))
                y += 1
                for library in specs.LIBRARIES[algorithm]:
                    if (algorithm, task, library) in specs.UNSUPPORTED:
                        continue
                    one = speeds.get((algorithm, task, 1, device), {}).get(library)
                    ten = speeds.get((algorithm, task, 10, device), {}).get(library)
                    colour = AJAX if library == "Ajax" else OTHER
                    if one and ten:
                        axis.plot(
                            [one[0], ten[0]],
                            [y, y],
                            color=colour,
                            linewidth=1,
                            alpha=0.6,
                        )
                    if one:
                        axis.plot(
                            one[0],
                            y,
                            "o",
                            markersize=5.5,
                            markerfacecolor="white",
                            markeredgecolor=colour,
                            markeredgewidth=1.3,
                        )
                    if ten:
                        axis.plot(
                            ten[0],
                            y,
                            "o",
                            markersize=5.5,
                            color=colour,
                            markeredgecolor="white",
                            markeredgewidth=0.6,
                        )
                    ratio, _ = factor(speeds.get((algorithm, task, 10, device), {}))
                    if library == "Ajax" and ratio is not None:
                        axis.annotate(
                            f"{ratio:.1f}x",
                            (1.0, y),
                            xycoords=("axes fraction", "data"),
                            xytext=(3, 0),
                            textcoords="offset points",
                            va="center",
                            fontsize=8,
                            color=AJAX,
                        )
                    physics = specs.physics(task, library)
                    labels.append(
                        (
                            library + (f" ({physics})" if physics else ""),
                            colour,
                            "normal",
                        )
                    )
                    y += 1
            axis.set_xscale("log")
            axis.set_yticks(
                range(len(labels)), [x[0] if column == 0 else "" for x in labels]
            )
            for tick, (_, colour, weight) in zip(
                axis.get_yticklabels(), labels, strict=True
            ):
                tick.set_color(colour)
                tick.set_fontweight(weight)
            axis.tick_params(axis="y", length=0)
            axis.set_ylim(len(labels) - 0.5, -0.5)
            axis.grid(axis="x", color="#e5e4e0", linewidth=0.6)
            axis.set_axisbelow(True)
            if family == next(iter(groups)):
                axis.set_title(DEVICES[device], fontsize=9, loc="left", color="#0b0b0b")
        axes[-1, column].set_xlabel(
            "Environment steps per second, every seed together (log scale)"
        )
    keys = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=OTHER,
            label="1 seed",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5.5,
            color=OTHER,
            label="10 seeds",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="",
            label="2.0x: Ajax over the fastest other library, 10 seeds",
        ),
    ]
    figure.legend(
        handles=keys,
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
        labelcolor=MUTED,
    )
    figure.tight_layout(rect=(0, 0, 0.97, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Date": None})


def replace_block(text: str, marker: str, block: str) -> str:
    """``text`` with ``block`` between the marker comments, indented as the
    start marker is (inside an admonition, say)."""
    pattern = re.compile(
        rf"^([ \t]*)(<!-- {marker}:start -->\n).*?^[ \t]*(<!-- {marker}:end -->)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"no {marker} markers")
    indent = match[1]
    body = "".join(indent + line + "\n" if line else "\n" for line in block.split("\n"))
    replaced = indent + match[2] + body + indent + match[3]
    return text[: match.start()] + replaced + text[match.end() :]


def main() -> None:
    speeds = read()
    draw(speeds)
    page = replace_block(PAGE.read_text(), "speed-by-family", table(speeds))
    PAGE.write_text(replace_block(page, "speed-settings", settings_block()))
    README.write_text(
        replace_block(README.read_text(), "speed-by-family", sentence(speeds))
    )


if __name__ == "__main__":
    main()
