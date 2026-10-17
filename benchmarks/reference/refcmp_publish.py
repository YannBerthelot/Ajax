"""Publish the protocol's records: a dataset, and the docs page drawn from it.

usage, from the repository root:
    uv run python benchmarks/reference/refcmp_publish.py dataset OUT_DIR
    uv run python benchmarks/reference/refcmp_publish.py docs
    uv run --extra reproduction python benchmarks/reference/refcmp_publish.py \
        upload OUT_DIR REPOSITORY

``dataset`` writes ``runs.parquet`` (one row per seed and evaluation, from
every record, current or not: ``fingerprint`` tells them apart),
``batches.jsonl`` (each batch's speed and manifest), ``summary.jsonl`` (each
arm's verdict, difference, resolution and speed), ``specs.json`` and the
dataset card. ``docs`` writes ``docs/reproduction/results.md`` and a plot per
comparison. ``upload`` sends a dataset directory to a Hugging Face dataset
repository, as the account logged in with ``huggingface-cli login``.
"""

import dataclasses
import json
import pathlib
import shutil
import sys

import numpy as np
import refcmp_protocol as protocol
from refcmp_specs import COMPARISONS

DOCS = protocol.ROOT / "docs" / "reproduction"
# The reference palette's first two categorical slots.
COLORS = {"reference": "#2a78d6", "ajax": "#eb6834"}


def recorded() -> list:
    return sorted(p.stem for p in protocol.CACHE.glob("*.jsonl"))


def dataset(out: pathlib.Path, reps: int = 200) -> None:
    import pandas as pd

    rows, batches = [], []
    for name in recorded():
        for i, r in enumerate(protocol.records(name)):
            batch = f"{name}/{i}"
            batches.append(
                {"batch": batch} | {k: v for k, v in r.items() if k != "curve"}
            )
            keys = {k: r[k] for k in ("side", "implementation", "fingerprint")}
            for seed, curve in zip(r["seeds"], r["curve"], strict=True):
                for step, value in zip(r["steps"], curve, strict=True):
                    rows.append(
                        {"comparison": name, **keys, "batch": batch, "seed": seed}
                        | {"step": step, "return": value}
                    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "runs.parquet", index=False)
    summaries = [
        row for name in recorded() for row in protocol.summary(name, reps=reps)
    ]
    for table, items in (("batches", batches), ("summary", summaries)):
        text = "".join(json.dumps(item) + "\n" for item in items)
        (out / f"{table}.jsonl").write_text(text)
    specs = {name: dataclasses.asdict(COMPARISONS[name]) for name in recorded()}
    (out / "specs.json").write_text(json.dumps(specs, indent=1))
    shutil.copy(protocol.HERE / "DATASET_CARD.md", out / "README.md")


def plot(name: str, path: pathlib.Path) -> None:
    """Each side's mean evaluation return over its current seeds, with a
    band of two standard errors."""
    import matplotlib.pyplot as plt

    spec = COMPARISONS[name]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for side, label in (("reference", spec.reference), ("matched", "Ajax")):
        color = COLORS["reference" if side == "reference" else "ajax"]
        values = np.array(list(protocol.curves(name, side).values()))
        steps = protocol.current(name, side)[0]["steps"]
        mean = values.mean(0)
        half = 2 * values.std(0, ddof=1) / np.sqrt(len(values))
        ax.fill_between(steps, mean - half, mean + half, color=color, alpha=0.2, lw=0)
        ax.plot(steps, mean, color=color, lw=2, label=f"{label} ({len(values)} seeds)")
    ax.set(title=f"{spec.agent} on {spec.env_id}", xlabel="steps", ylabel="return")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for item in (
        ax.title,
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.get_legend().get_texts(),
    ):
        item.set_color("#7a7974")
    ax.tick_params(colors="#7a7974")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def docs(reps: int = 200) -> None:
    import matplotlib

    matplotlib.use("Agg")
    words = {"equal": "no difference shown", "larger": "Ajax larger"}
    lines = [
        "# Results",
        "",
        "Written by `benchmarks/reference/refcmp_publish.py docs` from the",
        "records the dataset publishes. Ajax is its matched arm. The",
        "difference is Ajax's mean final return minus the reference's, with",
        "its 95% bootstrap interval; the resolution is the smallest gap",
        "AdaStop's budget detects 80% of the time. Speed is measured on the CPU,",
        "medians over batches.",
        "",
        "| Comparison | AdaStop | Seeds | Reference | Ajax | Difference | Resolution |",
        "|---|---|---|---|---|---|---|",
    ]
    speeds = [
        "",
        "## Speed",
        "",
        "| Comparison | Side | Steps per second | CPU seconds per 1M steps | Compile seconds |",
        "|---|---|---|---|---|",
    ]
    for name in recorded():
        for row in protocol.summary(name, reps=reps):
            low, high = row["interval"]
            decision = words.get(row["decision"], "Ajax smaller")
            lines.append(
                f"| {name} | {decision} | {row['seeds']} | {row['reference_mean']:.1f} "
                f"| {row['ajax_mean']:.1f} | {row['difference']:+.1f} "
                f"[{low:+.1f}, {high:+.1f}] | {row['resolution']:.1f} |"
            )
            for side, label in (
                ("reference", COMPARISONS[name].reference),
                ("ajax", "Ajax"),
            ):
                s = row["speed"][side]
                speeds.append(
                    f"| {name} | {label} | {s['env_steps_per_s']:,.0f} "
                    f"| {s['cpu_s_per_1m_steps']:,.0f} | {s['compile_s']:.1f} |"
                )
    machine = protocol.records(recorded()[0])[0]["manifest"]["hardware"]
    speeds += ["", f"Machine: {machine['cpu']}, {machine['cores']} cores.", ""]
    figures = ["## Learning curves", ""]
    for name in recorded():
        plot(name, DOCS / f"{name}.svg")
        figures += [f"![{name}]({name}.svg)", ""]
    (DOCS / "results.md").write_text("\n".join(lines + speeds + figures))


def upload(out: pathlib.Path, repository: str) -> None:
    from huggingface_hub import HfApi

    commit = protocol._run("git", "rev-parse", "--short", "HEAD")
    HfApi().upload_folder(
        folder_path=out,
        repo_id=repository,
        repo_type="dataset",
        commit_message=f"Records as of Ajax {commit}",
    )


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "dataset":
        dataset(pathlib.Path(args[0]))
    elif command == "docs":
        docs()
    elif command == "upload":
        upload(pathlib.Path(args[0]), args[1])
