"""Publish the protocol's records: a dataset, and the docs page drawn from it.

usage, from the repository root:
    uv run python benchmarks/reference/refcmp_publish.py dataset OUT_DIR
    uv run python benchmarks/reference/refcmp_publish.py docs
    uv run python benchmarks/reference/refcmp_publish.py specs
    uv run --extra reproduction python benchmarks/reference/refcmp_publish.py \
        upload OUT_DIR REPOSITORY

``dataset`` writes ``runs.parquet`` (one row per seed and evaluation, from
every record, current or not: ``fingerprint`` tells them apart),
``batches.jsonl`` (each batch's speed and manifest), ``summary.jsonl`` (each
arm's verdict, difference, resolution and speed), ``specs.json`` and the
dataset card. ``docs`` writes ``docs/reproduction/results.md`` and a plot per
comparison, ``specs`` ``docs/reproduction/specs.md``. ``upload`` sends a dataset directory to a Hugging Face dataset
repository, as the account logged in with ``huggingface-cli login``.
"""

import dataclasses
import json
import pathlib
import shutil
import sys

import numpy as np
import refcmp_protocol as protocol
from refcmp_specs import COMPARISONS, spec_sha256

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
            summary = {"batch": batch}
            summary.update({k: v for k, v in r.items() if k != "curve"})
            batches.append(summary)
            keys = {k: r[k] for k in ("side", "implementation", "fingerprint")}
            for seed, curve in zip(r["seeds"], r["curve"], strict=True):
                for step, value in zip(r["steps"], curve, strict=True):
                    rows.append(
                        {
                            "comparison": name,
                            **keys,
                            "batch": batch,
                            "seed": seed,
                            "step": step,
                            "return": value,
                        }
                    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "runs.parquet", index=False)
    summaries = [
        row for name in recorded() for row in protocol.summary(name, reps=reps)
    ]
    for table, items in (("batches", batches), ("summary", summaries)):
        text = "".join(json.dumps(item) + "\n" for item in items)
        (out / f"{table}.jsonl").write_text(text)
    specs = {
        name: {
            **dataclasses.asdict(COMPARISONS[name]),
            "sha256": spec_sha256(COMPARISONS[name]),
        }
        for name in recorded()
    }
    (out / "specs.json").write_text(json.dumps(specs, indent=1))
    shutil.copy(protocol.HERE / "DATASET_CARD.md", out / "README.md")


def margin_text(spec) -> str:
    if spec.margin is not None:
        return f"±{spec.margin:g}"
    if spec.relative_margin is not None:
        return f"±{spec.relative_margin:.0%} of the reference's mean score"
    return "none registered yet"


def specs_page() -> str:
    """docs/reproduction/specs.md: every comparison's settings, side by side."""
    lines = [
        "# Specifications",
        "",
        "Written by `benchmarks/reference/refcmp_publish.py specs` from",
        "`benchmarks/reference/refcmp_specs.py`, which a test keeps it equal to.",
        "Each comparison is set before its runs and changed only as a new",
        "version (see the [protocol](protocol.md)).",
        "",
    ]
    for name, spec in COMPARISONS.items():
        n, K, alpha = spec.adastop
        schedule = (
            f"every {spec.eval_every:,} steps"
            if spec.eval_every
            else "once, at the end"
        )
        lines += [
            f"## {name}, version {spec.version}",
            "",
            f"Ajax's {spec.agent} against {spec.source}.",
            "",
            f"- Environment: {spec.environment}.",
            f"- Budget: {spec.n_timesteps:,} steps per seed.",
            f"- Evaluation: {spec.n_eval_episodes} episodes of the deterministic"
            f" policy, {schedule}; a seed's score is the last one's mean return.",
            f"- AdaStop: n = {n}, K = {K}, alpha = {alpha}.",
            f"- Margin: {margin_text(spec)}.",
            f"- Spec hash: `{spec_sha256(spec)[:16]}`.",
            "- Reproduce: `uv run python benchmarks/reference/refcmp_protocol.py"
            f" run {name} --cache DIR`",
            "",
            "| Reference setting | Value | Ajax setting | Value |",
            "|---|---|---|---|",
        ]
        copied = set()
        for key, value in spec.reference_config.items():
            names = spec.renames.get(key, key)
            ajax = [
                k
                for k in ((names,) if isinstance(names, str) else names)
                if k in spec.matched
            ]
            copied.update(ajax)
            values = ", ".join(f"`{spec.matched[k]!r}`" for k in ajax)
            keys = ", ".join(f"`{k}`" for k in ajax) or "-"
            lines.append(f"| `{key}` | `{value!r}` | {keys} | {values} |")
        rest = ", ".join(
            f"`{k}={v!r}`" for k, v in spec.matched.items() if k not in copied
        )
        lines += ["", f"Ajax settings without a reference counterpart: {rest}.", ""]
        lines += ["Not matched:", "", *[f"- {u}" for u in spec.unmatched], ""]
        if spec.deviations:
            lines += [
                "Deviations found by the conformance checks (`tests/conformance`):",
                "",
            ]
            lines += [
                f"- {d.name} ({d.kind}, {d.submap}): {d.description} Code: {d.where}."
                f" Pinned by `{d.check}`."
                for d in spec.deviations
            ]
            lines.append("")
    return "\n".join(lines)


def plot(name: str, path: pathlib.Path) -> None:
    """Each side's mean evaluation return over the reported seeds (the
    latest spec version's), with a band of two standard errors."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _, keys, first = protocol.reported(name)
    spec = protocol.spec_of(name, first)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for side, label in (("reference", spec.reference), ("matched", "Ajax")):
        color = COLORS["reference" if side == "reference" else "ajax"]
        values = np.array(list(protocol.curves(name, side, keys[side]).values()))
        steps = protocol.current(name, side, keys[side])[0]["steps"]
        mean = values.mean(0)
        half = 2 * values.std(0, ddof=1) / np.sqrt(len(values))
        ax.fill_between(steps, mean - half, mean + half, color=color, alpha=0.2, lw=0)
        ax.plot(steps, mean, color=color, lw=2, label=f"{label} ({len(values)} seeds)")
    ax.set(title=f"{spec.agent} on {spec.env_id}", xlabel="steps", ylabel="return")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
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
    words = {"equal": "none shown", "larger": "Ajax larger"}
    lines = [
        "# Results",
        "",
        "Written by `benchmarks/reference/refcmp_publish.py docs` from the",
        "records the dataset publishes. Ajax is its matched arm. The",
        "difference is Ajax's mean final return minus the reference's, with",
        "its 95% bootstrap interval; the verdict sets that interval against the",
        "registered margin. AdaStop names the difference it detected, and the",
        "resolution is the smallest gap its budget detects 80% of the time.",
        "Speed is measured on the CPU, medians over batches.",
        "",
        "Each row reports its comparison's latest spec version, whose settings",
        "the [specifications](specs.md) list, and the Ajax commit it ran.",
        "",
        "| Comparison | Spec, Ajax | Verdict | Margin | Seeds | Reference | Ajax "
        "| Difference | AdaStop | Resolution |",
        "|---|---|---|---|---|---|---|---|---|---|",
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
            margin = "-" if row["margin"] is None else f"±{row['margin']:.3g}"
            commit = (row["ajax_commit"] or "")[:7]
            lines.append(
                f"| {name} | v{row['spec_version']}, {commit} | {row['verdict']} "
                f"| {margin} | {row['seeds']} | {row['reference_mean']:.4g} "
                f"| {row['ajax_mean']:.4g} | {row['difference']:+.3g} "
                f"[{low:+.3g}, {high:+.3g}] | {decision} | {row['resolution']:.2g} |"
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
    elif command == "specs":
        (DOCS / "specs.md").write_text(specs_page())
    elif command == "upload":
        upload(pathlib.Path(args[0]), args[1])
