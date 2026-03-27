"""
Plot training curves from plane_exps_awbc_sweep_clean.csv.

One line per experiment (mean across seeds, ±1 std shaded).
Groups are defined by the unique hyperparameter combination; the label is
taken from exp_name when available, otherwise reconstructed from hparams.

Usage:
    python plot_sweep.py                        # plots Eval/expert_bias
    python plot_sweep.py --metric Eval/episodic_mean_reward
    python plot_sweep.py --metric Eval/expert_bias --smooth 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_FILE = "ablation_study_awbc_debug_3.csv"
TB_DIR = Path("tensorboard")
OUT_DIR = Path("plots")

# Hyperparameters that distinguish experiments in the sweep.
# (All runs share augment_obs_with_expert_action; it's not needed for grouping.)
SWEEP_HPARAMS = [
    "use_expert_warmup",
    "use_mc_critic_pretrain",
    "use_bellman_critic_pretrain",
    "use_expert_guidance",
    "value_constraint_coef",
    "num_critics",
    "num_critic_updates",
    "expert_buffer_n_steps",
    "augment_obs_with_expert_action",
    "awbc_normalize",
    "awbc_use_relu",
    "fixed_awbc_lambda",
    "detach_obs_aug_action",
    "use_train_frac",
]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def build_label_from_hparams(row: pd.Series) -> str:
    """Compact human-readable label derived from hyperparameter values."""
    parts = []
    if not row.get("use_expert_warmup", True):
        parts.append("no_warmup")
    if row.get("augment_obs_with_expert_action", False):
        parts.append("obs_aug")
    if row.get("use_mc_critic_pretrain", False):
        parts.append("mc_pretrain")
    if row.get("use_expert_guidance", False):
        n = int(row.get("num_critic_updates", 1))
        parts.append(f"awbc_x{n}")
    vc = row.get("value_constraint_coef", 0.0)
    if vc and float(vc) > 0:
        parts.append(f"vc_{vc}")
    buf = row.get("expert_buffer_n_steps", 0)
    if not parts or (not row.get("augment_obs_with_expert_action") and not buf):
        parts = parts or ["baseline"]
    return "_".join(parts) if parts else "baseline"


def resolve_exp_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in a clean `label` column:
    - Use exp_name when it is not 'unknown'.
    - Rebuild from hparams otherwise.
    """
    df = df.copy()
    hparam_cols = [c for c in SWEEP_HPARAMS if c in df.columns]

    # Build per-run_id label (hparams are constant within a run)
    run_hparams = df.groupby("run_id")[["exp_name"] + hparam_cols].first()

    def _label(row):
        if row["exp_name"] != "unknown":
            return row["exp_name"]
        return build_label_from_hparams(row)

    run_label = run_hparams.apply(_label, axis=1).rename("label")
    df = df.join(run_label, on="run_id")
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

# Two runs started within this many seconds are considered the same batch.
_BATCH_GAP_SECONDS = 300


def deduplicate_runs(df: pd.DataFrame, tb_dir: Path = TB_DIR) -> pd.DataFrame:
    """
    When the same experiment was launched more than once, multiple batches of
    seeds end up in the data.  Keep only the most recent batch per label so
    every group has a consistent seed count.

    A "batch" is a cluster of runs whose tensorboard folders were all created
    within _BATCH_GAP_SECONDS of each other.
    """
    run_ids = df["run_id"].unique()
    mtime = {
        r: tb_dir.joinpath(r).stat().st_mtime
        for r in run_ids
        if tb_dir.joinpath(r).exists()
    }
    df = df.copy()
    df["mtime"] = df["run_id"].map(mtime)

    keep_ids: set[str] = set()
    for label, grp in df.groupby("label"):
        runs = (
            grp.groupby("run_id")["mtime"]
            .first()
            .sort_values()
            .reset_index()
        )
        # Cluster into batches by time gap
        batches: list[list[str]] = []
        batch: list[str] = [runs.iloc[0]["run_id"]]
        for prev, curr in zip(runs.itertuples(), runs.iloc[1:].itertuples()):
            if curr.mtime - prev.mtime < _BATCH_GAP_SECONDS:
                batch.append(curr.run_id)
            else:
                batches.append(batch)
                batch = [curr.run_id]
        batches.append(batch)

        if len(batches) > 1:
            kept = batches[-1]  # most recent batch
            dropped = sum(len(b) for b in batches[:-1])
            print(f"  {label}: keeping latest batch ({len(kept)} seeds), dropping {dropped} older seeds")
            keep_ids.update(kept)
        else:
            keep_ids.update(batch)

    return df[df["run_id"].isin(keep_ids)].drop(columns=["mtime"])


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    """Simple uniform moving-average over steps (min_periods=1)."""
    return s.rolling(window, min_periods=1, center=True).mean()


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap confidence interval of the mean for a 2-D array.

    Parameters
    ----------
    values : (n_steps, n_seeds) array
    n_bootstrap : number of bootstrap resamples
    ci : confidence level in percent (e.g. 95 → 2.5 % – 97.5 % interval)

    Returns
    -------
    lo, hi : arrays of shape (n_steps,)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    alpha = (100 - ci) / 2
    n_seeds = values.shape[1]
    # indices: (n_bootstrap, n_seeds)
    idx = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    # values[:, idx] → (n_steps, n_bootstrap, n_seeds); mean over seeds → (n_steps, n_bootstrap)
    boot_means = values[:, idx].mean(axis=2)
    lo = np.percentile(boot_means, alpha, axis=1)        # (n_steps,)
    hi = np.percentile(boot_means, 100 - alpha, axis=1)  # (n_steps,)
    return lo, hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour palette — distinct enough for up to ~15 groups
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def top_labels_by_asymptote(
    df: pd.DataFrame,
    metric: str,
    last_n_steps: int = 200_000,
    n: int = 10,
) -> list[str]:
    """Return the n labels with the highest mean metric over the last last_n_steps."""
    cutoff = df["step"].max() - last_n_steps
    tail = df[df["step"] >= cutoff]
    ranking = (
        tail.groupby("label")[metric]
        .mean()
        .sort_values(ascending=False)
    )
    return list(ranking.head(n).index)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    smooth: int = 1,
    figsize: tuple = (12, 6),
    labels: list[str] | None = None,
    ylim_bottom: float | None = None,
    n_bootstrap: int = 1000,
    ci: float = 95,
) -> plt.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in CSV. Available: {[c for c in df.columns if '/' in c]}")

    labels = sorted(labels if labels is not None else df["label"].unique())
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)

    for i, label in enumerate(labels):
        color = PALETTE[i % len(PALETTE)]
        group = df[df["label"] == label].copy()

        # Per-seed mean at each step → pivot to (n_steps × n_seeds) matrix
        per_seed = (
            group.groupby(["run_id", "step"])[metric]
            .mean()
            .unstack("run_id")          # rows=step, cols=run_id
            .sort_index()
        )
        steps = per_seed.index.values
        # Fill missing seed/step combos with the column mean (rare edge case)
        values = per_seed.fillna(per_seed.mean()).values  # (n_steps, n_seeds)

        mean = values.mean(axis=1)

        n_seeds = values.shape[1]
        if n_seeds > 1:
            ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        else:
            ci_lo, ci_hi = mean.copy(), mean.copy()

        if smooth > 1:
            mean = smooth_series(pd.Series(mean), smooth).values
            ci_lo = smooth_series(pd.Series(ci_lo), smooth).values
            ci_hi = smooth_series(pd.Series(ci_hi), smooth).values

        ax.plot(steps, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)

    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric}  (mean ± {ci:.0f}% bootstrap CI across seeds)", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"))
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.axhline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_asymptote_boxplot(
    df: pd.DataFrame,
    metric: str,
    labels: list[str],
    last_n_steps: int = 200_000,
    figsize: tuple = (12, 6),
    n_bootstrap: int = 1000,
    ci: float = 95,
) -> plt.Figure:
    """
    Boxplot of per-seed average return over the last `last_n_steps` steps,
    one column per label (sorted by mean descending).

    A diamond marker with bootstrap CI error bars is overlaid to show the
    uncertainty on the mean.
    """
    cutoff = df["step"].max() - last_n_steps

    # Per-seed tail mean for each label
    seed_means: dict[str, np.ndarray] = {}
    for label in labels:
        group = df[(df["label"] == label) & (df["step"] >= cutoff)]
        per_seed = group.groupby("run_id")[metric].mean()
        seed_means[label] = per_seed.values

    # Sort labels by mean descending
    labels_sorted = sorted(labels, key=lambda l: seed_means[l].mean(), reverse=True)

    rng = np.random.default_rng(42)
    alpha = (100 - ci) / 2

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(labels_sorted))

    box_data = [seed_means[l] for l in labels_sorted]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    for patch, label in zip(bp["boxes"], labels_sorted):
        color = PALETTE[labels_sorted.index(label) % len(PALETTE)]
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Bootstrap CI of the mean — shown as diamond + error bar
    for pos, label in zip(positions, labels_sorted):
        vals = seed_means[label]
        mean = vals.mean()
        color = PALETTE[labels_sorted.index(label) % len(PALETTE)]

        if len(vals) > 1:
            boots = rng.choice(vals, size=(n_bootstrap, len(vals)), replace=True).mean(axis=1)
            lo = np.percentile(boots, alpha)
            hi = np.percentile(boots, 100 - alpha)
        else:
            lo = hi = mean

        ax.errorbar(
            pos, mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="D",
            color=color,
            markersize=7,
            capsize=5,
            linewidth=1.8,
            zorder=5,
            label=f"{label}  (mean={mean:.1f})",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_sorted, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(
        f"{metric} — last {last_n_steps/1e3:.0f}k steps\n"
        f"box=seed distribution, ◆={ci:.0f}% bootstrap CI of mean",
        fontsize=12,
    )
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_FILE)
    parser.add_argument("--metric", default="Eval/expert_bias")
    parser.add_argument("--smooth", type=int, default=1, 
    help="Rolling-average window (steps)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CI")
    parser.add_argument("--ci", type=float, default=95, help="Confidence level in percent (default 95)")
    parser.add_argument("--top", type=int, default=5, 
    help="Number of best runs to plot")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    n = args.top


    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, {df['run_id'].nunique()} runs")

    df = resolve_exp_name(df)

    print("\nDeduplicating runs (keeping latest batch per label)...")
    df = deduplicate_runs(df)

    print(f"\nGroups ({df['label'].nunique()}):")
    for label, grp in df.groupby("label"):
        n_seeds = grp["run_id"].nunique()
        print(f"  {label:50s}  {n_seeds:3d} seeds")

    metric_slug = args.metric.replace("/", "_")
    smooth_tag = f"_smooth{args.smooth}" if args.smooth > 1 else ""

    print(f"\nPlotting {args.metric} — all groups...")
    fig = plot_metric(df, metric=args.metric, smooth=args.smooth, n_bootstrap=args.n_bootstrap, ci=args.ci)
    out_path = out_dir / f"{metric_slug}{smooth_tag}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)

    print(f"\nPlotting {args.metric} — top {n} by asymptotic return (last 200k steps)...")
    topn = top_labels_by_asymptote(df, metric=args.metric, last_n_steps=200_000, n=n)
    print(f"  Top {n}: {topn}")
    fig = plot_metric(df, metric=args.metric, smooth=args.smooth, labels=topn, ylim_bottom=-1000, n_bootstrap=args.n_bootstrap, ci=args.ci)
    out_path = out_dir / f"{metric_slug}{smooth_tag}_top{n}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)

    print(f"\nPlotting {args.metric} — asymptote boxplot (top {n})...")
    fig = plot_asymptote_boxplot(
        df, metric=args.metric, labels=topn,
        last_n_steps=200_000,
        n_bootstrap=args.n_bootstrap, ci=args.ci,
    )
    out_path = out_dir / f"{metric_slug}_boxplot_top{n}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
