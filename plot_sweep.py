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
import json
import struct
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D as _Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Custom legend handlers
# ---------------------------------------------------------------------------

class _InvisibleHandler(HandlerBase):
    """Renders nothing in the handle area — for seeds header and separator rows."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [_Line2D([], [], visible=False, transform=trans)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_FILE = "ablation_study_awbc_debug_3.csv"
TB_DIR = Path("tensorboard")
OUT_DIR = Path("plots")
RUN_REGISTRY_FILE = Path("run_registry.json")
ABLATION_REGISTRY_FILE = Path("ablation_run_registry.json")

# ---------------------------------------------------------------------------
# Display names — maps raw exp_name to short paper-ready label.
# UTD4 variants are handled automatically by appending " (UTD=4)".
# ---------------------------------------------------------------------------
EXP_DISPLAY_NAMES: dict[str, str] = {
    "sac_baseline":        "SAC",
    "ege_only":            "EGE",
}

# Metric axis labels and titles
METRIC_AXIS_LABELS: dict[str, str] = {
    "Eval/expert_bias":              "Expert Advantage",
    "Eval/episodic_mean_reward":     "Mean Episode Return",
    "ege_expert_action_fraction":    "EGE Expert Action Fraction",
    "ege_value_gap":                 "EGE Value Gap",
    "policy/altitude_error":         "Altitude Error",
    "temperature/alpha":             "Alpha (Temperature)",
}
METRIC_TITLES: dict[str, str] = {
    "Eval/expert_bias":              "Expert Advantage over Training",
    "Eval/episodic_mean_reward":     "Episode Return over Training",
    "ege_expert_action_fraction":    "EGE Expert Action Fraction",
    "ege_value_gap":                 "EGE Value Gap",
    "policy/altitude_error":         "Altitude Error",
    "temperature/alpha":             "Alpha (Temperature)",
}

# Experiments shown in Eval/expert_bias graphs (raw exp_name values).
EXPERT_BIAS_EXPERIMENTS = {
    "sac_baseline",
    "ege_only",
}

# Metrics shown in the per-experiment summary grid
SUMMARY_METRICS = [
    "Eval/expert_bias",
    "ege_expert_action_fraction",
    "ege_value_gap",
    "policy/altitude_error",
    "temperature/alpha",
]

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
# Registry helpers — link run_id → exp_name without W&B
# ---------------------------------------------------------------------------


def load_registry(registry_file: Path = RUN_REGISTRY_FILE) -> dict[str, dict]:
    """
    Load run_registry.json and return a dict keyed by run_id.

    Each value is the registry entry: {exp_name, wandb_group, tb_path, config}.
    Returns an empty dict if the file does not exist or is malformed.
    """
    if not registry_file.exists():
        return {}
    try:
        with open(registry_file) as f:
            entries = json.load(f)
        return {e["run_id"]: e for e in entries if "run_id" in e}
    except Exception as exc:
        print(f"[registry] Warning: could not load {registry_file}: {exc}")
        return {}


def apply_registry_labels(df: pd.DataFrame, registry: dict[str, dict]) -> pd.DataFrame:
    """
    Override the exp_name column with the value from the registry for any
    run_id that appears in both.  This takes priority over the CSV value so
    that the correct experiment name is always used regardless of what the
    CSV contains.
    """
    if not registry:
        return df
    df = df.copy()
    mapping = {rid: entry["exp_name"] for rid, entry in registry.items()}
    mask = df["run_id"].isin(mapping)
    df.loc[mask, "exp_name"] = df.loc[mask, "run_id"].map(mapping)
    return df


# ---------------------------------------------------------------------------
# TensorBoard event-file reader
# ---------------------------------------------------------------------------


def _read_scalar_from_event(data: bytes, metric: str) -> list[tuple[int, float]]:
    """
    Parse one TFRecord event and return (step, value) pairs for the requested tag.
    Handles both simple_value and single-element float tensor formats.
    """
    from tensorboardX.proto import event_pb2  # type: ignore

    e = event_pb2.Event()
    e.ParseFromString(data)
    results = []
    for v in e.summary.value:
        if v.tag != metric:
            continue
        # simple_value (legacy scalar summary)
        if v.simple_value != 0.0 or v.HasField("simple_value"):
            results.append((e.step, float(v.simple_value)))
        elif v.HasField("tensor") and v.tensor.tensor_content:
            # float32 tensor scalar
            val = struct.unpack("f", v.tensor.tensor_content[:4])[0]
            results.append((e.step, float(val)))
        elif v.HasField("tensor") and v.tensor.float_val:
            results.append((e.step, float(v.tensor.float_val[0])))
    return results


def load_tb_data(
    registry: dict[str, dict],
    tb_dir: Path,
    metric: str,
) -> pd.DataFrame:
    """
    Read TensorBoard event files for all runs in *registry* and return a
    DataFrame with columns [run_id, exp_name, step, <metric>].

    registry : dict  run_id → entry (must have "exp_name" key)
    tb_dir   : root directory; each run lives in tb_dir/<run_id>/
    metric   : TensorBoard tag to extract, e.g. "Eval/expert_bias"
    """
    rows = []
    missing = 0
    for run_id, entry in registry.items():
        run_dir = tb_dir / run_id
        if not run_dir.exists():
            missing += 1
            continue

        exp_name = entry.get("exp_name", "unknown")
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue

        for event_file in event_files:
            try:
                with open(event_file, "rb") as f:
                    while True:
                        hdr = f.read(12)
                        if len(hdr) < 12:
                            break
                        length = struct.unpack("Q", hdr[:8])[0]
                        data = f.read(length)
                        f.read(4)  # footer CRC
                        for step, val in _read_scalar_from_event(data, metric):
                            rows.append({
                                "run_id": run_id,
                                "exp_name": exp_name,
                                "step": step,
                                metric: val,
                            })
            except Exception as exc:
                warnings.warn(f"Could not read {event_file}: {exc}")

    if missing:
        print(f"  Warning: {missing} registered run_ids not found in {tb_dir}")
    if not rows:
        return pd.DataFrame(columns=["run_id", "exp_name", "step", metric])
    return pd.DataFrame(rows)


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


def _iqm(values: np.ndarray) -> np.ndarray:
    """Inter-quartile mean along the seeds axis (axis=1). Returns shape (n_steps,)."""
    n = values.shape[1]
    lo_idx = int(np.floor(0.25 * n))
    hi_idx = int(np.ceil(0.75 * n))
    return np.sort(values, axis=1)[:, lo_idx:hi_idx].mean(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95,
    rng: np.random.Generator | None = None,
    aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap confidence interval of the mean (or IQM) for a 2-D array.

    Parameters
    ----------
    values : (n_steps, n_seeds) array
    n_bootstrap : number of bootstrap resamples
    ci : confidence level in percent (e.g. 95 → 2.5 % – 97.5 % interval)
    aggregation : "mean" or "iqm"

    Returns
    -------
    lo, hi : arrays of shape (n_steps,)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    alpha = (100 - ci) / 2
    n_seeds = values.shape[1]
    # (n_bootstrap, n_seeds) → values[:, idx]: (n_steps, n_bootstrap, n_seeds)
    idx = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    boot_samples = values[:, idx]   # (n_steps, n_bootstrap, n_seeds)
    if aggregation == "iqm":
        n = n_seeds
        lo_q = int(np.floor(0.25 * n))
        hi_q = int(np.ceil(0.75 * n))
        boot_agg = np.sort(boot_samples, axis=2)[:, :, lo_q:hi_q].mean(axis=2)
    elif aggregation == "median":
        boot_agg = np.median(boot_samples, axis=2)
    else:
        boot_agg = boot_samples.mean(axis=2)
    lo = np.percentile(boot_agg, alpha, axis=1)
    hi = np.percentile(boot_agg, 100 - alpha, axis=1)
    return lo, hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour palette — DeepMind-inspired: rich, distinct, print-friendly
PALETTE = [
    "#2166AC",  # deep blue
    "#F4762A",  # warm orange
    "#1A9B6C",  # teal green
    "#9B59B6",  # violet
    "#C0392B",  # deep red
    "#17A8C4",  # cyan
    "#E8A820",  # golden amber
    "#5D6D7E",  # slate grey
    "#7DCE82",  # soft mint
    "#E84393",  # magenta
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


def first_permanent_crossing(steps: np.ndarray, mean: np.ndarray, threshold: float = 0.0) -> int | None:
    """Return the step at which `mean` crosses `threshold` and never falls back below.

    "Never goes back below" means: the last index where mean < threshold, then
    the crossing is the step immediately after. Returns None if the mean never
    crosses or is always below.
    """
    below = np.where(mean < threshold)[0]
    if len(below) == 0:
        # Always above — report first step
        return int(steps[0])
    last_below = below[-1]
    if last_below == len(steps) - 1:
        return None  # never permanently crosses
    return int(steps[last_below + 1])


def _display_name(raw: str) -> str:
    """Map a raw exp_name label to a short display name for figures."""
    is_utd4 = raw.endswith("_UTD4")
    base = raw[:-5] if is_utd4 else raw
    name = EXP_DISPLAY_NAMES.get(base, base)
    return f"{name} (UTD=4)" if is_utd4 else name


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    smooth: int = 1,
    figsize: tuple = (12, 6),
    labels: list[str] | None = None,
    ylim_bottom: float | None = None,
    n_bootstrap: int = 1000,
    ci: float = 95,
    show_crossing: bool = False,
    color_map: dict | None = None,
    aggregation: str = "mean",
) -> plt.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in CSV. Available: {[c for c in df.columns if '/' in c]}")

    all_labels = list(labels if labels is not None else df["label"].unique())

    # Pre-compute aggregate curves for sorting and annotations
    def _mean_curve(label):
        group = df[df["label"] == label].copy()
        per_seed = (
            group.groupby(["run_id", "step"])[metric]
            .mean()
            .unstack("run_id")
            .sort_index()
        )
        steps = per_seed.index.values
        values = per_seed.ffill().bfill().values
        if aggregation == "iqm":
            agg = _iqm(values)
        elif aggregation == "median":
            agg = np.median(values, axis=1)
        else:
            agg = values.mean(axis=1)
        if smooth > 1:
            agg = smooth_series(pd.Series(agg), smooth).values
        return steps, values, agg

    mean_cache = {lbl: _mean_curve(lbl) for lbl in all_labels}

    def fmt_step(s):
        return f"{s/1e6:.2f}M" if s >= 1e6 else f"{int(s/1e3)}k"

    # Sort by TTY0 (ascending); "never" goes last
    def _tty0(lbl):
        steps, _, mean = mean_cache[lbl]
        cx = first_permanent_crossing(steps, mean, threshold=0.0)
        return cx if cx is not None else float("inf")

    if show_crossing:
        all_labels = sorted(all_labels, key=_tty0)
    else:
        all_labels = sorted(all_labels)

    # Pre-compute crossing stats for table-aligned labels
    bold_labels: set = set()
    if show_crossing:
        crossing_stats = {}
        raw_cx0:   dict = {}
        raw_cx100: dict = {}
        raw_max:   dict = {}
        for lbl in all_labels:
            steps_l, _, mean_l = mean_cache[lbl]
            cx0   = first_permanent_crossing(steps_l, mean_l, 0.0)
            cx100 = first_permanent_crossing(steps_l, mean_l, 100.0)
            max_v = float(mean_l.max())
            raw_cx0[lbl]   = cx0    # None means "never"
            raw_cx100[lbl] = cx100
            raw_max[lbl]   = max_v
            crossing_stats[lbl] = (
                fmt_step(cx0)   if cx0   is not None else "never",
                fmt_step(cx100) if cx100 is not None else "never",
                f"{max_v:.0f}",
            )
        col_name  = max(len(_display_name(l)) for l in all_labels)
        col_cx0   = max(len(v[0]) for v in crossing_stats.values())
        col_cx100 = max(len(v[1]) for v in crossing_stats.values())
        col_max   = max(len(v[2]) for v in crossing_stats.values())

        # Per-column winners for selective bold
        cx0_finite   = {l: v for l, v in raw_cx0.items()   if v is not None}
        cx100_finite = {l: v for l, v in raw_cx100.items() if v is not None}
        win_cx0   = min(cx0_finite,   key=cx0_finite.get)   if cx0_finite   else None
        win_cx100 = min(cx100_finite, key=cx100_finite.get) if cx100_finite else None
        win_max   = max(raw_max,      key=raw_max.get)       if raw_max     else None
        bold_labels = {l for l in [win_cx0, win_cx100, win_max] if l is not None}
    else:
        win_cx0 = win_cx100 = win_max = None

    # Assign colors by base name (strip _UTD4 suffix so pairs share a color).
    # An external color_map can be passed to guarantee consistency across figures.
    base_names = sorted({lbl.replace("_UTD4", "") for lbl in all_labels})
    base_color = (color_map if color_map is not None
                  else {name: PALETTE[i % len(PALETTE)]
                        for i, name in enumerate(base_names)})

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)
    n_seeds_common: int | None = None

    for rank, label in enumerate(all_labels):
        is_utd4 = label.endswith("_UTD4")
        base = label.replace("_UTD4", "")
        color = base_color[base]
        linestyle = "--" if is_utd4 else "-"

        steps, values, mean = mean_cache[label]
        n_seeds_common = values.shape[1]
        if values.shape[1] > 1:
            ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci,
                                        rng=rng, aggregation=aggregation)
            if smooth > 1:
                ci_lo = smooth_series(pd.Series(ci_lo), smooth).values
                ci_hi = smooth_series(pd.Series(ci_hi), smooth).values
        else:
            ci_lo, ci_hi = mean.copy(), mean.copy()

        display = _display_name(label)
        if show_crossing:
            cx0_str, cx100_str, max_str = crossing_stats[label]
            plot_label = (
                f"{rank+1:2d}.  {display:<{col_name}}  "
                f"×0: {cx0_str:>{col_cx0}}   "
                f"×100: {cx100_str:>{col_cx100}}   "
                f"max: {max_str:>{col_max}}"
                "    "  # trailing pad so bold text never clips the box edge
            )
        else:
            plot_label = display

        ax.plot(steps, mean, label=plot_label, color=color, linewidth=1.8, linestyle=linestyle)
        ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)

    y_label = METRIC_AXIS_LABELS.get(metric, metric)
    title   = METRIC_TITLES.get(metric, metric)
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    agg_label = {"iqm": "IQM", "median": "median"}.get(aggregation, "mean")
    ax.set_title(f"{title}  ({agg_label} ± {ci:.0f}% bootstrap CI)", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"))
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)

    # Legend: seeds header (no separator row — underline drawn post-layout)
    handles, leg_labels = ax.get_legend_handles_labels()
    hdr = _Line2D([], [])
    rank_note = "  ·  ranked by time to y=0" if show_crossing else ""
    n_str = f"{n_seeds_common} seeds per curve{rank_note}" if n_seeds_common else ""
    # Centre-pad header in monospace so it sits roughly in the middle
    if leg_labels:
        pad = max(0, (len(leg_labels[0]) - len(n_str)) // 2)
        n_str = " " * pad + n_str
    from matplotlib.font_manager import FontProperties
    mono_prop = FontProperties(family="monospace", size=8)
    leg = ax.legend(
        [hdr] + handles,
        [n_str] + leg_labels,
        handler_map={hdr: _InvisibleHandler()},
        prop=mono_prop, ncol=1, loc="best",
    )

    # y=0 reference line and "Policy > / < Expert" annotations —
    # only meaningful for metrics that cross zero (expert_bias).
    is_bias_metric = metric == "Eval/expert_bias"
    ax.axhline(0, color="#999999", linewidth=1.1, linestyle="-", alpha=0.9)
    if is_bias_metric:
        from matplotlib.transforms import blended_transform_factory
        blend = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.97, 80,  "Policy > Expert", transform=blend,
                rotation=90, ha="center", va="bottom", fontsize=7.5, color="#777777")
        ax.text(0.97, -80, "Policy < Expert", transform=blend,
                rotation=90, ha="center", va="top",    fontsize=7.5, color="#777777")

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Bold winning rows — use set_fontweight directly on the legend Text objects.
    # No overlay: labels stay inside the legend box at any DPI / bbox setting.
    if show_crossing:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_texts = leg.get_texts()
        leg_texts[0].set_color("#555555")  # mute header
        for i, lbl in enumerate(all_labels):
            if lbl in bold_labels:
                leg_texts[i + 1].set_fontweight("bold")

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

    n = len(labels_sorted)
    # Tighter column spacing: fixed 0.4-inch per column, minimum figure width 4 inches
    fig_w = max(4.0, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, figsize[1]))
    positions = np.arange(n) * 0.5  # 0.5 unit spacing instead of 1.0

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
            markersize=8,
            capsize=6,
            linewidth=2.0,
            zorder=5,
            label=f"{_display_name(label)}  (mean={mean:.1f})",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_display_name(l) for l in labels_sorted], rotation=30, ha="right", fontsize=9)
    ax.set_xlim(positions[0] - 0.3, positions[-1] + 0.3)
    # Let matplotlib autoscale the y-axis; add a small margin
    ax.margins(y=0.15)
    y_label = METRIC_AXIS_LABELS.get(metric, metric)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"{METRIC_TITLES.get(metric, metric)} — last {last_n_steps/1e3:.0f}k steps"
        f"  (◆ = mean,  bars = {ci:.0f}% bootstrap CI)",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def plot_final_distribution(
    df: pd.DataFrame,
    metric: str,
    last_n_steps: int = 100_000,
    figsize: tuple = (10, 5),
    color_map: dict | None = None,
) -> plt.Figure:
    """
    Histogram of per-seed final performance for each experiment.

    For each seed the 'final value' is the mean of the last `last_n_steps` steps.
    Histograms are overlaid with semi-transparency; a KDE curve is drawn on top.
    """
    from scipy.stats import gaussian_kde  # type: ignore

    cutoff = df["step"].max() - last_n_steps
    labels = sorted(df["label"].unique())

    base_names = sorted({l.replace("_UTD4", "") for l in labels})
    base_color = (color_map if color_map is not None
                  else {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(base_names)})

    # Collect per-seed final means
    seed_finals: dict[str, np.ndarray] = {}
    for lbl in labels:
        grp = df[(df["label"] == lbl) & (df["step"] >= cutoff)]
        per_seed = grp.groupby("run_id")[metric].mean()
        seed_finals[lbl] = per_seed.values

    # Shared x range across all methods
    all_vals = np.concatenate(list(seed_finals.values()))
    x_min, x_max = all_vals.min(), all_vals.max()
    pad = (x_max - x_min) * 0.08
    x_grid = np.linspace(x_min - pad, x_max + pad, 400)

    fig, ax = plt.subplots(figsize=figsize)

    n_bins = max(20, int(2.5 * np.sqrt(max(len(v) for v in seed_finals.values()))))

    for lbl in labels:
        vals = seed_finals[lbl]
        color = base_color[lbl.replace("_UTD4", "")]
        display = _display_name(lbl)
        linestyle = "--" if lbl.endswith("_UTD4") else "-"

        ax.hist(vals, bins=n_bins, range=(x_min - pad, x_max + pad),
                color=color, alpha=0.25, density=True)

        if len(vals) > 2:
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2,
                    linestyle=linestyle, label=f"{display}  (n={len(vals)})")

    ax.axvline(0, color="#999999", linewidth=1.1, linestyle="-", alpha=0.9)

    y_top = ax.get_ylim()[1]
    ax.text(0, y_top * 0.97, "  Expert", color="#999999", fontsize=8,
            va="top", ha="left")

    y_label = METRIC_AXIS_LABELS.get(metric, metric)
    title   = METRIC_TITLES.get(metric, metric)
    ax.set_xlabel(f"Final {y_label}  (mean over last {last_n_steps/1e3:.0f}k steps)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{title} — seed distribution at end of training", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot ablation curves from TensorBoard or CSV.")
    parser.add_argument("--csv", default=None, help="CSV file (overrides TensorBoard mode)")
    parser.add_argument("--registry", default=str(ABLATION_REGISTRY_FILE),
                        help="Registry JSON (default: ablation_run_registry.json)")
    parser.add_argument("--project", default=None,
                        help="Filter to runs from this project (e.g. ablation_plane_final_2)")
    parser.add_argument("--tb-dir", default=str(TB_DIR), help="TensorBoard root directory")
    parser.add_argument("--metric", default="Eval/expert_bias")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling-average window (steps)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CI")
    parser.add_argument("--ci", type=float, default=95, help="Confidence level in percent (default 95)")
    parser.add_argument("--top", type=int, default=5, help="Number of best runs to plot")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    n = args.top

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tb_dir = Path(args.tb_dir)
    registry_file = Path(args.registry)

    # ------------------------------------------------------------------
    # Data loading: TensorBoard mode (default) or CSV fallback
    # ------------------------------------------------------------------
    if args.csv is None and registry_file.exists():
        registry = load_registry(registry_file)
        if args.project:
            before = len(registry)
            registry = {rid: e for rid, e in registry.items()
                        if e.get("project") == args.project}
            print(f"Project filter '{args.project}': {before} → {len(registry)} runs")
        print(f"Loading from TensorBoard ({registry_file}, {len(registry)} runs)...")
        df = load_tb_data(registry, tb_dir, args.metric)
        print(f"  {len(df)} scalar points, {df['run_id'].nunique()} runs, "
              f"{df['exp_name'].nunique()} experiments")
        if df.empty:
            print(f"  No data found for metric '{args.metric}'. Check tag name.")
            return
    else:
        csv_path = args.csv or CSV_FILE
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  {len(df)} rows, {df['run_id'].nunique()} runs")

        # Apply run_registry labels on top of CSV (legacy path)
        registry = load_registry()
        if registry:
            print(f"  Registry: {len(registry)} runs indexed in {RUN_REGISTRY_FILE}")
            df = apply_registry_labels(df, registry)

    df = resolve_exp_name(df)

    print("\nDeduplicating runs (keeping latest batch per label)...")
    df = deduplicate_runs(df)

    print(f"\nGroups ({df['label'].nunique()}):")
    for label, grp in df.groupby("label"):
        n_seeds = grp["run_id"].nunique()
        print(f"  {label:50s}  {n_seeds:3d} seeds")

    metric_slug = args.metric.replace("/", "_")
    smooth_tag = f"_smooth{args.smooth}" if args.smooth > 1 else ""

    # For Eval/expert_bias, restrict to the canonical set of experiments.
    plot_df = df
    if args.metric == "Eval/expert_bias":
        plot_df = df[df["label"].isin(EXPERT_BIAS_EXPERIMENTS)]
        dropped = set(df["label"].unique()) - set(plot_df["label"].unique())
        if dropped:
            print(f"  expert_bias filter: dropped {dropped}")

    print(f"\nPlotting {args.metric} — all groups...")
    fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth, n_bootstrap=args.n_bootstrap, ci=args.ci)
    out_path = out_dir / f"{metric_slug}{smooth_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    print(f"\nPlotting {args.metric} — all groups (focus ≥ -1000)...")
    show_cx = (args.metric == "Eval/expert_bias")
    fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth, n_bootstrap=args.n_bootstrap, ci=args.ci, ylim_bottom=-1000, show_crossing=show_cx)
    out_path = out_dir / f"{metric_slug}{smooth_tag}_focus.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    print(f"\nPlotting {args.metric} — top {n} by asymptotic return (last 200k steps)...")
    topn = top_labels_by_asymptote(plot_df, metric=args.metric, last_n_steps=200_000, n=n)
    print(f"  Top {n}: {topn}")
    fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth, labels=topn, ylim_bottom=-1000, n_bootstrap=args.n_bootstrap, ci=args.ci)
    out_path = out_dir / f"{metric_slug}{smooth_tag}_top{n}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    # IQM and median variants — only for Eval/expert_bias.
    if args.metric == "Eval/expert_bias":
        for agg in ("iqm", "median"):
            print(f"\nPlotting {args.metric} — {agg.upper()}, all groups...")
            fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth,
                              n_bootstrap=args.n_bootstrap, ci=args.ci,
                              aggregation=agg)
            out_path = out_dir / f"{metric_slug}{smooth_tag}_{agg}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
            plt.close(fig)

            print(f"\nPlotting {args.metric} — {agg.upper()}, focus ≥ -1000...")
            fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth,
                              n_bootstrap=args.n_bootstrap, ci=args.ci,
                              ylim_bottom=-1000, show_crossing=True,
                              aggregation=agg)
            out_path = out_dir / f"{metric_slug}{smooth_tag}_{agg}_focus.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
            plt.close(fig)

    if args.metric == "Eval/expert_bias":
        print(f"\nPlotting {args.metric} — final distribution histogram...")
        fig = plot_final_distribution(plot_df, metric=args.metric, last_n_steps=100_000)
        out_path = out_dir / f"{metric_slug}{smooth_tag}_dist.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
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

    # Summary plots — one figure per metric, shared colour map for consistency.
    # Only available in TB mode (requires registry + tb_dir).
    if args.csv is None and registry_file.exists():
        print("\nPlotting summary (5 key metrics, all experiments)...")

        # Build a shared colour map from all labels across all summary metrics
        # so that the same experiment always gets the same colour.
        all_summary_labels: set[str] = set()
        summary_dfs: dict[str, pd.DataFrame] = {}
        for s_metric in SUMMARY_METRICS:
            df_s = load_tb_data(registry, tb_dir, s_metric)
            if df_s.empty:
                continue
            df_s = resolve_exp_name(df_s)
            df_s = deduplicate_runs(df_s)
            summary_dfs[s_metric] = df_s
            all_summary_labels |= set(df_s["label"].unique())

        s_base_names = sorted({l.replace("_UTD4", "") for l in all_summary_labels})
        shared_color_map = {
            name: PALETTE[i % len(PALETTE)]
            for i, name in enumerate(s_base_names)
        }

        for s_metric, df_s in summary_dfs.items():
            slug = s_metric.replace("/", "_")
            print(f"  {s_metric}...")
            fig = plot_metric(
                df_s, metric=s_metric,
                smooth=args.smooth,
                n_bootstrap=args.n_bootstrap,
                ci=args.ci,
                color_map=shared_color_map,
            )
            out_path = out_dir / f"summary_{slug}{smooth_tag}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved → {out_path}")


if __name__ == "__main__":
    main()
