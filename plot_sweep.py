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
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
NOISY_EXPERT_REGISTRY_FILE = Path("noisy_expert_run_registry.json")

# ---------------------------------------------------------------------------
# Main result — change this one line to switch which experiment is "main"
# across all plots (curves, summaries, ablation anchors, expert-bias overview).
# ---------------------------------------------------------------------------
MAIN_EXP = "ege_decay_050"

# ---------------------------------------------------------------------------
# Optimality gap (Agarwal et al. 2021) — derived from Eval/expert_bias.
#   score       = expert_bias / OPTIMALITY_GAP_MAX   (0 = expert level)
#   optimality_gap = max(1 - score, 0)
# The "max theoretical performance" is OPTIMALITY_GAP_MAX units above expert.
# ---------------------------------------------------------------------------
OPTIMALITY_GAP_METRIC = "Eval/optimality_gap"
OPTIMALITY_GAP_MAX    = 10_000

# Metrics where lower values are better (affects crossing-panel direction).
METRIC_LOWER_IS_BETTER: frozenset[str] = frozenset({OPTIMALITY_GAP_METRIC})

# Default crossing thresholds per metric (used by summary/stats functions).
METRIC_THRESHOLDS: dict[str, list[float]] = {
    "Eval/expert_bias": [0.0, 200.0, 400.0],
    OPTIMALITY_GAP_METRIC: [1.0, 0.98, 0.96],
}

# ---------------------------------------------------------------------------
# Display names — maps raw exp_name to short paper-ready label.
# UTD4 variants are handled automatically by appending " (UTD=4)".
# ---------------------------------------------------------------------------
EXP_DISPLAY_NAMES: dict[str, str] = {
    "sac_baseline":        "SAC (baseline)",
    "residual_rl":         "Residual RL",
    # Main result — canonical reference across all ablation questions
    "ege_simple":          "EGE  ε=0.5  decay=0.15",
    # Decay horizon sweep
    "ege_decay_005":       "EGE  decay=0.05",
    "ege_decay_050":       "EGE  decay=0.50  (main)",
    "ege_no_decay":        "EGE  no decay  (always on)",
    # Gating style
    "ibrl_style":          "IBRL-style  argmax  (no decay)",
    "ibrl_style_decay":    "IBRL-style  argmax  decay=0.50",
    # Decay horizon sweep
    "ege_decay_075":       "EGE  decay=0.75",
    # Epsilon sweep (ε=0.5 anchor provided by ege_simple above)
    "ege_eps_0.1":         "EGE  ε=0.10",
    "ege_eps_0.25":        "EGE  ε=0.25",
    "ege_eps_0.75":        "EGE  ε=0.75",
    "ege_eps_0.9":         "EGE  ε=0.90",
    "ege_eps_0.95":        "EGE  ε=0.95",
    "ege_eps_0.99":        "EGE  ε=0.99",
    # Noisy expert degradation study — EGE
    "ege_noise_0pct":      "0%   (PID, reference)",
    "ege_noise_2pct":      "2%   (σ=100m)",
    "ege_noise_5pct":      "5%   (σ=250m)",
    "ege_noise_10pct":     "10%  (σ=500m)",
    "ege_noise_20pct":     "20%  (σ=1000m)",
    "ege_noise_40pct":     "40%  (σ=2000m)",
    "ege_noise_80pct":     "80%  (σ=4000m)",
    # Noisy expert degradation study — IBRL
    "ibrl_noise_0pct":     "0%   (PID, reference)",
    "ibrl_noise_2pct":     "2%   (σ=100m)",
    "ibrl_noise_5pct":     "5%   (σ=250m)",
    "ibrl_noise_10pct":    "10%  (σ=500m)",
    "ibrl_noise_20pct":    "20%  (σ=1000m)",
    "ibrl_noise_40pct":    "40%  (σ=2000m)",
    "ibrl_noise_80pct":    "80%  (σ=4000m)",
    # Noisy expert degradation study — Residual RL
    "residual_noise_0pct":  "0%   (PID, reference)",
    "residual_noise_2pct":  "2%   (σ=100m)",
    "residual_noise_5pct":  "5%   (σ=250m)",
    "residual_noise_10pct": "10%  (σ=500m)",
    "residual_noise_20pct": "20%  (σ=1000m)",
    "residual_noise_40pct": "40%  (σ=2000m)",
    "residual_noise_80pct": "80%  (σ=4000m)",
}

# ---------------------------------------------------------------------------
# Ablation question definitions — one figure per question.
# Each question compares a fixed set of experiments on a shared metric.
# ---------------------------------------------------------------------------

# Fixed colour map so the same run gets the same colour across all question
# figures. sac_baseline (grey) and MAIN_EXP (blue) are always anchors.
ABLATION_COLOR_MAP: dict[str, str] = {
    "sac_baseline":      "#5D6D7E",  # slate grey   — lower bound
    "residual_rl":       "#27AE60",  # green        — residual RL baseline
    "ege_simple":        "#9B59B6",  # violet       — decay=0.15
    "ege_decay_005":     "#7DCE82",  # soft mint    — shortest decay
    "ege_decay_050":     "#2166AC",  # deep blue    — main result
    "ege_decay_075":     "#1A5276",  # dark navy    — decay=0.75
    "ege_no_decay":      "#C0392B",  # deep red     — no decay (always on)
    "ibrl_style":        "#F4762A",  # warm orange  — argmax, no decay
    "ibrl_style_decay":  "#E8A820",  # golden amber — argmax, decay=0.50
    "ege_eps_0.1":       "#17A8C4",  # cyan
    "ege_eps_0.25":      "#1A9B6C",  # teal green
    "ege_eps_0.75":      "#9B59B6",  # violet
    "ege_eps_0.9":       "#E84393",  # magenta
    "ege_eps_0.95":      "#C0392B",  # deep red
    "ege_eps_0.99":      "#784212",  # brown
}

ABLATION_QUESTIONS: list[dict] = [
    {
        "key": "q1_decay_horizon",
        "title": "Q1: What is the right exploration decay horizon?",
        "subtitle": f"decay_frac ∈ {{0.0 (always on), 0.05, 0.15, 0.50 (main), 0.75}}",
        "experiments": ["sac_baseline", "ege_no_decay", "ege_decay_005", "ege_simple", "ege_decay_050", "ege_decay_075"],
        # No group filter: accept runs from any group for these experiments.
        "allowed_groups": None,
    },
    {
        "key": "q2_epsilon_sensitivity",
        "title": "Q2: How sensitive is EGE to the fixed-epsilon value?",
        "subtitle": "ε ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}  (best decay from Q1)",
        "experiments": ["sac_baseline", "ege_eps_0.1", "ege_eps_0.25", MAIN_EXP, "ege_eps_0.75", "ege_eps_0.9", "ege_eps_0.95", "ege_eps_0.99"],
        # Only accept eps runs from the dedicated Q2 group (decay=0.50).
        # sac_baseline and MAIN_EXP come from baselines group.
        "allowed_groups": {"ablation_baselines_plane", "ablation_q2_epsilon_plane"},
    },
    {
        "key": "q3_gating_style",
        "title": "Q3: Does gating style matter? Epsilon-greedy vs IBRL-style argmax",
        "subtitle": "Stochastic ε-greedy vs deterministic argmax gate (best decay from Q1)",
        "experiments": ["sac_baseline", MAIN_EXP, "ibrl_style", "ibrl_style_decay"],
        # ibrl_style_decay (decay=0.50) lives in q3_gating_plane; exclude q1_value_gap_gate_plane (decay=0.15).
        "allowed_groups": {"ablation_baselines_plane", "ablation_q3_gating_plane"},
    },
]

# ---------------------------------------------------------------------------
# Noisy expert degradation study
# ---------------------------------------------------------------------------

# Sequential blue→red palette: clean PID (blue) degrades to near-random (dark red).
# Shared across all three algorithms — same noise level always gets the same colour.
_NOISE_LEVEL_COLORS: dict[int, str] = {
    0:  "#2166AC",  # deep blue    — perfect PID
    2:  "#4393C3",  # medium blue
    5:  "#74ADD1",  # light blue
    10: "#F4A736",  # amber
    20: "#F46D43",  # orange-red
    40: "#D73027",  # deep red
    80: "#67001F",  # dark maroon
}
_NOISE_LEVELS_PCT = [0, 2, 5, 10, 20, 40, 80]

NOISY_EXPERT_COLOR_MAP: dict[str, str] = {
    **{f"ege_noise_{p}pct":      _NOISE_LEVEL_COLORS[p] for p in _NOISE_LEVELS_PCT},
    **{f"ibrl_noise_{p}pct":     _NOISE_LEVEL_COLORS[p] for p in _NOISE_LEVELS_PCT},
    **{f"residual_noise_{p}pct": _NOISE_LEVEL_COLORS[p] for p in _NOISE_LEVELS_PCT},
}

# Ordered from best to worst expert — also used as the x-axis of the degradation curve.
NOISY_EXPERT_EXPERIMENTS: list[str] = [
    "ege_noise_0pct",
    "ege_noise_2pct",
    "ege_noise_5pct",
    "ege_noise_10pct",
    "ege_noise_20pct",
    "ege_noise_40pct",
    "ege_noise_80pct",
]
NOISY_EXPERT_NOISE_PCTS: list[float] = [0, 2, 5, 10, 20, 40, 80]

IBRL_NOISY_EXPERIMENTS: list[str] = [f"ibrl_noise_{p}pct" for p in [0, 2, 5, 10, 20, 40, 80]]
RESIDUAL_NOISY_EXPERIMENTS: list[str] = [f"residual_noise_{p}pct" for p in [0, 2, 5, 10, 20, 40, 80]]

# Maps each 0pct reference name to its equivalent in ablation_run_registry.json
# (so we avoid re-running an identical experiment).
_ABLATION_EQUIV_0PCT: dict[str, str] = {
    "ege_noise_0pct":      "ege_decay_050",
    "ibrl_noise_0pct":     "ibrl_style",
    "residual_noise_0pct": "residual_rl",
}

# Metric axis labels and titles
METRIC_AXIS_LABELS: dict[str, str] = {
    "Eval/expert_bias":              "Expert Advantage",
    OPTIMALITY_GAP_METRIC:           "Optimality Gap",
    "Eval/episodic_mean_reward":     "Mean Episode Return",
    "ege_expert_action_fraction":    "EGE Expert Action Fraction",
    "ege_value_gap":                 "EGE Value Gap",
    "policy/altitude_error":         "Altitude Error",
    "temperature/alpha":             "Alpha (Temperature)",
}
METRIC_TITLES: dict[str, str] = {
    "Eval/expert_bias":              "Expert Advantage over Training",
    OPTIMALITY_GAP_METRIC:           "Optimality Gap over Training",
    "Eval/episodic_mean_reward":     "Episode Return over Training",
    "ege_expert_action_fraction":    "EGE Expert Action Fraction",
    "ege_value_gap":                 "EGE Value Gap",
    "policy/altitude_error":         "Altitude Error",
    "temperature/alpha":             "Alpha (Temperature)",
}

# Experiments shown in the top-level Eval/expert_bias overview plot.
EXPERT_BIAS_EXPERIMENTS = {
    "sac_baseline",
    "residual_rl",
    MAIN_EXP,       # canonical main result
    "ibrl_style",   # IBRL no decay
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


_CACHE_DIR = Path(".plot_cache")


def _tb_cache_path(registry: dict[str, dict], metric: str) -> Path:
    """Stable cache filename derived from the registry's run_ids and the metric."""
    import hashlib
    key = metric + "\n" + "\n".join(sorted(registry.keys()))
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    slug = metric.replace("/", "_")
    return _CACHE_DIR / f"{slug}_{digest}.parquet"


def _cache_is_fresh(cache_path: Path, registry: dict[str, dict], tb_dir: Path) -> bool:
    """Return True if the cache is newer than every event file it covers."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for run_id in registry:
        run_dir = tb_dir / run_id
        if not run_dir.exists():
            continue
        for ef in run_dir.glob("events.out.tfevents.*"):
            if ef.stat().st_mtime > cache_mtime:
                return False
    return True


def load_tb_data(
    registry: dict[str, dict],
    tb_dir: Path,
    metric: str,
    no_cache: bool = False,
) -> pd.DataFrame:
    """
    Read TensorBoard event files for all runs in *registry* and return a
    DataFrame with columns [run_id, exp_name, step, <metric>].

    Results are cached to parquet in _CACHE_DIR and reused on subsequent calls
    as long as no event file is newer than the cache.  Pass no_cache=True to
    force a full re-read.

    registry : dict  run_id → entry (must have "exp_name" key)
    tb_dir   : root directory; each run lives in tb_dir/<run_id>/
    metric   : TensorBoard tag to extract, e.g. "Eval/expert_bias"
    """
    cache_path = _tb_cache_path(registry, metric)

    if not no_cache and _cache_is_fresh(cache_path, registry, tb_dir):
        print(f"  Cache hit → {cache_path}")
        return pd.read_parquet(cache_path)

    rows = []
    missing = 0
    for run_id, entry in registry.items():
        run_dir = tb_dir / run_id
        if not run_dir.exists():
            missing += 1
            continue

        exp_name = entry.get("exp_name", "unknown")
        group = entry.get("group", "")
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
                                "group": group,
                                "step": step,
                                metric: val,
                            })
            except Exception as exc:
                warnings.warn(f"Could not read {event_file}: {exc}")

    if missing:
        print(f"  Warning: {missing} registered run_ids not found in {tb_dir}")
    if not rows:
        return pd.DataFrame(columns=["run_id", "exp_name", "group", "step", metric])

    df = pd.DataFrame(rows)

    if not no_cache:
        _CACHE_DIR.mkdir(exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  Cache saved → {cache_path}")

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
    """Return the n labels with the highest IQM metric over the last last_n_steps."""
    cutoff = df["step"].max() - last_n_steps
    tail = df[df["step"] >= cutoff]
    ranking = (
        tail.groupby("label")[metric]
        .apply(_scalar_iqm)
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
    linestyle_map: dict | None = None,
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
        if linestyle_map and label in linestyle_map:
            linestyle = linestyle_map[label]
        else:
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

        ax.plot(steps, mean, label=plot_label, color=color, linewidth=1.2, linestyle=linestyle, zorder=3 + (len(all_labels) - rank))
        ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.10, zorder=2 + (len(all_labels) - rank))

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

    For each seed the 'final value' is the IQM of the last `last_n_steps` steps.
    Histograms are overlaid with semi-transparency; a KDE curve is drawn on top.
    """
    from scipy.stats import gaussian_kde  # type: ignore

    cutoff = df["step"].max() - last_n_steps
    labels = sorted(df["label"].unique())

    base_names = sorted({l.replace("_UTD4", "") for l in labels})
    base_color = (color_map if color_map is not None
                  else {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(base_names)})

    # Collect per-seed final IQMs
    seed_finals: dict[str, np.ndarray] = {}
    for lbl in labels:
        grp = df[(df["label"] == lbl) & (df["step"] >= cutoff)]
        per_seed = grp.groupby("run_id")[metric].apply(_scalar_iqm)
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
    ax.set_xlabel(f"Final {y_label}  (IQM over last {last_n_steps/1e3:.0f}k steps)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{title} — seed distribution at end of training", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ablation question plots
# ---------------------------------------------------------------------------


def _bootstrap_scalar_iqm_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Bootstrap IQM ± CI for a 1-D array of per-seed scalar values.

    NaN entries (e.g. "never crossed") are excluded before resampling.
    Returns (iqm_center, ci_lo, ci_hi).  All three are NaN if no valid data.
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng(42)
    alpha = (100 - ci) / 2
    n = len(valid)
    lo_idx = int(np.floor(0.25 * n))
    hi_idx = int(np.ceil(0.75 * n))
    center = float(np.sort(valid)[lo_idx:hi_idx].mean())
    boots = rng.choice(valid, size=(n_bootstrap, n), replace=True)
    boot_iqm = np.sort(boots, axis=1)[:, lo_idx:hi_idx].mean(axis=1)
    return center, float(np.percentile(boot_iqm, alpha)), float(np.percentile(boot_iqm, 100 - alpha))


def _per_seed_crossing(
    df: pd.DataFrame,
    metric: str,
    label: str,
    threshold: float,
    censor_at: float | None = None,
    remove_censored: bool = False,
    invert: bool = False,
) -> tuple[np.ndarray, int]:
    """Return (crossing_steps, n_censored).

    crossing_steps: per-seed first-permanent-crossing steps.
      If remove_censored is True, runs that never cross are dropped.
      If censor_at is given, runs that never cross are filled with censor_at
      (RMST-style: cost is the full training budget).  Otherwise they are NaN.
    n_censored: number of runs that never reached the threshold.
    invert: if True, detect when the curve permanently falls *below* threshold
      (for lower-is-better metrics) by negating both curve and threshold.
    """
    group = df[df["label"] == label]
    per_seed = group.groupby(["run_id", "step"])[metric].mean().unstack("run_id").sort_index()
    results = []
    for run_id in per_seed.columns:
        curve = per_seed[run_id].ffill().bfill().values
        if invert:
            cx = first_permanent_crossing(per_seed.index.values, -curve, -threshold)
        else:
            cx = first_permanent_crossing(per_seed.index.values, curve, threshold)
        results.append(float(cx) if cx is not None else float("nan"))
    arr = np.array(results)
    n_censored = int(np.sum(np.isnan(arr)))
    if remove_censored:
        arr = arr[~np.isnan(arr)]
    elif censor_at is not None:
        arr = np.where(np.isnan(arr), censor_at, arr)
    return arr, n_censored


def _per_seed_final_perf(df: pd.DataFrame, metric: str, label: str, last_n_steps: int) -> np.ndarray:
    """Return a 1-D array of per-seed IQM performance over the last last_n_steps."""
    cutoff = df["step"].max() - last_n_steps
    group = df[(df["label"] == label) & (df["step"] >= cutoff)]
    per_seed = group.groupby("run_id")[metric].apply(_scalar_iqm)
    return per_seed.values


def _per_seed_max_perf(df: pd.DataFrame, metric: str, label: str) -> np.ndarray:
    """Return a 1-D array of per-seed maximum value over the full training run."""
    group = df[df["label"] == label]
    per_seed = group.groupby("run_id")[metric].max()
    return per_seed.values


def _per_seed_asymptote(
    df: pd.DataFrame,
    metric: str,
    label: str,
    last_n_steps: int = 200_000,
) -> np.ndarray:
    """Return a 1-D array of per-seed IQM over the last `last_n_steps` steps."""
    group = df[df["label"] == label]
    max_step = group["step"].max()
    tail = group[group["step"] >= max_step - last_n_steps]
    per_seed = tail.groupby("run_id")[metric].apply(_scalar_iqm)
    return per_seed.values


def make_latex_method_table(
    df: pd.DataFrame,
    metric: str,
    experiments: list[str],
    thresholds: list[float] | None = None,
    final_last_n_steps: int = 200_000,
    n_bootstrap: int = 1000,
    ci: float = 95,
    baseline: str = "sac_baseline",
    remove_censored: bool = False,
) -> str:
    """
    Build a LaTeX table matching the method-summary plot panels.

    Columns: one pair (IQM, Δ% vs baseline) per crossing threshold, one pair
    for peak performance, and one pair for mean metric over the last
    final_last_n_steps steps.
    For steps-to-X, negative Δ% means faster (better).
    For performance columns, positive Δ% means higher (better).
    Censored IQM entries are annotated with †.
    """
    if thresholds is None:
        thresholds = METRIC_THRESHOLDS.get(metric, [0.0, 200.0, 400.0])

    present = [e for e in experiments if e in df["label"].unique()]
    censor_at = float(df["step"].max()) if not remove_censored else None
    rng = np.random.default_rng(42)

    def _fmt_steps(s: float) -> str:
        return f"{s/1e6:.2f}M" if s >= 1e6 else f"{int(round(s/1e3))}k"

    def _pct(val: float, ref: float) -> str:
        if np.isnan(val) or np.isnan(ref) or ref == 0:
            return "—"
        return f"{(val - ref) / abs(ref) * 100:+.1f}\\%"

    # crossing_data[lbl][i] = (center, n_censored, n_total)
    # peak_iqm[lbl] = center
    # final_iqm[lbl] = center
    crossing_data: dict[str, list[tuple[float, int, int]]] = {}
    peak_iqm: dict[str, float] = {}
    final_iqm: dict[str, float] = {}

    _invert_crossing = metric in METRIC_LOWER_IS_BETTER
    for lbl in present:
        crossing_data[lbl] = []
        for thr in thresholds:
            vals, n_censored = _per_seed_crossing(
                df, metric, lbl, thr, censor_at=censor_at,
                remove_censored=remove_censored, invert=_invert_crossing,
            )
            center, _, _ = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            crossing_data[lbl].append((center, n_censored, len(vals)))
        peak_vals = _per_seed_max_perf(df, metric, lbl)
        peak_center, _, _ = _bootstrap_scalar_iqm_ci(peak_vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        peak_iqm[lbl] = peak_center
        final_vals = _per_seed_final_perf(df, metric, lbl, final_last_n_steps)
        final_center, _, _ = _bootstrap_scalar_iqm_ci(final_vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        final_iqm[lbl] = final_center

    ref_crossing = [crossing_data[baseline][i][0] if baseline in crossing_data else float("nan")
                    for i in range(len(thresholds))]
    ref_peak = peak_iqm.get(baseline, float("nan"))
    ref_final = final_iqm.get(baseline, float("nan"))

    # ---- Build LaTeX ----
    thr_col_specs = "rr" * len(thresholds)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{l{thr_col_specs}rrrr}}",
        "\\toprule",
    ]

    thr_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{Steps to $y={int(t)}$ (IQM)}}"
        for t in thresholds
    )
    lines.append(
        f"\\textbf{{Method}} & {thr_headers}"
        f" & \\multicolumn{{2}}{{c}}{{\\textbf{{Peak}}}}"
        f" & \\multicolumn{{2}}{{c}}{{\\textbf{{Avg last {final_last_n_steps//1000}k steps}}}} \\\\"
    )

    sub = " & ".join(["IQM & $\\Delta$\\%"] * (len(thresholds) + 2))
    lines.append(f" & {sub} \\\\")
    lines.append("\\midrule")

    for lbl in present:
        display = _display_name(lbl)
        is_baseline = lbl == baseline
        row_parts = [f"\\textbf{{{display}}}" if is_baseline else display]

        for i, thr in enumerate(thresholds):
            center, n_censored, n_total = crossing_data[lbl][i]
            n_reached = n_total - n_censored
            dagger = "$^\\dagger$" if n_censored > 0 else ""
            if np.isnan(center):
                iqm_str, pct_str = "—", "—"
            else:
                iqm_str = f"{_fmt_steps(center)}{dagger} ({n_reached}/{n_total})"
                pct_str = "—" if is_baseline else _pct(center, ref_crossing[i])
            row_parts += [iqm_str, pct_str]

        pc = peak_iqm[lbl]
        row_parts += [
            f"{pc:.1f}" if not np.isnan(pc) else "—",
            "—" if is_baseline else _pct(pc, ref_peak),
        ]

        fc = final_iqm[lbl]
        row_parts += [
            f"{fc:.1f}" if not np.isnan(fc) else "—",
            "—" if is_baseline else _pct(fc, ref_final),
        ]

        lines.append(" & ".join(row_parts) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[4pt]",
        "{\\footnotesize $^\\dagger$ Some runs did not reach this threshold; "
        "censored at training budget for IQM.}",
        f"\\caption{{Expert advantage summary (metric: \\texttt{{{metric}}}).}}",
        "\\label{tab:method_summary}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _shaffer_t_values(k: int) -> list[int]:
    """
    Return the sorted set of all possible numbers of simultaneously true pairwise
    null hypotheses for k groups (Shaffer 1986).

    The key insight: true null hypotheses must form equivalence classes (if A=B and
    B=C then A=C), so the possible true-null counts are exactly the sums of
    triangular numbers over all integer partitions of k.
    """
    possible: list[set[int]] = [set() for _ in range(k + 1)]
    possible[0] = {0}
    possible[1] = {0}
    for n in range(2, k + 1):
        for j in range(1, n + 1):
            pairs_j = j * (j - 1) // 2
            for t in possible[n - j]:
                possible[n].add(pairs_j + t)
    return sorted(possible[k])


def _scalar_iqm(vals: np.ndarray) -> float:
    """IQM of a 1-D array (trims bottom and top 25%)."""
    s = np.sort(vals)
    n = len(s)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(s[lo:hi].mean()) if hi > lo else float(s.mean())


def _print_pairwise_tests(
    seed_vals: dict[str, np.ndarray],
    present: list[str],
    display: dict[str, str],
    title: str,
    alpha: float,
    n_permutations: int,
    rng: np.random.Generator,
    unit: str = "",
    reference: str | None = None,
) -> None:
    """
    Print Tukey HSD (means) and IQM permutation test tables for one set of
    per-seed scalar values.  `unit` is appended to numeric columns (e.g. " steps").

    If `reference` is given, only pairs involving that group are tested.
    Tukey HSD is still fitted on all groups (for the pooled MSE estimate) but
    only reference rows are displayed — this is slightly conservative; Dunnett's
    test would be exact but is not available in statsmodels.
    For the IQM permutation test, comparisons against a single reference carry
    no logical constraints between them, so Holm-Bonferroni is used directly
    (Shaffer reduces to Holm in this case).
    """
    if reference is not None:
        pairs = [(reference, b) for b in present if b != reference]
    else:
        pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]

    # ---- Tukey HSD (mean-based) ----
    values, groups = [], []
    for lbl in present:
        values.extend(seed_vals[lbl].tolist())
        groups.extend([lbl] * len(seed_vals[lbl]))

    tukey = pairwise_tukeyhsd(endog=np.array(values), groups=np.array(groups), alpha=alpha)

    note = " [vs reference only; Tukey pooled MSE over all groups]" if reference else ""
    print("\n" + "=" * 72)
    print(f"Tukey HSD (means) — {title}  (α={alpha}){note}")
    print("=" * 72)
    print(f"{'Group 1':<28} {'Group 2':<28} {'Δ mean':>14} {'p-adj':>8} {'Reject':>8}")
    print("-" * 72)
    for row in tukey.summary().data[1:]:
        g1, g2, meandiff, p_adj, lower, upper, reject = row
        if reference is not None and reference not in (g1, g2):
            continue
        diff_str = f"{meandiff:,.0f}{unit}" if unit else f"{meandiff:.2f}"
        print(
            f"{display.get(g1, g1):<28} {display.get(g2, g2):<28}"
            f" {diff_str:>14} {float(p_adj):>8.4f} {'Yes' if reject else 'No':>8}"
        )
    print("=" * 72)

    # ---- IQM permutation test ----
    # With a single reference, no logical constraints exist between the tested
    # hypotheses, so Shaffer reduces to Holm; we use Holm directly.
    raw_p: list[float] = []
    iqm_diffs: list[float] = []
    for a, b in pairs:
        combined = np.concatenate([seed_vals[a], seed_vals[b]])
        na = len(seed_vals[a])
        obs = _scalar_iqm(seed_vals[a]) - _scalar_iqm(seed_vals[b])
        iqm_diffs.append(obs)
        count = sum(
            1 for _ in range(n_permutations)
            if abs(_scalar_iqm((perm := rng.permutation(combined))[:na])
                   - _scalar_iqm(perm[na:])) >= abs(obs)
        )
        raw_p.append(count / n_permutations)

    order = np.argsort(raw_p)
    adj_p = np.empty(len(raw_p))
    if reference is not None:
        # Holm-Bonferroni (= Shaffer when no logical constraints between hypotheses)
        running_max = 0.0
        for rank, idx in enumerate(order):
            running_max = max(running_max, min(1.0, raw_p[idx] * (len(pairs) - rank)))
            adj_p[idx] = running_max
        correction_label = "Holm-Bonferroni"
    else:
        # Shaffer: use logically-constrained multipliers
        t_set = _shaffer_t_values(len(present))
        running_max = 0.0
        for rank, idx in enumerate(order):
            remaining = len(pairs) - rank
            t_i = max((t for t in t_set if t <= remaining), default=0)
            running_max = max(running_max, min(1.0, raw_p[idx] * t_i))
            adj_p[idx] = running_max
        correction_label = "Shaffer"

    print(f"\nIQM permutation test + {correction_label} — {title}  (α={alpha}, {n_permutations:,} perms)")
    print("=" * 72)
    print(f"{'Group 1':<28} {'Group 2':<28} {'Δ IQM':>14} {'p-adj':>8} {'Reject':>8}")
    print("-" * 72)
    for (a, b), diff, p in zip(pairs, iqm_diffs, adj_p):
        diff_str = f"{diff:,.0f}{unit}" if unit else f"{diff:.2f}"
        print(
            f"{display.get(a, a):<28} {display.get(b, b):<28}"
            f" {diff_str:>14} {p:>8.4f} {'Yes' if p < alpha else 'No':>8}"
        )
    print("=" * 72)


def pairwise_significance_expert_bias(
    df: pd.DataFrame,
    metric: str,
    experiments: list[str],
    last_n_steps: int = 200_000,
    thresholds: list[float] | None = None,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    seed: int = 42,
    remove_censored: bool = False,
) -> None:
    """
    Pairwise significance tests for Eval/expert_bias.

    For each section, two tables are printed:
    - Tukey HSD (parametric, compares group means).
    - IQM permutation test + Shaffer correction (compares group IQMs,
      accounts for logically impossible null-hypothesis combinations).

    Sections:
    1. Asymptote: IQM over last `last_n_steps` steps per seed.
    2. Steps to threshold: per-seed first-permanent-crossing time for each
       threshold in `thresholds` (budget-censored RMST-style, or censored seeds
       removed if remove_censored=True).
    """
    if thresholds is None:
        thresholds = METRIC_THRESHOLDS.get(metric, [0.0, 200.0, 400.0])

    present = [e for e in experiments if e in df["label"].unique()]
    display = {lbl: _display_name(lbl) for lbl in present}
    rng = np.random.default_rng(seed)
    censor_at = float(df["step"].max()) if not remove_censored else None

    # ---- Section 1: asymptote ----
    seed_vals = {
        lbl: _per_seed_asymptote(df, metric, lbl, last_n_steps=last_n_steps)
        for lbl in present
    }
    reference = MAIN_EXP if MAIN_EXP in present else None
    _print_pairwise_tests(
        seed_vals, present, display,
        title=f"last {last_n_steps // 1000}k steps of {metric}",
        alpha=alpha, n_permutations=n_permutations, rng=rng,
        reference=reference,
    )

    # ---- Section 2: steps to threshold ----
    censor_label = "removed" if remove_censored else "budget-censored"
    _invert_crossing = metric in METRIC_LOWER_IS_BETTER
    direction_str = "≤" if _invert_crossing else "="
    for threshold in thresholds:
        threshold_str = f"{threshold:.2f}" if _invert_crossing else ("0" if threshold == 0 else str(int(threshold)))
        seed_crossing = {}
        for lbl in present:
            vals, n_censored = _per_seed_crossing(
                df, metric, lbl, threshold, censor_at=censor_at,
                remove_censored=remove_censored, invert=_invert_crossing,
            )
            seed_crossing[lbl] = vals
            if n_censored:
                print(f"  [{lbl}] steps-to-y{direction_str}{threshold_str}: {n_censored} seed(s) {censor_label}")
        _print_pairwise_tests(
            seed_crossing, present, display,
            title=f"steps to y{direction_str}{threshold_str} of {metric} ({censor_label})",
            alpha=alpha, n_permutations=n_permutations, rng=rng,
            unit=" steps",
            reference=reference,
        )
    print()


def plot_expert_bias_summary(
    df: pd.DataFrame,
    metric: str,
    experiments: list[str],
    title: str = "",
    thresholds: list[float] | None = None,
    figsize: tuple = (14, 5),
    n_bootstrap: int = 1000,
    ci: float = 95,
    remove_censored: bool = False,
) -> plt.Figure:
    """
    Multi-panel summary comparing EGE, IBRL and SAC on expert bias.

    Same layout as plot_ablation_summary_stats: one panel per threshold (time to
    first-permanent-crossing), plus a final panel ranking by peak (max) per-seed
    performance IQM.  Row order is fixed by the experiments list.
    """
    if thresholds is None:
        thresholds = METRIC_THRESHOLDS.get(metric, [0.0, 200.0, 400.0])

    present = [e for e in experiments if e in df["label"].unique()]
    n_panels = len(thresholds) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    rng = np.random.default_rng(42)

    _BAR_H = 0.30
    _IQM_LW = 2.5

    def _color(lbl):
        return ABLATION_COLOR_MAP.get(lbl, PALETTE[present.index(lbl) % len(PALETTE)])

    def _fmt_step(s):
        return f"{s/1e6:.2f}M" if s >= 1e6 else f"{int(s/1e3)}k"

    def _draw_ci_row(ax, yi, center, lo, hi, color, label_text):
        ax.fill_betweenx(
            [yi - _BAR_H, yi + _BAR_H], lo, hi,
            color=color, alpha=0.35, linewidth=0,
        )
        for x in (lo, hi):
            ax.plot([x, x], [yi - _BAR_H, yi + _BAR_H],
                    color=color, linewidth=1.2, alpha=0.7)
        ax.plot([center, center], [yi - _BAR_H, yi + _BAR_H],
                color=color, linewidth=_IQM_LW, solid_capstyle="butt")
        ax.text(hi, yi, f"  {label_text}", va="center", fontsize=7.5, color=color)

    def _fit_xlim(ax, ranges, extra_right_frac=4.0):
        if not ranges:
            return
        all_lo = min(lo for lo, hi in ranges)
        all_hi = max(hi for lo, hi in ranges)
        pad = (all_hi - all_lo) * 0.15 or abs(all_hi) * 0.05 or 1.0
        ax.set_xlim(all_lo - pad, all_hi + pad * extra_right_frac)

    censor_at = float(df["step"].max()) if not remove_censored else None
    _invert_crossing = metric in METRIC_LOWER_IS_BETTER

    # ---- Crossing-time panels ----
    for panel_idx, threshold in enumerate(thresholds):
        ax = axes[panel_idx]
        ranges = []
        for yi, lbl in enumerate(present):
            vals, n_censored = _per_seed_crossing(
                df, metric, lbl, threshold, censor_at=censor_at,
                remove_censored=remove_censored, invert=_invert_crossing,
            )
            n_total = len(vals) + (n_censored if remove_censored else 0)
            n_reached = n_total - n_censored
            center, lo, hi = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            color = _color(lbl)
            suffix = f"†" if n_censored > 0 else ""
            _draw_ci_row(ax, yi, center, lo, hi, color,
                         f"{_fmt_step(center)}{suffix}  ({n_reached}/{n_total})")
            ranges.append((lo, hi))

        if _invert_crossing:
            threshold_str = f"{threshold:.2f}"
            direction_str = "≤"
        else:
            threshold_str = "0" if threshold == 0 else str(int(threshold))
            direction_str = "="
        ax.set_yticks(np.arange(len(present)))
        if panel_idx == 0:
            ax.set_yticklabels([_display_name(l) for l in present], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Environment Steps", fontsize=10)
        censor_note = "censored seeds removed" if remove_censored else "†budget-censored"
        ax.set_title(f"Steps to y{direction_str}{threshold_str}\n(IQM  ·  {ci:.0f}% CI  ·  {censor_note})", fontsize=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
        )
        ax.set_ylim(-0.7, len(present) - 0.3)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        _fit_xlim(ax, ranges)

    # ---- Peak performance panel ----
    ax = axes[-1]
    ranges = []
    for yi, lbl in enumerate(present):
        vals = _per_seed_max_perf(df, metric, lbl)
        center, lo, hi = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        color = _color(lbl)
        if not np.isnan(center):
            _draw_ci_row(ax, yi, center, lo, hi, color, f"{center:.0f}")
            ranges.append((lo, hi))

    y_label = METRIC_AXIS_LABELS.get(metric, metric)
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([])
    ax.set_xlabel(y_label, fontsize=10)
    ax.set_title(
        f"Peak performance\n(IQM  ·  {ci:.0f}% bootstrap CI)",
        fontsize=10,
    )
    ax.set_ylim(-0.7, len(present) - 0.3)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    _fit_xlim(ax, ranges)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_ablation_summary_stats(
    df: pd.DataFrame,
    metric: str,
    experiments: list[str],
    title: str = "",
    thresholds: list[float] | None = None,
    last_n_steps: int = 200_000,
    figsize: tuple = (14, 5),
    n_bootstrap: int = 1000,
    ci: float = 95,
    remove_censored: bool = False,
) -> plt.Figure:
    """
    Three-panel summary for one ablation question.

    Panel 1+: Time to first-permanent-crossing of each threshold (in steps).
    Last panel: Final performance IQM over last last_n_steps steps.

    Each method is a filled CI rectangle with a bold vertical IQM tick.
    Axes are fitted to actual values — no forced zero origin.
    """
    if thresholds is None:
        thresholds = METRIC_THRESHOLDS.get(metric, [0.0, 200.0, 400.0])

    present = [e for e in experiments if e in df["label"].unique()]
    n_panels = len(thresholds) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    rng = np.random.default_rng(42)

    _BAR_H = 0.30
    _IQM_LW = 2.5

    def _color(lbl):
        return ABLATION_COLOR_MAP.get(lbl, PALETTE[present.index(lbl) % len(PALETTE)])

    def _fmt_step(s):
        return f"{s/1e6:.2f}M" if s >= 1e6 else f"{int(s/1e3)}k"

    def _draw_ci_row(ax, yi, center, lo, hi, color, label_text):
        ax.fill_betweenx(
            [yi - _BAR_H, yi + _BAR_H], lo, hi,
            color=color, alpha=0.35, linewidth=0,
        )
        for x in (lo, hi):
            ax.plot([x, x], [yi - _BAR_H, yi + _BAR_H],
                    color=color, linewidth=1.2, alpha=0.7)
        ax.plot([center, center], [yi - _BAR_H, yi + _BAR_H],
                color=color, linewidth=_IQM_LW, solid_capstyle="butt")
        ax.text(hi, yi, f"  {label_text}", va="center", fontsize=7.5, color=color)

    def _fit_xlim(ax, ranges, extra_right_frac=4.0):
        """Set x-limits fitted to actual data ranges with label margin."""
        if not ranges:
            return
        all_lo = min(lo for lo, hi in ranges)
        all_hi = max(hi for lo, hi in ranges)
        pad = (all_hi - all_lo) * 0.15 or abs(all_hi) * 0.05 or 1.0
        ax.set_xlim(all_lo - pad, all_hi + pad * extra_right_frac)

    censor_at = float(df["step"].max()) if not remove_censored else None
    _invert_crossing = metric in METRIC_LOWER_IS_BETTER

    # ---- Crossing-time panels ----
    for panel_idx, threshold in enumerate(thresholds):
        ax = axes[panel_idx]
        ranges = []
        for yi, lbl in enumerate(present):
            vals, n_censored = _per_seed_crossing(
                df, metric, lbl, threshold, censor_at=censor_at,
                remove_censored=remove_censored, invert=_invert_crossing,
            )
            n_total = len(vals) + (n_censored if remove_censored else 0)
            n_reached = n_total - n_censored
            center, lo, hi = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            color = _color(lbl)
            suffix = f"†" if n_censored > 0 else ""
            _draw_ci_row(ax, yi, center, lo, hi, color,
                         f"{_fmt_step(center)}{suffix}  ({n_reached}/{n_total})")
            ranges.append((lo, hi))

        if _invert_crossing:
            threshold_str = f"{threshold:.2f}"
            direction_str = "≤"
        else:
            threshold_str = "0" if threshold == 0 else str(int(threshold))
            direction_str = "="
        ax.set_yticks(np.arange(len(present)))
        if panel_idx == 0:
            ax.set_yticklabels([_display_name(l) for l in present], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Environment Steps", fontsize=10)
        censor_note = "censored seeds removed" if remove_censored else "†budget-censored"
        ax.set_title(f"Steps to y{direction_str}{threshold_str}\n(IQM  ·  {ci:.0f}% CI  ·  {censor_note})", fontsize=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
        )
        ax.set_ylim(-0.7, len(present) - 0.3)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        _fit_xlim(ax, ranges)

    # ---- Final performance panel ----
    ax = axes[-1]
    ranges = []
    for yi, lbl in enumerate(present):
        vals = _per_seed_final_perf(df, metric, lbl, last_n_steps)
        center, lo, hi = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        color = _color(lbl)
        if not np.isnan(center):
            _draw_ci_row(ax, yi, center, lo, hi, color, f"{center:.0f}")
            ranges.append((lo, hi))

    y_label = METRIC_AXIS_LABELS.get(metric, metric)
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([])
    ax.set_xlabel(y_label, fontsize=10)
    ax.set_title(
        f"Final performance  (last {last_n_steps//1000}k steps)\n(IQM  ·  {ci:.0f}% bootstrap CI)",
        fontsize=10,
    )
    ax.set_ylim(-0.7, len(present) - 0.3)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    _fit_xlim(ax, ranges)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_ablation_questions(
    registry: dict[str, dict],
    tb_dir: Path,
    out_dir: Path,
    metric: str = "Eval/expert_bias",
    smooth: int = 1,
    n_bootstrap: int = 1000,
    ci: float = 95,
    ylim_bottom: float = -250,
    remove_censored: bool = False,
    no_cache: bool = False,
) -> None:
    """
    Produce one subfolder per ablation question, each containing:
      - Training curve (IQM, full y range)
      - Training curve (IQM, focused)
      - Summary-stats figure (time-to-0, time-to-250, final perf)
    """
    print("\n" + "=" * 60)
    print("Ablation question plots")
    print("=" * 60)

    print(f"  Loading '{metric}' from TensorBoard...")
    df_full = load_tb_data(registry, tb_dir, metric, no_cache=no_cache)
    if df_full.empty:
        print(f"  No data found for metric '{metric}'. Skipping ablation plots.")
        return

    df_full = resolve_exp_name(df_full)
    df_full = deduplicate_runs(df_full)

    metric_slug = metric.replace("/", "_")
    smooth_tag = f"_smooth{smooth}" if smooth > 1 else ""

    for q in ABLATION_QUESTIONS:
        key = q["key"]
        title = q["title"]
        subtitle = q["subtitle"]
        want = q["experiments"]

        # Apply group filter: exclude runs whose W&B group is not in allowed_groups.
        # This prevents stale runs (e.g. eps experiments with wrong decay) from polluting plots.
        allowed_groups = q.get("allowed_groups")
        if allowed_groups is not None and "group" in df_full.columns:
            df_filtered = df_full[df_full["group"].isin(allowed_groups) | (df_full["group"] == "")]
        else:
            df_filtered = df_full

        available_q = set(df_filtered["label"].unique())
        present = [e for e in want if e in available_q]
        missing = [e for e in want if e not in available_q]
        if missing:
            print(f"  [{key}] Warning: missing experiments {missing} (not in registry/TB data for allowed groups).")
        if not present:
            print(f"  [{key}] No data — skipping.")
            continue

        q_dir = out_dir / key
        q_dir.mkdir(exist_ok=True)

        df_q = df_filtered[df_filtered["label"].isin(present)].copy()

        # For the decay-horizon question: dashed line for ege_decay_005 (shortest)
        # to visually distinguish it from the main result.
        is_decay_q = (key == "q1_decay_horizon")
        linestyle_map = {"ege_decay_005": "--"} if is_decay_q else None

        def _q_plot_metric(df, **kw):
            fig = plot_metric(df, metric=metric, smooth=smooth,
                              n_bootstrap=n_bootstrap, ci=ci,
                              color_map=ABLATION_COLOR_MAP,
                              labels=present,
                              linestyle_map=linestyle_map,
                              **kw)
            fig.axes[0].set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
            return fig

        print(f"\n  [{key}] Plotting into {q_dir}/")

        # Training curve — IQM, full y range
        fig = _q_plot_metric(df_q, aggregation="iqm")
        out_path = q_dir / f"{metric_slug}{smooth_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {out_path}")
        plt.close(fig)

        # Training curve — IQM, focused
        fig = _q_plot_metric(df_q, aggregation="iqm", ylim_bottom=ylim_bottom)

        # Decay-horizon question only: inset showing ege_decay_005 − MAIN_EXP
        # IQM difference with bootstrap CI.  Hovering near zero means
        # "shorter horizon is statistically indistinguishable from main result".
        if is_decay_q:
            zoom_labels = ["ege_decay_005", MAIN_EXP]
            zoom_present = [l for l in zoom_labels if l in present]
            if len(zoom_present) == 2:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_main = fig.axes[0]
                ax_ins = inset_axes(ax_main, width="38%", height="38%", loc="lower right",
                                    bbox_to_anchor=(0.0, 0.04, 0.98, 0.97),
                                    bbox_transform=ax_main.transAxes)
                inset_smooth = max(smooth, 5)
                per_label = {}
                for label in zoom_present:
                    group = df_q[df_q["label"] == label]
                    per_seed = (
                        group.groupby(["run_id", "step"])[metric]
                        .mean().unstack("run_id").sort_index()
                    )
                    per_label[label] = per_seed

                common_steps = np.intersect1d(
                    per_label["ege_decay_005"].index.values,
                    per_label[MAIN_EXP].index.values,
                )
                vals_005  = per_label["ege_decay_005"].reindex(common_steps).ffill().bfill().values
                vals_main = per_label[MAIN_EXP].reindex(common_steps).ffill().bfill().values

                n_boot = 500
                rng_b = np.random.default_rng(42)
                n_a, n_b = vals_005.shape[1], vals_main.shape[1]
                boot_diffs = np.empty((n_boot, len(common_steps)))
                for i in range(n_boot):
                    idx_a = rng_b.integers(0, n_a, n_a)
                    idx_b = rng_b.integers(0, n_b, n_b)
                    boot_diffs[i] = _iqm(vals_005[:, idx_a]) - _iqm(vals_main[:, idx_b])
                diff_mean = smooth_series(
                    pd.Series(_iqm(vals_005) - _iqm(vals_main)), inset_smooth).values
                alpha_tail = (100 - ci) / 2
                ci_lo_d = smooth_series(
                    pd.Series(np.percentile(boot_diffs, alpha_tail, axis=0)), inset_smooth).values
                ci_hi_d = smooth_series(
                    pd.Series(np.percentile(boot_diffs, 100 - alpha_tail, axis=0)), inset_smooth).values

                clip_mask = common_steps >= 200_000
                diff_color = ABLATION_COLOR_MAP.get("ege_decay_005", "#7DCE82")
                ax_ins.axhline(0, color="#999999", linewidth=0.8, linestyle="-", zorder=1)
                ax_ins.fill_between(common_steps[clip_mask], ci_lo_d[clip_mask], ci_hi_d[clip_mask],
                                    color=diff_color, alpha=0.20, zorder=2)
                ax_ins.plot(common_steps[clip_mask], diff_mean[clip_mask], color=diff_color,
                            linewidth=1.4, linestyle="--", zorder=3)
                ax_ins.xaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f"{int(x/1e3)}k"))
                ax_ins.set_title(f"decay=0.05 − {EXP_DISPLAY_NAMES.get(MAIN_EXP, MAIN_EXP)}", fontsize=8)
                ax_ins.set_ylabel("Δ IQM", fontsize=7, labelpad=2)
                ax_ins.tick_params(labelsize=7, pad=1)
                ax_ins.set_position([
                    ax_ins.get_position().x0 + 0.04,
                    ax_ins.get_position().y0,
                    ax_ins.get_position().width,
                    ax_ins.get_position().height,
                ])

        out_path = q_dir / f"{metric_slug}{smooth_tag}_focus.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {out_path}")
        plt.close(fig)

        # Summary-stats figure — time-to-0, time-to-200, time-to-400, final perf (all IQM)
        fig = plot_ablation_summary_stats(
            df_q, metric=metric, experiments=present,
            title=f"{title}\n{subtitle}",
            thresholds=METRIC_THRESHOLDS.get(metric, [0.0, 200.0, 400.0]),
            last_n_steps=200_000,
            n_bootstrap=n_bootstrap, ci=ci,
            remove_censored=remove_censored,
        )
        out_path = q_dir / f"{metric_slug}{smooth_tag}_summary_stats.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {out_path}")
        plt.close(fig)

    print("\nAblation plots complete.")


# ---------------------------------------------------------------------------
# Noisy expert degradation plots
# ---------------------------------------------------------------------------


def plot_degradation_curve(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float] | None = None,
    last_n_steps: int = 200_000,
    figsize: tuple | None = None,
    n_bootstrap: int = 1000,
    ci: float = 95,
    remove_censored: bool = False,
    title: str = "",
    df_sac_baseline: "pd.DataFrame | None" = None,
) -> plt.Figure:
    """
    1×(len(thresholds)+1) degradation curve: IQM statistic vs expert noise level.

    One line per algorithm (EGE / IBRL / Residual RL), one colour per algorithm.
    Panels: one per crossing threshold + one for final performance.
    y-axis is autoscaled to the data (no forced zero).
    """
    if thresholds is None:
        thresholds = [0.0, 200.0, 400.0]

    _ALGOS = [
        ("EGE",         NOISY_EXPERT_EXPERIMENTS,   "#2166AC"),
        ("IBRL",        IBRL_NOISY_EXPERIMENTS,      "#F4762A"),
        ("Residual RL", RESIDUAL_NOISY_EXPERIMENTS, "#27AE60"),
    ]

    n_cols = len(thresholds) + 1
    if figsize is None:
        figsize = (4.5 * n_cols, 4.5)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, n_cols)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]

    censor_at = float(df["step"].max()) if not remove_censored else None
    _invert_crossing = metric in METRIC_LOWER_IS_BETTER
    rng = np.random.default_rng(42)
    available = set(df["label"].unique())

    # ---- Crossing-time panels ----
    for panel_idx, threshold in enumerate(thresholds):
        ax = axes[panel_idx]

        for algo_label, exps, color in _ALGOS:
            present = [(e, NOISY_EXPERT_NOISE_PCTS[i]) for i, e in enumerate(exps) if e in available]
            if not present:
                continue
            xs = np.array([pct for _, pct in present], dtype=float)
            centers, lo_arr, hi_arr = [], [], []
            for exp, _ in present:
                vals, _ = _per_seed_crossing(
                    df, metric, exp, threshold, censor_at=censor_at,
                    remove_censored=remove_censored, invert=_invert_crossing,
                )
                c, lo, hi = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
                centers.append(c); lo_arr.append(lo); hi_arr.append(hi)

            centers = np.array(centers); lo_arr = np.array(lo_arr); hi_arr = np.array(hi_arr)
            valid = ~np.isnan(centers)
            if valid.any():
                ax.plot(xs[valid], centers[valid], color=color, linewidth=1.8,
                        marker="o", markersize=5, label=algo_label, zorder=3)
                ax.fill_between(xs[valid], lo_arr[valid], hi_arr[valid],
                                color=color, alpha=0.15, zorder=2)

        # SAC baseline hline
        if df_sac_baseline is not None and not df_sac_baseline.empty:
            sac_vals, _ = _per_seed_crossing(
                df_sac_baseline, metric, "sac_baseline", threshold,
                censor_at=censor_at, remove_censored=remove_censored,
                invert=_invert_crossing,
            )
            sac_c, _, _ = _bootstrap_scalar_iqm_ci(sac_vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            if not np.isnan(sac_c):
                ax.axhline(sac_c, color="#5D6D7E", linewidth=1.2, linestyle="--",
                           label="SAC (baseline)", zorder=1)

        if _invert_crossing:
            threshold_str, direction_str = f"{threshold:.2f}", "≤"
        else:
            threshold_str = "0" if threshold == 0 else str(int(threshold))
            direction_str = "="

        censor_note = "censored removed" if remove_censored else "†budget-censored"
        ax.set_title(
            f"Steps to y{direction_str}{threshold_str}\n(IQM · {ci:.0f}% CI · {censor_note})",
            fontsize=10,
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
        )
        ax.set_xticks(NOISY_EXPERT_NOISE_PCTS)
        ax.set_xticklabels([f"{p:.0f}%" for p in NOISY_EXPERT_NOISE_PCTS], fontsize=8)
        ax.set_xlabel("Expert noise level", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=9, loc="best")

    # ---- Final performance panel ----
    ax = axes[-1]
    y_label = METRIC_AXIS_LABELS.get(metric, metric)

    for algo_label, exps, color in _ALGOS:
        present = [(e, NOISY_EXPERT_NOISE_PCTS[i]) for i, e in enumerate(exps) if e in available]
        if not present:
            continue
        xs = np.array([pct for _, pct in present], dtype=float)
        centers, lo_arr, hi_arr = [], [], []
        for exp, _ in present:
            vals = _per_seed_final_perf(df, metric, exp, last_n_steps)
            c, lo, hi = _bootstrap_scalar_iqm_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            centers.append(c); lo_arr.append(lo); hi_arr.append(hi)

        centers = np.array(centers); lo_arr = np.array(lo_arr); hi_arr = np.array(hi_arr)
        valid = ~np.isnan(centers)
        if valid.any():
            ax.plot(xs[valid], centers[valid], color=color, linewidth=1.8,
                    marker="o", markersize=5, label=algo_label, zorder=3)
            ax.fill_between(xs[valid], lo_arr[valid], hi_arr[valid],
                            color=color, alpha=0.15, zorder=2)

    # SAC baseline hline on final performance panel
    if df_sac_baseline is not None and not df_sac_baseline.empty:
        sac_vals = _per_seed_final_perf(df_sac_baseline, metric, "sac_baseline", last_n_steps)
        sac_c, _, _ = _bootstrap_scalar_iqm_ci(sac_vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        if not np.isnan(sac_c):
            ax.axhline(sac_c, color="#5D6D7E", linewidth=1.2, linestyle="--",
                       label="SAC (baseline)", zorder=1)

    ax.set_title(
        f"Final {y_label} (last {last_n_steps // 1000}k steps)\n(IQM · {ci:.0f}% bootstrap CI)",
        fontsize=10,
    )
    ax.set_xticks(NOISY_EXPERT_NOISE_PCTS)
    ax.set_xticklabels([f"{p:.0f}%" for p in NOISY_EXPERT_NOISE_PCTS], fontsize=8)
    ax.set_xlabel("Expert noise level", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9, loc="best")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_noisy_expert_algo_lines(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float] | None = None,
    last_n_steps: int = 200_000,
    figsize: tuple | None = None,
    n_bootstrap: int = 1000,
    ci: float = 95,
    remove_censored: bool = False,
    title: str = "",
) -> plt.Figure:
    """
    1×(len(thresholds)+1) line plot comparing EGE, IBRL and Residual-RL.

    x-axis: expert noise level (%)
    y-axis: IQM statistic — one panel per crossing threshold + one for final perf
    One line per algorithm with bootstrap CI shading.
    """
    if thresholds is None:
        thresholds = [0.0, 200.0, 400.0]

    _ALGOS = [
        ("EGE",         NOISY_EXPERT_EXPERIMENTS,    "#2166AC"),
        ("IBRL",        IBRL_NOISY_EXPERIMENTS,       "#F4762A"),
        ("Residual RL", RESIDUAL_NOISY_EXPERIMENTS,  "#27AE60"),
    ]

    n_cols = len(thresholds) + 1
    if figsize is None:
        figsize = (4.5 * n_cols, 4.5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    censor_at = float(df["step"].max()) if not remove_censored else None
    _invert_crossing = metric in METRIC_LOWER_IS_BETTER
    rng = np.random.default_rng(42)
    available = set(df["label"].unique())
    alpha_tail = (100 - ci) / 2

    # ---- Crossing-time panels ----
    for panel_idx, threshold in enumerate(thresholds):
        ax = axes[panel_idx]

        for algo_label, exps, color in _ALGOS:
            present = [(e, NOISY_EXPERT_NOISE_PCTS[i]) for i, e in enumerate(exps) if e in available]
            if not present:
                continue
            xs = np.array([pct for _, pct in present], dtype=float)
            centers, lo_arr, hi_arr = [], [], []

            for exp, _ in present:
                vals, _ = _per_seed_crossing(
                    df, metric, exp, threshold, censor_at=censor_at,
                    remove_censored=remove_censored, invert=_invert_crossing,
                )
                c, lo, hi = _bootstrap_scalar_iqm_ci(
                    vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
                centers.append(c)
                lo_arr.append(lo)
                hi_arr.append(hi)

            centers = np.array(centers)
            lo_arr  = np.array(lo_arr)
            hi_arr  = np.array(hi_arr)
            valid   = ~np.isnan(centers)

            if valid.any():
                ax.plot(xs[valid], centers[valid], color=color, linewidth=1.8,
                        marker="o", markersize=5, label=algo_label, zorder=3)
                ax.fill_between(xs[valid], lo_arr[valid], hi_arr[valid],
                                color=color, alpha=0.15, zorder=2)

        if _invert_crossing:
            threshold_str, direction_str = f"{threshold:.2f}", "≤"
        else:
            threshold_str = "0" if threshold == 0 else str(int(threshold))
            direction_str = "="

        ax.set_xticks(NOISY_EXPERT_NOISE_PCTS)
        ax.set_xticklabels([f"{p:.0f}%" for p in NOISY_EXPERT_NOISE_PCTS], fontsize=8)
        ax.set_xlabel("Expert noise level", fontsize=10)
        censor_note = "censored seeds removed" if remove_censored else "†budget-censored"
        ax.set_title(
            f"Steps to y{direction_str}{threshold_str}\n"
            f"(IQM · {ci:.0f}% CI · {censor_note})",
            fontsize=10,
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        if panel_idx == 0:
            ax.legend(fontsize=9, loc="best")

    # ---- Final performance panel ----
    ax = axes[-1]
    y_label = METRIC_AXIS_LABELS.get(metric, metric)

    for algo_label, exps, color in _ALGOS:
        present = [(e, NOISY_EXPERT_NOISE_PCTS[i]) for i, e in enumerate(exps) if e in available]
        if not present:
            continue
        xs = np.array([pct for _, pct in present], dtype=float)
        centers, lo_arr, hi_arr = [], [], []

        for exp, _ in present:
            vals = _per_seed_final_perf(df, metric, exp, last_n_steps)
            c, lo, hi = _bootstrap_scalar_iqm_ci(
                vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            centers.append(c)
            lo_arr.append(lo)
            hi_arr.append(hi)

        centers = np.array(centers)
        lo_arr  = np.array(lo_arr)
        hi_arr  = np.array(hi_arr)
        valid   = ~np.isnan(centers)

        if valid.any():
            ax.plot(xs[valid], centers[valid], color=color, linewidth=1.8,
                    marker="o", markersize=5, label=algo_label, zorder=3)
            ax.fill_between(xs[valid], lo_arr[valid], hi_arr[valid],
                            color=color, alpha=0.15, zorder=2)

    ax.axhline(0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xticks(NOISY_EXPERT_NOISE_PCTS)
    ax.set_xticklabels([f"{p:.0f}%" for p in NOISY_EXPERT_NOISE_PCTS], fontsize=8)
    ax.set_xlabel("Expert noise level", fontsize=10)
    ax.set_title(
        f"Final {y_label} (last {last_n_steps // 1000}k steps)\n"
        f"(IQM · {ci:.0f}% bootstrap CI)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_noisy_expert_algo_rows(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float] | None = None,
    last_n_steps: int = 200_000,
    figsize: tuple | None = None,
    n_bootstrap: int = 1000,
    ci: float = 95,
    remove_censored: bool = False,
    title: str = "",
) -> plt.Figure:
    """
    3-row summary comparing EGE, IBRL and Residual-RL across all noise levels.

    Each row is one algorithm; within a row the y-axis lists noise levels (0%→80%)
    and the columns are: one panel per crossing threshold + one final-performance panel.
    Same noise level always gets the same colour across all rows.
    """
    if thresholds is None:
        thresholds = [0.0, 200.0, 400.0]

    n_cols = len(thresholds) + 1

    _ALGO_ROWS = [
        ("EGE  (ε-greedy, decay=0.50)", NOISY_EXPERT_EXPERIMENTS),
        ("IBRL  (argmax, no decay)",     IBRL_NOISY_EXPERIMENTS),
        ("Residual RL",                  RESIDUAL_NOISY_EXPERIMENTS),
    ]

    if figsize is None:
        figsize = (4.5 * n_cols, 3.5 * 3)

    fig, all_axes = plt.subplots(3, n_cols, figsize=figsize, sharex="col")

    _BAR_H = 0.30
    _IQM_LW = 2.5
    censor_at = float(df["step"].max()) if not remove_censored else None
    _invert_crossing = metric in METRIC_LOWER_IS_BETTER
    rng = np.random.default_rng(42)

    def _color(exp_name):
        return NOISY_EXPERT_COLOR_MAP.get(exp_name, "#999999")

    def _draw_ci_row(ax, yi, center, lo, hi, color, label_text):
        ax.fill_betweenx(
            [yi - _BAR_H, yi + _BAR_H], lo, hi,
            color=color, alpha=0.35, linewidth=0,
        )
        for x in (lo, hi):
            ax.plot([x, x], [yi - _BAR_H, yi + _BAR_H],
                    color=color, linewidth=1.2, alpha=0.7)
        ax.plot([center, center], [yi - _BAR_H, yi + _BAR_H],
                color=color, linewidth=_IQM_LW, solid_capstyle="butt")
        ax.text(hi, yi, f"  {label_text}", va="center", fontsize=7.5, color=color)

    def _fmt_step(s):
        return f"{s/1e6:.2f}M" if s >= 1e6 else f"{int(s/1e3)}k"

    available = set(df["label"].unique())

    # Accumulate lo/hi ranges per column across all rows so we can unify xlim.
    col_ranges: list[list[tuple[float, float]]] = [[] for _ in range(n_cols)]

    for row_idx, (algo_label, exps) in enumerate(_ALGO_ROWS):
        present = [e for e in exps if e in available]
        noise_pcts_present = [NOISY_EXPERT_NOISE_PCTS[i] for i, e in enumerate(exps) if e in available]
        axes = all_axes[row_idx]

        # ---- Crossing-time panels ----
        for panel_idx, threshold in enumerate(thresholds):
            ax = axes[panel_idx]
            for yi, lbl in enumerate(present):
                vals, n_censored = _per_seed_crossing(
                    df, metric, lbl, threshold, censor_at=censor_at,
                    remove_censored=remove_censored, invert=_invert_crossing,
                )
                n_total = len(vals) + (n_censored if remove_censored else 0)
                n_reached = n_total - n_censored
                center, lo, hi = _bootstrap_scalar_iqm_ci(
                    vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
                color = _color(lbl)
                suffix = "†" if n_censored > 0 else ""
                _draw_ci_row(ax, yi, center, lo, hi, color,
                             f"{_fmt_step(center)}{suffix}  ({n_reached}/{n_total})")
                if not np.isnan(lo) and not np.isnan(hi):
                    col_ranges[panel_idx].append((lo, hi))

            if _invert_crossing:
                threshold_str, direction_str = f"{threshold:.2f}", "≤"
            else:
                threshold_str = "0" if threshold == 0 else str(int(threshold))
                direction_str = "="

            ax.set_yticks(np.arange(len(present)))
            if panel_idx == 0:
                ax.set_yticklabels([f"{p:.0f}%" for p in noise_pcts_present], fontsize=9)
                ax.set_ylabel(algo_label, fontsize=9, labelpad=4)
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                censor_note = "censored seeds removed" if remove_censored else "†budget-censored"
                ax.set_title(
                    f"Steps to y{direction_str}{threshold_str}\n"
                    f"(IQM · {ci:.0f}% CI · {censor_note})",
                    fontsize=9,
                )
            if row_idx == 2:
                ax.set_xlabel("Environment Steps", fontsize=9)

            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
            )
            ax.set_ylim(-0.7, len(present) - 0.3)
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

        # ---- Final performance panel ----
        ax = axes[-1]
        for yi, lbl in enumerate(present):
            vals = _per_seed_final_perf(df, metric, lbl, last_n_steps)
            center, lo, hi = _bootstrap_scalar_iqm_ci(
                vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            color = _color(lbl)
            if not np.isnan(center):
                _draw_ci_row(ax, yi, center, lo, hi, color, f"{center:.0f}")
                col_ranges[-1].append((lo, hi))

        y_label = METRIC_AXIS_LABELS.get(metric, metric)
        ax.set_yticks(np.arange(len(present)))
        ax.set_yticklabels([])
        if row_idx == 0:
            ax.set_title(
                f"Final performance (last {last_n_steps // 1000}k steps)\n"
                f"(IQM · {ci:.0f}% bootstrap CI)",
                fontsize=9,
            )
        if row_idx == 2:
            ax.set_xlabel(y_label, fontsize=9)
        ax.set_ylim(-0.7, len(present) - 0.3)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    # Apply unified xlim per column (shared x-axis means setting it on row 0 propagates).
    for col_idx, ranges in enumerate(col_ranges):
        if not ranges:
            continue
        all_lo = min(lo for lo, hi in ranges)
        all_hi = max(hi for lo, hi in ranges)
        pad = (all_hi - all_lo) * 0.15 or abs(all_hi) * 0.05 or 1.0
        all_axes[0][col_idx].set_xlim(all_lo - pad, all_hi + pad * 4.0)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_noisy_expert_questions(
    registry: dict[str, dict],
    tb_dir: Path,
    out_dir: Path,
    ablation_registry: dict[str, dict] | None = None,
    metric: str = "Eval/expert_bias",
    smooth: int = 1,
    n_bootstrap: int = 1000,
    ci: float = 95,
    ylim_bottom: float = -250,
    remove_censored: bool = False,
    no_cache: bool = False,
) -> None:
    """
    Produce a subfolder 'noisy_expert_degradation' with:
      - Training curves (IQM, full + focused) for all noise levels
      - Summary-stats figure (time-to-0, time-to-200, final perf)
      - Degradation curve: final IQM vs noise_pct (the key result figure)

    If ablation_registry is provided, the ege_noise_0pct reference is taken
    from there (ege_decay_050) when not present in the noisy expert registry.
    """
    print("\n" + "=" * 60)
    print("Noisy expert degradation plots")
    print("=" * 60)

    print(f"  Loading '{metric}' from TensorBoard...")
    df = load_tb_data(registry, tb_dir, metric, no_cache=no_cache)

    # Supplement 0pct references from the ablation registry when not yet in
    # the noisy expert registry (avoids re-running identical experiments).
    df_sac_baseline: "pd.DataFrame | None" = None
    if ablation_registry:
        present_in_noisy = set(df["exp_name"].unique()) if not df.empty else set()
        df_abl = None  # load lazily — at most once
        for zero_exp, ablation_name in _ABLATION_EQUIV_0PCT.items():
            if zero_exp not in present_in_noisy:
                if df_abl is None:
                    df_abl = load_tb_data(ablation_registry, tb_dir, metric, no_cache=no_cache)
                if not df_abl.empty:
                    df_src = df_abl[df_abl["exp_name"] == ablation_name].copy()
                    if not df_src.empty:
                        df_src["exp_name"] = zero_exp
                        df = pd.concat([df, df_src], ignore_index=True) if not df.empty else df_src
                        print(f"  Supplemented {zero_exp} from ablation registry ({ablation_name}).")
        # Extract SAC baseline for reference hlines in the degradation curve.
        if df_abl is None:
            df_abl = load_tb_data(ablation_registry, tb_dir, metric, no_cache=no_cache)
        if not df_abl.empty:
            _sac_rows = df_abl[df_abl["exp_name"] == "sac_baseline"].copy()
            if not _sac_rows.empty:
                df_sac_baseline = deduplicate_runs(resolve_exp_name(_sac_rows))

    if df.empty:
        print(f"  No data found for metric '{metric}'. Skipping noisy expert plots.")
        return

    df = resolve_exp_name(df)
    df = deduplicate_runs(df)

    available = set(df["label"].unique())
    present = [e for e in NOISY_EXPERT_EXPERIMENTS if e in available]
    present_pcts = [pct for exp, pct in zip(NOISY_EXPERT_EXPERIMENTS, NOISY_EXPERT_NOISE_PCTS)
                    if exp in available]
    missing = [e for e in NOISY_EXPERT_EXPERIMENTS if e not in available]
    if missing:
        print(f"  Warning: missing experiments {missing}")
    if not present:
        print("  No data — skipping.")
        return

    q_dir = out_dir / "noisy_expert_degradation"
    q_dir.mkdir(exist_ok=True)
    print(f"  Saving plots to {q_dir}/")

    metric_slug = metric.replace("/", "_")
    smooth_tag = f"_smooth{smooth}" if smooth > 1 else ""
    df_q = df[df["label"].isin(present)].copy()

    title = "Expert Degradation: How does noise on the PID's target altitude affect EGE?"
    subtitle = "noise_pct ∈ {0%, 2%, 5%, 10%, 20%, 40%, 80%} of altitude range (5000 m)"

    def _q_plot_metric(df_in, **kw):
        fig = plot_metric(df_in, metric=metric, smooth=smooth,
                          n_bootstrap=n_bootstrap, ci=ci,
                          color_map=NOISY_EXPERT_COLOR_MAP,
                          labels=present,
                          **kw)
        fig.axes[0].set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
        return fig

    # Training curve — IQM, full y range
    fig = _q_plot_metric(df_q, aggregation="iqm")
    out_path = q_dir / f"{metric_slug}{smooth_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)

    # Training curve — IQM, focused
    fig = _q_plot_metric(df_q, aggregation="iqm", ylim_bottom=ylim_bottom)
    out_path = q_dir / f"{metric_slug}{smooth_tag}_focus.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)

    # Summary-stats figure — time-to-0, time-to-200, final perf (all IQM)
    fig = plot_ablation_summary_stats(
        df_q, metric=metric, experiments=present,
        title=f"{title}\n{subtitle}",
        thresholds=[0.0, 200.0, 400.0],
        last_n_steps=200_000,
        n_bootstrap=n_bootstrap, ci=ci,
        remove_censored=remove_censored,
    )
    out_path = q_dir / f"{metric_slug}{smooth_tag}_summary_stats.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)

    # 3-row comparison: EGE vs IBRL vs Residual-RL across all noise levels
    all_algo_exps = list(dict.fromkeys(
        NOISY_EXPERT_EXPERIMENTS + IBRL_NOISY_EXPERIMENTS + RESIDUAL_NOISY_EXPERIMENTS
    ))
    df_all_algos = df[df["label"].isin(all_algo_exps)].copy()
    if not df_all_algos.empty:
        fig = plot_noisy_expert_algo_rows(
            df_all_algos, metric=metric,
            thresholds=[0.0, 200.0, 400.0],
            last_n_steps=200_000,
            n_bootstrap=n_bootstrap, ci=ci,
            remove_censored=remove_censored,
            title="Expert Degradation: EGE vs IBRL vs Residual-RL\n"
                  "noise_pct ∈ {0%, 2%, 5%, 10%, 20%, 40%, 80%} of altitude range",
        )
        out_path = q_dir / f"{metric_slug}{smooth_tag}_algo_rows.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
        plt.close(fig)

        fig = plot_noisy_expert_algo_lines(
            df_all_algos, metric=metric,
            thresholds=[0.0, 200.0, 400.0],
            last_n_steps=200_000,
            n_bootstrap=n_bootstrap, ci=ci,
            remove_censored=remove_censored,
            title="Expert Degradation: EGE vs IBRL vs Residual-RL\n"
                  "noise_pct ∈ {0%, 2%, 5%, 10%, 20%, 40%, 80%} of altitude range",
        )
        out_path = q_dir / f"{metric_slug}{smooth_tag}_algo_lines.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
        plt.close(fig)

    # Degradation curve — all algorithms, all 4 summary panels vs noise level
    fig = plot_degradation_curve(
        df_all_algos if not df_all_algos.empty else df_q,
        metric=metric,
        thresholds=[0.0, 200.0, 400.0],
        last_n_steps=200_000,
        n_bootstrap=n_bootstrap, ci=ci,
        remove_censored=remove_censored,
        title="Expert Degradation: EGE vs IBRL vs Residual-RL\n"
              "noise_pct ∈ {0%, 2%, 5%, 10%, 20%, 40%, 80%} of altitude range",
        df_sac_baseline=df_sac_baseline,
    )
    out_path = q_dir / f"{metric_slug}{smooth_tag}_degradation_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)

    print("\nNoisy expert plots complete.")


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
    parser.add_argument("--stats-only", action="store_true",
                        help="Only run significance tests; skip all plot/table generation")
    parser.add_argument("--remove-censored", action="store_true",
                        help="Remove seeds that never reached a threshold instead of censoring at budget")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-read from TensorBoard event files, ignoring any parquet cache")
    parser.add_argument("--degradation", action="store_true",
                        help="Only generate noisy expert degradation plots (skips all other plots)")
    args = parser.parse_args()
    n = args.top

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tb_dir = Path(args.tb_dir)
    registry_file = Path(args.registry)

    # ------------------------------------------------------------------
    # --degradation: skip everything except noisy expert degradation plots
    # ------------------------------------------------------------------
    if args.degradation:
        if not NOISY_EXPERT_REGISTRY_FILE.exists():
            print(f"ERROR: {NOISY_EXPERT_REGISTRY_FILE} not found — run noisy_expert_study.py first.")
            return
        noisy_registry = load_registry(NOISY_EXPERT_REGISTRY_FILE)
        if not noisy_registry:
            print(f"ERROR: {NOISY_EXPERT_REGISTRY_FILE} is empty.")
            return
        ablation_registry = load_registry(ABLATION_REGISTRY_FILE) if ABLATION_REGISTRY_FILE.exists() else {}
        plot_noisy_expert_questions(
            registry=noisy_registry,
            tb_dir=tb_dir,
            out_dir=out_dir,
            ablation_registry=ablation_registry or None,
            metric="Eval/expert_bias",
            smooth=args.smooth,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
            remove_censored=args.remove_censored,
            no_cache=args.no_cache,
        )
        return

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
        _tb_metric = "Eval/expert_bias" if args.metric == OPTIMALITY_GAP_METRIC else args.metric
        df = load_tb_data(registry, tb_dir, _tb_metric, no_cache=args.no_cache)
        print(f"  {len(df)} scalar points, {df['run_id'].nunique()} runs, "
              f"{df['exp_name'].nunique()} experiments")
        if df.empty:
            print(f"  No data found for metric '{_tb_metric}'. Check tag name.")
            return
        if args.metric == OPTIMALITY_GAP_METRIC:
            df[OPTIMALITY_GAP_METRIC] = np.maximum(1.0 - df["Eval/expert_bias"] / OPTIMALITY_GAP_MAX, 0.0)
            df = df.drop(columns=["Eval/expert_bias"])
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

    # All basic metric plots go into a subfolder named after the metric slug.
    metric_dir = out_dir / metric_slug
    metric_dir.mkdir(exist_ok=True)
    print(f"\nBasic plots will be saved to {metric_dir}/")

    # For Eval/expert_bias and Eval/optimality_gap, restrict to the canonical set.
    plot_df = df
    if args.metric in {"Eval/expert_bias", OPTIMALITY_GAP_METRIC}:
        plot_df = df[df["label"].isin(EXPERT_BIAS_EXPERIMENTS)]
        dropped = set(df["label"].unique()) - set(plot_df["label"].unique())
        if dropped:
            print(f"  expert_bias filter: dropped {dropped}")

    if not args.stats_only:
        print(f"\nPlotting {args.metric} — all groups...")
        fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth, n_bootstrap=args.n_bootstrap, ci=args.ci)
        out_path = metric_dir / f"{metric_slug}{smooth_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)

        print(f"\nPlotting {args.metric} — all groups (focus ≥ -250)...")
        fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth, n_bootstrap=args.n_bootstrap, ci=args.ci, ylim_bottom=-250, show_crossing=False)
        out_path = metric_dir / f"{metric_slug}{smooth_tag}_focus.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)

    # Rank by peak (max) rather than asymptote for expert_bias / optimality_gap.
    if args.metric in {"Eval/expert_bias", OPTIMALITY_GAP_METRIC}:
        topn_labels = ["sac_baseline", "residual_rl", "ibrl_style", MAIN_EXP]
        topn_labels = [l for l in topn_labels if l in plot_df["label"].unique()]
    else:
        topn_labels = top_labels_by_asymptote(plot_df, metric=args.metric, last_n_steps=200_000, n=n)

    # IQM and median variants — only for Eval/expert_bias and optimality_gap.
    if args.metric in {"Eval/expert_bias", OPTIMALITY_GAP_METRIC}:
        if not args.stats_only:
            print(f"  Labels for ranked plot: {topn_labels}")
            for agg in ("iqm", "median"):
                print(f"\nPlotting {args.metric} — {agg.upper()}, all groups...")
                fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth,
                                  n_bootstrap=args.n_bootstrap, ci=args.ci,
                                  aggregation=agg)
                out_path = metric_dir / f"{metric_slug}{smooth_tag}_{agg}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Saved → {out_path}")
                plt.close(fig)

                print(f"\nPlotting {args.metric} — {agg.upper()}, focus ≥ -250...")
                fig = plot_metric(plot_df, metric=args.metric, smooth=args.smooth,
                                  n_bootstrap=args.n_bootstrap, ci=args.ci,
                                  ylim_bottom=-250, show_crossing=False,
                                  aggregation=agg)
                out_path = metric_dir / f"{metric_slug}{smooth_tag}_{agg}_focus.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Saved → {out_path}")
                plt.close(fig)

            print(f"\nPlotting {args.metric} — final distribution histogram...")
            fig = plot_final_distribution(plot_df, metric=args.metric, last_n_steps=100_000)
            out_path = metric_dir / f"{metric_slug}{smooth_tag}_dist.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
            plt.close(fig)

            _metric_thresholds = METRIC_THRESHOLDS.get(args.metric, [0.0, 200.0, 400.0])
            _summary_title = (
                "Optimality Gap — Method Comparison"
                if args.metric == OPTIMALITY_GAP_METRIC
                else "Expert Advantage — Method Comparison"
            )
            print(f"\nPlotting {args.metric} — EGE vs IBRL vs SAC summary (peak IQM)...")
            fig = plot_expert_bias_summary(
                plot_df, metric=args.metric,
                experiments=["sac_baseline", "residual_rl", "ibrl_style", MAIN_EXP],
                title=_summary_title,
                thresholds=_metric_thresholds,
                n_bootstrap=args.n_bootstrap, ci=args.ci,
                remove_censored=args.remove_censored,
            )
            out_path = metric_dir / f"{metric_slug}{smooth_tag}_method_summary.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
            plt.close(fig)

            print(f"\nGenerating {args.metric} — method summary LaTeX table...")
            tex = make_latex_method_table(
                plot_df, metric=args.metric,
                experiments=["sac_baseline", "residual_rl", "ibrl_style", MAIN_EXP],
                thresholds=_metric_thresholds,
                n_bootstrap=args.n_bootstrap, ci=args.ci,
                remove_censored=args.remove_censored,
            )
            out_path = metric_dir / f"{metric_slug}{smooth_tag}_method_table.tex"
            out_path.write_text(tex)
            print(f"Saved → {out_path}")

        print(f"\nRunning pairwise significance tests on {args.metric} (last 200k steps)...")
        pairwise_significance_expert_bias(
            plot_df, metric=args.metric,
            experiments=["sac_baseline", "residual_rl", "ibrl_style", MAIN_EXP],
            thresholds=METRIC_THRESHOLDS.get(args.metric, [0.0, 200.0, 400.0]),
            remove_censored=args.remove_censored,
        )
    else:
        if not args.stats_only:
            print(f"\nPlotting {args.metric} — asymptote boxplot (top {n})...")
            fig = plot_asymptote_boxplot(
                df, metric=args.metric, labels=topn_labels,
                last_n_steps=200_000,
                n_bootstrap=args.n_bootstrap, ci=args.ci,
            )
            out_path = metric_dir / f"{metric_slug}_boxplot_top{n}.png"
            fig.savefig(out_path, dpi=150)
            print(f"Saved → {out_path}")
            plt.close(fig)

    if not args.stats_only:
        # ------------------------------------------------------------------
        # Ablation question plots — one figure per question.
        # Only available in TB mode (requires registry + ablation_run_registry.json).
        # ------------------------------------------------------------------
        ablation_registry_file = Path(args.registry)
        ablation_registry = {}
        if args.csv is None and ablation_registry_file.exists():
            ablation_registry = load_registry(ablation_registry_file)
            if ablation_registry:
                plot_ablation_questions(
                    registry=ablation_registry,
                    tb_dir=tb_dir,
                    out_dir=out_dir,
                    metric="Eval/expert_bias",
                    smooth=args.smooth,
                    n_bootstrap=args.n_bootstrap,
                    ci=args.ci,
                    remove_censored=args.remove_censored,
                    no_cache=args.no_cache,
                )

        # ------------------------------------------------------------------
        # Noisy expert degradation plots — triggered automatically when
        # noisy_expert_run_registry.json exists next to the script.
        # ------------------------------------------------------------------
        if args.csv is None and NOISY_EXPERT_REGISTRY_FILE.exists():
            noisy_registry = load_registry(NOISY_EXPERT_REGISTRY_FILE)
            if noisy_registry:
                plot_noisy_expert_questions(
                    registry=noisy_registry,
                    tb_dir=tb_dir,
                    out_dir=out_dir,
                    ablation_registry=ablation_registry or None,
                    metric="Eval/expert_bias",
                    smooth=args.smooth,
                    n_bootstrap=args.n_bootstrap,
                    ci=args.ci,
                    remove_censored=args.remove_censored,
                    no_cache=args.no_cache,
                )

        # Summary plots — one figure per metric, shared colour map for consistency.
        # Only available in TB mode (requires registry + tb_dir).
        if args.csv is None and registry_file.exists():
            print("\nPlotting summary (5 key metrics, all experiments)...")

            # Build a shared colour map from all labels across all summary metrics
            # so that the same experiment always gets the same colour.
            all_summary_labels: set[str] = set()
            summary_dfs: dict[str, pd.DataFrame] = {}
            for s_metric in SUMMARY_METRICS:
                df_s = load_tb_data(registry, tb_dir, s_metric, no_cache=args.no_cache)
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
