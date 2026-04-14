"""
Compute cumulative compute time for ablation and degradation study runs.

Reads ablation_run_registry.json and noisy_expert_run_registry.json,
applies the same deduplication logic as plot_sweep.py (keeping the most
recent batch per experiment — typically the last ~100 seeds), then sums
wall-clock durations across three categories:

  • Main results  : sac_baseline, ege_decay_050, ibrl_style, residual_rl
  • Ablation study: all other experiments in ablation_run_registry.json
  • Degradation   : all experiments in noisy_expert_run_registry.json

Run duration is derived without reading every TensorBoard event:
  start ≈ timestamp embedded in the event filename
  end   ≈ mtime of the event file  (matches last wall_time within ~10 s)
"""

import json
import struct
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors plot_sweep.py
# ---------------------------------------------------------------------------

TB_DIR = Path("tensorboard")
ABLATION_REGISTRY_FILE = Path("ablation_run_registry.json")
NOISY_EXPERT_REGISTRY_FILE = Path("noisy_expert_run_registry.json")

# Experiments that count as "main results" (from the ablation registry).
MAIN_EXPERIMENTS = {"sac_baseline", "ege_decay_050", "ibrl_style", "residual_rl"}

# Batch gap used in deduplicate_runs (seconds).
_BATCH_GAP_SECONDS = 300


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [warn] Registry not found: {path}")
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Deduplication — keep the most recent batch of runs per experiment
# (same logic as deduplicate_runs() in plot_sweep.py).
# ---------------------------------------------------------------------------

def keep_latest_batch(runs: list[dict]) -> list[dict]:
    """
    Given a list of run dicts (each with 'run_id' and 'exp_name'), keep only
    the most recent time-cluster of runs per exp_name.

    Cluster boundary = gap > _BATCH_GAP_SECONDS between consecutive run
    folder mtimes.
    """
    # Group by exp_name
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        # Attach folder mtime so we can sort.
        run_dir = TB_DIR / r["run_id"]
        if not run_dir.exists():
            continue
        r = dict(r, _mtime=run_dir.stat().st_mtime)
        by_exp[r["exp_name"]].append(r)

    kept: list[dict] = []
    for exp_name, exp_runs in by_exp.items():
        exp_runs.sort(key=lambda x: x["_mtime"])

        # Cluster into batches.
        batches: list[list[dict]] = []
        batch = [exp_runs[0]]
        for prev, curr in zip(exp_runs, exp_runs[1:]):
            if curr["_mtime"] - prev["_mtime"] < _BATCH_GAP_SECONDS:
                batch.append(curr)
            else:
                batches.append(batch)
                batch = [curr]
        batches.append(batch)

        latest = batches[-1]
        if len(batches) > 1:
            dropped = sum(len(b) for b in batches[:-1])
            print(f"  {exp_name:40s}  keep {len(latest):3d} seeds  (dropped {dropped} older)")
        kept.extend(latest)

    return kept


# ---------------------------------------------------------------------------
# Duration estimation from the event file
# ---------------------------------------------------------------------------

def _event_file_duration(run_id: str) -> float | None:
    """
    Return wall-clock duration (seconds) for one run using:
      start = timestamp in the event filename
      end   = mtime of the event file

    Returns None if no event file exists.
    """
    run_dir = TB_DIR / run_id
    if not run_dir.exists():
        return None
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    total = 0.0
    for ef in event_files:
        parts = ef.name.split(".")
        # Filename: events.out.tfevents.<timestamp>.<host>
        try:
            start = float(parts[3])
        except (IndexError, ValueError):
            continue
        end = ef.stat().st_mtime
        duration = end - start
        if duration > 0:
            total += duration

    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m  ({seconds/3600:.1f} GPU-hours)"


def process_registry(
    registry_file: Path,
    label: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Load registry, deduplicate, compute per-experiment durations.
    Returns (durations_by_exp, seed_count_by_exp).
    """
    print(f"\n{'='*60}")
    print(f"Registry: {registry_file}  [{label}]")
    print(f"{'='*60}")

    runs = load_registry(registry_file)
    if not runs:
        return {}, {}

    print(f"  {len(runs)} total entries")
    print("\nDeduplicating (keeping latest batch per experiment)...")
    kept = keep_latest_batch(runs)
    print(f"\n  → {len(kept)} runs after dedup")

    durations: dict[str, float] = defaultdict(float)
    seed_counts: dict[str, int] = defaultdict(int)
    missing = 0

    for r in kept:
        d = _event_file_duration(r["run_id"])
        if d is None:
            missing += 1
            continue
        exp = r["exp_name"]
        durations[exp] += d
        seed_counts[exp] += 1

    if missing:
        print(f"  [warn] {missing} runs had no event file — excluded from compute time")

    return dict(durations), dict(seed_counts)


def main():
    # -----------------------------------------------------------------------
    # Ablation registry → split into main results + ablation study
    # -----------------------------------------------------------------------
    abl_durations, abl_counts = process_registry(ABLATION_REGISTRY_FILE, "ablation registry")

    main_durations: dict[str, float] = {}
    ablation_durations: dict[str, float] = {}
    for exp, dur in abl_durations.items():
        if exp in MAIN_EXPERIMENTS:
            main_durations[exp] = dur
        else:
            ablation_durations[exp] = dur

    main_counts = {e: abl_counts[e] for e in main_durations}
    ablation_counts = {e: abl_counts[e] for e in ablation_durations}

    # -----------------------------------------------------------------------
    # Noisy expert registry → degradation study
    # -----------------------------------------------------------------------
    deg_durations, deg_counts = process_registry(NOISY_EXPERT_REGISTRY_FILE, "noisy expert registry")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    def print_section(title: str, durations: dict[str, float], counts: dict[str, int]):
        total = sum(durations.values())
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
        col = max((len(e) for e in durations), default=10)
        for exp in sorted(durations):
            h = durations[exp] / 3600
            n = counts.get(exp, 0)
            print(f"  {exp:{col}s}  {n:4d} seeds   {h:7.1f} GPU-h")
        print(f"  {'TOTAL':{col}s}  {sum(counts.values()):4d} seeds   {total/3600:7.1f} GPU-h  →  {fmt_time(total)}")
        return total

    print("\n" + "="*60)
    print("  CUMULATIVE COMPUTE TIME SUMMARY")
    print("="*60)

    t_main = print_section("MAIN RESULTS  (sac_baseline, ege_decay_050, ibrl_style, residual_rl)", main_durations, main_counts)
    t_abl  = print_section("ABLATION STUDY", ablation_durations, ablation_counts)
    t_deg  = print_section("DEGRADATION STUDY  (noisy expert)", deg_durations, deg_counts)

    grand_total = t_main + t_abl + t_deg
    print(f"\n{'='*60}")
    print(f"  GRAND TOTAL   {fmt_time(grand_total)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
