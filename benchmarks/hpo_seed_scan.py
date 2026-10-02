"""Extract per-trial wall-clock from HPO result JSON files.

For each trial_NNN_result.json under hp_results/<env>/<method>/, reads
elapsed_s, n_seeds, phase. Cross-references the phase to a static
n_timesteps mapping (300k / 1M / 1M). Computes per-seed and
per-step-per-seed cost so we can see whether scaling stays sublinear
across the {20, 50, 100} seed boundaries the user actually used.

Usage:
    python benchmarks/hpo_seed_scan.py /path/to/hp_results
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

PHASE_TIMESTEPS = {1: 300_000, 2: 1_000_000, 3: 1_000_000}


def _scan(hp_results_root: str):
    rows = []
    for env in sorted(os.listdir(hp_results_root)):
        env_dir = os.path.join(hp_results_root, env)
        if not os.path.isdir(env_dir):
            continue
        for method in sorted(os.listdir(env_dir)):
            m_dir = os.path.join(env_dir, method)
            if not os.path.isdir(m_dir):
                continue
            for fn in os.listdir(m_dir):
                if not fn.endswith("_result.json"):
                    continue
                path = os.path.join(m_dir, fn)
                try:
                    with open(path) as f:
                        d = json.load(f)
                except Exception:
                    continue
                phase = d.get("phase")
                n_seeds = d.get("n_seeds")
                elapsed = d.get("elapsed_s")
                if phase is None or n_seeds is None or elapsed is None:
                    continue
                # Skip degenerate trials (pruned / errored fast).
                if elapsed < 5:
                    continue
                rows.append(
                    {
                        "env": env,
                        "method": method,
                        "phase": int(phase),
                        "n_seeds": int(n_seeds),
                        "elapsed_s": float(elapsed),
                        "trial": fn.replace("_result.json", ""),
                    }
                )
    return rows


def _summarise(rows):
    by_key: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["env"], r["method"], r["phase"], r["n_seeds"])
        by_key[key].append(r["elapsed_s"])

    out = []
    for (env, method, phase, n), elapsed in sorted(by_key.items()):
        elapsed_sorted = sorted(elapsed)
        med = elapsed_sorted[len(elapsed_sorted) // 2]
        n_steps = PHASE_TIMESTEPS.get(phase, 0)
        time_per_seed = med / n
        (med / n) / (n_steps / 1e3) if n_steps else 0
        out.append(
            {
                "env": env,
                "method": method,
                "phase": phase,
                "n_seeds": n,
                "n_trials": len(elapsed),
                "median_s": med,
                "time_per_seed_s": time_per_seed,
                "us_per_seed_per_step": (med / n) / max(n_steps, 1) * 1e6,
            }
        )
    return out


def main(root: str):
    rows = _scan(root)
    if not rows:
        print(f"no result.json files under {root}")
        return 1
    summary = _summarise(rows)

    # Print all rows.
    print(
        f"{'env':<22} {'method':<22} {'P':>1} {'N':>4} {'#':>3} "
        f"{'median_s':>9} {'s/seed':>7} {'us/seed/step':>13}"
    )
    print("-" * 90)
    for r in summary:
        print(
            f"{r['env']:<22} {r['method']:<22} "
            f"{r['phase']:>1} {r['n_seeds']:>4} {r['n_trials']:>3} "
            f"{r['median_s']:>9.1f} {r['time_per_seed_s']:>7.2f} "
            f"{r['us_per_seed_per_step']:>13.3f}"
        )

    # Cross-N comparison: per (env, method), show ratios across phases.
    print()
    print("Per-(env, method) per-seed-per-step ratios (lower is better):")
    print(
        f"{'env':<22} {'method':<22} "
        f"{'P1@20us/s/step':>15} {'P2@50us/s/step':>15} {'P3@100us/s/step':>16} "
        f"{'P2/P1':>6} {'P3/P1':>6}"
    )
    print("-" * 110)
    by_em: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in summary:
        by_em[(r["env"], r["method"])][r["phase"]] = r["us_per_seed_per_step"]
    for (env, method), pmap in sorted(by_em.items()):
        p1 = pmap.get(1, 0.0)
        p2 = pmap.get(2, 0.0)
        p3 = pmap.get(3, 0.0)
        r21 = (p2 / p1) if p1 else 0.0
        r31 = (p3 / p1) if p1 else 0.0
        print(
            f"{env:<22} {method:<22} "
            f"{p1:>15.3f} {p2:>15.3f} {p3:>16.3f} "
            f"{r21:>6.2f} {r31:>6.2f}"
        )


if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1]
            if len(sys.argv) > 1
            else "/home/yberthel/AjaxExperiments/hp_results"
        )
        or 0
    )
