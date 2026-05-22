"""Reads seed_scaling.jsonl and prints a scaling table.

Time-per-seed = measured_s / n_seeds. Ideal vmap parallelism keeps
measured_s flat as n_seeds grows (so time-per-seed drops as 1/n_seeds).
Saturation appears as measured_s climbing.

Usage:
    python benchmarks/seed_scaling_report.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


def main(path: str | None = None):
    path = path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "seed_scaling.jsonl"
    )
    by_n: dict[int, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_n[r["n_seeds"]].append(r)

    if not by_n:
        print(f"no rows in {path}")
        return 1

    rows = []
    for n in sorted(by_n.keys()):
        trials = by_n[n]
        measured = sorted(t["measured_s"] for t in trials)
        peaks = sorted(t["peak_bytes_in_use"] for t in trials)
        median_t = measured[len(measured) // 2]
        min_t = measured[0]
        median_peak = peaks[len(peaks) // 2]
        rows.append({
            "n": n,
            "median_s": median_t,
            "min_s": min_t,
            "peak_mb": median_peak / (1024 * 1024),
        })

    base_median = rows[0]["median_s"]
    base_peak = rows[0]["peak_mb"]
    base_n = rows[0]["n"]

    print(f"Per-seed scaling (baseline: n_seeds={base_n})")
    print(
        f"{'n':>3} | {'median_s':>9} | {'min_s':>8} | {'speedup':>8} | "
        f"{'time/seed':>10} | {'peak_mb':>9} | {'mem/seed_mb':>11}"
    )
    print("-" * 80)
    for r in rows:
        speedup = (r["median_s"] * base_n) / (base_median * r["n"])
        time_per_seed = r["median_s"] / r["n"]
        mem_per_seed = r["peak_mb"] / r["n"]
        print(
            f"{r['n']:>3} | {r['median_s']:>9.3f} | {r['min_s']:>8.3f} | "
            f"{speedup:>7.2f}x | {time_per_seed:>10.3f} | "
            f"{r['peak_mb']:>9.1f} | {mem_per_seed:>11.2f}"
        )
    print()
    print(
        "speedup: relative throughput vs n=1 (1.0 = identical wall clock per seed; "
        "ideal scales with n; saturation drives it back toward 1)"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None) or 0)
