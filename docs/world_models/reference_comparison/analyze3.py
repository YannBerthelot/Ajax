"""Round 3 analysis, exactly as pre-registered in PREREG3.md (written before the run).

usage: python analyze3.py [--root results/round3] [--out DIR] [--seeds 0 1 2 3 4]

Reads <root>/ref_s<seed>/evals.jsonl (reference) and <root>/ajax/s<seed>/evals.jsonl
(Ajax), or the same files gzipped (.jsonl.gz), one line per evaluation:
{"rows": N, "mean": eval mean return, ...}.
Checks every expected checkpoint (400, 800, ..., 24000) exists once per seed and
side, and that every DONE marker holds 0, then prints:
  LATE  = per-seed mean eval over checkpoints in [12000, 24000] (31 points), rule
          LEARNS / WITHIN_RANGE, Welch t-test (two-sided);
  S1    = per-seed mean eval at 16000, 18000, 20000, same rule;
  concordance at 20000 and 24000 (runs with eval >= 400), and each side's rate
          of eval >= 400 over the checkpoints in [16000, 24000];
  the H1 / H3 / H2 reading of PREREG3.md;
and writes analysis3.md and analysis3.json to --out (default: next to the data).
The "Root" line names --root relative to the repository root, else by its
directory name (provenance.shown).
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import provenance

ROWS, EVERY = 24000, 400
CHECKPOINTS = list(range(EVERY, ROWS + 1, EVERY))
LATE = [r for r in CHECKPOINTS if 12000 <= r <= 24000]
S1 = [16000, 18000, 20000]
CONC = [r for r in CHECKPOINTS if 16000 <= r <= 24000]
GOOD = 400.0


def load(path: pathlib.Path) -> dict[int, float]:
    curve: dict[int, float] = {}
    gz = path.with_name(path.name + ".gz")
    if not path.exists() and gz.exists():  # committed results: the same file, gzipped
        text = gzip.decompress(gz.read_bytes()).decode()
    else:
        text = path.read_text()
    for line in text.splitlines():
        rec = json.loads(line)
        rows = int(rec["rows"])
        if rows in curve:
            raise SystemExit(f"{path}: duplicate checkpoint {rows}")
        curve[rows] = float(rec["mean"])
    missing = [r for r in CHECKPOINTS if r not in curve]
    if missing:
        raise SystemExit(f"{path}: missing checkpoints {missing[:5]}...")
    return curve


def rule(name: str, ref: np.ndarray, ajax: np.ndarray) -> list[str]:
    a = float(ajax.mean())
    lo, hi = float(ref.min()), float(ref.max())
    t = stats.ttest_ind(ref, ajax, equal_var=False)
    return [
        f"### {name}",
        "",
        "| seed | " + " | ".join(str(i) for i in range(len(ref))) + " | mean |",
        "|---|" + "---|" * (len(ref) + 1),
        "| reference | "
        + " | ".join(f"{x:.1f}" for x in ref)
        + f" | {ref.mean():.1f} |",
        "| Ajax | " + " | ".join(f"{x:.1f}" for x in ajax) + f" | {a:.1f} |",
        "",
        f"* LEARNS = {a >= lo} (Ajax mean {a:.1f} >= reference min {lo:.1f})",
        f"* WITHIN_RANGE = {lo <= a <= hi} (reference range [{lo:.1f}, {hi:.1f}])",
        f"* Welch t = {t.statistic:.3f}, two-sided p = {t.pvalue:.4f}"
        " (positive t: reference higher)",
        "",
    ]


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here / "results" / "round3"))
    ap.add_argument("--out", help="where the outputs go (default: --root)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    out = pathlib.Path(a.out) if a.out else root
    out.mkdir(parents=True, exist_ok=True)

    bad = []
    for name in [f"ref_s{s}" for s in a.seeds] + ["ajax"]:
        p = root / f"{name}.DONE"
        if not p.exists() or p.read_text().strip() != "0":
            bad.append(f"{p}: {p.read_text().strip() if p.exists() else 'missing'}")
    if bad:
        raise SystemExit("DONE markers not all 0: " + "; ".join(bad))

    ref = {s: load(root / f"ref_s{s}" / "evals.jsonl") for s in a.seeds}
    ajx = {s: load(root / "ajax" / f"s{s}" / "evals.jsonl") for s in a.seeds}

    def per_seed(curves, points):
        return np.array([np.mean([curves[s][r] for r in points]) for s in a.seeds])

    late_r, late_a = per_seed(ref, LATE), per_seed(ajx, LATE)
    s1_r, s1_a = per_seed(ref, S1), per_seed(ajx, S1)

    def good_at(curves, r):
        return sum(curves[s][r] >= GOOD for s in a.seeds)

    def good_rate(curves):
        vals = [curves[s][r] >= GOOD for s in a.seeds for r in CONC]
        return float(np.mean(vals))

    n = len(a.seeds)
    conc = {
        "ref_20000": good_at(ref, 20000),
        "ajax_20000": good_at(ajx, 20000),
        "ref_24000": good_at(ref, 24000),
        "ajax_24000": good_at(ajx, 24000),
        "ref_rate_16_24k": good_rate(ref),
        "ajax_rate_16_24k": good_rate(ajx),
    }
    p_late = float(stats.ttest_ind(late_r, late_a, equal_var=False).pvalue)
    h1 = conc["ref_24000"] >= 4 and conc["ref_rate_16_24k"] <= 0.4  # PREREG3 amendment
    h3 = p_late < 0.05 and late_r.mean() > late_a.mean()
    ajax_better = p_late < 0.05 and late_a.mean() > late_r.mean()
    reading = (
        "H1 (end-of-run artifact on the reference side)"
        if h1
        else "H3 (real late-learning difference, reference better)"
        if h3
        else "Ajax significantly better (a difference to investigate)"
        if ajax_better
        else "H2 or noise (no reference concordance beyond its late rate,"
        " no significant LATE difference)"
    )
    if h1 and h3:
        reading = "H1 and H3 both met (report both)"

    shown = provenance.shown(root, here.parents[2])  # no absolute path
    lines = [
        "# Round 3 analysis (PREREG3.md)",
        "",
        f"Root `{shown}`; seeds {a.seeds}; {len(CHECKPOINTS)} checkpoints per run"
        f" ({EVERY} to {ROWS} rows); LATE = {len(LATE)} checkpoints in [12000, 24000].",
        "",
    ]
    lines += rule("LATE (primary)", late_r, late_a)
    lines += rule("S1 (16000, 18000, 20000)", s1_r, s1_a)
    lines += [
        "### Concordance (eval >= 400)",
        "",
        f"* at 20000: reference {conc['ref_20000']}/{n}, Ajax {conc['ajax_20000']}/{n}",
        f"* at 24000: reference {conc['ref_24000']}/{n}, Ajax {conc['ajax_24000']}/{n}",
        f"* rate over [16000, 24000]: reference {conc['ref_rate_16_24k']:.3f},"
        f" Ajax {conc['ajax_rate_16_24k']:.3f}",
        "",
        f"## Reading: {reading}",
        "",
        "## Curves (eval mean return; every 2000 rows shown, all points used above)",
        "",
        "| rows | "
        + " | ".join(f"ref s{s}" for s in a.seeds)
        + " | "
        + " | ".join(f"ajax s{s}" for s in a.seeds)
        + " |",
        "|---|" + "---|" * (2 * n),
    ]
    for r in range(2000, ROWS + 1, 2000):
        lines.append(
            f"| {r} | "
            + " | ".join(f"{ref[s][r]:.0f}" for s in a.seeds)
            + " | "
            + " | ".join(f"{ajx[s][r]:.0f}" for s in a.seeds)
            + " |"
        )
    text = "\n".join(lines) + "\n"
    (out / "analysis3.md").write_text(text)
    json.dump(
        {
            "late_ref": late_r.tolist(),
            "late_ajax": late_a.tolist(),
            "s1_ref": s1_r.tolist(),
            "s1_ajax": s1_a.tolist(),
            "p_late": p_late,
            "concordance": conc,
            "reading": reading,
        },
        open(out / "analysis3.json", "w"),
        indent=1,
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
