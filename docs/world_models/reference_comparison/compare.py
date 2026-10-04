"""Compare Ajax DreamerV3 with the reference DreamerV3 on CartPole-v1 (PROTOCOL.md 5.2-5.4).

Usage (Ajax venv: numpy, matplotlib, PyYAML; ``HERE`` is this directory):

    $APY $HERE/compare.py --root OUT [--seeds 0 1 2]               # full protocol
    $APY $HERE/compare.py --root OUT --seeds 0 1 --smoke           # smoke outputs
    $APY $HERE/compare.py --ref-root R --ajax-root A --out O       # a cross pair

Inputs: ``ROOT/ref_s{s}/`` (run_reference.py) and ``ROOT/ajax/`` (run_ajax.py),
with their ``.DONE`` markers next to them; ``--ref-root`` / ``--ajax-root`` take
the two sides from different runs (round 2's pairs E and R, README.md).
Every ``.jsonl`` input may be stored gzipped as ``.jsonl.gz`` (the committed
results under ``results/``). Outputs: printed tables, ``OUT/compare.md``,
``OUT/summary.json``, ``OUT/curves.png`` (``--out`` defaults to ``--root``).

Provenance checks (README.md "Recorded locations"): the reference run must
have imported the reference code from its reference checkout (recorded files
``<reference-checkout>/...``) at commit 29eb964 without uncommitted changes,
and the Ajax run Ajax from its configured src (recorded file
``<ajax-src>/...``) at the expected commit (``--ajax-commit``, default: HEAD
of ``--ajax-src``, itself defaulting to ``$AJAX_SRC``, else the ``src``
directory of the repository holding this script), src without uncommitted
changes. The commit and dirty flag are read from the run's own record
(``run_args.json`` / ``ajax/config.json``); records from before the runners
wrote them (the committed rounds 1-3) take them from the ``launch_info.txt``
that ``launch.sh`` wrote next to the run, where the reference's dirty flag was
not recorded. ``reproduce_tables.sh`` passes the Ajax commit the committed
results ran.

Pre-registered statistics (fixed before any full result existed; do not change):
* S1, per seed: mean of the evaluation means logged at rows 16000, 18000 and
  20000 (3 evaluations x 10 episodes).
* S2, per seed, at 20000 rows: Ajax's ``Train/episodic mean reward``
  definition: per env, the mean score of its last min(10, k) finished
  training episodes whose is_last row index (0-based, per env) is <= N / 16;
  the mean over the 16 envs; NaN while any env has no finished episode.
  Ajax: the logged value (audited against the final rolling buffer);
  reference: computed from episodes.jsonl.
* Rule (printed; the decision is the user's): A = mean over Ajax seeds of S1,
  [lo, hi] = [min, max] over reference seeds of S1; LEARNS = A >= lo,
  WITHIN_RANGE = lo <= A <= hi.

Checks (exit 1 if any fails): DONE markers with exit code 0 (launch.sh; full
mode: required), the reference config.yaml and run_args.json, the Ajax
config.json (widths, every DreamerV3Config value, batch / ratio / replay,
schedule, seeds, run arguments), every evaluation's episode count, the
reference episode log (consistency, tiling) and that the reference run
finished with its flush step, 10 records at rows 2000 k, the update counts
on both sides, and the Ajax S2 audit.

``--smoke``: checkpoints are whatever the runs logged; the S1 window is the
last three logged checkpoints; the protocol-value assertions (10 records,
9481 updates, full-size widths, 20000 rows, 10 eval episodes, DONE markers)
are replaced by consistency checks against each run's own arguments.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pathlib
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import namemap
import provenance

N_ENVS = 16
GATE = 1040
S1_ROWS = (16000, 18000, 20000)
FULL_ROWS = [2000 * k for k in range(1, 11)]
FULL_RUN = {"rows": 20000, "every": 2000, "eval_episodes": 10}
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_AJAX_SRC = os.environ.get("AJAX_SRC", str(REPO / "src"))
REF_COMMIT = "29eb964e2918a3f4db04086f7f51b60388e97f3d"  # danijar/dreamerv3
EMPTY_SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"  # sha1 of an empty diff

# Ajax resolved DreamerV3Config of DreamerV3("CartPole-v1", model_size="1m")
# with every other argument at its default (PROTOCOL.md section 1; audited
# field by field against the reference config in a one-off audit in the
# session scratch directory, refrun/audit_config, not committed; README.md). A
# --tiny smoke run replaces the widths by AJAX_TINY_WIDTHS (run_ajax.TINY).
AJAX_DREAMER_CONFIG = {
    "units": 64,
    "hidden": 64,
    "deter": 512,
    "stoch": 32,
    "classes": 4,
    "blocks": 8,
    "enc_layers": 3,
    "dec_layers": 3,
    "rew_layers": 1,
    "con_layers": 1,
    "bins": 255,
    "unimix": 0.01,
    "free_nats": 1.0,
    "rec_scale": 1.0,
    "rew_scale": 1.0,
    "con_scale": 1.0,
    "dyn_scale": 1.0,
    "rep_scale": 0.1,
    "return_horizon": 333.0,
    "actor_layers": 3,
    "critic_layers": 3,
    "actor_unimix": 0.01,
    "minstd": 0.1,
    "maxstd": 1.0,
    "imag_horizon": 15,
    "lam": 0.95,
    "repval_lam": 0.95,
    "actent": 0.0003,
    "slowreg": 1.0,
    "slow_rate": 0.02,
    "retnorm_rate": 0.01,
    "retnorm_limit": 1.0,
    "actor_scale": 1.0,
    "critic_scale": 1.0,
    "repval_scale": 0.3,
    "learning_rate": 4e-05,
    "agc": 0.3,
    "agc_pmin": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-20,
    "warmup": 1000,
}
AJAX_TINY_WIDTHS = {"units": 16, "hidden": 16, "deter": 64, "classes": 4}
AJAX_AGENT_CONFIG = {
    "train_ratio": 512,
    "batch_size": 16,
    "batch_length": 64,
    "replay_capacity": 5_000_000,
}


def expected_updates(n: int) -> int:
    """Updates once ``n`` rows are in: gate at 1040, then one per 2 rows."""
    return 0 if n < GATE else 1 + (n - GATE) // 2


def nanmean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def read_jsonl(path):
    path = pathlib.Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        # committed results: the same file, gzipped
        with gzip.open(path.with_name(path.name + ".gz"), "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- S2
def s2_from_episodes(episodes, n_rows, n_envs=N_ENVS, window=10):
    row_cut = n_rows // n_envs
    per = {w: [] for w in range(n_envs)}
    for e in sorted(episodes, key=lambda e: (e["worker"], e["row_last"])):
        if e["row_last"] <= row_cut:
            per[e["worker"]].append(e["score"])
    if any(len(v) == 0 for v in per.values()):
        return float("nan")
    return float(np.mean([np.mean(v[-window:]) for v in per.values()]))


# ---------------------------------------------------------------- checks
class Checks:
    def __init__(self):
        self.lines = []
        self.failed = 0

    def __call__(self, ok, msg):
        self.lines.append(("PASS" if ok else "FAIL") + "  " + msg)
        self.failed += not ok


def check_episodes(chk, seed, eps, n_rows):
    bad = [
        e
        for e in eps
        if e["length"] != e["score"] or e["length"] > 500 or e["length"] < 1
    ]
    chk(
        not bad,
        f"ref s{seed}: episodes length == score, 1 <= length <= 500 ({len(eps)} episodes)",
    )
    bad = [e for e in eps if (not e["terminal"]) and e["length"] != 500]
    chk(not bad, f"ref s{seed}: terminal False only for 500-step episodes")
    tiles = True
    for w in range(N_ENVS):
        prev_last = -1
        for e in sorted(
            (e for e in eps if e["worker"] == w), key=lambda e: e["row_last"]
        ):
            # reset row at prev_last + 1, then length rows: is_last at prev_last + 1 + length
            if e["row_last"] != prev_last + 1 + e["length"]:
                tiles = False
            prev_last = e["row_last"]
        if prev_last > n_rows // N_ENVS:
            tiles = False
    chk(
        tiles,
        f"ref s{seed}: per worker the episodes tile rows 0..row_last with L + 1 rows each",
    )


def check_ref_config(chk, seed, cfg, run_args):
    tiny = bool(run_args.get("tiny"))
    units = 16 if tiny else 64
    want = {
        "seed": seed,
        "task": "gymnax_cartpole",
        "jax.platform": "cpu",
        "jax.compute_dtype": "float32",
        "jax.param_dtype": "float32",
        "jax.prealloc": False,
        "jax.transfer_guard": False,
        "dyn.rssm.deter": 64 if tiny else 512,
        "dyn.rssm.hidden": 16 if tiny else 64,
        "dyn.rssm.classes": 4,
        "dyn.rssm.stoch": 32,
        "dyn.rssm.blocks": 8,
        "enc.simple.units": units,
        "dec.simple.units": units,
        "rewhead.units": units,
        "conhead.units": units,
        "actor.units": units,
        "critic.units": units,
        "run.train_ratio": 512.0,
        "run.steps": run_args["rows"],
        "run.num_envs": 16,
        "run.log_every": -1,
        "run.eval_every": -1,
        "run.save_every": -1,
        "run.driver_parallel": False,
        "run.train_fill": 0,
        "batch_size": 16,
        "batch_length": 65,
        "replay_context": 1,
        "replay_length": 65,
        "replay.size": 5e6,
        "replay.online": True,
        "opt.lr": 4e-5,
        "opt.warmup": 1000,
        "horizon": 333,
        "imag_length": 15,
        "contdisc": True,
    }
    bad = []
    for k, v in want.items():
        node = cfg
        for p in k.split("."):
            node = node[p]
        if (
            (float(node) != float(v))
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            else (node != v)
        ):
            bad.append(f"{k}={node!r} (want {v!r})")
    chk(
        not bad,
        f"ref s{seed}: config.yaml has the protocol values{' (tiny)' if tiny else ''}"
        + (f": {bad}" if bad else ""),
    )


def _diff(got: dict, want: dict) -> list:
    """Mismatches between two flat dicts (numbers compared as floats)."""
    bad = []
    for k in sorted(set(got) | set(want)):
        if k not in got or k not in want:
            bad.append(
                f"{k}: {got.get(k, '<missing>')!r} (want {want.get(k, '<absent>')!r})"
            )
            continue
        g, w = got[k], want[k]
        num = isinstance(w, (int, float)) and not isinstance(w, bool)
        if (float(g) != float(w)) if num else (g != w):
            bad.append(f"{k}={g!r} (want {w!r})")
    return bad


def recorded_revision(record: dict, side: str, launch_info: pathlib.Path):
    """(commit, dirty) of the tree a run imported its code from.

    ``side`` is "reference" or "ajax". From the run's own record
    (``<side>_commit`` / ``<side>_dirty``, written by the runners since
    2026-10-04), else from the ``launch_info.txt`` that launch.sh wrote next to
    the run: "reference: ... @ SHA[; dirty: yes|no]" (dirty None when not
    recorded) and "ajax: ... @ SHA; uncommitted diff vs HEAD sha1: H"
    (dirty = H is not the sha1 of an empty diff). (None, None) if neither.
    """
    if f"{side}_commit" in record:
        return record[f"{side}_commit"], record.get(f"{side}_dirty")
    if not launch_info.exists():
        return None, None
    for line in launch_info.read_text().splitlines():
        if side == "reference":
            m = re.match(r"reference: .* @ ([0-9a-f]{40})(?:; dirty: (yes|no))?$", line)
            if m:
                return m[1], None if m[2] is None else m[2] == "yes"
        else:
            m = re.match(
                r"ajax: .* @ ([0-9a-f]{40}); uncommitted diff vs HEAD sha1: ([0-9a-f]{40})",
                line,
            )
            if m:
                return m[1], m[2] != EMPTY_SHA1
    return None, None


def check_ref_run(chk, seed, run_args, recs, smoke, ref_rev):
    """The run's own arguments (full mode: the protocol values) and eval sizes.

    ``ref_rev``: the recorded (commit, dirty) of the reference checkout
    (recorded_revision).
    """
    src = [
        run_args.get("agent_file", "<missing>"),
        run_args.get("embodied_file", "<missing>"),
    ]
    commit, dirty = ref_rev
    ok_src = all(str(f).startswith("<reference-checkout>/") for f in src)
    ok_rev = commit == REF_COMMIT and dirty is not True
    chk(
        ok_src and ok_rev,
        f"ref s{seed}: reference code imported from the 29eb964 checkout ({src})"
        + ("" if ok_rev else f": recorded commit {commit}, dirty {dirty}"),
    )
    if not smoke:
        want = {
            "seed": seed,
            "tiny": False,
            "rows": FULL_RUN["rows"],
            "eval_every": FULL_RUN["every"],
            "eval_episodes": FULL_RUN["eval_episodes"],
        }
        bad = _diff({k: run_args.get(k) for k in want}, want)
        chk(
            not bad,
            f"ref s{seed}: run_args.json has the protocol values"
            + (f": {bad}" if bad else ""),
        )
    n = run_args["eval_episodes"]
    ok = all(len(r["eval_returns"]) == n and len(r["eval_lengths"]) == n for r in recs)
    chk(ok, f"ref s{seed}: every evaluation has {n} episodes")


def check_flush(chk, seed, d, eps, n_rows):
    """The run finished and the post-loop flush step ran (S2(final) needs it).

    The episodes whose is_last row is rows / 16 exist only through the flush
    step (run_reference.Instr.flush; the loop emits rows 0 .. rows/16 - 1), and
    timing.json is written by Instr.finish after it.
    """
    path = d / "timing.json"
    t = json.loads(path.read_text()) if path.exists() else {}
    n_last = sum(e["row_last"] == n_rows // N_ENVS for e in eps)
    ok = "t_end" in t and "flush_episodes" in t and t["flush_episodes"] == n_last
    chk(
        ok,
        f"ref s{seed}: run finished and the flush step ran (timing.json t_end present; "
        f"flush_episodes {t.get('flush_episodes', '<missing>')} == episodes with row_last "
        f"{n_rows // N_ENVS}: {n_last})",
    )


def check_ajax_config(chk, cfg, seeds, smoke, ajax_rev, ajax_commit):
    """Ajax resolved config (run_ajax.py config.json) against the protocol values.

    ``ajax_rev``: the recorded (commit, dirty) of the Ajax src
    (recorded_revision); it must be ``ajax_commit``, src clean.
    """
    run = cfg["run"]
    tiny = bool(run.get("tiny"))
    rows = int(run["rows"])
    want = dict(AJAX_DREAMER_CONFIG)
    if tiny:
        want.update(AJAX_TINY_WIDTHS)
    bad = _diff(cfg["dreamer_config"], want)
    bad += _diff(cfg["agent_config"], AJAX_AGENT_CONFIG)
    con = cfg["constructor"]
    bad += _diff(
        {k: con.get(k) for k in ("env_id", "model_size", "n_envs", "extensions")},
        {"env_id": "CartPole-v1", "model_size": "1m", "n_envs": 16, "extensions": []},
    )
    bad += _diff(
        {
            "n_envs": cfg["n_envs"],
            "ring_rows": cfg["ring_rows"],
            "first_update_step": cfg["schedule"]["first_update_step"],
            "total_updates": cfg["schedule"]["total_updates"],
            "seeds": sorted(run["seeds"]),
        },
        {
            "n_envs": N_ENVS,
            "ring_rows": rows // N_ENVS,
            "first_update_step": GATE,
            "total_updates": expected_updates(rows),
            "seeds": sorted(seeds),
        },
    )
    if not str(run.get("ajax_file", "")).startswith("<ajax-src>/"):
        bad.append(
            f"ajax_file={run.get('ajax_file', '<missing>')!r} (want under <ajax-src>)"
        )
    if ajax_rev[0] != ajax_commit or ajax_rev[1] is True:
        bad.append(
            f"recorded Ajax commit {ajax_rev[0]}, dirty {ajax_rev[1]}"
            f" (want {ajax_commit}, clean)"
        )
    if not smoke:
        bad += _diff(
            {k: run.get(k) for k in ("tiny", "rows", "log_every", "eval_episodes")},
            {
                "tiny": False,
                "rows": FULL_RUN["rows"],
                "log_every": FULL_RUN["every"],
                "eval_episodes": FULL_RUN["eval_episodes"],
            },
        )
    chk(
        not bad,
        f"ajax: config.json has the protocol values{' (tiny)' if tiny else ''}"
        + (f": {bad}" if bad else ""),
    )


def check_done(chk, root, names, required):
    """DONE markers written by launch.sh: one per process, holding its exit code."""
    for name in names:
        p = root / f"{name}.DONE"
        if not p.exists():
            if required:
                chk(False, f"{name}: {p.name} missing (process not finished?)")
            continue
        code = p.read_text().strip()
        chk(code == "0", f"{name}: exit code {code} ({p.name})")


# ---------------------------------------------------------------- main
def main(argv=None):  # noqa: C901 (one linear report; kept as written)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(HERE / "results" / "round1"),
        help="directory holding ref_s*/ and ajax/ (default: results/round1)",
    )
    ap.add_argument("--ref-root", help="ref_s*/ from here instead (default: --root)")
    ap.add_argument("--ajax-root", help="ajax/ from here instead (default: --root)")
    ap.add_argument("--out", help="where the outputs go (default: --root)")
    ap.add_argument(
        "--ajax-src",
        default=DEFAULT_AJAX_SRC,
        help="the Ajax src the run was configured with; its HEAD is the expected"
        " Ajax commit (default: $AJAX_SRC, else this repository's src)",
    )
    ap.add_argument(
        "--ajax-commit", help="expected Ajax commit (default: HEAD of --ajax-src)"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    ref_root = pathlib.Path(a.ref_root) if a.ref_root else root
    ajax_root = pathlib.Path(a.ajax_root) if a.ajax_root else root
    out = pathlib.Path(a.out) if a.out else root
    out.mkdir(parents=True, exist_ok=True)
    chk = Checks()
    seeds = a.seeds
    ajax_commit = a.ajax_commit or provenance.git_state(a.ajax_src)["commit"]

    # ---- load
    ref, aj, ref_eps = {}, {}, {}
    ajax_dir = ajax_root / "ajax"
    check_done(chk, ref_root, [f"ref_s{s}" for s in seeds], required=not a.smoke)
    check_done(chk, ajax_root, ["ajax"], required=not a.smoke)
    for s in seeds:
        d = ref_root / f"ref_s{s}"
        ref[s] = read_jsonl(d / "records.jsonl")
        ref_eps[s] = read_jsonl(d / "episodes.jsonl")
        run_args = json.loads((d / "run_args.json").read_text())
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        check_ref_config(chk, s, cfg, run_args)
        ref_rev = recorded_revision(run_args, "reference", ref_root / "launch_info.txt")
        check_ref_run(chk, s, run_args, ref[s], a.smoke, ref_rev)
        check_episodes(chk, s, ref_eps[s], run_args["rows"])
        check_flush(chk, s, d, ref_eps[s], run_args["rows"])
        aj[s] = read_jsonl(ajax_dir / f"s{s}" / "records.jsonl")
    ajax_cfg = json.loads((ajax_dir / "config.json").read_text())
    check_ajax_config(
        chk,
        ajax_cfg,
        seeds,
        a.smoke,
        recorded_revision(ajax_cfg["run"], "ajax", ajax_root / "launch_info.txt"),
        ajax_commit,
    )

    rows_ref = {s: [r["rows"] for r in ref[s]] for s in seeds}
    rows_aj = {s: [r["rows"] for r in aj[s]] for s in seeds}
    if a.smoke:
        rows = rows_ref[seeds[0]]
        chk(
            all(rows_ref[s] == rows and rows_aj[s] == rows for s in seeds),
            f"same checkpoints on both sides and all seeds: {rows}",
        )
        s1_rows = tuple(rows[-3:])
    else:
        rows = FULL_ROWS
        chk(
            all(rows_ref[s] == rows and rows_aj[s] == rows for s in seeds),
            "10 records per seed per side at rows 2000 k",
        )
        s1_rows = S1_ROWS
    final = rows[-1]

    # updates
    ok = True
    for s in seeds:
        for r in ref[s] + aj[s]:
            ok &= r["updates"] == expected_updates(r["rows"])
    chk(
        ok,
        f"updates = 1 + floor((N - 1040) / 2) at every N on both sides"
        f" ({expected_updates(final)} at {final})",
    )
    if not a.smoke:
        chk(
            all(ref[s][-1]["updates"] == 9481 == aj[s][-1]["updates"] for s in seeds),
            "9481 updates at 20000 rows on both sides",
        )

    # Ajax S2 audit against the final rolling buffer
    fin = np.load(ajax_dir / "ajax_final.npz")
    fin_seeds = fin["seeds"].tolist()
    audit = {}
    for s in seeds:
        i = fin_seeds.index(s)
        cnt = fin["count"][i].astype(np.float64).ravel()
        sm = fin["sum"][i].astype(np.float64).ravel()
        audited = float(np.mean(sm / cnt)) if (cnt > 0).all() else float("nan")
        logged = aj[s][-1]["s2_logged"]
        logged = float("nan") if logged is None else logged
        audit[s] = (logged, audited)
        same = (math.isnan(logged) and math.isnan(audited)) or abs(
            logged - audited
        ) <= 1e-4
        chk(
            same,
            f"ajax s{s}: logged S2 {fmt(logged, 3)} == final buffer mean {fmt(audited, 3)} (tol 1e-4)",
        )
    for s in seeds:
        i = fin_seeds.index(s)
        chk(
            int(fin["rows"][i]) == final
            and int(fin["n_updates"][i]) == expected_updates(final),
            f"ajax s{s}: final state rows {int(fin['rows'][i])}, n_updates {int(fin['n_updates'][i])}",
        )

    # ---- statistics
    def at(recs, n, key):
        for r in recs:
            if r["rows"] == n:
                v = r[key]
                return float("nan") if v is None else float(v)
        return float("nan")

    stats = {"ref": {}, "ajax": {}}
    for s in seeds:
        stats["ref"][s] = {
            "S1": float(np.mean([at(ref[s], n, "eval_mean") for n in s1_rows])),
            "S2": s2_from_episodes(ref_eps[s], final),
            "eval": [at(ref[s], n, "eval_mean") for n in rows],
            "S2_curve": [s2_from_episodes(ref_eps[s], n) for n in rows],
        }
        stats["ajax"][s] = {
            "S1": float(np.mean([at(aj[s], n, "eval_mean") for n in s1_rows])),
            "S2": at(aj[s], final, "s2_logged"),
            "eval": [at(aj[s], n, "eval_mean") for n in rows],
            "S2_curve": [at(aj[s], n, "s2_logged") for n in rows],
        }
    A = float(np.mean([stats["ajax"][s]["S1"] for s in seeds]))
    ref_s1 = [stats["ref"][s]["S1"] for s in seeds]
    lo, hi = float(np.min(ref_s1)), float(np.max(ref_s1))
    learns = bool(A >= lo)
    within = bool(lo <= A <= hi)

    # ---- report
    L = []
    P = L.append
    title = "Ajax vs reference DreamerV3, CartPole-v1" + (
        " (SMOKE: plumbing only, no verdict)" if a.smoke else ""
    )
    P(f"# {title}\n")

    def shown(p):  # relative to the repository root, else the directory name
        return provenance.shown(p, REPO)

    where = (
        f"`{shown(root)}`"
        if ref_root == root and ajax_root == root
        else f"`{shown(out)}` (reference runs `{shown(ref_root)}`,"
        f" Ajax run `{shown(ajax_root)}`)"
    )
    P(
        f"Root: {where}; seeds {seeds}; checkpoints (rows) {rows}; S1 window {list(s1_rows)}; S2 at {final}.\n"
    )
    P("## Checks\n")
    P("```")
    L.extend(chk.lines)
    P("```\n")
    P("## Pre-registered rule\n")
    P(f"* A = mean over Ajax seeds of S1 = **{fmt(A)}**")
    P(f"* reference S1 range [min, max] = **[{fmt(lo)}, {fmt(hi)}]**")
    P(f"* **LEARNS = {learns}** (A >= min reference S1)")
    P(f"* **WITHIN_RANGE = {within}** (min <= A <= max)\n")
    P("## Per-seed S1 and S2\n")
    P("| seed | ref S1 | ajax S1 | ref S2 | ajax S2 (logged) | ajax S2 (audit) |")
    P("|---|---|---|---|---|---|")
    for s in seeds:
        P(
            f"| {s} | {fmt(stats['ref'][s]['S1'])} | {fmt(stats['ajax'][s]['S1'])} | "
            f"{fmt(stats['ref'][s]['S2'])} | {fmt(stats['ajax'][s]['S2'])} | {fmt(audit[s][1])} |"
        )
    P(
        f"| mean | {fmt(np.mean(ref_s1))} | {fmt(A)} | "
        f"{fmt(nanmean([stats['ref'][s]['S2'] for s in seeds]))} | "
        f"{fmt(nanmean([stats['ajax'][s]['S2'] for s in seeds]))} | |\n"
    )

    def curve_table(name, key):
        P(f"## {name}\n")
        hdr = (
            "| rows | "
            + " | ".join(f"ref s{s}" for s in seeds)
            + " | ref mean | "
            + " | ".join(f"ajax s{s}" for s in seeds)
            + " | ajax mean |"
        )
        P(hdr)
        P("|" + "---|" * (2 * len(seeds) + 3))
        for j, n in enumerate(rows):
            rv = [stats["ref"][s][key][j] for s in seeds]
            av = [stats["ajax"][s][key][j] for s in seeds]
            P(
                f"| {n} | "
                + " | ".join(fmt(v) for v in rv)
                + f" | {fmt(nanmean(rv))} | "
                + " | ".join(fmt(v) for v in av)
                + f" | {fmt(nanmean(av))} |"
            )
        P("")

    known_differences(P, seeds, rows, aj)
    curve_table("Evaluation curve (mean return of the evaluation episodes)", "eval")
    curve_table("S2 curve (training-episode rolling mean, Ajax definition)", "S2_curve")

    # updates and reference eval returns
    P("## Updates per checkpoint\n")
    P(
        "| rows | expected | "
        + " | ".join(f"ref s{s}" for s in seeds)
        + " | "
        + " | ".join(f"ajax s{s}" for s in seeds)
        + " |"
    )
    P("|" + "---|" * (2 * len(seeds) + 2))
    for n in rows:
        P(
            f"| {n} | {expected_updates(n)} | "
            + " | ".join(fmt(at(ref[s], n, "updates"), 0) for s in seeds)
            + " | "
            + " | ".join(fmt(at(aj[s], n, "updates"), 0) for s in seeds)
            + " |"
        )
    P("")

    # training metrics
    def metric(recs, n, key):
        for r in recs:
            if r["rows"] == n:
                v = r["train"].get(key)
                return float("nan") if v is None else float(v)
        return float("nan")

    def derived_scale(recs, n):
        num, den = metric(recs, n, "ret/std"), metric(recs, n, "ret_normed/std")
        return num / den if den and np.isfinite(den) and den != 0 else float("nan")

    def exact_scale_ref(s, n):
        for r in ref[s]:
            if r["rows"] == n and r.get("retnorm") and "scale" in r["retnorm"]:
                return float(r["retnorm"]["scale"])
        return float("nan")

    keys = [*namemap.COMPARE_KEYS, "retnorm_scale_derived", "retnorm_scale_exact"]

    def value(side, s, n, key):
        recs = ref[s] if side == "ref" else aj[s]
        if key == "retnorm_scale_derived":
            return derived_scale(recs, n)
        if key == "retnorm_scale_exact":
            if side == "ref":
                return exact_scale_ref(s, n)
            if n == final:
                i = fin_seeds.index(s)
                return max(
                    1.0, float(fin["retnorm_hi"][i]) - float(fin["retnorm_lo"][i])
                )
            return float("nan")
        return metric(recs, n, key)

    windows = [n for n in rows if n > GATE]
    P("## Training metrics, window means (mean over seeds [min, max])\n")
    P(
        "Reference windows hold updates n0-1 .. n1-1 (JAXAgent returns the previous call's metrics); "
        "Ajax windows n0 .. n1. Loss names mapped: "
        + ", ".join(
            f"`{k}`->`{v}`"
            for k, v in namemap.REF_TO_AJAX.items()
            if not k.endswith("_std")
        )
        + ".\n"
    )
    for n in windows:
        P(f"### window ending at {n} rows\n")
        P(
            "| metric (Ajax name) | ref mean [min, max] | ajax mean [min, max] | ajax/ref |"
        )
        P("|---|---|---|---|")
        for k in keys:
            rv = [value("ref", s, n, k) for s in seeds]
            av = [value("ajax", s, n, k) for s in seeds]
            rm, am = nanmean(rv), nanmean(av)
            ratio = (
                am / rm
                if np.isfinite(rm) and rm != 0 and np.isfinite(am)
                else float("nan")
            )
            P(
                f"| {k} | {fmt(rm, 4)} [{fmt(nanmin(rv), 4)}, {fmt(nanmax(rv), 4)}] | "
                f"{fmt(am, 4)} [{fmt(nanmin(av), 4)}, {fmt(nanmax(av), 4)}] | {fmt(ratio, 3)} |"
            )
        P("")
    P("## Training metrics per seed (ref / ajax)\n")
    for s in seeds:
        P(f"### seed {s}\n")
        P("| metric | " + " | ".join(str(n) for n in windows) + " |")
        P("|" + "---|" * (len(windows) + 1))
        for k in keys:
            cells = [
                f"{fmt(value('ref', s, n, k), 3)} / {fmt(value('ajax', s, n, k), 3)}"
                for n in windows
            ]
            P(f"| {k} | " + " | ".join(cells) + " |")
        P("")

    # reference per-episode eval returns
    P("## Reference evaluation returns per episode\n")
    for s in seeds:
        for r in ref[s]:
            P(f"* s{s} rows {r['rows']}: {r['eval_returns']}")
    P("")

    text = "\n".join(L)
    print(text)
    (out / "compare.md").write_text(text + "\n")

    summary = {
        "smoke": a.smoke,
        "seeds": seeds,
        "rows": rows,
        "s1_rows": list(s1_rows),
        "A": A,
        "ref_S1_min": lo,
        "ref_S1_max": hi,
        "LEARNS": learns,
        "WITHIN_RANGE": within,
        "per_seed": {
            side: {str(s): v for s, v in d.items()} for side, d in stats.items()
        },
        "ajax_s2_audit": {
            str(s): {"logged": v[0], "audited": v[1]} for s, v in audit.items()
        },
        "checks": chk.lines,
        "checks_failed": chk.failed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    plot(out / "curves.png", rows, seeds, stats, title)
    print(f"\nwrote {out / 'compare.md'}, {out / 'summary.json'}, {out / 'curves.png'}")
    if chk.failed:
        print(f"{chk.failed} CHECK(S) FAILED", file=sys.stderr)
        return 1
    return 0


def known_differences(P, seeds, rows, aj):
    """Known, disclosed differences between the two sides (PROTOCOL.md 6.1-6.2)."""
    nonfinite = []
    for s in seeds:
        for r in aj[s]:
            if r["rows"] in rows and r["rows"] > GATE:
                nonfinite += [
                    f"s{s}@{r['rows']}:{k}" for k, v in r["train"].items() if v is None
                ]
    P("## Known differences between the two sides (read before interpreting)\n")
    P(
        "* **D1 / U1, asynchrony (reference only).** The reference acts with policy parameters that "
        "lag the trained ones by about 2 vector steps (about 16 of 9481 updates; measured on the real "
        "reference in `refrun/audit_loop`), its first training batch is prefetched before the gate "
        "opens (11 distinct items of 16 in the audit), and its write-back and metrics are one call "
        "late. Ajax acts with the latest parameters. Evaluation uses the latest parameters on both "
        "sides."
    )
    P(
        "* **D2 / U2, shared replay-sampler stream (reference only).** Stock `make_replay` passes no "
        "seed, so `Uniform(seed=0)` samples the replay for every reference seed: reference seeds "
        f"{seeds} share one index stream (the replay contents differ). The reference S1 [min, max] "
        "range therefore comes from runs that are not fully independent. Ajax seeds its sampler from "
        "the run seed."
    )
    P(
        "* **Report prefetch stream (stock, D1 / D2).** With `run.eval_every = -1` the stock report "
        "stream (`train.py:76-77`) is created but never read: when the first replay items appear "
        "(about row 1040) its prefetch thread draws 2 batches of 16 (32 samples, popping up to 32 of "
        "the earliest online-queue items, plus Uniform-RNG and `agent.rng` draws) and then blocks for "
        "good. A stock run (`eval_every` 180 s) would draw 16 more every 180 s of wall time. "
        "Negligible: training draws 128 samples per vector step against 16 new items."
    )
    P(
        "* **NaN in training-metric windows.** The reference `Agg` skips NaN values; Ajax's "
        "accumulator propagates them, so one NaN makes the Ajax window mean NaN. A `nan` Ajax window "
        "against a finite reference window means 'a NaN occurred in that Ajax window', not a "
        "divergence of the finite values. Non-finite Ajax window means here: "
        + (
            f"{len(nonfinite)} ({', '.join(nonfinite[:20])}{' ...' if len(nonfinite) > 20 else ''})"
            if nonfinite
            else "none"
        )
        + "."
    )
    P(
        "* Reference metric windows hold updates n0-1 .. n1-1 (one-call lag); Ajax per-episode "
        "evaluation returns are not logged (only the mean).\n"
    )


def nanmin(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.min(xs)) if xs else float("nan")


def nanmax(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.max(xs)) if xs else float("nan")


def plot(path, rows, seeds, stats, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ref": "#2a78d6", "ajax": "#eb6834"}  # categorical slots 1, 2
    labels = {"ref": "reference 29eb964", "ajax": "Ajax"}
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, key, name in zip(
        axes,
        ["eval", "S2_curve"],
        ["Evaluation return (10 episodes)", "S2: training-episode rolling mean"],
    ):
        for side in ("ref", "ajax"):
            for i, s in enumerate(seeds):
                ax.plot(
                    rows,
                    stats[side][s][key],
                    color=colors[side],
                    lw=1.2,
                    alpha=0.55,
                    marker=markers[i % len(markers)],
                    ms=5,
                    label=f"{labels[side]} seed {s}",
                )
            mean = [
                nanmean([stats[side][s][key][j] for s in seeds])
                for j in range(len(rows))
            ]
            ax.plot(
                rows, mean, color=colors[side], lw=2.6, label=f"{labels[side]} mean"
            )
        ax.set_title(name, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("rows (one per env per vector step)", color="#52514e")
        ax.set_ylabel("return", color="#52514e")
        ax.set_ylim(0, 520)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b5b4ae")
        ax.tick_params(colors="#52514e")
    axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig.suptitle(title, fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
