# DreamerV3: Ajax against the reference code on CartPole

This directory reproduces the comparison of Ajax's DreamerV3 with the real
reference code (`danijar/dreamerv3` at `29eb964`, the version Ajax follows) on
gymnax CartPole-v1, run on 2026-10-03 and 2026-10-04. It holds the harness, the
protocol and pre-registrations, the root-cause investigation that came between
rounds 1 and 2, and the results of all three rounds in compact form. The
conclusions are in `PERFORMANCE_REPORT.md` (section "DreamerV3 agent (M7)",
subsection "Reference comparison").

Like `../parity/`, everything here is run by hand: the reference side needs a
throwaway venv with the reference's pinned dependencies. Nothing here is a
test: pytest does not collect it and CI does not run it.

## The question each round answered

The first CartPole learning check of the Ajax agent failed a bar of our own
(eval return > 400 at 20,000 rows). Each round ran the real reference code and
Ajax on the same protocol (`PROTOCOL.md`): the 1m model, 16 envs, train ratio
512, batches of 16 x 64, float32, CPU, evaluation on 10 episodes of the
sampling policy from fresh resets. Statistics and decision rules were written
down before each round's results existed.

| Round | Question | Reference side | Ajax side | Seeds, rows, eval every |
|---|---|---|---|---|
| 1 (`PROTOCOL.md`) | Does the reference learn CartPole better than Ajax, i.e. is there an end-to-end bug in Ajax? | stock 29eb964 (`run_reference.py`) | 5cb0738 (`run_ajax.py`) | 0-2, 20,000, 2,000 |
| 2 (`PREREG2.md`) | Is deviation D22 (the two-hot expectation) the whole difference? Each side is run with the other side's expectation and compared with round 1's other side: pair E (exact expectation on both sides) and pair R (the reference's expectation on both sides). | 29eb964 + Ajax's exact expectation (`run_reference_exact.py`) | 5cb0738 + the reference's expectation (`run_ajax_refdecode.py`) | 0-2, 20,000, 2,000 |
| 3 (`PREREG3.md`) | Is the reference's jump at its last checkpoint an end-of-run artifact (H1), a product of its correlated seeds (H2, deviation D2), or a real late-learning difference (H3)? | 29eb964 + exact expectation + replay sampler seeded per run (`run_reference_round3.py`) | 5cb0738 (`run_ajax.py`) | 0-4, 24,000, 400 |

## Results

- **Round 1.** The training-episode score S2 matched within 1% at 20,000
  rows (reference 106.6, Ajax 107.6); along the way the means differed by up
  to 22% (8,000 rows: 42.1 against 33.0). The dynamics, representation and
  reward losses matched within 1.1% from 4,000 rows; the reconstruction loss
  was 16-25% lower in Ajax at 10k, 12k, 16k, 18k and 20k rows (4% higher at
  14k; seed ranges disjoint only at 10k). The pre-registered evaluation
  statistic S1 (mean evaluation at 16k, 18k and 20k rows) failed: Ajax 204.1
  against reference seeds 241.4-386.1. The actor's entropy was lower in Ajax
  in every window from 4k to 18k rows (window means 6-23% lower; seed ranges
  disjoint in 5 of 10 windows), and the mean |encoder output| 5-10% lower, the
  gap shrinking over training (seed ranges disjoint up to 8k rows).
  `results/round1/compare.md`.
- **Investigation** (`DIAGNOSIS.md`): the encoder gap is the seeds' initial
  draws; the entropy gap is deviation D22. Under `jit`, XLA contracts the
  reference's mirror-pair sum into fused multiply-adds, which adds noise of std
  about 0.085 to every reward and value prediction early in training.
- **Round 2.** With the same expectation on both sides, the entropy seed
  ranges overlap in all 10 windows in both pairs. S1 still failed in both
  pairs (pair E: Ajax 204.1 against 265.2-307.0; pair R: 233.2 against
  241.4-386.1), driven by the 20,000-row checkpoint: all six reference runs
  (rounds 1 and 2) scored at least 399 there, against one of six at 18,000
  rows and none at 16,000; one of the six Ajax runs scored 500 at 20,000
  rows (round 1, seed 2), with no such concordance (`results/round2/pairE/`,
  `pairR/`).
- **Round 3.** The 20,000-row concordance disappears (2 of 5 runs reach 400 on
  each side); the pre-registered reading is H2 or noise. Primary statistic, the
  mean evaluation over [12k, 24k] rows (31 checkpoints): reference 312.7
  [270.3, 346.7], Ajax 265.4 [198.3, 321.1], Welch p = 0.12; the strict rule
  (Ajax mean at least the lowest reference seed) fails by 4.9. S1: reference
  282.1 [201.5, 358.1], Ajax 305.8 [250.7, 373.8], within range, p = 0.52.
  `results/round3/analysis3.md`.

No Ajax bug was found. The remaining structural difference is the reference's
asynchronous acting (D1), which cannot be removed without changing the
reference. `PERFORMANCE_REPORT.md` has the full account and the redefined
CartPole check that followed.

## Files

| File | What it is |
|---|---|
| `PROTOCOL.md` | Round 1 protocol (config mapping, environment port, training loop, evaluation, statistics, departures from the stock reference). Kept as pre-registered, except that its absolute local paths were replaced by placeholders (see "Recorded locations"); its path abbreviations name the scratch directories the runs used (`RR` is the scratch harness directory, whose scripts are now this directory, `REF` the reference checkout, `AJ` the Ajax checkout); the paths that moved or were not committed are listed under "Results layout". |
| `PREREG2.md`, `PREREG3.md` | Round 2 and round 3 pre-registrations, as written before the runs (round 3 with its amendment). |
| `DIAGNOSIS.md` | Root-cause investigation between rounds 1 and 2; paths edited to point here; its probe scripts and outputs were not committed. |
| `launch.sh` | Launches one round (`--round 1/2/3`), all processes in parallel in the background; `--smoke` for a tiny run. |
| `run_reference.py` | Reference runner: stock `main.py` train logic with the protocol's overrides, `train_instrumented.train`, evaluation and records (reference venv). |
| `run_reference_exact.py` | Round 2 reference: `run_reference.py` with `TwoHotDist.mean` replaced by Ajax's exact form at import time. |
| `run_reference_round3.py` | Round 3 reference: as `run_reference_exact.py`, plus the replay sampler seeded per run. |
| `train_instrumented.py` | Copy of the reference's `embodied/run/train.py` plus marked INSTRUMENTATION blocks. |
| `cartpole_env.py` | Numpy float32 port of gymnax CartPole-v1 as an `embodied.Env` (the reference side's environment). |
| `check_cartpole_port.py` | Self-test of the port against gymnax, step by step (Ajax venv). Named `check_*` so that pytest does not collect it. |
| `namemap.py` | Reference-to-Ajax training-metric name map. |
| `run_ajax.py` | Ajax runner: one vmapped run over the seeds (Ajax venv). |
| `run_ajax_refdecode.py`, `refdecode.py` | Round 2 Ajax: `run_ajax.py` with `TwoHot.decode` replaced by the reference's literal expectation at import time. |
| `compare.py` | Round 1 and round 2 analysis (S1, S2, checks, curves, training-metric tables). |
| `analyze3.py` | Round 3 analysis (LATE, S1, concordance, reading). |
| `reproduce_tables.sh` | Recomputes every committed table from `results/` and diffs it against the committed one. |
| `provenance.py` | How locations and revisions are recorded (see "Recorded locations"); used by the runners and the analysis scripts. Self-test: `python -m doctest provenance.py`. |
| `results/` | Compact results of the three rounds (below). |
| `LICENSE.dreamerv3` | The reference's MIT license (see "License"). |
| `LICENSE.gymnax` | gymnax's Apache License 2.0 (see "License"). |

## Setup

All paths below are relative to this directory. `launch.sh` defaults to the
locations shown, under `.work/` (git-ignored); every one can be overridden
(`bash launch.sh --help`).

**Reference checkout** (`--reference-checkout`, default `.work/dreamerv3_29eb964`):

    git clone https://github.com/danijar/dreamerv3 .work/dreamerv3_29eb964
    git -C .work/dreamerv3_29eb964 checkout 29eb964e2918a3f4db04086f7f51b60388e97f3d

The checkout is never edited: the round 2 and round 3 patches replace methods
at import time.

**Reference venv** (`--reference-python`, default `.work/venv/bin/python`).
Python 3.11 (the runs used 3.11.15), CPU only, no tensorflow or gym. The exact
versions the runs used (`uv pip list` of that venv):

    uv venv --python 3.11 .work/venv
    uv pip install --python .work/venv/bin/python \
        absl-py==2.5.0 attrs==26.1.0 chex==0.1.86 cloudpickle==3.1.2 colored==2.3.2 \
        decorator==5.3.1 dm-tree==0.1.10 einops==0.8.2 gast==0.7.0 jax==0.4.26 \
        jaxlib==0.4.26 ml-dtypes==0.5.4 msgpack==1.2.3 numpy==1.26.4 opt-einsum==3.4.0 \
        optax==0.2.2 psutil==7.2.2 pyzmq==27.2.0 ruamel-yaml==0.19.1 scipy==1.17.1 \
        six==1.17.0 tensorflow-probability==0.24.0 toolz==1.1.0 \
        typing-extensions==4.16.0 wrapt==2.5.0

**Ajax** (`--ajax-root`, default this repository; `--ajax-python`, default
`<ajax-root>/.venv/bin/python`, the project's Poetry venv: jax 0.7.2 in the
runs). The committed results ran commit `5cb0738` with no uncommitted changes
under `src/` (`results/round*/launch_info.txt`); rounds 2 and 3 ran from a
worktree frozen at that commit, so that edits elsewhere could not reach a
running process. To rerun that exact version:

    git worktree add .work/ajax-5cb0738 5cb0738
    bash launch.sh --round 1 --ajax-root .work/ajax-5cb0738 --ajax-python <repo>/.venv/bin/python

The analysis scripts (`compare.py`, `analyze3.py`, `reproduce_tables.sh`) need
numpy, scipy, matplotlib and PyYAML, which the Ajax venv has.

## Running a round

    bash launch.sh --round 1          # or 2, 3; output in .work/round<N>/

The script checks its inputs, writes `launch_info.txt` (time, OS and
architecture, load, revisions and their dirty flags; see "Recorded
locations"), starts one reference process per seed (working directory: the
reference checkout, as the stock `main.py`) and one vmapped Ajax process
(working directory: the output directory), and returns. A process is finished
when its `<name>.DONE` file exists; it holds the exit code. The script prints
the analysis command for the round:

- round 1: `compare.py --root .work/round1` (plus the `--ajax-src` it prints);
- round 2: `compare.py` twice, for the pre-registered cross pairs with round 1:
  pair E `--ref-root .work/round2 --ajax-root .work/round1 --out .work/round2/pairE`
  and pair R `--ref-root .work/round1 --ajax-root .work/round2 --out .work/round2/pairR`;
- round 3: `analyze3.py --root .work/round3`.

`compare.py` exits 1 if any of its checks fails (DONE markers, resolved
configs on both sides, provenance of the imported code, episode tiling, update
counts, the Ajax S2 audit). Its provenance checks require the reference files
the runs imported to be under `<reference-checkout>` at commit 29eb964 with no
uncommitted changes, and the Ajax file under `<ajax-src>` at the expected
commit with no uncommitted changes under `src` (`--ajax-commit`; default: the
HEAD of `--ajax-src`, itself defaulting to `$AJAX_SRC`, else this repository's
`src`).

**Expected wall time.** The committed runs ran all processes of a round in
parallel on a 14-core Apple Silicon Mac shared with other jobs (load average
17-25 at launch):

| Round | Reference, per seed (all seeds in parallel) | Ajax (all seeds, one vmapped process) |
|---|---|---|
| 1 | 4.7 h (build and compile 4 min) | 4.5 h |
| 2 | 3.6 h (build and compile 3 min) | 3.3 h |
| 3 | 5.8 h (build and compile 3 min) | 7.9 h |

`bash launch.sh --round N --smoke` (2 seeds, tiny widths, 1,600 rows,
evaluation every 400 rows on 2 episodes) checks the plumbing: about 5 minutes
for the reference processes and 2.5 minutes for Ajax on the same machine (the
three rounds' smokes run side by side). Check its output with the
`compare.py ... --smoke` command it prints. The port self-test takes about
10 s:

    JAX_PLATFORMS=cpu <ajax venv python> check_cartpole_port.py

Thread settings change speed, not results; `launch.sh` documents the
`REF_XLA_FLAGS` / `AJAX_XLA_FLAGS` variables. The reference is not bitwise
reproducible across reruns (thread timing of its prefetch seeds and
online-queue pops, `PROTOCOL.md` U6), so a rerun gives different numbers with
the same distribution.

## Reproducing the tables

    bash reproduce_tables.sh <ajax venv python>

recomputes round 1's `compare.md`, round 2's pair E and pair R, and round 3's
`analysis3.md` from `results/`, into a temporary directory, and diffs them
against the committed ones (every line except the `Root` line, and the whole
`summary.json` / `analysis3.json`). The `Root` line names the directory the
tables were computed from: a scratch output directory (`<scratch>/refrun/...`)
for the committed tables, `results/...` for the recomputed ones, so it differs
by construction and is not compared. `compare.py`'s provenance checks verify
that the records name the reference's 29eb964 and the Ajax commit the
committed results ran (5cb0738, passed as `--ajax-commit`). The committed
`compare.md`, `summary.json`, `analysis3.md` and `analysis3.json` are the files
written when the rounds were analysed, unchanged except for the local paths
(see "Recorded locations").

## Results layout

    results/round1/            round 1 (stock reference, Ajax 5cb0738)
    results/round2/            round 2 runs (reference + exact expectation, Ajax + reference expectation)
    results/round2/pairE/      compare.py: round-2 reference vs round-1 Ajax
    results/round2/pairR/      compare.py: round-1 reference vs round-2 Ajax
    results/round3/            round 3 (reference + exact expectation + seeded sampler, Ajax 5cb0738)

Each round directory holds what the runners wrote, file for file and unchanged
in content except for the local paths and the host name (see "Recorded
locations"), minus what is not needed to recompute the statistics (below); the
two largest kinds of file are gzipped (`gzip -d` gives the original file):

| File | Contents |
|---|---|
| `launch_info.txt` | Launch time, OS and architecture, load, reference and Ajax revisions, flags, rows / every / episodes / seeds. |
| `<name>.DONE` | Exit code of each process (all 0). |
| `ref_s<S>/config.yaml` | The full resolved reference config (stock `config.save`). |
| `ref_s<S>/run_args.json` | The runner's arguments, the protocol overrides, the jax version and the imported reference files. |
| `ref_s<S>/timing.json` | Build and compile time, first-update step, total wall time, flush-step episode count. |
| `ref_s<S>/PATCH.txt` | Rounds 2 and 3: the import-time patch applied. |
| `ref_s<S>/evals.jsonl` | One line per evaluation: rows, updates, the 10 returns and lengths, their means. |
| `ref_s<S>/records.jsonl.gz` | One line per checkpoint (`PROTOCOL.md` 5.1): evaluation, training-metric window means under both names (`train`: Ajax names, `train_raw`: reference names), metric-sample counts, return-normaliser state. |
| `ref_s<S>/episodes.jsonl.gz` | Every finished training episode: env, last row, length, score, terminal. S2 is computed from it. |
| `ajax/config.json` | Ajax's resolved config (constructor, `DreamerV3Config`, agent config, schedule, seeds, run arguments, imported file). |
| `ajax/timing.json` | Wall time of the vmapped `train` call (compilation included). |
| `ajax/PATCH.txt` | Round 2: the import-time patch applied. |
| `ajax/ajax_final.npz` | Final state per seed: the rolling episodic-return buffer (S2 audit), return-normaliser lo / hi, update and row counts. |
| `ajax/s<S>/evals.jsonl` | One line per evaluation: rows, updates, mean return and length (Ajax logs only the mean). |
| `ajax/s<S>/records.jsonl.gz` | One line per checkpoint, same schema as the reference's: evaluation mean, every `Train/` metric's window mean, the logged S2. |
| `compare.md`, `summary.json` | `compare.py` output (round 1; round 2 under `pairE/` and `pairR/`). |
| `analysis3.md`, `analysis3.json` | `analyze3.py` output (round 3). |

Not committed (all redundant with the files above, or not used by any
statistic): the 7.6 MB reference checkpoints and empty `replay/` directories,
the process logs, the reference's `train_metrics.jsonl` (the `train` fields of
`records.jsonl`), `progress.jsonl` (update count and time every 160 steps) and
`metrics.jsonl` (the stock logger's episode scores, a cross-check of
`episodes.jsonl`), Ajax's `s<S>/train_metrics.jsonl` (the `train` fields of
`records.jsonl`), `ajax_records.jsonl` (the per-seed `records.jsonl`
concatenated) and `all_metrics.npz` (every logged metric at the logging
ticks; `records.jsonl` holds all of them except `env_frames`), and the
`curves.png` plots, which `compare.py` redraws.

`refrun/audit_loop`, cited in `compare.md`, `refrun/audit_config`, cited in a
comment of `compare.py`, and the probe scripts cited in `PROTOCOL.md` and
`DIAGNOSIS.md` were one-off audits and were not committed.

`PROTOCOL.md` paths (`RR` = the scratch harness directory, `SP` = the session
scratch directory) map to this directory as follows:

| `PROTOCOL.md` path | Here |
|---|---|
| `RR/cartpole_env.py`, `RR/run_reference.py`, `RR/run_ajax.py`, `RR/train_instrumented.py`, `RR/compare.py` | same name in this directory |
| `RR/launch.sh` | `launch.sh --round 1` |
| `RR/test_cartpole_env.py` | `check_cartpole_port.py` |
| `RR/full/` (`ref_s{s}/`, `ajax/`) | `results/round1/` (`records.jsonl` gzipped) |
| `RR/full/*.log`, `RR/test_cartpole_env.log`, `RR/out_smoke/`, `RR/out_speed/` | not committed (logs, smoke and speed-test outputs) |
| `RR/inspect_*.py`, `RR/probe_ref_agent.py`, `RR/audit_loop`, `SP/m7/lc/diag.py` | not committed (one-off probes) |

## Recorded locations

No file the harness writes for keeping, and no committed file, holds an
absolute local path or a host name. A location is recorded relative to a named
root, `<name>/relative/path` (`provenance.py`):

| Placeholder | Root |
|---|---|
| `<reference-checkout>` | the reference checkout (`launch.sh --reference-checkout`; `DREAMERV3_REF` for the reference runners) |
| `<ajax-root>` | the Ajax checkout that was run (`launch.sh --ajax-root`) |
| `<ajax-src>` | its `src` directory, from which the Ajax run imported `ajax` (`AJAX_SRC` for the Ajax runners) |
| `<out>` | the output directory of a launch (`launch.sh --out`), which holds `ref_s<S>/` and `ajax/` |
| `<repo>` | the Ajax repository holding this directory |
| `<scratch>` | committed results and `PROTOCOL.md` only: the session scratch directory the rounds were run from |
| `<elsewhere>` | a file under none of the roots: only its name is recorded, and the provenance checks fail on it |

What identifies the code that ran is a commit, not a path. `launch.sh` writes
to `launch_info.txt` the commit of the reference checkout and of the Ajax
checkout, each with a dirty flag (`git status` lists something under the
checkout, under `src` for Ajax; for Ajax also the sha1 of `git diff HEAD --
src`, as before), the commit and dirty flag of this directory (the harness:
the runners, `train_instrumented.py`, `cartpole_env.py`, ...; recorded as
`<repo>/docs/...`, a line added after the committed rounds ran), and the OS
and architecture (`uname -sm`), not the host name. The runners record the
files they imported (`<reference-checkout>/...`, `<ajax-src>/...`) with the
commit and dirty flag of that tree: `reference_commit` / `reference_dirty` in
`run_args.json`, `run.ajax_commit` / `run.ajax_dirty` in `ajax/config.json`.
Their own output directory is recorded as `<out>/<name>`, also as the `logdir`
of the reference's `config.yaml` (the run itself uses the real directory).
Process logs (`*.log`, not committed) are the programs' own output and do
print real paths and, in Ajax's compile-cache message, the host name.

The committed results (rounds 1 to 3) were recorded with absolute paths; on
2026-10-04 these were replaced by the placeholders and nothing else was
changed, except that the host line of `launch_info.txt` was reduced to the OS
and architecture. The reference checkout (`<reference-checkout>`) was at
29eb964; the Ajax checkouts (`<ajax-root>`, imported from `<ajax-src>`) were
two worktrees at 5cb0738 with no uncommitted changes under `src`: a plain one
for round 1 and one frozen at that commit for rounds 2 and 3. The output
directories `<scratch>/refrun/full`, `full2`, `full3`, `cmpE` and `cmpR` are
now `results/round1`, `round2`, `round3`, `round2/pairE` and `round2/pairR`.
Those runners did not yet record commits in their own files; `compare.py`
reads them from the `launch_info.txt` next to each run, which did not record
the reference checkout's dirty flag.

## License

`train_instrumented.py` is a modified copy of the reference's
`embodied/run/train.py`; `run_reference.py`, `run_reference_exact.py`,
`run_reference_round3.py` and `refdecode.py` contain code derived from the
reference's `dreamerv3/main.py`, `jaxutils.py` and `nets.py`. Each carries a
header naming the reference file and commit, what was changed, and the
reference's copyright and MIT permission notice. `LICENSE.dreamerv3` is the
reference's license, copied from its checkout at 29eb964. `cartpole_env.py` is
a modified (numpy) port of gymnax's CartPole-v1 and says so in its docstring;
`LICENSE.gymnax` is gymnax's Apache License 2.0, copied from the gymnax 1.0.0
distribution (`YannBerthelot/gymnax@61ff068`) that Ajax runs.
