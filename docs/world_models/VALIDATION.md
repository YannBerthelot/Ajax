# Validating DreamerV3 and TD-MPC2 against the published curves (M9)

Milestone M9 of `DESIGN.md` §11: the scripts, reference data, acceptance criteria
and CI smoke tests of the GPU validation. **The GPU runs themselves are not part
of M9 and have not been run**; the criteria below were fixed before any run.

**Caveat, everywhere results are reported.** Ajax runs the mujoco_playground
`dm_control_suite` tasks: MJX re-implementations of dm_control whose physics,
observations and sometimes rewards differ from dm_control's, on which the
published curves were measured. Every comparison here is approximate; a task
that falls outside its reference range calls for a look at the playground task
before the agent.

| File (`benchmarks/world_models/`) | What it does |
|---|---|
| `extract_references.py` | Reads the official checkouts once and writes the committed reference curves `references/*.json`. |
| `paper_protocol.py` | Registry and CLI of the paper-protocol runs (DreamerV3 on 18 tasks, TD-MPC2 on 22), resumable, one JSON line per chunk. |
| `paper_report.py` | Judges the runs against the references: markdown table, plots, verdict. |
| `multitask_validation.py` | TD-MPC2 multi-task pipeline: source runs, dataset, offline training, report. |
| `wm_runs.py` | Chunked, checkpointed runs and the metric probes (shared). |
| `wm_acceptance.py` | The acceptance criteria (their only definition). |

Tests: `tests/world_models/validation/` (CPU only, no network or GPU) and the
dataset round trip in `tests/agents/TDMPC2/test_tdmpc2_dataset.py`.

## 1. Reference data

`extract_references.py --dreamerv3 <danijar/dreamerv3@29eb964> --tdmpc2
<nicklashansen/tdmpc2@5f6fade>` writes two compact JSON files (both repositories
are MIT-licensed; each file records the repository, commit, file, license and
copyright line, the units and the protocol):

- `dreamerv3_dmc_proprio.json`: `scores/dmc_proprio-dreamerv3.json.gz` (added in
  `2411f7d`): the 18 DMC-proprio tasks of Table 11, 5 seeds, 49 points from 10K
  to 490K **env steps** (agent steps x action repeat 2, summed over the 16 envs:
  `main.py:150`). The values are the returns of the **training episodes of the
  stochastic policy** (`episode/score`, `embodied/run/train.py:36-61`;
  dreamerv3_spec 7.3, 7.7), binned every 10K env steps. The file does not say how
  it binned: with 16 lockstep envs an episode ends only every ~16K env steps, so
  its bins were filled or smoothed. Returns are rounded to 0.01. The aggregates
  reproduce dreamerv3_spec 7.7: task mean / median 675.4 / 790.4 at 250K and
  757.8 / 868.5 at 490K (the paper's printed Table 11 mean and median rows are
  swapped).
- `tdmpc2_dmc.json`: `results/<task>.csv` (added in `b67b21c`), columns `step,
  reward, seed`, 3 seeds (1, 2, 3), for the 22 DMC tasks playground implements.
  `step` counts **env steps** (agent steps x 2; tdmpc2_spec 4.22): one point per
  50K agent steps; `reward` is the mean return of 10 evaluation episodes of the
  planner in `eval_mode` (`online_trainer.py:27-48`; spec 4.21). Budgets: 4M env
  steps, 14M for humanoid. It also records the mt30 task list
  (`tdmpc2/common/__init__.py:26-37`) and its playground equivalents.

Task maps (reference name → playground id) are explicit in the script; it raises
on any DMC task it cannot classify.

| Reference | Mapped to playground | Unavailable in playground |
|---|---|---|
| DreamerV3 Table 11 (18) | all 18 (`dmc_ball_in_cup_catch` → `BallInCup`; the others by name) | none |
| TD-MPC2 DMC results (39) | 22: acrobot-swingup, cartpole-{balance, balance-sparse, swingup, swingup-sparse}, cheetah-run, cup-catch (`BallInCup`), finger-{spin, turn-easy, turn-hard}, fish-swim, hopper-{hop, stand}, humanoid-{run, stand, walk}, pendulum-swingup, reacher-{easy, hard}, walker-{run, stand, walk} | 17: dog-{run, stand, trot, walk}, quadruped-{run, walk} and the 11 custom tasks (cheetah-jump, cheetah-run-{back, backwards, front}, cup-spin, hopper-hop-backwards, pendulum-spin, reacher-three-{easy, hard}, walker-{run, walk}-backwards) |
| TD-MPC2 mt30 (30) | the 19 original DMC tasks | the 11 custom tasks |

The other 65 TD-MPC2 result files are Meta-World, MyoSuite and ManiSkill2 tasks
(out of scope: vector DMC only).

## 2. Paper-protocol runs (`paper_protocol.py`)

One run = one agent on one task with the reference's seed count, the seeds
vmapped (one jitted program per seed). Every hyperparameter not listed is the
agent's default, which is the paper-era value (`deviations.md`).

| | DreamerV3 (`dreamerv3-<task>`) | TD-MPC2 (`tdmpc2-<task>`) |
|---|---|---|
| Tasks | the 18 of Table 11 | the 22 mapped DMC tasks |
| Model | `model_size="12m"` | `model_size=5` (README.md:102, :115; spec 4.30) |
| Envs, repeat, episode | 16, 2, 1000 env steps (500 agent steps) | 1, 2, 1000 env steps (T = 500) |
| Ratio / UTD | `train_ratio=512` | 1 update per agent step (2 env steps) |
| Budget | 500K env steps = 250K rows (Table 2; dreamerv3_spec 7.5, OQ 1) | the CSV's: 4M env steps = 2M agent steps; humanoid 14M = 7M |
| Seeds | 5 (0-4) | 3 (0-2) |
| Metric (the reference's) | mean return of the training episodes that ended in the chunk (stochastic policy) | mean return of 10 eval episodes (planner, `eval_mode`), fresh initial states |
| Chunk (one record) | 10K rows = 20K env steps (625 ticks) | 50K agent steps (`eval_freq`) |
| Citations | Table 2 p.19; `29eb964:dreamerv3/configs.yaml:44, :84, :232-236` | `5f6fade:tdmpc2/config.yaml:10-11, :27`; spec 4.11, 4.21, 4.22 |

**Chunks and resume.** A run is a sequence of `train()` calls of one chunk each,
through the agents' own resume path (`initial_state=`), which continues every
schedule from the absolute tick and resizes the replay for the run so far
(`DESIGN.md` §5.5-5.6): a chunked run computes what the uninterrupted run
computes (pinned by the agents' resume tests and, for the scripts, by the smoke
tests). The state is saved with `ajax.checkpoint` at most every 900 s (DreamerV3's
own `run.save_every`, `configs.yaml:48`) and restored into the agent's
`n_timesteps=0` skeleton. Rerunning the same command resumes from the last save
(the records written after it are dropped and recomputed); a finished run is
skipped and its state deleted. A run directory with another specification is
refused. `run.json` is written atomically; a `curve.jsonl` line cut short by a
kill (it comes after the last save) is dropped and recomputed on resume. Each
chunk recompiles the training program: the replay ring is sized to the rows so
far plus the chunk (`DESIGN.md` §5.5), so its shape changes every chunk until
it reaches the replay capacity (DreamerV3: never at this budget, 25 compiles per
task; TD-MPC2: about 20, until the ring holds 1M steps; after that the
persistent compile cache hits). §5 counts them.

**Outputs.** `<out>/<run>/run.json` (the run's specification and the agent's
resolved configuration, with provenance) and `curve.jsonl`, one line per chunk:
`progress` in the agent's `unit` (DreamerV3 `rows`, TD-MPC2 `agent_steps`),
`env_frames` (= progress x action repeat 2: the references' env-step axis),
`metric`, per-seed `value` and `episodes` (DreamerV3 also `return_sum`),
`n_updates`, replay bytes, chunk wall time and provenance (git sha and dirty
flag, jax version, backend, device).

**DreamerV3 metric, how it is read.** The row collector keeps each env's last 10
episode returns. On DMC (fixed length, no terminations) an env's `k`-th episode
ends on tick `k (T + 1) + T - 1` (`T` agent steps plus the reset row), so the
probe knows how many episodes each env ended in the chunk, checks that against
the window's write position (a termination raises) and averages them. A chunk of
625 ticks ends 1 or 2 of the 501-row episodes per env (16-32 episodes).

**TD-MPC2 metric.** `evaluate_tdmpc2`, the agent's own evaluation (planner in
`eval_mode`, a rebuilt env, fresh initial states), vmapped over the seeds after
each chunk, with key `fold_in(PRNGKey(seed), agent_steps)` (a resumed run uses
the uninterrupted run's keys). The reference also
evaluates the untrained agent at step 0; Ajax does not (no criterion uses it).

### GPU commands

```bash
cd Ajax
python benchmarks/world_models/paper_protocol.py --list
# one task per GPU (CUDA_VISIBLE_DEVICES), all writing to the same directory:
python benchmarks/world_models/paper_protocol.py --out runs/ --only dreamerv3-walker_walk
python benchmarks/world_models/paper_protocol.py --out runs/ --only tdmpc2-walker-walk
# or everything, sequentially:
python benchmarks/world_models/paper_protocol.py --out runs/
# at any time, on the runs so far:
python benchmarks/world_models/paper_report.py --runs runs/ --out runs/report.md --plot-dir runs/
```

`--seeds` overrides the seeds with a prefix of the protocol's (e.g. `--seeds 0 1
2` for DreamerV3 on a smaller GPU; a verdict needs at least 3); other seeds make
the run off protocol. `--smoke` runs one tiny task per agent on CPU (plumbing
only).

**What the report checks besides the criteria.** Each run's `run.json` must be
its registry entry's (`paper_runs()[<directory name>]`: agent, env, kwargs,
budget, chunk, evaluation episodes; seeds the protocol's or a prefix of them).
A run that is not is reported as **off protocol**, with the differing keys, and
never judged, so its windows stay `INCOMPLETE`: a verdict cannot come from
changed settings, a changed budget or chosen seeds. A table lists every run's
provenance (commits, a tree with changes, JAX versions, devices) and flags runs
whose records span several of them, so a verdict can be traced to the code
that produced it.

## 3. Acceptance criteria (`wm_acceptance.py`)

Fixed before any run; changing one is a new protocol, recorded as such.

Scores are averaged over **windows** of env steps, per seed (Ajax's DreamerV3
points weighted by their episodes):

| Agent | Windows (env steps) |
|---|---|
| DreamerV3 | (200K, 300K] around the 250K midpoint; (400K, 500K] |
| TD-MPC2 | (500K, 1M], up to the paper's DMC headline budget; the last 1M of the task's budget ((3M, 4M], humanoid (13M, 14M]) |

At each window:

- **per task**: Ajax seed mean ≥ lowest reference seed − 50 (return units; DMC
  returns lie in [0, 1000]);
- **aggregate** over the protocol's tasks: Ajax median and mean of the task
  means each ≥ the reference's − 50, and at least 80% of the tasks passing (15
  of DreamerV3's 18, 18 of TD-MPC2's 22).

A task's window is judged only when its run is on protocol (§2), reached the
window's end (its last record at or past it) and has at least 3 seeds; until
every protocol task is judged the window is `INCOMPLETE`. The
verdict is `FAIL` if any window fails, else `INCOMPLETE` if any is, else `PASS`
(exit code 1 on `FAIL`).

Why these:

- *Windows, not points*: a point is one 10-episode evaluation (TD-MPC2) or the
  training episodes of 20K env steps (DreamerV3), and DreamerV3's bins have an
  undocumented convention; a 100K-1M window absorbs both. Two windows test both
  sample efficiency and the end of training.
- *The reference's lowest seed as the per-task bar*: robust to seed noise on the
  high-variance tasks (DreamerV3's cartpole_swingup_sparse, hopper_stand and
  finger_turn_easy have seed SDs above 200; spec 7.7) without letting a task
  fall below everything the reference produced; exceeding the reference never
  fails.
- *50 return units*: room for the playground ports and for Ajax's 3-5 seeds;
  5% of the DMC maximum.
- *Median, mean and 80%*: a few tasks may legitimately differ in playground; a
  systematic shortfall, or one hidden by wide reference ranges, may not (the
  aggregates catch what lenient per-task bars let through).
- *What it validates*: that the agents reproduce the papers' learning on the
  playground ports of the tasks, within the reference's own seed spread. It
  cannot separate an agent bug from a playground difference on one task.

## 4. TD-MPC2 multi-task pipeline (`multitask_validation.py`)

`DESIGN.md` §7 "Validation (M9)": the paper's multi-task models train offline on
single-task agents' replay buffers (Sec. 4.1; spec 4.18-4.19); the official
datasets are out of scope, so the pipeline makes its own on the 19 original mt30
DMC tasks (playground versions, mt30 order = task ids):

| Stage | What | Settings |
|---|---|---|
| `sources` | single-task TD-MPC2 per task, k seeds, evaluated every 50K steps; the full history exported (`export_episodes`, which raises on any eviction) and saved as a one-task dataset | `model_size=5`, repeat 2, 1000-step episodes, **500K agent steps** (1M env steps), `buffer_size` 1M ≥ run length, k = 3 |
| `dataset` | pooled (`pool_tasks`), saved (`save_dataset`: `.npz` of the arrays + JSON metadata, no pickle), loaded back (`load_dataset`, which runs `MultiTaskDataset.check`) and compared equal; `dataset.json` records the summary and the size and SHA-256 of `dataset.npz` and of every source export | 19 tasks x 3 seeds x 1000 episodes of 501 rows, obs 24, action 6 |
| `train` | `TDMPC2MultiTask` offline, every task evaluated after each chunk | `model_size=19` (the paper's ablation size), batch 1024 (paper), 1M updates in chunks of 100K, 1 seed, 10 eval episodes per task (`eval_mode`) |
| `report` | per task, offline return vs the source agents' final return | below |

**Criterion** (`wm_acceptance.judge_multitask`): per task, offline mean return ≥
0.5 x the source seeds' mean final return; tasks whose sources stay below 100
(did not learn) are reported, not judged; at least 80% of the tasks must be
judged (16 of 19), else `INCOMPLETE`: with most sources below the floor the
comparison cannot test the mechanisms; `PASS` when ≥ 80% of the judged tasks
pass. The paper's multi-task models fall well short of single-task agents
(mt30 normalised score 28.3 at 5M, 54.2 at 19M; 78.0 at about 19M on a 15-task
DMC subset, spec 4.28, against about 87 for single-task agents at 1M env steps),
so the bar is a fraction, set below that ratio for per-task spread and a smaller
dataset. **It validates the multi-task mechanisms on Ajax's own data, not the
paper's numbers.**

The budgets are ours: 1M updates of 1024 slices sample each of the 28.5M stored
transitions about 36 times, close to the paper's mt30 run (10M x 1024 / 345M ≈
30). Evaluations of successive chunks start from the same initial states (the
agent's evaluation key does not advance without its logger), a common-random-
numbers comparison.

```bash
python benchmarks/world_models/multitask_validation.py --out mt/             # all stages
python benchmarks/world_models/multitask_validation.py --out mt/ --stage sources \
    --tasks walker-stand walker-walk                                        # split over GPUs
python benchmarks/world_models/multitask_validation.py --out mt/ --stage dataset
python benchmarks/world_models/multitask_validation.py --out mt/ --stage train
python benchmarks/world_models/multitask_validation.py --out mt/ --stage report
```

Every stage resumes or skips what is done. The `dataset` and `train` stages
check `dataset.json` against the files (hashing ~7 GB: under a minute) and
refuse a dataset whose sources changed since it was built (a source run redone:
delete `dataset.npz`, `dataset.json` and `offline/` to rebuild); the offline
run's specification holds `dataset.json`, so neither a resume nor the report
uses an offline run of another dataset. The report uses only the final records
of runs whose `run.json` is the protocol's, lists the others as off protocol and
gives every run's provenance, as the paper-protocol report does. `--smoke`: 2
tasks, tiny models, CPU.

## 5. Resource estimates

**Estimates, not measurements**: from the model and replay sizes, the agents'
docstrings, the papers' reported costs and the CPU timings below. Measure the
first GPU run of each kind before scheduling the rest.

| Run | GPU memory | Host RAM / disk | Wall time (one GPU) |
|---|---|---|---|
| DreamerV3, one task, 5 seeds | replay 2.1 GB + working set 1.1 GB per seed (`DreamerV3` docstring: 8.3 KB per row x 250K rows); 16 GB for 5 seeds in the first chunk, then **~27 GB from the second chunk on**: every chunk grows the ring (the capacity, 5M rows, exceeds the run) while the previous state, still referenced by the caller, stays alive. Use a 40 GB GPU, or `--seeds 0 1 2` for ~16 GB | checkpoint up to ~11 GB (deleted at the end); host RAM ~25 GB to save it | 125K updates per seed. The reference took 0.3 A100-days per seed (Table 2, bf16, asynchronous): 5 seeds at most ~1.5 A100-days per task if vmapping saved nothing; 18 tasks ≤ ~27 A100-days. **Plus 25 compiles per task** (one per chunk): ~1.5-2 min each measured on CPU for `12m`, so ~40-50 min per task, ~12-15 h over 18 tasks (measure on GPU) |
| TD-MPC2, one task, 3 seeds | replay 124 MB per seed (walker; 357 MB humanoid) + ~0.1 GB model and optimizers: < 2 GB | checkpoint ≤ 1.2 GB | 2M agent steps (humanoid 7M), each one planning decision (6 MPPI iterations of 512 samples) and one update (batch 256), latency-bound: ~3-6 h per task, ~10-20 h for humanoid, + ~1 min per evaluation (40 / 140 of them), + ~20 compiles until the ring reaches 1M steps (later chunks hit the compile cache); 22 tasks ~90-180 GPU-hours |
| Multi-task sources, 19 tasks x 3 seeds | as TD-MPC2 above (replay ≤ 62 MB per seed) | 186 MB of episodes per task | 500K agent steps per task: ~1-3 h each, ~20-60 GPU-hours |
| Multi-task dataset | - | 3.5 GB file; ~10 GB host RAM to build and check | minutes |
| Multi-task offline training | dataset 3.5 GB (one copy shared by the seeds) + 19M model, optimizers and batch ~1 GB | checkpoint ~0.3 GB | 1M updates: the paper's 19M model took 5.3 RTX 3090-days for 10M updates (Table 1), ~46 ms per update, so ~13 h, + 10 evaluations of 19 tasks |

CPU smoke timings (Apple M-series, 14 cores shared at load average ~40 with other
jobs): `paper_protocol.py --smoke` took 160 s for DreamerV3 (2 chunks of 168 rows)
and 190 s for TD-MPC2 (2 chunks of 40 steps), dominated by compilation. The CI
smoke tests took, at that load: DreamerV3 (uninterrupted + killed and resumed from
disk + report) 261 s, TD-MPC2 (killed and resumed) 170 s, the multi-task pipeline
(2 tasks, then a second invocation) 206 s; a quiet CI runner should be faster. For
scale, the CPU learning checks (`PERFORMANCE_REPORT.md`) took about 0.4 s per
update for DreamerV3 at `1m` and 0.1 s per agent step for TD-MPC2 at size 1:
the paper sizes are not runnable on CPU.

## 6. Choices of ours (not the papers')

- The acceptance criteria, windows and tolerances (§3) and the multi-task
  budgets, model size, seeds and fraction (§4).
- The DreamerV3 metric is read per chunk of 20K env steps (the reference logs its
  episodes as they end and bins them every 10K); the comparison windows make the
  difference immaterial.
- TD-MPC2 evaluations use their own keys (`fold_in(PRNGKey(seed), agent_steps)`)
  and start at the first chunk, not at step 0.
- Checkpoints every 900 s at most, deleted when a paper-protocol run finishes.
