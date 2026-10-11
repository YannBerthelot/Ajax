# Performance

Speed is part of what a change must preserve. This page gives the rule, the
benchmark that checks it, and where results are recorded. For how to make a
training run fast as a user (seeds, environments, the compile cache), read
[Performance](../guide/performance.md) in the user guide.

## The guardrail

The rule, from [CLAUDE.md](../../CLAUDE.md): any agent that becomes more than
about 10% slower on the standardised benchmark run is investigated to its
root cause **before** the change lands. Small deltas that are understood are
written down in [PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md).
Benchmarks run on CPU by default, so they never compete with an experiment
using the GPU.

A regression is not automatically a veto. The recurrent replay rework of
October 2026 made SAC with memory 17-24% slower in exchange for exact
bootstraps at time limits; the report records the cost, where it goes, the
variants measured on the way and the owner's decision. What the rule forbids
is a slowdown nobody has explained.

## The benchmark

[`benchmarks/agent_bench.py`](../../benchmarks/agent_bench.py) times a fixed
training run for each agent:

- **The run.** Discrete-action agents train on `CartPole-v1`, continuous ones
  on `Pendulum-v1`, with 4 environments for 8,000 steps, logging off. Only
  the environment, `n_envs` and the step count are pinned; every other
  hyperparameter is the agent's default, so the benchmark follows the
  agent's own code path rather than a tuned configuration.
- **Presets.** Three agents cannot run that way and get a small documented
  preset in the script: APG trains on a Pendulum tracking task (it
  differentiates through the environment), and DreamerV3 and TD-MPC2 run
  shrunken models that keep every code path. UDRL and the offline
  `TDMPC2MultiTask` are not benchmarked.
- **The measurement.** For each agent the script first runs a 400-step
  warm-up (recorded as `compile_s`, compilation included), then three
  measured trials, each on a freshly built agent. `steps_per_s` is the
  trial's 8,000 steps over its wall time. Comparisons use the median of the
  trials.

Each result is one JSON line per trial, tagged with the git commit and the
JAX backend.

### Checking a change

Measure both sides on the same machine, one after the other, with the same
settings:

```bash
export JAX_PLATFORMS=cpu
# on main
uv run python benchmarks/agent_bench.py --only TD3 SAC --tag main --out /tmp/bench_main.jsonl
# on your branch
uv run python benchmarks/agent_bench.py --only TD3 SAC --tag branch --out /tmp/bench_branch.jsonl
uv run python benchmarks/agent_bench.py --compare /tmp/bench_main.jsonl /tmp/bench_branch.jsonl
```

`--compare` prints each agent's median steps per second on both sides and
the change, flags every agent slower than `--tol` (10% by default), and
exits with code 1 when there is one. An agent present on one side only shows
as `MISSING` or `NEW (no baseline)`.

Always pass `--out`. It defaults to `benchmarks/agent_baseline.jsonl`, and a
capture without it overwrites the recorded baseline.

### Reading the numbers

- **They belong to the machine.** Steps per second move with the CPU and
  with whatever else runs on it. The report's DreamerV3 and TD-MPC2 entries
  record runs at a load average of 25-35 and say plainly that those numbers
  are not a baseline. Compare files measured on one machine, under similar
  load.
- **The compile cache moves `compile_s`.** `import ajax` turns on a
  persistent compile cache, so a second process can load a program instead
  of compiling it. Each measured trial builds a fresh agent, and the
  script's notes measure tracing and lowering at 15-20% of a DreamerV3
  trial at its preset. Use the same cache setting on both sides;
  `AJAX_NO_COMPILE_CACHE=1` turns the cache off, as the refactoring's
  before-and-after comparison did.
- **Noise.** Single runs swing; the report has seen 15% between identical
  runs under load. Look at the medians, and repeat a measurement before
  calling a 10% change real.

### Recorded results

The `benchmarks/*.jsonl` files are records that `PERFORMANCE_REPORT.md`
cites; keep scratch measurements outside the repository.

| File | What it holds |
| --- | --- |
| `agent_baseline.jsonl` | the original baseline: 10 agents, including the since-deleted SafeSAC, at commit 04e3001 |
| `agent_refactor_before.jsonl`, `agent_refactor_after.jsonl` | the 12 benchmarked agents (and, before, SafeSAC) before and after the 2026 refactoring, at a1658d9 and 91ee10f, compile cache off |
| `agent_phase*.jsonl`, `agent_sac_rework.jsonl`, ... | the steps of earlier reworks |

The most complete reference is `agent_refactor_after.jsonl`. The
refactoring made no agent more than 10% slower, and AVG more than doubled
its speed: the shared loop no longer stacks its metrics at every step, and
its length-1 inner scans became direct calls.

## Recording a result

[PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md) is the summary: one
dated section per change that moved the numbers. A section says what changed
and why, how it was measured (the command, the machine, its load, the
number of trials), the table of before and after, and the cause of each
delta where it is known. [PERFORMANCE_LOG.md](../../PERFORMANCE_LOG.md) is
the running log of optimisation attempts, the reverted ones included, with
the tricks worth reusing.

## Other measurement tools

| Script | What it measures |
| --- | --- |
| [`benchmarks/perf_bench.py`](../../benchmarks/perf_bench.py) | SAC scenarios one process per measurement, splitting compile from steady-state time and recording peak GPU memory |
| [`benchmarks/seed_scaling_report.py`](../../benchmarks/seed_scaling_report.py) | prints time per seed against the number of seeds, from `benchmarks/seed_scaling.jsonl` |
| [`benchmarks/learning_checks.py`](../../benchmarks/learning_checks.py) | whether an agent learns its task on CPU at a small preset (the world models' learning gates); hours long, run outside CI |

## What keeps Ajax fast

Most of Ajax's speed comes from structure, not from tuning each line. A
change that breaks one of these patterns usually shows up in the benchmark.

- **One program per run, seeds as a `vmap`.** The report measured 28 times
  the single-seed throughput with 32 seeds on Pendulum, and 86 times with
  100 seeds on Plane3DCircle: the wall time barely moves as seeds are added.
- **The persistent compile cache.** Compilation dominated the start of a
  fresh process: with the cache, compile and warm-up fell by 77% on a warm
  process. A program containing a host callback is never cached, so a run
  with a `LoggingConfig` recompiles in each new process.
- **Nothing per-iteration piles up.** The loop writes each evaluation's
  metrics into rows the scan carries instead of stacking metrics at every
  iteration; `final_aux_scan` in
  [`perf_utils.py`](../../src/ajax/perf_utils.py) keeps only an inner scan's
  last metrics.
- **Unbatched gates.** Every predicate that decides whether to evaluate is
  built from values shared by all seeds, so `jax.lax.cond` skips the
  evaluation instead of computing it on every iteration (see
  [Architecture](architecture.md#evaluation-and-logging)).
- **Donated state.** A resumed run's state is donated to the compiled
  program, which reuses its buffers instead of holding a second copy.
- **No scatters in hot loops on CPU.** XLA runs a scatter one index at a
  time on CPU. The recurrent replay builds its bootstrap stream with one-hot
  selects; building it with scatters cost about 15% more on the
  transformer memory.
