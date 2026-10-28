# Performance

Speed is part of what a change must preserve. This page gives the rule, the
benchmark that checks it, and where results are recorded. For how to make a
training run fast as a user (seeds, environments, the compile cache), read
[Performance](../guide/performance.md) in the user guide. The benchmarks run
on CPU; Ajax claims no GPU speed-up.

## The guardrail

The rule, from [CLAUDE.md](../../CLAUDE.md): any agent that becomes more than
about 10% slower on the standardised benchmark run is investigated to its
root cause **before** the change lands. Small deltas that are understood are
written down in [PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md).
Benchmarks run on CPU by default, so they never compete with an experiment
using the GPU. The benchmark is run by hand
([Checking a change](#checking-a-change)); CI only tests the script itself
(`tests/benchmarks`).

A regression is not automatically a veto. The recurrent replay rework of
October 2026 made SAC with memory 17-24% slower in exchange for exact
bootstraps at time limits; the report records the cost, where it goes, the
variants measured on the way and the owner's decision. What the rule forbids
is a slowdown nobody has explained.

## The benchmark

[`benchmarks/agent_bench.py`](../../benchmarks/agent_bench.py) times a fixed
training run for each agent:

- **The run.** Discrete-action agents train on `CartPole-v1`, continuous ones
  on `Pendulum-v1`, with 4 environments for 8,000 steps, logging off. The
  environment, `n_envs` and the step count are pinned, and so is
  `learning_starts=1000` for SAC, TD3, REDQ and ASAC, whose defaults (10,000)
  would outlast the run. Every other hyperparameter is the agent's default,
  so the benchmark follows the agent's own code path rather than a tuned
  configuration. `tests/benchmarks/test_agent_bench.py` checks that every
  entry updates within the run.
- **Presets and scenarios.** Three agents cannot run that way and get a small
  documented preset in the script: APG trains on a Pendulum tracking task (it
  differentiates through the environment), and DreamerV3 and TD-MPC2 run
  shrunken models that keep every code path. SAC has a second scenario,
  `SAC/inverted_pendulum`, a Brax task whose episodes terminate, for the
  auto-reset and terminal targets. The offline `TDMPC2MultiTask` is not
  benchmarked.
- **The measurement.** For each agent the script first runs a 400-step
  warm-up (recorded as `compile_s`, compilation included), then three
  measured trials, each on a freshly built agent. A fresh agent traces and
  lowers its program again, and compiles it unless the compile cache holds
  it. So each trial records two times:
    - `measured_s`, the whole call, and `steps_per_s`, the 8,000 steps over
      it;
    - `exec_s`, the same time less the spans JAX reports for tracing,
      lowering and compiling (overlapping spans counted once), and
      `exec_steps_per_s`. This is the run's execution alone.

Each result is one JSON line per trial, tagged with the git commit and the
JAX backend. `--n-seeds N` trains N seeds at once, the vmapped program
experiments run; steps per second then count every seed's steps.

### The baseline

[`benchmarks/agent_baseline.jsonl`](../../benchmarks/agent_baseline.jsonl)
is the reference a change is compared with. It holds the run of 2026-10-12:
`main` at `a8a13c2`, macOS on Apple silicon, CPU only, the compile cache off
(`AJAX_NO_COMPILE_CACHE=1`), the script's defaults (4 envs, 8,000 steps, one
seed, three trials), 14 entries (every exported agent but `TDMPC2MultiTask`,
plus `SAC/inverted_pendulum`). The report's
[section of that date](../../PERFORMANCE_REPORT.md) gives the medians.

That baseline breaks with every earlier `benchmarks/agent_*.jsonl` file in
two ways. Before it, SAC, TD3, REDQ and ASAC ran at their default
`learning_starts=10_000`, past the end of the run, so their rows timed
collection alone, with no update. And the earlier files have no `exec_s`.
Do not compare against them.

### Checking a change

Measure both sides on the same machine, one after the other, with the same
settings:

```bash
export JAX_PLATFORMS=cpu AJAX_NO_COMPILE_CACHE=1
# on main
uv run python benchmarks/agent_bench.py --only TD3 SAC --tag main --out /tmp/bench_main.jsonl
# on your branch
uv run python benchmarks/agent_bench.py --only TD3 SAC --tag branch --out /tmp/bench_branch.jsonl
uv run python benchmarks/agent_bench.py --compare /tmp/bench_main.jsonl /tmp/bench_branch.jsonl
```

`--compare` reads `exec_steps_per_s` when both files have it, and
`steps_per_s` otherwise; it says which at the top. It prints each agent's
median on both sides and the change, flags every agent slower than `--tol`
(10% by default), and exits with code 1 when there is one. An agent present
on one side only shows as `MISSING` or `NEW (no baseline)`.

`--out` has no default: a capture without it stops with an error, so a quick
check of one agent can never overwrite the recorded baseline. To compare with
the baseline itself, capture on the same kind of machine and pass
`benchmarks/agent_baseline.jsonl` as the first file of `--compare`.

### Reading the numbers

- **They belong to the machine.** Steps per second move with the CPU and
  with whatever else runs on it. The report's first DreamerV3 and TD-MPC2
  bench entries, run at a load average of 25 to 35, read 81 and 64-66 steps
  per second, against 275 and 245 for the whole run in the quiet baseline.
  Compare files measured on one machine, under similar load.
- **Execution, not compilation.** `exec_s` leaves out tracing, lowering and
  compiling, which a fresh agent repeats in every trial: at the DreamerV3
  preset tracing and lowering alone take about 15-20% of a trial. That is
  why `--compare` judges `exec_steps_per_s`. The compile cache still moves
  `compile_s` and `measured_s`, so use the same cache setting on both
  sides; the baseline was captured with it off.
- **Noise.** Single runs swing; the report has seen 15% between identical
  runs under load, and UDRL's three baseline trials spread from 9,604 to
  15,069 execution steps per second. Look at the medians, and repeat a
  measurement before calling a 10% change real.

### Recorded results

The `benchmarks/*.jsonl` files are the recorded results; keep scratch
measurements outside the repository.

| File | What it holds |
| --- | --- |
| `agent_baseline.jsonl` | the current baseline (above) |
| `agent_refactor_before.jsonl`, `agent_refactor_after.jsonl` | the agents before (13, the since-deleted SafeSAC included) and after (12) the 2026 refactoring, at a1658d9 and 91ee10f, compile cache off; SAC, TD3, REDQ and ASAC timed collection only |
| `results.jsonl` | `perf_bench.py` scenarios, May 2026, on a GPU |
| `seed_scaling.jsonl`, `seed_scaling_p3d.jsonl` | the seed-scaling runs of the user guide's [Performance](../guide/performance.md) page, May 2026, on a GPU |

Older records (per-phase benchmarks, the compile-cache and bf16
experiments) were removed on 2026-10-12; git history has them before
`d05a9f2`.

## Recording a result

[PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md) is the summary: one
dated section per change that moved the numbers. A section says what changed
and why, how it was measured (the command, the machine, its load, the
number of trials), the table of before and after, and the cause of each
delta where it is known. [PERFORMANCE_LOG.md](../../PERFORMANCE_LOG.md) is
the running log of optimisation attempts, the reverted ones included, with
the tricks worth reusing. A new baseline replaces `agent_baseline.jsonl` and
gets its own section, saying how it breaks with the one before.

## Other measurement tools

| Script | What it measures |
| --- | --- |
| [`benchmarks/perf_bench.py`](../../benchmarks/perf_bench.py) | SAC scenarios one process per measurement, splitting compile from steady-state time and recording peak GPU memory |
| [`benchmarks/seed_scaling_report.py`](../../benchmarks/seed_scaling_report.py) | prints time per seed against the number of seeds, from `benchmarks/seed_scaling.jsonl` |
| [`benchmarks/world_models/learning_checks.py`](../../benchmarks/world_models/learning_checks.py) | whether an agent learns its task on CPU at a small preset (the world models' learning gates); hours long, run outside CI |

## What keeps Ajax fast

Most of Ajax's speed comes from structure, not from tuning each line. A
change that breaks one of these patterns usually shows up in the benchmark.

- **One program per run, seeds as a `vmap`.** The report measured 28 times
  the single-seed throughput with 32 seeds on Pendulum, and 86 times with
  100 seeds on Plane3DCircle: the wall time barely moves as seeds are added.
- **The persistent compile cache.** Compilation dominated the start of a
  fresh process: with the cache, compile and warm-up fell by 77% on a warm
  process. JAX 0.10 never caches a program containing a host callback, so
  there a run with a `LoggingConfig` recompiles in each new process; JAX
  0.11 caches it. CI restores each
  job's cache from the last push to `main`.
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
