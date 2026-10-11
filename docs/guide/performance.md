# Performance

A run's time splits into two parts: compiling the program, and running it.
For short runs and hyperparameter sweeps the first often dominates; for long
runs, the second. This page says what Ajax does about each and how to get
the most out of a machine. The numbers come from
[PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md), where every
measurement is described in full. They belong to the machine they were
measured on: use them for orders of magnitude, not as promises.

## The compile cache

`train` compiles the whole run, environment included, into one program.
That takes seconds to tens of seconds, and a fresh Python process would pay
it every time. So `import ajax` points JAX's persistent compile cache at
`~/.cache/ajax/jax_compile_cache`: a later process that compiles the same
program loads it from disk instead.

On the report's Plane3DCircle SAC scenario, a second launch of the same
script took 3.35 s instead of 16.26 s, compile and warm-up fell from 21.5 s
to 4.9 s, and peak GPU memory fell from 302 MB to 15 MB, because the
compiler's scratch memory is never allocated. It matters most where every
trial is a new process, as in a hyperparameter sweep.

What to know about it:

- **A logged run is not cached.** JAX never stores a program that contains
  a host callback, and the [logging worker](logging.md) is fed by one. A run
  with a `LoggingConfig` compiles in every new process.
- **A different run is a different program.** The budget, the agent's
  settings and the shapes are part of the program, so changing any of them
  compiles again. A repeated identical run hits the cache.
- **One subdirectory per machine and version.** Entries are kept per CPU
  model, architecture, and Python, JAX and jaxlib versions, so environments
  with different versions, or machines sharing a home directory, do not
  load each other's programs. Nothing is ever deleted: clear the directory
  yourself if it grows too large.
- `AJAX_JAX_COMPILE_CACHE_DIR` moves the cache; `AJAX_NO_COMPILE_CACHE=1`
  turns it off.

## Seeds are nearly free

`train` runs all seeds as one vmapped program. Up to the point where the
device is busy, more seeds cost almost no extra time. On the report's
benchmarks:

| Seeds | Pendulum, wall time | Plane3DCircle, wall time |
| --- | --- | --- |
| 1 | 9.81 s | 15.77 s |
| 4 | 10.51 s | 17.46 s |
| 16 | 10.43 s | 18.75 s |
| 32 | 11.08 s | 19.12 s |
| 100 | | 18.37 s |

Thirty-two seeds give 28 times the throughput of one on Pendulum, and a
hundred seeds 86 times on Plane3DCircle. The fixed cost of compiling and
launching is shared by every seed, so running seeds in separate calls pays
it again for each. On Plane3DCircle each seed past 16 added roughly 10-11 MB
of GPU memory: memory, not compute, limits the seed count there.

The scaling depends on the workload. In the report's scan of real
hyperparameter sweeps (wide networks, critic ensembles), going from 20 to 50
seeds made the time per seed and step worse for several method and
environment pairs, by up to 81%, once the seeds' working set outgrew the
GPU's caches. Measure your own configuration before settling on a seed
count.

## Parallel environments

`n_envs` copies of the environment step together inside each seed, as one
vectorised operation. Unlike seeds, they change the algorithm: an
off-policy agent makes one update per iteration, so more copies at a fixed
budget mean fewer updates ([Training](training.md)). Pick `n_envs` for the
learning you want, then use seeds to fill the device.

## CPU or GPU

Small networks on gymnax tasks train well on a CPU: every benchmark number
below is a CPU number. A GPU pays off when each step holds more parallel
work: many seeds, many environment copies, wide networks, or a physics
simulator such as Brax or Playground. The logging worker always runs on the
CPU, so it never takes GPU memory. On a CPU, Playground uses MJX's pure-JAX
implementation.

To keep a run on the CPU of a machine with a GPU, set `JAX_PLATFORMS=cpu`
(and `CUDA_VISIBLE_DEVICES=""` to hide the GPU entirely). The repository's
tests and benchmarks run this way, so they never compete with an experiment
on the GPU.

## Logging

Evaluation runs `num_episode_test` whole episodes every `log_frequency`
steps, on top of training: evaluate as often as your curves need, not more.
Without a `LoggingConfig` the evaluation is not compiled at all, and the
program can be cached.

## Agent speeds

`benchmarks/agent_bench.py` times every agent but UDRL on a fixed task
(CartPole for DQN, PQN, PPO and DreamerV3, a Pendulum tracking task for APG,
Pendulum for the others), 4 environment copies, 8,000 steps, 3 trials, the
compile cache off, on a Mac's CPU. Median training speed and compile time,
after the 2026 refactoring:

| Agent | Steps per second | Compile and warm-up (s) |
| --- | --- | --- |
| PQN | 7,332 | 1.01 |
| DQN | 6,375 | 0.94 |
| TD3 | 6,163 | 1.30 |
| APG | 5,494 | 1.48 |
| SAC | 5,051 | 4.71 |
| ASAC | 5,006 | 1.55 |
| REDQ | 4,868 | 1.58 |
| APO | 4,486 | 2.03 |
| PPO | 4,281 | 1.98 |
| AVG | 4,056 | 1.58 |
| DreamerV3 | 273 | 12.31 |
| TDMPC2 | 243 | 7.25 |

DreamerV3 and TD-MPC2 run at reduced presets here (DreamerV3's smallest
size, `"1m"`; a TD-MPC2 narrowed to 64-unit layers and a smaller planner).
SAC, TD3, REDQ and ASAC run below their default `learning_starts` of 10,000,
so their figures time collection, not updates. At its paper's
defaults TD-MPC2 would take minutes per bench run on a CPU: it updates its
model and plans at every environment step. Speeds on your machine will
differ; the ratios between agents are the useful part.
[Developer performance](../developer/performance.md) explains how to run
the benchmark and the 10% no-regression rule every change follows.

## Other measured costs

- **Playground fresh resets.** Drawing a new initial state per episode
  (the default) costs a reset per episode. On CheetahRun, whose reset runs a
  200-step stabilisation, SAC's iterations took about 2% longer, and compile
  plus first run grew from 36 s to 59 s per cold process.
- **Brax resets.** Ajax's Brax wrapper computes a reset only on steps where
  some copy's episode ended. `differentiable_reset=True` computes one every
  step, which cost about 7-13% of a step's time on CPU in the report's
  measurement.
- **Memory blocks** in the replay agents: see [Memory](memory.md).
