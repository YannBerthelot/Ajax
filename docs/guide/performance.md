# Performance

A run's time splits into two parts: compiling the program, and running it.
For short runs and hyperparameter sweeps the first often dominates; for long
runs, the second. This page says what Ajax does about each and how to get
the most out of a machine. The numbers come from
[PERFORMANCE_REPORT.md](../../PERFORMANCE_REPORT.md), where every
measurement is described in full. They belong to the machine they were
measured on: use them for orders of magnitude, not as promises. They are
CPU measurements unless a line says otherwise.

## The compile cache

`train` compiles the whole run, environment included, into one program.
That takes seconds to tens of seconds, and a fresh Python process would pay
it every time. So `import ajax` points JAX's persistent compile cache at
`~/.cache/ajax/jax_compile_cache`: a later process that compiles the same
program loads it from disk instead.

On the report's Plane3DCircle SAC scenario, run on a GPU, a second launch
of the same script took 3.35 s instead of 16.26 s, compile and warm-up fell from 21.5 s
to 4.9 s, and peak GPU memory fell from 302 MB to 15 MB, because the
compiler's scratch memory is never allocated.

What to know about it:

- **It serves relaunches, not new trials.** A rerun of the same script, or
  a repeated or resumed trial, loads its program from the cache. A new trial
  of a hyperparameter sweep is a different program: its hyperparameters are
  compiled in as constants, so it compiles from scratch. The budget, the
  agent's settings and the shapes are all part of the program.
- **A logged run is cached from JAX 0.11.** The [logging worker](logging.md)
  is fed by a host callback, and JAX 0.10 never stores a program that
  contains one, so on 0.10 a run with a `LoggingConfig` compiles in every
  new process.
- **One subdirectory per machine and version.** Entries are kept per CPU
  model, architecture, and Python, JAX and jaxlib versions, so environments
  with different versions, or machines sharing a home directory, do not
  load each other's programs.
- **Nothing trims it on its own.** Clear the directory yourself if it grows
  too large, or call `ajax._compile_cache.prune(root, max_bytes)`, which
  deletes the least recently used files until the rest fit (the repository's
  CI does this once per job).
- `AJAX_JAX_COMPILE_CACHE_DIR` moves the cache; `AJAX_NO_COMPILE_CACHE=1`
  turns it off.

## Seeds share one program

`train` runs all seeds as one vmapped program, so the fixed cost of
compiling and launching is paid once for all of them; running seeds in
separate calls pays it again for each. Until the device is busy, extra
seeds add little time. An early GPU measurement of SAC, from May 2026 on an
older version of Ajax, with small replay buffers and short runs
(`benchmarks/seed_scaling.jsonl`, `benchmarks/seed_scaling_p3d.jsonl`):

| Seeds | Pendulum, wall time | Plane3DCircle, wall time |
| --- | --- | --- |
| 1 | 9.81 s | 15.77 s |
| 4 | 10.51 s | 17.46 s |
| 16 | 10.43 s | 18.75 s |
| 32 | 11.08 s | 19.12 s |
| 100 | | 18.37 s |

Thirty-two seeds gave 28 times the throughput of one on Pendulum, and a
hundred seeds 86 times on Plane3DCircle. Each seed of a replay agent keeps
its own buffer, so memory grows with the number of seeds.

The scaling depends on the workload. In the report's scan of real
hyperparameter sweeps (wide networks, critic ensembles), going from 20 to 50
seeds made the time per seed and step worse for several method and
environment pairs, by up to 81%. Measure your own configuration before
settling on a seed count.

## Parallel environments

`n_envs` copies of the environment step together inside each seed, as one
vectorised operation. Unlike seeds, they change the algorithm: an
off-policy agent makes one update per iteration, so more copies at a fixed
budget mean fewer updates ([Training](training.md)). Pick `n_envs` for the
learning you want, then use seeds to fill the device.

## CPU or GPU

Small networks on gymnax tasks train well on a CPU: every agent-speed
number below is a CPU number, and Ajax makes no claim about its speed on a
GPU. A GPU has more room for parallel work: many seeds, many environment
copies, wide networks, or a physics simulator such as Brax or Playground.
The logging worker always runs on the CPU, so it never takes GPU memory. On
a CPU, Playground uses MJX's pure-JAX implementation.

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

`benchmarks/agent_bench.py` times every agent on a fixed task (CartPole for
DQN, PQN, PPO and DreamerV3, a Pendulum tracking task for APG, Pendulum for
the others, DAgger imitating a PD controller), 4 environment copies, 8,000
steps, one seed, 5 trials, the compile cache off, on a Mac's CPU. Every
agent updates within the run: the replay agents start at step 1,000 instead
of their defaults (10,000 for SAC, TD3, REDQ and ASAC). Medians of the five
trials, from the baseline recorded on 2026-10-28:

| Agent | Steps per second, execution | Steps per second, whole run | Compile and warm-up (s) |
| --- | --- | --- | --- |
| DAgger | 42,406 | 9,508 | 0.8 |
| PQN | 36,260 | 9,040 | 0.9 |
| APO | 31,243 | 7,203 | 1.2 |
| APG | 28,105 | 6,385 | 1.3 |
| PPO | 25,391 | 6,698 | 1.3 |
| UDRL | 21,876 | 6,419 | 1.1 |
| DQN | 21,542 | 8,256 | 0.8 |
| AVG | 11,560 | 5,366 | 1.1 |
| TD3 | 4,063 | 2,911 | 1.2 |
| IQL | 3,145 | 2,319 | 1.1 |
| SAC | 3,094 | 2,302 | 4.6 |
| ASAC | 2,135 | 1,709 | 1.2 |
| DreamerV3 | 508 | 407 | 7.6 |
| TDMPC2 | 412 | 360 | 4.5 |
| TQC | 392 | 374 | 1.3 |
| CrossQ | 91 | 90 | 1.7 |
| REDQ | 63 | 63 | 1.2 |

"Execution" leaves out the time JAX spends tracing, lowering and compiling
the program; "whole run" includes it, because every trial builds a fresh
agent, whose program is traced again. REDQ is slow by design: from step
1,000 it makes 20 updates of 10 critics per environment step. CrossQ's
critics are 2,048 units wide, and TQC updates 5 critics of three 512-unit
layers. ASAC's and IQL's trials spread the most. DreamerV3 and TD-MPC2 run at reduced presets (DreamerV3's
smallest size, `"1m"`; a TD-MPC2 narrowed to 64-unit layers and a smaller
planner); at its paper's defaults TD-MPC2 would take minutes per bench run
on a CPU, since it updates its model and plans at every environment step.
Speeds on your machine will differ; the ratios between agents are the
useful part. [Developer performance](../developer/performance.md) explains
how to run the benchmark and the 10% no-regression rule every change
follows.

## Other measured costs

- **The action map.** Mapping every continuous action onto the
  environment's bounds ([Actions](environments.md#actions)) cost nothing
  measurable: SAC +0.1%, TD3 +2.8% and PPO +1.0% steps per second, within
  noise.
- **Playground fresh resets.** Drawing a new initial state per episode
  (the default) costs a reset per episode. On CheetahRun, whose reset runs a
  200-step stabilisation, SAC's iterations took about 2% longer, and compile
  plus first run grew from 36 s to 59 s per cold process.
- **Brax resets.** Ajax's Brax wrapper computes a reset only on steps where
  some copy's episode ended. `differentiable_reset=True` computes one every
  step, which cost about 7-13% of a step's time in the report's
  measurement.
- **Memory blocks** in the replay agents: see [Memory](memory.md).
