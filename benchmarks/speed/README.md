# Training speed against other libraries

How fast does Ajax train, against the libraries people use for the same
algorithms? Each cell times one algorithm on one task in one library:
environment steps per second, over every seed, during a stretch of training
once the updates run. `docs/guide/performance.md` shows the figure and the
table; `results/speed.jsonl` holds the records.

| Family | Algorithm | Tasks | Libraries |
|---|---|---|---|
| SAC and TD3 | SAC | Pendulum-v1, HalfCheetah-v5 | Ajax, SB3, SBX, CleanRL, Brax |
| | TD3 | Pendulum-v1, HalfCheetah-v5 | Ajax, SB3, SBX, CleanRL |
| PPO | PPO | CartPole-v1, HalfCheetah-v5 | Ajax, SB3, SBX, CleanRL, PureJaxRL, Brax (HalfCheetah only: no discrete actions) |
| DQN and PQN | DQN | CartPole-v1 | Ajax, SB3, SBX, CleanRL, PureJaxRL |
| | PQN | CartPole-v1 | Ajax, CleanRL, purejaxql |

## The same work in every library

`speed_specs.py` holds one set of settings per algorithm and task, set in each
library under its own names: the number of environments, rollout and batch
sizes, epochs, minibatches, network sizes, how often the networks update. They
are CleanRL's defaults where its scripts and PureJaxRL's agree (PPO on
CartPole, DQN), the defaults SB3 and CleanRL share for SAC, TD3 and PPO on
MuJoCo, and purejaxql's configuration for PQN. `EXCEPTIONS` lists what a
library cannot change and keeps of its own. Nothing was tuned towards Ajax:
the settings were fixed before any timing. No library evaluates during the
timed stretch.

## Each library its own way

| Library | Runs as | 10 seeds |
|---|---|---|
| Ajax | `train(seed=...)` | one program, the seeds vmapped |
| PureJaxRL, purejaxql | `jax.jit(jax.vmap(make_train(config)))` | one program, the seeds vmapped |
| SB3, SBX | `Algorithm("MlpPolicy", env).learn(n)`, Gymnasium environments in a DummyVecEnv | ten processes at once, torch at one thread each |
| CleanRL | its single-file scripts, unmodified (`../reference/vendor/cleanrl`): JAX versions for TD3 and DQN, which its docs call faster | ten processes at once, torch at one thread each |
| Brax | `brax.training.agents.{sac,ppo}.train` | ten processes at once on the CPU; on a GPU, one after another: its one-seed speed |

A single seed runs with each library's own threading. JAX's CPU thread pool
takes the run's four cores in every process; torch's threads are set to one
for ten processes. A library whose ten processes run slower together than one
after another is shown at its one-after-another speed, its one-seed speed (the
records keep both). On a GPU only Ajax and the JAX libraries are timed; SB3,
SBX and CleanRL step their environments in Python on the CPU.

On HalfCheetah, SB3, SBX and CleanRL step C MuJoCo through Gymnasium
(CleanRL's scripts need Gymnasium 0.29's vector API, so they step
HalfCheetah-v4, the same model). Ajax, Brax and PureJaxRL step
[gymnasium-mjx](https://github.com/YannBerthelot/gymnasium-mjx)'s
HalfCheetah-v5 on MJX, MuJoCo in JAX: the same task, so the three train on the
same physics code. On a CPU, C MuJoCo steps one environment faster than MJX.

## Versions

`envs/<library>/` pins each library in a uv project with its lockfile, at
its latest release. JAX is Ajax's (0.11.2) everywhere but in Brax's project:
Brax's trainers call `jax.device_put_replicated`, which JAX 0.10 removed, so
it pins 0.9.2. torch is 2.15.0 for SB3, SBX and CleanRL. The projects of
`../reference/envs` serve the learning comparisons, on gymnax environments
and an older JAX; changing them would change the fingerprints of their
recorded runs, so the speed comparison has its own.

Two seams adapt third-party code without editing it, as `../reference`
does: gymnax 1.0's step returns `terminated, truncated`, and PureJaxRL and
purejaxql, written for gymnax 0.0, get one `done`; PureJaxRL's `dqn.py` calls
`jax.tree_map`, which JAX 0.6 removed, and gets `jax.tree.map`, the same
function.

## How a stretch is timed

The stretch starts once the updates run, after `warmup` steps per seed (past
`learning_starts`, and past the first compilation), and lasts `segment`
steps per seed. It counts every step and every update between its ends.

- **SB3, SBX, CleanRL**: their environments come from
  `gymnasium.make("speed_clock:speed/<id>")`, which wraps the task in a clock
  (`speed_clock.py`). The clock counts the process's steps; at the start it
  waits for the other processes of the run, so ten processes time the same
  stretch, and at the end it writes the seconds. Ten processes' speed is
  their steps over the time from the first start to the last end.
- **Brax**: two training epochs of `segment` steps; Brax reports each
  epoch's steps per second, evaluation left out. The first compiles; the
  second, after the same wait, is timed.
- **PureJaxRL, purejaxql**: a run is one compiled program, so two are timed,
  of `warmup` and of `warmup + segment` steps, each compiled first; the
  difference of their run times is the stretch's.
- **Ajax**: a first call trains `warmup` steps and compiles; two calls then
  resume from its state, one of a single update's steps and one of `segment`
  steps, each timed less the seconds JAX spends compiling its program. The
  difference of the two is the stretch's: a call's fixed cost (seconds on
  HalfCheetah) cancels, as the set-up of the other libraries is left out.

The segments are sized so that a stretch lasts about a minute (twenty
seconds at least) on four of bob's cores (`SEGMENTS` in `speed_specs.py`). Each
record notes how busy its cores and their SMT siblings were meanwhile, against
the run's own CPU time, so a stretch that shared its cores shows.

## Running it

Every library's project syncs once, then `drive.py` times one cell, from the
repository root on the machine to time:

```bash
for library in sb3 sbx cleanrl; do
    uv sync --locked --project benchmarks/speed/envs/$library
done
for library in brax purejaxrl purejaxql; do  # --extra cuda for the GPU
    uv sync --locked --project benchmarks/speed/envs/$library --extra cuda
done
uv sync --locked --extra gymnasium-mjx --extra cuda
python3 benchmarks/speed/drive.py SAC HalfCheetah-v5 SB3 --seeds 10 --cores 44-47 \
    --warmup 2000 --segment 3000
python3 benchmarks/speed/drive.py --list cpu   # every cell, with its stretch
uv run python benchmarks/speed/speed_figure.py  # the figure and the docs tables
```

`--gpu 0` times a cell on GPU 0 instead of the CPU (`--list gpu` gives the
GPU's cells). `speed_figure.py` takes the latest record of each cell.
