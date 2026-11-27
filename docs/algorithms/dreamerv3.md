# DreamerV3

DreamerV3 learns a model of the environment and trains its policy inside
that model. A recurrent world model learns, from replayed sequences, to
predict the next latent state, the reward and whether the episode
continues; an actor and a critic then learn mostly from trajectories the
model imagines.

**Paper:** Hafner, Pasukonis, Ba and Lillicrap, *Mastering Diverse Domains
through World Models*, arXiv:2301.04104v2, 2023; Nature, 2025
([arXiv:2301.04104v2](https://arxiv.org/abs/2301.04104v2)). Official code:
[danijar/dreamerv3](https://github.com/danijar/dreamerv3). Ajax reproduces
the code that produced the paper's numbers (commit `2411f7d`), plus one
later upstream bug fix (`29eb964`, the replay context). The
[world-model design notes](../world_models/README.md) give the full
specification, and [deviations.md](../world_models/deviations.md) lists
every version choice and every departure from the reference.

## When to use it

A world model reuses every transition many times, so DreamerV3 tends to
need fewer environment steps than a model-free agent, at a much higher
compute cost per step. [World models](../guide/world-models.md) compares
it with [TD-MPC2](tdmpc2.md), which plans with its model at every step.

| Feature | DreamerV3 |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes |
| Memory | built in: the world model is recurrent (`memory=` is not accepted) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Observations | flat vectors only; the environment must not normalise observations or rewards |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe` (the features join the observation the encoder reads and the decoder reconstructs) |
| Resume and checkpoints | yes; a resumed run continues the schedules and the replay, and equals the run left whole |

## Example

```python
# doc: skip (compiles and trains for about a minute on a CPU; tests/agents/DreamerV3/test_dreamerv3_agent.py covers this path)
from ajax import DreamerV3
from ajax.logging import LoggingConfig

# The smallest preset and a low train ratio, for a quick look on a CPU.
agent = DreamerV3("CartPole-v1", model_size="1m", n_envs=4, train_ratio=32,
                  batch_size=4, batch_length=16)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=0, n_timesteps=2_000, num_episode_test=2,
                                 logging_config=logging)
print(evaluations["Train/episodic mean reward"])  # the score the paper reports
print(evaluations["Eval/episodic mean reward"])
```

The paper's DMC-proprio protocol (Table 2): the 12M model, 16 environments, a
train ratio of 512, action repeat 2, and 500K environment frames, which is
250K rows at repeat 2:

```python
# doc: skip (the paper's full budget; benchmarks/world_models/paper_protocol.py runs it)
from ajax import DreamerV3

agent = DreamerV3("CheetahRun", model_size="12m", n_envs=16, train_ratio=512,
                  action_repeat=2, episode_length=1000)
state, _ = agent.train(seed=0, n_timesteps=250_000)  # 500K env frames
```

## Key hyperparameters

Every default is the paper-era value. `model_size` and `train_ratio` are
those of the reference's `dmc_proprio` configuration, its one for vector
observations (its defaults block has the 200m widths and a ratio of 32).

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 16 | environments stepped in parallel |
| `model_size` | `"12m"` | preset of `"1m"`, `"12m"`, `"25m"`, `"50m"`, `"100m"`, `"200m"`, `"400m"`; sets `units`, `deter`, `hidden`, `classes` |
| `train_ratio` | 512 | replayed steps trained per collected row |
| `batch_size`, `batch_length` | 16, 64 | sequences per batch, and their length |
| `imag_horizon` | 15 | imagined steps per start state |
| `return_horizon` | 333 | `1 / (1 - gamma)`: sets the discount |
| `lam` | 0.95 | lambda of the imagined returns |
| `learning_rate` | 4e-5 | a float or a schedule of the update count; a 1,000-update linear warmup (`warmup`) multiplies it |
| `agc`, `eps` | 0.3, 1e-20 | adaptive gradient clipping, and LaProp's epsilon |
| `actent` | 3e-4 | actor entropy scale |
| `free_nats` | 1.0 | free bits of the KL losses |
| `dyn_scale`, `rep_scale` | 1.0, 0.1 | dynamics and representation loss scales |
| `replay_capacity` | 5,000,000 | rows kept, over all environments |
| `replay_on_host` | False | keep the stored latents in pinned host memory instead of the accelerator's |
| `action_repeat`, `episode_length` | 1, 1000 | simulator steps per agent step, and per episode (Brax, Playground) |

The other arguments (network depths, `bins`, `unimix`, the other loss
scales, the slow critic, the return normaliser) are in the [API](#api)
below.

## Presets

`DreamerV3.preset("DMC", env_id)` builds the DMC-proprio configuration
above, the reference's `dmc_proprio` (`configs.yaml:44`, `:84`,
`:232-236` at `29eb964`; [Presets](../guide/training.md#presets)):
`model_size="12m"`, `train_ratio=512`, `n_envs=16` and `action_repeat=2`.
The reference's other domains (Atari, Crafter, DMLab, Procgen) observe
images, which Ajax's DreamerV3 does not encode yet.

## Differences from the reference

Each is in [deviations.md](../world_models/deviations.md), under its
number.

- **Synchronous, in float32** (D1, D3, D4). The reference acts and trains
  asynchronously, and computes in bfloat16.
- **One decoder head** for the flat observation vector (D5), where the
  reference has one per observation key; the loss is the same.
- **Everything checkpointed and seeded** (D2, D6).
- **The two-hot expectation** is computed in an exact form (D22). The
  reference's form adds noise to early predictions under JIT, which
  measurably changes the early actor entropy.
- **The replay** keeps one ring per environment, where the reference keeps
  one first-in first-out store over all its workers (D27).
- **The DMC tasks** run on MuJoCo Playground's MJX ports, whose physics
  differ from dm_control's (E18), so comparisons with the published curves
  are approximate.

## Notes

- **What `n_timesteps` counts.** One *row* per environment per vector step,
  reset rows included, as the reference's `step` does. `env_frames = rows *
  action_repeat` is logged next to it.
- **The score.** The paper reports the returns of the stochastic policy's
  training episodes, logged as `Train/episodic mean reward`. The evaluation
  also samples its actions, since the reference has no deterministic mode.
- **Actions.** Continuous actions are clipped to `[-1, 1]` and mapped onto
  the environment's bounds, as the reference's `ClipAction` and
  `NormalizeAction` wrappers do.
- **Replay memory.** Each row stores the observation, the action, the
  reward, the flags and the posterior latent: about 8.3 KB at 12M on a
  24-dimensional observation, so the DMC protocol's 250K rows take about
  2.1 GB per seed (3.2 GB with the training step's working set).
  `agent.replay_bytes_per_seed` gives the figure after `train`. Each
  environment's ring keeps `min(ceil(replay_capacity / n_envs), rows per
  env of the run)` rows, sized again by every `train` call, so that a run
  split into resumed calls keeps the replay of the run left whole.
- **`replay_on_host=True`.** The posterior latents are almost all of a row;
  with this option they stay in pinned host memory and the device holds
  little more than the working set. The results are the same, bit for bit,
  and each update is a little slower. XLA caps pinned host memory at 64 GB
  by default: set `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` higher for more.
- **Recompilation.** Widths, depths, counts, `bins`, `imag_horizon`,
  `batch_size`, `batch_length`, `train_ratio` and `replay_capacity` fix the
  compiled program; changing one recompiles.

## Results

On a CPU, with the 1M preset and 20,000 rows, Pendulum swings up from
12,000 rows on (final evaluation -222, against about -1,200 at the start).
On CartPole, Ajax was compared with the reference code itself on the same
protocol, over three pre-registered rounds. Once both used the same
two-hot expectation, the training statistics' seed ranges overlapped the
reference's in at least 52 of 58 windows, and all five Ajax seeds passed
the CartPole check that followed, as did all five of the reference's.
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)
has the full account. The GPU validation against the published DMC curves
is specified in [VALIDATION.md](../world_models/VALIDATION.md) and has not
been run yet.

The probes (`tests/probing`) train DreamerV3 briefly on tiny environments
with known answers: discounting, a continuous advantage, and a risky or
safe choice it can only weigh in imagination.

## API

::: ajax.agents.DreamerV3.DreamerV3.DreamerV3
