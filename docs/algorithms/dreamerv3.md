# DreamerV3

DreamerV3 learns a model of the environment and trains its policy inside that
model. A recurrent world model learns, from replayed sequences, to predict the
next latent state, the reward and whether the episode continues. An actor and a
critic then learn mostly from trajectories imagined by the model. All three
are trained together, one gradient step per update, with the LaProp optimizer
and adaptive gradient clipping.

**Paper:** Hafner, Pasukonis, Ba and Lillicrap, *Mastering Diverse Domains
through World Models*, arXiv:2301.04104v2, 2023; Nature, 2025
([arXiv:2301.04104v2](https://arxiv.org/abs/2301.04104v2)). Official code:
[danijar/dreamerv3](https://github.com/danijar/dreamerv3).

Ajax reproduces the code that produced the paper's numbers (commit `2411f7d`)
plus one later upstream bug fix (`29eb964`, the replay context). The
[world-model design notes](../world_models/README.md) give the full
specification, and [deviations.md](../world_models/deviations.md) lists every
version choice and every departure from the reference.

## Can I use?

| Feature | DreamerV3 |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes |
| Memory | built in: the world model is recurrent (`memory=` is not accepted) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Observations | flat vectors only; the environment must not normalise observations or rewards |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update` (after every update), `eval_metrics` |
| Resume and checkpoints | yes; a resumed run continues the schedules and the replay, and equals the uninterrupted run (test) |

## Example

```python
# doc: skip (compiles and trains for about a minute on a CPU; tests/agents/DreamerV3/test_dreamerv3_agent.py covers this path)
from ajax import DreamerV3
from ajax.logging.wandb_logging import LoggingConfig

# The smallest preset and a low train ratio, for a quick look on a CPU.
agent = DreamerV3("CartPole-v1", model_size="1m", n_envs=4, train_ratio=32,
                  batch_size=4, batch_length=16)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
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

Every default is the paper-era value.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 16 | environments stepped in parallel |
| `model_size` | `"12m"` | preset of `"1m"`, `"12m"`, `"25m"`, `"50m"`, `"100m"`, `"200m"`, `"400m"`; sets `units`, `deter`, `hidden`, `classes` |
| `train_ratio` | 512 | replayed steps trained per collected row |
| `batch_size`, `batch_length` | 16, 64 | sequences per batch, and their length |
| `imag_horizon` | 15 | imagined steps per start state |
| `return_horizon` | 333 | `1 / (1 - gamma)`: sets the discount |
| `lam` | 0.95 | lambda of the imagined returns |
| `learning_rate` | 4e-5 | a float or a schedule of the update count; a 1,000-update linear warmup multiplies it |
| `agc`, `eps` | 0.3, 1e-20 | adaptive gradient clipping, LaProp's epsilon |
| `actent` | 3e-4 | actor entropy scale |
| `free_nats` | 1.0 | free bits of the KL losses |
| `dyn_scale`, `rep_scale` | 1.0, 0.1 | dynamics and representation loss scales |
| `replay_capacity` | 5,000,000 | rows kept, over all environments |
| `action_repeat`, `episode_length` | 1, 1000 | simulator steps per agent step, and per episode (Brax, Playground) |

The other arguments (network depths, `bins`, `unimix`, the remaining loss
scales, the slow critic and the return normaliser) are listed in the API
below.

## Notes

- **What `n_timesteps` counts.** One *row* per environment per vector step,
  reset rows included, as the reference's `step` does. `env_frames = rows *
  action_repeat` is logged next to it.
- **The score.** The paper reports the returns of the stochastic policy's
  training episodes, logged as `Train/episodic mean reward`. The evaluation
  also samples its actions, since the reference has no deterministic mode,
  and draws fresh keys at each evaluation (new initial states and action
  noise); a rerun of the same seed draws the same ones.
- **Actions.** Continuous actions are clipped to `[-1, 1]` and mapped onto
  the environment's bounds, the reference's `ClipAction` and
  `NormalizeAction`.
- **Memory use.** Each replay row stores the observation, the action, the
  reward, the flags and the posterior latent: about 8.3 KB at 12M on a
  24-dimensional observation, so the 250K rows of the DMC protocol take about
  2.1 GB per seed, and about 3.2 GB with the training step's working set.
  `agent.replay_bytes_per_seed` gives the figure after `train`. Running several
  seeds multiplies it. The replay keeps `min(ceil(replay_capacity / n_envs),
  rows per env of the run)` rows per env (a capacity in rows, deviation D27),
  sized again by every `train` call, so that a run split into resumed calls
  keeps the replay of the uninterrupted run.
- **Recompilation.** Widths, depths, counts, `bins`, `imag_horizon`,
  `batch_size`, `batch_length`, `train_ratio` and `replay_capacity` fix the
  compiled program; changing one recompiles.
- **Main deviations** (all in [deviations.md](../world_models/deviations.md)):
  training is synchronous and in float32 (the reference is asynchronous and
  computes in bfloat16); one decoder head for the flat observation; everything
  is checkpointed and seeded. The two-hot expectation is computed in an exact
  form (D22); the reference's form adds noise to early predictions under JIT,
  which measurably changes the early actor entropy.
- The DMC tasks run on MuJoCo Playground's MJX ports, whose physics differ
  from dm_control's (E18), so comparisons with the published curves are
  approximate.

## Results

Recorded on CPU in [PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md),
with the 1M preset and 20,000 rows, before evaluations drew fresh keys:
Pendulum swings up from 12,000 rows on (final evaluation -222, against about
-1,200 at the start). On CartPole Ajax
was compared with the reference code itself on the same protocol, over three
pre-registered rounds; once both used the same two-hot expectation, the
training statistics' seed ranges overlapped the reference's in at least 52 of
58 windows, and all five Ajax seeds passed the redefined CartPole check, as did
all five of the reference's. The GPU validation against the published DMC
curves is specified in [VALIDATION.md](../world_models/VALIDATION.md) and has
not been run yet. The probes (`tests/probing`) train DreamerV3 briefly on
tiny environments with known answers (discounting, a continuous advantage,
and a risky-or-safe choice it can only weigh in imagination), and its own
tests (`tests/agents/DreamerV3`) check that a run split by a checkpoint
equals the run left whole. On the
CPU speed benchmark (`benchmarks/agent_bench.py`: CartPole, 4 environments,
8,000 rows, a smaller model than the paper's) DreamerV3 ran at 367 rows per
second of execution (275 with compilation) at main `a8a13c2`.

## API

::: ajax.agents.DreamerV3.DreamerV3.DreamerV3
