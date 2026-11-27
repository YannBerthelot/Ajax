# CrossQ

CrossQ is SAC without target networks. Its actor and critics are
batch-renormalised, and the critics are trained on one pass over the
replayed rows and their next states together, so that both share the batch
statistics; the bootstrap reads the second half of that pass, held
constant. The actor and the temperature step on every third update.

**Paper:** Bhatt et al., *CrossQ: Batch Normalization in Deep Reinforcement
Learning for Greater Sample Efficiency and Simplicity*, ICLR 2024
([arXiv:1902.05605](https://arxiv.org/abs/1902.05605)). Where the authors'
code is more specific, Ajax follows the code: `adityab/CrossQ` at
`a318a26`.

## When to use it

CrossQ suits the tasks [SAC](sac.md) suits. It keeps SAC's one update per
step and drops the target networks, relying on wide batch-renormalised
critics instead. Those critics make each step costly: on a CPU, narrow
them (`critic_width=256`), or train on a GPU
([Performance](../guide/performance.md)).

| Feature | CrossQ |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a `tanh`-squashed Gaussian, mapped onto the environment's bounds |
| Memory (`memory=`) | no: the paper defines no recurrent batch normalisation |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | no: its networks are set by `actor_width` and `critic_width` |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_warmup_action`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import CrossQ
from ajax.logging import LoggingConfig

# Narrow critics for a CPU; the authors' are 2048 wide.
agent = CrossQ("Pendulum-v1", critic_width=256, learning_starts=1_000)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

## Key hyperparameters

The defaults are those of the authors' `train.py -algo crossq`. The
[API](#api) section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `learning_rate` | `1e-3` | the actor's, the critics' and the temperature's Adam step size |
| `alpha_learning_rate` | `None` | the temperature's own step size, when given |
| `adam_beta_1` | `0.5` | the actor's and the critics' Adam beta1 (the temperature's stays 0.9) |
| `actor_width`, `critic_width` | `256`, `2048` | units of the two hidden layers |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | rows per update |
| `learning_starts` | `5_000` | steps of uniform random actions before the first update |
| `num_critics` | `2` | critics; the target takes their minimum |
| `policy_delay` | `3` | updates per actor and temperature step |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy per action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

## Presets

`CrossQ.preset("Swimmer", env_id)` discounts at `gamma=0.9999`, the one
task the authors set apart (`train.py:200`, on Swimmer-v4;
[Presets](../guide/training.md#presets)). Every other task takes the
defaults.

## Differences from the reference

None in the update: one update of each from the same state agrees in every
parameter, running statistic and Adam moment, to within float32 rounding
([Results](#results)). Ajax follows the authors' code even where it
departs from SAC: its temperature loss has `alpha` where SAC's has
`log alpha` (`sbx/sac/sac.py:414-416`), and, as in their layer, Batch
Renorm's gradient is NaN where a batch's variance is zero.

## Notes

- **Batch Renorm.** Each network normalises its input and each hidden ReLU
  layer's output with `BatchRenorm` (the `"batchrenorm"` architecture
  token), the authors' layer: plain batch norm for its first 100,000
  training passes, then renormalised towards the running statistics (r
  clipped to 3, d to 5), momentum 0.99. Only the losses run it on the
  batch's statistics; acting, evaluation, the target's next actions and
  the critics in the actor's loss read the running ones.
- **No target networks.** The critics keep no target parameters, so
  `IBRL` and `LCBGatedBootstrap`, which bootstrap on target critics, are
  refused.
- **Initialisation.** As in the authors' code (flax's defaults):
  lecun-normal kernels and zero biases on every layer, the mean and
  log-std heads included.

## Results

The repository records no learning curves for CrossQ yet. A conformance
check (`tests/conformance/test_crossq_original.py`) runs one update of
Ajax's CrossQ and of `adityab/CrossQ` from the same state, batch and
draws: a critic step, an actor and temperature step, and the same past
Batch Renorm's warm-up, comparing every parameter, running statistic and
Adam moment. The probes (`tests/probing`) train it briefly on tiny
environments with known answers. [Performance](../guide/performance.md)
compares its speed with the other agents'.

## API

::: ajax.agents.CrossQ.CrossQ.CrossQ
