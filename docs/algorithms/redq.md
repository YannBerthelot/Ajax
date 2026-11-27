# REDQ

Randomized Ensembled Double Q-learning is SAC with many critics and many
critic updates per environment step. Each critic target takes the minimum
over a random subset of the target critics, which keeps over-estimation in
check even at a high update-to-data ratio, and the actor maximises the
ensemble's mean value.

**Paper:** Chen, Wang, Zhou and Ross, *Randomized Ensembled Double
Q-Learning: Learning Fast Without a Model*, ICLR 2021
([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)). Ajax follows its
Algorithm 1 and, where the authors' code is more specific, the code:
`watchernyu/REDQ` at `7b5d1bf`.

## When to use it

REDQ suits tasks where environment steps are scarce and compute is not: by
default it makes 20 critic steps on an ensemble of 10 critics per
environment step (with one environment), so it learns more from each
sample than [SAC](sac.md), at far more compute per step.

| Feature | REDQ |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a `tanh`-squashed Gaussian, mapped onto the environment's bounds |
| Memory (`memory=`) | yes, on replayed sequences, without `expert_policy` or most extensions ([Memory](../guide/memory.md)) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_warmup_action`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
# doc: skip (20 critic steps of 10 critics per update: minutes on a CPU)
from ajax import REDQ
from ajax.logging import LoggingConfig

# 10 critics, targets from the min over 2 of them, 20 critic steps per update.
agent = REDQ("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(log_frequency=2_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

An SVGD-style variant, which pushes the critics apart in function space,
is the `KernelRepulsion` extension, an Ajax research feature:

```python
from ajax import REDQ
from ajax.extensions.ensemble import KernelRepulsion

agent = REDQ("Pendulum-v1", extensions=(KernelRepulsion(coef=0.1),))
```

## Key hyperparameters

The defaults are the authors' code's, but for the target entropy (see
[Differences from the reference](#differences-from-the-reference)). The
[API](#api) section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (epsilon `1e-8`, torch's) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers |
| `gamma` | `0.99` | discount |
| `tau` | `0.005` | Polyak rate of the target critics |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `None` | rows per update: 256, or 64 sequences with `memory=` |
| `learning_starts` | `5_000` | steps of uniform random actions before the first update |
| `num_critics` | `10` | critics in the ensemble |
| `subset_size` | `2` | target critics the target takes the minimum over |
| `num_critic_updates` | `20` | critic steps per update, each on a fresh batch |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy per action dimension; `"mbpo"` for MBPO's per-task value |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `0`, `64`, `False` | recurrent memory and its sequence replay ([Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground ([Environments](../guide/environments.md)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's (`Eval/expert bias`) |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

Cloning an expert into the actor before training is the `CloningPretrain`
extension; a cloning term during training is `ImitationLoss`
([Expert guidance](../guide/expert-guidance.md)).

## Presets

`REDQ.preset(task, env_id)` sets MBPO's target entropy for the task, the
authors' code's default (`redq/algos/core.py:11-13`;
[Presets](../guide/training.md#presets)), as `target_entropy_per_dim`: the
whole action's value over the task's action dimensions.

| Preset | Whole action | Action dimensions | `target_entropy_per_dim` |
| --- | --- | --- | --- |
| `Hopper` | -1 | 3 | -1/3 |
| `HalfCheetah` | -3 | 6 | -0.5 |
| `Walker2d` | -3 | 6 | -0.5 |
| `Ant` | -4 | 8 | -0.5 |
| `Humanoid` | -2 | 17 | -2/17 |

## Differences from the reference

- **Target entropy.** The default is `-1` per action dimension, the
  authors' code's `"auto"` option (`redq_sac.py:55-56`), since its own
  default, MBPO's, is defined for five tasks only.
  `target_entropy_per_dim="mbpo"` takes MBPO's value instead for the Brax
  ids `hopper`, `halfcheetah`, `walker2d`, `ant` and `humanoid`, and raises
  on any other environment; the presets give the same values by task.
- **Where the code departs from the paper,** Ajax follows the code, in four
  choices. The actor step reads the critics as they were
  before the last critic step, since the code computes the actor loss
  before that step's optimiser update. The critic loss is the sum over the
  critics of each one's mean squared error. `alpha_init=1.0` matches the
  code's `log_alpha` starting at 0. And the code then reads `alpha=0.2`
  until its first temperature step (its constructor sets the temperature
  twice), so the first update's critic targets and actor loss read 0.2 in
  Ajax too.

## Notes

- **One update.** `num_critic_updates` critic steps, each on a fresh
  replay batch and each followed by a target update; then one actor step
  and one temperature step on the last batch.
- **Update-to-data ratio.** Each iteration steps every environment once,
  then makes one update, so the ratio of critic steps to environment steps
  is `num_critic_updates / n_envs`: 20 with the defaults.
- **Initialisation.** As in the authors' code, every layer of the actor
  and the critics, the mean and log-std layers included, starts
  Xavier-uniform with a zero bias, each critic of the ensemble drawn on its
  own. With `memory=`, REDQ builds Ni et al. (2022)'s recurrent networks
  and their initialisation, by analogy with their SAC.
- **Actions and evaluation.** As SAC; evaluation uses the mean action. The
  log-probability subtracts `log(1 - tanh(u)^2 + 1e-6)`, as the authors'
  code does (`core.py:232`). The critics read actions in the environment's
  units, as the authors' critics do (`act_limit * tanh`, `core.py:238`).
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **Diagnostics.** REDQ logs how spread out its ensemble is
  (`ensemble_q_std`, `mean_pairwise_q_dist`), computed without gradient.

## Results

The repository records no learning curves for REDQ yet. A conformance
check (`tests/conformance/test_redq_original.py`) runs one training step
of Ajax's REDQ and of `watchernyu/REDQ` from the same state, with the same
batches, subsets and noises, and compares every critic and target, the
actor, the temperature and their optimisers, the first step (temperature
0.2) included. The probes (`tests/probing`) train it briefly on tiny
environments with known answers: its targets, discounting, time limits,
memory, extension phases, and a run split by a checkpoint matching the run
left whole. [Performance](../guide/performance.md) compares its speed with
the other agents'.

## API

::: ajax.agents.REDQ.REDQ.REDQ
