# REDQ

Randomized Ensembled Double Q-learning is SAC with many critics and many
critic updates per environment step. Each critic target takes the minimum
over a random subset of the target critics, which keeps over-estimation in
check even at a high update-to-data ratio, and the actor maximises the
ensemble's mean value. Each environment step costs far more compute than
SAC's, in exchange for many more critic updates per step.

**Paper.** Chen et al., *Randomized Ensembled Double Q-Learning*, 2021
([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)). The update follows its
Algorithm 1.

## Can I use?

| | REDQ |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions, `expert_policy` or cloning |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

## Example

```python
# doc: skip (about 2.5 minutes on a laptop CPU: 20 critic steps per update)
from ajax import REDQ
from ajax.logging.wandb_logging import LoggingConfig

# 10 critics, targets from the min over 2 of them, 20 critic steps per update.
agent = REDQ("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=2_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

An SVGD-style variant, which pushes the critics apart in function space, is
the `KernelRepulsion` extension, an Ajax research feature:

```python
from ajax import REDQ
from ajax.extensions.ensemble import KernelRepulsion

agent = REDQ("Pendulum-v1", extensions=(KernelRepulsion(coef=0.1),))
```

## Key hyperparameters

Defaults are read from `REDQ.__init__`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target critics |
| `num_critics` | `10` | critics in the ensemble |
| `subset_size` | `2` | target critics the target takes the minimum over |
| `num_critic_updates` | `20` | critic steps per update, each on a fresh minibatch |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `expert_policy`, `pre_train_n_steps` | `None`, `0` | behaviour-clone an expert before training; the expert also feeds an evaluation metric |
| `actor_cloning_epochs`, `actor_cloning_lr`, `actor_cloning_batch_size` | `10`, `1e-3`, `64` | that cloning's settings |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **One update.** `num_critic_updates` critic steps, each on a fresh replay
  minibatch and each followed by a target update; then one actor step and
  one temperature step on the last minibatch.
- **Update-to-data ratio.** Each iteration steps every env once, then makes
  one update, so the ratio of critic steps to environment steps is
  `num_critic_updates / n_envs`: 20 with the defaults.
- **Cost.** The first example took about 2.5 minutes on a laptop CPU; the
  same run of SAC took about 8 seconds.
- **Actions and evaluation.** As SAC: a `tanh`-squashed Gaussian in
  `[-1, 1]`, clipped to the environment's bounds, not rescaled; evaluation
  uses the mean action.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **Diagnostics.** REDQ logs how spread out its ensemble is
  (`ensemble_q_std`, `mean_pairwise_q_dist`), computed without gradient.
- **Shared maths.** REDQ imports SAC's maths (`ajax.agents.SAC.core`) instead
  of copying it.

## Results

The repository records no learning curves for REDQ yet. The probing tests
(`tests/probing`) check its targets, discounting, time limits, memory,
extension phases and checkpoint resume on tiny problems with exact answers.
The speed benchmark (`benchmarks/agent_bench.py`) runs REDQ with its defaults
for 8,000 steps, below `learning_starts`, so its figure times collection, not
updates.

## API

::: ajax.agents.REDQ.REDQ.REDQ
