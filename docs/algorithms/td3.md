# TD3

Twin Delayed DDPG learns a deterministic policy and two critics. It fixes
DDPG's over-estimation with three changes: the target takes the smaller of
two target critics, the target action gets clipped noise (target policy
smoothing), and the actor and the targets update only every `policy_delay`
critic steps. It explores by adding Gaussian noise to its actions and learns
off-policy from a replay buffer.

**Paper.** Fujimoto et al., *Addressing Function Approximation Error in
Actor-Critic Methods*, 2018 ([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)).
The defaults follow its Table 1 and Algorithm 1, and the critic follows the
authors' code (`sfujim/TD3`).

## Can I use?

| | TD3 |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions, `expert_policy` or cloning |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

TD3 is not exported from `ajax`: import it from `ajax.agents.TD3.TD3`.

## Example

```python
from ajax.agents.TD3.TD3 import TD3
from ajax.logging.wandb_logging import LoggingConfig

agent = TD3("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=2_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

Defaults are read from `TD3.__init__`; the class docstring ties them to the
paper.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate` | `1e-3` | Adam step sizes |
| `actor_architecture`, `critic_architecture` | `("400", "relu", "300", "relu")` | hidden layers (plain MLPs) |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `100` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target networks |
| `num_critics` | `2` | critics; the target takes their minimum |
| `policy_delay` | `2` | critic steps per actor and target update |
| `exploration_noise` | `0.1` | standard deviation of the behaviour noise |
| `target_policy_noise`, `target_noise_clip` | `0.2`, `0.5` | target smoothing noise and its clip |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `expert_policy`, `pre_train_n_steps` | `None`, `0` | behaviour-clone an expert before training |
| `actor_cloning_epochs`, `actor_cloning_lr`, `actor_cloning_batch_size`, `skip_actor_pretrain` | `10`, `1e-3`, `64`, `False` | that cloning's settings |
| `action_pipeline` | `None` | replaces the collection-time action choice (used by a downstream project) |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **Actor.** A deterministic MLP ending in `tanh`, its last layer
  initialised uniformly in `[-3e-3, 3e-3]` (the DDPG recipe). Actions lie in
  `[-1, 1]`, and Ajax clips them to the environment's bounds without
  rescaling them, so the noise levels are in those `[-1, 1]` units (the
  environment's own on Brax and Playground).
- **Behaviour policy.** `clip(mu(s) + N(0, exploration_noise), -1, 1)` after
  the warm-up; uniform random actions before `learning_starts`. Evaluation
  uses `mu(s)` without noise.
- **Target.** `r + gamma (1 - terminated) min_i Q'_i(s', a')`, with
  `a' = clip(mu'(s') + clip(N(0, target_policy_noise), -c, c), -1, 1)` and
  `c = target_noise_clip`.
- **Actor loss.** `-Q_1(s, mu(s))`: the first critic only.
- **Updates per step.** Each iteration steps every env once, then makes one
  critic step; every `policy_delay`-th one also steps the actor and the
  targets.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **PID policy.** With `actor_architecture=PIDPolicy(...)` the actor is a
  PID whose output goes through `tanh`; see [Networks](../guide/networks.md).
- **Memory.** Sequence replay also burns in the target actor's memory, since
  TD3's bootstrap action comes from it.

## Results

The repository records no learning curves for TD3 yet. The probing tests
(`tests/probing`) check its target smoothing, discounting, time limits,
memory, extension phases and checkpoint resume on tiny problems with exact
answers. The speed benchmark (`benchmarks/agent_bench.py`) runs TD3 with its
defaults for 8,000 steps, below `learning_starts`, so its figure times
collection, not updates.

## API

::: ajax.agents.TD3.TD3.TD3
