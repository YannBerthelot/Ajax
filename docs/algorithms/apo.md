# APO

Average Policy Optimization is PPO for the average-reward criterion. Instead of
discounting future rewards, it learns how much reward the policy earns per step
on average and judges each action by how much better than that average it
does. This suits continuing tasks, such as holding a setpoint, where there is
no natural horizon and a discount factor only shortens the agent's view. The
clipped update and the minibatch epochs are PPO's own code.

**Paper:** Ma et al., *Average-Reward Reinforcement Learning with Trust Region
Methods*, 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)). Ajax
follows the authors' official code, `xtma/apo`, where it is more specific
than the paper.

## Can I use?

| Feature | APO |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, tanh-bounded, mapped onto the environment's bounds |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probe and resume contract) |

## Example

```python
from ajax import APO
from ajax.logging.wandb_logging import LoggingConfig

agent = APO("Pendulum-v1", n_envs=8, n_steps=256, batch_size=64, normalize_observations=True)
logging = LoggingConfig(config={}, log_frequency=10_240, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=51_200, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
print(state.average_reward)  # the estimated reward per step, one per seed
```

This budget shows the calls; it is too short to learn the swing-up.

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 4 | environments stepped in parallel |
| `n_steps` | 2048 | steps per environment in each rollout |
| `batch_size` | 64 | sets the number of minibatches, as in [PPO](ppo.md) |
| `num_minibatches` | 0 | when positive, the number of minibatches |
| `n_epochs` | 10 | passes over the rollout per update, each reshuffled |
| `actor_learning_rate`, `critic_learning_rate` | 3e-4 | Adam step sizes |
| `adam_eps` | 1e-8 | Adam's epsilon |
| `alpha` | 0.1 | update rate of the running averages of the reward rate and of the values |
| `nu` | 0.1 | weight of the value-bias penalty |
| `gae_lambda` | 0.95 | GAE's lambda (there is no discount) |
| `clip_range` | 0.2 | the ratio clip |
| `ent_coef` | 0 | entropy bonus weight |
| `max_grad_norm` | 0.5 | gradient clipping by global norm |
| `normalize_advantage` | True | advantages standardised over the whole rollout |
| `actor_architecture`, `critic_architecture` | `("128", "tanh", "128", "tanh")` | hidden layers |
| `normalize_observations`, `normalize_rewards` | False | running normalisation |
| `reward_normalization_gamma` | None | with `normalize_rewards`, divide by the spread of the discounted return |
| `expert_policy` | None | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `actor_bias_init`, `critic_bias_init`, `encoder_kernel_init` | None | initialisers by name ([Networks](../guide/networks.md)) |
| `log_std_bounds` | (-20.0, 2.0) | the `(low, high)` clip of the continuous actor's log-std; `None` leaves it unclipped (see the notes) |
| `episode_length`, `action_repeat` | None, 1 | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `expose_recent_rollout` | False | keep the last rollout on the state, for measurement extensions |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## How it works

Each update first moves two running averages towards the rollout's means, at
rate `alpha`: the reward rate `rho` (the mean reward) and `b` (the mean
predicted value). The advantages are GAE without a discount on the
differential reward `r - rho`. The critic fits the differential value; its
target is shifted by `nu * b`, which pulls the values back towards a zero mean.
Without that penalty the values are defined only up to a constant and can
drift. The actor then takes PPO's clipped step.

## Notes

- **Episode ends.** The advantage chain stops at every episode end, and a
  terminal state's next value is dropped, as in the official code. One
  difference: the reference drops time-limited steps from the loss because it
  has only the reset observation there. Ajax keeps the final observation, so a
  truncated step bootstraps on it and is trained on, as PPO does.
- **Advantage normalisation** runs once over the whole rollout, not per
  minibatch as in Ajax's PPO.
- **Shuffling.** Every epoch draws a new split of the rollout into
  minibatches, as APO's official code does.
- **Actor head.** APO always uses Brax's continuous head: a tanh-bounded
  Gaussian with one learnable log-std per action dimension (initial std 1)
  and a `lecun_uniform` mean layer. A discrete APO ignores it.
- **Log-std clip.** By default the log-std is clipped to `[-20, 2]`, the
  bounds of SAC's head, a deviation from the official code that the run
  config records (`log_std_bounds`); `log_std_bounds=None` removes the clip.
  A PID policy keeps its own log-std, always clipped to `[-20, 2]`.
- `n_steps` and `batch_size` must divide one another unless
  `num_minibatches` is set; the constructor refuses them otherwise.
- **Expert.** Cloning an expert into the actor before training is the
  `CloningPretrain` extension, and a cloning term during training the
  `ImitationLoss` extension; APO supports both
  ([Expert guidance](../guide/expert-guidance.md)). With `expert_policy` set,
  the evaluation also reports the expert's return.
- APO is the on-policy counterpart of [ASAC](asac.md), the average-reward
  off-policy agent.

## Results

No learning curves are published for APO yet. Unit tests
(`tests/agents/APO` and the contracts every agent shares) check its pieces
against exact contracts, without learning, such as its differential
advantages. The probes (`tests/probing`) train it briefly on tiny
environments with known answers: the reward rate it learns, episode ends,
extension phases, and a run split by a checkpoint matching the run left
whole. On the CPU speed benchmark (`benchmarks/agent_bench.py`: Pendulum, 4
environments, 8,000 steps) APO ran at 16,390 steps per second of execution
(4,024 with compilation) at main `a8a13c2`
([PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.APO.APO.APO
