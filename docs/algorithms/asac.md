# ASAC

Average-Reward Soft Actor-Critic is SAC for the average-reward criterion: it
maximises the long-run reward per step instead of a discounted sum, so it has
no `gamma`. Its critics learn a differential soft value on `r - theta`, where
`theta` tracks the entropy-regularised reward rate. Use it for continuing
tasks, where no discount is natural; episodic tasks fit by charging each
termination a learned penalty.

**Paper.** Adamczyk et al., *Average-Reward Soft Actor-Critic*, 2025
([arXiv:2501.09080v2](https://arxiv.org/abs/2501.09080v2)).

## Can I use?

| | ASAC |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions or `expert_policy` |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

## Example

```python
from ajax import ASAC
from ajax.logging.wandb_logging import LoggingConfig

agent = ASAC("Pendulum-v1", n_envs=4, learning_starts=1_000)  # no gamma
logging = LoggingConfig(config={}, log_frequency=2_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)

print(evaluations["Eval/episodic mean reward"])  # one row per seed
print(state.theta)  # the reward-rate estimate, one per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

Defaults are read from `ASAC.__init__`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target critics, and the rate of the `theta` and penalty averages |
| `p_0` | `20` | scale of the termination penalty |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, divide by the spread of the discounted return |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

With an `expert_policy`, ASAC stores the expert's actions along each
trajectory, from its state when it has one, and takes the `CloningPretrain`
and `ImitationLoss` extensions ([Expert guidance](../guide/expert-guidance.md)).

## Notes

- **Target.** `r - theta + min(Q'_1, Q'_2)(s', a') - Q'(0, 0) - alpha log pi(a'|s')`
  with `a' ~ pi(.|s')`. The shift by the target critics' value at the origin,
  `Q'(0, 0)`, is their reference point: a differential value is defined only
  up to a constant.
- **Episode ends.** The target bootstraps through every end. A termination
  is charged a penalty instead, subtracted from the terminal transition's
  reward: an average (at rate `tau`) of `p_0` times the batch's mean reward,
  terminal rewards counted as zero.
- **Reward rate.** After each update `theta` moves, at rate `tau`, towards
  the batch mean of the penalised reward minus `alpha log pi(a|s)`, with
  fresh actions `a ~ pi(.|s)`.
- **One update.** The penalty, then a critic step, an actor step on
  `alpha log pi - min_i Q_i`, a temperature step on fresh samples of the
  updated policy, the `theta` step and the target update. Each iteration
  steps every env once, then makes one update.
- **Two critics.** ASAC always uses two critics; it has no `num_critics`.
- **Extensions.** ASAC tells its extensions there is no discount: they
  receive `gamma=None`.
- **Actions and evaluation.** As SAC: a `tanh`-squashed Gaussian in
  `[-1, 1]`, mapped onto the environment's bounds; evaluation uses the mean
  action. The logged evaluation return is the usual episodic sum. ASAC
  reports no `Eval/mean bias`.

## Results

The repository records no learning curves for ASAC yet. Unit tests
(`tests/agents/ASAC` and the contracts every agent shares) check its pieces
against exact contracts, without learning. The probes (`tests/probing`)
train it briefly on tiny environments with known answers: its differential
target, its average-reward bookkeeping (reward scale, episode ends), memory,
and a run split by a checkpoint matching the run left whole. On the CPU
speed benchmark (`benchmarks/agent_bench.py`: Pendulum, 4 envs, 8,000 steps,
updates from step 1,000) ASAC ran at 1,254 steps per second of execution
(1,032 with compilation) at main `a8a13c2` (see
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.ASAC.ASAC.ASAC
