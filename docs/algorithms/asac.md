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
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `1e-4`, `5e-4`, `1e-4` | Adam step sizes (the authors'; Adam's epsilon is torch's 1e-8) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target critics |
| `tau_theta` | `0.005` | rate of the `theta` and penalty averages (the authors tune it per environment, 0.0005 to 0.1) |
| `p_0` | `20` | scale of the termination penalty |
| `alpha_init` | `0.2` | entropy weight (its start when it is learned) |
| `fixed_alpha` | `True` | keep `alpha` at `alpha_init`; `False` learns it towards the target entropy |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension (with `fixed_alpha=False`) |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `10.0` | gradient clipping, each critic and the actor on its own (the authors'); `None` turns it off |
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

- **Target.** `r - theta + (1 - done) (min(Q'_1, Q'_2)(s', a') - alpha (log pi(a'|s') - log(1/nA))) - Q'(0, 0)`
  with `a' ~ pi(.|s')`, `done` a termination and `nA` the action
  dimension, as the authors' code measures the entropy term. The shift by the target
  critics' value at the origin, `Q'(0, 0)`, is their reference point: a
  differential value is defined only up to a constant. It applies to every
  row, terminal ones included, as in the authors' code.
- **Episode ends.** A termination drops the next value and is charged a
  penalty, subtracted from the terminal transition's reward: an average (at
  rate `tau_theta`) of `p_0` times the larger of the rewards of the batch's
  first row (read for the rows that do not terminate) and its second row
  (for those that do). The authors' code computes it this way (`gather` by
  the done flags);
  its comment describes the mean reward of the steps that did not
  terminate. A time limit bootstraps from the final observation.
- **Reward rate.** After each update `theta` moves, at rate `tau_theta`, towards
  the batch mean of the reward, without the penalty, minus `alpha (log pi(a|s)
  - log(1/nA))`, at the actions `a ~ pi(.|s)` the actor step sampled.
- **One update.** The penalty, then a critic step, an actor step on
  `alpha log pi - min_i Q_i`, a temperature step on the actor step's samples
  (skipped with `fixed_alpha`), the `theta` step and the target update. Each
  iteration steps every env once, then makes one update.
- **Temperature.** As in the authors' published runs, `alpha` is fixed:
  0.2 by default. Those runs use 0.05 for Humanoid and Swimmer; pass
  `alpha_init=0.05` for those. `fixed_alpha=False` learns it as SAC does.
- **Two critics.** ASAC always uses two critics; it has no `num_critics`.
- **Extensions.** ASAC tells its extensions there is no discount: they
  receive `gamma=None`.
- **Initialisation.** As the authors' code (stable-baselines3's SAC actor
  and plain torch critics): every layer, the mean and log-std heads
  included, starts as torch's `nn.Linear` does, its kernel and bias
  uniform in `±1/sqrt(fan_in)`. With `memory=`, ASAC keeps its previous
  initialisation until Ni et al.'s is audited.
- **Actions and evaluation.** As SAC: a `tanh`-squashed Gaussian in
  `[-1, 1]`, mapped onto the environment's bounds; evaluation uses the mean
  action. Its log-probability subtracts `log(1 - tanh(u)^2 + 1e-6)`, as the
  authors' stable-baselines3 actor does. The logged evaluation return is the usual episodic sum. ASAC
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
