# RVI-SAC

RVI-SAC is SAC for the average-reward criterion, learned by relative value
iteration (RVI) Q-learning: it maximises the long-run reward per step
instead of a discounted sum, so it has no `gamma`. Its critics learn a
differential soft value on `r - xi`, where the offset `xi` follows the
batch mean of the target's next values. A termination does not end its
target: the agent goes on from the reset state and pays a reset cost, which
it learns to keep episodes from ending more often than a target rate. Use it
for continuing tasks, or for episodic ones read as continuing.

**Paper.** Hisaki and Ono, *RVI-SAC: Average Reward Off-Policy Deep
Reinforcement Learning*, ICML 2024
([arXiv:2408.01972](https://arxiv.org/abs/2408.01972)); the authors' code is
[yhisaki/average-reward-drl](https://github.com/yhisaki/average-reward-drl).

## Can I use?

| | RVI-SAC |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions or `expert_policy` |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `transform_action`, `actuate`, `choose_warmup_action`, `choose_replay_action` |

## Example

```python
from ajax import RVISAC
from ajax.logging import LoggingConfig

agent = RVISAC("Pendulum-v1", n_envs=4, learning_starts=1_000)  # no gamma
logging = LoggingConfig(log_frequency=2_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)

print(evaluations["Eval/episodic mean reward"])  # one row per seed
print(state.offset)  # xi, the reward-rate estimate, one per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

Defaults are read from `RVISAC.__init__`. They are the authors' (their
Table 1 and `rvi_sac.yaml`).

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (Adam's epsilon is torch's 1e-8) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target critics and of the reset critic's target |
| `centering` | `RVIBatchMean(rate=0.005)` | how the offset `xi` follows the critics ([Notes](#notes)) |
| `reset_cost` | `ResetCostSettings()` | the reset cost: target rate `1e-3`, a `("64", "relu", "64", "relu")` reset critic, its rate followed at `0.005`, Adam at `3e-4`; `None` charges no cost |
| `alpha_init` | `1.0` | entropy weight at the start |
| `fixed_alpha` | `False` | keep `alpha` at `alpha_init` |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping; the authors use none |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **Target.** `r - c 1(terminated) - xi + min(Q'_1, Q'_2)(s', a') - alpha log pi(a'|s')`,
  `a' ~ pi(.|s')`. There is no cut at a termination: the replay's next
  observation after a termination is the reset one, and the target reads
  its value, as the authors' reset scheme does. A time limit bootstraps
  from the final observation. Each critic's squared error is summed over the
  two critics (no factor 1/2), as in the authors' code.
- **Centering.** After the critics' step, `xi` moves to `(1 - rate) xi +
  rate * mean(V'(s'))`, the batch mean of the values the target read
  (`ajax.agents.centering.RVIBatchMean`, the paper's Eq. 18). The paper's
  analysis clips `xi`; the authors' code does not, and neither does Ajax.
  `centering=ReferencePoint(...)` reads the target critics at one
  state-action pair instead (the authors' `rvi_sac_with_reference`).
- **Reset cost.** A reset critic, a 64-64 network, learns the rate at
  which the policy terminates by the same RVI rule on the termination
  indicator (`ajax.agents.reset_cost`); the cost `c`, a Lagrange multiplier
  starting at 0, takes an Adam step up while that rate exceeds the target
  rate and down while it is below, and never goes below 0.
- **One update.** The critics' and the reset critic's steps, the offsets,
  an actor step on `alpha log pi - min_i Q_i`, a temperature step on the
  actor step's sample (`alpha (-log pi - H_target)`, its gradient through
  `alpha`), the reset cost's step, then the target updates, in the
  authors' order. Each iteration steps every env once, then makes one
  update.
- **Two critics.** RVI-SAC always uses two critics; it has no `num_critics`.
- **Extensions.** RVI-SAC tells its extensions there is no discount: they
  receive `gamma=None`, and `dones` is all zeros (the target cuts
  nowhere). It centres its values and prices terminations itself, so
  `ValueCentering` and `ResetCost` refuse it.
- **Initialisation.** As the authors' code: every layer, the reset
  critic's too, starts orthogonal with gain `sqrt(1/3)` and a zero bias.
  The authors' actor draws its mean and log-std from one output layer; here
  each head draws its own. With `memory=`, RVI-SAC keeps Ajax's
  initialisation, as recurrent SAC and ASAC do.
- **Actions and evaluation.** A `tanh`-squashed Gaussian in `[-1, 1]`
  (log-std clipped to `[-20, 2]`), mapped onto the environment's bounds.
  Evaluation, as the authors', takes the mean of `tanh` at the latent mean
  plus and minus its standard deviation, not `tanh` of the mean.

## Results

The repository records no learning curves for RVI-SAC yet. A conformance
check (`tests/conformance/test_rvisac_original.py`) makes one update of
Ajax's RVI-SAC and of the authors' code from the same state and batch, at
four recorded updates (two from initialisation, two after 400 warm-up
updates, a quarter of each batch terminating), and compares every
network and target, the five optimisers, the temperature, the reset cost,
`xi` and the reset rate, and the evaluation action. The probes
(`tests/probing/test_average_reward.py`) train it on a reset cycle whose
differential values, reward rate and termination rate have closed forms.

## API

::: ajax.agents.RVISAC.RVISAC.RVISAC
