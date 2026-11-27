# RVI-SAC

RVI-SAC is SAC for the average-reward criterion, learned by relative value
iteration (RVI) Q-learning: it maximises the long-run reward per step, so
it has no `gamma`. Its critics learn a differential soft value on
`r - xi`, where the offset `xi` follows the batch mean of the target's
next values. A termination does not end its target: the agent goes on
from the reset state and pays a learned reset cost.

**Paper:** Hisaki and Ono, *RVI-SAC: Average Reward Off-Policy Deep
Reinforcement Learning*, ICML 2024
([arXiv:2408.01972](https://arxiv.org/abs/2408.01972)). Ajax follows the
authors' code, `yhisaki/average-reward-drl` at `b5b0d31`.

## When to use it

RVI-SAC suits continuing tasks, or episodic ones read as continuing, where
a discount factor only shortens the agent's view. Where [ASAC](asac.md)
charges a termination a penalty learned from the rewards, RVI-SAC holds
the termination rate under a target. [APO](apo.md) is the on-policy
average-reward agent.

| Feature | RVI-SAC |
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

Every default is the authors' (their Table 1 and `rvi_sac.yaml`). The
[API](#api) section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (epsilon `1e-8`, torch's) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers |
| `tau` | `0.005` | Polyak rate of the target critics and of the reset critic's target |
| `centering` | `RVIBatchMean(rate=0.005)` | how the offset `xi` follows the critics ([Notes](#notes)) |
| `reset_cost` | `ResetCostSettings()` | the reset cost: target rate `1e-3`, a `("64", "relu", "64", "relu")` reset critic whose rate is followed at `0.005`, Adam at `3e-4`; `None` charges no cost |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `None` | rows per update: 256, or 64 sequences with `memory=` |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `alpha_init` | `1.0` | initial entropy weight |
| `fixed_alpha` | `False` | keep `alpha` at `alpha_init` |
| `target_entropy_per_dim` | `-1.0` | target entropy per action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping; the authors use none |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `0`, `64`, `False` | recurrent memory and its sequence replay ([Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground ([Environments](../guide/environments.md)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's (`Eval/expert bias`) |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

## Differences from the reference

One update of each from the same state agrees to within float32 rounding
([Results](#results)). Two details differ from the authors' code or paper:

- The authors' actor draws its mean and log-std from one output layer,
  one orthogonal matrix; here each head draws its own.
- The paper's analysis clips `xi`; the authors' code does not, and neither
  does Ajax.

## Notes

- **Target.** `r - c 1(terminated) - xi + min(Q'_1, Q'_2)(s', a') - alpha log pi(a'|s')`,
  with `a' ~ pi(.|s')`. Nothing cuts at a termination: the replay's next
  observation after one is the reset observation, whose value the target
  reads, as in the authors' reset scheme. A time limit bootstraps from the
  final observation. The critic loss sums the two critics' squared errors.
- **Centering.** After the critics' step, `xi` moves to
  `(1 - rate) xi + rate * mean(V'(s'))`, the batch mean of the values the
  target read (`RVIBatchMean`, the paper's Eq. 18).
  `centering=ReferencePoint(...)` reads the target critics at one
  state-action pair instead (the authors' `rvi_sac_with_reference`); both
  are in `ajax.agents.centering`.
- **Reset cost.** A 64-64 reset critic learns the rate at which the policy
  terminates, by the same RVI rule on the termination indicator. The cost
  `c`, a Lagrange multiplier starting at 0, takes an Adam step up while
  that rate exceeds the target rate and down while it is below, and never
  goes below 0.
- **One update.** The critics' and the reset critic's steps, the offsets,
  an actor step on `alpha log pi - min_i Q_i`, a temperature step on the
  actor step's sample (`alpha (-log pi - H_target)`, its gradient through
  `alpha`), the reset cost's step, then the target updates, in the
  authors' order.
- **Two critics,** always: RVI-SAC has no `num_critics`. It tells its
  extensions there is no discount (`gamma=None`, `dones` all zeros). It
  centres its values and prices terminations itself, so `ValueCentering`
  and `ResetCost` refuse it.
- **Initialisation.** As in the authors' code, every layer, the reset
  critic's too, starts orthogonal with gain `sqrt(1/3)` and a zero bias.
  With `memory=`, RVI-SAC builds Ni et al. (2022)'s recurrent networks and
  their initialisation, as the other recurrent off-policy agents do.
- **Evaluation**, as the authors', takes the mean of `tanh` at the latent
  mean plus and minus its standard deviation, not `tanh` of the mean.

## Results

The repository records no learning curves for RVI-SAC yet. A conformance
check (`tests/conformance/test_rvisac_original.py`) makes one update of
Ajax's RVI-SAC and of the authors' code from the same state and batch, at
four recorded updates (two from initialisation, two after 400 warm-up
updates), and compares every network and target, the five optimisers, the
temperature, the reset cost, `xi`, the reset rate and the evaluation
action. A probe (`tests/probing/test_average_reward.py`) trains it on a
reset cycle whose differential values, reward rate and termination rate
have closed forms. [Performance](../guide/performance.md) compares its
speed with the other agents'.

## API

::: ajax.agents.RVISAC.RVISAC.RVISAC
