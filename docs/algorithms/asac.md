# ASAC

Average-Reward Soft Actor-Critic is SAC for the average-reward criterion:
it maximises the long-run reward per step instead of a discounted sum, so
it has no `gamma`. Its critics learn a differential soft value on
`r - theta`, where `theta` tracks the entropy-regularised reward rate.

**Paper:** Adamczyk et al., *Average-Reward Soft Actor-Critic*, 2025
([arXiv:2501.09080v2](https://arxiv.org/abs/2501.09080v2)). Ajax follows
the authors' code, `BobMak/asac` at `4b68862`.

## When to use it

ASAC suits continuing tasks, such as holding a setpoint, where there is no
natural horizon and a discount factor only shortens the agent's view.
Episodic tasks fit too: each termination is charged a learned penalty.
[RVI-SAC](rvisac.md) is the other off-policy average-reward agent, and
[APO](apo.md) the on-policy one.

| Feature | ASAC |
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
from ajax import ASAC
from ajax.logging import LoggingConfig

agent = ASAC("Pendulum-v1", n_envs=4, learning_starts=1_000)  # no gamma
logging = LoggingConfig(log_frequency=2_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)

print(evaluations["Eval/episodic mean reward"])  # one row per seed
print(state.theta)  # the reward-rate estimate, one per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

The learning rates, the gradient clip and `tau_theta` are the authors'.
The [API](#api) section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `1e-4`, `5e-4`, `1e-4` | Adam step sizes (epsilon `1e-8`, torch's) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers |
| `tau` | `0.005` | Polyak rate of the target critics |
| `tau_theta` | `0.005` | rate of the `theta` and penalty averages; the authors tune it per task ([Presets](#presets)) |
| `p_0` | `20` | scale of the termination penalty |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `None` | rows per update: 256, or 64 sequences with `memory=` |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `alpha_init` | `0.2` | entropy weight (its start when it is learned) |
| `fixed_alpha` | `True` | keep `alpha` at `alpha_init`; `False` learns it towards the target entropy |
| `target_entropy_per_dim` | `-1.0` | target entropy per action dimension, with `fixed_alpha=False` |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `10.0` | gradient clipping, each critic and the actor on its own; `None` turns it off |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)); without a `gamma`, the rewards' return is discounted at 0.99 |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `0`, `64`, `False` | recurrent memory and its sequence replay ([Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground ([Environments](../guide/environments.md)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's (`Eval/expert bias`) |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

## Presets

`ASAC.preset(task, env_id)` takes the authors' tuned values for a
Gymnasium MuJoCo task (`hparams/<task>-v5/asac.yaml`;
[Presets](../guide/training.md#presets)). Their temperature is 1 / `beta`.

| Preset | `alpha_init` | `tau_theta` | `learning_starts` | `buffer_size` |
| --- | --- | --- | --- | --- |
| `Ant`, `HalfCheetah`, `Pusher` | 0.2 | 0.005 | 10,000 | 1,000,000 |
| `Hopper` | 0.2 | 0.001 | 10,000 | 1,000,000 |
| `Humanoid` | 0.05 | 0.001 | 10,000 | 1,000,000 |
| `InvertedDoublePendulum` | 0.2 | 0.001 | 0 | 1,000,000 |
| `InvertedPendulum` | 0.2 | 0.005 | 0 | 1,000,000 |
| `Reacher` | 0.2 | 0.005 | 10,000 | 100,000 |
| `Swimmer` | 0.05 | 0.005 | 10,000 | 1,000,000 |
| `Walker2d` | 0.2 | 0.0005 | 10,000 | 1,000,000 |

Their Swimmer-v3 configuration (`tau_theta` 0.1, the critics averaged
rather than minimised) has no preset: Ajax runs v5.

## Differences from the reference

None in the update: one update of Ajax's ASAC and of `BobMak/asac` from the
same state agree to within float32 rounding ([Results](#results)). Ajax
follows the authors' code even where it reads like a slip: the
termination penalty reads two rows of the batch (below), and the entropy
term subtracts `log(1/nA)`, a uniform discrete policy's log-probability,
for a continuous policy.

## Notes

- **Target.** `r - theta + (1 - done) (min(Q'_1, Q'_2)(s', a') - alpha (log pi(a'|s') - log(1/nA))) - Q'(0, 0)`,
  with `a' ~ pi(.|s')`, `done` a termination and `nA` the action
  dimension. A differential value is defined only up to a constant: the
  shift by the target critics' value at the origin, `Q'(0, 0)`, is the
  authors' reference point, and applies to every row, terminal ones
  included.
- **Episode ends.** A termination drops the next value and is charged a
  penalty, subtracted from the terminal transition's reward: an average,
  at rate `tau_theta`, of `p_0` times the larger of the rewards of the
  batch's first row (read for the rows that do not terminate) and its
  second row (for those that do), as the authors' code computes it; its
  comment describes the mean reward of the steps that did not terminate.
  A time limit bootstraps from the final observation.
- **Reward rate.** After each update `theta` moves, at rate `tau_theta`,
  towards the batch mean of the reward, without the penalty, minus
  `alpha (log pi(a|s) - log(1/nA))` at the actions the actor step sampled.
- **One update.** The penalty, then a critic step, an actor step on
  `alpha log pi - min_i Q_i`, a temperature step on the actor step's
  samples (skipped with `fixed_alpha`), the `theta` step and the target
  update.
- **Temperature.** Fixed, as in the authors' published runs;
  `fixed_alpha=False` learns it as SAC does.
- **Two critics,** always: ASAC has no `num_critics`. It tells its
  extensions there is no discount: they receive `gamma=None`.
- **Initialisation.** As in the authors' code (Stable-Baselines3's SAC
  actor and plain torch critics): every layer, the mean and log-std heads
  included, starts as torch's `nn.Linear` does, kernel and bias uniform in
  `±1/sqrt(fan_in)`. With `memory=`, ASAC builds Ni et al. (2022)'s
  recurrent networks and their initialisation, by analogy with their SAC.
- **Actions and evaluation.** As SAC; evaluation uses the mean action. The
  log-probability subtracts `log(1 - tanh(u)^2 + 1e-6)`, as the authors'
  Stable-Baselines3 actor does. The logged evaluation return is the usual
  episodic sum; ASAC reports no `Eval/mean bias`.

## Results

The repository records no learning curves for ASAC yet. A conformance
check (`tests/conformance/test_asac_original.py`) makes one update of
Ajax's ASAC and of `BobMak/asac` from the same state and batch, on two
Gymnasium tasks and on a Pendulum with a two-dimensional torque that
terminates every 20th step, and compares the actor, both critics and
their targets, the three optimisers, `theta` and the penalty. The probes
(`tests/probing`) train it briefly on tiny environments with known
answers: its differential target, its average-reward bookkeeping and
memory. [Performance](../guide/performance.md) compares its speed with the
other agents'.

## API

::: ajax.agents.ASAC.ASAC.ASAC
