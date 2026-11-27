# SAC

Soft Actor-Critic learns a stochastic policy that maximises the reward plus
the policy's entropy, so it keeps exploring while it improves. Two critics
estimate the action value, the target takes the smaller of the two target
critics' values to curb over-estimation, and the entropy weight `alpha`
tunes itself towards a target entropy.

**Papers:** Haarnoja et al., *Soft Actor-Critic*, 2018
([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)), with the learned
temperature of Haarnoja et al., *Soft Actor-Critic Algorithms and
Applications*, 2018 ([arXiv:1812.05905](https://arxiv.org/abs/1812.05905)).
Ajax follows the authors' code for the latter, `rail-berkeley/softlearning`
at `a1253bc`.

## When to use it

SAC is the default agent for continuous actions when environment steps are
costly: it learns off-policy from a replay buffer and reuses each
transition many times. It also runs more expert-guidance modules than any
other agent. With many cheap parallel environments, [PPO](ppo.md) learns
from fresh rollouts instead; for a deterministic policy, see [TD3](td3.md).

| Feature | SAC |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a `tanh`-squashed Gaussian, mapped onto the environment's bounds |
| Memory (`memory=`) | yes, on replayed sequences, without `expert_policy` or most extensions ([Memory](../guide/memory.md)) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)); also [CQL and Cal-QL](#cql-and-cal-ql-offline) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_warmup_action`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import SAC
from ajax.logging import LoggingConfig

agent = SAC("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(log_frequency=2_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)

print(evaluations["timestep"][0])                # when each evaluation ran
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

`train` runs both seeds side by side in one compiled program. This budget
only shows the call: Pendulum needs more steps to be solved. See
[Training](../guide/training.md) for resuming, and
[Logging](../guide/logging.md) for Weights & Biases or TensorBoard.

## Key hyperparameters

The defaults are softlearning's; [Coming from SB3](../getting-started/coming-from-sb3.md)
lists where they differ from Stable-Baselines3's. The [API](#api) section
lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (epsilon `1e-8`, TensorFlow's) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers |
| `gamma` | `0.99` | discount |
| `tau` | `0.005` | Polyak rate of the target critics |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `None` | rows per update: 256, or 64 sequences with `memory=` |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `alpha_init` | `1.0` | initial entropy weight |
| `fixed_alpha` | `False` | keep `alpha` at `alpha_init` |
| `target_entropy_per_dim` | `-1.0` | target entropy per action dimension: a number, or a schedule of the timestep |
| `num_critics` | `2` | critics; the target takes their minimum |
| `num_critic_updates` | `1` | critic steps per update, on the same batch |
| `n_step` | `1` | rewards each target sums before it bootstraps |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping of the actor and the critics |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `0`, `64`, `False` | recurrent memory and its sequence replay ([Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground ([Environments](../guide/environments.md)) |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

### Expert options

| Argument | Default | What it does |
| --- | --- | --- |
| `expert_policy` | `None` | an expert, in the environment's units, that the run tracks and the expert extensions read; each evaluation logs its return |
| `eval_expert_policy` | `None` | an expert whose return each evaluation logs, which takes no other part in the run |

SAC's other expert options are extensions now
([Expert guidance](../guide/expert-guidance.md)): `ExpertPrefill` writes
the expert's steps to the replay, `ExpertWarmup` lets it take part of the
warm-up, `CloningPretrain` clones it into the actor, `ExpertGains` makes the
actor output its PID's gains, and `ExpertActionInput` and `ExpertStateInput`
append its action and state to the observation. `MCPretrain`, an expert
critic fitted on its Monte-Carlo returns, runs on SAC alone.

## Differences from the reference

- **Update order.** softlearning runs the critic, actor and temperature
  steps in one `session.run` (`sac.py:348-353`): every gradient reads the
  parameters from before the update, and the temperature reads the actor
  loss's own sample. Ajax makes the steps one after the other: the actor
  reads the stepped critics, and the temperature a fresh sample of the
  stepped actor, as softlearning's TensorFlow 2 port (`13cf187`,
  `sac.py:277-281`) and CleanRL do.
- **Output heads.** The reference draws the mean and the log-std from one
  output layer; Ajax's two heads each take Glorot's bound for their own
  width (about 1% wider with 256 inputs).
- **Ajax's own options,** off by default: `n_step` returns,
  `policy_update_start` and `alpha_update_start` (the timesteps from which
  the actor and `alpha` update), a schedule as `target_entropy_per_dim`,
  and the expert options above.

## Notes

- **Actions.** The warm-up's uniform actions cover the whole box, and the
  replay stores the agent's `[-1, 1]` action. Evaluation uses the mean
  action.
- **Initialisation.** Glorot-uniform kernels and zero biases in every
  layer, as in `haarnoja/sac` and softlearning. With `memory=`, SAC is Ni
  et al. (2022)'s recurrent SAC, networks, losses and initialisation
  included ([Memory](../guide/memory.md#the-networks)).
- **Standard deviation.** The log-std is clipped to `[-20, 2]` and the
  standard deviation is its exponential, as in `haarnoja/sac` and in
  softlearning at the time of the paper; later softlearning uses
  `softplus(x) + 1e-5`.
- **Updates per step.** Each iteration steps every environment once, then
  makes one update: `num_critic_updates` critic steps, one actor step, one
  temperature step and one target update. With `n_envs=4` that is one
  update per four environment steps.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **Temperature.** The `alpha` step is never gradient-clipped, even with
  `max_grad_norm` set, as in the reference: clipping would turn it into a
  sign step.
- **Memory.** With `memory=`, SAC replays 64 sequences of 64 steps, each
  from a zero memory, as Ni et al. (2022) do, with R2D2's `burn_in` and
  `stored_state` as options (Kapturowski et al. 2019). Of the extensions
  it then takes only observation features and the warm-up's or the
  replay's action choices, refusing the others when training starts.

## CQL and Cal-QL: offline

CQL (Kumar, Zhou, Tucker and Levine, *Conservative Q-Learning for Offline
Reinforcement Learning*, NeurIPS 2020,
[arXiv:2006.04779](https://arxiv.org/abs/2006.04779)) adds a critic term
that pushes the values down on actions away from the data's and up on the
data's. Cal-QL (Nakamoto et al., *Cal-QL: Calibrated Offline RL
Pre-Training for Efficient Online Fine-Tuning*, NeurIPS 2023,
[arXiv:2303.05479](https://arxiv.org/abs/2303.05479)) floors that push at
the data's Monte-Carlo returns, so that fine-tuning online does not start
from values far too low. Both are extensions of SAC, trained with
[`train_offline`](../guide/training.md#training-offline):

```python
from ajax import SAC
from ajax.extensions.offline import CalQL, ConservativeQ

agent = SAC(
    "Pendulum-v1",
    actor_learning_rate=1e-4,  # the references' policy rate
    learning_starts=0,  # online fine-tuning updates from the first step
    extensions=(ConservativeQ(alpha=5.0),),  # or CalQL(alpha=5.0)
)
# state, evaluations = agent.train_offline(dataset, n_updates=1_000_000)
```

The term is the authors' importance-sampled CQL(H)
(`aviralkumar2907/CQL` at `d67dbe9`, `cql.py:234-264`): for each critic,
the logsumexp of its values at 10 uniform actions and at 10 of the
policy's at `s` and at `s'`, each less its log-density, minus its value at
the data's action, times `alpha` (5 in the references' gym runs). Cal-QL
first takes the larger of each policy action's value and the row's return
to go (`nakamotoo/Cal-QL` at `ac6eafe`,
`JaxCQL/conservative_sac.py:227-240`), which the transition dataset holds
at its `gamma`; collected rows have none and are not floored, where the
reference computes theirs as episodes end. Both need a policy with a
density, so TD3 refuses them. One more difference from the references:
SAC's target keeps its entropy term, which the CQL reference's
`deterministic_backup` drops.

## Results

The [reproduction results](../reproduction/results.md) set SAC beside
Stable-Baselines3's and SBX's on Pendulum, with learning curves. The probes
(`tests/probing`) train it briefly on tiny environments with known answers:
the soft bootstrap, discounting, time limits, reward scale, memory, every
extension phase it folds, and a run split by a checkpoint matching the run
left whole. [Performance](../guide/performance.md) compares its speed with
the other agents'.

## API

::: ajax.agents.SAC.SAC.SAC
