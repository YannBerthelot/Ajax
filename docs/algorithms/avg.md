# AVG

Action Value Gradient is an incremental actor-critic: after each environment
step it makes one update on that single transition, with no replay buffer and
no target network. It stays stable through normalisation instead: the
observations are normalised, the networks' penultimate features are
L2-normalised, and the TD error is divided by a running estimate of its
scale. Use it when the agent must learn from each step as it comes, without
storing past experience.

**Paper.** Vasan et al., *Deep Policy Gradient Methods Without Batch Updates,
Target Networks, or Replay Buffers*, 2024
([arXiv:2411.15370](https://arxiv.org/abs/2411.15370)). The update follows its
Algorithm 1 and the authors' code (`gauthamvasan/avg`).

## Can I use?

| | AVG |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Incremental: one update per step, no replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | No |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

AVG has no memory option: with one-transition updates, backpropagation
through time has length 1, so a memory block could not learn anything
temporal without eligibility traces or real-time recurrent learning.

## Example

```python
from ajax import AVG
from ajax.logging.wandb_logging import LoggingConfig

agent = AVG("Pendulum-v1")  # one env, learning from its first step
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

Defaults are read from `AVG.__init__`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate` | `6.3e-3` | Adam step size of the actor |
| `critic_learning_rate` | `8.7e-3` | Adam step size of the critic |
| `beta_1`, `beta_2` | `0`, `0.999` | Adam's moment decays (AVG uses `beta_1 = 0`; Adam's epsilon is torch's 1e-8) |
| `actor_architecture`, `critic_architecture` | `("256", "leaky_relu", "256", "leaky_relu")` | hidden layers |
| `gamma` | `0.99` | discount |
| `learning_starts` | `0` | steps of uniform random actions before the first update |
| `alpha_init` | `0.07` | the entropy weight, fixed for the whole run |
| `num_critics` | `1` | critics; with more, the actor and the target use their minimum |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `expose_recent_rollout` | `False` | keep the last transition on `state.last_rollout` for extensions |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

AVG keeps `alpha` fixed at `alpha_init`, as the authors' code does, so it has
no `alpha_learning_rate` or `target_entropy_per_dim`. There is no
`normalize_observations` argument, since AVG always normalises its
observations, and no reward normalisation.

## Notes

- **TD-error scale.** The critic minimises the squared TD error divided by
  `sqrt(var(r) + E[G^2] var(gamma))`, where `r` is the entropy-regularised
  reward, `gamma` the per-step discount (zero at a termination) and `G` the
  returns of finished episodes. The scale is 1 until two returns have been
  seen. The statistics are updated after every step, warm-up included, and
  shared by all envs.
- **Actor gradient.** The actor loss is `alpha log pi(a|s) - Q(s, a)` at the
  action actually executed, reparameterised from its stored pre-squash
  sample, so the gradient flows through that action.
- **Update order, a deliberate choice.** Algorithm 1 writes the critic
  update before the actor's; the authors' code builds both losses from the
  same parameters, then steps the actor and the critic. Ajax follows the
  code, which produced the paper's results: the actor's loss sees the critic
  before its step, and the critic's target the actor before its step.
- **Time limits.** Only a termination stops the bootstrap, as in the
  authors' code; a transition cut by a time limit bootstraps on its final
  observation.
- **Several envs.** The paper trains on one environment. With `n_envs > 1`,
  each update uses the transitions of all envs from the same step.
- **Initialisation.** As in the authors' code, every layer starts
  orthogonal with gain 1 and a zero bias, the mean and log-std heads
  included. So the first policy's mean and log-std are unit-scale
  projections of the normalised features, not a near-zero mean and a
  constant std.
- **Actions and evaluation.** A `tanh`-squashed Gaussian in `[-1, 1]`,
  mapped onto the environment's bounds; evaluation uses the mean action.

## Results

The repository records no learning curves for AVG yet. Unit tests
(`tests/agents/AVG` and the contracts every agent shares) check its pieces
against exact contracts, without learning. The probes (`tests/probing`)
train it briefly on tiny environments with known answers: its target,
discounting, time limits, observation normalisation, and a run split by a
checkpoint matching the run left whole. On the CPU speed benchmark
(`benchmarks/agent_bench.py`: Pendulum, 4 envs, 8,000 steps) AVG ran at
9,460 steps per second of execution (4,027 with compilation), updates
included, at main `a8a13c2` (see
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.AVG.AVG.AVG
