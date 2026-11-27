# AVG

Action Value Gradient is an incremental actor-critic: after each
environment step it makes one update on that single transition, with no
replay buffer and no target network. It stays stable through normalisation
instead: the observations are normalised, the networks' penultimate
features are L2-normalised, and the TD error is divided by a running
estimate of its scale.

**Paper:** Vasan et al., *Deep Policy Gradient Methods Without Batch
Updates, Target Networks, or Replay Buffers*, 2024
([arXiv:2411.15370](https://arxiv.org/abs/2411.15370)). Ajax follows its
Algorithm 1 as the authors' code makes it, `gauthamvasan/avg` at
`6b6822c`.

## When to use it

AVG suits an agent that must learn from each step as it comes, without
storing past experience: on-device or real-time learning, or a memory
budget too small for a replay buffer. Where a replay buffer fits,
[SAC](sac.md) reuses each transition many times.

| Feature | AVG |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a `tanh`-squashed Gaussian, mapped onto the environment's bounds |
| Memory (`memory=`) | no: with one-transition updates, a memory block could learn nothing temporal |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | no: it keeps no replay |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_warmup_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import AVG
from ajax.logging import LoggingConfig

agent = AVG("Pendulum-v1")  # one env, learning from its first step
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

The defaults are the authors'. The [API](#api) section lists every
argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate` | `6.3e-3` | Adam step size of the actor |
| `critic_learning_rate` | `8.7e-3` | Adam step size of the critic |
| `beta_1`, `beta_2` | `0`, `0.999` | Adam's moment decays (epsilon `1e-8`, torch's) |
| `actor_architecture`, `critic_architecture` | `("256", "leaky_relu", "256", "leaky_relu")` | hidden layers |
| `gamma` | `0.99` | discount |
| `learning_starts` | `0` | steps of uniform random actions; AVG learns from each step after them, never from one of these |
| `alpha_init` | `0.07` | the entropy weight, fixed for the whole run |
| `num_critics` | `1` | critics; with more, the actor and the target use their minimum |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's (`Eval/expert bias`) |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

AVG keeps `alpha` fixed, as the authors' code does, so it has no
`alpha_learning_rate` or `target_entropy_per_dim`. It always normalises
its observations and never its rewards, so it has no
`normalize_observations` or `normalize_rewards` argument.

## Differences from the reference

- **Actions.** The authors' code hands its `tanh` action to the
  environment as it is; Ajax maps it onto the environment's bounds. On a
  `[-1, 1]` box the two are the same.
- **Several environments.** The paper trains on one environment. With
  `n_envs > 1`, each update uses the transitions of all the environments
  from the same step.
- **Update order.** Algorithm 1 writes the critic update before the
  actor's; the authors' code builds both losses from the same parameters,
  then steps the actor and the critic. Ajax follows the code, which
  produced the paper's results: the actor's loss sees the critic before
  its step, and the critic's target the actor before its step.

## Notes

- **TD-error scale.** The critic minimises the squared TD error divided by
  `sqrt(max(var(r) + E[G^2] var(gamma), 1e-4))`, where `r` is the
  entropy-regularised reward, `gamma` the per-step discount (zero at a
  termination) and `G` the returns of the episodes that terminated, as
  the authors' `TDErrorScaler` does. The scale is 1 while that variance is
  at most 0.01 and no episode has terminated; on a task without
  terminations it follows the reward's variance alone. The statistics are
  updated after every step, warm-up included, and shared by all the
  environments.
- **Actor gradient.** The actor loss is `alpha log pi(a|s) - Q(s, a)` at
  the action actually executed, reparameterised from its stored
  pre-squash sample, so the gradient flows through that action.
- **Time limits.** Only a termination stops the bootstrap, as in the
  authors' code; a transition cut by a time limit bootstraps on its final
  observation.
- **Initialisation.** As in the authors' code, every layer starts
  orthogonal with gain 1 and a zero bias, the mean and log-std heads
  included. So the first policy's mean and log-std are unit-scale
  projections of the normalised features, not a near-zero mean and a
  constant standard deviation.
- **Log-std head.** As in the authors' code, whose `MultivariateNormal`
  takes it as the covariance's diagonal, the head's output is a
  log-variance: the standard deviation is `exp(out / 2)`, and its clip to
  `[-20, 2]` bounds the log-variance.
- **Evaluation** uses the mean action.

## Results

The repository records no learning curves for AVG yet. A conformance
check (`tests/conformance/test_avg_original.py`) makes one update of
Ajax's AVG and of `gauthamvasan/avg` from the same state and transition,
and compares the actor, the critic, their optimisers and the TD-error
scaler: every leaf conforms, to within float32 rounding. It asserts the
one difference from that code, the action handed to the environment. The
probes
(`tests/probing`) train it briefly on tiny environments with known
answers: its target, discounting, time limits, observation normalisation,
and a run split by a checkpoint matching the run left whole.
[Performance](../guide/performance.md) compares its speed with the other
agents'.

## API

::: ajax.agents.AVG.AVG.AVG
