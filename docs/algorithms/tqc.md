# TQC

TQC is SAC with distributional critics: each critic predicts quantiles of
the return. The target pools every target critic's quantiles at the next
state and action, sorts them, and drops the largest, which controls the
over-estimation a maximising actor feeds on. Every critic regresses its
quantiles on the remaining atoms with the quantile Huber loss, and the
actor maximises the mean of all the critics' quantiles.

**Paper:** Kuznetsov, Shvechikov, Grishin and Vetrov, *Controlling
Overestimation Bias with Truncated Mixture of Continuous Distributional
Quantile Critics*, ICML 2020
([arXiv:2005.04269](https://arxiv.org/abs/2005.04269)). Ajax follows the
authors' code, `bayesgroup/tqc_pytorch` as Software Heritage archives it
(`swh:1:rev:65114eba5d868e9f0355e33b96acab7530191f16`).

## When to use it

TQC suits the tasks [SAC](sac.md) suits, with one knob more for
over-estimation: the number of quantiles dropped, which its authors tune
per task (the presets). Its five critics of three 512-unit layers make
each step costly on a CPU.

| Feature | TQC |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a `tanh`-squashed Gaussian, mapped onto the environment's bounds |
| Memory (`memory=`) | yes, on replayed sequences, without `expert_policy` or most extensions ([Memory](../guide/memory.md)) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import TQC
from ajax.logging import LoggingConfig

# Narrow critics for a CPU; the authors' are 3 x 512.
agent = TQC("Pendulum-v1", critic_architecture=("128", "relu", "128", "relu"))
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

## Key hyperparameters

The defaults are the authors' code's (`main.py` and `tqc/`). The
[API](#api) section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (epsilon `1e-8`, torch's) |
| `actor_architecture` | `("256", "relu", "256", "relu")` | the actor's hidden layers |
| `critic_architecture` | `("512", "relu", "512", "relu", "512", "relu")` | each critic's hidden layers |
| `num_critics` | `5` | distributional critics |
| `n_quantiles` | `25` | quantiles each critic predicts |
| `top_quantiles_to_drop_per_net` | `2` | pooled target quantiles dropped per critic; the paper tunes it per task ([Presets](#presets)) |
| `gamma` | `0.99` | discount |
| `tau` | `0.005` | Polyak rate of the target critics |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `None` | rows per update: 256, or 64 sequences with `memory=` |
| `learning_starts` | `257` | the step of the first update; the policy acts from the first step, with no uniform warm-up |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy per action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `0`, `64`, `False` | recurrent memory and its sequence replay ([Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

## Presets

`TQC.preset(task, env_id)` drops the authors' number of quantiles for the
task (`README.md:42-48`, the paper's Table 6;
[Presets](../guide/training.md#presets)).

| Preset | `top_quantiles_to_drop_per_net` |
| --- | --- |
| `Hopper` | 5 |
| `HalfCheetah` | 0 |
| `Walker2d`, `Ant`, `Humanoid` | 2 |

## Differences from the reference

- **The actor's step.** As in the authors' code, the critics step and the
  targets update first, then the actor and the temperature, the actor's
  loss built on the critics from before their step. The authors' torch
  (1.3.0) backpropagated that loss through the critics' stepped weights;
  Ajax differentiates it at the critics from before their step. Nothing
  else differs ([Results](#results)).

## Notes

- **Quantile critics.** Each critic's head has `n_quantiles` outputs. A
  value query (an evaluation, a probe, an extension) reads their mean,
  Q(s, a); the target and the critics' loss read the quantiles.
- **Extensions.** Those that shift the target by a value per row (`IBRL`,
  `LCBGatedBootstrap`, `TargetBC`) shift every quantile alike. Those that
  read one target value (`CriticBlend`, `MCVarianceCorrection`,
  `BiasVorePenalty`) refuse TQC.
- **Initialisation.** As in the authors' code (torch's `nn.Linear`
  defaults): kernels and biases uniform in `±1/sqrt(fan_in)` on every
  layer. With `memory=`, TQC builds Ni et al. (2022)'s recurrent networks
  and their initialisation, as the other recurrent off-policy agents do.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.

## Results

The repository records no learning curves for TQC yet. A conformance
check (`tests/conformance/test_tqc_original.py`) runs one update of Ajax's
TQC and of the authors' code, unchanged on torch 1.4, from the same state,
with the same batch and noises: the critics, their targets, the
temperature and their optimisers conform, to within float32 rounding, and
the actor's step differs, as above. The probes (`tests/probing`) train it
briefly on tiny environments with known answers.
[Performance](../guide/performance.md) compares its speed with the other
agents'.

## API

::: ajax.agents.TQC.TQC.TQC
