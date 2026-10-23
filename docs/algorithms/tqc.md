# TQC

TQC is SAC with distributional critics. Each of its critics predicts
quantiles of the return. The target pools every target critic's quantiles
at the next state and action, sorts them, and drops the largest, which
controls the overestimation a maximising actor feeds on. Every critic
regresses its quantiles on the remaining atoms with the quantile Huber loss.
The actor maximises the mean of all the critics' quantiles.

**Paper.** Kuznetsov, Shvechikov, Grishin and Vetrov, *Controlling
Overestimation Bias with Truncated Mixture of Continuous Distributional
Quantile Critics*, ICML 2020
([arXiv:2005.04269](https://arxiv.org/abs/2005.04269)). Where the authors'
code (`bayesgroup/tqc_pytorch`, archived by Software Heritage as
`swh:1:rev:65114eba5d868e9f0355e33b96acab7530191f16`) is more specific, Ajax
follows the code.

## Can I use?

| | TQC |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes: trains on replayed sequences (R2D2-style burn-in) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

## Example

```python
from ajax import TQC
from ajax.logging.wandb_logging import LoggingConfig

# Narrow critics for a CPU; the authors' are 3 x 512.
agent = TQC("Pendulum-v1", critic_architecture=("128", "relu", "128", "relu"))
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

## Key hyperparameters

Defaults are read from `TQC.__init__`, the authors' `main.py` and `tqc/`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes |
| `actor_architecture` | `("256", "relu", "256", "relu")` | actor hidden layers |
| `critic_architecture` | three `"512", "relu"` | each critic's hidden layers |
| `num_critics` | `5` | distributional critics |
| `n_quantiles` | `25` | quantiles each critic predicts |
| `top_quantiles_to_drop_per_net` | `2` | pooled target quantiles dropped per critic; the paper tunes it per task (5 on Hopper, 0 on HalfCheetah, 2 on the others) |
| `gamma` | `0.99` | discount |
| `tau` | `0.005` | target networks' Polyak rate |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `257` | the step of the first update; the policy acts from the first step (no uniform warm-up) |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory and its sequence replay |
| `max_grad_norm`, `weight_decay` | `None`, `0.0` | gradient clipping, decoupled weight decay |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **Quantile critics.** Each critic's head has `n_quantiles` outputs
  (`NetworkConfig.critic_quantiles`). Every value query, for evaluation,
  the probes or an extension, reads their mean, Q(s, a). The target and the
  critics' loss read the quantiles (`q_values(..., quantiles=True)`).
- **Update order.** As the authors' code: the critics' step and the target
  update, then the actor's and the temperature's steps. The actor's loss
  reads the critics from before their step, since the authors' code builds
  all three losses first.
- **Extensions.** Those that shift the target by a value per row (IBRL,
  LCBGatedBootstrap, TargetBC) shift every quantile alike. Those that read one
  target value (CriticBlend, MCVarianceCorrection, BiasVorePenalty) need a
  `scalar_target` and refuse TQC.
- **Initialisation.** As the authors' code (torch's `nn.Linear` defaults):
  kernels and biases uniform in ±1/sqrt(fan_in) on every layer. Recurrent
  TQC keeps Ajax's initialisation, as recurrent SAC and REDQ do.
- **Shared maths.** The target is SAC's soft Bellman target
  (`ajax.agents.SAC.core`) on the kept quantiles, the actor step is SAC's on
  the critics' mean, and the critic step is the shared one.

## Results

The repository records no learning curves for TQC yet. Unit tests
(`tests/agents/TQC` and the contracts every agent shares) check the
truncation, the quantile Huber loss and the authors' settings. The probes
(`tests/probing`) train it briefly on tiny environments with known answers.

## API

::: ajax.agents.TQC.TQC.TQC
