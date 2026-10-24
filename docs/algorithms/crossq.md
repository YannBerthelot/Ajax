# CrossQ

CrossQ is SAC without target networks. Its actor and critics are
batch-renormalised. The critics are trained on one pass over the replayed
rows and their next states together, so both share the batch statistics.
The bootstrap reads the second half of that pass, held constant. The actor
and the temperature step on every third update.

**Paper.** Bhatt et al., *CrossQ: Batch Normalization in Deep Reinforcement
Learning for Greater Sample Efficiency and Simplicity*, ICLR 2024
([arXiv:1902.05605](https://arxiv.org/abs/1902.05605)). Where the authors'
code (`adityab/CrossQ` at `a318a26`) is more specific, Ajax follows the code.

## Can I use?

| | CrossQ |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | No: the paper defines no recurrent batch normalisation |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `transform_action`, `actuate` |

## Example

```python
from ajax import CrossQ
from ajax.logging.wandb_logging import LoggingConfig

# Narrow critics for a CPU; the authors' are 2048 wide.
agent = CrossQ("Pendulum-v1", critic_width=256, learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

## Key hyperparameters

Defaults are read from `CrossQ.__init__`, the authors' `train.py -algo crossq`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `learning_rate` | `1e-3` | the actor's, the critics' and the temperature's Adam step size |
| `alpha_learning_rate` | `None` | the temperature's own, when given |
| `adam_beta_1` | `0.5` | the actor's and the critics' Adam beta1 (the temperature's stays 0.9) |
| `actor_width`, `critic_width` | `256`, `2048` | units of the two hidden layers |
| `num_critics` | `2` | critics; the target takes their minimum |
| `policy_delay` | `3` | updates per actor and temperature step |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `5_000` | steps of uniform random actions before the first update |
| `alpha_init` | `1.0` | initial entropy weight |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm`, `weight_decay` | `None`, `0.0` | gradient clipping, decoupled weight decay |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **Batch Renorm.** Each network normalises its input and the output of each
  hidden ReLU layer with `BatchRenorm`, the `"batchrenorm"` architecture
  token. This is the authors' layer: plain batch norm for its first 100,000
  training passes, then renormalised towards the running statistics (r
  clipped to 3, d to 5), momentum 0.99. It trains when a loss runs it in
  training (`apply_train`). Every other read uses its running statistics:
  acting, evaluation, the next actions of the target, and the critics in the
  actor's loss. The statistics live on the network's state (`batch_stats`),
  apart from the trained parameters.
- **Temperature.** The authors' code steps the temperature down `alpha (H - H_target)`, alpha
  where SAC's loss has log alpha, so its gradient is alpha times SAC's (`sbx/sac/sac.py:414-416`;
  SBX later took the log). Ajax follows the authors (`core.alpha_temperature_loss_fn`).
- **Initialisation.** As the authors' code (flax's defaults): lecun-normal
  kernels and zero biases on every layer, the mean and log-std heads
  included.
- **No target networks.** The critics' state keeps no target parameters.
  `IBRL` and `LCBGatedBootstrap`, which bootstrap on target critics, are
  refused.
- **Shared maths.** The target, the critics' regression and the actor's
  objective are SAC's (`ajax.agents.SAC.core`), and the extensions fold
  through the shared critic step's two halves (`fold_on_target`,
  `fold_critic_loss`).
- **Cost.** The critics' pass doubles its batch, and the default critics are
  2048 wide. On a CPU, narrower critics (`critic_width=256`) are much
  cheaper.

## Results

The repository records no learning curves for CrossQ yet. Unit tests
(`tests/agents/CrossQ` and the contracts every agent shares) check the
authors' settings, the critics' joint pass and the policy delay. The probes
(`tests/probing`) train it briefly on tiny environments with known answers.
`BatchRenorm` matches the authors' layer bit for bit. A conformance check
(`tests/conformance/test_crossq_original.py`) runs one update of Ajax's CrossQ
and of `adityab/CrossQ` from the same state, with the same batch and draws.
It covers a critic step, the policy delay's actor and temperature step, and
the same step past Batch Renorm's warm-up. It compares every parameter,
running statistic, Adam moment and count.

## API

::: ajax.agents.CrossQ.CrossQ.CrossQ
