# PPO

Proximal Policy Optimization collects a fixed-length rollout from `n_envs`
parallel environments, then takes several passes of minibatch gradient steps on
it. A clipped objective keeps each update close to the policy that collected the
data, so the same rollout can be reused for `n_epochs` passes without the policy
running away. PPO is the default on-policy agent in Ajax: it handles discrete and
continuous actions, memory, image observations and the minibatch layouts of
Brax's PPO.

**Paper:** Schulman, Wolski, Dhariwal, Radford and Klimov, *Proximal Policy
Optimization Algorithms*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)).

## Can I use?

| Feature | PPO |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes (Gaussian; `squash=True` for a tanh-bounded one) |
| Memory (`memory=`) | yes: GRU, LSTM, transformer, Mamba ([Memory](../guide/memory.md)) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probing test) |

## Example

```python
from ajax import PPO
from ajax.logging.wandb_logging import LoggingConfig

agent = PPO("CartPole-v1", n_envs=4, n_steps=128, batch_size=32)
logging = LoggingConfig(config={}, log_frequency=4_096, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=20_480, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed, one column per evaluation
```

With memory, the update replays the rollout in contiguous chunks of
`bptt_length` steps, each started from the memory the policy had there:

```python
from ajax import PPO
from ajax.networks.memory import MemoryConfig

agent = PPO("CartPole-v1", n_envs=4, n_steps=128, batch_size=32,
            memory=MemoryConfig(kind="gru", hidden_size=32), bptt_length=32)
state, _ = agent.train(seed=0, n_timesteps=4_096)
```

## Key hyperparameters

Defaults are those of the constructor.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 4 | environments stepped in parallel |
| `n_steps` | 2048 | steps per environment in each rollout |
| `batch_size` | 64 | sets the number of minibatches (see the notes) |
| `num_minibatches` | 0 | when positive, the number of minibatches, as in Brax |
| `n_epochs` | 10 | passes over the rollout per update |
| `actor_learning_rate`, `critic_learning_rate` | 3e-4 | Adam step sizes (each a float or a schedule) |
| `adam_eps` | 1e-8 | Adam's epsilon |
| `gamma` | 0.99 | discount |
| `gae_lambda` | 0.95 | GAE's lambda |
| `clip_range` | 0.2 | the ratio clip; a float or a schedule of the timestep |
| `ent_coef` | 0 | entropy bonus weight |
| `vf_coef` | 1.0 | critic loss weight (Brax uses 0.5) |
| `max_grad_norm` | 0.5 | gradient clipping by global norm (`None` turns it off) |
| `normalize_advantage` | True | advantages standardised in each minibatch |
| `actor_architecture`, `critic_architecture` | `("128", "tanh", "128", "tanh")` | hidden layers, as width and activation strings |
| `normalize_observations`, `normalize_rewards` | False | running normalisation, as SB3's `VecNormalize` |
| `memory`, `bptt_length` | None | memory block, and the length of the replayed chunks |
| `expose_recent_rollout` | False | keep the last rollout on the state, for measurement extensions |

## Notes

- **`batch_size` sets the number of minibatches, not their size.** Unless
  `num_minibatches` is positive, the rollout of `n_steps * n_envs` samples
  splits into `max(n_steps, batch_size) // min(n_steps, batch_size)`
  minibatches. With the defaults that is 32 minibatches of 256 samples
  (`batch_size * n_envs`). This differs from Stable-Baselines3, where
  `batch_size` is the minibatch size.
- **Where the advantages are computed.** When the environments split evenly
  into the minibatches, each minibatch holds whole rollout fragments and its
  advantages are recomputed with the current critic, as Brax's PPO does.
  `unroll_length` cuts the rollout into shorter fragments that bootstrap at
  their ends. Otherwise the advantages are computed once on the whole rollout
  and the samples are shuffled flat; that is the case with the defaults (4
  environments, 32 minibatches) and with a single environment.
- **Brax-style options.** `use_vtrace_gae` (zero TD error at truncations),
  `fused_grad_clip` (one clip over actor and critic gradients),
  `log_std_state_independent`, `log_std_init`, `mean_kernel_init`, `squash`,
  `num_resets_per_eval` and `num_evals` reproduce choices of Brax's PPO. They
  are off by default.
- **Separate networks.** The actor and the critic are separate networks with
  separate optimizers; the value loss is not clipped. An entropy bonus on a
  squashed policy is a one-sample estimate of the executed action's entropy,
  as in Brax.
- **Normalisation.** With `normalize_observations=True` the loss sees the
  observations normalised with the statistics of the step that collected them,
  as SB3's `VecNormalize` and CleanRL do. `reward_normalization_gamma` divides
  rewards by the running spread of the discounted return instead of the
  single-step reward's.
- **Reward shaping.** `reward_shaping_fn(agent_state, rollout)` returns a bonus
  added to the rewards before the advantages are computed.
- **Extensions.** PPO folds the loss and target phases, so instrumentation such
  as `ConditioningMetrics` works on it (pass `expose_recent_rollout=True` so it
  can read the latest rollout). It has no `action` phase: extensions that act
  during collection need SAC. See [Extensions](../guide/extensions.md).
- A recurrent PPO whose sequences do not split into the requested minibatches
  trains on one minibatch and warns. `unroll_length` is for feedforward
  policies; recurrent ones use `bptt_length`.

## Results

No learning curves are published for PPO yet. The probing tests
(`tests/probing`) check that it learns the exact values of small problems
whose answers are known, and that a run split by a checkpoint equals the run
left whole. On the CPU speed benchmark (`benchmarks/agent_bench.py`, 4
environments, 8,000 steps) PPO ran at about 4,300 steps per second at main
`91ee10f` ([PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.PPO.PPO.PPO
