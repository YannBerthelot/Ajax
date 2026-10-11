# PQN

Parallelised Q-Network is Q-learning without a replay buffer and without a
target network. It collects short rollouts from many environments at once,
as PPO does, and regresses the Q-network on Q(lambda) returns computed from
them. A LayerNorm after every hidden layer keeps this stable. It is built
for many vectorised environments, which replace the buffer as the source of
varied data.

**Paper.** Gallici et al., *Simplifying Deep Temporal Difference Learning*,
2024.

## Can I use?

| | PQN |
| --- | --- |
| Continuous actions (`Box`) | No |
| Discrete actions (`Discrete`) | Yes |
| Learning | On-policy rollouts, no replay buffer |
| Vectorised environments (`n_envs`) | Yes, and it relies on them |
| Recurrent memory (`memory=`) | No |
| Image observations (CNN encoder) | Yes, with `cnn_image_shape` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss` |

## Example

```python
from ajax import PQN
from ajax.logging.wandb_logging import LoggingConfig

agent = PQN("CartPole-v1")  # 8 envs, 128-step rollouts
logging = LoggingConfig(config={}, log_frequency=32_768, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=131_072, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

## Key hyperparameters

Defaults are read from `PQN.__init__`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `8` | parallel environments |
| `learning_rate` | `2.5e-4` | Adam step size |
| `architecture` | `("128", "relu", "128", "relu")` | hidden layers; each Dense is followed by a LayerNorm |
| `gamma` | `0.99` | discount |
| `q_lambda` | `0.65` | the lambda of the Q(lambda) returns |
| `n_steps` | `128` | rollout length per env |
| `n_epochs` | `4` | passes over each rollout |
| `num_minibatches` | `4` | minibatches per pass |
| `epsilon_start`, `epsilon_end` | `1.0`, `0.05` | exploration rate at the start and the end of its decay |
| `epsilon_decay_frac` | `0.5` | fraction of the run over which epsilon decays |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `td_loss_fn` | `None` | the loss; `None` is the squared error, `make_huber_td_loss(delta)` the Huber loss |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | `None`, `0`, `None` | convolutional encoder for flat image observations |
| `expose_recent_rollout` | `False` | keep the last rollout on `state.last_rollout` for extensions |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

A rollout holds `n_steps * n_envs` rows (1,024 by default), which must divide
evenly into `num_minibatches`; PQN raises otherwise.

## Notes

- **Returns.** With `V(s) = max_a Q(s, a)`, each step's return is
  `G_t = r_t + gamma ((1 - lambda) V(s_{t+1}) + lambda G_{t+1})`, and
  `G_t = r_t + gamma V(s_{t+1})` at the rollout's last step. A termination
  zeroes `V(s_{t+1})`; a time limit bootstraps on `V(s_{t+1})` but stops
  the lambda trace there.
- **One update.** The returns are computed once per rollout with the
  current network, then held fixed for `n_epochs` passes of
  `num_minibatches` gradient steps, the rollout reshuffled each pass.
- **Why several envs.** Without a buffer, the data's diversity comes from
  the parallel envs. The probing tests run PQN with 8 envs because with one
  env the value of the action it stops exploring drifts.
- **Exploration.** As DQN: epsilon decays linearly with the training
  fraction; evaluation is greedy.
- **Extensions.** `on_target` extensions reshape the returns before they are
  held fixed; `critic_loss` extensions add a term to every minibatch's TD
  loss.
- **Shared code.** PQN reuses DQN's Q-network wrapper, loss and
  epsilon-greedy collection, and PPO's minibatch shuffler; its own code is
  the LayerNorm network, the Q(lambda) returns and the on-policy loop.

## Results

The repository records no learning curves for PQN yet. The probing tests
(`tests/probing`) check its Q(lambda) returns, discounting, episode ends,
extension phases and checkpoint resume on tiny problems with exact answers.
On the speed benchmark (`benchmarks/agent_bench.py`: CartPole, 4 envs, 8,000
steps, on a Mac CPU) PQN ran at a median 7,332 steps per second (see
`PERFORMANCE_REPORT.md`).

## API

::: ajax.agents.PQN.PQN.PQN
