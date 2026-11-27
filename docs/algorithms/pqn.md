# PQN

Parallelised Q-Network is Q-learning without a replay buffer and without a
target network. It collects short rollouts from many environments at once,
as PPO does, and regresses the Q-network on Q(lambda) returns computed from
them. A LayerNorm after every hidden layer keeps this stable.

**Paper:** Gallici et al., *Simplifying Deep Temporal Difference Learning*,
2024. Ajax follows the authors' code, purejaxql (`mttga/purejaxql` at
`47af6d7`).

## When to use it

PQN is built for many vectorised environments, which replace the buffer as
the source of varied data. It suits discrete-action tasks whose steps are
cheap, as in JAX simulators. With few environments, or costly steps,
[DQN](dqn.md) reuses its data through a replay buffer.

| Feature | PQN |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | no |
| Memory (`memory=`) | no |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `action`, `eval_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import PQN
from ajax.logging import LoggingConfig

agent = PQN("CartPole-v1")  # 128 envs, 32-step rollouts
logging = LoggingConfig(log_frequency=32_768, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=131_072, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

## Key hyperparameters

The defaults are purejaxql's MinAtar configuration
(`config/alg/pqn_minatar.yaml`), with a 128-128 network on vector
observations, which that configuration does not cover. Its CartPole
configuration is a [preset](#presets).

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `128` | parallel environments |
| `learning_rate` | `5e-4` | RAdam step size (the authors' optimiser, `optax.radam`) |
| `learning_rate_decay` | `"linear"` | decay the rate linearly to ~0 over the run's gradient steps, as every purejaxql configuration does; `None` keeps it constant |
| `architecture` | `("128", "relu", "128", "relu")` | hidden layers; each Dense is followed by a LayerNorm |
| `gamma` | `0.99` | discount |
| `q_lambda` | `0.65` | the lambda of the Q(lambda) returns |
| `n_steps` | `32` | rollout length per environment |
| `n_epochs` | `2` | passes over each rollout |
| `num_minibatches` | `32` | minibatches per pass |
| `epsilon_start`, `epsilon_end` | `1.0`, `0.05` | exploration rate at the start and the end of its decay |
| `epsilon_decay_frac` | `0.1` | fraction of the run over which epsilon decays |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `10.0` | gradient clipping by global norm (the authors'); `None` turns it off |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, the discount of the return whose spread divides the rewards: `gamma` if None, 0 for the single reward's |
| `td_loss_fn` | `None` | the loss; `None` is half the squared error (the authors'), `make_huber_td_loss(delta)` the Huber loss |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | `None`, `0`, `None` | convolutional encoder for flat image observations; `None` builds the reference network (see the notes) |
| `expose_recent_rollout` | `False` | keep the last rollout on `state.last_rollout` even when no extension reads it |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `expert_policy` | `None` | an expert, as an action index per row (see the notes) |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

A rollout holds `n_steps * n_envs` rows (4,096 by default), which must
divide evenly into `num_minibatches`; PQN raises otherwise.

## Presets

`PQN.preset(task, env_id)` takes purejaxql's configuration for the task
(`purejaxql/config/alg/pqn_<task>.yaml`; [Presets](../guide/training.md#presets));
its MinAtar configuration is the defaults.

| Argument | `CartPole` | `Atari` |
| --- | --- | --- |
| `n_envs`, `n_steps` | 32, 64 | 128, 32 |
| `num_minibatches`, `n_epochs` | 16, 4 | 32, 2 |
| `learning_rate`, `learning_rate_decay` | `1e-4`, `"linear"` | `2.5e-4`, `None` |
| `epsilon_end`, `epsilon_decay_frac` | 0.2, 0.2 | 0.001, 0.1 |
| `q_lambda`, `reward_scale` | 0.95, 0.1 | 0.65, 1.0 |
| `architecture` | 256-256 ReLU | the default |

Atari's frame skip, life loss and reward clipping are the environment's.

## Differences from the reference

- **Pixels.** The authors' Atari network divides pixels by 255; Ajax
  leaves that scaling to the environment.
- **Evaluation.** Ajax scores one episode per environment; purejaxql
  scores the last episode each of its test environments ends within a
  fixed window of steps.

## Notes

- **Returns.** As purejaxql computes them. With `V(s) = max_a Q(s, a)`,
  each step's return is
  `G_t = r_t + gamma ((1 - lambda) V(s_{t+1}) + lambda G_{t+1})`, and
  `G_t = r_t + gamma V(s_{t+1})` at the rollout's last step. At the end of
  an episode the return is the reward. A time limit ends an episode as a
  termination does, since gymnax's done flag does not tell them apart. The
  rollout's second-to-last step reads the last step's `V(s_T)` where its
  own `V(s_{T-1})` belongs, as purejaxql's backward pass does.
- **One update.** The returns are computed once per rollout with the
  current network, then held fixed for `n_epochs` passes of
  `num_minibatches` gradient steps, the rollout reshuffled each pass.
- **Several environments.** Without a buffer, the data's diversity comes
  from the parallel environments. With one environment, the value of an
  action PQN stops exploring drifts.
- **Exploration.** Epsilon decays linearly with the training fraction,
  read once per update at the rollout's start, as purejaxql reads it;
  evaluation is greedy.
- **Initialisation.** As the authors' code, which sets no initialiser:
  every Dense layer, the Q head included, starts with Flax's default
  `lecun_normal` kernel and a zero bias.
- **Images.** With `cnn_image_shape` and no `cnn_spec`, PQN builds the
  authors' image network: Mnih et al.'s (`MNIH_CNN`) for images of at
  least 36 pixels a side, MinAtar's (`MINATAR_CNN`) for smaller ones, each
  layer `he_normal` and followed by a LayerNorm. The image features go
  straight to the Q head; `architecture` is not used.
- **Extensions.** `on_target` extensions reshape the returns before they are
  held fixed; `critic_loss` extensions add a term to every minibatch's TD
  loss.
- **Expert.** With `expert_policy=` (an action index per row), collection
  tracks the expert, its observation features read it, and evaluations log
  its return. Under `JSRLCurriculum` the expert acts the first steps of
  each episode; Q is fitted at the actions the steps executed, whose
  rewards the returns hold, not at the policy's choice. JSRL with PQN is
  Ajax's composition, not a published pairing; as with epsilon-greedy, the
  expert's steps lean PQN's returns towards its behaviour.

## Results

The [reproduction results](../reproduction/results.md) set PQN beside its
authors' code on CartPole and Asterix-MinAtar, with learning curves. A
conformance check (`tests/conformance/test_pqn_purejaxql.py`) runs one
training iteration of each from the same state with the same random
numbers and compares them step by step. The probes (`tests/probing`) train
it briefly on tiny environments with known answers: its Q(lambda) returns,
discounting, episode ends, extension phases, and a run split by a
checkpoint matching the run left whole. [Performance](../guide/performance.md)
compares its speed with the other agents'.

## API

::: ajax.agents.PQN.PQN.PQN
