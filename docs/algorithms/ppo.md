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
| Memory (`memory=`) | yes: GRU, LSTM, transformer, Mamba ([Memory](../guide/memory.md)), without extensions |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probe and resume contract) |

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
| `n_epochs` | 10 | passes over the rollout per update, each reshuffled |
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
| `reward_normalization_gamma` | None | with `normalize_rewards`, divide by the spread of the discounted return |
| `memory`, `bptt_length` | None | memory block, and the length of the replayed chunks |
| `unroll_length` | None | use Brax's minibatching: fragments of this length, advantages per minibatch (feedforward only) |
| `log_std_state_independent`, `log_std_init` | True, 0.0 | one learned log standard deviation per action dimension, and its initial value (see the notes) |
| `actor_bias_init`, `critic_bias_init` | None | initialisers by name ([Networks](../guide/networks.md)) |
| `encoder_kernel_init` | None | the hidden layers' initialiser; None is `"orthogonal(np.sqrt(2))"` (Baselines' ppo2; PureJaxRL's recurrent PPO, with `memory=`) |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | None, 0, None | convolutional encoder for flat image observations |
| `episode_length`, `action_repeat` | None, 1 | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `expose_recent_rollout` | False | keep the last rollout on the state, for measurement extensions |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **`batch_size` sets the number of minibatches, not their size.** Unless
  `num_minibatches` is positive, the rollout of `n_steps * n_envs` samples
  splits into `max(n_steps, batch_size) // min(n_steps, batch_size)`
  minibatches. With the defaults that is 32 minibatches of 256 samples
  (`batch_size * n_envs`). This differs from Stable-Baselines3, where
  `batch_size` is the minibatch size.
- **Shuffling.** Every epoch draws a new split of the rollout into
  minibatches, as OpenAI Baselines' and CleanRL's PPO do.
- **Where the advantages are computed.** The advantages are computed once on
  the whole rollout and its `n_steps * n_envs` samples are shuffled flat into
  the minibatches, as in the PPO paper's Algorithm 1 and OpenAI Baselines'
  ppo2. `unroll_length` switches to Brax's PPO: the rollout is cut into
  fragments of that length that bootstrap at their ends, each minibatch holds
  whole fragments, and its advantages are recomputed with the current critic.
  `unroll_length=n_steps` keeps each environment's whole rollout as one
  fragment. PPO refuses at construction a length that does not split the
  rollout evenly into the minibatches.
- **Actions.** A continuous PPO samples an unbounded Gaussian by default;
  Ajax clips the sample to `[-1, 1]` and maps that range onto the
  environment's bounds, so on gymnax Pendulum (torque in `[-2, 2]`) one unit
  of the Gaussian is two units of torque. `squash=True` bounds it with
  `tanh` instead. Evaluation uses the mean action.
- **Exploration noise.** The Gaussian learns one log standard deviation per
  action dimension, whatever the observation, starting at 0 (a standard
  deviation of 1) and never clipped, as PPO's reference code does: OpenAI
  Baselines at `ea25b9e` (ppo2's `common/distributions.py:96-106`; ppo1's
  `ppo1/mlp_policy.py:36-39`, `gaussian_fixed_var=True`), CleanRL at
  `fe8d8a0` (`ppo_continuous_action.py:129`) and PureJaxRL at `31756b1`
  (`ppo_continuous_action.py:41`). Unlike theirs, it is updated by the
  actor's own optimizer and gradient clip, the critic having its own (see
  separate networks). `log_std_state_independent=False` with
  `log_std_init=-1.0` builds the head PPO had before: a log standard
  deviation computed from the observation, starting at -1 and clipped to
  `[-20, 2]`, as SAC's is; a checkpoint saved with that head restores only
  into that network, and restoring it into the default one is refused with
  this advice. A PID policy keeps its own log standard deviation, which
  starts at `log_std_init` and is clipped to `[-20, 2]`.
- **Brax-style options.** `fused_grad_clip` (one clip over actor and critic
  gradients), `mean_kernel_init`, `squash`, `num_resets_per_eval` and
  `num_evals` reproduce choices of Brax's PPO. They are off by default, as
  is `encoder_layer_norm` (a LayerNorm on the encoder's output).
- **Separate networks.** The actor and the critic are separate networks with
  separate optimizers; the value loss is not clipped. As in Baselines' ppo2,
  their hidden layers start orthogonal with gain `sqrt(2)`, the policy's
  output layer with gain 0.01 and the value layer with gain 1. An entropy bonus on a
  squashed policy is a one-sample estimate of the executed action's entropy,
  as in Brax.
- **Normalisation.** With `normalize_observations=True` the loss sees the
  observations normalised with the statistics of the step that collected them,
  as SB3's `VecNormalize` and CleanRL do.
- **Extensions.** PPO folds the loss and target phases, so instrumentation such
  as `ConditioningMetrics` works on it (pass `expose_recent_rollout=True` so it
  can read the latest rollout). It has no `action` phase: extensions that act
  during collection need SAC. A recurrent PPO refuses extensions. See
  [Extensions](../guide/extensions.md).
- A recurrent PPO whose sequences do not split into the requested minibatches
  trains on one minibatch and warns. `unroll_length` is for feedforward
  policies; recurrent ones use `bptt_length`.

## Results

No learning curves are published for PPO yet. Unit tests (`tests/agents/PPO`
and the contracts every agent shares) check its pieces against exact
contracts, without learning, such as the fresh split of each epoch. The
probes (`tests/probing`) train it briefly on tiny environments with known
answers: the exact values of small problems, episode ends, memory, extension
phases, and a run split by a checkpoint matching the run left whole. On the
CPU speed benchmark (`benchmarks/agent_bench.py`: CartPole, 4 environments,
8,000 steps) PPO ran at 17,817 steps per second of execution (4,298 with
compilation) at main `a8a13c2`
([PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.PPO.PPO.PPO
