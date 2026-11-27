# DQN

Deep Q-Network learns one Q-value per discrete action and acts greedily on
them. It samples its training data from a replay buffer and regresses the
Q-network on temporal-difference targets computed with a separate target
network, refreshed every few hundred environment steps. It explores with an
epsilon-greedy rule whose epsilon decays linearly during training.

**Papers:** Mnih et al., *Human-level control through deep reinforcement
learning*, Nature 2015. Options: van Hasselt et al., *Deep Reinforcement
Learning with Double Q-learning*, 2016 (Double DQN); Wang et al., *Dueling
Network Architectures for Deep Reinforcement Learning*, 2016 (dueling
network). On vector observations Ajax follows CleanRL's `dqn.py`
(`vwxyzjn/cleanrl` at `fe8d8a0`); on images, the authors' code
(`google-deepmind/dqn` at `9d9b1d1`), which runs Atari only.

## When to use it

DQN learns off-policy from a replay buffer, reusing each transition many
times, so it suits discrete-action tasks where environment steps are
costly. With many cheap parallel environments, [PQN](pqn.md) learns
without a replay buffer; for continuous actions, see [SAC](sac.md).

| Feature | DQN |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | no |
| Memory (`memory=`) | no |
| Environments | gymnax (Brax and Playground tasks have continuous actions) |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `action`, `eval_action`, `choose_warmup_action`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import DQN
from ajax.logging import LoggingConfig

# A short run that shows the call, too short to learn CartPole: the
# default warm-up, CleanRL's, is 10,000 steps.
agent = DQN("CartPole-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(log_frequency=5_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=10_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

Double DQN, the dueling network and the loss are three arguments of the
same agent:

```python
from ajax import DQN
from ajax.networks.q_networks import DuelingQNetwork
from ajax.agents.critics import compute_mse_td_loss
from ajax.agents.DQN.DQN import compute_double_dqn_td_target

agent = DQN(
    "CartPole-v1",
    td_target_fn=compute_double_dqn_td_target,  # Double DQN
    q_network_cls=DuelingQNetwork,               # dueling heads
    td_loss_fn=compute_mse_td_loss,              # the squared error, named
)
state, _ = agent.train(seed=0, n_timesteps=2_000)
```

## Key hyperparameters

The defaults are CleanRL's on vector observations, its Adam and squared
loss included; on images the optimiser and the loss are the authors' (see
the notes). On CartPole-v1 they reach a return near 500 within CleanRL's
500,000 steps.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `learning_rate` | `2.5e-4` | step size of Adam (of centred RMSProp on images) |
| `architecture` | `("120", "relu", "84", "relu")` | hidden layers of the Q-network |
| `gamma` | `0.99` | discount |
| `buffer_size` | `10_000` | replay capacity |
| `batch_size` | `128` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `train_frequency` | `10` | environment steps between updates |
| `n_gradient_steps` | `1` | gradient steps per update |
| `target_update_interval` | `50` | gradient steps between target refreshes (500 environment steps at the defaults) |
| `tau` | `1.0` | refresh rate: `1.0` copies the weights, below `1` is a Polyak step |
| `epsilon_start`, `epsilon_end` | `1.0`, `0.05` | exploration rate at the start and the end of its decay |
| `epsilon_decay_frac` | `0.5` | fraction of the run over which epsilon decays |
| `epsilon_decay_start` | `0` | step at which the decay starts; epsilon is `epsilon_start` until then |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, the discount of the return whose spread divides the rewards: `gamma` if None, 0 for the single reward's |
| `td_target_fn` | `None` | the target; `None` is vanilla DQN, `compute_double_dqn_td_target` is Double DQN |
| `td_loss_fn` | `None` | the loss; `None` is the squared error on vector observations and the Huber loss with delta 1 on images |
| `q_network_cls` | `None` | the network; `None` is `QNetwork`, `DuelingQNetwork` the dueling one |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | `None`, `0`, `None` | convolutional encoder for flat image observations; `None` builds the reference network |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expert_policy` | `None` | an expert, as an action index per row: collection tracks it and stores its actions, and evaluations log its return |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Presets

`DQN.preset("Atari", env_id)` takes Mnih et al.'s Atari configuration
(`run_gpu:12-32`; [Presets](../guide/training.md#presets)): a
1,000,000-row replay, batches of 32, 50,000 warm-up steps, one update every
4 steps (`train_frequency=4`), the target refreshed every 10,000 steps
(`target_update_interval=2_500` updates), and epsilon held at 1 for the
warm-up (`epsilon_decay_start=50_000`) then falling to 0.1 over a million
steps. That million is a fraction of the reference's 50M-step run
(`epsilon_decay_frac=0.02`), so a shorter run decays faster. Atari's frame
skip and stack, no-op starts and reward clipping are the environment's.

## Differences from the reference

- **Step counting.** CleanRL numbers its steps from 0, so an update or
  refresh at Ajax's 10,500th step is CleanRL's at `global_step` 10,500.
  Ajax also updates and refreshes at step 10,000, where CleanRL waits for
  `global_step > learning_starts`.
- **The first convolution.** The authors' code gives it a padding argument
  of 1; Ajax does not pad it, as the paper describes. On 84 x 84 images
  both read 20 x 20.

## Notes

- **Target.** `r + gamma (1 - terminated) max_a' Q'(s', a')`; Double DQN
  picks `a'` with the online network and values it with the target network.
  A transition cut by a time limit bootstraps on its final observation;
  only a termination stops the bootstrap.
- **Updates per step.** Each iteration steps every environment once. From
  `learning_starts`, each time the steps collected reach a multiple of
  `train_frequency`, it takes `n_gradient_steps` gradient steps: one per 10
  environment steps at the defaults, whatever `n_envs` (with `n_envs` at
  or above `train_frequency`, one per iteration). `train_frequency=1`
  updates every iteration. An evaluation between two updates logs the
  training metrics of the last one.
- **Target refresh.** One mechanism covers both classic schemes: every
  `target_update_interval` gradient steps the target moves by `tau`. The
  hard copy is `tau=1.0`; a Polyak average is `tau < 1` with interval `1`.
  The interval counts updates so that it keeps its meaning in
  `train_offline` and pretraining, where no environment step is taken.
- **Exploration.** Epsilon decays linearly with the training fraction (the
  timestep over the run's total), from `epsilon_start` to `epsilon_end`
  over `epsilon_decay_frac` of the run, starting at step
  `epsilon_decay_start`. Before `learning_starts` every action is uniform.
  Evaluation is greedy.
- **Loss.** On vector observations, the squared error, as CleanRL. On
  images (`cnn_image_shape`), the Huber loss with delta 1, whose gradient is
  the TD error clipped to [-1, 1], as the original DQN clips it on Atari's
  rewards, themselves clipped to [-1, 1]. On unclipped rewards that loss
  learns poorly: with it, CartPole-v1 stays near a return of 10.
  `td_loss_fn` sets either: `compute_mse_td_loss`, `compute_huber_1_td_loss`
  or `make_huber_td_loss(delta)`.
- **Optimiser.** On vector observations, Adam at torch's defaults (eps
  `1e-8`), as CleanRL. On images, the authors' centred RMSProp (Graves
  2013): each step is the gradient over `sqrt(E[g^2] - E[g]^2 + eps)`,
  both averages decaying at 0.95. Their eps is 0.01 on a gradient summed
  over the batch; on Ajax's mean loss the same step takes
  `eps = 0.01 / batch_size**2`.
- **Network.** A plain MLP without LayerNorm (Mnih et al.'s network has
  none) and a linear head. As in the authors' Torch7 code, every layer
  starts as `nn.Linear` does, kernel and bias uniform in
  `±1/sqrt(fan_in)`.
- **Images.** With `cnn_image_shape` and no `cnn_spec`, DQN builds Mnih et
  al.'s network (`MNIH_CNN`: 32, 64 and 64 channels, kernels 8, 4 and 3,
  strides 4, 2 and 1, no padding, then 512 units) when the image is at
  least 36 pixels a side, the smallest it fits. Smaller images, such as
  MinAtar's 10 x 10, get MinAtar's network (`MINATAR_CNN`: one 3 x 3
  convolution of 16 channels, then 128 units). Both start as the MLP does.
- **Extensions.** `on_target` extensions reshape the target; `critic_loss`
  extensions add a term to the TD loss, with its gradient. With an
  `expert_policy`, `ExpertPrefill`, `ExpertWarmup`, `JSRLCurriculum`,
  `ExpertValueMetrics` (Q at the stored index) and the expert's
  observation features run on it.

## Results

The [reproduction results](../reproduction/results.md) set DQN beside
Stable-Baselines3's on CartPole, with learning curves; the comparison is
inconclusive so far. The probes (`tests/probing`) train it briefly on tiny
environments with known answers: its max bootstrap and action indexing,
discounting, time limits, extension phases, and a run split by a
checkpoint matching the run left whole. [Performance](../guide/performance.md)
compares its speed with the other agents'.

## API

::: ajax.agents.DQN.DQN.DQN

::: ajax.networks.q_networks.DuelingQNetwork

::: ajax.agents.DQN.DQN.compute_double_dqn_td_target

::: ajax.agents.critics.make_huber_td_loss
