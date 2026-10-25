# DQN

Deep Q-Network learns one Q-value per discrete action and acts greedily on
them. It regresses the Q-network on temporal-difference targets computed
with a separate target network, refreshed every few hundred updates, and
samples its training data from a replay buffer. It explores with an
epsilon-greedy rule whose epsilon decays linearly during training. Double
DQN, the dueling network and the squared loss are options of the same agent.

**Papers.** Mnih et al., *Human-level control through deep reinforcement
learning*, Nature 2015. Options: van Hasselt et al., *Deep Reinforcement
Learning with Double Q-learning*, 2016 (Double DQN); Wang et al., *Dueling
Network Architectures for Deep Reinforcement Learning*, 2016 (dueling
network).

## Can I use?

| | DQN |
| --- | --- |
| Continuous actions (`Box`) | No |
| Discrete actions (`Discrete`) | Yes |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | No |
| Image observations (CNN encoder) | Yes, with `cnn_image_shape` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `choose_warmup_action` |

## Example

```python
from ajax import DQN
from ajax.logging.wandb_logging import LoggingConfig

agent = DQN("CartPole-v1", n_envs=4)
logging = LoggingConfig(config={}, log_frequency=5_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=10_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

The variants compose through three arguments:

```python
from ajax import DQN
from ajax.networks.q_networks import DuelingQNetwork
from ajax.agents.critics import compute_mse_td_loss
from ajax.agents.DQN.DQN import compute_double_dqn_td_target

agent = DQN(
    "CartPole-v1",
    td_target_fn=compute_double_dqn_td_target,  # Double DQN
    q_network_cls=DuelingQNetwork,               # dueling heads
    td_loss_fn=compute_mse_td_loss,              # squared error instead of Huber
)
state, _ = agent.train(seed=0, n_timesteps=2_000)
```

## Key hyperparameters

Defaults are read from DQN's fields.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `learning_rate` | `2.5e-4` | step size of centered RMSProp (see Notes) |
| `architecture` | `("128", "relu", "128", "relu")` | hidden layers of the Q-network |
| `gamma` | `0.99` | discount |
| `buffer_size` | `100_000` | replay capacity |
| `batch_size` | `64` | replay minibatch |
| `learning_starts` | `1_000` | steps of uniform random actions before the first update |
| `n_gradient_steps` | `1` | gradient steps per iteration |
| `target_update_interval` | `500` | gradient steps between target refreshes |
| `tau` | `1.0` | refresh rate: `1.0` copies the weights, below `1` is a Polyak step |
| `epsilon_start`, `epsilon_end` | `1.0`, `0.05` | exploration rate at the start and the end of its decay |
| `epsilon_decay_frac` | `0.5` | fraction of the run over which epsilon decays |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, divide by the spread of the discounted return |
| `td_target_fn` | `None` | the target; `None` is vanilla DQN, `compute_double_dqn_td_target` is Double DQN |
| `td_loss_fn` | `None` | the loss; `None` is the Huber loss with delta 1, `compute_mse_td_loss` the squared error |
| `q_network_cls` | `None` | the network; `None` is `QNetwork`, `DuelingQNetwork` the dueling one |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | `None`, `0`, `None` | convolutional encoder for flat image observations; `None` builds the reference network (see Notes) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## Notes

- **Target.** `r + gamma (1 - terminated) max_a' Q'(s', a')`; Double DQN
  picks `a'` with the online network and values it with the target network.
- **Updates per step.** Each iteration steps every env once, then (from
  `learning_starts`) takes `n_gradient_steps` gradient steps. With `n_envs=4`
  that is one gradient step per four environment steps.
- **Target refresh.** One mechanism covers both classic schemes: every
  `target_update_interval` gradient steps the target moves by `tau`. The
  hard copy is `tau=1.0`; a Polyak average is `tau < 1` with interval `1`.
- **Exploration.** Epsilon decays linearly with the training fraction (the
  timestep over the run's total), from `epsilon_start` to `epsilon_end`
  over the first `epsilon_decay_frac` of the run. Before `learning_starts`
  every action is uniform. Evaluation is greedy.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **Loss.** The default is the Huber loss with delta 1, whose gradient is
  the TD error clipped to [-1, 1], as the original DQN clips it.
  `make_huber_td_loss(delta)` sets another delta; `compute_mse_td_loss` is the
  squared error.
- **Optimiser.** The authors' centered RMSProp (Graves 2013): each step is
  the gradient over `sqrt(E[g^2] - E[g]^2 + eps)`, both averages decaying
  at 0.95. Their eps is 0.01 on a gradient summed over the batch; on Ajax's
  mean loss the same step takes `eps = 0.01 / batch_size**2`.
- **Network.** A plain MLP without LayerNorm (Mnih et al.'s network has
  none) and a linear head. It lives in `state.actor_state`, so the shared
  evaluation reads it like any actor. As in the authors' Torch7 code, every
  layer of the MLP and the head starts as `nn.Linear` does, kernel and bias
  uniform in `±1/sqrt(fan_in)`.
- **Images.** With `cnn_image_shape` and no `cnn_spec`, DQN builds Mnih et
  al.'s network (`MNIH_CNN`: 32, 64 and 64 channels, kernels 8, 4 and 3,
  strides 4, 2 and 1, no padding, then 512 units) when the image is at
  least 36 pixels a side, the smallest it fits. Smaller images, such as
  MinAtar's 10 x 10, get MinAtar's network (`MINATAR_CNN`: one 3 x 3
  convolution of 16 channels, then 128 units). Both start as the MLP does.
  The authors' first convolution is given a padding argument of 1; Ajax
  does not pad it, as the paper describes. On 84 x 84 images both read 20 x
  20.
- **Extensions.** `on_target` extensions reshape the target; `critic_loss`
  extensions add a term to the TD loss, with its gradient.

## Results

The repository records no learning curves for DQN yet. Unit tests
(`tests/agents/DQN` and the contracts every agent shares) check its pieces
against exact contracts, without learning. The probes (`tests/probing`)
train it briefly on tiny environments with known answers: its max bootstrap
and action indexing, discounting, time limits, extension phases, and a run
split by a checkpoint matching the run left whole. On the CPU speed
benchmark (`benchmarks/agent_bench.py`: CartPole, 4 envs, 8,000 steps,
updates from step 1,000) DQN ran at 16,662 steps per second of execution
(6,395 with compilation) at main `a8a13c2` (see
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.DQN.DQN.DQN

::: ajax.networks.q_networks.DuelingQNetwork

::: ajax.agents.DQN.DQN.compute_double_dqn_td_target

::: ajax.agents.critics.make_huber_td_loss
