# UDRL

Upside-Down RL turns reinforcement learning into supervised learning. The
policy takes a *command* next to the observation: a return to obtain and a
number of steps to obtain it in. It learns, from its own past experience, which
action led to which outcome, then is asked for slightly better outcomes than
the best it has seen. There is no critic, no value estimate and no discount by
default.

**Papers:** Schmidhuber, *Reinforcement Learning Upside Down: Don't Predict
Rewards -- Just Map Them to Actions*, 2019
([arXiv:1912.02875](https://arxiv.org/abs/1912.02875)), which sets out the
idea. The training procedure Ajax implements (its Algorithms 3 to 5) is that
of Srivastava, Shyam, Mutz, Jaśkowski and Schmidhuber, *Training Agents using
Upside-Down Reinforcement Learning*, 2019 (arXiv:1912.02877).

## Can I use?

| Feature | UDRL |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, tanh-bounded |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `actor_loss` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probing test) |

UDRL is not exported from `ajax`: import it from `ajax.agents.UDRL.UDRL`.

## Example

```python
from ajax.agents.UDRL.UDRL import UDRL
from ajax.logging.wandb_logging import LoggingConfig

agent = UDRL("CartPole-v1", n_envs=4)
logging = LoggingConfig(config={}, log_frequency=10_240, use_wandb=False)
state, metrics = agent.train(seed=[1, 2], n_timesteps=51_200, logging_config=logging)
print(metrics["Train/episodic mean reward"])  # returns of the episodes collected
print(state.command_target_return)            # the return the agent now asks for
```

UDRL has no separate evaluation: a command-conditioned policy needs a command
to be evaluated with. With a `logging_config`, `train` returns the training
metrics of the iterations it logged (`Train/episodic mean reward`,
`Train/actor_loss`, `Train/mean_rtg`, `Train/mean_horizon`,
`Train/n_completed_episodes`).

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 1 | environments stepped in parallel |
| `n_steps` | 64 | length of the segment collected per iteration |
| `buffer_capacity` | 64 | segments kept in the replay |
| `n_updates_per_iter` | 64 | gradient steps per iteration |
| `batch_size` | 64 | samples per gradient step |
| `actor_learning_rate` | 3e-4 | Adam step size |
| `actor_architecture` | `("128", "relu", "128", "relu")` | hidden layers |
| `max_grad_norm` | None | gradient clipping by global norm |
| `gamma` | 1.0 | discount of the returns in the commands |
| `bc_loss_type` | `"nll"` | `"nll"` (log-likelihood) or `"mse"` |
| `command_return_init`, `command_horizon_init` | 1.0, 100.0 | the first command |
| `command_topk` | 32 | best episodes the next command is built from |
| `command_return_boost` | 1.0 | multiplies the top episodes' mean return |
| `command_target_tau` | 1.0 | smoothing of the command (1: replaced every iteration) |
| `command_scale_r`, `command_scale_h` | 0.02, 0.01 | scaling of the command before the network |

## How it works

Each iteration:

1. collects `n_steps` steps per environment with the current policy and
   commands, decreasing each command as reward is received and steps pass;
2. stores that segment in a first-in first-out replay of `buffer_capacity`
   segments;
3. sets the next command from the `command_topk` best episodes in the replay:
   a return between their mean and their mean plus one standard deviation, and
   their mean length as the horizon;
4. takes `n_updates_per_iter` supervised steps. Each sample is a state, the
   action taken there, and as command what actually followed: the reward
   summed up to a random later step of the same episode, and that number of
   steps.

## Notes

- **Deviation: segments, not episodes.** The replay stores fixed-length
  segments rather than whole episodes, so arrays keep a fixed shape. A training
  command therefore spans at most `n_steps` steps. The top-K statistics read
  episodes whole across segments.
- **Command scaling.** The command enters the network multiplied by
  `command_scale_r` and `command_scale_h` (the paper's LunarLander values), so
  that it is of order one; the replay keeps it unscaled.
- The policy's encoder ends with a LayerNorm, as Ajax's earlier networks did.
- `n_epochs` is accepted and ignored: UDRL takes `n_updates_per_iter` steps
  per iteration (its Algorithm 3).
- The command is carried in the last two entries of the observation; with
  `cnn_image_shape` the image is flattened in front of them.

## Results

No learning curves are published for UDRL yet. The probing tests check, on
small chains whose answers are known, that it learns from the commands that
actually followed and scales its command alike when acting and when training,
and that a run split by a checkpoint equals the run left whole.

## API

::: ajax.agents.UDRL.UDRL.UDRL
