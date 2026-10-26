# UDRL

Upside-Down RL turns reinforcement learning into supervised learning. The
policy takes a *command* next to the observation: a return to obtain and a
number of steps to obtain it in. It learns, from its own past experience, which
action led to which outcome, then is asked for slightly better outcomes than
the best it has seen. There is no critic, no value estimate and no discount by
default.

**Papers:** Schmidhuber, *Reinforcement Learning Upside Down: Don't Predict
Rewards -- Just Map Them to Actions*, 2019
([arXiv:1912.02875](https://arxiv.org/abs/1912.02875)). Ajax follows its
training procedure; the code cites it as Algorithms 3 to 5. The companion
paper by Srivastava, Shyam, Mutz, Jaśkowski and Schmidhuber, *Training Agents
using Upside-Down Reinforcement Learning*, 2019
([arXiv:1912.02877](https://arxiv.org/abs/1912.02877)), reports the first
experiments.

## Can I use?

| Feature | UDRL |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, tanh-bounded, mapped onto the environment's bounds |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `actor_loss` , `transform_action`, `actuate`, `observe` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probe and resume contract) |

## Example

```python
from ajax import UDRL
from ajax.logging.wandb_logging import LoggingConfig

agent = UDRL("CartPole-v1", n_envs=4)
logging = LoggingConfig(config={}, log_frequency=10_240, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=51_200, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])   # the policy, commanded as below
print(evaluations["Train/episodic mean reward"])  # returns of the episodes collected
print(state.command_target_return)                # the return the agent now asks for
```

With a `logging_config`, every `log_frequency` steps UDRL evaluates its
policy on `num_episode_test` episodes and `train` returns, next to the
evaluation (`Eval/episodic mean reward`, `Eval/episodic entropy`,
`Eval/mean episodic length`, and `Eval/commanded return` and
`Eval/commanded horizon`, the command each episode starts from), the
training metrics of those iterations
(`Train/episodic mean reward`, `Train/actor_loss`, `Train/mean_rtg`,
`Train/mean_horizon`, `Train/n_completed_episodes`).

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
| `gamma` | 1.0 | discount of the returns-to-go logged as `Train/mean_rtg`, a diagnostic only: commands, the top-K statistics and the training targets sum rewards undiscounted, as in the paper |
| `command_return_init`, `command_horizon_init` | 1.0, 100.0 | the first command |
| `command_topk` | 32 | best episodes the next command is built from |
| `command_return_boost` | 1.0 | multiplies the top episodes' mean return |
| `command_target_tau` | 1.0 | smoothing of the command (1: replaced every iteration) |
| `command_scale_r`, `command_scale_h` | 0.02, 0.01 | scaling of the command before the network |
| `normalize_observations`, `normalize_rewards`, `reward_normalization_gamma` | False, False, None | running normalisation |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | None, 0, None | convolutional encoder for flat image observations, the features after the image, and the encoder's layers; the command joins the encoder's output |
| `episode_length`, `action_repeat` | None, 1 | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `expert_policy` | `None` | an expert the rollout runs alongside: a residual policy (`ResidualPolicy`) or gain mode (`ExpertGains`) acts through it and the observation features read it |
| `eval_expert_policy` | `None` | an expert whose return each evaluation logs, which takes no other part in the run |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## How it works

Each iteration:

1. collects `n_steps` steps per environment with the current policy and
   commands, decreasing each command as reward is received and steps pass;
2. stores that segment in a first-in first-out replay of `buffer_capacity`
   segments;
3. sets the next command from the `command_topk` best episodes in the replay:
   a return between their mean and their mean plus one standard deviation, and
   their mean length as the horizon;
4. takes `n_updates_per_iter` supervised steps, each maximising the
   log-likelihood of the action taken. Each sample is a state, the action
   taken there, and as command what actually followed: the reward summed up
   to a random later step of the same episode, and that number of steps.

## Notes

- **Segments, read as episodes.** The replay stores fixed-length segments
  rather than whole episodes, so arrays keep a fixed shape. The sampler and
  the top-K statistics read episodes whole across segments, so a training
  command can span any part of an episode the replay holds, not only one
  segment.
- **Evaluation.** Each evaluated episode starts from the current command
  (`state.command_target_return` and its horizon), without the random spread
  that collection adds, and the command decreases as in collection. The
  policy acts deterministically.
- **Command scaling.** The command enters the network multiplied by
  `command_scale_r` and `command_scale_h` (the paper's LunarLander values), so
  that it is of order one; the replay keeps it unscaled.
- The policy's encoder ends with a LayerNorm, as Ajax's earlier networks did.
- The command is carried in the last two entries of the observation, after
  the observation features; with `cnn_image_shape` the image is flattened in
  front of them.

## Results

No learning curves are published for UDRL yet. Unit tests
(`tests/agents/UDRL` and the contracts every agent shares) check its pieces
against exact contracts, without learning. The probes (`tests/probing`)
train it briefly on tiny environments with known answers: that it learns
from the commands that actually followed and scales its command alike when
acting and when training, its extension phases, and a run split by a
checkpoint matching the run left whole. On the CPU speed benchmark
(`benchmarks/agent_bench.py`: Pendulum, 4 environments, 8,000 steps) UDRL
ran at 12,575 steps per second of execution (4,441 with compilation) at main
`a8a13c2`; its trials spread from 9,604 to 15,069
([PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.UDRL.UDRL.UDRL
