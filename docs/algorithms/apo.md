# APO

Average Policy Optimization is PPO for the average-reward criterion.
Instead of discounting future rewards, it learns how much reward the policy
earns per step on average, and judges each action by how much better than
that average it does.

**Paper:** Ma et al., *Average-Reward Reinforcement Learning with Trust Region
Methods*, 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)). Ajax
follows the authors' official code, `xtma/apo` at `bfcd997` (on their fork
of rlpyt, `xtma/rlpyt` at `e28bb20`), where it is more specific than the
paper.

## When to use it

APO suits continuing tasks, such as holding a setpoint, where there is no
natural horizon and a discount factor only shortens the agent's view. On
episodic tasks with a clear end, [PPO](ppo.md) is the usual choice. Its
off-policy counterpart is [ASAC](asac.md).

| Feature | APO |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, a Gaussian around a `tanh`-bounded mean, clipped and mapped onto the environment's bounds |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `eval_action`, `transform_action`, `actuate`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import APO
from ajax.logging import LoggingConfig

agent = APO("Pendulum-v1", normalize_observations=True)
logging = LoggingConfig(log_frequency=10_000, backends=())
state, evaluations = agent.train(seed=[1, 2], n_timesteps=50_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
print(state.average_reward)  # the estimated reward per step, one per seed
```

This budget shows the calls; it is too short to learn the swing-up.

## Key hyperparameters

The defaults are the official code's MuJoCo configuration
(`apo/experiments/configs/mujoco/apg/mujoco_appo.py`). There is no
discount, as in the authors' run grid (`run_exp.py`), which replaces the
configuration's 0.99. The grid also sweeps `gae_lambda`, `alpha` and `nu`
per task; the paper does not report which values it kept, so Ajax takes the
configuration's.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 10 | environments stepped in parallel |
| `n_steps` | 200 | steps per environment in each rollout |
| `num_minibatches` | 20 | minibatches per epoch; `n_steps * n_envs` must be a multiple of it |
| `batch_size` | 64 | with `num_minibatches=0`, sets the number of minibatches as in [PPO](ppo.md) |
| `n_epochs` | 10 | passes over the rollout per update, each reshuffled |
| `actor_learning_rate`, `critic_learning_rate` | 3e-4 | the step sizes of the one Adam over both networks (see the notes) |
| `adam_eps` | 1e-8 | Adam's epsilon |
| `vf_coef` | 1 | the value loss's weight in the summed loss |
| `alpha` | 0.1 | update rate of the running averages of the reward rate and of the values |
| `nu` | 1 | weight of the value-bias penalty |
| `gae_lambda` | 0.95 | GAE's lambda |
| `clip_range` | 0.2 | the ratio clip; a float or a schedule of the timestep |
| `ent_coef` | 0.01 | entropy bonus weight |
| `max_grad_norm` | 10 | clips the gradient's global norm over both networks at once; `None` turns it off |
| `normalize_advantage` | False | standardise the advantages over the whole rollout |
| `actor_architecture`, `critic_architecture` | `("64", "tanh", "64", "tanh")` | hidden layers |
| `normalize_observations`, `normalize_rewards` | False | running normalisation |
| `reward_normalization_gamma` | None | with `normalize_rewards`, the discount of the return whose spread divides the rewards: 0.99 if None, 0 for the single reward's |
| `expert_policy` | None | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `actor_bias_init`, `critic_bias_init` | None | the heads' bias initialisers by name ([Networks](../guide/networks.md)) |
| `encoder_kernel_init` | `"torch_linear"` | the hidden layers start as torch's `nn.Linear`; any initialiser by name sets their kernels instead |
| `log_std_bounds` | None | the `(low, high)` clip of the continuous actor's log standard deviation; `None` leaves it unclipped |
| `episode_length`, `action_repeat` | None, 1 | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `expose_recent_rollout` | False | keep the last rollout on the state even when no extension reads it |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## How it works

Each update first moves two running averages towards the rollout's means, at
rate `alpha`: the reward rate `rho` (the mean reward) and `b` (the mean
predicted value). The first update sets them to the first rollout's means,
as the official code does. The advantages are GAE without a discount on the
differential reward `r - rho`. The critic fits the differential value; its
target is shifted by `nu * b`, which pulls the values back towards a zero mean.
Without that penalty the values are defined only up to a constant and can
drift. Each minibatch then takes one step of both networks down the sum of
PPO's clipped surrogate, the value loss and the entropy bonus.

## Differences from the reference

Each is a default the constructor can set.

- **Output layers.** The mean layer starts `lecun_uniform` and the value
  layer `orthogonal(1.0)`, both with zero biases; the official code leaves
  them at torch's `nn.Linear` default, as it does the hidden layers.
- **Actions.** As for every agent, Ajax clips the sample to `[-1, 1]` and
  maps it onto the environment's bounds
  ([Environments](../guide/environments.md#actions)). The official code
  hands the raw sample to the environment: its MuJoCo tasks clamp it to
  their control range in the simulation, and Gymnasium's charge the
  control cost on the unclamped action.

## Notes

- **Episode ends.** As in the official code, the advantage chain stops at
  every episode end and the next value is dropped there, a time limit
  included (the reference's done flag does not tell them apart). A step
  that hit a time limit leaves the losses, which average over the other
  steps.
- **Minibatches.** Every epoch draws a new split of the rollout into
  minibatches, as the official code does. `n_steps * n_envs` must be a
  multiple of `num_minibatches`; with `num_minibatches=0`, `n_steps` and
  `batch_size` must divide one another. The constructor refuses them
  otherwise.
- **Advantage normalisation** is off, as in the official configuration.
  With `normalize_advantage=True` it runs once over the whole rollout, not
  per minibatch as in Ajax's PPO, by the trained steps' mean and unbiased
  standard deviation (floored at `1e-6`), as the official code would.
- **Evaluation: the return and the reward rate count episodes
  differently.** Each evaluation runs the deterministic policy and logs
  both:
  - `Eval/episodic mean reward` (and the length and entropy) is the mean
    over `num_episode_test` episodes, one per environment, as for every
    agent. `num_episode_test` defaults to 10.
  - `Eval/mean average reward`, the reward rate, follows the official
    evaluation, whatever `num_episode_test` is: 10 environments of 1,000
    steps each, each restarting when its episode ends, a time limit
    included. The run stops after the step at which 10 trajectories have
    ended, and the rate is the mean, over every trajectory ended by then,
    of its return divided by its length; one still running is left out.

  The official sampler checks the trajectory count every 0.1 s of wall
  clock, so where it stops depends on timing; Ajax checks after every step,
  as rlpyt's serial sampler does. With the default `num_episode_test=10`,
  both measures read one run of the same 10 environments, so the rate adds
  steps only on a task whose time limit is under 1,000. With any other
  `num_episode_test`, the rate runs 10 environments of its own.
- **Networks.** The actor and the critic each have two hidden layers of 64
  `tanh` units, as the official code's mean and value networks. They start
  as its torch `nn.Linear` layers do: kernel and bias uniform in
  `±1/sqrt(fan_in)` (`encoder_kernel_init="torch_linear"`).
- **Actor head.** A continuous APO's actor is the official code's: a
  Gaussian whose mean is `tanh` of the mean layer, with one learned log
  standard deviation per action dimension, starting at 0 and not clipped.
  The sample itself is not squashed: the log-probability and the entropy
  are the Gaussian's, of the raw sample. The evaluation acts with the mean,
  as the official evaluation does. `log_std_bounds` adds a clip. A PID
  policy's output takes the same `tanh`, and its log standard deviation the
  same clip. A discrete APO ignores the head.
- **Optimiser.** As in the official code, one Adam steps the whole model
  once per minibatch, without a learning-rate schedule, down the clipped
  surrogate plus `vf_coef` times the value loss minus `ent_coef` times the
  entropy. Its gradient is clipped to global norm 10 over every parameter,
  actor and critic at once, with torch's formula. Adam acts on each
  coordinate on its own, so Ajax keeps one Adam state per network, both
  stepped together on that one gradient: the same optimiser, with a
  learning rate per network as torch's parameter groups allow.
- **Entropy bonus.** As in the official code, the loss subtracts 0.01 times
  the batch mean of the policy's entropy. A continuous APO's is its
  Gaussian's, `log_std + log sqrt(2 pi e)` summed over the action
  dimensions; the log standard deviation does not depend on the state, so
  the bonus pushes it up and leaves the mean alone.
- **Expert.** Cloning an expert into the actor before training is the
  `CloningPretrain` extension, and a cloning term during training the
  `ImitationLoss` extension; APO supports both
  ([Expert guidance](../guide/expert-guidance.md)). With `expert_policy` set,
  the rollout stores the expert's actions, from its state when it has one,
  and the evaluation also reports the expert's return.

## Results

No learning curves are published for APO yet. A conformance check
(`tests/conformance/test_apo_original.py`) makes one update of Ajax's APO
and of the official code's from the same state and rollout, on two gymnax
tasks, one of them with time limits, and compares both networks, the log
standard deviation, the optimiser, the reward rate and the value bias. The
probes (`tests/probing`) train it briefly on tiny environments with known
answers: the reward rate it learns, episode ends, extension phases, and a
run split by a checkpoint matching the run left whole.
[Performance](../guide/performance.md) compares its speed with the other
agents'.

## API

::: ajax.agents.APO.APO.APO
