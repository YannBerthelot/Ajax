# APO

Average Policy Optimization is PPO for the average-reward criterion. Instead of
discounting future rewards, it learns how much reward the policy earns per step
on average and judges each action by how much better than that average it
does. This suits continuing tasks, such as holding a setpoint, where there is
no natural horizon and a discount factor only shortens the agent's view. The
clipped update and the minibatch epochs are PPO's own code.

**Paper:** Ma et al., *Average-Reward Reinforcement Learning with Trust Region
Methods*, 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)). Ajax
follows the authors' official code, `xtma/apo`, where it is more specific
than the paper.

## Can I use?

| Feature | APO |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, a Gaussian around a tanh-bounded mean, clipped and mapped onto the environment's bounds |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probe and resume contract) |

## Example

```python
from ajax import APO
from ajax.logging.wandb_logging import LoggingConfig

agent = APO("Pendulum-v1", n_envs=8, n_steps=256, batch_size=64, normalize_observations=True)
logging = LoggingConfig(config={}, log_frequency=10_240, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=51_200, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
print(state.average_reward)  # the estimated reward per step, one per seed
```

This budget shows the calls; it is too short to learn the swing-up.

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 4 | environments stepped in parallel |
| `n_steps` | 2048 | steps per environment in each rollout |
| `batch_size` | 64 | sets the number of minibatches, as in [PPO](ppo.md) |
| `num_minibatches` | 0 | when positive, the number of minibatches |
| `n_epochs` | 10 | passes over the rollout per update, each reshuffled |
| `actor_learning_rate`, `critic_learning_rate` | 3e-4 | the step sizes of the one Adam over both networks (see the notes) |
| `adam_eps` | 1e-8 | Adam's epsilon |
| `vf_coef` | 1 | the value loss's weight in the summed loss |
| `alpha` | 0.1 | update rate of the running averages of the reward rate and of the values |
| `nu` | 0.1 | weight of the value-bias penalty |
| `gae_lambda` | 0.95 | GAE's lambda (there is no discount) |
| `clip_range` | 0.2 | the ratio clip; a float or a schedule of the timestep |
| `ent_coef` | 0.01 | entropy bonus weight, the official code's (see the notes) |
| `max_grad_norm` | 10 | clips the gradient's global norm over both networks at once (`None` turns it off) |
| `normalize_advantage` | True | advantages standardised over the whole rollout |
| `actor_architecture`, `critic_architecture` | `("64", "tanh", "64", "tanh")` | hidden layers, the official code's (see the notes) |
| `normalize_observations`, `normalize_rewards` | False | running normalisation |
| `reward_normalization_gamma` | None | with `normalize_rewards`, divide by the spread of the discounted return |
| `expert_policy` | None | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `actor_bias_init`, `critic_bias_init` | None | the heads' bias initialisers by name ([Networks](../guide/networks.md)) |
| `encoder_kernel_init` | `"torch_linear"` | the hidden layers start as torch's `nn.Linear`, as in the official code; any initialiser by name sets their kernels instead |
| `log_std_bounds` | None | the `(low, high)` clip of the continuous actor's log-std; `None`, the official code's, leaves it unclipped (see the notes) |
| `episode_length`, `action_repeat` | None, 1 | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `expose_recent_rollout` | False | keep the last rollout on the state, for measurement extensions |
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

## Notes

- **Episode ends.** As in the official code, the advantage chain stops at
  every episode end and the next value is dropped there, a time limit
  included (the reference's done flag does not tell them apart), and a
  time-limited step leaves the loss (`bootstrap_timelimit`): the losses
  average over the other steps.
- **Advantage normalisation** runs once over the whole rollout, not per
  minibatch as in Ajax's PPO, by the trained steps' mean and unbiased
  standard deviation (floored at `1e-6`), as the official code would; its
  config turns it off (the differences below).
- **Shuffling.** Every epoch draws a new split of the rollout into
  minibatches, as APO's official code does.
- **Hidden layers.** The actor and the critic each have two hidden layers
  of 64 `tanh` units, as the official code's mean and value networks:
  `xtma/apo`'s config sets only `normalize_observation`
  (`apo/experiments/configs/mujoco/apg/mujoco_appo.py:20`), so rlpyt's
  `MujocoFfModel` builds both with its defaults
  (`rlpyt/models/pg/mujoco_ff_model.py:21-22, :33-49`). They start as its
  torch `Linear` layers (`rlpyt/models/mlp.py:27-31`) do: kernel and bias
  uniform in `±1/sqrt(fan_in)` (`encoder_kernel_init="torch_linear"`;
  torch 1.5.1, which rlpyt's environment files pin,
  `torch/nn/modules/linear.py:79-84`). Not the official code's: the output
  layers start as Ajax's, the mean layer `lecun_uniform` (below) and the
  value layer `orthogonal(1.0)`, both with zero biases, where the official
  code leaves them at torch's default too. Until October 2026 APO's hidden
  layers were two of 128 units, starting orthogonal with zero biases; a
  checkpoint saved then restores into an APO whose `actor_architecture`
  and `critic_architecture` are both `("128", "tanh", "128", "tanh")`, and
  the default network refuses it with that advice.
- **Actor head.** A continuous APO's actor is the official code's:
  `xtma/apo` at `bfcd997` runs rlpyt's `MujocoFfAgent` (`exp.py:17, :38`)
  from the authors' fork, `xtma/rlpyt` at `e28bb20`. It is a Gaussian whose
  mean is `tanh` of the mean layer
  (`rlpyt/models/pg/mujoco_ff_model.py:23, :40-41`), with one learned
  log-std per action dimension, starting at 0 (std 1) and not clipped
  (`:24, :50, :78`; `rlpyt/agents/pg/gaussian.py:34-38` sets no bounds, no
  clip and no squashing). The sample itself is not squashed: the
  log-probability and the entropy bonus are the Gaussian's, of the raw
  sample (`rlpyt/distributions/gaussian.py:81-97, :108-134, :208`). The
  evaluation acts with the mean, as the official evaluation agent sets the
  std to 0 (`apo/agents/mujoco_eval.py:15`). `log_std_bounds` adds a clip,
  which the run config records. A PID policy's output takes the same
  `tanh`, and its log-std the same clip. A discrete APO ignores the head.
  Not the official code's: the mean layer starts `lecun_uniform` (torch's
  default initialisation there). Until October 2026 APO squashed its
  samples with `tanh` and clipped the log-std to `[-20, 2]`; the parameters
  are the same, so a checkpoint saved then restores, with the same
  deterministic action, `tanh` of the mean layer, and samples as the new
  head does.
- **Actions in the environment.** As for every agent, Ajax clips the
  sample to `[-1, 1]` and maps it onto the environment's bounds
  ([Environments](../guide/environments.md#actions)). The official code
  hands the raw sample to the environment (`rlpyt/envs/gym.py:66-67`);
  its MuJoCo tasks clamp it to their control range in the simulation, and
  Gymnasium's charge the control cost on the unclamped action
  (`gymnasium/envs/mujoco/hopper_v4.py:97, :138` in Gymnasium 1.2.0).
- **Optimiser.** As in the official code, one Adam steps the whole model
  (`apo/algos/apg/base.py:28`) at 3e-4, without a schedule
  (`apo/experiments/configs/mujoco/apg/mujoco_appo.py:5, :16`), down the
  clipped surrogate plus `vf_coef` (1, `appo.py:27`) times the value loss
  minus `ent_coef` times the entropy (`apo/algos/apg/appo.py:145-158`),
  once per minibatch. Its gradient is clipped to global norm 10 over every
  parameter, actor and critic at once (`appo.py:106-112`,
  `mujoco_appo.py:8`), with torch's formula (torch 1.5.1,
  `torch/nn/utils/clip_grad.py:33-37`). Adam acts on each coordinate on its
  own, so Ajax keeps one Adam state per network, both stepped together on
  that one gradient: the same optimiser, with a learning rate per network
  as torch's parameter groups allow. Until October 2026 each network took
  its own step, clipped to norm 0.5 on its own; a checkpoint saved then
  restores, its moments and step counts kept.
- **Entropy bonus.** As in the official code, the loss subtracts 0.01
  (`entropy_loss_coeff`, `mujoco_appo.py:9`) times the batch mean of the
  policy's entropy (`appo.py:155-158`). A continuous APO's is its
  Gaussian's, `log_std + log sqrt(2 pi e)` summed over the action
  dimensions (`rlpyt/distributions/gaussian.py:81-97, :102-103`); the
  log-std does not depend on the state, so the bonus pushes it up and
  leaves the mean alone. Until October 2026 the weight was 0; `ent_coef=0`
  reproduces runs from then (the parameters are unchanged, so their
  checkpoints restore).
- **Remaining differences from the official code** (`xtma/apo` at
  `bfcd997`, its config `apo/experiments/configs/mujoco/apg/mujoco_appo.py`,
  on `xtma/rlpyt` at `e28bb20`), each a default the constructor can set:
  - Advantage normalisation: off there (`normalize_advantage=False`,
    `mujoco_appo.py:15`; `apo/algos/apg/base.py:90-98` would standardise
    them), on here (`normalize_advantage=True`).
  - Rollout shape: 10 environments of 200 steps there (`batch_B=10`,
    `batch_T=200`, `mujoco_appo.py:27-28`), 2,000 samples in 20 minibatches
    of 100 (`minibatches=20`, `:12`); 4 environments of 2,048 steps here,
    8,192 samples in minibatches of 64. Both take 10 epochs (`:13`).
  - Value-bias coefficient: `rm_vbias_coeff=1.0` in the config `exp.py`
    runs (`mujoco_appo.py:7`), while the repository's sweep, `run_exp.py`,
    tries 0, 0.03, 0.1, 0.3 and 1.0 (`run_exp.py:55`) and `APPO`'s own
    default is 0.1 (`apo/algos/apg/appo.py:27`); `nu=0.1` here. The sweep
    also sets the discount to 1.0 (`run_exp.py:52`, the other values
    commented out), as Ajax's APO has none, where the config alone has
    0.99 (`mujoco_appo.py:10`).
  - Output layers: the mean layer starts `lecun_uniform` and the value
    layer `orthogonal(1.0)`, with zero biases, where the official code
    leaves both at torch's `nn.Linear` default (see Hidden layers).
  - Episode ends and actions: time-limited steps are trained on, and the
    sample is clipped to `[-1, 1]` before the environment (the notes
    above).
- `n_steps` and `batch_size` must divide one another unless
  `num_minibatches` is set; the constructor refuses them otherwise.
- **Expert.** Cloning an expert into the actor before training is the
  `CloningPretrain` extension, and a cloning term during training the
  `ImitationLoss` extension; APO supports both
  ([Expert guidance](../guide/expert-guidance.md)). With `expert_policy` set,
  the rollout stores the expert's actions, from its state when it has one,
  and the evaluation also reports the expert's return.
- APO is the on-policy counterpart of [ASAC](asac.md), the average-reward
  off-policy agent.

## Results

No learning curves are published for APO yet. Unit tests
(`tests/agents/APO` and the contracts every agent shares) check its pieces
against exact contracts, without learning, such as its differential
advantages. The probes (`tests/probing`) train it briefly on tiny
environments with known answers: the reward rate it learns, episode ends,
extension phases, and a run split by a checkpoint matching the run left
whole. On the CPU speed benchmark (`benchmarks/agent_bench.py`: Pendulum, 4
environments, 8,000 steps) APO ran at 16,390 steps per second of execution
(4,024 with compilation) at main `a8a13c2`
([PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.APO.APO.APO
