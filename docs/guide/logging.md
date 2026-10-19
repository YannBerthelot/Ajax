# Logging

A `LoggingConfig` turns on evaluation and logging. Every `log_frequency`
steps the agent is evaluated and its metrics are recorded: in memory always,
and in Weights & Biases or TensorBoard when you ask for them.

## `LoggingConfig`

```python
from ajax.logging.wandb_logging import LoggingConfig

logging = LoggingConfig(
    config={"experiment": "baseline"},  # extra fields stored with the run
    log_frequency=5_000,  # evaluate and log every 5,000 environment steps
    use_wandb=False,  # True by default
    use_tensorboard=False,
)
```

| Field | Default | Meaning |
| --- | --- | --- |
| `config` | (required) | a dictionary stored with each run, with the agent's run configuration added (below) |
| `log_frequency` | 1000 | environment steps between evaluations |
| `use_wandb` | `True` | log to Weights & Biases |
| `use_tensorboard` | `False` | write TensorBoard event files |
| `project_name`, `group_name` | `None` | the W&B project and group |
| `run_name` | `None` | the W&B run name's prefix: each run is named `<run_name>_<run id>` |
| `folder` | `None` | where TensorBoard runs go: `<folder>/tensorboard/<run id>/` (the working directory when `None`) |
| `sweep` | `False` | for hyperparameter sweeps: SAC evaluates only in the last 20% of training, and the logging worker sends nothing to W&B |

With both backends off, `train` still evaluates and returns the evaluations
in memory: the quickest way to get learning curves.

### The run configuration

Each run stores `config` merged with the agent's `agent.config`: every
argument of the agent's constructor, by name, with the value it was given or
its default (`TDMPC2MultiTask` leaves out its dataset and evaluation
environments), and `algo_name`, the agent's name. `train` merges them into a
copy, so one `LoggingConfig` can serve several agents' runs. DreamerV3 and TD-MPC2 add the values they
resolve from those arguments (their discount, replay size and so on). W&B
stores the result as the run's configuration; TensorBoard stores it as text
and, in each run directory, as `config.json` with the `seed` and the
`run_id` added, so a run can be reproduced and paired with others by seed.
Values that are not JSON (an environment object, a function) are written as
their text.

## What is logged

Each evaluation records:

| Name | Meaning |
| --- | --- |
| `timestep` | environment steps so far |
| `Eval/episodic mean reward` | mean return of the `num_episode_test` test episodes, with the deterministic policy |
| `Eval/mean episodic length` | mean length of those episodes |
| `Eval/episodic entropy` | the policy's mean entropy along them |
| `Eval/mean bias` | the critic's error on those episodes ([below](#the-value-bias)) |
| `Eval/mean average reward` | APO's long-run reward per step; NaN for the others |
| `Eval/episodic mean expert reward`, `Eval/expert bias` | with an expert policy: the expert's return on the same episodes, and the agent's minus it; NaN without |
| `Train/episodic mean reward` | mean return of the training episodes that ended recently |
| `<group>/<metric>` | the agent's training metrics, such as `value/critic_loss` or `policy/policy_loss` |

Extensions add their own, such as `ConditioningMetrics`'s
`Cond/critic_srank`. UDRL logs `timestep`, its training metrics
(`Train/episodic mean reward`, `Train/actor_loss`, `Train/mean_rtg`,
`Train/mean_horizon`, `Train/n_completed_episodes`), its evaluation's return,
entropy and length, the command the evaluation starts from (below), and its
extensions' metrics. The world models log their
own sets: see their pages.

Each evaluation runs on episodes of its own: their initial states and noise
are drawn afresh from the evaluation count ([Training](training.md#evaluation)).
Returns are in the environment's units even when the agent normalises
rewards. A metric that is NaN (an agent that has not started updating yet,
an expert metric without an expert) is dropped from W&B and TensorBoard,
but kept as NaN in the returned evaluations.

### Training diagnostics

Beside its losses, each agent logs a few measures of how its updates go.
Like the other training metrics, they are the last update's values; PPO,
APO and PQN average theirs over the update's epochs and minibatches.

| Name | Agents | Meaning |
| --- | --- | --- |
| `policy/approx_kl` | PPO, APO | the mean of `(r - 1) - log r`, `r` the ratio of the new policy's probability of each action to the old one's: an estimate of how far an update moved the policy, as CleanRL and Stable-Baselines3 log it |
| `policy/policy_std` | PPO, APO | the mean standard deviation of the policy's Gaussian, before any squashing (SAC logs the same); NaN for a discrete policy |
| `value/explained_variance` | PPO, APO | `1 - Var[target - V] / Var[target]` over the minibatch: 1 for a critic that predicts its targets, 0 for one no better than their mean; NaN when the targets do not vary |
| `policy/grad_norm`, `value/grad_norm` | PPO, APO | the global norm of each network's gradient, before clipping. PPO clips the two networks' gradients together (`max_grad_norm`): the clip acts when the norm of both, `sqrt(policy² + value²)`, exceeds it |
| `value/value_bias`, `value/average_reward` | APO | the value bias `b` and the reward rate `rho` the update used |
| `policy/epsilon` | DQN, PQN | the exploration rate of the epsilon-greedy schedule at the current timestep |
| `value/abs_td_error` | DQN, PQN | the mean of `abs(Q(s, a) - target)` over the batch |
| `value/grad_norm` | DQN, PQN | the global norm of the Q-network's gradient, before clipping |
| `policy/grad_norm`, `value/grad_norm` | SAC, TD3 | the global norm of the actor's and the critics' gradients, before clipping. TD3 logs 0 for the actor on the updates its `policy_delay` skips |
| `value/q_gap` | TD3 | the mean over the batch of the critics' spread, the largest value minus the smallest (`abs(Q1(s, a) - Q2(s, a))` for the default twin critics): how far they disagree |
| `value/target_noise_clip_frac` | TD3 | the share of the target policy's noise samples that the clip to `±target_noise_clip` changed |
| `Train/grad_norm` | APG | the global norm of the controller's gradient, before clipping (`max_grad_norm`) |
| `Eval/commanded return`, `Eval/commanded horizon` | UDRL | the return and horizon each evaluation episode is commanded with at its start, to set beside the return and length it achieves; in normalised units when rewards are normalised |

### Extension metrics

The expert-guided extensions log their schedules and how often the expert
acts, and the pretraining extensions what their fit computed, read from the
agent state at each evaluation:

| Name | Extension | Meaning |
| --- | --- | --- |
| `edge/live_expert_frac` | `EDGEExploration` | the share of envs the expert drove on the last collection step |
| `edge/decay`, `edge/beta` | `EDGEExploration` | the gate's schedule: the decay (`"fixed"`, `"argmax"`, `"boltzmann"`) or the LCB pessimism (`"lcb"`, `"argmax_lcb"`); the `"thompson"` gate has none |
| `jsrl/horizon`, `jsrl/expert_frac` | `JSRLCurriculum` | the guide horizon `H_t`, and the share of envs the expert drove on the last collection step |
| `lcb_bootstrap/beta` | `LCBGatedBootstrap` | the LCB pessimism `β` |
| `critic_blend/expert_weight` | `CriticBlend` | the weight `α` of the expert's target in the blend |
| `value_box/threshold` | `ValueBox` | the expert value above which the expert acts |
| `value_box/in_box_frac`, `value_box/expert_frac` | `ValueBox` | the shares of envs inside the box and driven by the expert on the last collection step |
| `mc_pretrain/v_min`, `mc_pretrain/v_max` | `MCPretrain` | the expert's value range, which `ValueBox`'s threshold runs between |
| `mc_pretrain/first_loss`, `mc_pretrain/last_loss` | `MCPretrain` | the squared error of `φ*`'s fit to the Monte-Carlo returns, at its first and its last step |
| `mc_pretrain/q_expert_mean` | `MCPretrain` | the mean of `φ*`'s value at the expert's actions after the fit, over the rollout's states |
| `cloning/first_loss`, `cloning/last_loss` | `CloningPretrain` | the mean cloning loss of its first and its last epoch (NaN with `skip_actor`) |
| `bellman_pretrain/first_loss`, `bellman_pretrain/last_loss` | `BellmanPretrain` | the critic loss of its first and its last update |

A pretraining result is the same at every evaluation. It is NaN in a run
resumed from a checkpoint saved before the results were kept.

### The value bias

`Eval/mean bias` measures how far the critic's value estimates are from
the returns the policy actually obtains. At each step of each evaluation
episode it takes the critic's estimate at the observation and the action
taken, subtracts the discounted return realised from that step to the
episode's end, and averages over the steps. Positive means the critic
overestimates. It is computed in the critic's own units: rewards normalised
as in training and multiplied by the agent's `reward_scale`. An episode cut
by the time limit realises no return past it, so even an exact critic shows
a positive bias over its last steps.

SAC, TD3, REDQ and AVG report it with their action-value critic, DQN and
PQN with their Q-network, PPO with its state-value critic. ASAC and APO,
which learn average-reward values rather than discounted ones, report NaN,
as does a critic with memory.

## In memory

`train` returns the evaluations as a dictionary of arrays, one row per seed
and one column per evaluation:

```python
from ajax import PPO
from ajax.logging.wandb_logging import LoggingConfig

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
logging = LoggingConfig(config={}, log_frequency=4_096, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=16_384, logging_config=logging)

print(sorted(evaluations))  # every logged name
print(evaluations["timestep"][0])  # [ 4096  8192 12288 16384]
print(evaluations["value/critic_loss"].shape)  # (2, 4)
print(agent.config["algo_name"], agent.config["n_steps"])  # PPO 128
```

A call that evaluates fewer times than its maximum pads the remaining
columns with -1 (`timestep`) and NaN (the rest). Pass an `n_timesteps` that
is a whole number of the agent's iterations (`n_envs` steps for the replay
agents, `n_envs × n_steps` for PPO, APO, PQN and UDRL, `n_envs × horizon`
for APG): the iterations stop at a multiple of one, so otherwise an
evaluation due at the end of the run does not happen, and `train` warns.

## Weights & Biases and TensorBoard

Set `use_wandb=True` (log in once with `wandb login`) or
`use_tensorboard=True`. Each seed is its own run, with its own id; after
`train`, `agent.run_ids` lists them, and `train(..., on_ids_ready=callback)`
hands them to `callback` before training starts.

!!! warning "Guard your script's entry point"
    The logging worker is a separate Python process started with `spawn`,
    which imports your script again. A script that trains at the top level
    then tries to train a second time inside the worker, and the run hangs.
    Put the training under `if __name__ == "__main__":`, as below.

```python
import json
import tempfile

from ajax import PPO
from ajax.logging.wandb_logging import LoggingConfig, load_scalars_from_tfevents


def main():
    folder = tempfile.mkdtemp()
    logging = LoggingConfig(
        config={"experiment": "cartpole-demo"},
        log_frequency=4_096,
        use_wandb=False,
        use_tensorboard=True,
        folder=folder,
    )
    agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
    agent.train(seed=[0, 1], n_timesteps=8_192, logging_config=logging)

    run = f"{folder}/tensorboard/{agent.run_ids[0]}"
    # TensorBoard writes spaces in names as underscores.
    scalars = load_scalars_from_tfevents(run)
    print(scalars["Eval/episodic_mean_reward"])  # [(step, value), ...]
    with open(f"{run}/config.json") as f:
        print(json.load(f)["seed"])  # 0


if __name__ == "__main__":
    main()
```

`load_scalars_from_tfevents` reads the scalars back without TensorFlow.

## The logging worker

Training runs inside one compiled program, which must not wait on a network
or a disk. At each evaluation, the program hands the metrics to Python
through a `jax.debug.callback`, which puts them on a queue; a separate worker
process takes them off and writes them. The worker:

- runs on the CPU, so it never reserves GPU memory;
- writes TensorBoard events as they arrive;
- buffers W&B steps and sends them ten at a time, in one W&B session per
  batch, because opening a W&B run is slow and W&B allows only one active
  run per process. W&B charts therefore update in steps of ten evaluations;
- is drained and stopped when `train` returns, so every metric is written
  before `train` hands back the state.

The worker starts only when a backend is on. The callback has one cost:
JAX never stores a program that contains one in its persistent compile
cache, so a run with a `LoggingConfig` compiles again in every new process.
See [Performance](performance.md).
