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
| `config` | (required) | a dictionary stored with each run; `train` adds the agent's hyperparameters to it |
| `log_frequency` | 1000 | environment steps between evaluations |
| `use_wandb` | `True` | log to Weights & Biases |
| `use_tensorboard` | `False` | write TensorBoard event files |
| `project_name`, `group_name` | `None` | the W&B project and group |
| `run_name` | `None` | the W&B run name's prefix: each run is named `<run_name>_<run id>` |
| `folder` | `None` | where TensorBoard runs go: `<folder>/tensorboard/<run id>/` (the working directory when `None`) |
| `sweep` | `False` | for hyperparameter sweeps: SAC evaluates only in the last 20% of training, and the logging worker sends nothing to W&B |

With both backends off, `train` still evaluates and returns the evaluations
in memory: the quickest way to get learning curves.

## What is logged

Each evaluation records:

| Name | Meaning |
| --- | --- |
| `timestep` | environment steps so far |
| `Eval/episodic mean reward` | mean return of the `num_episode_test` test episodes, with the deterministic policy |
| `Eval/mean episodic length` | mean length of those episodes |
| `Eval/episodic entropy` | the policy's mean entropy along them |
| `Train/episodic mean reward` | mean return of the training episodes that ended recently |
| `Eval/mean average reward`, `Eval/mean bias` | the average-reward agents' (APO) long-run reward per step and bias; NaN for the others |
| `Eval/episodic mean expert reward`, `Eval/expert bias` | with an expert policy: the expert's return, and the agent's minus it; NaN without |
| `<group>/<metric>` | the agent's training metrics, such as `value/critic_loss` or `policy/policy_loss` |

Extensions add their own, such as `ConditioningMetrics`'s
`Cond/critic_srank`. The world models and UDRL log their own sets: see
their pages.

Returns are in the environment's units even when the agent normalises
rewards. A metric that is NaN (an agent that has not started updating yet,
an expert metric without an expert) is dropped from W&B and TensorBoard,
but kept as NaN in the returned evaluations.

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
```

A call that evaluates fewer times than its maximum pads the remaining
columns with -1 (`timestep`) and NaN (the rest).

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

    # TensorBoard writes spaces in names as underscores.
    scalars = load_scalars_from_tfevents(f"{folder}/tensorboard/{agent.run_ids[0]}")
    print(scalars["Eval/episodic_mean_reward"])  # [(step, value), ...]


if __name__ == "__main__":
    main()
```

Each TensorBoard run directory also holds a `config.json`: the
configuration, the agent's hyperparameters, the seed and the run id, so a
run can be reproduced and paired with others by seed.
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

The callback has one cost: JAX never stores a program that contains one in
its persistent compile cache, so a run with a `LoggingConfig` compiles again
in every new process. See [Performance](performance.md).
