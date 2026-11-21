# Quickstart

This page trains an agent, reads its results, runs the trained policy and
saves it. Each example runs on a laptop CPU in under a minute, compilation
included.

The same steps are a notebook,
[`notebooks/quickstart.ipynb`](https://github.com/YannBerthelot/Ajax/blob/main/notebooks/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/YannBerthelot/Ajax/blob/main/notebooks/quickstart.ipynb)),
and complete scripts are in
[`examples/`](https://github.com/YannBerthelot/Ajax/tree/main/examples).

## Train an agent

An agent takes an environment id and its hyperparameters; `train` runs it.
Here PPO learns CartPole with three seeds at once:

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960)
```

It prints a line when it starts, one per evaluation and one at the end:

```text
PPO, 3 run(s) of 40,960 steps: compiling, then a line per evaluation (pass logging_config=None for a silent run with none).
      4,096 / 40,960 steps | return 171.1 (min 69.6, max 280.0 over 3 seeds) | 2.5 s
      8,192 / 40,960 steps | return 259.1 (min 156.9, max 450.1 over 3 seeds) | 4.2 s
...
     40,960 / 40,960 steps | return 500.0 (min 500.0, max 500.0 over 3 seeds) | 5.5 s
Done in 5.5 s.
```

What happened:

- `PPO("CartPole-v1", ...)` looked the id up in gymnax and built 8 copies of
  the environment, which step in parallel inside each seed.
- `train` compiled the training, environment included, and ran the three
  seeds side by side. The first line's time includes the compilation.
- Ten times over the run it evaluated each seed's policy on 10 test
  episodes (`num_episode_test`) and printed their mean return, with the
  lowest and highest seed.

`evaluations` holds what was printed, and every other logged value:

```python
import matplotlib.pyplot as plt

from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960)

steps = evaluations.timesteps  # [ 4096  8192 ... 40960]
returns = evaluations.returns  # shape (3, 10): seeds x evaluations

plt.plot(steps, returns.mean(axis=0))
plt.fill_between(steps, returns.min(axis=0), returns.max(axis=0), alpha=0.3)
plt.xlabel("environment steps")
plt.ylabel("evaluation return")
plt.savefig("cartpole.png")
```

It is a dict of every logged name to an array with one row per seed and one
column per evaluation; `returns` and `timesteps` are its two most read
entries. To choose when the evaluations come, or to log to TensorBoard or
Weights & Biases, pass a `LoggingConfig` ([Logging](../guide/logging.md));
`logging_config=None` trains with no evaluation and prints nothing, which is
the fastest. [Training an agent](../guide/training.md) has the details.

## Continuous actions

The same pattern works for every agent. SAC on Pendulum:

```python
from ajax import SAC

# SAC waits learning_starts steps (10,000 by default) before its first update;
# this short demonstration starts after 1,000.
agent = SAC("Pendulum-v1", learning_starts=1_000)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=3_000)
```

Three thousand steps show the mechanics, not a trained pendulum: give SAC
tens of thousands of steps for that. The [algorithm pages](../algorithms/index.md)
help pick an agent and list each one's defaults.

## Use the trained policy

`episode_returns` runs fresh episodes of each seed's trained policy and
returns their returns, one row per seed:

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=16_384, logging_config=None)

returns = agent.episode_returns(state, episodes=20)  # shape (2, 20)
print(returns.mean(axis=1))  # each seed's score
```

To act in a loop of your own, `agent.predict(state, observation)` returns
one seed's action for an observation, in the environment's units, and
`agent.policy(state)` carries a memory from step to step.
[Train, evaluate, deploy](../guide/deploy.md) shows both.

## Save and load

`agent.save` writes the state with the agent's class and arguments;
`ajax.load` builds both again:

```python
import os
import tempfile

import ajax
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, logging_config=None)
path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
agent.save(state, path)

# Later, in another process.
agent, state = ajax.load(path)
print(agent.episode_returns(state, episodes=5).mean(axis=1))
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, initial_state=state)
```

[Checkpoints and resuming](../guide/checkpoints.md) explains what a resume
keeps, and which arguments must be passed again.

## Where next

- [Train, evaluate, deploy](../guide/deploy.md): act with a trained policy in
  a loop of your own, save it and load it elsewhere.
- [Environments](../guide/environments.md): gymnax, Brax and MuJoCo
  Playground ids, your own environments, normalisation and time limits.
- [Networks and the PID policy](../guide/networks.md): layer sizes,
  activations, encoders, and a policy that is a PID controller.
- [Memory](../guide/memory.md): GRU, LSTM, transformer and Mamba blocks for
  partially observed tasks.
- [Extensions](../guide/extensions.md): learn from an expert, change the
  critic's target, measure the critic.
- [Logging](../guide/logging.md): Weights & Biases and TensorBoard.
