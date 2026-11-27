# Quickstart

This page trains a first agent, reads its results, acts with the trained
policy and saves it. It assumes Ajax is [installed](installation.md). Each
example runs on a laptop CPU in under a minute, compilation included, and
each builds on the ones before it.

The same steps are a notebook,
[`notebooks/quickstart.ipynb`](https://github.com/YannBerthelot/Ajax/blob/main/notebooks/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/YannBerthelot/Ajax/blob/main/notebooks/quickstart.ipynb)),
and complete scripts are in
[`examples/`](https://github.com/YannBerthelot/Ajax/tree/main/examples).

## Train an agent

An agent takes an environment id and its hyperparameters, and `train` runs
it. Here PPO learns CartPole with three seeds at once:

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960)
```

It prints a line when it starts, one per evaluation and one at the end:

```text
PPO, 3 run(s) of 40,960 steps: compiling, then a line per evaluation (pass logging_config=None for a silent run with none).
      4,096 / 40,960 steps | return 96.1 (min 86.1, max 113.0 over 3 seeds) | 3.0 s
      8,192 / 40,960 steps | return 361.9 (min 263.1, max 487.2 over 3 seeds) | 3.7 s, about 6 s left
     12,288 / 40,960 steps | return 366.1 (min 222.0, max 500.0 over 3 seeds) | 4.0 s, about 2 s left
...
     40,960 / 40,960 steps | return 437.0 (min 311.1, max 500.0 over 3 seeds) | 6.1 s
Done in 6.1 s. evaluations.returns has a row per seed and a column per evaluation: shape (3, 10).
```

What happened:

- `PPO("CartPole-v1", ...)` found the id in
  [gymnax](https://github.com/RobertTLange/gymnax), a library of
  environments written in JAX, and built 8 copies of the environment, which
  step in parallel within each seed.
- `train` compiled the training, environment included, and ran the three
  seeds side by side. The first line's time includes the compilation.
- Ten times over the run, it evaluated each seed's policy on 10 test
  episodes (`num_episode_test`) and printed their mean return, with the
  lowest and highest seed. From the second evaluation on, a line ends with
  the time left.

## Read the results

`evaluations` holds what was printed and every other logged value: a dict
of each logged name to an array with a row per seed and a column per
evaluation. `returns` and `timesteps` are the two most read:

```python
import matplotlib.pyplot as plt

steps = evaluations.timesteps  # [ 4096  8192 ... 40960]
returns = evaluations.returns  # shape (3, 10): seeds x evaluations

plt.plot(steps, returns.mean(axis=0))
plt.fill_between(steps, returns.min(axis=0), returns.max(axis=0), alpha=0.3)
plt.xlabel("environment steps")
plt.ylabel("evaluation return")
plt.savefig("cartpole.png")
```

A single seed has its row too: after `train(seed=0)`, the run's curve is
`evaluations.returns[0]`. To choose when the evaluations come, or to log to
TensorBoard or Weights & Biases, pass a `LoggingConfig`
([Logging](../guide/logging.md)). `logging_config=None` trains with no
evaluation and prints nothing, which is the fastest.
[Training an agent](../guide/training.md) has the details.

## Continuous actions

The same calls work for every agent. SAC on Pendulum:

```python
from ajax import SAC

# SAC waits learning_starts steps (10,000 by default) before its first update;
# this short demonstration starts after 1,000.
sac = SAC("Pendulum-v1", learning_starts=1_000)
sac_state, sac_evaluations = sac.train(seed=[1, 2], n_timesteps=3_000)
```

Three thousand steps show the mechanics, not a trained pendulum: give SAC
tens of thousands of steps for that. The [algorithm pages](../algorithms/index.md)
help choose an agent and list each one's defaults.

## Use the trained policy

`evaluate` plays fresh episodes with each seed's trained policy and returns
their returns, a row per seed:

```python
returns = agent.evaluate(state, episodes=20)  # shape (3, 20)
print(returns.mean(axis=1))  # each seed's score
```

`render` saves one episode of a seed's policy as an animation, and returns
its path. It needs the `render` extra (`pip install "ajax-rl[render]"`):

```python
print(agent.render(state, "cartpole.gif", max_steps=100))
```

Drawing takes about a tenth of a second a frame, so `max_steps=100` keeps
this to about ten seconds; without it, the episode plays to its end, 500
steps for a policy that balances the pole.
[Watch the policy](../guide/deploy.md#watch-the-policy) covers the other
environments.

To act in a loop of your own, `agent.predict(state, observation)` returns
the first seed's action for an observation (`seed_index` picks another), in
the environment's units. `agent.policy(state)` gives a policy that also
carries a memory from step to step. [Train, evaluate, deploy](../guide/deploy.md)
shows both.

## Save and load

`agent.save` writes the state with the agent's class and arguments, and
`ajax.load` builds both again:

```python
import os
import tempfile

import ajax

path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
agent.save(state, path)

# Later, in another process.
agent, state = ajax.load(path)
print(agent.evaluate(state, episodes=5).mean(axis=1))
state, _ = agent.train(seed=[1, 2, 3], n_timesteps=8_192, initial_state=state)
```

The last line trains the loaded run further.
[Checkpoints and resuming](../guide/checkpoints.md) explains what a resume
keeps, and which arguments must be passed again.

## Gymnasium environments

`"CartPole-v1"` above is gymnax's CartPole, not Gymnasium's. Ajax is meant
for environments written in JAX: gymnax, Brax, MuJoCo Playground,
Gymnasium's MuJoCo tasks in JAX (`"mujoco:<id>"`), or your own with gymnax's
interface. The environment then compiles into the same program as the
agent, and the whole run stays on one device, a CPU or a GPU: that is where
the speed comes from.

A [Gymnasium](https://gymnasium.farama.org) environment works with the same
calls, asked for with the `gymnasium:` prefix:

```python
gymnasium_agent = PPO("gymnasium:CartPole-v1", n_envs=8, n_steps=128)
gymnasium_state, _ = gymnasium_agent.train(seed=[1, 2], n_timesteps=2_048)
print(gymnasium_agent.evaluate(gymnasium_state, episodes=5).mean(axis=1))
```

It steps in Python, so training is slower, a GPU does not help, the seeds
train one after another, and some features refuse it. Use it when your task
has no JAX version: building one warns that it steps in Python and names the
closest JAX tasks. [Gymnasium environments](../guide/environments.md#gymnasium-environments)
says what changes, and Gymnasium's MuJoCo tasks need the `gymnasium` extra
([Installation](installation.md#extras)).

## Next

[Why Ajax](why-ajax.md) shows what else Ajax offers, and when another
library fits better. Coming from Stable-Baselines3,
[Coming from SB3](coming-from-sb3.md) puts its calls beside Ajax's. The user
guide starts with [Training an agent](../guide/training.md).
