# Quickstart

This page trains an agent, reads its results, runs the trained policy and
saves it. Each example runs on a laptop CPU in under a minute, compilation
included.

## Train an agent

An agent takes an environment id and its hyperparameters; `train` runs it.
Here PPO learns CartPole with three seeds at once:

```python
import matplotlib.pyplot as plt

from ajax import PPO
from ajax.logging import LoggingConfig

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)  # 1,024 steps per rollout
# Evaluate every 4,096 steps; backends=() keeps the results in memory only.
logging = LoggingConfig(log_frequency=4_096, backends=())
state, evaluations = agent.train(
    seed=[1, 2, 3], n_timesteps=32_768, logging_config=logging
)

steps = evaluations["timestep"][0]  # [ 4096  8192 ... 32768]
returns = evaluations["Eval/episodic mean reward"]  # shape (3, 8): seeds x evaluations
print(returns.mean(axis=0))

plt.plot(steps, returns.mean(axis=0))
plt.fill_between(steps, returns.min(axis=0), returns.max(axis=0), alpha=0.3)
plt.xlabel("environment steps")
plt.ylabel("evaluation return")
plt.savefig("cartpole.png")
```

What happened:

- `PPO("CartPole-v1", ...)` looked the id up in gymnax and built 8 copies of
  the environment, which step in parallel inside each seed.
- `train` compiled the whole run, environment included, into one program and
  ran the three seeds side by side with `jax.vmap`.
- Every 4,096 steps it evaluated the policy on 10 test episodes
  (`num_episode_test`) and stored the mean return.

`evaluations` maps every logged name to an array with one row per seed and
one column per evaluation. Without a `logging_config`, `train` does not
evaluate at all and returns `(state, None)`.
[Training an agent](../guide/training.md) has the details.

## Continuous actions

The same pattern works for every agent. SAC on Pendulum:

```python
from ajax import SAC
from ajax.logging import LoggingConfig

# SAC waits learning_starts steps (10,000 by default) before its first update;
# this short demonstration starts after 1,000.
agent = SAC("Pendulum-v1", learning_starts=1_000)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=[1, 2], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
```

Three thousand steps show the mechanics, not a trained pendulum: give SAC
tens of thousands of steps for that. The [algorithm pages](../algorithms/index.md)
help pick an agent and list each one's defaults.

## Use the trained policy

`predict` returns the action of one seed's policy for an observation, in the
environment's units. It acts as the evaluation acts: it normalises the
observation as the agent does and maps the action onto the environment's
bounds.

```python
import gymnax
import jax

from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=16_384)

env, params = gymnax.make("CartPole-v1")
key = jax.random.PRNGKey(0)
obs, env_state = env.reset(key, params)
episode_return, done = 0.0, False
while not done:
    action = agent.predict(state, obs)  # seed 0's policy
    key, step_key = jax.random.split(key)
    obs, env_state, reward, terminated, truncated, _ = env.step(
        step_key, env_state, action, params
    )
    episode_return += float(reward)
    done = bool(terminated or truncated)
print(episode_return)
```

An agent with memory, a recurrent actor or a world model, acts through
`agent.policy(state)`, which carries the memory from step to step.
[Train, evaluate, deploy](../guide/deploy.md) shows both.

## Save and resume

`save_checkpoint` writes a state to disk. `load_checkpoint` reads it back
into an agent built with the same arguments, which `train` then continues:

```python
import os
import tempfile

from ajax import PPO
from ajax.checkpoint import load_checkpoint, save_checkpoint

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192)
path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
save_checkpoint(state, path)

# Later, in another process: the same agent.
agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state = load_checkpoint(agent, path)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, initial_state=state)
print(state.collector_state.timestep)  # [18432 18432]: PPO adds a rollout per call
```

[Checkpoints and resuming](../guide/checkpoints.md) explains what a resume
keeps.

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
