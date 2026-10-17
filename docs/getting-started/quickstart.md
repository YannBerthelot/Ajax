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
from ajax.logging.wandb_logging import LoggingConfig

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)  # 1,024 steps per rollout
# Evaluate every 4,096 steps; use_wandb=False keeps the results in memory only.
logging = LoggingConfig(config={}, log_frequency=4_096, use_wandb=False)
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
from ajax.logging.wandb_logging import LoggingConfig

# SAC waits learning_starts steps (10,000 by default) before its first update;
# this short demonstration starts after 1,000.
agent = SAC("Pendulum-v1", learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
```

Three thousand steps show the mechanics, not a trained pendulum: give SAC
tens of thousands of steps for that. The [algorithm pages](../algorithms/index.md)
help pick an agent and list each one's defaults.

## Use the trained policy

Every array in `state` has a seed axis first. Take one seed's actor and ask it
for actions:

```python
import gymnax
import jax

from ajax import PPO
from ajax.environments.interaction import get_pi

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=16_384)

actor = jax.tree.map(lambda x: x[0], state.actor_state)  # seed 0

env, params = gymnax.make("CartPole-v1")
key = jax.random.PRNGKey(0)
obs, env_state = env.reset(key, params)
episode_return, done = 0.0, False
while not done:
    pi, _ = get_pi(actor, actor.params, obs[None])  # a batch of one observation
    action = pi.mode()[0]  # the most likely action
    key, step_key = jax.random.split(key)
    obs, env_state, reward, terminated, truncated, _ = env.step(
        step_key, env_state, action, params
    )
    episode_return += float(reward)
    done = bool(terminated or truncated)
print(episode_return)
```

`get_pi` returns the policy's action distribution. It applies the agent's own
observation normalisation when the agent keeps one (a replay agent's
`normalize_observations=True`). If you trained an agent that normalises in
the environment (PPO, APO, PQN, UDRL), normalise the observation the
same way first: see [Environments](../guide/environments.md).

For continuous actions, take `pi.mean()` instead of `pi.mode()`. It is in the
agent's units, `[-1, 1]`: map it onto the environment's bounds before
stepping, as training and evaluation do, with
`agent_action_to_env(pi.mean(), env, params)[0]` from
`ajax.environments.utils` (Pendulum's torque spans `[-2, 2]`).

## Save and resume

`save_checkpoint` writes a state to disk. `restore_into` loads it into the
state of a fresh agent, which `train` then continues:

```python
import os
import tempfile

from ajax import PPO
from ajax.checkpoint import restore_into, save_checkpoint

seeds = [0, 1]
agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=seeds, n_timesteps=8_192)
path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
save_checkpoint(state, path)

# Later, in another process: the same agent, and a 0-step run as the skeleton.
agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
skeleton, _ = agent.train(seed=seeds, n_timesteps=0)
state = restore_into(skeleton, path)
state, _ = agent.train(seed=seeds, n_timesteps=8_192, initial_state=state)
print(state.collector_state.timestep)  # [18432 18432]: PPO adds a rollout per call
```

[Checkpoints and resuming](../guide/checkpoints.md) explains what a resume
keeps.

## Where next

- [Environments](../guide/environments.md): gymnax, Brax and MuJoCo
  Playground ids, your own environments, normalisation and time limits.
- [Networks and the PID policy](../guide/networks.md): layer sizes,
  activations, encoders, and a policy that is a PID controller.
- [Memory](../guide/memory.md): GRU, LSTM, transformer and Mamba blocks for
  partially observed tasks.
- [Extensions](../guide/extensions.md): learn from an expert, change the
  critic's target, measure the critic.
- [Logging](../guide/logging.md): Weights & Biases and TensorBoard.
