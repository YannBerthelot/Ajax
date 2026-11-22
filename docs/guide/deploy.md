# Train, evaluate, deploy

This page takes an agent from training to acting in a loop of your own:
train and read the evaluations, act with the trained policy, and save it
for later.

## Train and evaluate

`train` evaluates the policy every `log_frequency` steps on
`num_episode_test` episodes and returns the results, one row per seed:

```python
from ajax import SAC
from ajax.logging import LoggingConfig

agent = SAC("Pendulum-v1", learning_starts=1_000)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=3_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # seeds x evaluations
```

[Training an agent](training.md) and [Logging](logging.md) have the details.

## Act with the trained policy

`agent.policy(state, seed_index)` returns the policy of one seed. It acts on
the environment's own observations and returns actions in the environment's
units, exactly as the evaluation acts. It normalises the observation as the
agent does, appends the observation features, applies a residual policy or
gain mode and maps the action onto the environment's bounds. You do not
repeat any of that yourself.

A policy keeps what it needs from one step to the next in a carry: the
memory of a recurrent actor or a world model, a stateful expert's state,
UDRL's command. `reset(n_envs)` starts one, and `act` returns the next:

```python
import gymnax
import jax
import jax.numpy as jnp

env, params = gymnax.make("Pendulum-v1")
n_envs = 4
reset = jax.vmap(env.reset, in_axes=(0, None))
step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

policy = agent.policy(state, seed_index=0)
carry = policy.reset(n_envs)
key = jax.random.PRNGKey(0)
obs, env_state = reset(jax.random.split(key, n_envs), params)
done = reward = None
returns = jnp.zeros(n_envs)
for _ in range(200):  # one Pendulum episode
    action, carry = policy.act(obs, carry, done, reward)
    key, step_key = jax.random.split(key)
    keys = jax.random.split(step_key, n_envs)
    obs, env_state, reward, terminated, truncated, _ = step(keys, env_state, action, params)
    done = jnp.logical_or(terminated, truncated)
    returns = returns + reward
print(returns)  # one return per environment
```

`act(obs, carry, done, reward, key)` takes:

- `obs`: a batch of observations, one row per environment;
- `done`: True where `obs` starts a new episode, the last action having
  ended one (gymnax resets the environment itself). The memory, the
  expert's state, the features and the command restart there. `None` means
  no episode ended;
- `reward`: what the last action earned. Only UDRL reads it, to count its
  command down; it asks for it with `done`;
- `key`: the key of the policy's own draws (a world model samples its
  action). Without one, the policy draws its keys from its carry.

`act` compiles the first time it runs and not again.

## One action at a time

For an agent without memory, `predict` returns one action, without a carry.
It acts as the policy acts at the start of an episode:

```python
action = agent.predict(state, obs[0])  # one observation; a batch works too
print(action)
```

It acts for the first seed; `seed_index` picks another. With it, an episode
in one environment is a plain loop:

```python
single_reset_key, single_key = jax.random.split(jax.random.PRNGKey(1))
single_obs, single_state = env.reset(single_reset_key, params)
episode_return, episode_over = 0.0, False
while not episode_over:
    single_action = agent.predict(state, single_obs)
    single_key, step_key = jax.random.split(single_key)
    single_obs, single_state, single_reward, terminated, truncated, _ = env.step(
        step_key, single_state, single_action, params
    )
    episode_return += float(single_reward)
    episode_over = bool(terminated or truncated)
print(episode_return)
```

Each step draws on a fresh key, split from the last: JAX has no hidden
random state.

An agent whose policy carries memory refuses `predict`, since one call cannot
carry it between steps: a recurrent actor, a world model (DreamerV3,
TD-MPC2), UDRL, an expert or a feature with a state. Use `policy` instead.
For one environment, it is the same loop with a carry and a batch of one:
`obs[None]` adds the batch axis and `actions[0]` removes it.

```python
from ajax import PPO
from ajax.networks.memory import MemoryConfig

memory_agent = PPO(
    "CartPole-v1",
    n_envs=4,
    n_steps=32,
    memory=MemoryConfig(kind="gru", hidden_size=16),
    bptt_length=16,
)
memory_state, _ = memory_agent.train(seed=0, n_timesteps=512, logging_config=None)

cartpole, cartpole_params = gymnax.make("CartPole-v1")
memory_policy = memory_agent.policy(memory_state)
memory_carry = memory_policy.reset(1)  # one environment
reset_key, loop_key = jax.random.split(jax.random.PRNGKey(2))
one_obs, one_state = cartpole.reset(reset_key, cartpole_params)
episode_return, episode_over = 0.0, False
while not episode_over:
    actions, memory_carry = memory_policy.act(one_obs[None], memory_carry)
    loop_key, step_key = jax.random.split(loop_key)
    one_obs, one_state, one_reward, terminated, truncated, _ = cartpole.step(
        step_key, one_state, actions[0], cartpole_params
    )
    episode_return += float(one_reward)
    episode_over = bool(terminated or truncated)
print(episode_return)
```

TDMPC2MultiTask has no policy yet: its planner acts for one task of its set.

## Save and load

`agent.save(state, path)` writes the state with the agent's class and
arguments, and `agent, state = ajax.load(path)` builds both again: the
shortest way, shown in [Checkpoints](checkpoints.md#save-and-load-an-agent).
The pair below stores the state alone: `save_checkpoint` writes it, and
`load_checkpoint` reads it back into an agent you build with the same
arguments, in this process or another:

```python
import os
import tempfile

from ajax.checkpoint import load_checkpoint, save_checkpoint

path = os.path.join(tempfile.mkdtemp(), "sac_pendulum.pkl")
save_checkpoint(state, path)

# Later, in another process: the same agent, then its policy.
agent = SAC("Pendulum-v1", learning_starts=1_000)
state = load_checkpoint(agent, path)
policy = agent.policy(state)
```

The loaded state also resumes training with `train(initial_state=state)`.
A run whose `train` call took options that shape its update (APG's
learning-rate schedule, a `total_timesteps`) resumes through `restore_into`
instead, on a skeleton built with those options:
[Checkpoints and resuming](checkpoints.md) explains it.
