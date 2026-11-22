# Coming from Stable-Baselines3

This page puts the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
(SB3) calls you know beside Ajax's. The SB3 blocks are for comparison; the
Ajax blocks run as written.

## The main difference

In SB3 a model holds its networks, and `learn` changes it in place. In Ajax
an agent holds only its settings. `train` returns the trained **state** (the
networks, the optimisers, the replay), and everything that uses a trained
agent takes that state. One `train` call trains several seeds, so a state
holds one copy per seed.

## Train

```python
# doc: skip (Stable-Baselines3, for comparison)
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1", seed=0)
model.learn(total_timesteps=4_096)
```

```python
from ajax import PPO

agent = PPO("CartPole-v1")
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_096)
print(evaluations.returns)  # one row per seed, one column per evaluation
```

`train` evaluates the policy about ten times over the run and prints a line
for each, with no callback to set up; `logging_config=None` turns both off.

There is no policy name to give: the networks follow the environment's
observations (an MLP, or a CNN encoder for images, see
[Networks](../guide/networks.md)).

## The calls side by side

| In SB3 | In Ajax | Page |
| --- | --- | --- |
| `PPO("MlpPolicy", env)` | `PPO(env_id)`, or a prebuilt JAX environment | [Training an agent](../guide/training.md) |
| `model.learn(total_timesteps=N)` | `state, evaluations = agent.train(seed=[...], n_timesteps=N)` | [Training an agent](../guide/training.md#train) |
| one model per seed | `seed=[0, 1, 2]`: every seed in one compiled program | [Performance](../guide/performance.md) |
| `make_vec_env(env_id, n_envs=8)` | `PPO(env_id, n_envs=8)` | [Training an agent](../guide/training.md#parallel-environments) |
| `VecNormalize(venv)` | `normalize_observations=True`, `normalize_rewards=True` | [Environments](../guide/environments.md#normalisation) |
| `policy_kwargs=dict(net_arch=[64, 64])` | `actor_architecture=("64", "tanh", "64", "tanh")`, and `critic_architecture` | [Networks](../guide/networks.md) |
| `learning_rate=3e-4` | `actor_learning_rate=3e-4, critic_learning_rate=3e-4`: one per network | [Algorithms](../algorithms/index.md) |
| `EvalCallback(eval_env, eval_freq=...)` | nothing: `train` evaluates and returns the evaluations; `LoggingConfig(log_frequency=...)` sets how often | [Logging](../guide/logging.md) |
| `evaluate_policy(model, env, n_eval_episodes=10)` | `agent.evaluate(state, episodes=10)`, one row of returns per seed | [Quickstart](quickstart.md#use-the-trained-policy) |
| `tensorboard_log="runs"` | `LoggingConfig(backends=(TensorBoardBackend("runs"),))` | [Logging](../guide/logging.md#backends) |
| `model.predict(obs, deterministic=True)` | `agent.predict(state, obs)`, or `agent.policy(state)` for a policy with memory | [Train, evaluate, deploy](../guide/deploy.md) |
| `model.save(path)`, `PPO.load(path)` | `agent.save(state, path)`, `agent, state = PPO.load(path)` | [Checkpoints](../guide/checkpoints.md#save-and-load-an-agent) |
| `model.learn(..., reset_num_timesteps=False)` | `agent.train(initial_state=state, ...)` | [Training an agent](../guide/training.md#continuing-a-run) |
| a Gymnasium environment | `"gymnasium:<id>"`, stepped in Python and slower | [Environments](../guide/environments.md#gymnasium-environments) |
| `RecurrentPPO` (sb3-contrib) | `PPO(env_id, memory={"kind": "lstm", "hidden_size": 64})` | [Memory](../guide/memory.md) |

## Act with the trained agent

```python
# doc: skip (Stable-Baselines3, for comparison)
action, _ = model.predict(obs, deterministic=True)
```

```python
import gymnax
import jax

env, params = gymnax.make("CartPole-v1")
obs, _ = env.reset(jax.random.PRNGKey(0), params)

action = agent.predict(state, obs)  # the first seed's policy
print(action)
```

`predict` acts as the evaluation does: the most likely action of a discrete
policy, the mean action of a continuous one. An agent whose policy carries
something between steps (a memory, a world model) refuses `predict`;
`agent.policy(state)` gives a policy with `reset` and `act` instead
([Train, evaluate, deploy](../guide/deploy.md)).

## Save and load

`save` takes the state, and `load` returns the agent with it:

```python
# doc: skip (Stable-Baselines3, for comparison)
model.save("ppo_cartpole")
model = PPO.load("ppo_cartpole")
```

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "ppo_cartpole.pkl")
agent.save(state, path)

# Later, in another process.
agent, state = PPO.load(path)
print(agent.predict(state, obs))
```

The file records the agent's plain arguments. Objects (a prebuilt
environment, extensions) are passed to `load` again
([Checkpoints](../guide/checkpoints.md#save-and-load-an-agent)).

## What Ajax does not have

- **Callbacks.** A training run is one compiled program, so no Python runs
  between its steps. What a callback would do goes elsewhere: evaluations
  come with `train`, changes to the training loop are
  [extensions](../guide/extensions.md), and a run can be cut into several
  `train` calls with your own code between them
  ([Continuing a run](../guide/training.md#continuing-a-run)).
- **Fast training on a Python environment.** The speed comes from the
  environment compiling with the agent. A Gymnasium environment works, at
  the speed of Python, and some features refuse it
  ([Gymnasium environments](../guide/environments.md#gymnasium-environments)).
- **A2C, DDPG and HER.** [Algorithms](../algorithms/index.md) lists what
  Ajax has.

[Why Ajax](why-ajax.md#when-to-use-something-else) says when SB3 is the
better choice.
