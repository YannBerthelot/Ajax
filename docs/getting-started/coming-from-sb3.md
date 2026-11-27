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

There is no policy name to give: an MLP reads the observations. For images,
pass `cnn_image_shape` to read them with a CNN encoder
([Networks](../guide/networks.md#image-observations)); Ajax warns when
images reach an MLP.

## The defaults are the original's

Ajax implements each algorithm as its paper or its authors' code does, and
its defaults are theirs; each [algorithm page](../algorithms/index.md) cites
them. SB3 sets some of its own. So the same call does not train the same
thing in both libraries, and settings brought from SB3 need the lines below.
The SB3 values are those of its 2.9.0 constructors.

| Agent | Ajax, as the original | SB3 |
| --- | --- | --- |
| PPO | value loss clipped around the rollout's value (ppo2) | not clipped |
| PPO | observations and rewards normalised (ppo2's MuJoCo run) | as the environment gives them, unless wrapped in `VecNormalize` |
| PPO | `vf_coef` weighs half the squared error | weighs the whole squared error: SB3's 0.5 is 1.0 here |
| PPO | `batch_size=64` gives 32 minibatches, whatever `n_envs` is | `batch_size` is a minibatch's size |
| SAC | `learning_starts=10_000` random steps before the first update | 100 |
| TD3 | `learning_starts=25_000`; learning rates `3e-4`; exploration noise 0.1 | 100; `1e-3`; no noise unless given |
| DQN | CleanRL's vector-observation settings: Adam at `2.5e-4`, the squared loss, a replay of 10,000, batches of 128, an update every 10 steps, the target copied every 50 updates, `learning_starts=10_000`, epsilon decaying over half the run | Adam at `1e-4`, the Huber loss, a replay of 1,000,000, batches of 32, an update every 4 steps, the target copied every 10,000 steps, `learning_starts=100`, epsilon decaying over a tenth of the run |

A short run shows the warm-up most: SAC on 20,000 steps acts at random for
the first half unless `learning_starts` is lowered.

For PPO, `PPO.preset("SB3", env_id)` sets SB3's defaults in one call, and
[its page](../algorithms/ppo.md#presets) says what stays yours to set (the
number of minibatches with several environments) and what no option
matches:

```python
from ajax import PPO

ported = PPO.preset("SB3", "CartPole-v1", n_envs=8, n_steps=256, num_minibatches=32)
print(ported.clip_value_loss, ported.normalize_observations)  # False False
```

The other agents have no such preset: pass the values of the SB3 column.
For DQN, SB3's Huber loss is `td_loss_fn=compute_huber_1_td_loss`
([its notes](../algorithms/dqn.md#notes)). An argument SB3 calls otherwise is refused with the
closest Ajax names, or, for SB3's own (`verbose`, `net_arch`,
`tensorboard_log`, ...), with where Ajax takes it; `total_timesteps` exists in both and means another
thing here: `train(n_timesteps=...)` is SB3's `learn(total_timesteps=...)`.

## The calls side by side

| In SB3 | In Ajax | Page |
| --- | --- | --- |
| `PPO("MlpPolicy", env)` | `PPO(env_id)`, or a prebuilt JAX environment | [Training an agent](../guide/training.md) |
| `model.learn(total_timesteps=N)` | `state, evaluations = agent.train(seed=[...], n_timesteps=N)` | [Training an agent](../guide/training.md#train) |
| one model per seed | `seed=[0, 1, 2]`: every seed in one compiled program | [Performance](../guide/performance.md) |
| `make_vec_env(env_id, n_envs=8)` | `PPO(env_id, n_envs=8)` | [Training an agent](../guide/training.md#parallel-environments) |
| `VecNormalize(venv)` | `normalize_observations=True`, `normalize_rewards=True` (PPO's defaults; off by default elsewhere) | [Environments](../guide/environments.md#normalisation) |
| `policy_kwargs=dict(net_arch=[64, 64])` | `actor_architecture=("64", "tanh", "64", "tanh")`, and `critic_architecture`: each width is followed by its activation | [Networks](../guide/networks.md) |
| `learning_rate=3e-4` | `actor_learning_rate=3e-4, critic_learning_rate=3e-4`: one per network, and `alpha_learning_rate` for SAC's temperature | [Algorithms](../algorithms/index.md) |
| `EvalCallback(eval_env, eval_freq=...)` | nothing: `train` evaluates and returns the evaluations; `LoggingConfig(log_frequency=...)` sets how often | [Logging](../guide/logging.md) |
| `evaluate_policy(model, env, n_eval_episodes=10)` | `agent.evaluate(state, episodes=10)`, one row of returns per seed | [Quickstart](quickstart.md#use-the-trained-policy) |
| `tensorboard_log="runs"` | `LoggingConfig(backends=(TensorBoardBackend("runs"),))`, in a script under `if __name__ == "__main__":`; the viewer is `pip install tensorboard` | [Logging](../guide/logging.md#backends) |
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

`predict` returns the action alone, where SB3's returns an `(action,
state)` pair, and as a JAX array. A Gymnasium environment wants a NumPy
value, so pass it through `np.asarray`:

```python
import gymnasium
import numpy as np

gym_env = gymnasium.make("CartPole-v1")
gym_obs, _ = gym_env.reset(seed=0)
action = np.asarray(agent.predict(state, gym_obs))
gym_obs, reward, terminated, truncated, _ = gym_env.step(action)
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

`load` returns two values, the agent and its state, where SB3's returns the
model alone. The file records the agent's plain arguments. Objects (a prebuilt
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
