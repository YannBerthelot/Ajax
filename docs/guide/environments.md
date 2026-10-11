# Environments

Ajax trains on JAX environments: [gymnax](https://github.com/RobertTLange/gymnax),
[Brax](https://github.com/google/brax) and
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), or
any environment with gymnax's interface, such as
[TargetGym](https://github.com/YannBerthelot/TargetGym). The environment
runs inside the same compiled program as the agent, so it must be written in
JAX: a Gymnasium environment that steps in Python cannot be used.

## Environments by name

Pass an id as the agent's first argument. Ajax looks it up in this order and
uses the first match:

1. gymnax: `"Pendulum-v1"`, `"CartPole-v1"`, `"MountainCarContinuous-v0"`, ...
2. MuJoCo Playground: `"CheetahRun"`, `"CartpoleBalance"`, ...
3. Brax: `"ant"`, `"halfcheetah"`, ...

An unknown id raises `ValueError`. The agent builds `n_envs` copies, which
step in parallel.

```python
from ajax import PPO, SAC

agent = SAC("Pendulum-v1")  # gymnax
agent = PPO("CartPole-v1")  # gymnax, discrete actions
```

gymnax environments with image observations are flattened to one vector,
because the networks read flat vectors. A [CNN encoder](networks.md) can
rebuild the image from it.

### Brax and Playground

Brax and Playground episodes last `episode_length` simulator steps, 1,000 by
default. PPO and the world models take `episode_length` as an argument; for
the other agents, build the environment as below and pass it in. On a CPU,
Playground environments run MJX's pure-JAX implementation.

Playground draws a new initial state for every episode (`fresh_reset=True`),
as dm_control does. Runs made before this became the default restarted every
episode of copy `i` from one cached state. To reproduce them, or to change
how the environment is built, build it with `build_env_from_id` and pass it
to the agent with the same `n_envs`:

```python
from ajax import SAC
from ajax.environments.create import build_env_from_id

# Every episode of copy i restarts from the same cached state, as older runs did.
env, _ = build_env_from_id("CartpoleBalance", n_envs=4, episode_length=1_000, fresh_reset=False)
agent = SAC(env, n_envs=4)
```

`build_env_from_id(..., differentiable_reset=True)` lets `jax.grad` flow
through the reset itself, for instance to physics parameters, at the cost of
computing a reset at every step. Without it, gradients still flow through
the actions, the policy and the environment state. See
[Differentiable simulation](differentiable-simulation.md).

Evaluation rebuilds a Brax or Playground environment from its id. A
prebuilt Brax environment with wrappers of your own would lose them there.
Register a builder instead, and every rebuild uses it:

```python
from ajax.environments.create import register_brax_builder, register_playground_builder
```

A builder takes `(n_envs, episode_length)`, returns the fully wrapped
environment and sets its `_ajax_env_id` attribute to the id. To support
action repeat it also accepts an `action_repeat` keyword.

## Your own environment

Any gymnax environment works: pass the instance, and its parameters as
`env_params`. Wrappers you add stay on during training and evaluation.

```python
import gymnax

from ajax import SAC

env, params = gymnax.make("Pendulum-v1")
params = params.replace(max_steps_in_episode=100)  # shorter episodes
agent = SAC(env, env_params=params, learning_starts=500)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

`SAC("Pendulum-v1", env_params=params)` does the same with an id. Brax and
Playground ids take no `env_params`: passing some raises.

### TargetGym

TargetGym's set-point tasks (aircraft, process control, industrial plants)
use gymnax's interface, so an instance goes straight in:

```python
# doc: skip (needs the optional plane group: uv sync --group plane)
from target_gym import Plane3DCircle

from ajax import SAC

agent = SAC(Plane3DCircle(), n_envs=4, learning_starts=1_000)
state, _ = agent.train(seed=[0, 1], n_timesteps=20_000)
```

Ajax's benchmarks use it ([Performance](performance.md)). The `plane`
dependency group installs it.

### Wrappers

Ajax ships gymnax wrappers for common changes to a task:

| Wrapper (`ajax.wrappers`) | What it does |
| --- | --- |
| `InitialStateWrapper(env, init_state_fn)` | replaces the reset state, for every episode (fixed or narrowed initial conditions) |
| `TransformObservation(env, fn)` | applies `fn` to every observation |
| `TransformReward(env, fn)` | applies `fn` to every reward |
| `TerminatedTruncatedWrapper(env)` | splits an old-style `done` into `terminated` and `truncated` (see [time limits](#time-limits)) |
| `PIDTermsWrapper` (`ajax.environments.pid_terms`) | adds a PID's error terms to the observation (see [below](#pid-error-terms)) |

```python
import gymnax
import jax.numpy as jnp

from ajax import SAC
from ajax.wrappers import InitialStateWrapper, TransformReward

env, params = gymnax.make("Pendulum-v1")


def hanging_down(key, state, params):
    """Start every episode at rest, hanging down."""
    return state.replace(theta=jnp.asarray(jnp.pi), theta_dot=jnp.asarray(0.0))


env = InitialStateWrapper(env, hanging_down)
env = TransformReward(env, lambda reward: reward / 10.0)
agent = SAC(env, env_params=params, learning_starts=500)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

Two more modules turn one environment into a family of tasks:
`ajax.environments.model_reference` (follow a reference model's response)
and `ajax.environments.system_class` (a distribution over physical
parameters, one system per copy). [Differentiable simulation](differentiable-simulation.md)
covers both.

## Normalisation

Two arguments normalise what the agent learns from, with running statistics
pooled over the `n_envs` copies:

- `normalize_observations=True`: each observation dimension to mean 0 and
  variance 1.
- `normalize_rewards=True`: divides rewards by a running standard
  deviation, by default of the single-step reward. With
  `reward_normalization_gamma=0.99` it uses the standard deviation of the
  discounted return instead, as Stable-Baselines3's `VecNormalize` and
  Gymnasium's `NormalizeReward` do.

The logged returns stay in the environment's units. Evaluation normalises
observations with the training statistics, frozen, and never normalises
rewards.

The statistics live in the environment state:
`state.collector_state.env_state.normalization_info.obs` for gymnax (in
`info["normalization_info"]` for Brax), with `mean` and `var`. Every copy
holds the same pooled values. To run the policy on a new observation
yourself, normalise it as `(obs - mean) / sqrt(var + 1e-8)` first.

SAC also offers `normalize_obs_running=True`, which keeps the statistics in
the agent instead, so they also cover the dimensions SAC appends to the
observation itself (an expert's action or state). An expert policy always
reads the raw observation, in the environment's own units: see
[Expert guidance](expert-guidance.md).

### Raw observation columns

Some observation entries must keep their units: a PID's gains only mean
something on errors in the plant's units. An environment or wrapper declares
its first `k` columns raw with a `raw_obs_columns = k` attribute. Every
normaliser, in the environment or in the agent, then leaves those columns
unchanged and normalises the rest. `PIDTermsWrapper` declares its terms this
way.

## PID error terms

A PID controller acts on the tracking error `e = target - current`, its
integral `I` and its difference `D`. `PIDTermsWrapper` puts `(e, I, D)` in
front of a gymnax environment's observation, so any agent can read them:

```python
# doc: skip (needs the optional plane group: uv sync --group plane)
from target_gym import FirstOrderSystem

from ajax import SAC
from ajax.environments.pid_terms import PIDTermsWrapper

# FirstOrderSystem observes [x, target]: index 0 is controlled, index 1 its set-point.
env = PIDTermsWrapper(FirstOrderSystem(), current_idx=(0,), target_idx=(1,), dt=1.0)
agent = SAC(env, n_envs=4, normalize_observations=True, learning_starts=500)
state, _ = agent.train(seed=0, n_timesteps=1_000)
stats = state.collector_state.env_state.normalization_info.obs
print(stats.mean[0, 0], stats.var[0, 0])  # the three terms are pinned at mean 0, variance 1
```

The observation becomes `[e, I, D, raw observation]`, `3 * n_errors`
columns longer. `dt` is the sampling period: the integral sums `dt * e` and
the difference is divided by `dt`, so gains keep their continuous-time
units. `integral_limit` clamps the integral (anti-windup). The terms restart
with every episode, auto-resets included. The [PID policy](networks.md#the-pid-policy)
reads them.

## Actions

Ajax clips every continuous action to the environment's bounds and does not
rescale it otherwise. SAC, TD3, REDQ, ASAC, AVG, APO, UDRL and APG squash
their continuous actions into `[-1, 1]` with `tanh`; PPO samples an
unbounded Gaussian unless you pass `squash=True`. On an environment whose
bounds are wider than
`[-1, 1]`, a squashed agent reaches only part of the range: on gymnax's
Pendulum, whose torque goes from -2 to 2, SAC applies at most 1. Every Brax
and Playground action lies in `[-1, 1]`. DreamerV3 and TD-MPC2 map `[-1, 1]`
onto the bounds instead, as their papers do.

### Action repeat

DreamerV3 and TD-MPC2 take `action_repeat`: each agent step runs the
simulator that many times with the same action and returns the summed
reward. It works on Brax and Playground environments built from an id;
`episode_length` then counts simulator steps, so an episode lasts
`episode_length // action_repeat` agent steps. gymnax environments and
prebuilt environments support only 1.

## Time limits

An episode can end in two ways. It *terminates* when the task itself ends:
the pole fell, the robot crashed. It is *truncated* when it hits the time
limit, which is not part of the task. Ajax keeps the two apart: after a
termination the value of what follows is zero, but after a truncation the
agent bootstraps on the value of the last observation, as Stable-Baselines3
and CleanRL do. Treating a time limit as a termination would teach the agent
that the world ends there, and underestimate the value of long-running
states.

Auto-resetting environments return the next episode's first observation
when an episode ends. Ajax recovers the last observation of the ended
episode from the step's `info` (`final_obs` on Brax and Playground, where
Ajax's wrapper stack records it) and stores it for the bootstrap.

Each backend reports the two flags its own way:

- **gymnax 1.0** returns `terminated` and `truncated` apart. Ajax uses them
  as they are.
- **Older gymnax-style environments** return one `done` that folds in the
  time limit. Ajax recovers the time limit from `info["truncated"]` when the
  environment publishes it, else from the state's step counter. Wrapping
  such an environment in `TerminatedTruncatedWrapper` makes the split
  explicit.
- **Brax and Playground** report the time limit in `info["truncation"]`,
  set by their episode wrapper.

The [algorithm pages](../algorithms/index.md) note the agents that treat
episode ends differently, such as the average-reward agents.
