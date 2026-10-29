# Environments

Ajax trains on JAX environments: [gymnax](https://github.com/RobertTLange/gymnax),
[Brax](https://github.com/google/brax) and
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), or
any environment with gymnax's interface, such as
[TargetGym](https://github.com/YannBerthelot/TargetGym). The environment
runs inside the same compiled program as the agent, which is what makes
training fast. A Gymnasium environment, which steps in Python, can be used
too if you ask for it, at a cost: see
[Gymnasium environments](#gymnasium-environments).

## Environments by name

Pass an id as the agent's first argument. Ajax looks it up in this order and
uses the first match:

1. gymnax: `"Pendulum-v1"`, `"CartPole-v1"`, `"MountainCarContinuous-v0"`, ...
2. MuJoCo Playground: `"CheetahRun"`, `"CartpoleBalance"`, ...
3. Brax: `"ant"`, `"halfcheetah"`, ...

Brax and Playground ids need Ajax's `brax` or `playground` extra
([Installation](../getting-started/installation.md#extras)). An unknown id
raises `ValueError`, naming the extras that are not installed. The agent
builds `n_envs` copies, which step in parallel.

```python
from ajax import PPO, SAC

agent = SAC("Pendulum-v1")  # gymnax
agent = PPO("CartPole-v1")  # gymnax, discrete actions
```

gymnax environments with image observations are flattened to one vector,
because the networks read flat vectors. A [CNN encoder](networks.md) can
rebuild the image from it.

### Brax and Playground

Every agent takes `episode_length` and `action_repeat` for an environment
built from a Brax or Playground id. `episode_length` counts simulator steps,
1,000 by default; gymnax environments keep their own time limit, set in
their parameters (below). On a CPU, Playground environments run MJX's
pure-JAX implementation.

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

### Episode keys

Every environment's resets are keyed by episode: copy `i`'s `k`-th episode
starts on a key of `(i, k)` alone, and on gymnax its steps draw on keys of
`(i, k, step)`. Two runs that differ by an option, an extension or their
updates then start each copy's `k`-th episode from the same state, even where
their actions end the episodes at other steps. `episode_keys=False` keeps
each backend's own auto-reset, as runs made before episode keys did:

```python
import gymnax

from ajax.environments.create import build_env_from_id, with_episode_keys

env, params = build_env_from_id("Pendulum-v1", episode_keys=False)
# An environment you built yourself:
my_gymnax_env, my_params = gymnax.make("Pendulum-v1")
env = with_episode_keys(my_gymnax_env, on=False)
```

A gymnax environment that overrides gymnax's own `step` or `reset` (bsuite's
DeepSea) is refused with episode keys on, since they would bypass it.

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

The Plane3DCircle scenario of `benchmarks/perf_bench.py` trains on it
([Performance](performance.md)); the per-agent benchmark does not. The
`plane` dependency group installs it.

### Wrappers

Ajax ships gymnax wrappers for common changes to a task:

| Wrapper (`ajax.wrappers`) | What it does |
| --- | --- |
| `InitialStateWrapper(env, init_state_fn)` | replaces the reset state, for every episode (fixed or narrowed initial conditions) |
| `ObservationNoiseWrapper(env, sigma, scale=1.0)` | adds sensor noise to the observations (see [below](#sensor-noise)) |
| `TerminatedTruncatedWrapper(env)` | splits an old-style `done` into `terminated` and `truncated` (see [time limits](#time-limits)) |

```python
import gymnax
import jax.numpy as jnp

from ajax import SAC
from ajax.wrappers import InitialStateWrapper, ObservationNoiseWrapper

env, params = gymnax.make("Pendulum-v1")


def hanging_down(key, state, params):
    """Start every episode at rest, hanging down."""
    return state.replace(theta=jnp.asarray(jnp.pi), theta_dot=jnp.asarray(0.0))


env = InitialStateWrapper(env, hanging_down)
env = ObservationNoiseWrapper(env, sigma=0.05)
agent = SAC(env, env_params=params, learning_starts=500)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

For anything else, subclass `ajax.wrappers.GymnaxWrapper`: it forwards
every method to the wrapped environment (`self._env`), so you override only
what you change.

```python
import gymnax

from ajax import SAC
from ajax.wrappers import GymnaxWrapper


class ScaleReward(GymnaxWrapper):
    """Divide every reward by 10."""

    def step(self, key, state, action, params=None):
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        return obs, state, reward / 10.0, terminated, truncated, info


env, params = gymnax.make("Pendulum-v1")
agent = SAC(ScaleReward(env), env_params=params, learning_starts=500)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

Two more modules turn one environment into a family of tasks:
`ajax.environments.model_reference` (follow a reference model's response)
and `ajax.environments.system_class` (a distribution over physical
parameters, one system per copy). [Differentiable simulation](differentiable-simulation.md)
covers both.

### Sensor noise

`ObservationNoiseWrapper(env, sigma, scale)` adds `sigma * scale * N(0, 1)`
to every observation, drawn afresh at every reset and step. `scale` is one
number or one per observation dimension, such as the observations' standard
deviations. The final observation of an ended episode gets the same draw as
the step, so a stored transition holds the observation the agent saw. The
environment's own random draws are unchanged.

The noise sits on the observation only: the state's `get_obs` stays
noiseless, and that is what an [expert](expert-guidance.md) reads on a
gymnax environment. `ObservationNoiseWrapperBrax` does the same on a Brax or
Playground stack built by `build_env_from_id`; since evaluation rebuilds
those from their id, add it inside a registered builder.

## Gymnasium environments

A [Gymnasium](https://gymnasium.farama.org) environment steps in Python, so
Ajax cannot compile it with the agent. You can still train on one, MuJoCo's
tasks included, by asking for it with the `gymnasium:` prefix:

```python
from ajax import SAC

agent = SAC("gymnasium:Pendulum-v1", learning_starts=500)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

The rest of the id goes to `gymnasium.make`, so `"gymnasium:my_pkg:MyEnv-v0"`
imports `my_pkg` first. For an environment that needs arguments, build it
with `GymnasiumEnv`, which passes them to `gymnasium.make`:

```python
# doc: skip (needs the gymnasium extra: pip install "ajax-rl[gymnasium]")
from ajax import SAC
from ajax.environments.gymnasium import GymnasiumEnv

agent = SAC(GymnasiumEnv("HalfCheetah-v5", ctrl_cost_weight=0.05))
```

An id without the prefix never reaches Gymnasium: `"CartPole-v1"` is
gymnax's. MuJoCo's tasks need the `gymnasium` extra
([Installation](../getting-started/installation.md#extras)).

Every step goes from the compiled program to Python and back, and the copies
step one after another in Python. Training is slower than on a JAX
environment, and a GPU does not help. Building one raises a
`PythonEnvironmentWarning` that says so and names the closest JAX tasks.
Tasks with the same name do not follow the same rules in Gymnasium, Brax and
MuJoCo Playground, so the warning says what differs. Returns are not
comparable across them:

| Gymnasium | Closest JAX task | What differs |
| --- | --- | --- |
| `HalfCheetah-v5` | Brax `"halfcheetah"` | Brax's physics only |
| `Hopper-v5`, `Walker2d-v5` | Brax `"hopper"`, `"walker2d"` | Brax's physics; the healthy reward is also paid on the step that ends the episode |
| `Ant-v5` | Brax `"ant"` | No contact cost; 27 observations instead of 105 |
| `Humanoid-v5` | Brax `"humanoid"` | No contact cost; other observations |
| `HumanoidStandup-v5` | Brax `"humanoidstandup"` | A height reward 5 times smaller, other costs and observations |
| `InvertedPendulum-v5` | Brax `"inverted_pendulum"` | The reward is also paid on the step that ends the episode |
| `InvertedDoublePendulum-v5` | Brax `"inverted_double_pendulum"` | A 0.02 s control step instead of 0.05 s; smaller reset noise; 8 observations instead of 9 |
| `Reacher-v5` | Brax `"reacher"` | 1,000-step episodes by default instead of 50; other targets and observations |
| `Pusher-v5` | Brax `"pusher"` | The reward is read before the step; other placements |
| `Swimmer-v5` | Brax `"swimmer"` | Brax's physics only |
| `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0` | gymnax, same id | The same task, in float32 |
| `MountainCarContinuous-v0` | gymnax, same id | The car is not stopped at the left wall, and the goal also counts when reached moving left |

Brax's physics is its own solver, not MuJoCo's: Euler steps, fewer solver
iterations, and fewer bodies that collide (usually only the feet). MuJoCo
Playground has DeepMind Control tasks close to several of them
(`"CheetahRun"`, `"HopperHop"`, `"WalkerWalk"`, `"HumanoidRun"`,
`"ReacherEasy"`, ...): other tasks, with another model, a reward between 0
and 1 and no early termination. The warning names them too.

What changes with a Gymnasium environment:

- **Seeds train one after another**, each the run it would be alone: JAX
  cannot run a Python step inside the batch of seeds.
- **Evaluation steps copies of its own**, so training's episodes go on where
  they were.
- **Resets follow Gymnasium's generator**: each copy is seeded at its first
  reset from the run's key, and Ajax's [episode keys](#episode-keys) do not
  apply. The same seed gives the same run.
- **Refused**: resuming from a saved state (the simulator's state lives in
  Python, outside the agent's state), gradients through the environment
  (APG), `action_repeat` above 1, observation spaces other than a Box,
  action spaces other than a Discrete starting at 0 or a one-dimensional Box,
  and environments without a time limit (pass `max_episode_steps`).
- The environment state Ajax sees holds only the last observation, so
  wrappers and experts that read gymnax state fields do not apply.

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

The agents learning from the rollout they just collected (PPO, APO, PQN and
AVG) store each observation normalised with the statistics of its
collection, which count it: the rollout holds what the actor saw, as
openai/baselines ppo2's `VecNormalize` keeps it. The statistics live in
`state.collector_state.obs_norm_info`, with `mean` and `var`. To run the
policy on a new observation yourself, normalise it as `(obs - mean) /
sqrt(var + 1e-8)` first.

The replay agents (SAC, TD3, REDQ, ASAC, DQN, UDRL) learn from rows
collected long before, so their replay keeps the environment's own
observations and rewards, and a replayed row is normalised with the current
statistics, as Stable-Baselines3's replay buffer does under `VecNormalize`.
UDRL's rewards are the exception for now: its segments keep them normalised
by the environment. The agent
normalises observations itself: the statistics live in
`state.collector_state.obs_norm_info` (mirrored on its networks, which apply
them, so `get_pi` takes raw observations) and cover the dimensions SAC
appends to the observation (an expert's action or state), not UDRL's
command. A replayed reward
is divided by the environment normaliser's current statistics. An expert
policy always reads the raw observation, in the environment's own units: see
[Expert guidance](expert-guidance.md).

### Raw observation columns

Some observation entries must keep their units: a PID's gains only mean
something on errors in the plant's units. An environment or wrapper declares
its first `k` columns raw with a `raw_obs_columns = k` attribute. Every
normaliser, in the environment or in the agent, then leaves those columns
unchanged and normalises the rest. Observation features keep their columns
raw by declaring it (`observes_normalised = False`) instead.

## PID error terms

A PID controller acts on the tracking error `e = target - current`, its
integral `I` and its difference `D`. The observation feature `PIDTermsInput`
appends `(e, I, D)` to the observation, so any agent that takes features can
read them:

```python
# doc: skip (needs the optional plane group: uv sync --group plane)
from target_gym import FirstOrderSystem

from ajax import SAC
from ajax.extensions.observe import PIDTermsInput

# FirstOrderSystem observes [x, target]: index 0 is controlled, index 1 its set-point.
terms = PIDTermsInput(current_idx=(0,), target_idx=(1,), dt=1.0)
agent = SAC(
    FirstOrderSystem(),
    n_envs=4,
    normalize_observations=True,
    learning_starts=500,
    extensions=(terms,),
)
state, _ = agent.train(seed=0, n_timesteps=1_000)
stats = state.collector_state.obs_norm_info  # SAC normalises observations itself
print(stats.mean.shape)  # (1, 2): the plant's columns; the terms after them stay raw
```

The observation becomes `[raw observation, e, I, D]`, `3 * n_errors`
columns longer. `dt` is the sampling period: the integral sums `dt * e` and
the difference is divided by `dt`, so gains keep their continuous-time
units. `integral_limit` clamps the integral (anti-windup). The terms restart
with every episode, auto-resets included. The [PID policy](networks.md#the-pid-policy)
reads them.

## Actions

Every continuous agent acts in `[-1, 1]` on each action dimension, whatever
the environment's bounds: these are the agent's units. Before each step,
Ajax maps the action onto the environment's box `[low, high]`: it clips the
action to `[-1, 1]`, then sends `c + h * a`, with `c = (high + low) / 2` the
centre of the box and `h = (high - low) / 2` its half-width (CleanRL's
`action_bias` and `action_scale`). On gymnax's Pendulum, whose torque goes
from -2 to 2, an action of 1 applies a torque of 2. A dimension with an
infinite bound gets the clipped action as it is, and discrete actions pass
through. Brax and Playground actions already lie in `[-1, 1]`, where the map
is just the clip. With per-copy parameters from a system class, each copy
is mapped onto its own system's bounds.

The map is the same for every agent and everywhere an action reaches the
environment: collection, evaluation and expert rollouts. SAC, TD3, REDQ,
ASAC, AVG, UDRL and APG (by default) squash their actions into `[-1, 1]`
with `tanh`; PPO samples an unbounded Gaussian unless you pass
`squash=True`, and APO one whose mean is bounded by `tanh`: their samples
are clipped by the map. The replay buffers and
rollouts store the action in agent units.

`ajax.environments.utils.agent_action_to_env` is the map and
`env_action_to_agent` its inverse, which does not clip. An expert policy
returns actions in the environment's units, and Ajax reads them through the
inverse: see [Expert guidance](expert-guidance.md).

### Action repeat

Every agent takes `action_repeat`: each agent step runs the simulator that
many times with the same action and returns the summed reward. It works on
Brax and Playground environments built from an id; `episode_length` then
counts simulator steps, so an episode lasts `episode_length // action_repeat`
agent steps. gymnax environments and prebuilt environments support only 1
and raise otherwise. The world models use it for their papers' protocols
(see [World models](world-models.md)).

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
episode from the step's `info` (`final_observation` on gymnax, `final_obs`
on Brax and Playground, where Ajax's wrapper stack records it) and stores
it for the bootstrap.

Each backend reports the two flags its own way:

- **gymnax 1.0** returns `terminated` and `truncated` apart. Ajax uses them
  as they are.
- **Older gymnax-style environments** return one `done` that folds in the
  time limit, which gymnax hands back as `terminated`: every time limit then
  looks like the end of the task. Wrap such an environment in
  `TerminatedTruncatedWrapper`, which keeps `terminated` only where the
  time limit was not reached.
- **Brax and Playground** report the time limit in `info["truncation"]`,
  set by their episode wrapper.

The [algorithm pages](../algorithms/index.md) note the agents that treat
episode ends differently, such as the average-reward agents.
