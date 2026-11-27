# Environments

Ajax trains on JAX environments: [gymnax](https://github.com/RobertTLange/gymnax),
[Brax](https://github.com/google/brax) and
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), or
any environment with gymnax's interface, such as
[TargetGym](https://github.com/YannBerthelot/TargetGym). The environment
runs inside the same compiled program as the agent, which is what makes
training fast. A Gymnasium environment, which steps in Python, works too,
more slowly ([Gymnasium environments](#gymnasium-environments)).

## Environments by name

Pass an id as the agent's first argument. Ajax looks it up in this order and
uses the first match:

1. gymnax: `"Pendulum-v1"`, `"CartPole-v1"`, `"MountainCarContinuous-v0"`, ...
2. MuJoCo Playground: `"CheetahRun"`, `"CartpoleBalance"`, ...
3. Brax: `"ant"`, `"halfcheetah"`, ...

Two prefixes choose a source outright: `"mujoco:<id>"` for
[Gymnasium's MuJoCo tasks in JAX](#gymnasiums-mujoco-tasks-in-jax), and
`"gymnasium:<id>"` for [Gymnasium environments](#gymnasium-environments).
The agent builds `n_envs` copies, which step in parallel.

| Source | Its ids | Needs |
| --- | --- | --- |
| gymnax | `gymnax.registered_envs` | nothing more |
| MuJoCo Playground | `mujoco_playground.registry.ALL_ENVS` | the `playground` extra |
| Brax | the modules of [`brax.envs`](https://github.com/google/brax/tree/main/brax/envs) | the `brax` or `playground` extra |
| `mujoco:` | `gymnasium_mjx.TASKS`, [listed below](#gymnasiums-mujoco-tasks-in-jax) | the `gymnasium-mjx` extra |
| `gymnasium:` | any id `gymnasium.make` takes | nothing more; the `gymnasium` extra for MuJoCo's tasks |

[Installation](../getting-started/installation.md#extras) lists the extras.
An unknown id raises `ValueError`, naming the closest ids and the extras
that are not installed; a `"mujoco:"` id without its package raises
`ImportError`, naming the extra.

```python
import gymnax

print(gymnax.registered_envs)  # ['Pendulum-v1', 'CartPole-v1', ...]
```

```python
from ajax import PPO, SAC

agent = SAC("Pendulum-v1")  # gymnax
agent = PPO("CartPole-v1")  # gymnax, discrete actions
```

gymnax environments with image observations are flattened to one vector,
because the networks read flat vectors. An MLP reads that vector, with a
warning, unless you pass `cnn_image_shape`, which turns on a
[CNN encoder](networks.md#image-observations) that rebuilds the image.

### Brax and Playground

An environment built from a Brax or Playground id takes the agent's
`episode_length`, in simulator steps (1,000 by default), and
[`action_repeat`](#action-repeat); a gymnax environment keeps the time
limit of its parameters ([below](#your-own-environment)). Playground runs
MJX's pure-JAX physics, on a CPU or a GPU, or
[MuJoCo Warp](#mujoco-warp-on-a-gpu) on an NVIDIA GPU.

To change how an environment is built, build it with `build_env_from_id`
and pass it to the agent with the same `n_envs`:

```python
from ajax import SAC
from ajax.environments.create import build_env_from_id

# Every episode of copy i restarts from one cached state.
env, _ = build_env_from_id("CartpoleBalance", n_envs=4, episode_length=1_000, fresh_reset=False)
agent = SAC(env, n_envs=4)
```

| `build_env_from_id` argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | the copies to build |
| `episode_length`, `action_repeat` | `None`, `1` | as the agent's arguments; `None` is 1,000 steps, or a `mujoco:` task's own time limit |
| `fresh_reset` | `True` | Playground: a new initial state for every episode, as dm_control draws; `False` restarts each copy's episodes from one cached state |
| `physics` | `"jax"` | Playground and `mujoco:`: MJX's pure-JAX physics, or `"warp"` ([below](#mujoco-warp-on-a-gpu)) |
| `episode_keys` | `True` | key each copy's resets by episode ([below](#episode-keys)) |
| `differentiable_reset` | `False` | let `jax.grad` flow through the reset itself, to physics parameters say, at the cost of a reset every step; gradients flow through the actions, the policy and the environment state either way ([Differentiable simulation](differentiable-simulation.md)) |

Evaluation rebuilds a Brax or Playground environment from its id, so a
prebuilt one would lose wrappers of your own there. Register a builder
instead, and every rebuild uses it:

```python
from ajax.environments.create import register_brax_builder, register_playground_builder
```

A builder takes `(n_envs, episode_length)`, returns the fully wrapped
environment and sets its `_ajax_env_id` attribute to the id. To support
action repeat it also accepts an `action_repeat` keyword.

!!! warning "HopperHop, HopperStand and the finger tasks with several seeds"
    `HopperHop`, `HopperStand`, `FingerSpin`, `FingerTurnEasy` and
    `FingerTurnHard` do not compile when the body is batched two ways with
    different sizes, such as several seeds each with several environments
    or evaluated over several episodes: training stops at its first
    evaluation with `JaxRuntimeError: INTERNAL: during context [hlo
    verifier]`. The fault is in XLA's algebraic simplifier (seen with jax
    0.11.2 and 0.12.0.1, on a CPU). Until it is fixed there, turn that pass
    off for these runs:

    ```bash
    XLA_FLAGS=--xla_disable_hlo_passes=algsimp python train.py
    ```

    Results are the same up to floating-point rounding; training ran 8 to
    15% slower on a CPU. A single seed compiles without it. A GPU fails the
    same way, and there the workaround does not combine with
    `--xla_gpu_deterministic_ops=true`.

### MuJoCo Warp on a GPU

On an NVIDIA GPU, Playground tasks and the `"mujoco:"` tasks can run on
MuJoCo Warp, Playground's own default, instead of MJX's pure-JAX physics.
On an L40S, SAC and PPO trained about five times faster on it
([Performance](performance.md)). Build the environment with
`physics="warp"` and pass it to the agent with the same `n_envs`;
evaluation runs on the same physics:

```python
# doc: skip (needs an NVIDIA GPU and the gymnasium-mjx extra)
from ajax import SAC
from ajax.environments.create import build_env_from_id

env, _ = build_env_from_id("mujoco:Hopper-v5", n_envs=4, physics="warp")
agent = SAC(env, n_envs=4)
```

- **It needs** an NVIDIA GPU and MuJoCo 3.13 or newer (the first whose Warp
  reports the convex collisions it drops). Nothing can be differentiated
  through it: it refuses an older MuJoCo, a CPU, `differentiable_reset=True`
  and APG.
- **The runs differ from pure JAX's.** After 15 steps from the same state,
  on `mujoco:Hopper-v5` both stayed within 2e-6 of C MuJoCo's joint
  positions; on `mujoco:Walker2d-v5`, Warp stayed within 2e-6 and pure JAX
  moved 1.3e-3 away.
- **Contacts have a fixed room.** Warp keeps one contact buffer for all the
  worlds a run steps at once, seeds times copies: the `"mujoco:"` tasks
  give it 49,152 contacts (1,024 worlds of 48) and a convex-collision slot
  per geom pair in those worlds, and Playground's tasks set their own. A
  step that runs out of room stops the run with an error rather than drop
  contacts.

### Gymnasium's MuJoCo tasks in JAX

`"mujoco:<id>"` builds one of Gymnasium's eleven v5 MuJoCo tasks on MJX, so
it runs in the compiled program like any Playground task: `Ant-v5`,
`HalfCheetah-v5`, `Hopper-v5`, `Humanoid-v5`, `HumanoidStandup-v5`,
`InvertedDoublePendulum-v5`, `InvertedPendulum-v5`, `Pusher-v5`,
`Reacher-v5`, `Swimmer-v5` and `Walker2d-v5`. The tasks come from the
[gymnasium-mjx](https://github.com/YannBerthelot/gymnasium-mjx) package,
which the `gymnasium-mjx` extra installs.

```python
# doc: skip (needs the gymnasium-mjx extra: pip install "ajax-rl[gymnasium-mjx]")
from ajax import SAC

agent = SAC("mujoco:Reacher-v5")
```

Each task loads Gymnasium's own model file and follows its `<task>_v5.py`:
the same observations, rewards, terminations, reset distribution and time
limit (50 steps for Reacher, 100 for Pusher, 1,000 for the others, unless
you pass `episode_length`).

- Resets draw from Gymnasium's distributions, but not its random stream.
- Humanoid's models ask for the PGS solver, which MJX lacks: they run with
  Newton's.
- One step from the same state gives Gymnasium's values to about 1e-10 in
  float64 and 1e-4 in float32, the precision training runs in. Over a whole
  episode the two drift apart where bodies touch the ground, as two builds
  of MuJoCo do. gymnasium-mjx's README says how each task is checked.

For a task's own keywords (`reset_noise_scale`, `ctrl_cost_weight`, ...),
build it with `gymnasium_mjx.make` and pass the task object in place of an
id. Any mujoco_playground environment object works this way: Ajax adds its
usual wrappers, keeps the object's time limit unless you pass
`episode_length`, and evaluates on the same object.

```python
# doc: skip (needs the gymnasium-mjx extra: pip install "ajax-rl[gymnasium-mjx]")
import gymnasium_mjx
from gymnasium_mjx.playground import PlaygroundTask

from ajax import SAC

agent = SAC(PlaygroundTask(gymnasium_mjx.make("Swimmer-v5", ctrl_cost_weight=1e-3)))
```

### Diverging simulators

A simulator can diverge: Playground's humanoid tasks do, rarely, after a
hard impact. Left alone, one such step would reach the agent as a NaN
reward and observation, and turn every network parameter NaN in the next
update.

So on a Brax or Playground environment built from an id, a step whose
observation or reward is not finite ends its episode: a termination with
reward 0, then the usual reset. The agent and the evaluations never see the
NaN. An evaluation episode that diverges keeps the return it had and shows
as a shorter `Eval/mean episodic length`, and the run logs the training
steps that diverged as `Train/non-finite env steps`. This is a workaround:
if that count grows, the task's physics needs a look.

Only non-finite values are caught: the very large finite steps just before
them still reach the agent. A registered builder's stack is kept as it
returns it, so add `ajax.wrappers.NonFiniteStepWrapper` there, directly
over the raw environment. gymnax environments do not get this guard.

### Episode keys

Every environment's resets are keyed by episode: copy `i`'s `k`-th episode
starts on a key of `(i, k)` alone, and on gymnax its steps draw on keys of
`(i, k, step)`. Two runs that differ by an option, an extension or their
updates then start each copy's `k`-th episode from the same state, even
where their actions end the episodes at other steps. `episode_keys=False`
keeps each backend's own auto-reset, as runs made before episode keys did:

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

### Writing one

An environment of your own is a subclass of gymnax's `Environment`: a state,
parameters, a reset, a step and the two spaces. Here a point moves along a
line and is rewarded for nearing a target:

```python
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ajax import PPO


@struct.dataclass
class State(environment.EnvState):  # EnvState holds `time`, the step count
    x: jax.Array  # the point's position
    target: jax.Array


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = 50  # the time limit


class Reach(environment.Environment):
    """A point moves by 0.1 * action towards a random target."""

    @property
    def default_params(self):
        return Params()

    def reset_env(self, key, params):
        target = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        state = State(time=0, x=jnp.float32(0.0), target=target)
        return self.get_obs(state), state

    def step_env(self, key, state, action, params):
        x = state.x + 0.1 * jnp.clip(action[0], -1.0, 1.0)
        state = state.replace(x=x, time=state.time + 1)
        reward = -jnp.abs(state.target - x)
        terminated = jnp.abs(x) > 2.0  # the task's own end, not the time limit
        return self.get_obs(state), state, reward, terminated, {}

    def get_obs(self, state, params=None, key=None):
        return jnp.stack([state.x, state.target])

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, (2,))

    def action_space(self, params):
        return spaces.Box(-1.0, 1.0, (1,))


agent = PPO(Reach(), env_params=Params(), n_envs=8, n_steps=64)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=5_120)
```

What to know when writing one:

- **`step_env` returns five values, and the fourth is the termination
  only.** Count the steps in `state.time`. gymnax's own `step`, the one Ajax
  and your loops call, compares it with `params.max_steps_in_episode`, adds
  the `truncated` flag and resets the environment, so `env.step` returns six
  values: `obs, state, reward, terminated, truncated, info`.
- **Leave the time limit out of that fourth value.** Returned there, it
  would be a termination, and the agent would stop bootstrapping at the
  limit ([Time limits](#time-limits)).
- **The step is compiled.** It reads its randomness from `key`, and chooses
  between values with `jnp.where`, not with a Python `if` on them.
- **A saved agent does not hold the environment.** `agent.save` stores the
  agent's plain arguments; pass the environment again when loading:
  `ajax.load(path, env_id=Reach(), env_params=Params())`.

### TargetGym

TargetGym's set-point tasks (aircraft, process control, industrial plants)
use gymnax's interface, so an instance goes straight in. Install it with
`pip install target-gym`, or `uv sync --group plane` in a checkout:

```python
# doc: skip (needs the optional plane group: uv sync --group plane)
from target_gym import Plane3DCircle

from ajax import SAC

agent = SAC(Plane3DCircle(), n_envs=4, learning_starts=1_000)
state, _ = agent.train(seed=[0, 1], n_timesteps=20_000)
```

### Wrappers

Ajax ships gymnax wrappers for common changes to a task:

| Wrapper (`ajax.wrappers`) | What it does |
| --- | --- |
| `InitialStateWrapper(env, init_state_fn)` | replaces the reset state, for every episode (fixed or narrowed initial conditions) |
| `ObservationNoiseWrapper(env, sigma, scale=1.0)` | adds sensor noise to the observations ([below](#sensor-noise)) |
| `ObservationWrapper(env)` | a base class: your `observation(obs)` changes every observation ([below](#changing-the-observations)) |
| `TerminatedTruncatedWrapper(env)` | splits an old-style `done` into `terminated` and `truncated` ([Time limits](#time-limits)) |

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

Two modules turn one environment into a family of tasks:
`ajax.environments.model_reference` (follow a reference model's response)
and `ajax.environments.system_class` (a distribution over physical
parameters, one system per copy).
[Differentiable simulation](differentiable-simulation.md) covers both.

#### Changing the observations

To change the observations, subclass `ObservationWrapper` and write
`observation`:

```python
import gymnax
import jax.numpy as jnp

from ajax import PPO
from ajax.wrappers import ObservationWrapper


class HideVelocities(ObservationWrapper):
    """CartPole showing positions only: both velocities read zero."""

    def observation(self, obs):
        # CartPole observes [position, velocity, angle, angular velocity].
        return obs * jnp.array([1.0, 0.0, 1.0, 0.0])


env, params = gymnax.make("CartPole-v1")
agent = PPO(HideVelocities(env), env_params=params, n_envs=8, n_steps=64)
state, _ = agent.train(seed=0, n_timesteps=1_024)
```

!!! warning "Do not override `reset` and `step` to change observations"
    When an episode ends, gymnax has already reset the environment, and the
    last observation of the finished episode travels in
    `info["final_observation"]`, which the agent learns from.
    `ObservationWrapper` changes it as it changes the others. A
    `GymnaxWrapper` that changes only what `reset` and `step` return leaves
    it as the environment made it: the agent then trains on two kinds of
    observations and learns poorly, with no error.

A wrapper that changes the observations' shape also defines
`observation_space(params)`. This CartPole needs memory to be solved:
[Memory](memory.md) trains on it.

### Sensor noise

`ObservationNoiseWrapper(env, sigma, scale)` adds `sigma * scale * N(0, 1)`
to every observation, drawn afresh at every reset and step. `scale` is one
number or one per observation dimension, such as the observations' standard
deviations. The final observation of an ended episode gets the step's draw,
so a stored transition holds what the agent saw, and the environment's own
random draws are unchanged.

The noise sits on the observation only: the state's `get_obs` stays
noiseless, and that is what an [expert](expert-guidance.md) reads on a
gymnax environment. `ObservationNoiseWrapperBrax` does the same on a Brax or
Playground stack; since evaluation rebuilds those from their id, add it in a
[registered builder](#brax-and-playground).

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
gymnax's. What changes with a Gymnasium environment:

- **It is slower.** Every step goes from the compiled program to Python and
  back, and the copies step one after another; a GPU does not help. A
  `PythonEnvironmentWarning` says so and names the closest JAX tasks.
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
- **No state fields.** The environment state Ajax sees holds only the last
  observation, so wrappers and experts that read gymnax state fields do not
  apply.

Tasks with the same name follow other rules in Gymnasium, Brax and MuJoCo
Playground, so returns are not comparable across them. The warning says
what differs from the closest JAX task:

| Gymnasium | Closest JAX task | What differs |
| --- | --- | --- |
| `HalfCheetah-v5` | Ajax `"mujoco:HalfCheetah-v5"` | Nothing: the same task on MJX |
| `HalfCheetah-v5` | Brax `"halfcheetah"` | Brax's physics only |
| `Hopper-v5`, `Walker2d-v5` | Ajax `"mujoco:Hopper-v5"`, `"mujoco:Walker2d-v5"` | Nothing: the same tasks on MJX |
| `Hopper-v5`, `Walker2d-v5` | Brax `"hopper"`, `"walker2d"` | Brax's physics; the healthy reward is also paid on the step that ends the episode |
| `Ant-v5` | Ajax `"mujoco:Ant-v5"` | Nothing: the same task on MJX |
| `Ant-v5` | Brax `"ant"` | No contact cost; 27 observations instead of 105 |
| `Humanoid-v5`, `HumanoidStandup-v5` | Ajax `"mujoco:Humanoid-v5"`, `"mujoco:HumanoidStandup-v5"` | The Newton solver instead of PGS, which MJX lacks |
| `Humanoid-v5` | Brax `"humanoid"` | No contact cost; other observations |
| `HumanoidStandup-v5` | Brax `"humanoidstandup"` | A height reward 5 times smaller, other costs and observations |
| `InvertedPendulum-v5` | Ajax `"mujoco:InvertedPendulum-v5"` | Nothing: the same task on MJX |
| `InvertedPendulum-v5` | Brax `"inverted_pendulum"` | The reward is also paid on the step that ends the episode |
| `InvertedDoublePendulum-v5` | Ajax `"mujoco:InvertedDoublePendulum-v5"` | Nothing: the same task on MJX |
| `InvertedDoublePendulum-v5` | Brax `"inverted_double_pendulum"` | A 0.02 s control step instead of 0.05 s; smaller reset noise; 8 observations instead of 9 |
| `Reacher-v5` | Ajax `"mujoco:Reacher-v5"` | Nothing: the same task on MJX |
| `Reacher-v5` | Brax `"reacher"` | 1,000-step episodes by default instead of 50; other targets and observations |
| `Pusher-v5` | Ajax `"mujoco:Pusher-v5"` | Nothing: the same task on MJX |
| `Pusher-v5` | Brax `"pusher"` | The reward is read before the step; other placements |
| `Swimmer-v5` | Ajax `"mujoco:Swimmer-v5"` | Nothing: the same task on MJX |
| `Swimmer-v5` | Brax `"swimmer"` | Brax's physics only |
| `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0` | gymnax, same id | The same task, in float32 |
| `MountainCarContinuous-v0` | gymnax, same id | The car is not stopped at the left wall, and the goal also counts when reached moving left |

Brax's physics is its own solver, not MuJoCo's: Euler steps, fewer solver
iterations, and fewer bodies that collide (usually only the feet). MuJoCo
Playground has DeepMind Control tasks close to several of them
(`"CheetahRun"`, `"HopperHop"`, `"WalkerWalk"`, `"HumanoidRun"`,
`"ReacherEasy"`, ...): other tasks, with another model, a reward between 0
and 1 and no early termination. The warning names them too.

## Normalisation

Running statistics, pooled over the `n_envs` copies, normalise what the
agent learns from:

| Argument | What it does | Default |
| --- | --- | --- |
| `normalize_observations` | each observation dimension to mean 0 and variance 1 | `False`; `True` for PPO, as ppo2's MuJoCo run; AVG always normalises its observations, as its paper does |
| `normalize_rewards` | divides the rewards by the running standard deviation of the discounted return, then clips them to ±10, as Stable-Baselines3's `VecNormalize` does | `False`; `True` for PPO |
| `reward_normalization_gamma` | that return's discount; `0` divides by the standard deviation of the single-step reward (which is 0 for a constant reward, such as CartPole's) | the agent's `gamma`, or 0.99 for an agent without one (APO, ASAC, RVI-SAC) |
| `observation_clip` | clips the normalised observations an on-policy agent stores to ± this value; a replay agent, whose networks normalise without clipping, refuses it | `None`; 10 for PPO, as ppo2's `VecNormalize` |

The logged returns stay in the environment's units. Evaluation normalises
observations with the training statistics, frozen, and never normalises
rewards. The statistics live in `state.collector_state.obs_norm_info`, with
`mean` and `var`; an observation is normalised as
`(obs - mean) / sqrt(var + 1e-8)`. [`agent.policy`](deploy.md) applies
them for you.

Where the agent stores what it learns from depends on how long it keeps it:

- **The agents learning from the rollout they just collected** (PPO, APO,
  REINFORCE, PQN and AVG) store each observation normalised with the
  statistics of its collection, which count it: the rollout holds what the
  actor saw, as openai/baselines ppo2's `VecNormalize` keeps it.
- **The [replay agents](../algorithms/index.md#agent-families) and UDRL**
  (<!-- agents: raw-replay -->SAC, ASAC, RVI-SAC, REDQ, CrossQ, TQC, TD3, IQL, DQN, DAgger, UDRL<!-- /agents -->)
  learn from rows collected long before, so their replay keeps the
  environment's own observations and rewards, and a replayed row is
  normalised with the current statistics, as Stable-Baselines3's replay
  buffer is under `VecNormalize`. UDRL's segments still keep their rewards
  normalised by the environment. The networks apply the observation
  statistics themselves, so `get_pi` takes raw observations; the statistics
  cover the columns SAC appends (an expert's action or state), not UDRL's
  command.

An expert policy always reads the raw observation, in the environment's
own units ([Expert guidance](expert-guidance.md)).

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
with every episode, auto-resets included. The
[PID policy](networks.md#the-pid-policy) reads them.

## Actions

Every continuous agent acts in `[-1, 1]` on each action dimension, whatever
the environment's bounds: these are the agent's units. Before each step,
Ajax clips the action to `[-1, 1]` and maps it onto the environment's box
`[low, high]` as `c + h * a`, with `c = (high + low) / 2` the centre of the
box and `h = (high - low) / 2` its half-width (CleanRL's `action_bias` and
`action_scale`). On gymnax's Pendulum, whose torque goes from -2 to 2, an
action of 1 applies a torque of 2.

- A dimension with an infinite bound gets the clipped action as it is, and
  discrete actions pass through. Brax and Playground actions already lie in
  `[-1, 1]`, where the map is just the clip. Copies with their own system
  parameters are each mapped onto their own bounds.
- The map is the same for every agent and wherever an action reaches the
  environment: collection, evaluation and expert rollouts. Replay buffers
  and rollouts store the action in agent units.
- SAC, TQC, RVI-SAC, CrossQ, TD3, REDQ, ASAC, AVG, DAgger, UDRL and APG (by
  default) squash their actions into `[-1, 1]` with `tanh`. PPO and
  REINFORCE sample an unbounded Gaussian (PPO unless `squash=True`), and APO
  and IQL one whose mean is bounded by `tanh`: the map clips their samples.

`ajax.environments.utils.agent_action_to_env` is the map and
`env_action_to_agent` its inverse, which does not clip. An expert policy
returns actions in the environment's units, and Ajax reads them through the
inverse: see [Expert guidance](expert-guidance.md).

### Action repeat

Every agent takes `action_repeat`: each agent step runs the simulator that
many times with the same action and returns the summed reward. It works on
Brax and Playground environments built from an id, where an episode lasts
`episode_length // action_repeat` agent steps; gymnax and prebuilt
environments take only 1. The world models use it for their papers'
protocols ([World models](world-models.md)).

## Time limits

An episode can end in two ways. It *terminates* when the task itself ends:
the pole fell, the robot crashed. It is *truncated* when it hits the time
limit, which is not part of the task. Ajax keeps the two apart, as
Stable-Baselines3 and CleanRL do: after a termination the value of what
follows is zero, but after a truncation the agent bootstraps on the value
of the last observation. Treating a time limit as a termination would
teach the agent that the world ends there.

An auto-resetting environment returns the next episode's first observation
when an episode ends, so Ajax takes the ended episode's last observation
from the step's `info` (`final_observation` on gymnax, `final_obs` on Brax
and Playground) and stores it for the bootstrap. Each backend reports the
two flags its own way:

| Backend | Reports | What Ajax does |
| --- | --- | --- |
| gymnax 1.0 | `terminated` and `truncated` apart, an environment of your own included ([Writing one](#writing-one)) | uses them as they are |
| older gymnax-style environments | one `done` that folds in the time limit, which gymnax hands back as `terminated`: every time limit then looks like the end of the task | wrap the environment in `TerminatedTruncatedWrapper`, which keeps `terminated` only where the time limit was not reached |
| Brax and Playground | the time limit in `info["truncation"]`, set by their episode wrapper | reads it there |

The [algorithm pages](../algorithms/index.md) note the agents that treat
episode ends differently, such as the average-reward agents.
