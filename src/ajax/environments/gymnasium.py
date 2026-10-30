"""Gymnasium environments, stepped in Python from Ajax's compiled training.

Ajax compiles the environment and the agent into one program, which is why it
trains on JAX environments. A Gymnasium environment steps in Python instead:
``GymnasiumEnv`` gives it gymnax's interface, so every agent trains on it
unchanged, and its ``reset`` and ``step`` leave the compiled program through
``jax.experimental.io_callback`` (JAX's callback for code with side effects,
which the compiler never skips, merges or repeats). EnvPool's XLA interface
steps its C++ environments from compiled code the same way.

One Python call steps every copy: a ``custom_vmap`` rule flattens each
``vmap`` around the callback into one batch, and a copy's place in that batch
is its slot in the pool of Gymnasium environments. JAX cannot batch such a
callback inside a ``lax.cond``, where Ajax's training runs its evaluations, so
the seeds of a run train one after another (``ajax.agents.base``).

The pool lives in Python, outside the JAX state: evaluation steps copies of
its own (:func:`with_own_copies`), a saved state cannot resume
the episodes it was in, gradients cannot flow through a step, and resets
follow each copy's own Gymnasium generator, seeded at its reset from the
run's key, rather than Ajax's episode keys.
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax.custom_batching import custom_vmap
from jax.experimental import io_callback


class PythonEnvironmentWarning(UserWarning):
    """A Gymnasium environment steps in Python, outside the compiled training."""


@dataclass(frozen=True)
class JaxCounterpart:
    """A JAX task close to a Gymnasium one, and what differs between them."""

    backend: str
    env_ids: tuple[str, ...]
    differences: str


_BRAX_PHYSICS = "Brax's own physics (Euler steps, fewer colliding bodies)"
_DM_CONTROL = (
    "a DeepMind Control task: another model, a reward between 0 and 1, no early"
    " termination"
)
# The Gymnasium task each entry was compared with, and its closest JAX tasks.
# Read in the source of Gymnasium 1.2.0, Brax 0.14.2, MuJoCo Playground 0.2 and
# gymnax 1.0.1: tasks of the same name do not share the same rules.
# fmt: off
COUNTERPARTS: dict[str, tuple[str, tuple[JaxCounterpart, ...]]] = {
    "HalfCheetah": ("HalfCheetah-v5", (
        JaxCounterpart("Brax", ("halfcheetah",), f"same reward, observation and reset; {_BRAX_PHYSICS}"),
        JaxCounterpart("MuJoCo Playground", ("CheetahRun",), _DM_CONTROL),
    )),
    "Hopper": ("Hopper-v5", (
        JaxCounterpart("Brax", ("hopper",), f"same reward weights and health ranges; {_BRAX_PHYSICS}; the healthy reward is also paid on the step that ends the episode"),
        JaxCounterpart("MuJoCo Playground", ("HopperHop", "HopperStand"), _DM_CONTROL),
    )),
    "Walker2d": ("Walker2d-v5", (
        JaxCounterpart("Brax", ("walker2d",), f"same reward weights and health ranges; {_BRAX_PHYSICS}; the healthy reward is also paid on the step that ends the episode"),
        JaxCounterpart("MuJoCo Playground", ("WalkerWalk", "WalkerRun"), _DM_CONTROL),
    )),
    "Ant": ("Ant-v5", (
        JaxCounterpart("Brax", ("ant",), f"no contact cost; 27 observations instead of 105; {_BRAX_PHYSICS}"),
    )),
    "Humanoid": ("Humanoid-v5", (
        JaxCounterpart("Brax", ("humanoid",), f"no contact cost; other observations (244 instead of 348); {_BRAX_PHYSICS}"),
        JaxCounterpart("MuJoCo Playground", ("HumanoidWalk", "HumanoidRun"), _DM_CONTROL),
    )),
    "HumanoidStandup": ("HumanoidStandup-v5", (
        JaxCounterpart("Brax", ("humanoidstandup",), "a height reward 5 times smaller, another control cost, no impact cost, other observations"),
        JaxCounterpart("MuJoCo Playground", ("HumanoidStand",), _DM_CONTROL),
    )),
    "InvertedPendulum": ("InvertedPendulum-v5", (
        JaxCounterpart("Brax", ("inverted_pendulum",), f"the reward is also paid on the step that ends the episode; {_BRAX_PHYSICS}"),
        JaxCounterpart("MuJoCo Playground", ("CartpoleBalance",), _DM_CONTROL),
    )),
    "InvertedDoublePendulum": ("InvertedDoublePendulum-v5", (
        JaxCounterpart("Brax", ("inverted_double_pendulum",), "a control step of 0.02 s instead of 0.05 s; reset noise 10 times smaller; 8 observations instead of 9"),
    )),
    "Reacher": ("Reacher-v5", (
        JaxCounterpart("Brax", ("reacher",), "1,000-step episodes by default instead of 50 (episode_length=50 matches); other target placement and observations"),
        JaxCounterpart("MuJoCo Playground", ("ReacherEasy", "ReacherHard"), f"{_DM_CONTROL}; the reward is 0 or 1"),
    )),
    "Pusher": ("Pusher-v5", (
        JaxCounterpart("Brax", ("pusher",), "the reward is computed from the state before the step; another fingertip point and object placement"),
    )),
    "Swimmer": ("Swimmer-v5", (
        JaxCounterpart("Brax", ("swimmer",), f"same reward, observation and termination; {_BRAX_PHYSICS}"),
        JaxCounterpart("MuJoCo Playground", ("SwimmerSwimmer6",), f"{_DM_CONTROL}; reach a target with 6 links"),
    )),
    "CartPole": ("CartPole-v1", (JaxCounterpart("gymnax", ("CartPole-v1",), "the same task, in float32"),)),
    "Pendulum": ("Pendulum-v1", (JaxCounterpart("gymnax", ("Pendulum-v1",), "the same task, in float32"),)),
    "Acrobot": ("Acrobot-v1", (JaxCounterpart("gymnax", ("Acrobot-v1",), "the same task, in float32"),)),
    "MountainCar": ("MountainCar-v0", (JaxCounterpart("gymnax", ("MountainCar-v0",), "the same task, in float32"),)),
    "MountainCarContinuous": ("MountainCarContinuous-v0", (
        JaxCounterpart("gymnax", ("MountainCarContinuous-v0",), "the car is not stopped at the left wall, and the goal also counts when reached moving left"),
    )),
}
# fmt: on
_EXTRAS = {"Brax": " (ajax-rl[brax])", "MuJoCo Playground": " (ajax-rl[playground])"}


def warning_text(env_id: str) -> str:
    """What :class:`PythonEnvironmentWarning` says for ``env_id``."""
    text = (
        f"Gymnasium's {env_id} runs in Python, outside Ajax's compiled training"
        " program. Every step goes from the compiled program to Python and"
        " back, so training is slower than on a JAX environment and gains"
        " nothing from a GPU, and seeds train one after another. Episode keys"
        " and resuming from a saved state do not apply."
    )
    _, name, _ = gymnasium.envs.registration.parse_env_id(env_id.rpartition(":")[2])
    if name not in COUNTERPARTS:
        return text
    compared_with, counterparts = COUNTERPARTS[name]
    lines = [f"\nClosest JAX tasks, compared with {compared_with}:"]
    for counterpart in counterparts:
        ids = ", ".join(f'"{task_id}"' for task_id in counterpart.env_ids)
        extra = _EXTRAS.get(counterpart.backend, "")
        lines.append(
            f"- {counterpart.backend} {ids}{extra}: {counterpart.differences}."
        )
    return text + "\n".join(lines)


@struct.dataclass
class GymnasiumState:
    slot: jax.Array  # the copy's place in the pool
    obs: jax.Array  # its last observation, which get_obs returns


@struct.dataclass
class GymnasiumParams(environment.EnvParams):
    max_steps_in_episode: int = 1000


def _one_call_per_batch(host_call: Callable, result_shapes: Callable) -> Callable:
    """``host_call`` on a batch (a leading axis on every argument), as a
    function whose every ``vmap`` level reaches it as that one batch, flattened.
    ``result_shapes(batch_size)`` gives its results' shapes and dtypes."""

    @custom_vmap
    def call(*args):
        return io_callback(host_call, result_shapes(args[0].shape[0]), *args)

    @call.def_vmap
    def batched(axis_size, in_batched, *args):
        args = [
            arg if is_batched else jnp.broadcast_to(arg, (axis_size, *arg.shape))
            for arg, is_batched in zip(args, in_batched, strict=True)
        ]
        flat = call(*[arg.reshape((-1, *arg.shape[2:])) for arg in args])
        out = tuple(x.reshape((axis_size, -1, *x.shape[1:])) for x in flat)
        return out, (True,) * len(out)

    return call


class _Copies:
    """The Gymnasium environments one :class:`GymnasiumEnv` steps, by slot."""

    def __init__(self, make: Callable[[], gymnasium.Env], discrete: bool):
        self.make = make
        self.discrete = discrete
        self.envs: list = []

    def reset(self, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        while len(self.envs) < len(seeds):
            self.envs.append(self.make())
        envs = self.envs[: len(seeds)]
        obs = [
            env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds, strict=True)
        ]
        return np.arange(len(seeds), dtype=np.int32), _flat(obs)

    def step(self, slots: np.ndarray, actions: np.ndarray) -> tuple:
        if np.max(slots) >= len(self.envs):
            raise RuntimeError("A Gymnasium environment was stepped before its reset.")
        obs, rewards, terminated, truncated, final_obs = [], [], [], [], []
        for slot, action in zip(slots, actions, strict=True):
            env = self.envs[slot]
            action = int(action) if self.discrete else np.asarray(action)
            step_obs, reward, step_terminated, step_truncated, _ = env.step(action)
            final_obs.append(step_obs)
            # Gymnasium's own auto-reset happens a step later; Ajax's (and
            # gymnax's) returns the next episode's first observation at once.
            if step_terminated or step_truncated:
                step_obs, _ = env.reset()
            obs.append(step_obs)
            rewards.append(reward)
            terminated.append(step_terminated)
            truncated.append(step_truncated)
        return (
            _flat(obs),
            np.asarray(rewards, np.float32),
            np.asarray(terminated, bool),
            np.asarray(truncated, bool),
            _flat(final_obs),
        )


def _flat(observations: list) -> np.ndarray:
    return np.stack([np.asarray(obs, np.float32).reshape(-1) for obs in observations])


class GymnasiumEnv(environment.Environment):
    """A Gymnasium environment, ``gymnasium.make(env_id, **make_kwargs)``,
    with gymnax's interface. ``"gymnasium:<env_id>"`` builds one by id.

    It needs a time limit (``max_episode_steps``), a Box observation space
    (flattened) and a Discrete or one-dimensional Box action space.
    """

    steps_in_python = True
    # Its resets happen in Python: Ajax's episode keys cannot key them.
    _ajax_episode_keys = False

    def __init__(self, env_id: str, **make_kwargs: Any):
        super().__init__()
        self.env_id = env_id

        def make() -> gymnasium.Env:
            try:
                return gymnasium.make(env_id, **make_kwargs)
            except gymnasium.error.DependencyNotInstalled as error:
                raise ImportError(
                    f"{env_id} needs Ajax's gymnasium extra: pip install"
                    f" 'ajax-rl[gymnasium]' ({error})"
                ) from error

        sample = make()
        max_steps = sample.spec.max_episode_steps if sample.spec else None
        observation_space, action_space = sample.observation_space, sample.action_space
        sample.close()
        if max_steps is None:
            raise ValueError(
                f"{env_id} has no time limit; evaluation runs a fixed number of"
                f" steps. Pass one: GymnasiumEnv({env_id!r}, max_episode_steps=...)"
            )
        self.max_steps = int(max_steps)
        if not isinstance(observation_space, gymnasium.spaces.Box):
            raise ValueError(
                f"{env_id}'s observations are {observation_space}; Ajax's"
                " networks take a Box of numbers."
            )
        self.obs_size = int(np.prod(observation_space.shape))
        self._observation_space = spaces.Box(
            np.asarray(observation_space.low, np.float32).reshape(-1),
            np.asarray(observation_space.high, np.float32).reshape(-1),
            (self.obs_size,),
        )
        if (
            isinstance(action_space, gymnasium.spaces.Discrete)
            and action_space.start == 0
        ):
            self._action_space: Any = spaces.Discrete(int(action_space.n))
        elif (
            isinstance(action_space, gymnasium.spaces.Box)
            and len(action_space.shape) == 1
        ):
            self._action_space = spaces.Box(
                np.asarray(action_space.low, np.float32),
                np.asarray(action_space.high, np.float32),
                action_space.shape,
            )
        else:
            raise ValueError(
                f"{env_id}'s actions are {action_space}; Ajax acts in a Discrete"
                " space starting at 0 or a one-dimensional Box."
            )
        self._copies = _Copies(make, isinstance(self._action_space, spaces.Discrete))
        warnings.warn(warning_text(env_id), PythonEnvironmentWarning, stacklevel=2)

    @property
    def default_params(self) -> GymnasiumParams:
        return GymnasiumParams(max_steps_in_episode=self.max_steps)

    @property
    def name(self) -> str:
        return self.env_id

    # The methods below take gymnax's signatures; the Gymnasium environment
    # has no gymnax params and draws its own randomness.
    def observation_space(
        self,
        params: Optional[GymnasiumParams] = None,  # noqa: ARG002
    ) -> spaces.Box:
        return self._observation_space

    def action_space(self, params: Optional[GymnasiumParams] = None) -> Any:  # noqa: ARG002
        return self._action_space

    def get_obs(self, state: GymnasiumState, params=None, key=None) -> jax.Array:  # noqa: ARG002
        return state.obs

    def reset(self, key: jax.Array, params=None) -> tuple[jax.Array, GymnasiumState]:  # noqa: ARG002
        # Built at each trace from this env's pool, which with_own_copies
        # replaces on a copy.
        reset_copies = _one_call_per_batch(
            self._copies.reset,
            lambda n: (
                jax.ShapeDtypeStruct((n,), jnp.int32),
                jax.ShapeDtypeStruct((n, self.obs_size), jnp.float32),
            ),
        )
        seed = jax.random.bits(key, dtype=jnp.uint32)
        slots, obs = reset_copies(seed[None])
        return obs[0], GymnasiumState(slot=slots[0], obs=obs[0])

    def step(
        self,
        key: jax.Array,  # noqa: ARG002
        state: GymnasiumState,
        action: jax.Array,
        params=None,  # noqa: ARG002
    ):
        def shapes(n: int) -> tuple:
            obs = jax.ShapeDtypeStruct((n, self.obs_size), jnp.float32)
            flag = jax.ShapeDtypeStruct((n,), jnp.bool_)
            return obs, jax.ShapeDtypeStruct((n,), jnp.float32), flag, flag, obs

        step_copies = _one_call_per_batch(self._copies.step, shapes)
        out = step_copies(state.slot[None], jnp.asarray(action)[None])
        obs, reward, terminated, truncated, final_obs = (x[0] for x in out)
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": final_obs,
        }
        next_state = GymnasiumState(slot=state.slot, obs=obs)
        return obs, next_state, reward, terminated, truncated, info


def with_own_copies(env: Any) -> Any:
    """``env`` with every :class:`GymnasiumEnv` under its wrappers given a
    pool of its own, so that stepping it (evaluation does) leaves ``env``'s
    episodes where they are. Other environments come back as they are."""
    if isinstance(env, GymnasiumEnv):
        separate = copy.copy(env)
        separate._copies = _Copies(env._copies.make, env._copies.discrete)
        return separate
    inner = getattr(env, "__dict__", {}).get("_env")
    if inner is None:
        return env
    rewrapped = copy.copy(env)
    rewrapped._env = with_own_copies(inner)
    return rewrapped
