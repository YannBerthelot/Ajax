from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnax import EnvParams

from ajax.environments.system_class import env_params_is_batched
from ajax.types import (
    BraxEnv,
    EnvType,
    GymnaxEnv,
    NormalizationInfo,
)


def check_if_environment_has_continuous_actions(
    env: EnvType,
    env_params: Optional[EnvParams] = None,
) -> bool:
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return True
    # Discriminate on the space *type*, never on its repr: gymnax>=1.0
    # formats Box bounds with ``np.asarray`` in ``__repr__``, which raises
    # when the env builds them with ``jnp.array`` under a trace (this
    # check runs inside the jitted init).
    return "discrete" not in type(env.action_space(env_params)).__name__.lower()


def get_action_dim(env: EnvType, env_params: Optional[EnvParams] = None) -> int:
    """Get the action dimension (continuous case) or the number of action (discrete case) of the environment."""
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return env.action_size

    return (
        env.action_space(env_params).n
        if not check_if_environment_has_continuous_actions(env)
        else env.action_space(env_params).shape[0]
    )


def get_state_action_shapes(
    env: EnvType,
) -> Tuple[tuple, tuple]:
    """Returns the (obs_shape, action_shape) of a gymnax, brax, or gymnasium environment.

    - Discrete action spaces return shape (1,)
    - Shapes are returned as `tuple`s (not multiplied)
    - Works for wrapped environments
    """
    # Gymnax
    if check_env_is_gymnax(env):
        env_params = env.default_params
        obs_space = env.observation_space(env_params)
        act_space = env.action_space(env_params)
        obs_shape = obs_space.shape
        # Discrete check via duck typing
        if hasattr(act_space, "n") and isinstance(act_space.n, int):
            action_shape = (1,)
        else:
            action_shape = act_space.shape
        return obs_shape, action_shape

    # Brax
    if check_env_is_brax(env):
        obs_shape = (env.observation_size,)
        action_shape = (env.action_size,)
        return obs_shape, action_shape

    raise ValueError(f"Unsupported environment type: {type(env)}")


def get_raw_env(env: EnvType) -> EnvType:
    """Get the raw environment from the given environment."""
    if hasattr(env, "_env"):
        return get_raw_env(env._env)
    if hasattr(env, "env"):
        return get_raw_env(env.env)
    return env


def check_env_is_playground(env) -> bool:
    env = get_raw_env(env)
    if "mujoco_playground" in str(type(env)).lower():
        return True
    # Catch external subclasses of mujoco_playground's MjxEnv that live
    # outside the mujoco_playground package (e.g. SafetyExperiments'
    # AntMjx) by walking the MRO and looking for any class declared in the
    # mujoco_playground namespace.
    try:
        for cls in type(env).__mro__:
            if "mujoco_playground" in cls.__module__:
                return True
    except AttributeError:
        pass
    return False


def check_env_is_brax(env) -> bool:
    raw = get_raw_env(env)
    if isinstance(raw, BraxEnv) or "brax" in str(type(raw)).lower():
        return True
    # Treat mujoco_playground envs as brax-compatible: same State API
    # (obs, reward, done, info), same observation_size / action_size, same
    # wrap_for_brax_training stack. The only difference is state.data vs
    # state.pipeline_state, handled where accessed.
    return check_env_is_playground(raw)


def check_env_is_gymnax(env) -> bool:
    env = get_raw_env(env)
    return isinstance(env, GymnaxEnv) or "gymnax" in str(type(env)).lower()


def get_env_type(env: EnvType) -> str:
    """Get the type of the environment.

    Playground envs return "brax": they share the same State API, wrapper
    stack, and training code path. The only divergence is state.data vs
    state.pipeline_state, branched on via check_env_is_playground where needed.
    """
    if check_env_is_brax(env):
        return "brax"
    if check_env_is_gymnax(env):
        return "gymnax"
    raise ValueError(f"Unsupported env type: {type(env)}")


def unnormalize_observation(obs: jax.Array, norm_info: NormalizationInfo) -> jax.Array:
    """Unnormalize the observation using the normalization info."""
    if norm_info is None or norm_info.var is None:
        return obs
    return obs * jnp.sqrt(norm_info.var + 1e-8) + norm_info.mean


def maybe_append_train_frac(
    obs: jax.Array,
    train_frac: Optional[float],
) -> jax.Array:
    """
    Append train_time_fraction as a final observation dimension if provided.
    This is used to give the agent a curriculum signal (how far through
    training we are). Pass train_frac=0.0 for pre-collected expert data
    (collected before training begins) and None to leave obs unchanged.
    """
    if train_frac is None:
        return obs
    new_col = jnp.full((obs.shape[0], 1), train_frac)
    return jnp.concatenate([obs, new_col], axis=-1)


def env_action_repeat(env: EnvType) -> int:
    """Simulator steps one ``env.step`` runs: the ``action_repeat`` of the
    brax ``EpisodeWrapper`` in a brax / playground stack (read through the
    wrappers' attribute forwarding), 1 for gymnax envs and for brax stacks
    without an ``EpisodeWrapper``."""
    if not check_env_is_brax(env):
        return 1
    repeat = getattr(env, "action_repeat", None)
    return 1 if repeat is None else int(repeat)


def check_action_repeat(action_repeat: int) -> None:
    if int(action_repeat) != action_repeat or action_repeat < 1:
        raise ValueError(f"action_repeat must be a positive int, got {action_repeat!r}")


def _n_param_envs(env_params: EnvParams) -> int:
    """Leading (per-env) axis of a batched params (see ``system_class``)."""
    return int(jnp.shape(jax.tree.leaves(env_params)[0])[0])


def agent_episode_length(
    env: EnvType, env_params: Optional[EnvParams], action_repeat: int = 1
) -> int:
    """Episode length in *agent* steps (the T of fixed-length schedules).

    brax / playground: ``env.episode_length // action_repeat``
    (``EpisodeWrapper`` counts simulator steps, see
    ``ajax.environments.create``); the wrapper's own repeat
    (:func:`env_action_repeat`) must equal ``action_repeat``. gymnax:
    ``env_params.max_steps_in_episode // action_repeat`` (the env's own
    params when ``env_params`` is None); per-env (batched) params must
    agree on it, since a fixed-length schedule has one ``T``. The
    simulator-step length must be a multiple of ``action_repeat``,
    otherwise the last agent step of an episode would be cut short; that
    raises.
    """
    check_action_repeat(action_repeat)
    if get_env_type(env) == "brax":
        sim_steps = getattr(env, "episode_length", None)
        if sim_steps is None:
            raise ValueError(
                "Cannot infer the episode length of this brax/playground env:"
                " it has no `episode_length` (no brax EpisodeWrapper in its"
                " stack). Build it with ajax.environments.create."
            )
        env_repeat = env_action_repeat(env)
        if env_repeat != action_repeat:
            raise ValueError(
                f"The env repeats each action {env_repeat} times (its"
                f" EpisodeWrapper), not action_repeat={action_repeat}."
            )
    else:
        params = env_params if env_params is not None else env.default_params
        sim_steps = params.max_steps_in_episode
        if env_params_is_batched(params):
            lengths = np.unique(np.asarray(sim_steps))
            if lengths.size != 1:
                raise ValueError(
                    "The per-env params disagree on max_steps_in_episode"
                    f" ({lengths.tolist()}); an episode length in agent steps"
                    " needs one value."
                )
            sim_steps = lengths[0]
    sim_steps = int(sim_steps)
    if sim_steps % action_repeat:
        raise ValueError(
            f"The episode length ({sim_steps} simulator steps) is not a multiple"
            f" of action_repeat={action_repeat}."
        )
    return sim_steps // action_repeat


def _concrete(x) -> Optional[np.ndarray]:
    """``x`` as a NumPy array, or None when it is a tracer (unknown at trace time)."""
    if isinstance(x, jax.core.Tracer):
        return None
    return np.asarray(x)


def agent_action_to_env(
    action: jax.Array, env: EnvType, env_params: Optional[EnvParams] = None
) -> jax.Array:
    """Map an agent action in ``[-1, 1]`` to the env's action bounds.

    ``action`` is ``[n_envs, *action_shape]``. Continuous actions are
    clipped to ``[-1, 1]`` and, on every dimension of the Box with finite
    bounds ``[low, high]``, mapped affinely: ``low + (clip(a, -1, 1) + 1) /
    2 * (high - low)``. This is DreamerV3's ``ClipAction`` +
    ``NormalizeAction`` (a dimension with an infinite bound keeps the
    clipped action) and TD-MPC2's action scaling. For ``[-1, 1]`` bounds --
    every brax / playground env, gymnax envs like MountainCarContinuous --
    the map is the identity on in-range actions (only the clip acts);
    gymnax Pendulum's torque is ``[-2, 2]``. Per-env (batched) gymnax
    params (``ajax.environments.system_class``) give per-env bounds: env
    ``e``'s action is mapped with its own system's bounds. Discrete actions
    pass through. Replay stores the agent's raw action, not this one.
    """
    if not check_if_environment_has_continuous_actions(env, env_params):
        return action
    clipped = jnp.clip(action, -1.0, 1.0)
    if check_env_is_brax(env):
        return clipped  # brax / playground actions live in [-1, 1]
    if env_params_is_batched(env_params):
        if _n_param_envs(env_params) != action.shape[0]:
            raise ValueError(
                f"Per-env params for {_n_param_envs(env_params)} envs, but"
                f" actions for {action.shape[0]}."
            )

        def bounds(params):
            space = env.action_space(params)
            per_env_shape = action.shape[1:]
            return (
                jnp.broadcast_to(space.low, per_env_shape),
                jnp.broadcast_to(space.high, per_env_shape),
            )

        low, high = jax.vmap(bounds)(env_params)  # [n_envs, *action_shape]
    else:
        space = env.action_space(env_params)
        low, high = space.low, space.high
    concrete_low, concrete_high = _concrete(low), _concrete(high)
    if (
        concrete_low is not None
        and concrete_high is not None
        and bool(np.all(concrete_low == -1.0))
        and bool(np.all(concrete_high == 1.0))
    ):
        return clipped  # exact, without the affine round trip
    low = jnp.asarray(low, dtype=action.dtype)
    high = jnp.asarray(high, dtype=action.dtype)
    finite = jnp.isfinite(low) & jnp.isfinite(high)
    # Substitute [-1, 1] on infinite dims so the arithmetic stays finite;
    # those dims take the clipped action in the final select anyway.
    safe_low = jnp.where(finite, low, -1.0)
    safe_high = jnp.where(finite, high, 1.0)
    scaled = safe_low + (clipped + 1.0) / 2.0 * (safe_high - safe_low)
    return jnp.where(finite, scaled, clipped)
