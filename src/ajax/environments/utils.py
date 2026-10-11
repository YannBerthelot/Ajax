from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnax import EnvParams

from ajax.environments.system_class import env_params_is_batched
from ajax.types import (
    BraxEnv,
    EnvType,
    GymnaxEnv,
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
        return (traced_observation_size(env),), (env.action_size,)

    raise ValueError(f"Unsupported environment type: {type(env)}")


def traced_observation_size(env: BraxEnv) -> Any:
    """``env.observation_size``, evaluated while tracing: brax's
    ``PipelineEnv`` reads it off a reset it runs (seconds of physics on
    CPU), which a trace only stages."""
    size: list = []
    jax.eval_shape(lambda: size.append(env.observation_size))
    return size[0]


def wrapper_chain(env: Any):
    """``env`` and every env it wraps, outermost first.

    Follows the attribute each wrapper stores its inner env in (``_env`` for
    gymnax wrappers, ``env`` for brax / Ajax brax-stack wrappers), read from
    the instance dict so attribute forwarding (``__getattr__``) is never
    mistaken for a wrapped env.
    """
    seen: set = set()
    layer = env
    while layer is not None and id(layer) not in seen:
        seen.add(id(layer))
        yield layer
        attrs = getattr(layer, "__dict__", {})
        layer = attrs.get("_env", attrs.get("env"))


def get_raw_env(env: EnvType) -> EnvType:
    """The env under every wrapper of ``env``."""
    *_, raw = wrapper_chain(env)
    return raw


def env_keys(key: jax.Array, mode: str, n_envs: int) -> jax.Array:
    """One key per env on gymnax (vmapped), a single key on brax / playground."""
    return jax.random.split(key, n_envs) if mode == "gymnax" else key


def per_env(mask: jax.Array, like: jax.Array) -> jax.Array:
    """Reshape an ``[n_envs]`` mask to broadcast against ``like``."""
    return mask.reshape(mask.shape + (1,) * (jnp.ndim(like) - mask.ndim))


def check_env_is_playground(env) -> bool:
    # Catch external subclasses of mujoco_playground's MjxEnv that live
    # outside the mujoco_playground package (e.g. SafetyExperiments'
    # AntMjx) by walking the MRO and looking for any class declared in the
    # mujoco_playground namespace.
    return any(
        "mujoco_playground" in cls.__module__ for cls in type(get_raw_env(env)).__mro__
    )


def check_env_is_brax(env) -> bool:
    raw = get_raw_env(env)
    if isinstance(raw, BraxEnv) or "brax" in str(type(raw)).lower():
        return True
    # Treat mujoco_playground envs as brax-compatible: same State API
    # (obs, reward, done, info), same observation_size / action_size, same
    # wrap_for_brax_training stack. The only difference is state.data vs
    # state.pipeline_state, handled where accessed.
    return check_env_is_playground(raw)


def raw_obs_columns(env: EnvType) -> int:
    """How many leading observation columns ``env`` declares raw (its
    ``raw_obs_columns``, 0 when undeclared): every observation normaliser
    leaves them in the env's own units."""
    return int(getattr(env, "raw_obs_columns", 0))


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


def _action_bounds(
    env: EnvType, env_params: Optional[EnvParams], n_envs: Optional[int] = None
) -> Optional[tuple[jax.Array, jax.Array]]:
    """The centre ``(high + low) / 2`` and half-range ``(high - low) / 2``
    of ``env``'s continuous action box, or None for discrete actions.

    A dimension with an infinite bound gets ``(0, 1)``, as a ``[-1, 1]``
    one. brax / playground actions live in ``[-1, 1]``. Per-env (batched)
    gymnax params (``ajax.environments.system_class``) give each env its
    own box, ``[n_envs, *action_shape]``; ``n_envs``, the number of actions
    they apply to, must match.
    """
    if not check_if_environment_has_continuous_actions(env, env_params):
        return None
    if check_env_is_brax(env):
        n_actions = get_action_dim(env)
        return jnp.zeros(n_actions), jnp.ones(n_actions)

    def box(params):
        space = env.action_space(params)
        return (
            jnp.broadcast_to(space.low, space.shape),
            jnp.broadcast_to(space.high, space.shape),
        )

    if not env_params_is_batched(env_params):
        low, high = box(env_params)
    elif n_envs is not None and _n_param_envs(env_params) != n_envs:
        raise ValueError(
            f"Per-env params for {_n_param_envs(env_params)} envs, but"
            f" actions for {n_envs}."
        )
    else:
        low, high = jax.vmap(box)(env_params)
    finite = jnp.isfinite(low) & jnp.isfinite(high)
    return (
        jnp.where(finite, (high + low) / 2, 0.0),
        jnp.where(finite, (high - low) / 2, 1.0),
    )


def agent_action_to_env(
    action: jax.Array, env: EnvType, env_params: Optional[EnvParams] = None
) -> jax.Array:
    """Map an agent action in ``[-1, 1]`` to the env's action bounds.

    Every Ajax env step carrying an agent's or an expert's action calls it:
    every continuous agent acts in ``[-1, 1]`` (squashed or not), whatever
    the env's box. ``action`` is ``[n_envs, *action_shape]``. Continuous
    actions are clipped to ``[-1, 1]`` and, on every dimension of the Box
    with finite bounds ``[low, high]``, mapped affinely onto them: ``c + h
    * clip(a, -1, 1)``, with centre ``c = (high + low) / 2`` and half-range
    ``h = (high - low) / 2`` (CleanRL's ``action_bias`` and
    ``action_scale``).
    This is DreamerV3's ``ClipAction`` + ``NormalizeAction`` (a dimension
    with an infinite bound keeps the clipped action) and TD-MPC2's action
    scaling. For ``[-1, 1]`` bounds -- every brax / playground env, gymnax
    envs like MountainCarContinuous -- the map is exactly the clip, even
    when the bounds are traced; gymnax Pendulum's torque is ``[-2, 2]``.
    Per-env (batched) gymnax params (``ajax.environments.system_class``)
    give per-env bounds: env ``e``'s action is mapped with its own system's
    bounds. Discrete actions pass through. Replay stores the agent's raw
    action, not this one. :func:`env_action_to_agent` is the inverse.
    """
    bounds = _action_bounds(env, env_params, action.shape[0])
    if bounds is None:
        return action
    centre, half = (jnp.asarray(x, dtype=action.dtype) for x in bounds)
    return centre + half * jnp.clip(action, -1.0, 1.0)


def env_action_to_agent(
    action: jax.Array, env: EnvType, env_params: Optional[EnvParams] = None
) -> jax.Array:
    """Map an action in the env's bounds to agent units, ``[-1, 1]``.

    The inverse of :func:`agent_action_to_env` on the env's box, ``(a - c)
    / h``, with the same ``action`` layout and per-env bounds. It does not
    clip: an action outside the box stays outside ``[-1, 1]``. It is the
    identity on ``[-1, 1]`` boxes, on a dimension with an infinite bound
    and on discrete actions.
    """
    return _to_agent_units(action, _action_bounds(env, env_params, action.shape[0]))


def _to_agent_units(action: jax.Array, bounds: Optional[tuple[Any, Any]]) -> jax.Array:
    """``(action - centre) / half``, given :func:`_action_bounds`'s output."""
    if bounds is None:
        return action
    centre, half = (jnp.asarray(x, dtype=action.dtype) for x in bounds)
    return (action - centre) / half


class ExpertInAgentUnits:
    """An expert written in env units, read in agent units (``[-1, 1]``).

    What the expert returns -- ``expert(obs)``, a stateful ``expert(state,
    obs)``'s action, ``step_with_gains``'s action -- goes through
    :func:`env_action_to_agent`'s map, unclipped. ``init_state``,
    ``learnable_fields``, ``anchor_gains`` and ``step_with_gains`` exist
    exactly when the expert's do (``hasattr`` still tells a stateful
    expert). It hashes by identity, for jit's static arguments.

    Its bounds are one ``(A,)`` box read from concrete params, also inside
    jit (static params): a stateless expert is also vmapped over single
    observations, where per-env bounds would broadcast its action. Per-env
    boxes that differ raise.
    """

    __slots__ = ("expert", "_bounds")
    _FORWARDED = frozenset({"init_state", "learnable_fields", "anchor_gains"})

    def __init__(
        self, expert: Any, env: EnvType, env_params: Optional[EnvParams] = None
    ) -> None:
        with jax.ensure_compile_time_eval():
            bounds = _action_bounds(env, env_params)
        if bounds is not None:
            bounds = tuple(np.asarray(x) for x in bounds)
        if bounds is not None and env_params_is_batched(env_params):
            if any(np.any(x != x[0]) for x in bounds):
                raise ValueError(
                    "The per-env params have different action bounds: an"
                    " expert's actions cannot be read in agent units without"
                    " the env each one comes from (a replayed row has none)."
                )
            bounds = tuple(x[0] for x in bounds)
        self.expert = expert
        self._bounds = bounds

    @classmethod
    def wrap(
        cls, expert: Any, env: EnvType, env_params: Optional[EnvParams] = None
    ) -> Any:
        """``expert`` in agent units; None and a wrapped expert stay as they are."""
        if expert is None or isinstance(expert, cls):
            return expert
        return cls(expert, env, env_params)

    def __call__(self, *args: Any) -> Any:
        out = self.expert(*args)
        # The call, not init_state, says what came back: an expert with a
        # state may also be called on observations alone (expert(obs)).
        if len(args) == 2:  # expert(state, obs) -> (action, state)
            return _to_agent_units(out[0], self._bounds), out[1]
        return _to_agent_units(out, self._bounds)

    def __getattr__(self, name: str) -> Any:
        if name in ExpertInAgentUnits._FORWARDED:
            return getattr(self.expert, name)
        if name == "step_with_gains":
            step = self.expert.step_with_gains

            def step_with_gains(state: Any, obs: Any, gains: Any) -> Any:
                action, state = step(state, obs, gains)
                return _to_agent_units(action, self._bounds), state

            return step_with_gains
        raise AttributeError(name)
