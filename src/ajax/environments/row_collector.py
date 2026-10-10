"""Row collector: one observation-aligned row per env per tick.

The collector the world-model agents share (docs/world_models/DESIGN.md
§5.2). Each call (one *tick*) emits one :class:`Row` per env and advances
the envs by at most one agent step. Rows are obs-aligned, DreamerV3's
replay convention, which is also exactly TD-MPC2's ``T + 1`` rows per
episode of ``T`` steps:

==============  ===========================================================
field           meaning
==============  ===========================================================
``obs``         observation acted on at this tick
``reward``      reward received on entering ``obs`` (0 when ``is_first``)
``is_first``    ``obs`` is the reset observation of an episode
``is_last``     ``obs`` is the final observation of an episode
``is_terminal`` the episode ended by a true termination (not a time limit)
``action``      the agent's raw action at this tick, zeroed when ``is_last``
``extras``      agent-defined per-row outputs returned by the policy
==============  ===========================================================

Deferred reset. When ``env.step`` ends an episode, the *next* tick emits
the terminal observation (``get_final_obs``) as ``is_last`` with
``is_terminal = terminated``; on that tick the env is not stepped (it is
*held*), and the tick after emits the reset observation with
``is_first = 1`` and reward 0. The policy is called on every tick,
including ``is_last`` ones (a recurrent policy filters the final
observation too); its action there is discarded.

Two reset modes, chosen by the caller:

* ``"static"`` -- fixed-length lockstep episodes of ``T`` agent steps.
  The held tick is ``tick mod (T + 1) == T`` for every env, computed from
  the *unbatched* absolute tick index the caller passes in. On it a fresh
  ``env.reset`` with a fresh key replaces the env state, inside a
  ``lax.cond`` on that unbatched predicate (a real cond under the seed
  ``vmap``, so the reset only runs on held ticks). Every episode therefore
  starts from a freshly randomised initial state on every backend,
  including a mujoco_playground env built with ``fresh_reset=False``,
  whose auto-reset returns a cached first state (deviation E20). Episode
  ends are the schedule's; a ``done`` the
  env reports off the schedule is counted in
  ``RowCollectorState.n_offschedule_dones`` (an error for the caller to
  surface: the env is then auto-reset mid-episode).
* ``"dynamic"`` -- data-dependent episode ends (tasks with terminations).
  Every env is stepped and the held envs are restored per env with
  ``jnp.where`` from a snapshot of the env state taken before the step
  (except the batch-shared reset seed of Ajax's brax ``AutoResetWrapper``,
  which keeps its stepped value, see :func:`_dynamic_transition`).
  The reset observation is the backend's auto-reset observation, stashed
  when the episode ended: fresh on gymnax, brax and mujoco_playground
  (Ajax's default ``FreshAutoResetWrapper``), a cached first state on a
  playground env built with ``fresh_reset=False`` (deviation E20; a
  warning is issued).

Random-action phase. The caller may pass an *unbatched* boolean
``random_phase`` (derived from the tick, so it stays a real ``lax.cond``
under the seed ``vmap``); when it is set the policy is not called, the
action is uniform (``U[-1, 1]^A`` continuous, ``randint`` discrete), the
extras are zeros and the policy carry is left unchanged.

The action sent to the env is mapped to the env's bounds
(:func:`ajax.environments.utils.agent_action_to_env`); the row keeps the
raw agent action.

The tick is a pure function of ``(state, tick)``: usable inside
``lax.scan``, jittable and vmappable over seeds. gymnax envs get one key
per env (``vmap``) and may carry per-env (batched) params
(:mod:`ajax.environments.system_class`); brax / playground envs are
natively batched and take a single key, exactly as in
:mod:`ajax.environments.interaction`.

Precondition: the env stack does not normalise observations or rewards
(:func:`check_unnormalized_env`, enforced by
:func:`init_row_collector_state`). Rows store the env's observations and
rewards as stepped, the house return metric sums those rewards, and
:func:`ajax.evaluate.evaluate_policy` evaluates the raw env.
"""

import warnings
from functools import partial
from typing import Any, Literal, NamedTuple, Optional, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.environments.interaction import (
    get_final_obs,
    init_rolling_mean,
    reset,
    step,
    update_episodic_return,
)
from ajax.environments.utils import (
    agent_action_to_env,
    check_env_is_playground,
    check_if_environment_has_continuous_actions,
    get_action_dim,
    get_env_type,
)
from ajax.state import CollectorState, EnvironmentConfig
from ajax.wrappers import (
    AutoResetWrapper,
    NormalizeVecObservationBrax,
    NormalizeVecObservationGymnax,
)

ResetMode = Literal["static", "dynamic"]
TimestepUnit = Literal["rows", "env_steps"]


class PolicyFn(Protocol):
    """``(carry, obs, is_first, key) -> (action, carry, extras)``.

    ``obs`` is ``[n_envs, *obs_shape]`` and ``is_first`` ``[n_envs]``; a
    stateful policy resets its own carry where ``is_first`` is set. The
    action is the agent's raw action (``[-1, 1]`` for continuous control);
    ``extras`` is any pytree with a leading ``n_envs`` axis (``None`` for
    none), stored on the row.
    """

    def __call__(
        self, carry: Any, obs: jax.Array, is_first: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, Any, Any]: ...


@struct.dataclass
class Row:
    """One obs-aligned row per env (leading axis ``n_envs``)."""

    obs: jax.Array
    reward: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    action: jax.Array
    extras: Any = None


@partial(struct.dataclass, kw_only=True)
class RowCollectorState(CollectorState):
    """:class:`CollectorState` plus the row collector's bookkeeping.

    ``last_obs`` is the observation the next row emits; ``reward`` /
    ``is_first`` / ``is_last`` / ``is_terminal`` are that row's flags.
    ``timestep`` advances by the unit the caller chose (``"rows"``:
    ``n_envs`` per tick, held ticks included; ``"env_steps"``: the number of
    envs actually stepped). ``episodic_return_state`` /
    ``episodic_mean_return`` follow the house rolling-mean convention
    (``Train/episodic mean reward``); ``step_in_episode`` counts the agent
    steps of each env's current episode and ``last_episode_length`` keeps
    the length of its last finished one. ``max_timesteps`` stays None: the
    env state is used as is (no ``train_time_fraction`` view).
    """

    reward: jax.Array  # [n_envs] f32
    is_first: jax.Array  # [n_envs] bool
    is_last: jax.Array  # [n_envs] bool
    is_terminal: jax.Array  # [n_envs] bool
    reset_obs: jax.Array  # [n_envs, *obs] auto-reset obs stash (dynamic mode)
    env_steps: jax.Array  # [] i32, agent steps taken, summed over envs
    rows: jax.Array  # [] i32, rows emitted, summed over envs
    n_offschedule_dones: jax.Array  # [] i32 (static mode)
    last_episode_length: jax.Array  # [n_envs] i32, of the last finished one
    policy_carry: Any = None


def _wrapper_chain(env: Any):
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


def check_unnormalized_env(env: Any, consumer: str) -> None:
    """Raise if ``env``'s wrapper stack normalises observations or rewards.

    The row collector stores the env's observations and rewards as stepped
    (and sums those rewards into the house ``Train/episodic mean reward``),
    and :func:`ajax.evaluate.evaluate_policy` rebuilds and evaluates the raw
    env: with a normalising wrapper the rows and the training metric would
    be normalised while evaluation is not (``prepare_env``'s normalisation
    stack also clips actions to ``[-1, 1]``, defeating the bound mapping).
    """
    normalizing = (NormalizeVecObservationBrax, NormalizeVecObservationGymnax)
    for layer in _wrapper_chain(env):
        if isinstance(layer, normalizing):
            raise ValueError(
                f"{consumer} stores and evaluates the env's raw observations and"
                f" rewards, but the env is wrapped in {type(layer).__name__}"
                " (observation / reward normalisation). Build the env without"
                " normalisation (normalize_observations=False,"
                " normalize_rewards=False)."
            )


def init_row_collector_state(
    key: jax.Array,
    env_args: EnvironmentConfig,
    policy_carry: Any = None,
    window_size: int = 10,
) -> RowCollectorState:
    """Reset the envs; the first row of every env is ``is_first``.

    Raises if the env stack normalises observations or rewards (module
    docstring).
    """
    env, n_envs = env_args.env, env_args.n_envs
    check_unnormalized_env(env, "The row collector")
    mode = get_env_type(env)
    reset_key, rng = jax.random.split(key)
    obs, env_state = reset(
        _env_keys(reset_key, mode, n_envs), env, mode, env_args.env_params
    )
    zeros_f = jnp.zeros(n_envs, jnp.float32)
    zeros_b = jnp.zeros(n_envs, bool)
    zeros_i = jnp.zeros(n_envs, jnp.int32)
    scalar_zero = jnp.zeros((), jnp.int32)
    return RowCollectorState(
        rng=rng,
        _env_state=env_state,
        last_obs=obs,
        last_terminated=zeros_f,
        last_truncated=zeros_f,
        episodic_return_state=init_rolling_mean(
            window_size=window_size,
            last_return=jnp.full((n_envs, 1), jnp.nan),
            cumulative_reward=jnp.zeros((n_envs, 1)),
        ),
        episodic_mean_return=jnp.asarray(jnp.nan, jnp.float32),
        timestep=scalar_zero,
        reward=zeros_f,
        is_first=jnp.ones(n_envs, bool),
        is_last=zeros_b,
        is_terminal=zeros_b,
        reset_obs=obs,
        env_steps=scalar_zero,
        rows=scalar_zero,
        n_offschedule_dones=scalar_zero,
        step_in_episode=zeros_i,
        last_episode_length=zeros_i,
        policy_carry=policy_carry,
    )


class _Next(NamedTuple):
    """What a tick hands to the next one (per env unless noted)."""

    env_state: Any
    obs: jax.Array
    reward: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    reset_obs: jax.Array
    stepped: jax.Array  # the env advanced one agent step this tick
    n_offschedule_dones: jax.Array  # [] i32, this tick's


def collect_row(
    collector_state: RowCollectorState,
    tick: jax.Array,
    policy_fn: PolicyFn,
    *,
    env_args: EnvironmentConfig,
    reset_mode: ResetMode,
    episode_length: Optional[int] = None,
    random_phase: Optional[jax.Array] = None,
    timestep_unit: TimestepUnit = "rows",
) -> tuple[RowCollectorState, Row]:
    """Emit one row per env and advance the envs by one tick.

    Args:
        collector_state: from :func:`init_row_collector_state` or the
            previous tick.
        tick: the *unbatched* absolute tick index (the scan input, offset
            on resume). Drives the static schedule.
        policy_fn: see :class:`PolicyFn`; called with the carry stored on
            the state.
        env_args: the training env (static).
        reset_mode: ``"static"`` or ``"dynamic"`` (module docstring).
        episode_length: ``T`` in agent steps, required in static mode
            (:func:`ajax.environments.utils.agent_episode_length`).
        random_phase: optional unbatched boolean; when true, act uniformly
            at random instead of calling the policy. ``None`` (no random
            phase) does not trace the random branch at all.
        timestep_unit: ``"rows"`` or ``"env_steps"``, what
            ``collector_state.timestep`` counts.

    Returns:
        ``(new_state, row)``.
    """
    if reset_mode not in ("static", "dynamic"):
        raise ValueError(
            f"reset_mode must be 'static' or 'dynamic', got {reset_mode!r}"
        )
    if timestep_unit not in ("rows", "env_steps"):
        raise ValueError(
            f"timestep_unit must be 'rows' or 'env_steps', got {timestep_unit!r}"
        )
    if reset_mode == "static" and (episode_length is None or episode_length < 1):
        raise ValueError("static reset mode needs episode_length >= 1 (agent steps)")

    cs = collector_state
    env, env_params, n_envs = env_args.env, env_args.env_params, env_args.n_envs
    mode = get_env_type(env)
    rng, policy_key, random_key, step_key, reset_key = jax.random.split(cs.rng, 5)

    action, policy_carry, extras = _act(
        policy_fn,
        cs.policy_carry,
        cs.last_obs,
        cs.is_first,
        policy_key,
        random_key,
        random_phase,
        env_args,
    )
    action = jnp.where(_per_env(cs.is_last, action), jnp.zeros_like(action), action)
    row = Row(
        obs=cs.last_obs,
        reward=cs.reward,
        is_first=cs.is_first,
        is_last=cs.is_last,
        is_terminal=cs.is_terminal,
        action=action,
        extras=extras,
    )
    env_action = agent_action_to_env(action, env, env_params)
    step_keys = _env_keys(step_key, mode, n_envs)

    if reset_mode == "static":
        assert episode_length is not None  # checked above
        nxt = _static_transition(
            cs,
            jnp.asarray(tick),
            episode_length,
            env_action,
            step_keys,
            _env_keys(reset_key, mode, n_envs),
            env_args,
            mode,
        )
    else:
        if check_env_is_playground(env) and not getattr(env, "_ajax_fresh_reset", True):
            warnings.warn(
                "Dynamic reset mode on a mujoco_playground env built with"
                " fresh_reset=False: its auto-reset returns a cached first"
                " state, so episodes after the first do not start from a fresh"
                " random state (deviation E20). Build the env with the default"
                " fresh_reset=True, or use the static mode on fixed-length"
                " tasks.",
                stacklevel=2,
            )
        nxt = _dynamic_transition(cs, env_action, step_keys, env_args, mode)

    ended = nxt.is_last & nxt.stepped
    step_reward = jnp.where(nxt.stepped, nxt.reward, 0.0)
    # Episode returns: the house rolling mean, over the steps actually taken.
    # Rewards are the env's as stepped: raw, since the env stack does not
    # normalise them (checked by init_row_collector_state).
    episodic_return_state, episodic_mean_return = update_episodic_return(
        cs.episodic_return_state, step_reward, ended
    )
    length = cs.step_in_episode + nxt.stepped.astype(jnp.int32)
    env_steps = cs.env_steps + jnp.sum(nxt.stepped, dtype=jnp.int32)
    rows = cs.rows + n_envs
    new_state = cs.replace(
        rng=rng,
        _env_state=nxt.env_state,
        last_obs=nxt.obs,
        last_terminated=nxt.is_terminal.astype(jnp.float32),
        last_truncated=(nxt.is_last & ~nxt.is_terminal).astype(jnp.float32),
        episodic_return_state=episodic_return_state,
        episodic_mean_return=episodic_mean_return,
        timestep=rows if timestep_unit == "rows" else env_steps,
        reward=nxt.reward,
        is_first=nxt.is_first,
        is_last=nxt.is_last,
        is_terminal=nxt.is_terminal,
        reset_obs=nxt.reset_obs,
        env_steps=env_steps,
        rows=rows,
        n_offschedule_dones=cs.n_offschedule_dones + nxt.n_offschedule_dones,
        step_in_episode=jnp.where(ended, 0, length),
        last_episode_length=jnp.where(ended, length, cs.last_episode_length),
        policy_carry=policy_carry,
    )
    return new_state, row


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


def _act(
    policy_fn: PolicyFn,
    carry: Any,
    obs: jax.Array,
    is_first: jax.Array,
    policy_key: jax.Array,
    random_key: jax.Array,
    random_phase: Optional[jax.Array],
    env_args: EnvironmentConfig,
) -> tuple[jax.Array, Any, Any]:
    def act_with_policy(carry):
        return policy_fn(carry, obs, is_first, policy_key)

    if random_phase is None:
        return act_with_policy(carry)

    out_struct = jax.eval_shape(act_with_policy, carry)

    def act_randomly(carry):
        action = _uniform_action(random_key, out_struct[0], env_args)
        extras = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), out_struct[2])
        return action, carry, extras

    return jax.lax.cond(random_phase, act_randomly, act_with_policy, carry)


def _uniform_action(
    key: jax.Array, action_struct: jax.ShapeDtypeStruct, env_args: EnvironmentConfig
) -> jax.Array:
    """``U[-1, 1]`` per dimension (continuous) or ``randint(0, n)`` (discrete),
    with the policy's action shape and dtype."""
    shape, dtype = action_struct.shape, action_struct.dtype
    if check_if_environment_has_continuous_actions(env_args.env, env_args.env_params):
        return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0).astype(dtype)
    n_actions = get_action_dim(env_args.env, env_args.env_params)
    return jax.random.randint(key, shape, 0, n_actions).astype(dtype)


# ---------------------------------------------------------------------------
# Env transitions
# ---------------------------------------------------------------------------


def _static_transition(
    cs: RowCollectorState,
    tick: jax.Array,
    episode_length: int,
    env_action: jax.Array,
    step_keys: jax.Array,
    reset_keys: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
) -> _Next:
    n_envs = env_args.n_envs
    phase = tick % (episode_length + 1)
    hold = phase == episode_length  # unbatched: a real cond under vmap
    scheduled_end = phase == episode_length - 1
    ones_b, zeros_b = jnp.ones(n_envs, bool), jnp.zeros(n_envs, bool)

    def held(_):
        obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)
        return _Next(
            env_state=env_state,
            obs=obs,
            reward=jnp.zeros(n_envs, jnp.float32),
            is_first=ones_b,
            is_last=zeros_b,
            is_terminal=zeros_b,
            reset_obs=cs.reset_obs,
            stepped=zeros_b,
            n_offschedule_dones=jnp.zeros((), jnp.int32),
        )

    def stepped(_):
        obs, env_state, reward, terminated, truncated, info = jax.lax.stop_gradient(
            step(
                step_keys,
                _fresh_containers(cs._env_state),
                env_action,
                env_args.env,
                mode,
                env_args.env_params,
            )
        )
        done = (terminated > 0) | (truncated > 0)
        final_obs = get_final_obs(info, obs).astype(obs.dtype)
        end = jnp.broadcast_to(scheduled_end, (n_envs,))
        return _Next(
            env_state=env_state,
            # off-schedule dones keep the (auto-reset) obs the env continues from
            obs=jnp.where(_per_env(end, obs), final_obs, obs),
            reward=reward.astype(jnp.float32),
            is_first=zeros_b,
            is_last=end,
            is_terminal=end & (terminated > 0),
            reset_obs=cs.reset_obs,
            stepped=ones_b,
            n_offschedule_dones=jnp.sum(done & ~end, dtype=jnp.int32),
        )

    return jax.lax.cond(hold, held, stepped, None)


def _dynamic_transition(
    cs: RowCollectorState,
    env_action: jax.Array,
    step_keys: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
) -> _Next:
    held = cs.is_last
    snapshot = cs._env_state
    obs, env_state, reward, terminated, truncated, info = jax.lax.stop_gradient(
        step(
            step_keys,
            _fresh_containers(snapshot),
            env_action,
            env_args.env,
            mode,
            env_args.env_params,
        )
    )
    done = ((terminated > 0) | (truncated > 0)) & ~held
    final_obs = get_final_obs(info, obs).astype(obs.dtype)
    restored = jax.tree.map(
        partial(_restore_held, held, env_args.n_envs), env_state, snapshot
    )
    env_state = _keep_batch_shared_seed(env_args.env, restored, env_state)
    next_obs = jnp.where(
        _per_env(held, obs),
        cs.reset_obs,
        jnp.where(_per_env(done, obs), final_obs, obs),
    )
    return _Next(
        env_state=env_state,
        obs=next_obs,
        reward=jnp.where(held, 0.0, reward.astype(jnp.float32)),
        is_first=held,
        is_last=done,
        is_terminal=done & (terminated > 0),
        # after an episode end the env already holds its auto-reset obs
        reset_obs=jnp.where(_per_env(done, obs), obs, cs.reset_obs),
        stepped=~held,
        n_offschedule_dones=jnp.zeros((), jnp.int32),
    )


def _restore_held(
    held: jax.Array, n_envs: int, stepped_leaf: jax.Array, snapshot_leaf: jax.Array
) -> jax.Array:
    """Per-env leaf select: the snapshot where held, the stepped value elsewhere.

    Leaves without a leading env axis (none in the supported stacks) take
    the stepped value. Batch-shared leaves stored per row are put back by
    :func:`_keep_batch_shared_seed`.
    """
    if jnp.ndim(stepped_leaf) == 0 or jnp.shape(stepped_leaf)[0] != n_envs:
        return stepped_leaf
    return jnp.where(_per_env(held, stepped_leaf), snapshot_leaf, stepped_leaf)


def _keep_batch_shared_seed(env: Any, restored: Any, stepped: Any) -> Any:
    """Undo the per-env restore of a batch-shared reset seed.

    Ajax's brax :class:`ajax.wrappers.AutoResetWrapper` keeps one reset seed
    for the whole batch in ``info["rng"]``, tiled per row and read from row
    0; it advances whenever any env's episode ends. That leaf belongs to the
    batch, not to a held env: restoring a held env 0's row would roll the
    seed back, and the next auto-reset would redraw the initial states it
    has already used (an env restarting from its previous initial state).
    It therefore keeps its stepped value. mujoco_playground's auto-reset
    keeps one key per env (``AutoResetWrapper_rng``), restored per env like
    the rest of the held env's state.
    """
    if not any(isinstance(layer, AutoResetWrapper) for layer in _wrapper_chain(env)):
        return restored
    info = dict(restored.info)
    info["rng"] = stepped.info["rng"]
    return restored.replace(info=info)


# ---------------------------------------------------------------------------
# Helpers
def resume_tick(collector_state: Any, n_envs: int) -> int:
    """The absolute tick a resumed run starts at (host-side).

    The :class:`RowCollectorState`'s ``rows`` (summed over envs, one per env per tick) over
    ``n_envs``; every seed must be at the same tick, so that every schedule
    of the tick continues for all of them.
    """
    rows = np.asarray(jax.device_get(collector_state.rows)).reshape(-1)
    if rows.size == 0 or np.any(rows != rows[0]) or rows[0] % n_envs:
        raise ValueError(
            "cannot resume: the collector row counts differ across seeds or"
            f" are not a multiple of n_envs={n_envs} ({rows.tolist()})"
        )
    return int(rows[0]) // n_envs


# ---------------------------------------------------------------------------


def _fresh_containers(tree: Any) -> Any:
    """The same leaves in newly built containers.

    brax and mujoco_playground wrappers update ``state.info`` *in place*
    during ``step`` (e.g. the auto-reset wrappers' ``info.update(steps=...)``)
    and brax states share that dict with their successors. Stepping a copy
    keeps the caller's env state -- the snapshot a held env is restored
    from, the state a ``lax.cond`` branch closes over -- untouched.
    """
    return jax.tree.map(lambda leaf: leaf, tree)


def _env_keys(key: jax.Array, mode: str, n_envs: int) -> jax.Array:
    """One key per env on gymnax (vmapped), a single key on brax / playground."""
    return jax.random.split(key, n_envs) if mode == "gymnax" else key


def _per_env(mask: jax.Array, like: jax.Array) -> jax.Array:
    """Reshape an ``[n_envs]`` mask to broadcast against ``like``."""
    return mask.reshape(mask.shape + (1,) * (jnp.ndim(like) - mask.ndim))


__all__ = [
    "PolicyFn",
    "ResetMode",
    "Row",
    "RowCollectorState",
    "TimestepUnit",
    "check_unnormalized_env",
    "collect_row",
    "init_row_collector_state",
    "resume_tick",
]
