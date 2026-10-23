import inspect
import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple

import chex
import distrax
import flashbax as fbx
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial

from ajax.buffers.utils import ReplayBatch, init_buffer
from ajax.controller import (
    ActingState,
    ObserveInputs,
    acting_state,
    augment_obs,
    feature_columns,
    feature_width,
    flatten_expert_state,
    init_observers,
    observe_features,
)
from ajax.environments.obs_norm import (
    apply_obs_norm,
    init_agent_obs_norm,
    normalized_obs,
    update_obs_norm,
)
from ajax.environments.system_class import env_params_is_batched
from ajax.environments.utils import (
    ExpertInAgentUnits,
    agent_action_to_env,
    check_env_is_playground,
    check_if_environment_has_continuous_actions,
    env_keys,
    get_raw_env,
    get_state_action_shapes,
    maybe_append_train_frac,
    raw_obs_columns,
)
from ajax.networks.distributions import sample_with_raw
from ajax.networks.memory import flatten_carry, tree_where_reset
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    LoadedTrainState,
    RollinEpisodicMeanRewardState,
    Transition,
    zeros_like_abstract_pytree,
)
from ajax.types import BufferType, NormalizationInfo
from ajax.wrappers import (
    _FINAL_OBS_KEYS,
    RAW_FINAL_OBS_KEY,
    RAW_OBS_KEY,
    RAW_REWARD_KEY,
)


def augment_from_batch(batch: ReplayBatch) -> Tuple[jax.Array, jax.Array]:
    """A replay batch's ``obs`` and ``next_obs`` augmented (:func:`augment_obs`)
    with the expert's actions collection stored for them: ``a_expert`` and
    ``next_a_expert``, from raw observations and the expert's state."""
    return (
        augment_obs(batch.obs, expert_action=batch.a_expert),
        augment_obs(batch.next_obs, expert_action=batch.next_a_expert),
    )


def expert_state_dim(expert_policy, n_envs_probe: int = 1) -> int:
    """Return the flattened expert_state dim for `expert_policy`. Zero for
    stateless experts (no `init_state` attribute)."""
    if expert_policy is None or not hasattr(expert_policy, "init_state"):
        return 0
    state = expert_policy.init_state(n_envs_probe)
    flat = flatten_expert_state(state)
    if flat is None:
        return 0
    return int(flat.shape[-1])


@partial(jax.jit, static_argnames=["window_size"])
def init_rolling_mean(
    window_size: int, last_return: jnp.ndarray, cumulative_reward: jnp.ndarray
) -> RollinEpisodicMeanRewardState:
    return RollinEpisodicMeanRewardState(
        buffer=jnp.zeros((window_size, cumulative_reward.shape[0], 1)),
        index=jnp.zeros_like(cumulative_reward).astype("int8"),
        count=jnp.zeros_like(cumulative_reward).astype("int8"),
        sum=jnp.zeros_like(cumulative_reward),
        cumulative_reward=cumulative_reward,
        last_return=last_return,
    )


def get_final_obs(info: Any, fallback: jax.Array) -> jax.Array:
    """Return the transition's pre-reset observation.

    Auto-resetting environments overwrite the returned observation with the
    reset one whenever the episode ends, so the terminal observation has to be
    recovered from ``info`` -- truncation must bootstrap on ``V(s_T)``, not on
    ``V(s_reset)``. Falls back to ``fallback`` (the post-step observation) for
    environments that expose neither key.
    """
    for key in _FINAL_OBS_KEYS:
        if key in info:
            return info[key]
    return fallback


def reward_statistics(env_state: Any) -> Optional[NormalizationInfo]:
    """The env reward normaliser's current statistics (None without one): a
    brax state keeps them in its info, a gymnax one as a field."""
    info = getattr(env_state, "info", None)
    stats = (
        info.get("normalization_info")
        if isinstance(info, dict)
        else getattr(env_state, "normalization_info", None)
    )
    return None if stats is None else stats.reward


def with_reward_statistics(env_state: Any, stats: NormalizationInfo) -> Any:
    """``env_state`` with its reward normaliser's statistics ``stats``,
    where :func:`reward_statistics` reads them."""
    info = getattr(env_state, "info", None)
    if isinstance(info, dict):
        norm = info["normalization_info"].replace(reward=stats)
        return env_state.replace(info={**info, "normalization_info": norm})
    norm = env_state.normalization_info.replace(reward=stats)
    return env_state.replace(normalization_info=norm)


def get_raw_final_obs(info: Any, fallback: jax.Array) -> jax.Array:
    """:func:`get_final_obs` in the env's own units: an observation
    normaliser normalises the final observation and keeps the raw one."""
    return info.get(RAW_FINAL_OBS_KEY, get_final_obs(info, fallback))


def bootstrap_obs(
    next_obs: jax.Array,
    final_obs: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
) -> jax.Array:
    """The observation a replayed transition's target bootstraps on.

    ``next_obs`` is the one the agent sees next, the reset observation after
    an end. A time limit is not part of the task: where it alone ended the
    episode, the episode was cut, not finished, so the target bootstraps on
    the final pre-reset observation ``final_obs`` (SB3's
    ``handle_timeout_termination``, CleanRL's ``real_next_obs``). After a
    termination the reset observation stays: a discounted target masks it,
    and ASAC's differential target, which never cuts, continues into the
    next episode from it. Flags are shaped ``(n,)`` or ``(n, 1)``.
    """
    cut = (truncated > 0) & ~(terminated > 0)
    cut = cut.reshape(cut.shape[0], *(1,) * (next_obs.ndim - 1))
    return jnp.where(cut, final_obs, next_obs)


@partial(jax.jit, static_argnames=["mode", "env"])
def reset(
    rng: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> tuple[jax.Array, EnvState]:
    """Reset the environment and return the initial observation and state.

    Args:
        mode: Environment mode ("gymnax" or "brax").
        env_params: Parameters for Gymnax environments.
    """
    if mode == "gymnax":
        # Batched params (one system per env, see ajax.environments.system_class)
        # vmap alongside the keys; unbatched params broadcast.
        params_axis = 0 if env_params_is_batched(env_params) else None
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, params_axis))(rng, env_params)
    else:
        env_state = env.reset(rng)
        obsv = env_state.obs
    # Observations feed neural networks and a float32-schema replay buffer.
    # Some envs (e.g. the discrete probing envs) emit integer observations;
    # coerce to float32 here so every downstream consumer is consistent.
    # No-op for the float-obs envs (gymnax classic control, brax).
    obsv = obsv.astype(jnp.float32)
    return obsv, env_state


def step(
    rng: jax.Array,
    state: jax.Array,
    action: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, jax.Array, Any]:
    """Perform a step in the environment.

    Args:
        mode: Environment mode ("gymnax" or "brax").
        env_params: Parameters for Gymnax environments.

    Returns:
        The observation, new state, reward, terminated and truncated flags,
        and info.
    """
    if env_params is None and mode == "gymnax":
        env_params = env.default_params

    if mode == "gymnax":
        # Ajax carries actions as float32 end-to-end (the replay buffer uses a
        # float32 schema). Discrete gymnax envs index with the action --
        # e.g. MinAtar's `action_set[action]` -- and JAX rejects a float
        # indexer, so restore the integer dtype at the env boundary. Envs
        # whose dynamics only compare the action (classic control) accepted
        # the float silently, which is why this stayed hidden.
        if not check_if_environment_has_continuous_actions(env, env_params):
            action = action.astype(jnp.int32)

        def step_wrapper(
            rng: jax.Array,
            state: EnvState,
            action: jax.Array,
            env_params: EnvParams,
        ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, jax.Array, Any]:
            return env.step(key=rng, state=state, action=action, params=env_params)

        params_axis = 0 if env_params_is_batched(env_params) else None
        if rng.ndim > 1:
            out = jax.vmap(
                step_wrapper,
                in_axes=(0, 0, 0, params_axis),
            )(rng, state, action, env_params)
        else:
            out = step_wrapper(rng, state, action, env_params)
        # Gymnax >= 1.0 reports terminated and truncated apart; an env still
        # folding the time limit into one ``done`` needs
        # :class:`ajax.wrappers.TerminatedTruncatedWrapper`.
        obsv, env_state, reward, terminated, truncated, info = out
        terminated, truncated = jnp.float_(terminated), jnp.float_(truncated)
        if "normalization_info" in env_state.__dict__:
            obs_norm_info = jax.tree.map(
                lambda x: jnp.broadcast_to(x.mean(axis=0, keepdims=True), x.shape),
                env_state.normalization_info.obs,
            )
            reward_norm_info = env_state.normalization_info.reward
            if reward_norm_info is not None:
                # One normaliser per env: the statistics are pooled, each
                # env's discounted-return accumulator stays its own.
                reward_norm_info = jax.tree.map(
                    lambda x: jnp.broadcast_to(x.mean(axis=0, keepdims=True), x.shape),
                    reward_norm_info.replace(returns=None),
                ).replace(returns=reward_norm_info.returns)
            env_state = env_state.replace(
                normalization_info=env_state.normalization_info.replace(
                    obs=obs_norm_info, reward=reward_norm_info
                )
            )
    elif mode == "brax":
        env_state = env.step(state=state, action=action)
        obsv, reward, done, info = (
            env_state.obs,
            env_state.reward,
            env_state.done,
            env_state.info,
        )
        truncated = env_state.info["truncation"]
        terminated = done * (1 - truncated)

    else:
        raise ValueError(f"Unrecognized mode for step {mode}")

    # See reset(): keep observations float32 regardless of the env's
    # native obs dtype so networks and the replay buffer stay consistent.
    # The reward is coerced too -- some envs emit integer rewards, which
    # otherwise mismatch the float32 buffer schema and eval scan carry.
    obsv = obsv.astype(jnp.float32)
    reward = reward.astype(jnp.float32)
    return obsv, env_state, reward, terminated, truncated, info


def get_pi(
    actor_state: LoadedTrainState,
    actor_params: FrozenDict,
    obs: jax.Array,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
) -> Tuple[distrax.Distribution, LoadedTrainState]:
    """Get the policy distribution for the given observation and actor state.

    Args:
        done: Done flags for recurrent mode.

    Returns:
        The policy distribution and the updated actor state.
    """
    # Agent-side obs normalisation, applied here rather than by each caller
    # of get_pi. Stats are synced into actor_state.obs_norm_info after every
    # online collection step. None disables normalisation.
    obs = apply_obs_norm(obs, getattr(actor_state, "obs_norm_info", None))
    obs = maybe_add_axis(obs, recurrent)
    if recurrent:
        assert done is not None
        done = maybe_add_axis(done, recurrent)
        pi, new_actor_hidden_state = actor_state.apply(
            actor_params,
            obs,
            hidden_state=actor_state.hidden_state,
            done=done,
        )
    else:
        pi = actor_state.apply(actor_params, obs)
        new_actor_hidden_state = None
    return pi, actor_state.replace(hidden_state=new_actor_hidden_state)


def get_pi_sequence(
    actor_state: LoadedTrainState,
    actor_params: FrozenDict,
    obs: jax.Array,
    resets: jax.Array,
    initial_hidden: Any,
) -> Tuple[distrax.Distribution, Any]:
    """Run a recurrent actor over a time-major sequence (training/BPTT path).

    Unlike :func:`get_pi` (single-step path, which reads/advances the live
    carry on ``actor_state``), this takes an explicit initial carry — e.g.
    the rollout-start carry for on-policy updates, or a burned-in carry for
    replayed sequences — and does NOT mutate the actor state.

    Args:
        obs: (T, B, obs_dim) observations.
        resets: (T, B) episode-start flags aligned with ``obs``.
        initial_hidden: carry valid for obs[0].

    Returns:
        (pi, final_hidden): distribution over (T, B, ...) and the carry
        after consuming the sequence.
    """
    obs = apply_obs_norm(obs, getattr(actor_state, "obs_norm_info", None))
    return actor_state.apply(
        actor_params, obs, hidden_state=initial_hidden, done=resets
    )


def maybe_add_axis(arr: jax.Array, recurrent: bool) -> jax.Array:
    """Add an axis to the array if in recurrent mode."""
    return arr[jnp.newaxis, :] if recurrent else arr


@partial(jax.jit, static_argnames=["recurrent"])
def get_action_and_new_agent_state(
    rng,
    agent_state: BaseAgentState,
    obs: jnp.ndarray,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
):
    """Get the action and updated agent state based on the current observation.

    Args:
        done: Done flags for recurrent mode.

    Returns:
        ``(action, log_probs, raw_action, agent_state)``.
    """
    if recurrent:
        chex.assert_tree_no_nones(done)

    pi, new_actor_state = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=obs,
        done=done,
        recurrent=recurrent,
    )
    # raw_action, the pre-tanh sample, lets PPO/APO recompute the log-prob
    # without arctanh.
    action, log_probs, raw_action = sample_with_raw(pi, rng)
    if recurrent:
        # get_pi ran the actor on a single-step sequence (1, B, ...);
        # drop the time axis so callers (env step, buffer writes) see the
        # same (B, ...) shapes as in the feedforward path.
        action = action.squeeze(0)
        log_probs = log_probs.squeeze(0)
        raw_action = raw_action.squeeze(0)

    return (
        action,
        log_probs,
        raw_action,
        agent_state.replace(actor_state=new_actor_state),
    )


def update_episodic_return(
    episodic_return_state: RollinEpisodicMeanRewardState,
    reward: jnp.ndarray,
    done: jnp.ndarray,
) -> tuple[RollinEpisodicMeanRewardState, jnp.ndarray]:
    """Advance the per-env rolling mean of episode returns by one step.

    ``reward`` (``[n_envs]``, raw) is added to every env's running return;
    an env whose episode is ``done`` pushes that return into its rolling
    window and restarts at 0. Returns the new state and the mean over envs
    of each env's rolling mean (NaN until every env finished an episode),
    the house ``Train/episodic mean reward``.
    """
    state = episodic_return_state
    done = done.reshape(done.shape[0], 1)
    new_cumulative_reward = state.cumulative_reward + reward.reshape(reward.shape[0], 1)
    # NaN until the env's first episode ends.
    last_return = jnp.where(done, new_cumulative_reward, state.last_return)
    # A done env writes its return over its window's oldest one; the others
    # leave their window, sum, count and index as they are.
    rows, cols = state.index.flatten(), jnp.arange(done.shape[0])
    oldest = state.buffer[rows, cols]
    window = state.buffer.shape[0]
    total = jnp.where(done, state.sum - oldest + last_return, state.sum)
    count = jnp.where(done, jnp.minimum(state.count + 1, window), state.count)
    new_state = state.replace(
        buffer=state.buffer.at[rows, cols].set(jnp.where(done, last_return, oldest)),
        index=jnp.where(done, (state.index + 1) % window, state.index),
        count=count,
        sum=total,
        cumulative_reward=new_cumulative_reward * (1 - done),
        last_return=last_return,
    )
    return new_state, jnp.mean(total / count)


def get_action_and_log_probs(
    action_key: chex.PRNGKey,
    agent_state: BaseAgentState,
    recurrent: bool,
    acting: ActingState,
) -> Tuple[jax.Array, jax.Array, jax.Array, BaseAgentState]:
    """Sample an action from the policy at ``acting``'s observation, from
    its carry.

    Returns (action, log_probs, raw_action, agent_state). raw_action is
    the pre-tanh sample (PPO/APO recompute log_prob without arctanh).
    The updated agent state must be kept by callers: recurrent actors
    advance their hidden state on every forward pass, and dropping it
    would freeze the policy's memory at zero for the whole collection
    phase.
    """
    actor = agent_state.actor_state.replace(hidden_state=acting.carry)
    return get_action_and_new_agent_state(
        action_key,
        agent_state.replace(actor_state=actor),
        acting.obs,
        acting.done,
        recurrent=recurrent,
    )


def maybe_vmap(f, vmap_on, **kwargs):
    if vmap_on:
        return jax.vmap(f, **kwargs)
    return f


def get_raw_obs(env_state: EnvState, env: Environment, mode: str) -> jax.Array:
    vmap_on = (
        max(jax.tree_util.tree_leaves(jax.tree.map(lambda x: x.ndim, env_state))) > 0
    )
    if mode == "gymnax":
        # gymnax 1.0's signature is `get_obs(state, params, key=None)`, and some
        # envs (e.g. MemoryChain) make `params` required rather than optional.
        env_params = getattr(env, "default_params", None)
        return maybe_vmap(lambda st: env.get_obs(st, env_params), vmap_on)(env_state)
    # An observation normaliser overwrites a brax state's obs and keeps the
    # raw one in its info.
    if RAW_OBS_KEY in getattr(env_state, "info", {}):
        return env_state.info[RAW_OBS_KEY]
    if check_env_is_playground(env):
        # env_state.obs is already the post-wrapper observation (e.g. with
        # safety-wrapper augmentations). Recomputing it via the raw env's
        # `_get_obs(env_state.data)` would strip those augmentations and
        # mismatch the buffer schema (which was sized from the wrapped
        # `env.observation_size`). Trust the env's own bookkeeping.
        return env_state.obs
    # Brax: prefer env._get_obs for pre-normalization obs. Some minimal envs
    # (brax `fast`) don't expose `_get_obs`; fall back to env_state.obs.
    raw_env = get_raw_env(env)
    if not hasattr(raw_env, "_get_obs"):
        return env_state.obs
    # Some brax envs (e.g. humanoid) have `_get_obs(pipeline_state, action)`
    # and cannot be recomputed without the action; env_state.obs already
    # holds the correct pre-normalization observation -> trust it.
    try:
        n_obs_params = len(inspect.signature(raw_env._get_obs).parameters)
    except (TypeError, ValueError):
        n_obs_params = 1
    if n_obs_params > 1:
        return env_state.obs
    return maybe_vmap(env._get_obs, vmap_on)(env_state.pipeline_state)


class ActionPipelineResult(NamedTuple):
    """What an ``action_pipeline`` hands :func:`collect_experience`.

    The first two fields are required. The others default to ``None``, and
    ``collect_experience`` then falls back: zero log-probs, no expert acting,
    no value box, the key it handed the pipeline, the env action stored, no
    expert state or action threaded, and the actor's carry as the pipeline
    found it.
    """

    # Executed in the environment once agent_action_to_env maps it onto the
    # env's bounds: in agent units, [-1, 1], as every action below.
    env_action: jax.Array
    policy_action: jax.Array  # the actor's action, stored in the transition
    log_probs: Optional[jax.Array] = None
    is_expert_flag: Optional[jax.Array] = None  # 1 where an expert acted
    in_value_box: Optional[jax.Array] = None  # value-box membership
    entry_bonus: Optional[jax.Array] = None  # value-box entry bonus
    rng: Optional[jax.Array] = None  # the collector's key after its draws
    # A stateful expert's state after this step (PID integrator, CPG phase).
    new_expert_state: Optional[Any] = None
    # Written to the replay buffer instead of env_action: a gain policy's
    # raw output, a discrete action's (n_envs, 1) column.
    buffer_action: Optional[jax.Array] = None
    # The expert's action this step, computed with its true internal state,
    # so the residual-RL losses need not recompute it from a zero state.
    a_expert: Optional[jax.Array] = None
    # A recurrent actor's advanced carry: a pipeline that runs the policy
    # hands it back, or the actor's memory would stay frozen.
    new_actor_hidden: Optional[Any] = None
    # A stateful expert's state at an episode start (its init_state): the
    # state the next row's observation carries where this step ended one.
    initial_expert_state: Optional[Any] = None
    # The pre-tanh sample PPO/APO recompute a log-prob from. Without it the
    # transition stores the action itself (pipelines act post-tanh, and no
    # agent with a pipeline of its own recomputes log-probs).
    raw_action: Optional[jax.Array] = None
    # 1 where the step entered the value box, which ends the agent's
    # episode whatever the sign of the entry bonus. None: no box.
    box_entry: Optional[jax.Array] = None


def _policy_pipeline(
    agent_state: BaseAgentState,
    acting: ActingState,
    _raw_obs: jax.Array,
    rng: jax.Array,
    uniform: jax.Array,
    mix_key: jax.Array,
    action_key: jax.Array,
    *,
    recurrent: bool,
    n_envs: int,
) -> ActionPipelineResult:
    """The default action pipeline: the policy's action, or a uniform one
    in [-1, 1] while ``uniform`` (the warm-up)."""
    action, log_probs, raw_action, agent_state = get_action_and_log_probs(
        action_key=action_key,
        agent_state=agent_state,
        recurrent=recurrent,
        acting=acting,
    )
    uniform_action = jax.random.uniform(
        mix_key, minval=-1.0, maxval=1.0, shape=action.shape
    )
    # A discrete policy's action is stepped and stored as a float, the
    # uniform draw's dtype, which the cond needs both branches to share.
    action = action.astype(uniform_action.dtype)
    no_box = jnp.zeros((n_envs, 1), dtype=jnp.float32)
    return ActionPipelineResult(
        env_action=jax.lax.cond(uniform, lambda: uniform_action, lambda: action),
        policy_action=action,
        log_probs=log_probs,
        is_expert_flag=jnp.zeros_like(action[..., :1], dtype=jnp.float32),
        in_value_box=no_box,
        entry_bonus=no_box,
        rng=rng,
        new_actor_hidden=agent_state.actor_state.hidden_state,
        raw_action=raw_action,
    )


#: ``explore(pi, key, step) -> action``: a behaviour policy. From the
#: policy ``pi`` at the last observations (a distribution; DQN's greedy
#: Q-policy), a key of its own and the timestep, the actions to take: in
#: [-1, 1], or a discrete action's index (:func:`explore_pipeline`).
Explore = Callable[[Any, jax.Array, jax.Array], jax.Array]


def explore_pipeline(
    explore: Explore,
    recurrent: bool,
    n_actions: Optional[int] = None,
    warmup: bool = True,
) -> Callable:
    """The action pipeline of the behaviour policy ``explore`` (``n_actions``:
    a discrete action space's size), or uniform actions, on ``mix_key``,
    during the warm-up; an agent that never warms up (``warmup=False``, an
    on-policy one) draws none.

    A recurrent actor advances its carry, its one-step time axis dropped
    from the action. A discrete action is stored as an ``(n_envs, 1)``
    column. No log-probs, expert or value box.
    """

    def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        del raw_obs
        actor = agent_state.actor_state.replace(hidden_state=acting.carry)
        pi, actor = get_pi(actor, actor.params, acting.obs, acting.done, recurrent)
        action = explore(pi, action_key, acting.timestep)
        action = action.squeeze(0) if recurrent else action
        env_action = action
        if warmup:
            if n_actions is None:
                uniform_action = jax.random.uniform(
                    mix_key, minval=-1.0, maxval=1.0, shape=action.shape
                )
            else:
                uniform_action = jax.random.randint(
                    mix_key, action.shape, minval=0, maxval=n_actions
                ).astype(action.dtype)
            env_action = jax.lax.cond(uniform, lambda: uniform_action, lambda: action)
        discrete = n_actions is not None
        return ActionPipelineResult(
            env_action=env_action,
            policy_action=action[:, None] if discrete else action,
            rng=rng,
            buffer_action=env_action[:, None] if discrete else None,
            new_actor_hidden=actor.hidden_state if recurrent else None,
        )

    return pipeline


def with_acting_state(pipeline: Optional[Callable]) -> Optional[Callable]:
    """``pipeline`` in the protocol :func:`collect_experience` calls, the
    step's :class:`~ajax.controller.ActingState` second. One written before
    it, ``(agent_state, raw_obs, rng, uniform, mix_key, action_key)``, which
    cannot take the seven arguments, is adapted with a warning. A deliberate
    transitional shim (CLAUDE.md rule 10): remove it once AjaxExperiments'
    TD3 variants take ``acting``."""
    if pipeline is None:
        return pipeline
    try:
        inspect.signature(pipeline).bind(*range(7))
        return pipeline
    except TypeError:
        pass
    warnings.warn(
        "An action pipeline now takes the step's ActingState second:"
        " (agent_state, acting, raw_obs, rng, uniform, mix_key, action_key)."
        " The old signature will stop working.",
        FutureWarning,
        stacklevel=3,
    )

    def adapted(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        del acting
        return pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key)

    return adapted


def with_expert_state(collector_state: Any, expert_policy: Any, n_envs: int) -> Any:
    """``collector_state`` holding a stateful expert's initial state (a PID's
    integrator), so the scan carry keeps one shape; unchanged for no expert
    or a stateless one. Collection resets it per env at episode starts."""
    if expert_policy is None or not hasattr(expert_policy, "init_state"):
        return collector_state
    return collector_state.replace(expert_state=expert_policy.init_state(n_envs))


def expert_state_at_step(expert_policy: Any, acting: ActingState, n_envs: int) -> Any:
    """A stateful expert's state for this step: the tracked one, reset to
    its initial state where the previous step ended an episode (the
    auto-reset has already advanced the observation)."""
    initial = expert_policy.init_state(n_envs)
    current = acting.expert_state if acting.expert_state is not None else initial
    return tree_where_reset(acting.done.astype(jnp.bool_), initial, current)


def expert_step(
    expert_policy: Any, acting: ActingState, raw_obs: jax.Array, n_envs: int
) -> Tuple[jax.Array, Any]:
    """The expert's action at ``raw_obs`` and its state after the call: a
    stateful expert (with ``init_state``) is called from its state for this
    step (:func:`expert_state_at_step`), a stateless one on the observation
    alone (its state ``None``). Both held constant."""
    if hasattr(expert_policy, "init_state"):
        state = expert_state_at_step(expert_policy, acting, n_envs)
        action, new_state = expert_policy(state, raw_obs)
        return jax.lax.stop_gradient(action), jax.lax.stop_gradient(new_state)
    return jax.lax.stop_gradient(expert_policy(raw_obs)), None


def make_next_expert_fn(expert_policy: Any) -> Optional[Callable]:
    """``next_expert_fn(post_step_state, raw_next_obs)``: the expert's action
    at ``s_{t+1}`` from its state after the call at ``s_t``, the action it
    would take next (stored as ``next_a_expert``); None without an expert,
    when collection stores zeros."""
    if expert_policy is None:
        return None
    expert_is_stateful = hasattr(expert_policy, "init_state")

    def next_expert_fn(post_step_expert_state, raw_next_obs):
        if expert_is_stateful and post_step_expert_state is not None:
            a_next, _ = expert_policy(post_step_expert_state, raw_next_obs)
            return a_next
        return expert_policy(raw_next_obs)

    return next_expert_fn


def expert_collect_kwargs(
    expert_policy: Any,
    action_pipeline: Optional[Callable],
    recurrent: bool,
    n_envs: int,
) -> dict:
    """:func:`collect_experience`'s ``action_pipeline`` and ``next_expert_fn``
    for an agent storing its expert's actions (``a_expert`` at s,
    ``next_a_expert`` at s'), the expert called from its state tracked
    along each trajectory (:func:`expert_step`). ``action_pipeline`` (None:
    the policy's) still chooses the action, with the same draws; one that
    tracks the expert itself (returns its state or action) is left as is.
    Without an expert, ``action_pipeline`` alone."""
    if expert_policy is None:
        return {"action_pipeline": action_pipeline}
    inner = action_pipeline or partial(
        _policy_pipeline, recurrent=recurrent, n_envs=n_envs
    )

    def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        result = inner(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key)
        if result.new_expert_state is not None or result.a_expert is not None:
            return result
        a_expert, new_state = expert_step(expert_policy, acting, raw_obs, n_envs)
        return result._replace(a_expert=a_expert, new_expert_state=new_state)

    return {
        "action_pipeline": pipeline,
        "next_expert_fn": make_next_expert_fn(expert_policy),
    }


def step_features(
    env_args: EnvironmentConfig,
    mode: str,
    expert: Any,
    collector: Any,
    env_state: Any,
    raw_final_obs: jax.Array,
    expert_state: Any,
    ended: jax.Array,
) -> tuple[jax.Array, jax.Array, Any]:
    """A step's observation features (``env_args.observers``): those of the
    next row's observation, the reset one where an episode ended, from fresh
    observer and expert states; those of the step's final observation (the
    transition's next) from the continuing ones, the expert's ``expert_state``
    after this step's call; and the observers' states after the next
    observation."""
    observers, n_envs = env_args.observers, env_args.n_envs
    done = ended.astype(jnp.bool_)
    continuing = collector.observer_states
    start = tree_where_reset(done, init_observers(observers, n_envs), continuing)
    expert_start = expert_state
    if (
        expert is not None
        and hasattr(expert, "init_state")
        and expert_state is not None
    ):
        expert_start = tree_where_reset(done, expert.init_state(n_envs), expert_state)
    at = {"expert": expert, "timestep": collector.timestep}
    at["horizon"] = collector.max_timesteps
    raw_next = get_raw_obs(env_state=env_state, env=env_args.env, mode=mode)
    next_inputs = ObserveInputs(
        raw_next.astype(jnp.float32), expert_state=expert_start, **at
    )
    final_inputs = ObserveInputs(raw_final_obs, expert_state=expert_state, **at)
    next_columns, states = feature_columns(next_inputs, observers, start)
    final_columns, _ = feature_columns(final_inputs, observers, continuing)
    return next_columns, final_columns, states


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "mode",
        "env_args",
        "buffer",
        "action_pipeline",
        "next_expert_fn",
        "store_hidden",
        "store_raw_next_obs",
        "expert",
    ],
)
def collect_experience(
    agent_state: BaseAgentState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: Optional[BufferType] = None,
    uniform: bool = False,
    action_pipeline: Optional[Callable] = None,
    next_expert_fn: Optional[Callable] = None,
    store_hidden: bool = False,
    store_raw_next_obs: bool = False,
    expert: Any = None,
) -> tuple[BaseAgentState, Transition]:
    """Collect one step of experience. ``store_raw_next_obs`` keeps the
    env's own next observation on the transition (``raw_next_obs``), for
    an agent updating on it (AVG) whose extensions read it. ``expert`` (the
    agent's, in agent units) is what its observation features read
    (``env_args.observers``).

    Args:
        action_pipeline: Optional callable that handles agent-specific action
            selection (obs augmentation, expert exploration, residual RL, etc.).
            When None, the policy's (:func:`_policy_pipeline`): uniform during
            the warm-up, the policy's action after.
            Signature: (agent_state, acting, raw_obs, rng, uniform, mix_key,
                action_key) -> :class:`ActionPipelineResult`, ``acting`` the
                step's :class:`~ajax.controller.ActingState`.
    """
    rng, action_key, step_key, mix_key = jax.random.split(agent_state.rng, 4)

    rng_step = env_keys(step_key, mode, env_args.n_envs)

    # R2D2-style stored-state replay: the actor's carry BEFORE this step is
    # the hidden state valid for last_obs (the transition's obs). Snapshot
    # it now — the action selection below advances the live carry.
    if store_hidden:
        _pre_actor_carry_flat = jax.lax.stop_gradient(
            flatten_carry(agent_state.actor_state.hidden_state)
        )

    # Update obs running stats with the current last_obs (BEFORE the
    # forward pass), then sync into actor/critic so the very first step
    # of training already uses non-zero stats. This avoids a step-1
    # divide-by-≈0 in apply_obs_norm.
    if (
        agent_state.collector_state.obs_norm_info is not None
        and not env_args.store_normalized_obs
    ):
        # The span the statistics cover: the env's columns and the
        # normalised features (the raw features last).
        last_obs = agent_state.collector_state.last_obs
        span = last_obs.shape[-1] - feature_width(env_args, normalised=False)
        _, _new_obs_norm = update_obs_norm(
            last_obs[..., :span],
            agent_state.collector_state.obs_norm_info,
            raw_obs_columns(env_args.env),
        )
        agent_state = agent_state.replace(
            collector_state=agent_state.collector_state.replace(
                obs_norm_info=_new_obs_norm
            ),
            actor_state=agent_state.actor_state.replace(obs_norm_info=_new_obs_norm),
            critic_state=agent_state.critic_state.replace(obs_norm_info=_new_obs_norm),
        )

    # Raw obs (needed by action_pipeline for expert policy). It is written
    # to the float32-schema buffer; coerce int obs (e.g. discrete probing
    # envs) so flashbax's dtype check passes.
    raw_obs = get_raw_obs(
        env_state=agent_state.collector_state.env_state,
        env=env_args.env,
        mode=mode,
    ).astype(jnp.float32)

    # The agent's own pipeline (SAC's expert, EDGE, the value box, ...) or
    # the policy's.
    pipeline = action_pipeline or partial(
        _policy_pipeline, recurrent=recurrent, n_envs=env_args.n_envs
    )
    acting = acting_state(agent_state)
    result = pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key)
    env_action = result.env_action
    action = result.policy_action
    zeros = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)
    log_probs = zeros if result.log_probs is None else result.log_probs
    is_expert_flag = zeros if result.is_expert_flag is None else result.is_expert_flag
    in_value_box = zeros if result.in_value_box is None else result.in_value_box
    entry_bonus = zeros if result.entry_bonus is None else result.entry_bonus
    rng = rng if result.rng is None else result.rng
    new_expert_state = result.new_expert_state
    # A recurrent actor's memory would stay frozen without its advanced carry.
    if result.new_actor_hidden is not None:
        agent_state = agent_state.replace(
            actor_state=agent_state.actor_state.replace(
                hidden_state=result.new_actor_hidden
            )
        )
    raw_action = action if result.raw_action is None else result.raw_action
    buffer_action = env_action if result.buffer_action is None else result.buffer_action

    obsv, env_state, reward, terminated, truncated, info = jax.lax.stop_gradient(
        step(
            rng_step,
            agent_state.collector_state.env_state,
            agent_action_to_env(env_action, env_args.env, env_args.env_params),
            env_args.env,
            mode,
            env_args.env_params,
        )
    )

    # Read before the train_frac column below, which info's final obs lacks.
    final_obs = get_final_obs(info, obsv).astype(obsv.dtype)
    # The same in the environment's own units, what an expert reads.
    raw_final_obs = get_raw_final_obs(info, final_obs)
    obs_norm_info = agent_state.collector_state.obs_norm_info
    if env_args.store_normalized_obs:
        # The next observation counts in the statistics it is normalised
        # with, and so is the step's final one (the transition's next).
        cols = raw_obs_columns(env_args.env)
        obsv, obs_norm_info = update_obs_norm(obsv, obs_norm_info, cols)
        final_obs = normalized_obs(final_obs, obs_norm_info, cols)

    # An agent conditioned on the training fraction sees it as a last
    # column of every observation, the next one (the final observation off
    # an episode end) included.
    train_frac = (
        agent_state.collector_state.train_time_fraction
        if agent_state.collector_state.max_timesteps is not None
        else None
    )
    raw_next_obs = maybe_append_train_frac(final_obs, train_frac)
    obsv = maybe_append_train_frac(obsv, train_frac)

    # The replay keeps the env's own reward, which a replayed batch divides
    # by the normaliser's current statistics (get_batch_from_buffer).
    own_reward = info.get(RAW_REWARD_KEY, reward).astype(reward.dtype)
    # The value box: its entry pays the bonus (zeros without a box) and ends
    # the agent's episode, the bonus standing in for the bootstrap.
    reward = reward + entry_bonus[..., 0]
    if result.box_entry is not None:
        terminated = jnp.logical_or(
            terminated.astype(bool), result.box_entry[..., 0] > 0
        ).astype(terminated.dtype)

    # Computed once for the buffer write and the on-policy Transition
    # below; zeros without an expert. The buffer keeps them only when its
    # schema has the fields (init_buffer's include_expert_fields).
    _a_expert_for_buf = (
        result.a_expert
        if result.a_expert is not None
        else jnp.zeros_like(buffer_action)
    )
    if next_expert_fn is not None:
        _next_a_expert_for_buf = jax.lax.stop_gradient(
            next_expert_fn(new_expert_state, raw_final_obs)
        )
    else:
        _next_a_expert_for_buf = jnp.zeros_like(buffer_action)

    # If the env runs with augment_obs_with_expert_state, the
    # collector's last_obs is already augmented with the (BEFORE-expert)
    # expert_state. Augment next_obs symmetrically with the AFTER-expert
    # new_expert_state so the buffer stores consistent shapes.
    _next_state_aug = flatten_expert_state(new_expert_state)
    augmented = (
        _next_state_aug is not None
        and agent_state.collector_state.last_obs.shape[-1]
        == raw_next_obs.shape[-1] + _next_state_aug.shape[-1]
    )
    if augmented:
        next_obs_for_buffer = augment_obs(raw_next_obs, _next_state_aug)
    else:
        next_obs_for_buffer = raw_next_obs

    # The next row's obs is the env's (the reset one where the episode
    # ended), joined when augmented to the state the expert's next call
    # starts from: its initial state where the episode ended (the pipeline
    # resets it on this step's done flag), else the post-step one.
    if augmented:
        assert _next_state_aug is not None
        _start_state_aug = _next_state_aug
        if result.initial_expert_state is not None:
            _ended = jnp.logical_or(terminated, truncated).reshape(-1, 1)
            _initial_state_aug = flatten_expert_state(result.initial_expert_state)
            assert _initial_state_aug is not None
            _start_state_aug = jnp.where(_ended, _initial_state_aug, _next_state_aug)
        new_last_obs = augment_obs(obsv, _start_state_aug)
    else:
        new_last_obs = obsv

    observer_states = agent_state.collector_state.observer_states
    if env_args.observers:
        next_columns, final_columns, observer_states = step_features(
            env_args,
            mode,
            expert,
            agent_state.collector_state,
            env_state,
            raw_final_obs,
            new_expert_state,
            jnp.logical_or(terminated, truncated),
        )
        new_last_obs = jnp.concatenate([new_last_obs, next_columns], axis=-1)
        next_obs_for_buffer = jnp.concatenate(
            [next_obs_for_buffer, final_columns], axis=-1
        )

    buffer_state = agent_state.collector_state.buffer_state
    if buffer_state is not None and buffer is not None:
        # The final observation as wide as the next row's.
        final_obs = maybe_append_train_frac(final_obs, train_frac=train_frac)
        if augmented:
            final_obs = augment_obs(final_obs, _next_state_aug)
        if env_args.observers:
            final_obs = jnp.concatenate([final_obs, final_columns], axis=-1)
        row = {
            "obs": agent_state.collector_state.last_obs,
            "action": buffer_action,
            "reward": (own_reward + entry_bonus[..., 0])[:, None],
            "terminated": terminated[:, None],
            "truncated": truncated[:, None],
            "next_obs": bootstrap_obs(new_last_obs, final_obs, terminated, truncated),
            "raw_obs": raw_obs,
            "is_expert": is_expert_flag,
            "a_expert": _a_expert_for_buf,
            "next_a_expert": _next_a_expert_for_buf,
            "raw_next_obs": raw_final_obs.astype(jnp.float32),
            # Unknown until the episode ends: no Cal-QL floor on the row.
            "return_to_go": jnp.full((env_args.n_envs, 1), -jnp.inf),
            # The next step's action, unknown yet: no TargetBC penalty.
            "next_action": jnp.full(buffer_action.shape, jnp.nan, jnp.float32),
        }
        if store_hidden:
            row["actor_carry"] = _pre_actor_carry_flat
        # The fields the buffer holds: those a reader asked for
        # (init_buffer), or all a checkpoint from before kept.
        schema = buffer_state.experience
        buffer_state = buffer.add(
            buffer_state, {k: v for k, v in row.items() if k in schema}
        )

    transition = Transition(
        obs=agent_state.collector_state.last_obs,
        action=action,
        raw_action=raw_action,
        reward=reward[:, None],
        terminated=terminated[:, None],
        truncated=truncated[:, None],
        raw_obs=raw_obs,
        next_obs=next_obs_for_buffer,
        log_prob=log_probs,
        a_expert=_a_expert_for_buf,
        next_a_expert=_next_a_expert_for_buf,
        raw_next_obs=(
            raw_final_obs.astype(jnp.float32) if store_raw_next_obs else None
        ),
    )

    # The Train return sums what the env paid, in its own units (a reward
    # normaliser keeps it in info), and the value box's bonus.
    paid = (
        info[RAW_REWARD_KEY].astype(jnp.float32) + entry_bonus[..., 0]
        if RAW_REWARD_KEY in info
        else reward
    )
    new_episodic_return_state, episodic_mean_return = update_episodic_return(
        agent_state.collector_state.episodic_return_state,
        paid,
        done=jnp.logical_or(terminated, truncated),
    )

    # Per-env step counter for the JSRL curriculum, None unless the agent
    # initialised it (int32 zeros of shape (n_envs,)).
    _prev_step_in_ep = agent_state.collector_state.step_in_episode
    if _prev_step_in_ep is not None:
        _ep_done = terminated.astype(jnp.bool_) | truncated.astype(jnp.bool_)
        _ep_done = _ep_done.reshape(_prev_step_in_ep.shape)
        _new_step_in_ep = jnp.where(
            _ep_done, jnp.zeros_like(_prev_step_in_ep), _prev_step_in_ep + 1
        )
    else:
        _new_step_in_ep = None

    new_collector_state = agent_state.collector_state.replace(
        rng=rng,
        _env_state=env_state,
        last_obs=new_last_obs,
        timestep=agent_state.collector_state.timestep + env_args.n_envs,
        last_terminated=terminated,
        last_truncated=truncated,
        episodic_return_state=new_episodic_return_state,
        episodic_mean_return=episodic_mean_return,
        buffer_state=buffer_state,
        last_expert_frac=jnp.mean(is_expert_flag),
        last_in_box=in_value_box.astype(jnp.float32),
        obs_norm_info=obs_norm_info,
        observer_states=observer_states,
        **({"expert_state": new_expert_state} if new_expert_state is not None else {}),
        **({"step_in_episode": _new_step_in_ep} if _new_step_in_ep is not None else {}),
    )

    agent_state = agent_state.replace(collector_state=new_collector_state, rng=rng)
    return agent_state, transition


@partial(
    jax.jit,
    static_argnames=[
        "expert_policy",
        "mode",
        "env_args",
        "n_timesteps",
        "augment_obs_with_expert_state",
    ],
)
def collect_experience_from_expert_policy(
    expert_policy: Any,
    rng: jax.Array,
    mode: str,
    env_args: EnvironmentConfig,
    n_timesteps: int = int(1e5),
    augment_obs_with_expert_state: bool = False,
    train_frac: Optional[float] = None,
) -> Transition:
    """Collect experience by running an expert policy in the environment.

    Stateful experts (those with an ``init_state`` attribute, e.g. PID) are
    called as ``expert_policy(state, obs) -> (action, new_state)`` and the
    state is reset on episode termination so the integrator restarts each
    episode. Stateless experts are called as ``expert_policy(obs) -> action``.
    The expert speaks env units (:class:`ExpertInAgentUnits` reads it in
    agent units, as the transitions store it) and its action is mapped back
    onto the env's bounds to step it.

    The observations are the live collector's: the training fraction
    ``train_frac`` appended when given, then, with
    ``augment_obs_with_expert_state``, the expert's state before its call
    at s and after it at s' (the state an episode end resets comes after).
    """
    expert_policy = ExpertInAgentUnits.wrap(
        expert_policy, env_args.env, env_args.env_params
    )
    reset_key, step_key, rng = jax.random.split(rng, 3)
    reset_keys = env_keys(reset_key, mode, env_args.n_envs)
    last_obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)

    expert_is_stateful = hasattr(expert_policy, "init_state")
    init_expert_state = (
        expert_policy.init_state(env_args.n_envs)
        if expert_is_stateful
        else jnp.zeros((1,))  # dummy carry; never read
    )

    def step_fn(carry, _):
        env_state, rng, last_obs, expert_state = carry
        # The step's key is also the one carried to the next step: the
        # stream the goldens pin.
        rng = jax.random.split(rng)[0]
        rng_step = env_keys(rng, mode, env_args.n_envs)

        # The expert reads the raw env obs (NOT normalized).
        raw_obs = get_raw_obs(
            env_state=env_state,
            env=env_args.env,
            mode=mode,
        )
        if expert_is_stateful:
            action, new_expert_state = expert_policy(expert_state, raw_obs)
        else:
            action = jax.vmap(expert_policy, in_axes=0)(raw_obs)
            new_expert_state = expert_state

        obsv, new_env_state, reward, terminated, truncated, info = (
            jax.lax.stop_gradient(
                step(
                    rng_step,
                    env_state,
                    agent_action_to_env(action, env_args.env, env_args.env_params),
                    env_args.env,
                    mode,
                    env_args.env_params,
                )
            )
        )

        # Build transition — prefer the env's final observation so truncation
        # bootstraps on V(s_final), not V(s_reset).
        next_obs = get_final_obs(info, obsv)
        # The expert's action there, on the observation in the environment's
        # units and its state after this step (the live collector's
        # next_a_expert), before an episode end resets the state.
        raw_next_obs = get_raw_final_obs(info, next_obs).astype(jnp.float32)
        if expert_is_stateful:
            next_action, _ = expert_policy(new_expert_state, raw_next_obs)
        else:
            next_action = jax.vmap(expert_policy, in_axes=0)(raw_next_obs)

        post_step_state = new_expert_state
        # Reset expert state on episode end so the integrator restarts
        # at the start of the next episode (autoreset has already produced
        # a fresh first obs in `obsv`).
        if expert_is_stateful:
            new_expert_state = tree_where_reset(
                jnp.logical_or(terminated, truncated),
                expert_policy.init_state(env_args.n_envs),
                new_expert_state,
            )

        # The live collector's layout (augment_obs): the training fraction,
        # then the expert's state before its call at s, after it (before an
        # episode end's reset) at s'.
        last_obs_aug = maybe_append_train_frac(last_obs, train_frac)
        next_obs_aug = maybe_append_train_frac(next_obs, train_frac)
        if augment_obs_with_expert_state and expert_is_stateful:
            last_obs_aug = augment_obs(last_obs_aug, flatten_expert_state(expert_state))
            next_obs_aug = augment_obs(
                next_obs_aug, flatten_expert_state(post_step_state)
            )

        transition = Transition(
            obs=last_obs_aug,
            action=action,
            # The env's own reward, as the live collector's replay rows.
            reward=info.get(RAW_REWARD_KEY, reward)[:, None],
            terminated=terminated[:, None],
            truncated=truncated[:, None],
            next_obs=next_obs_aug,
            raw_obs=raw_obs,
            a_expert=action,
            next_a_expert=next_action,
            raw_next_obs=raw_next_obs,
        )

        carry = (new_env_state, rng, obsv, new_expert_state)
        return carry, transition

    _, transitions = jax.lax.scan(
        step_fn,
        (env_state, step_key, last_obs, init_expert_state),
        length=n_timesteps,
    )

    return transitions


def preallocate_last_rollout(
    agent_state: BaseAgentState, length: int, **collect_kwargs: Any
) -> BaseAgentState:
    """``agent_state`` with a zero ``last_rollout`` shaped like one rollout.

    An agent exposing its latest rollout (``expose_recent_rollout``) writes
    it on ``agent_state.last_rollout`` every iteration. Allocating it before
    the first one keeps the scan carry's structure fixed (``None`` would
    become a :class:`Transition`). ``collect_kwargs`` are the agent's
    :func:`collect_experience` arguments; ``length`` its steps per rollout.
    """
    collect = partial(collect_experience, **collect_kwargs)
    _, rollout = jax.eval_shape(
        lambda state: jax.lax.scan(collect, state, xs=None, length=length),
        agent_state,
    )
    return agent_state.replace(last_rollout=zeros_like_abstract_pytree(rollout))


def init_collector_state(
    rng: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
    buffer: Optional[fbx.flat_buffer.TrajectoryBuffer] = None,
    window_size: int = 10,
    max_timesteps: Optional[int] = None,
    action_dim_override: Optional[int] = None,
    expert_state_aug_dim: int = 0,
    include_expert_fields: bool = False,
    actor_carry_dim: int = 0,
    include_raw_next_obs: bool = False,
    include_raw_obs: bool = False,
    include_is_expert: bool = False,
    expert_policy: Any = None,
    include_return_to_go: bool = False,
    include_next_action: bool = False,
):
    """Initialise the rollout collector. ``expert_state_aug_dim`` (>0)
    grows the buffered obs by that many trailing dimensions, holding the
    flattened expert state. Initial values are zero (matches the
    expert's `init_state`, which returns a zero pytree on this codebase
    for the PID expert). With the agent's ``expert_policy`` the replay
    keeps the expert's actions (``a_expert``, ``next_a_expert``) and the
    collector holds a stateful expert's initial state
    (:func:`with_expert_state`)."""
    include_expert_fields = include_expert_fields or expert_policy is not None
    last_done = jnp.zeros(env_args.n_envs)

    reset_key, rng = jax.random.split(rng)
    reset_keys = env_keys(reset_key, mode, env_args.n_envs)
    last_obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)
    obs_shape, action_shape = get_state_action_shapes(env_args.env)
    if action_dim_override is not None:
        action_shape = (action_dim_override,)

    # The training fraction's column (the schedule's observation), then the
    # expert's state, as collection appends them (augment_obs).
    add_train_frac = max_timesteps is not None
    last_obs = maybe_append_train_frac(last_obs, 0.0 if add_train_frac else None)
    if expert_state_aug_dim > 0:
        zero_aug = jnp.zeros((last_obs.shape[0], expert_state_aug_dim))
        last_obs = augment_obs(last_obs, zero_aug)
    extra = add_train_frac + expert_state_aug_dim
    # The observation features (env_args.observers) of the first
    # observations, from fresh observer and expert states at step 0.
    observer_states = None
    if env_args.observers:
        n_envs = env_args.n_envs
        stateful = expert_policy is not None and hasattr(expert_policy, "init_state")
        raw = get_raw_obs(env_state=env_state, env=env_args.env, mode=mode)
        inputs = ObserveInputs(
            raw.astype(jnp.float32),
            expert=expert_policy,
            expert_state=expert_policy.init_state(n_envs) if stateful else None,
            timestep=0,
            horizon=max_timesteps,
        )
        last_obs, observer_states = observe_features(
            last_obs,
            inputs,
            env_args.observers,
            init_observers(env_args.observers, n_envs),
        )
        extra += feature_width(env_args)
    if extra:
        obs_shape = (*obs_shape[:-1], obs_shape[-1] + extra)

    transition = Transition(
        obs=jnp.ones((env_args.n_envs, *obs_shape)),
        action=jnp.ones((env_args.n_envs, *action_shape)),
        raw_action=jnp.ones((env_args.n_envs, *action_shape)),
        next_obs=jnp.ones((env_args.n_envs, *obs_shape)),
        reward=jnp.ones((env_args.n_envs, 1)),
        terminated=jnp.ones((env_args.n_envs, 1)),
        truncated=jnp.ones((env_args.n_envs, 1)),
        log_prob=jnp.ones((env_args.n_envs, *action_shape)),
        raw_obs=jnp.ones((env_args.n_envs, *obs_shape)),
        # The expert action computed with the correct stateful expert
        # state at both s_t and s_{t+1}. The residual-RL TD target reads
        # next_a_expert.
        a_expert=jnp.zeros((env_args.n_envs, *action_shape)),
        next_a_expert=jnp.zeros((env_args.n_envs, *action_shape)),
    )
    episodic_return_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=jnp.zeros((env_args.n_envs, 1)),
        last_return=jnp.nan * jnp.zeros((env_args.n_envs, 1)),
    )

    buffer_state = (
        init_buffer(
            buffer,
            env_args,
            add_train_frac=add_train_frac,
            action_dim_override=action_dim_override,
            expert_state_aug_dim=expert_state_aug_dim,
            include_expert_fields=include_expert_fields,
            actor_carry_dim=actor_carry_dim,
            include_raw_next_obs=include_raw_next_obs,
            include_raw_obs=include_raw_obs,
            include_is_expert=include_is_expert,
            include_return_to_go=include_return_to_go,
            include_next_action=include_next_action,
        )
        if buffer is not None
        else None
    )
    obs_norm_info = None
    if env_args.normalize_obs_in_agent:
        # The span the statistics cover: all but the raw features.
        span = last_obs.shape[-1] - feature_width(env_args, normalised=False)
        obs_norm_info = init_agent_obs_norm(span)
    if env_args.store_normalized_obs:
        last_obs, obs_norm_info = update_obs_norm(
            last_obs, obs_norm_info, raw_obs_columns(env_args.env)
        )

    collector_state = CollectorState(
        rng=rng,
        _env_state=env_state,
        last_obs=last_obs,
        buffer_state=buffer_state,
        timestep=0,
        last_terminated=last_done,
        last_truncated=last_done,
        rollout=transition,
        episodic_return_state=episodic_return_state,
        max_timesteps=max_timesteps,
        last_in_box=jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32),
        obs_norm_info=obs_norm_info,
        observer_states=observer_states,
    )
    return with_expert_state(collector_state, expert_policy, env_args.n_envs)


def should_use_uniform_sampling(timestep: jax.Array, learning_starts: int) -> jax.Array:
    """Whether the collector acts uniformly: before ``learning_starts``."""
    return timestep < learning_starts
