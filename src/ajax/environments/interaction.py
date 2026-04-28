from typing import Any, Callable, Optional, Sequence, Tuple

import chex
import distrax
import flashbax as fbx
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial

from ajax.buffers.utils import init_buffer
from ajax.environments.utils import get_state_action_shapes, maybe_append_train_frac
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    LoadedTrainState,
    RollinEpisodicMeanRewardState,
    Transition,
)
from ajax.types import BufferType


def flatten_expert_state(expert_state) -> jnp.ndarray:
    """Pack a stateful expert's pytree state into a flat (..., dim) array.

    Used to augment obs with `expert_state` (the missing Markov component
    when the expert has hidden state — e.g. PID integrator). All leaves
    must have a leading batch axis; the function concatenates along the
    last axis. Returns a zero-shape array if the state is None or empty,
    so the caller can fall back gracefully for stateless experts.
    """
    if expert_state is None:
        return None
    leaves = jax.tree_util.tree_leaves(expert_state)
    if not leaves:
        return None
    cols = []
    for leaf in leaves:
        leaf = jnp.asarray(leaf)
        if leaf.ndim == 0:
            cols.append(leaf[None])
        elif leaf.ndim == 1:
            # (n_envs,) -> (n_envs, 1)
            cols.append(leaf[..., None])
        else:
            cols.append(leaf)
    return jnp.concatenate(cols, axis=-1)


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


def update_rolling_mean(
    state: RollinEpisodicMeanRewardState, new_value: jax.Array
) -> tuple[RollinEpisodicMeanRewardState, jax.Array]:
    rows = state.index.flatten()  # [2, 3]
    cols = jnp.arange(new_value.shape[0])  # column indices

    old_value = state.buffer[rows, cols]

    new_sum = state.sum - old_value + new_value  # remove oldest value to add new one
    buffer = state.buffer.at[rows, cols].set(new_value)
    new_index = (state.index + 1) % buffer.shape[0]
    new_count = jnp.minimum(state.count + 1, buffer.shape[0])

    new_state = RollinEpisodicMeanRewardState(
        buffer=buffer,
        index=new_index,
        count=new_count,
        sum=new_sum,
        cumulative_reward=state.cumulative_reward,
        last_return=state.last_return,
    )
    mean = new_sum / new_count
    return new_state, mean


@partial(jax.jit, static_argnames=["mode", "env"])
def reset(
    rng: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> tuple[jax.Array, EnvState]:
    """
    Reset the environment and return the initial observation and state.

    Args:
        rng (jax.Array): Random number generator key.
        env (Environment): Environment to reset.
        mode (str): Environment mode ("gymnax" or "brax").
        env_params (Optional[EnvParams]): Parameters for Gymnax environments.

    Returns:
        tuple[jax.Array, EnvState]: Initial observation and environment state.
    """
    if mode == "gymnax":
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(rng, env_params)
    else:
        env_state = env.reset(rng)  # ✅ no vmap
        obsv = env_state.obs
    return obsv, env_state


# @partial(
#     jax.jit,
#     static_argnames=["mode", "env"],
# )
def step(
    rng: jax.Array,
    state: jax.Array,
    action: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, jax.Array, Any]:
    """
    Perform a step in the environment.

    Args:
        rng (jax.Array): Random number generator key.
        state (jax.Array): Current environment state.
        action (jax.Array): Action to take.
        env (Environment): Environment to step in.
        mode (str): Environment mode ("gymnax" or "brax").
        env_params (Optional[EnvParams]): Parameters for Gymnax environments.

    Returns:
        Tuple[jax.Array, EnvState, jax.Array, jax.Array, Any]: Observation, new state, reward, done flag, and info.
    """
    if env_params is None and mode == "gymnax":
        env_params = env.default_params

    if mode == "gymnax":

        def step_wrapper(
            rng: jax.Array,
            state: EnvState,
            action: jax.Array,
            env_params: EnvParams,
        ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Any]:
            """Wrapper to step the environment."""
            return env.step(key=rng, state=state, action=action, params=env_params)

        time = state.time if hasattr(state, "time") else state.t
        truncated = time >= env_params.max_steps_in_episode - 1  # type: ignore[union-attr]
        if rng.ndim > 1:
            out = jax.vmap(
                step_wrapper,
                in_axes=(0, 0, 0, None),
            )(rng, state, action, env_params)
        else:
            out = step_wrapper(rng, state, action, env_params)

        # jax.debug.print("Action: {action}", action=action)
        if len(out) == 5:
            obsv, env_state, reward, done, info = out
            truncated = info["truncated"]
            # type: ignore[union-attr]
            terminated = done * (1 - truncated)
            terminated, truncated = jnp.float_(terminated), jnp.float_(truncated)
        else:
            obsv, env_state, reward, terminated, truncated, info = out
            terminated, truncated = jnp.float_(terminated), jnp.float_(truncated)
        if "normalization_info" in env_state.__dict__:
            obs_norm_info = jax.tree.map(
                lambda x: jnp.broadcast_to(x.mean(axis=0, keepdims=True), x.shape),
                env_state.normalization_info.obs,
            )
            reward_norm_info = jax.tree.map(
                lambda x: jnp.broadcast_to(x.mean(axis=0, keepdims=True), x.shape),
                env_state.normalization_info.reward,
            )
            env_state = env_state.replace(
                normalization_info=env_state.normalization_info.replace(
                    obs=obs_norm_info, reward=reward_norm_info
                )
            )
    elif mode == "brax":  # ✅ no vmap for brax
        env_state = env.step(state=state, action=action)
        obsv, reward, done, info = (
            env_state.obs,
            env_state.reward,
            env_state.done,
            env_state.info,
        )
        # if "truncation" in info:
        truncated = env_state.info["truncation"]
        # else:
        #     truncated = jnp.zeros_like(done)
        terminated = done * (1 - truncated)

    else:
        raise ValueError(f"Unrecognized mode for step {mode}")

    return obsv, env_state, reward, terminated, truncated, info


def get_pi(
    actor_state: LoadedTrainState,
    actor_params: FrozenDict,
    obs: jax.Array,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
) -> Tuple[distrax.Distribution, LoadedTrainState]:
    """
    Get the policy distribution for the given observation and actor state.

    Args:
        actor_state (LoadedTrainState): Actor's train state.
        actor_params (FrozenDict): Parameters of the actor.
        obs (jax.Array): Current observation.
        done (Optional[jax.Array]): Done flags for recurrent mode.
        recurrent (bool): Whether the actor is recurrent.

    Returns:
        Tuple[distrax.Distribution, LoadedTrainState]: Policy distribution and updated actor state.
    """
    # Agent-side obs normalisation: applied here so all callers of get_pi
    # stay unchanged. Stats are synced into actor_state.obs_norm_info
    # after every online collection step. None disables normalisation.
    if getattr(actor_state, "obs_norm_info", None) is not None:
        from ajax.agents.obs_norm import apply_obs_norm
        obs = apply_obs_norm(obs, actor_state.obs_norm_info)
    obs = maybe_add_axis(obs, recurrent)
    done = maybe_add_axis(done, recurrent)
    if recurrent:
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


def maybe_add_axis(arr: jax.Array, recurrent: bool) -> jax.Array:
    """
    Add an axis to the array if in recurrent mode.

    Args:
        arr (jax.Array): Input array.
        recurrent (bool): Whether to add an axis.

    Returns:
        jax.Array: Array with an additional axis if recurrent.
    """
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
        rng (jax.Array): Random number generator key.
        agent_state (BaseAgentState): Current agent state.
        obs (jnp.ndarray): Current observation.
        done (Optional[jax.Array]): Done flags for recurrent mode.
        recurrent (bool): Whether the agent is recurrent.

    Returns:
        Tuple[jax.Array, BaseAgentState]: Action and updated agent state.

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
    action, log_probs = pi.sample_and_log_prob(seed=rng)

    return (
        action,
        log_probs,
        agent_state.replace(actor_state=new_actor_state),
    )


def assert_shape(x, expected_shape, name="tensor"):
    assert (
        x.shape == expected_shape
    ), f"{name} has shape {x.shape}, expected {expected_shape}"


def compute_episodic_reward_mean(
    agent_state: BaseAgentState,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    env: Environment,
    mode: str,
) -> tuple[RollinEpisodicMeanRewardState, jnp.ndarray]:
    reward = get_raw_reward(reward, env, agent_state, mode)
    new_cumulative_reward = (
        agent_state.collector_state.episodic_return_state.cumulative_reward
        + reward.reshape(reward.shape[0], 1)
    )
    last_return = jax.lax.select(
        done.reshape(done.shape[0], 1),
        new_cumulative_reward,
        agent_state.collector_state.episodic_return_state.last_return,
    )  # nan if no episode has finished yet
    updated_episodic_return_state, updated_episodic_mean_return = update_rolling_mean(
        agent_state.collector_state.episodic_return_state,
        last_return,
    )
    previous_episodic_mean_return = (
        agent_state.collector_state.episodic_return_state.sum
        / agent_state.collector_state.episodic_return_state.count
    )

    episodic_mean_return = jax.lax.select(
        done.reshape(done.shape[0], 1),
        updated_episodic_mean_return,
        previous_episodic_mean_return,
    )

    def broadcast_and_select(done, new_val, old_val):
        # Compute broadcastable shape
        # `done` is (2,1) — reshape to [2, 1, 1, ..., 1] to match new_val.ndim
        extra_dims = new_val.ndim - done.ndim
        broadcast_shape = (1,) * extra_dims + done.shape
        done_broadcasted = jnp.reshape(done, broadcast_shape)
        # Broadcast done to new_val's shape (safely)
        done_broadcasted = jnp.broadcast_to(done_broadcasted, new_val.shape)

        return jax.lax.select(done_broadcasted, new_val, old_val)

    new_episodic_return_state = jax.tree_util.tree_map(
        lambda new_val, old_val: broadcast_and_select(
            done.reshape(done.shape[0], 1), new_val, old_val
        ),
        updated_episodic_return_state,
        agent_state.collector_state.episodic_return_state,
    )

    new_episodic_return_state = new_episodic_return_state.replace(
        cumulative_reward=new_cumulative_reward * (1 - done.reshape(done.shape[0], 1)),
        last_return=last_return,
    )

    return new_episodic_return_state, jnp.mean(episodic_mean_return)


def get_raw_reward(reward, env, agent_state, mode):
    if "unnormalize_reward" in dir(env):
        raw_reward = env.unnormalize_reward(
            reward,
            (
                agent_state.collector_state.env_state.norm_info.reward
                if mode == "gymnax"
                else agent_state.collector_state.env_state.info["norm_info"]
            ),
        )
    else:
        raw_reward = reward
    return raw_reward


def get_action_and_log_probs(
    action_key: chex.PRNGKey,
    agent_state: BaseAgentState,
    recurrent: bool,
    uniform: bool,
) -> Tuple[jax.Array, jax.Array]:
    action, log_probs, agent_state = get_action_and_new_agent_state(
        action_key,
        agent_state,
        agent_state.collector_state.last_obs,
        jnp.logical_or(
            agent_state.collector_state.last_terminated,
            agent_state.collector_state.last_truncated,
        ),
        recurrent=recurrent,
    )
    uniform_action = jax.random.uniform(
        key=action_key, minval=-1, maxval=1, shape=action.shape
    )  # TODO : add actual bounds
    return uniform * uniform_action + (1 - uniform) * action, log_probs


def maybe_vmap(f, vmap_on, **kwargs):
    if vmap_on:
        return jax.vmap(f, **kwargs)
    return f


def get_raw_obs(
    env_state: EnvState, env: Environment, mode: str
) -> Optional[jax.Array]:
    from ajax.environments.utils import check_env_is_playground, get_raw_env

    vmap_on = (
        max(jax.tree_util.tree_leaves(jax.tree.map(lambda x: x.ndim, env_state))) > 0
    )
    if mode == "gymnax":
        return maybe_vmap(env.get_obs, vmap_on)(env_state)
    if check_env_is_playground(env):
        import inspect

        raw = get_raw_env(env)
        n_params = len(inspect.signature(raw._get_obs).parameters)
        if n_params >= 2:
            return maybe_vmap(raw._get_obs, vmap_on)(env_state.data, env_state.info)
        return maybe_vmap(raw._get_obs, vmap_on)(env_state.data)
    # Brax: prefer env._get_obs for pre-normalization obs. Some minimal envs
    # (brax `fast`) don't expose `_get_obs`; fall back to env_state.obs.
    if not hasattr(get_raw_env(env), "_get_obs"):
        return env_state.obs
    return maybe_vmap(env._get_obs, vmap_on)(env_state.pipeline_state)


def identity(*args):
    return args


def return_first(*args):
    return args[0]


def get_buffer_action_and_env_action(
    rng: jax.Array,
    expert_action: jax.Array,
    policy_action: jax.Array,
    is_warmup: jax.Array,
    expert_fraction: float = 0.7,
) -> tuple[jax.Array, jax.Array]:
    """
    During warmup: randomly choose between expert and uniform for BOTH
    the environment step and the buffer write, so rewards are consistent.
    After warmup: use the policy action for both.

    Returns (env_action, buffer_action) — always the same during warmup,
    always policy_action after warmup.
    """
    mix_key, _ = jax.random.split(rng)
    uniform_action = jax.random.uniform(
        mix_key, minval=-1.0, maxval=1.0, shape=expert_action.shape
    )
    use_expert = jax.random.uniform(mix_key) < expert_fraction
    warmup_action = jnp.where(use_expert, expert_action, uniform_action)

    env_action = jax.lax.cond(
        is_warmup,
        lambda: warmup_action,
        lambda: policy_action,
    )
    return env_action, env_action  # buffer_action = env_action always


@partial(
    jax.jit,
    static_argnames=["recurrent", "mode", "env_args", "buffer", "action_pipeline"],
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
) -> tuple[BaseAgentState, Transition]:
    """Collect one step of experience.

    Args:
        action_pipeline: Optional callable that handles agent-specific action
            selection (obs augmentation, expert exploration, residual RL, etc.).
            When None, uses vanilla behavior: uniform during warmup, policy after.
            Signature: (agent_state, raw_obs, rng, uniform, mix_key, action_key)
                -> ActionPipelineResult(env_action, policy_action, log_probs,
                   is_expert_flag, in_value_box, entry_bonus, rng)
    """
    rng, action_key, step_key, mix_key = jax.random.split(agent_state.rng, 4)

    rng_step = (
        jax.random.split(step_key, env_args.n_envs) if mode == "gymnax" else step_key
    )

    # Update obs running stats with the current last_obs (BEFORE the
    # forward pass), then sync into actor/critic so the very first step
    # of training already uses non-zero stats. This avoids a step-1
    # divide-by-≈0 in apply_obs_norm.
    if agent_state.collector_state.obs_norm_info is not None:
        from ajax.agents.obs_norm import update_obs_norm
        _, _new_obs_norm = update_obs_norm(
            agent_state.collector_state.last_obs,
            agent_state.collector_state.obs_norm_info,
        )
        agent_state = agent_state.replace(
            collector_state=agent_state.collector_state.replace(
                obs_norm_info=_new_obs_norm
            ),
            actor_state=agent_state.actor_state.replace(obs_norm_info=_new_obs_norm),
            critic_state=agent_state.critic_state.replace(obs_norm_info=_new_obs_norm),
        )

    # Raw obs (needed by action_pipeline for expert policy)
    assert agent_state.collector_state.rollout is not None
    has_raw_obs = agent_state.collector_state.rollout.raw_obs is not None
    raw_obs = (
        get_raw_obs(
            env_state=agent_state.collector_state.env_state,
            env=env_args.env,
            mode=mode,
        )
        if has_raw_obs
        else None
    )

    if action_pipeline is not None:
        # Agent-specific action pipeline (SAC with expert, EDGE, box, etc.)
        result = action_pipeline(
            agent_state, raw_obs, rng, uniform, mix_key, action_key
        )
        env_action = result.env_action
        action = result.policy_action
        log_probs = result.log_probs
        is_expert_flag = result.is_expert_flag
        in_value_box = result.in_value_box
        entry_bonus = result.entry_bonus
        rng = result.rng
        new_expert_state = getattr(result, "new_expert_state", None)
        _buffer_action_override = getattr(result, "buffer_action", None)
        # Live LCB / Thompson telemetry — NaN if the gate didn't compute it.
        _live_q_advantage = getattr(result, "q_advantage", None)
        _live_sigma_actor = getattr(result, "critic_sigma_actor", None)
        _live_sigma_expert = getattr(result, "critic_sigma_expert", None)
    else:
        new_expert_state = None
        _buffer_action_override = None
        _live_q_advantage = None
        _live_sigma_actor = None
        _live_sigma_expert = None
        # Vanilla: uniform during warmup, policy action after
        action, log_probs = get_action_and_log_probs(
            action_key=action_key,
            agent_state=agent_state,
            recurrent=recurrent,
            uniform=False,
        )
        uniform_action = jax.random.uniform(
            mix_key, minval=-1.0, maxval=1.0, shape=action.shape
        )
        env_action = jax.lax.cond(uniform, lambda: uniform_action, lambda: action)
        is_expert_flag = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        in_value_box = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)
        entry_bonus = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)

    buffer_action = (
        _buffer_action_override if _buffer_action_override is not None else env_action
    )

    # --- Step environment ---
    obsv, env_state, reward, terminated, truncated, info = jax.lax.stop_gradient(
        step(
            rng_step,
            agent_state.collector_state.env_state,
            env_action,
            env_args.env,
            mode,
            env_args.env_params,
        )
    )

    # Append train_time_fraction to observation
    obsv = maybe_append_train_frac(
        obsv,
        train_frac=(
            agent_state.collector_state.train_time_fraction
            if agent_state.collector_state.max_timesteps is not None
            else None
        ),
    )

    # in_box_after for buffer write suppression and cumulative reward tracking
    in_box_after = (
        env_args.env.trunc_condition(
            agent_state.collector_state.env_state, env_args.env_params
        )
        if "trunc_condition" in dir(env_args.env)
        else jnp.zeros_like(terminated)
    )

    raw_next_obs = info.get("final_obs", info.get("obs_st", obsv))

    # Box reward/termination modification (no-op when entry_bonus is zeros)
    reward = reward + entry_bonus[..., 0]
    terminated = jnp.logical_or(
        terminated.astype(bool), entry_bonus[..., 0] > 0
    ).astype(terminated.dtype)

    # --- Buffer write ---
    buffer_state = agent_state.collector_state.buffer_state
    if buffer_state is not None and buffer is not None:
        _transition = {
            "obs": agent_state.collector_state.last_obs,
            "action": buffer_action,
            "reward": reward[:, None],
            "terminated": terminated[:, None],
            "truncated": truncated[:, None],
            "raw_obs": raw_obs,
            "is_expert": is_expert_flag,
        }
        should_write = jnp.logical_or(
            uniform,
            jnp.logical_not(
                in_box_after[0]
                * jnp.logical_not(jnp.logical_or(terminated, truncated))[0]
            ),
        )
        buffer_state = jax.lax.cond(
            should_write,
            buffer.add,
            return_first,
            agent_state.collector_state.buffer_state,
            _transition,
        )

    # If the env runs with augment_obs_with_expert_state, the
    # collector's last_obs is already augmented with the (BEFORE-expert)
    # expert_state. Augment next_obs symmetrically with the AFTER-expert
    # new_expert_state so the buffer stores consistent shapes.
    _next_state_aug = flatten_expert_state(new_expert_state)
    if (
        _next_state_aug is not None
        and agent_state.collector_state.last_obs.shape[-1]
        == raw_next_obs.shape[-1] + _next_state_aug.shape[-1]
    ):
        next_obs_for_buffer = jnp.concatenate(
            [raw_next_obs, _next_state_aug], axis=-1
        )
    else:
        next_obs_for_buffer = raw_next_obs

    # Transition for on-policy mix (keeps policy action, not env_action)
    transition = Transition(
        obs=agent_state.collector_state.last_obs,
        action=action,
        reward=reward[:, None],
        terminated=terminated[:, None],
        truncated=truncated[:, None],
        raw_obs=raw_obs,
        next_obs=next_obs_for_buffer,
        log_prob=log_probs,
        inside_box=in_value_box if action_pipeline is not None else None,
    )

    new_episodic_return_state, episodic_mean_return = compute_episodic_reward_mean(
        agent_state=agent_state,
        reward=reward,
        done=jnp.logical_or(terminated, truncated),
        env=env_args.env,
        mode=mode,
    )

    # If running with augment_obs_with_expert_state, the next iteration's
    # last_obs must carry the post-step (after-expert) expert_state so the
    # actor and critic see the right Markov state. Detect by shape parity
    # with the buffer-stored next_obs above.
    new_last_obs = (
        next_obs_for_buffer
        if next_obs_for_buffer.shape[-1] != obsv.shape[-1]
        else obsv
    )

    # Live gating telemetry (per-step batch means).
    _gate_diag_updates = {
        "last_expert_frac": jnp.mean(is_expert_flag),
    }
    if _live_q_advantage is not None:
        _gate_diag_updates["last_q_advantage"] = _live_q_advantage
    if _live_sigma_actor is not None:
        _gate_diag_updates["last_critic_sigma_actor"] = _live_sigma_actor
    if _live_sigma_expert is not None:
        _gate_diag_updates["last_critic_sigma_expert"] = _live_sigma_expert

    # Per-env step_in_episode counter for JSRL curriculum: increment
    # by 1 each step, reset to 0 on episode end (terminated|truncated).
    # Stored as int32 in collector_state.step_in_episode (None when
    # not initialised — callers that need it must initialise to zeros
    # of shape (n_envs,) at agent setup).
    _prev_step_in_ep = agent_state.collector_state.step_in_episode
    if _prev_step_in_ep is not None:
        _ep_done = (terminated.astype(jnp.bool_) | truncated.astype(jnp.bool_))
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
        cumulative_reward=(
            in_box_after * (agent_state.collector_state.cumulative_reward + reward)
        ),
        **_gate_diag_updates,
        **(
            {"last_in_box": in_value_box.astype(jnp.float32)}
            if action_pipeline is not None
            else {}
        ),
        **(
            {"expert_state": new_expert_state}
            if new_expert_state is not None
            else {}
        ),
        **(
            {"step_in_episode": _new_step_in_ep}
            if _new_step_in_ep is not None
            else {}
        ),
    )

    agent_state = agent_state.replace(collector_state=new_collector_state, rng=rng)
    return agent_state, transition


@partial(
    jax.jit,
    static_argnames=[
        "expert_policy", "mode", "env_args", "n_timesteps",
        "augment_obs_with_expert_state",
    ],
)
def collect_experience_from_expert_policy(
    expert_policy: Callable[[jnp.ndarray], jnp.ndarray],
    rng: jax.Array,
    mode: str,
    env_args: EnvironmentConfig,
    n_timesteps: int = int(1e5),
    augment_obs_with_expert_state: bool = False,
) -> Sequence[Transition]:
    """Collect experience by running an expert policy in the environment.

    Stateful experts (those with an ``init_state`` attribute, e.g. PID) are
    called as ``expert_policy(state, obs) -> (action, new_state)`` and the
    state is reset on episode termination so the integrator restarts each
    episode. Stateless experts are called as ``expert_policy(obs) -> action``.

    Returns:
        transitions: A `Transition` object containing all collected transitions.
    """
    # Split RNGs for environment reset and stepping
    reset_key, step_key, rng = jax.random.split(rng, 3)

    # Reset environment
    reset_keys = (
        jax.random.split(reset_key, env_args.n_envs) if mode == "gymnax" else reset_key
    )
    last_obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)

    expert_is_stateful = hasattr(expert_policy, "init_state")
    init_expert_state = (
        expert_policy.init_state(env_args.n_envs)
        if expert_is_stateful
        else jnp.zeros((1,))  # dummy carry; never read
    )

    def step_fn(carry, _):
        env_state, rng, last_obs, expert_state = carry
        # Prepare RNGs for stepping
        rng_step, new_rng = jax.random.split(rng)
        rng_step = (
            jax.random.split(rng_step, env_args.n_envs)
            if mode == "gymnax"
            else rng_step
        )

        # Compute action using expert policy. Stateful experts get the
        # running PID/etc state and return an updated state; stateless
        # experts ignore it. obs is the raw env obs (NOT normalized).
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

        # Step environment and stop gradient to avoid backprop through env
        obsv, new_env_state, reward, terminated, truncated, info = (
            jax.lax.stop_gradient(
                step(
                    rng_step, env_state, action, env_args.env, mode, env_args.env_params
                )
            )
        )

        # Reset expert state on episode end so the integrator restarts
        # at the start of the next episode (autoreset has already produced
        # a fresh first obs in `obsv`).
        if expert_is_stateful:
            zero_state = expert_policy.init_state(env_args.n_envs)
            done = jnp.logical_or(terminated, truncated)
            new_expert_state = jax.tree.map(
                lambda cur, zero: jnp.where(
                    done.reshape(done.shape + (1,) * (cur.ndim - done.ndim)),
                    zero,
                    cur,
                ),
                new_expert_state,
                zero_state,
            )

        # Build transition — prefer info['final_obs'] so truncation bootstraps
        # on V(s_final), not V(s_reset).
        next_obs = info.get("final_obs", info.get("obs_st", obsv))

        # When augment_obs_with_expert_state is on, mirror the live
        # collector: append the BEFORE-expert state to obs, and the
        # AFTER-expert state to next_obs. The actor was init with this
        # inflated obs dim, so BC must feed it the same shape.
        if augment_obs_with_expert_state and expert_is_stateful:
            cur_state_aug = flatten_expert_state(expert_state)
            next_state_aug = flatten_expert_state(new_expert_state)
            if cur_state_aug is not None:
                last_obs_aug = jnp.concatenate([last_obs, cur_state_aug], axis=-1)
                next_obs_aug = jnp.concatenate([next_obs, next_state_aug], axis=-1)
            else:
                last_obs_aug = last_obs
                next_obs_aug = next_obs
        else:
            last_obs_aug = last_obs
            next_obs_aug = next_obs

        transition = Transition(
            obs=last_obs_aug,
            action=action,
            reward=reward[:, None],
            terminated=terminated[:, None],
            truncated=truncated[:, None],
            next_obs=next_obs_aug,
        )

        new_rng, _ = jax.random.split(rng)
        carry = (new_env_state, new_rng, obsv, new_expert_state)
        return carry, transition

    # Run for n_timesteps and collect transitions
    _, transitions = jax.lax.scan(
        step_fn,
        (env_state, step_key, last_obs, init_expert_state),
        length=n_timesteps,
    )

    return transitions


# @partial(jax.jit, static_argnames=["mode", "env_args", "buffer", "window_size"])
def init_collector_state(
    rng: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
    buffer: Optional[fbx.flat_buffer.TrajectoryBuffer] = None,
    window_size: int = 10,
    max_timesteps: Optional[int] = None,
    action_dim_override: Optional[int] = None,
    expert_state_aug_dim: int = 0,
    normalize_obs_running: bool = False,
):
    """Initialise the rollout collector. ``expert_state_aug_dim`` (>0)
    grows the buffered obs by that many trailing dimensions, holding the
    flattened expert state. Initial values are zero (matches the
    expert's `init_state`, which returns a zero pytree on this codebase
    for the PID expert)."""
    last_done = jnp.zeros(env_args.n_envs)

    reset_key, rng = jax.random.split(rng)
    reset_keys = (
        jax.random.split(reset_key, env_args.n_envs) if mode == "gymnax" else reset_key
    )
    last_obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)
    obs_shape, action_shape = get_state_action_shapes(env_args.env)
    if action_dim_override is not None:
        action_shape = (action_dim_override,)

    if max_timesteps is not None:
        _obs_shape = list(obs_shape)
        _obs_shape[-1] += 1
        obs_shape = tuple(
            _obs_shape
        )  # To account for the schedule observation for the agent
        new_col = jnp.full((last_obs.shape[0], 1), 0)
        last_obs = jnp.concatenate([last_obs, new_col], axis=-1)

    if expert_state_aug_dim > 0:
        _obs_shape = list(obs_shape)
        _obs_shape[-1] += expert_state_aug_dim
        obs_shape = tuple(_obs_shape)
        zero_aug = jnp.zeros((last_obs.shape[0], expert_state_aug_dim))
        last_obs = jnp.concatenate([last_obs, zero_aug], axis=-1)

    transition = Transition(
        obs=jnp.ones((env_args.n_envs, *obs_shape)),
        action=jnp.ones((env_args.n_envs, *action_shape)),
        next_obs=jnp.ones((env_args.n_envs, *obs_shape)),
        reward=jnp.ones((env_args.n_envs, 1)),
        terminated=jnp.ones((env_args.n_envs, 1)),
        truncated=jnp.ones((env_args.n_envs, 1)),
        log_prob=jnp.ones((env_args.n_envs, *action_shape)),
        raw_obs=jnp.ones((env_args.n_envs, *obs_shape)),
    )
    episodic_return_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=jnp.zeros((env_args.n_envs, 1)),
        last_return=jnp.nan * jnp.zeros((env_args.n_envs, 1)),
    )

    add_train_frac = max_timesteps is not None

    buffer_state = (
        init_buffer(
            buffer, env_args, max_timesteps,
            add_train_frac=add_train_frac,
            action_dim_override=action_dim_override,
            expert_state_aug_dim=expert_state_aug_dim,
        )
        if buffer is not None
        else None
    )
    obs_norm_info = None
    if normalize_obs_running:
        from ajax.agents.obs_norm import init_agent_obs_norm
        obs_norm_info = init_agent_obs_norm(env_args.n_envs, last_obs.shape[-1])

    return CollectorState(
        rng=rng,
        _env_state=env_state,
        last_obs=last_obs,
        buffer_state=buffer_state,
        timestep=0,
        last_terminated=last_done,
        last_truncated=last_done,
        rollout=transition,
        episodic_return_state=episodic_return_state,
        cumulative_reward=jnp.zeros(
            (env_args.n_envs)
        ),  # TODO : switch to (env_args.n_envs,1)
        max_timesteps=max_timesteps,
        last_in_box=jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32),
        obs_norm_info=obs_norm_info,
    )


@jax.jit
def _select_uniform_action(uniform_action, _):
    return uniform_action


@jax.jit
def _select_policy_action(_, action):
    return action


@jax.jit
def _return_true(_):
    return True


@jax.jit
def _return_false(_):
    return False


@jax.jit
def should_use_uniform_sampling(timestep: jax.Array, learning_starts: int) -> bool:
    """Check if we should use uniform sampling based on timestep and learning starts.

    Args:
        timestep: Current timestep
        learning_starts: Number of timesteps before learning starts

    Returns:
        bool: Whether to use uniform sampling
    """

    return jnp.where(
        timestep < learning_starts, jnp.ones_like(timestep), jnp.zeros_like(timestep)
    )
