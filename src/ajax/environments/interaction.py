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
from ajax.networks.networks import predict_value
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    LoadedTrainState,
    RollinEpisodicMeanRewardState,
    Transition,
)
from ajax.types import BufferType


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
        if action.ndim > 1:
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
    vmap_on = (
        max(jax.tree_util.tree_leaves(jax.tree.map(lambda x: x.ndim, env_state))) > 0
    )
    return (
        maybe_vmap(env.get_obs, vmap_on)(env_state)
        if mode == "gymnax"
        else maybe_vmap(env._get_obs, vmap_on)(env_state.pipeline_state)
    )


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
    static_argnames=[
        "recurrent",
        "mode",
        "env_args",
        "buffer",
        "expert_policy",
        "expert_fraction",
        "augment_obs_with_expert_action",
        "use_box",
        "use_expert_guided_exploration",
        "exploration_decay_frac",
        "exploration_boltzmann",
        "exploration_argmax",
        "use_residual_rl",
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
    expert_policy: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    action_scale: float = 1.0,
    gamma: Optional[float] = None,
    expert_fraction: float = 0.7,
    augment_obs_with_expert_action: bool = False,
    use_box: bool = False,
    box_v_min: float = 0.0,
    box_v_max: float = 0.0,
    total_timesteps: int = 1,
    use_expert_guided_exploration: bool = False,
    exploration_decay_frac: float = 0.30,
    exploration_tau: float = 1.0,
    exploration_boltzmann: bool = False,
    fixed_exploration_prob: float = 0.5,
    exploration_argmax: bool = False,
    use_residual_rl: bool = False,
) -> tuple[BaseAgentState, Transition]:
    """Collect one step of experience.

    During warmup (uniform=True):
      - If expert_policy is provided: randomly picks expert or uniform action
        (controlled by expert_fraction) for BOTH env step and buffer write,
        ensuring reward is consistent with the action stored.
      - If expert_policy is None (vanilla SAC): pure uniform random action.

    After warmup:
      - Policy action, with expert override inside the box when expert_policy
        is provided and the environment has a trunc_condition.
    """
    rng, action_key, step_key, mix_key = jax.random.split(agent_state.rng, 4)

    rng_step = (
        jax.random.split(step_key, env_args.n_envs) if mode == "gymnax" else step_key
    )

    # Raw obs (needed for expert policy and augmentation — must come before actor call)
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

    # --- Value-threshold box logic ---
    if use_box and agent_state.expert_critic_params is not None:
        # Compute effective threshold with curriculum
        train_frac = agent_state.collector_state.timestep / total_timesteps
        effective_threshold = box_v_min + (box_v_max - box_v_min) * train_frac

        # Compute V_expert using frozen critic
        obs_for_box = agent_state.collector_state.last_obs
        raw_for_box = raw_obs if raw_obs is not None else obs_for_box[..., :-1]
        a_box = jax.lax.stop_gradient(expert_policy(raw_for_box))
        v_box = jnp.min(
            predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.expert_critic_params,
                x=jnp.concatenate([obs_for_box, a_box], axis=-1),
            ),
            axis=0,
        )
        in_value_box = (v_box > effective_threshold)  # shape (n_envs, 1)

        last_in_box = agent_state.collector_state.last_in_box
        if last_in_box is None:
            last_in_box = jnp.zeros_like(in_value_box)

        # Terminal reward bonus on entry (outside→inside transition)
        entry_bonus = jnp.where(
            (last_in_box < 0.5) & (in_value_box > 0.5),
            v_box,
            jnp.zeros_like(v_box),
        )
    else:
        in_value_box = jnp.zeros(
            (env_args.n_envs, 1), dtype=jnp.float32
        )
        entry_bonus = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)

    # If obs augmentation is enabled, temporarily replace last_obs with the
    # augmented version so the actor sees [obs | a_expert | train_frac].
    # The buffer still receives the original (unaugmented) last_obs below.
    if augment_obs_with_expert_action and expert_policy is not None:
        _raw_for_aug = raw_obs if raw_obs is not None else agent_state.collector_state.last_obs
        _a_expert = jax.lax.stop_gradient(expert_policy(_raw_for_aug))
        _last_obs = agent_state.collector_state.last_obs
        # Layout: [env_obs | a_expert | train_frac]  (train_frac is the last dim)
        _augmented_obs = jnp.concatenate(
            [_last_obs[..., :-1], _a_expert, _last_obs[..., -1:]], axis=-1
        )
        _agent_state_for_actor = agent_state.replace(
            collector_state=agent_state.collector_state.replace(last_obs=_augmented_obs)
        )
    else:
        _agent_state_for_actor = agent_state

    # Policy action (used post-warmup)
    action, log_probs = get_action_and_log_probs(
        action_key=action_key,
        agent_state=_agent_state_for_actor,
        recurrent=recurrent,
        uniform=False,  # never blend here — we handle warmup explicitly below
    )

    # --- Compute env_action ---
    # During warmup: expert/uniform mix (or pure uniform if no expert)
    # After warmup: policy action with optional in-box expert override
    uniform_action = jax.random.uniform(
        mix_key, minval=-1.0, maxval=1.0, shape=action.shape
    )

    if expert_policy is not None:
        expert_action = jax.lax.stop_gradient(expert_policy(raw_obs))

        # In-box expert override for post-warmup steps
        in_box = (
            env_args.env.trunc_condition(
                agent_state.collector_state.env_state, env_args.env_params
            )
            if "trunc_condition" in dir(env_args.env)
            else jnp.zeros_like(action[..., :1])
        )
        if use_residual_rl:
            # Residual RL (Johannink et al.): policy outputs a residual correction
            # added on top of the expert action, then clipped to valid action range.
            post_warmup_action = jnp.clip(expert_action + action, -1.0, 1.0)
        else:
            post_warmup_action = (1 - in_box) * action + in_box * expert_action

        # Expert-guided exploration: value-gap-based stochastic substitution.
        # Uses frozen φ* (expert_critic_params) when available for a stable reference;
        # falls back to the live critic when φ* is not present (e.g. no MC pre-train).
        # Decays to zero after exploration_decay_frac of training.
        if use_expert_guided_exploration:
            obs_for_ege = agent_state.collector_state.last_obs
            # Use frozen φ* as the reference critic when available (Q2 ablation gate).
            ege_critic_params = (
                agent_state.expert_critic_params
                if agent_state.expert_critic_params is not None
                else agent_state.critic_state.params
            )
            q_policy = jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=ege_critic_params,
                    x=jnp.concatenate([obs_for_ege, action], axis=-1),
                ), axis=0,
            )
            q_expert_ege = jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=ege_critic_params,
                    x=jnp.concatenate([obs_for_ege, expert_action], axis=-1),
                ), axis=0,
            )
            gap = q_expert_ege - q_policy
            train_frac_ege = agent_state.collector_state.timestep / total_timesteps
            # decay_frac=0.0 means never decay (gate stays active for full training)
            if exploration_decay_frac == 0.0:
                decay = jnp.ones_like(train_frac_ege)
            else:
                decay = jnp.maximum(1.0 - train_frac_ege / exploration_decay_frac, 0.0)
            if exploration_argmax:
                # IBRL-style: deterministically pick expert when Q(s, π*(s)) > Q(s, π(s))
                use_expert_ege = (gap > 0.0) & (decay > 0.0)
            elif exploration_boltzmann:
                # Adaptive value-gap gate: p ∝ sigmoid(gap / τ·|Q|)
                q_scale = jax.lax.stop_gradient(jnp.abs(q_policy).mean() + 1e-6)
                p_expert_state = jax.nn.sigmoid(gap / (exploration_tau * q_scale))
                p_expert_final = decay * p_expert_state
                rng, ege_key = jax.random.split(rng)
                use_expert_ege = jax.random.uniform(ege_key, shape=p_expert_final.shape) < p_expert_final
            else:
                # Fixed-epsilon: constant probability regardless of value gap
                p_expert_final = decay * fixed_exploration_prob
                rng, ege_key = jax.random.split(rng)
                use_expert_ege = jax.random.uniform(ege_key, shape=p_expert_final.shape) < p_expert_final
            post_warmup_action = jnp.where(use_expert_ege, expert_action, post_warmup_action)

        # Warmup mix: expert_fraction of steps use expert, rest use uniform.
        # Residual RL uses pure uniform warmup (matching vanilla SAC) since the
        # expert is only meant to serve as a base for post-warmup residual actions.
        if use_residual_rl:
            warmup_action = uniform_action
            use_expert_this_step = jnp.zeros((), dtype=jnp.bool_)
        else:
            use_expert_this_step = jax.random.uniform(mix_key) < expert_fraction
            warmup_action = jnp.where(use_expert_this_step, expert_action, uniform_action)

        # Track whether the stored action came from the expert (for buffer logging)
        _post_expert = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        if use_expert_guided_exploration:
            _post_expert = jnp.maximum(_post_expert, use_expert_ege.astype(jnp.float32))
        if use_box:
            _post_expert = jnp.maximum(_post_expert, in_value_box.astype(jnp.float32))
        _warmup_expert = jnp.ones_like(action[..., :1], dtype=jnp.float32) * use_expert_this_step.astype(jnp.float32)
        is_expert_flag = jax.lax.cond(uniform, lambda: _warmup_expert, lambda: _post_expert)
    else:
        # No expert — vanilla SAC:
        #   warmup: pure uniform random
        #   post-warmup: policy action only (no in-box override)
        expert_action = jnp.zeros_like(action)  # never sent to env
        in_box = jnp.zeros_like(action[..., :1])
        post_warmup_action = action
        warmup_action = uniform_action
        is_expert_flag = jnp.zeros_like(action[..., :1], dtype=jnp.float32)

    env_action = jax.lax.cond(
        uniform,                          # True during warmup
        lambda: warmup_action,
        lambda: post_warmup_action,
    )

    # Inside box: override with expert action
    if use_box and expert_policy is not None:
        env_action = jnp.where(in_value_box, expert_action, env_action)

    buffer_action = env_action            # always store what the env actually received

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

    # Recompute in_box AFTER step for reward accumulation logic
    in_box_after = (
        env_args.env.trunc_condition(
            agent_state.collector_state.env_state, env_args.env_params
        )
        if "trunc_condition" in dir(env_args.env)
        else jnp.zeros_like(terminated)
    )

    raw_next_obs = info["obs_st"] if "obs_st" in info else obsv

    # --- Box reward/termination modification ---
    if use_box:
        reward = reward + entry_bonus[..., 0]
        terminated = jnp.logical_or(terminated.astype(bool), entry_bonus[..., 0] > 0).astype(terminated.dtype)

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

    # Transition for on-policy mix (keeps policy action, not env_action)
    transition = Transition(
        obs=agent_state.collector_state.last_obs,
        action=action,
        reward=reward[:, None],
        terminated=terminated[:, None],
        truncated=truncated[:, None],
        raw_obs=raw_obs,
        next_obs=raw_next_obs,
        log_prob=log_probs,
        inside_box=in_value_box if use_box else None,
    )

    new_episodic_return_state, episodic_mean_return = compute_episodic_reward_mean(
        agent_state=agent_state,
        reward=reward,
        done=jnp.logical_or(terminated, truncated),
        env=env_args.env,
        mode=mode,
    )

    new_collector_state = agent_state.collector_state.replace(
        rng=rng,
        _env_state=env_state,
        last_obs=obsv,
        timestep=agent_state.collector_state.timestep + env_args.n_envs,
        last_terminated=terminated,
        last_truncated=truncated,
        episodic_return_state=new_episodic_return_state,
        episodic_mean_return=episodic_mean_return,
        buffer_state=buffer_state,
        cumulative_reward=(
            in_box_after * (agent_state.collector_state.cumulative_reward + reward)
        ),
        last_in_box=in_value_box.astype(jnp.float32),
    )

    agent_state = agent_state.replace(collector_state=new_collector_state, rng=rng)
    return agent_state, transition


@partial(
    jax.jit,
    static_argnames=["expert_policy", "mode", "env_args", "n_timesteps"],
)
def collect_experience_from_expert_policy(
    expert_policy: Callable[[jnp.ndarray], jnp.ndarray],
    rng: jax.Array,
    mode: str,
    env_args: EnvironmentConfig,
    n_timesteps: int = int(1e5),
) -> Sequence[Transition]:
    """Collect experience by running an expert policy in the environment.

    Args:
        expert_policy: A function mapping observations to actions.
        rng: JAX random key.
        mode: Environment type ("gymnax" or "brax").
        env_args: Environment configuration.
        n_timesteps: Number of steps to run the policy.

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

    def step_fn(carry, _):
        env_state, rng, last_obs = carry
        # Prepare RNGs for stepping
        rng_step, new_rng = jax.random.split(rng)
        rng_step = (
            jax.random.split(rng_step, env_args.n_envs)
            if mode == "gymnax"
            else rng_step
        )

        # Compute action using expert policy, obs should NOT be normalized (unless expert_policy was designed using normed-obs)
        raw_obs = get_raw_obs(
            env_state=env_state,
            env=env_args.env,
            mode=mode,
        )
        action = jax.vmap(expert_policy, in_axes=0)(raw_obs)
        # Step environment and stop gradient to avoid backprop through env
        obsv, new_env_state, reward, terminated, truncated, info = (
            jax.lax.stop_gradient(
                step(
                    rng_step, env_state, action, env_args.env, mode, env_args.env_params
                )
            )
        )

        # Build transition
        transition = Transition(
            obs=last_obs,
            action=action,
            reward=reward[:, None],
            terminated=terminated[:, None],
            truncated=truncated[:, None],
            next_obs=obsv,
        )

        # Update carry with new environment state and RNG
        new_rng, _ = jax.random.split(rng)
        carry = (new_env_state, new_rng, obsv)
        return carry, transition

    # Run for n_timesteps and collect transitions
    _, transitions = jax.lax.scan(
        step_fn, (env_state, step_key, last_obs), length=n_timesteps
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
):
    last_done = jnp.zeros(env_args.n_envs)

    reset_key, rng = jax.random.split(rng)
    reset_keys = (
        jax.random.split(reset_key, env_args.n_envs) if mode == "gymnax" else reset_key
    )
    last_obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)
    obs_shape, action_shape = get_state_action_shapes(env_args.env)

    if max_timesteps is not None:
        _obs_shape = list(obs_shape)
        _obs_shape[-1] += 1
        obs_shape = tuple(
            _obs_shape
        )  # To account for the schedule observation for the agent
        new_col = jnp.full((last_obs.shape[0], 1), 0)
        last_obs = jnp.concatenate([last_obs, new_col], axis=-1)

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
        init_buffer(buffer, env_args, max_timesteps, add_train_frac=add_train_frac)
        if buffer is not None
        else None
    )
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
