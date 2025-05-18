from typing import Any, Optional, Tuple

import chex
import distrax
import flashbax as fbx
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial

from ajax.buffers.utils import init_buffer
from ajax.environments.utils import get_state_action_shapes
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
    state: RollinEpisodicMeanRewardState, new_value: float
) -> tuple[RollinEpisodicMeanRewardState, float]:
    rows = state.index.flatten()  # [2, 3]
    cols = jnp.array(range(new_value.shape[0]))  # column indices

    old_value = state.buffer[rows, cols]

    new_sum = state.sum - old_value + new_value
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


@partial(jax.jit, static_argnames=["mode", "env", "env_params"])
def reset_env(
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


@partial(
    jax.jit,
    static_argnames=["mode", "env", "env_params"],
)
def step_env(
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
    if mode == "gymnax":
        obsv, env_state, reward, done, info = jax.vmap(
            env.step,
            in_axes=(0, 0, 0, None),
        )(rng, state, action, env_params)
        truncated = env_state.time >= env_params.max_steps_in_episode  # type: ignore[union-attr]
        terminated = done * (1 - truncated)
        terminated, truncated = jnp.float_(terminated), jnp.float_(truncated)
    elif mode == "brax":  # ✅ no vmap for brax
        env_state = env.step(state, action)
        obsv, reward, done, info = (
            env_state.obs,
            env_state.reward,
            env_state.done,
            env_state.info,
        )
        truncated = env_state.info["truncation"]
        terminated = done * (1 - truncated)

    else:
        raise ValueError(f"Unrecognized mode for step_env {mode}")

    return obsv, env_state, reward, terminated, truncated, info


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
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


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
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
    agent_state: BaseAgentState, reward: jnp.ndarray, done: jnp.ndarray
) -> tuple[RollinEpisodicMeanRewardState, jnp.ndarray]:
    new_cumulative_reward = (
        agent_state.collector_state.episodic_return_state.cumulative_reward
        + reward.reshape(reward.shape[0], 1)
    )

    last_return = jax.lax.select(
        done.reshape(done.shape[0], 1),
        new_cumulative_reward,
        agent_state.collector_state.episodic_return_state.last_return,
    )
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


@partial(
    jax.jit,
    static_argnames=["recurrent", "mode", "env_args", "buffer"],
    # donate_argnums=0,
)
def collect_experience(
    agent_state: BaseAgentState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: Optional[BufferType] = None,
    uniform: bool = False,
):
    """Collect experience by interacting with the environment.

    Args:
        agent_state (BaseAgentState): Current agent state.
        _ (Any): Placeholder argument for compatibility.
        recurrent (bool): Whether the agent is recurrent.
        mode (str): The mode of the environment ("gymnax" or "brax").
        env_args (EnvironmentConfig): Configuration for the environment.
        buffer (fbx.flat_buffer.TrajectoryBuffer): Buffer to store trajectory data.

    Returns:
        Tuple[BaseAgentState, None]: Updated agent state and None.

    """
    rng, action_key, step_key = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)
    agent_action, log_probs, agent_state = get_action_and_new_agent_state(
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
        action_key,
        shape=agent_action.shape,
        minval=-1.0,
        maxval=1.0,
    )

    assert_shape(uniform_action, agent_action.shape)

    # Use jax.lax.cond to choose between uniform sampling and policy sampling
    action = jax.lax.cond(
        uniform,
        _select_uniform_action,
        _select_policy_action,
        uniform_action,
        agent_action,
    )

    rng_step = (
        jax.random.split(step_key, env_args.num_envs) if mode == "gymnax" else step_key
    )
    obsv, env_state, reward, terminated, truncated, info = jax.lax.stop_gradient(
        step_env(
            rng_step,
            agent_state.collector_state.env_state,
            action,
            env_args.env,
            mode,
            env_args.env_params,
        )
    )
    _transition = {
        "obs": agent_state.collector_state.last_obs,
        "action": action,  # if action.ndim == 2 else action[:, None]
        "reward": reward[:, None],
        "terminated": terminated[:, None],
        "truncated": truncated[:, None],
    }
    if buffer is not None:
        buffer_state = buffer.add(
            agent_state.collector_state.buffer_state,
            _transition,
        )
    else:
        _transition.update(
            {"next_obs": obsv, "log_prob": log_probs}
        )  # not included if using buffer to reduce weight, as flashbax can rebuild it.
        transition = Transition(**_transition)
    done = jnp.logical_or(terminated, truncated)

    new_episodic_return_state, episodic_mean_return = compute_episodic_reward_mean(
        agent_state, reward, done
    )

    new_collector_state = agent_state.collector_state.replace(
        rng=rng,
        env_state=env_state,
        last_obs=obsv,
        buffer_state=buffer_state if buffer is not None else None,
        timestep=agent_state.collector_state.timestep + 1,
        last_terminated=terminated,
        last_truncated=truncated,
        episodic_return_state=new_episodic_return_state,
        episodic_mean_return=episodic_mean_return,
    )
    agent_state = agent_state.replace(collector_state=new_collector_state)
    return agent_state, transition if buffer is None else None


@partial(jax.jit, static_argnames=["mode", "env_args", "buffer", "window_size"])
def init_collector_state(
    rng: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
    buffer: Optional[fbx.flat_buffer.TrajectoryBuffer] = None,
    window_size: int = 10,
):
    last_done = jnp.zeros(env_args.num_envs)

    reset_key, rng = jax.random.split(rng)
    reset_keys = (
        jax.random.split(reset_key, env_args.num_envs)
        if mode == "gymnax"
        else reset_key
    )
    last_obs, env_state = reset_env(reset_keys, env_args.env, mode, env_args.env_params)
    obs_shape, action_shape = get_state_action_shapes(env_args.env, env_args.env_params)
    transition = Transition(
        obs=jnp.ones((env_args.num_envs, *obs_shape)),
        action=jnp.ones((env_args.num_envs, *action_shape)),
        next_obs=jnp.ones((env_args.num_envs, *obs_shape)),
        reward=jnp.ones((env_args.num_envs, 1)),
        terminated=jnp.ones((env_args.num_envs, 1)),
        truncated=jnp.ones((env_args.num_envs, 1)),
        log_prob=jnp.ones((env_args.num_envs, *action_shape)),
    )
    episodic_return_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=jnp.zeros((env_args.num_envs, 1)),
        last_return=jnp.nan * jnp.zeros((env_args.num_envs, 1)),
    )
    return CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=last_obs,
        buffer_state=init_buffer(buffer, env_args) if buffer is not None else None,
        timestep=0,
        last_terminated=last_done,
        last_truncated=last_done,
        rollout=transition,
        episodic_return_state=episodic_return_state,
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
    return jax.lax.cond(
        timestep < learning_starts,
        _return_true,
        _return_false,
        operand=None,
    )
