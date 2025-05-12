import gc
import math
from typing import Optional

import jax
import jax.numpy as jnp
from brax.envs import create
from gymnax.environments.environment import EnvParams
from jax.tree_util import Partial as partial

from ajax.environments.interaction import get_pi, reset_env, step_env
from ajax.environments.utils import check_env_is_gymnax


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "env_params",
        "num_episodes",
        "lstm_hidden_size",
        "env",
    ],
)
def evaluate(
    env,
    actor_state,
    num_episodes: int,
    rng: jax.Array,
    env_params: Optional[EnvParams],
    recurrent: bool = False,
    lstm_hidden_size: Optional[int] = None,
    gamma: float = 0.99,  # TODO : propagate
) -> jax.Array:
    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    if mode == "brax":
        env_name = type(env.unwrapped).__name__.lower()
        env = create(
            env_name=env_name, batch_size=num_episodes
        )  # no need for autoreset with random init as we only done one episode
    key, reset_key = jax.random.split(rng, 2)
    reset_keys = (
        jax.random.split(reset_key, num_episodes) if mode == "gymnax" else reset_key
    )

    def get_deterministic_action_and_entropy(
        obs: jax.Array,
        done: Optional[bool] = None,
    ) -> tuple[jax.Array, jax.Array]:
        if actor_state is None:
            raise ValueError("Actor not initialized.")
        pi, _ = get_pi(
            actor_state,
            actor_state.params,
            obs,
            done,
            recurrent,
        )

        action = pi.mean()
        entropy = pi.entropy()
        return action, entropy

    obs, state = reset_env(reset_keys, env, mode, env_params)
    done = jnp.zeros(num_episodes, dtype=jnp.int8)
    rewards = jnp.zeros(num_episodes)
    entropy_sum = jnp.zeros(1)
    step_count = jnp.zeros(1)

    carry = (rewards, key, obs, done, state, entropy_sum, step_count)

    def sample_action_and_step_env(carry):
        rewards, rng, obs, done, state, entropy_sum, step_count = carry
        rng, step_key = jax.random.split(rng)
        step_keys = (
            jax.random.split(step_key, num_episodes) if mode == "gymnax" else step_key
        )
        actions, entropy = get_deterministic_action_and_entropy(
            obs,
            done if recurrent else None,
        )
        obs, state, new_rewards, new_terminated, new_truncated, _ = step_env(
            step_keys,
            state,
            actions.squeeze(0) if recurrent else actions,
            env,
            mode,
            env_params,
        )
        new_done = jnp.logical_or(new_terminated, new_truncated)

        still_running = 1 - done  # only count unfinished envs
        step_count += still_running.mean()
        entropy_sum += (entropy.mean() * still_running).mean()
        rewards += new_rewards * still_running
        done = done | jnp.int8(new_done)
        return rewards, rng, obs, done, state, entropy_sum, step_count

    def env_not_done(carry):
        done = carry[3]
        return jnp.logical_not(done.all())

    rewards, _, _, _, _, entropy_sum, step_count = jax.lax.while_loop(
        env_not_done,
        sample_action_and_step_env,
        carry,
    )

    avg_entropy = entropy_sum / jnp.maximum(step_count, 1.0)  # avoid divide by zero
    return rewards.mean(axis=-1), avg_entropy.mean(axis=-1)


@partial(
    jax.jit,
    static_argnames=["chunk_size", "horizon"],
)
def compute_episodic_mean_reward(agent_state, chunk_size=1000, horizon=10_000):
    buffer_state = agent_state.collector_state.buffer_state
    current_index = buffer_state.current_index
    start_idx = current_index - horizon

    reward_buffer = buffer_state.experience["reward"]  # [num_envs, T, 1]
    terminated_buffer = buffer_state.experience["terminated"]  # [num_envs, T, 1]
    truncated_buffer = buffer_state.experience["truncated"]  # [num_envs, T, 1]
    done_buffer = jnp.logical_or(
        terminated_buffer, truncated_buffer
    )  # TODO : check correctness
    num_envs = reward_buffer.shape[0]
    num_chunks = math.ceil(horizon / chunk_size)

    def masked_cumsum_include_done(rewards, dones):
        def step(carry, inputs):
            running_sum = carry
            reward, done = inputs
            new_sum = reward + running_sum
            next_carry = new_sum * (1.0 - done)
            return next_carry, new_sum

        init = jnp.zeros_like(rewards[0])
        _, out = jax.lax.scan(step, init=init, xs=(rewards, dones))
        return out

    def process_chunk(i, carry):
        reward_sum, done_count = carry

        chunk_start = start_idx + i * chunk_size
        max_available = current_index - chunk_start
        mask_len = jnp.minimum(chunk_size, max_available)

        # Always slice `chunk_size` elements and then apply a mask
        reward_chunk = jax.lax.dynamic_slice(
            reward_buffer,
            start_indices=(0, chunk_start, 0),
            slice_sizes=(num_envs, chunk_size, 1),
        ).squeeze(-1)

        done_chunk = jax.lax.dynamic_slice(
            done_buffer,
            start_indices=(0, chunk_start, 0),
            slice_sizes=(num_envs, chunk_size, 1),
        ).squeeze(-1)

        # Mask out the extra elements beyond the actual length of data in the chunk
        mask = jnp.arange(chunk_size) < mask_len  # [chunk_size] mask
        reward_chunk = reward_chunk * mask  # Apply mask to reward
        done_chunk = done_chunk * mask  # Apply mask to done

        # Apply masked cumulative sum over the rewards and dones
        masked = jax.vmap(masked_cumsum_include_done)(reward_chunk, done_chunk)

        reward_sum += jnp.sum(masked * done_chunk)
        done_count += jnp.sum(done_chunk)
        del masked, reward_chunk, done_chunk, mask
        gc.collect()
        jax.clear_caches()
        return reward_sum, done_count

    # Initialize accumulator for sum and count
    init_carry = (jnp.array(0.0), jnp.array(0.0))

    # Use `lax.fori_loop` to process chunks sequentially, minimizing memory use
    reward_sum, done_count = jax.lax.fori_loop(0, num_chunks, process_chunk, init_carry)

    # Compute the final episodic mean reward
    episodic_mean_reward = reward_sum / done_count
    return episodic_mean_reward
