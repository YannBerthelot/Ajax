from typing import Optional, TypeVar

import jax
import jax.numpy as jnp
from brax.envs import create
from gymnax.environments.environment import EnvParams
from jax.tree_util import Partial as partial

from ajax.agents.sac.utils import SquashedNormal
from ajax.environments.interaction import get_pi, reset, step
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
from ajax.wrappers import (
    ClipAction,
    ClipActionBrax,
    NormalizationInfo,
    NormalizeVecObservationBrax,
    NormalizeVecObservationGymnax,
)

T = TypeVar("T")  # generic type for pytrees


def repeat_first_entry(tree: T, num_repeats: int) -> T:
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x[0], (num_repeats, *x.shape[1:])), tree
    )
    return jax.tree.map(lambda x: jnp.repeat(x[0:1], repeats=num_repeats, axis=0), tree)


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "env_params",
        "num_episodes",
        "lstm_hidden_size",
        "env",
        "avg_reward_mode",
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
    norm_info: Optional[NormalizationInfo] = None,
    avg_reward_mode: bool = False,
) -> jax.Array:
    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    if mode == "gymnax":
        clip_wrapper = ClipAction
        norm_wrapper = NormalizeVecObservationGymnax
    else:
        clip_wrapper = ClipActionBrax
        norm_wrapper = NormalizeVecObservationBrax
    continuous = check_if_environment_has_continuous_actions(env, env_params)

    env_name = (
        type(env.unwrapped).__name__.lower()
        if "unwrapped" in dir(env)
        else type(env).__name__.lower()
    )
    if mode == "brax":
        env = clip_wrapper(
            create(env_name=env_name, batch_size=num_episodes)
        )  # no need for autoreset with random init as we only done one episode, still need for normalization though
    else:
        env = env.unwrapped if "unwrapped" in dir(env) else env
        env = clip_wrapper(env)
    if norm_info is not None:
        norm_info = repeat_first_entry(norm_info, num_repeats=num_episodes)
        env = norm_wrapper(
            env,
            train=False,
            norm_info=norm_info,
            gamma=gamma,
            normalize_obs=norm_info.obs is not None,
            normalize_reward=False,
        )

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

        action = pi.mean() if continuous else pi.mode()

        # jax.debug.print(
        #     "Action: {action} Obs:{obs} State:{state}",
        #     action=action,
        #     obs=obs,
        #     state=actor_state.params,
        # )

        entropy = (
            pi.unsquashed_entropy() if isinstance(pi, SquashedNormal) else pi.entropy()
        )
        return action, entropy

    obs, state = reset(reset_keys, env, mode, env_params)

    done = jnp.zeros(num_episodes, dtype=jnp.int8)
    rewards = jnp.zeros(num_episodes)
    entropy_sum = jnp.zeros(1)
    step_count = jnp.zeros(1)
    step_count_2 = jnp.zeros(1)

    carry = (
        rewards,
        key,
        obs,
        done,
        state,
        entropy_sum,
        step_count,
        step_count_2,
        rewards,
    )

    def sample_action_and_step(carry):
        (
            rewards,
            rng,
            obs,
            done,
            state,
            entropy_sum,
            step_count,
            step_count_2,
            _,
        ) = carry
        rng, step_key = jax.random.split(rng)
        step_keys = (
            jax.random.split(step_key, num_episodes) if mode == "gymnax" else step_key
        )
        actions, entropy = get_deterministic_action_and_entropy(
            obs,
            done if recurrent else None,
        )
        # jax.debug.print("action:{x}, obs:{y}", x=actions, y=obs)
        obs, new_state, new_rewards, new_terminated, new_truncated, _ = step(
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
        step_count_2 += 1
        entropy_sum += (entropy.mean() * still_running).mean()
        rewards += new_rewards * still_running
        done = done | jnp.int8(new_done)
        # reward_buf = reward_buf.at[step_count_2.astype(int)].set(
        #     jnp.where(still_running, new_rewards, jnp.nan)
        # )
        return (
            rewards,
            rng,
            obs,
            done,
            new_state,
            entropy_sum,
            step_count,
            step_count_2,
            new_rewards,
        )

    def sample_action_and_step_scan(carry, _):
        carry = sample_action_and_step(carry)
        reward = carry[-1]
        return carry, reward

    def env_not_done(carry):
        done = carry[3]
        return jnp.logical_not(done.all())

    # if not avg_reward_mode:
    rewards, _, _, _, _, entropy_sum, step_count, step_count_2, _ = jax.lax.while_loop(
        env_not_done,
        sample_action_and_step,
        carry,
    )
    avg_entropy = entropy_sum / jnp.maximum(step_count, 1.0)  # avoid divide by zero
    bias = jnp.nan * jnp.ones(num_episodes)
    avg_reward = jnp.nan * jnp.ones(num_episodes)
    if avg_reward_mode:
        _, _rewards = jax.lax.scan(
            f=sample_action_and_step_scan,
            init=carry,
            xs=None,
            length=env_params.max_steps_in_episode,  # type: ignore[union-attr]
        )
        avg_reward = _rewards.mean(axis=0)
        bias = jnp.nansum(_rewards - avg_reward, axis=0)
        # avg_entropy = entropies.mean(axis=0)
    return (
        rewards.mean(axis=-1),
        avg_entropy.mean(axis=-1),
        avg_reward.mean(axis=-1),
        bias.mean(axis=-1),
        step_count.mean(),
    )
