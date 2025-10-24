from typing import Callable, Optional, TypeVar

import jax
import jax.numpy as jnp
from brax.envs import create
from gymnax.environments.environment import EnvParams
from jax.tree_util import Partial as partial

from ajax.agents.SAC.utils import SquashedNormal
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


def setup_environment(env, env_params, num_episodes, norm_info, gamma):
    """Prepare and wrap the environment (gymnax or brax)."""
    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    clip_wrapper = ClipAction if mode == "gymnax" else ClipActionBrax
    norm_wrapper = (
        NormalizeVecObservationGymnax
        if mode == "gymnax"
        else NormalizeVecObservationBrax
    )
    continuous = check_if_environment_has_continuous_actions(env, env_params)

    env_name = (
        type(env.unwrapped).__name__.lower()
        if hasattr(env, "unwrapped")
        else type(env).__name__.lower()
    )

    if mode == "brax":
        env = clip_wrapper(create(env_name=env_name, batch_size=num_episodes))
    else:
        env = env.unwrapped if hasattr(env, "unwrapped") else env
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

    return env, mode, continuous


def get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous):
    """Return a function mapping obs → (action, entropy)."""

    def fn(obs: jax.Array, done: Optional[bool] = None):
        if actor_state is None:
            raise ValueError("Actor not initialized.")
        pi, _ = get_pi(actor_state, actor_state.params, obs, done, recurrent)
        action = pi.mean() if continuous else pi.mode()
        entropy = (
            pi.unsquashed_entropy() if isinstance(pi, SquashedNormal) else pi.entropy()
        )
        return action, entropy

    return fn


def step_environment(
    mode,
    env,
    env_params,
    recurrent,
    actor_state,
    continuous,
    expert_policy=None,
    action_scale=1.0,
    early_termination_condition=None,
):
    """Return a pure function for environment stepping."""

    def fn(carry):
        rewards, rng, obs, done, state, entropy_sum, step_count, step_count_2, _ = carry
        rng, step_key = jax.random.split(rng)
        step_keys = (
            jax.random.split(step_key, obs.shape[0]) if mode == "gymnax" else step_key
        )
        raw_actions, entropy = get_deterministic_action_and_entropy_fn(
            actor_state, recurrent, continuous
        )(obs, done if recurrent else None)
        expert_actions = expert_policy(obs) if expert_policy is not None else 0.0

        inside_the_box = (
            early_termination_condition(state, env_params).reshape(-1, 1)
            if early_termination_condition is not None
            else 0.0
        )
        actions = (1.0 - inside_the_box) * raw_actions * action_scale + expert_actions
        obs, new_state, new_rewards, new_term, new_trunc, _ = step(
            step_keys,
            state,
            actions.squeeze(0) if recurrent else actions,
            env,
            mode,
            env_params,
        )
        new_done = jnp.logical_or(new_term, new_trunc)
        still_running = 1 - done

        # distance_to_expert_action = abs(
        #     raw_actions * action_scale - expert_actions
        # )  # TODO : propagate to expose it in logs
        return (
            rewards + new_rewards * still_running,
            rng,
            obs,
            done | new_done,
            new_state,
            entropy_sum + (entropy.mean() * still_running).mean(),
            step_count + still_running.mean(),
            step_count_2 + 1,
            new_rewards,
        )

    return fn


def step_environment_expert(mode, env, env_params, expert_policy):
    """Step function for expert policy."""

    def fn(carry):
        rewards, rng, obs, done, state, entropy_sum, step_count, step_count_2, _ = carry
        rng, step_key = jax.random.split(rng)
        step_keys = (
            jax.random.split(step_key, obs.shape[0]) if mode == "gymnax" else step_key
        )
        actions = expert_policy(obs)
        obs, new_state, new_rewards, new_term, new_trunc, _ = step(
            step_keys, state, actions, env, mode, env_params
        )
        new_done = jnp.logical_or(new_term, new_trunc)
        still_running = 1 - done
        return (
            rewards + new_rewards * still_running,
            rng,
            obs,
            done | new_done,
            new_state,
            entropy_sum,
            step_count + still_running.mean(),
            step_count_2 + 1,
            new_rewards,
        )

    return fn


def while_env_not_done(carry):
    """Condition for while_loop."""
    done = carry[3]
    return jnp.logical_not(done.all())


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "env_params",
        "num_episodes",
        "lstm_hidden_size",
        "env",
        "avg_reward_mode",
        "expert_policy",
        "imitation_coef",
        "action_scale",
        "early_termination_condition",
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
    gamma: float = 0.99,
    norm_info: Optional[NormalizationInfo] = None,
    avg_reward_mode: bool = False,
    num_steps_average_reward: int = int(1e4),
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.0,
    action_scale: float = 1.0,
    early_termination_condition: Optional[Callable] = None,
) -> jax.Array:
    # Setup
    env, mode, continuous = setup_environment(
        env, env_params, num_episodes, norm_info, gamma
    )
    key, reset_key = jax.random.split(rng, 2)
    reset_keys = (
        jax.random.split(reset_key, num_episodes) if mode == "gymnax" else reset_key
    )
    obs, state = reset(reset_keys, env, mode, env_params)

    # Initial carry
    init_carry = (
        jnp.zeros(num_episodes),  # rewards
        key,
        obs,
        jnp.zeros(num_episodes, dtype=jnp.int8),  # done
        state,
        jnp.zeros(1),  # entropy_sum
        jnp.zeros(1),  # step_count
        jnp.zeros(1),  # step_count_2
        jnp.zeros(num_episodes),  # last reward
    )

    # Choose step function
    step_fn = step_environment(
        mode,
        env,
        env_params,
        recurrent,
        actor_state,
        continuous,
        expert_policy=expert_policy if imitation_coef is not None else None,
        action_scale=action_scale,
        early_termination_condition=early_termination_condition,
    )

    # Main loop
    rewards, _, _, _, _, entropy_sum, step_count, step_count_2, _ = jax.lax.while_loop(
        while_env_not_done, step_fn, init_carry
    )

    # Optionally compute expert comparison
    rewards_expert = jnp.nan
    if expert_policy is not None:
        rewards_expert, *_ = jax.lax.while_loop(
            while_env_not_done,
            step_environment_expert(mode, env, env_params, expert_policy),
            init_carry,
        )

    # Optional average-reward mode
    avg_reward, bias = jnp.nan, jnp.nan
    if avg_reward_mode:

        def scan_step(carry, _):
            carry = step_fn(carry)
            return carry, carry[-1]

        _, rewards_over_time = jax.lax.scan(
            scan_step, init_carry, None, length=num_steps_average_reward
        )
        avg_reward = rewards_over_time.mean(axis=0)
        bias = jnp.nansum(rewards_over_time - avg_reward, axis=0)

    avg_entropy = entropy_sum / jnp.maximum(step_count, 1.0)

    return (
        rewards.mean(axis=-1),
        avg_entropy.mean(axis=-1),
        jnp.nanmean(avg_reward),
        jnp.nanmean(bias),
        step_count.mean(),
        jnp.nanmean(rewards_expert) if expert_policy is not None else jnp.nan,
    )
