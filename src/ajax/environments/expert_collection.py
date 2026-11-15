"""Rows collected by an expert policy acting alone: a dataset for cloning,
offline training and prefilled replays, built with the collection's own
steps and features (ajax.environments.interaction).
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.controller import (
    ObserveInputs,
    init_observers,
    observe_features,
    previous_action_input,
    previous_reward_input,
)
from ajax.environments.interaction import (
    bootstrap_obs,
    get_final_obs,
    get_raw_final_obs,
    get_raw_obs,
    reset,
    step,
    step_features,
)
from ajax.environments.utils import (
    ExpertInAgentUnits,
    agent_action_to_env,
    env_keys,
    flatten_action_column,
)
from ajax.networks.memory import tree_where_reset
from ajax.random_keys import split_carry_first
from ajax.state import EnvironmentConfig, Transition
from ajax.wrappers import RAW_REWARD_KEY


@partial(
    jax.jit,
    static_argnames=[
        "expert_policy",
        "mode",
        "env_args",
        "n_timesteps",
        "bootstrap_next_obs",
    ],
)
def collect_experience_from_expert_policy(
    expert_policy: Any,
    rng: jax.Array,
    mode: str,
    env_args: EnvironmentConfig,
    n_timesteps: int = int(1e5),
    bootstrap_next_obs: bool = False,
) -> Transition:
    """Collect experience by running an expert policy in the environment.

    A transition's next observation is the final one at every episode end;
    with bootstrap_next_obs, the one the live collector's replay keeps
    (bootstrap_obs): the final one at a time limit, the reset one after a
    termination. Returns and episodes (record, MCPretrain, CloningPretrain)
    need the final one.

    Stateful experts (those with an ``init_state`` attribute, e.g. PID) are
    called as ``expert_policy(state, obs) -> (action, new_state)`` and the
    state is reset on episode termination so the integrator restarts each
    episode. Stateless experts are called as ``expert_policy(obs) -> action``.
    The expert speaks env units (:class:`ExpertInAgentUnits` reads it in
    agent units, as the transitions store it) and its action is mapped back
    onto the env's bounds to step it.

    The observations are the live collector's, their features
    (``env_args.observers``) included: the data comes before training, so a
    feature reading the training progress holds 0.
    """
    expert_policy = ExpertInAgentUnits.wrap(
        expert_policy, env_args.env, env_args.env_params
    )
    reset_key, carry_key = jax.random.split(rng)
    reset_keys = env_keys(reset_key, mode, env_args.n_envs)
    last_obs, env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)

    expert_is_stateful = hasattr(expert_policy, "init_state")
    init_expert_state = (
        expert_policy.init_state(env_args.n_envs)
        if expert_is_stateful
        else jnp.zeros((1,))  # dummy carry; never read
    )
    observers = env_args.observers
    before_training: dict[str, Any] = {"progress": 0.0}
    observer_states = None
    if observers:
        raw = get_raw_obs(env_state, env_args.env, mode, env_args.env_params)
        inputs = ObserveInputs(
            raw.astype(jnp.float32),
            expert_policy,
            init_expert_state if expert_is_stateful else None,
            previous_action=previous_action_input(
                observers, None, env_args.n_envs, env_args.env, env_args.env_params
            ),
            previous_reward=previous_reward_input(observers, None, env_args.n_envs),
            **before_training,
        )
        last_obs, observer_states = observe_features(
            last_obs, inputs, observers, init_observers(observers, env_args.n_envs)
        )

    def step_fn(carry, _):
        env_state, rng, last_obs, expert_state, observer_states = carry
        # The carry first, then the step's key, which no later step splits.
        rng, env_step_key = split_carry_first(rng, 1)
        rng_step = env_keys(env_step_key, mode, env_args.n_envs)

        # The expert reads the raw env obs (NOT normalized).
        raw_obs = get_raw_obs(env_state, env_args.env, mode, env_args.env_params)
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
                    agent_action_to_env(
                        flatten_action_column(action, env_args.continuous),
                        env_args.env,
                        env_args.env_params,
                    ),
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
        ended = jnp.logical_or(terminated, truncated)
        # Reset expert state on episode end so the integrator restarts
        # at the start of the next episode (autoreset has already produced
        # a fresh first obs in `obsv`).
        if expert_is_stateful:
            new_expert_state = tree_where_reset(
                ended, expert_policy.init_state(env_args.n_envs), new_expert_state
            )

        # The live collector's features: the transition's next observation's
        # from the continuing states, the next row's from fresh ones where
        # the episode ended.
        if observers:
            next_columns, final_columns, observer_states = step_features(
                env_args,
                mode,
                expert_policy,
                observer_states,
                new_env_state,
                raw_next_obs,
                post_step_state if expert_is_stateful else None,
                ended,
                action,
                info.get(RAW_REWARD_KEY, reward).astype(reward.dtype),
                **before_training,
            )
            obsv = jnp.concatenate([obsv, next_columns], axis=-1)
            next_obs = jnp.concatenate([next_obs, final_columns], axis=-1)
        if bootstrap_next_obs:
            next_obs = bootstrap_obs(obsv, next_obs, terminated, truncated)

        transition = Transition(
            obs=last_obs,
            action=action,
            # The env's own reward, as the live collector's replay rows.
            reward=info.get(RAW_REWARD_KEY, reward)[:, None],
            terminated=terminated[:, None],
            truncated=truncated[:, None],
            next_obs=next_obs,
            raw_obs=raw_obs,
            a_expert=action,
            next_a_expert=next_action,
            raw_next_obs=raw_next_obs,
        )

        carry = (new_env_state, rng, obsv, new_expert_state, observer_states)
        return carry, transition

    _, transitions = jax.lax.scan(
        step_fn,
        (env_state, carry_key, last_obs, init_expert_state, observer_states),
        length=n_timesteps,
    )

    return transitions
