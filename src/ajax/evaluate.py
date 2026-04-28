from typing import Callable, Optional, TypeVar

import jax
import jax.numpy as jnp
from gymnax.environments.environment import EnvParams
from jax.tree_util import Partial as partial

from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.interaction import get_pi, reset, step
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_env_is_playground,
    check_if_environment_has_continuous_actions,
    get_raw_env,
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

    if mode == "brax":
        from ajax.environments.create import (
            _build_brax_env,
            _build_playground_env,
        )

        ajax_env_id = getattr(env, "_ajax_env_id", None)
        if ajax_env_id is None:
            raw = get_raw_env(env)
            ajax_env_id = type(raw).__name__.lower()
        if check_env_is_playground(env):
            env = _build_playground_env(
                ajax_env_id, n_envs=num_episodes, episode_length=1000
            )
        else:
            env = _build_brax_env(
                ajax_env_id, n_envs=num_episodes, episode_length=1000
            )
        env = clip_wrapper(env)
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
    expert_handover: bool = False,
    train_frac: Optional[float] = None,
    eval_action_transform: Optional[Callable] = None,
    agent_state=None,
    pid_gain_policy: bool = False,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
):
    """Return a pure function for environment stepping.

    Carry is a 10-tuple; the last slot holds the expert's internal state
    (e.g. PID integrator). For stateless experts it's an unused dummy.

    pid_gain_policy: when True, the actor output is interpreted as PID gain
    modulation: env_action = expert.step_with_gains(state, obs, anchor*exp(ln10*a)).

    augment_obs_with_expert_action: when True, the actor receives
    [env_obs, expert_action] as input. The expert is queried statefully
    (matching training) so the policy sees the same augmented obs at eval
    that it saw during training.
    """

    expert_is_stateful = expert_policy is not None and hasattr(
        expert_policy, "init_state"
    )
    if pid_gain_policy:
        if not (expert_is_stateful and hasattr(expert_policy, "learnable_fields")):
            raise ValueError(
                "pid_gain_policy=True requires a stateful expert with learnable_fields."
            )
        _anchor_gains = expert_policy.anchor_gains
        _gain_log_scale = jnp.log(10.0)

    def fn(carry):
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
            expert_state,
        ) = carry
        rng, step_key = jax.random.split(rng)
        step_keys = (
            jax.random.split(step_key, obs.shape[0]) if mode == "gymnax" else step_key
        )

        # When augmenting obs with the expert action OR with the expert's
        # internal state, compute the (stateful) expert call BEFORE the
        # actor sees obs, so (a) the augmented obs matches the training-
        # time format, and (b) the expert_state carry advances every
        # step — matching the online action pipeline, which calls the
        # expert every step regardless of selection. Without this, an
        # expert_state-only augmentation eval keeps expert_state frozen
        # at zeros while training saw an evolving integrator, causing a
        # silent train/eval distribution mismatch.
        _need_expert_call = (
            (augment_obs_with_expert_action or augment_obs_with_expert_state)
            and expert_policy is not None
        )
        if _need_expert_call:
            if expert_is_stateful:
                _aug_expert_action, _aug_new_expert_state = expert_policy(
                    expert_state, obs
                )
            else:
                _aug_expert_action = expert_policy(obs)
                _aug_new_expert_state = expert_state
            if augment_obs_with_expert_action:
                obs_for_actor = jnp.concatenate(
                    [obs, jax.lax.stop_gradient(_aug_expert_action)], axis=-1
                )
            else:
                obs_for_actor = obs
        else:
            obs_for_actor = obs
            _aug_new_expert_state = None

        # Augment obs with the expert's flattened internal state
        # (PID integrator etc.) so the actor's input matches the
        # training-time augmented obs format. The state attached is
        # the BEFORE-expert state at this step, mirroring how the
        # collector stores it: collector_state.expert_state at obs t
        # is the state that has NOT yet seen obs t.
        if augment_obs_with_expert_state and expert_is_stateful:
            from ajax.environments.interaction import flatten_expert_state
            _es_flat = flatten_expert_state(expert_state)
            if _es_flat is not None:
                obs_for_actor = jnp.concatenate(
                    [obs_for_actor, jax.lax.stop_gradient(_es_flat)], axis=-1
                )

        raw_actions, entropy = get_deterministic_action_and_entropy_fn(
            actor_state, recurrent, continuous
        )(obs_for_actor, done if recurrent else None)

        if pid_gain_policy:
            gains = _anchor_gains * jnp.exp(_gain_log_scale * raw_actions)
            actions, new_expert_state = expert_policy.step_with_gains(
                expert_state, obs, gains
            )
        elif eval_action_transform is not None or early_termination_condition is not None:
            if expert_policy is not None:
                if expert_is_stateful:
                    expert_actions, new_expert_state = expert_policy(expert_state, obs)
                else:
                    expert_actions = expert_policy(obs)
                    new_expert_state = expert_state
            else:
                expert_actions = 0.0
                new_expert_state = expert_state
            inside_the_box = (
                early_termination_condition(state, env_params).reshape(-1, 1)
                if early_termination_condition is not None
                else 0.0
            )
            if eval_action_transform is not None:
                actions = eval_action_transform(
                    raw_actions, expert_actions, obs, agent_state
                )
            else:
                actions = (1.0 - inside_the_box) * raw_actions + inside_the_box * expert_actions
        else:
            actions = raw_actions
            # If we already advanced the expert state for obs augmentation,
            # use that updated state so the next step's augmented obs has
            # the correct PID integral; otherwise keep the carry as-is.
            new_expert_state = (
                _aug_new_expert_state
                if (
                    (augment_obs_with_expert_action or augment_obs_with_expert_state)
                    and _aug_new_expert_state is not None
                )
                else expert_state
            )
        obs, new_state, new_rewards, new_term, new_trunc, _ = step(
            step_keys,
            state,
            actions.squeeze(0) if recurrent else actions,
            env,
            mode,
            env_params,
        )
        if train_frac is not None:
            new_col = jnp.full((obs.shape[0], 1), train_frac)
            obs = jnp.concatenate([obs, new_col], axis=-1)

        new_done = jnp.logical_or(new_term, new_trunc)
        still_running = 1 - done

        # Reset the expert's internal state per-env when the episode just ended
        # (autoreset produces the fresh first obs of the next episode).
        if expert_is_stateful:
            zero_state = expert_policy.init_state(obs.shape[0])
            reset_mask = jnp.logical_or(new_term, new_trunc).astype(jnp.bool_)
            new_expert_state = jax.tree.map(
                lambda cur, zero: jnp.where(
                    reset_mask.reshape(
                        reset_mask.shape + (1,) * (cur.ndim - reset_mask.ndim)
                    ),
                    zero,
                    cur,
                ),
                new_expert_state,
                zero_state,
            )

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
            new_expert_state,
        )

    return fn


def step_environment_expert(mode, env, env_params, expert_policy):
    """Step function for expert policy. expert_policy must be a FunctionalExpertPolicy."""

    def fn(carry):
        rewards, rng, obs, done, state, entropy_sum, step_count, step_count_2, _, expert_state = carry
        rng, step_key = jax.random.split(rng)
        step_keys = (
            jax.random.split(step_key, obs.shape[0])
            if mode == "gymnax" and obs.ndim > 1
            else step_key
        )

        actions, new_expert_state = expert_policy(expert_state, obs)
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
            new_expert_state,
        )

    return fn


def while_env_not_done(carry):
    """Condition for while_loop."""
    done = carry[3]
    return jnp.logical_not(done.all())


def _infer_max_eval_steps(env, env_params) -> int:
    """Derive an upper bound on eval rollout length from the env.

    Gymnax envs expose `max_steps_in_episode` via env_params; brax/playground
    envs expose `episode_length` through the EpisodeWrapper (propagated by
    __getattr__ through outer wrappers).
    """
    if env_params is not None and hasattr(env_params, "max_steps_in_episode"):
        return int(env_params.max_steps_in_episode)
    if hasattr(env, "episode_length"):
        return int(env.episode_length)
    return 1000


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
        "action_scale",
        "early_termination_condition",
        "eval_action_transform",
        "max_eval_steps",
        "pid_gain_policy",
        "augment_obs_with_expert_action",
        "augment_obs_with_expert_state",
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
    action_scale: float = 1.0,
    early_termination_condition: Optional[Callable] = None,
    train_frac: Optional[float] = None,
    eval_action_transform: Optional[Callable] = None,
    max_eval_steps: Optional[int] = None,
    agent_state=None,
    pid_gain_policy: bool = False,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
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
    if train_frac is not None:
        new_col = jnp.full((obs.shape[0], 1), train_frac)
        obs_agent = jnp.concatenate([obs, new_col], axis=-1)
    else:
        obs_agent = obs

    # Initial carry
    _expert_is_stateful = expert_policy is not None and hasattr(
        expert_policy, "init_state"
    )
    _init_agent_expert_state = (
        expert_policy.init_state(num_episodes)
        if _expert_is_stateful
        else jnp.zeros((1,))  # dummy; unused when expert is stateless
    )
    init_carry_agent = (
        jnp.zeros(num_episodes),  # rewards
        key,
        obs_agent,
        jnp.zeros(num_episodes, dtype=jnp.int8),  # done
        state,
        jnp.zeros(1),  # entropy_sum
        jnp.zeros(1),  # step_count
        jnp.zeros(1),  # step_count_2
        jnp.zeros(num_episodes),  # last reward
        _init_agent_expert_state,  # expert_state (used only for stateful experts)
    )
    init_carry_expert = (
        jnp.zeros(num_episodes),  # rewards
        key,
        obs,
        jnp.zeros(num_episodes, dtype=jnp.int8),  # done
        state,
        jnp.zeros(1),  # entropy_sum
        jnp.zeros(1),  # step_count
        jnp.zeros(1),  # step_count_2
        jnp.zeros(num_episodes),  # last reward
        _init_agent_expert_state,  # expert_state
    )

    # Choose step function
    step_fn = step_environment(
        mode,
        env,
        env_params,
        recurrent,
        actor_state,
        continuous,
        expert_policy=expert_policy,
        action_scale=action_scale,
        early_termination_condition=early_termination_condition,
        train_frac=train_frac,
        eval_action_transform=eval_action_transform,
        agent_state=agent_state,
        pid_gain_policy=pid_gain_policy,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        augment_obs_with_expert_state=augment_obs_with_expert_state,
    )

    # Main loop. We use `scan` with a fixed length rather than `while_loop`
    # because mjx-warp kernels (mujoco_playground's physics backend) fail
    # `contact_dim` shape assertions under `vmap` + `while_loop` but compose
    # cleanly under `vmap` + `scan`. The step_fn already done-masks via
    # `still_running = 1 - done`, so iterating past the natural termination
    # of every lane is a no-op on the accumulated reward/entropy/step_count.
    steps_bound = (
        int(max_eval_steps) if max_eval_steps is not None
        else _infer_max_eval_steps(env, env_params)
    )

    def _scan_body(carry, _):
        return step_fn(carry), None

    final_carry, _ = jax.lax.scan(
        _scan_body, init_carry_agent, None, length=steps_bound
    )
    rewards, _, _, _, _, entropy_sum, step_count, step_count_2, _, _ = final_carry

    # Optionally compute expert comparison
    rewards_expert = jnp.nan
    if expert_policy is not None:
        expert_step_fn = step_environment_expert(mode, env, env_params, expert_policy)

        def _expert_scan_body(carry, _):
            return expert_step_fn(carry), None

        final_expert_carry, _ = jax.lax.scan(
            _expert_scan_body, init_carry_expert, None, length=steps_bound
        )
        rewards_expert = final_expert_carry[0]

    # Optional average-reward mode
    avg_reward, bias = jnp.nan, jnp.nan
    if avg_reward_mode:

        def scan_step(carry, _):
            carry = step_fn(carry)
            return carry, carry[-1]

        _, rewards_over_time = jax.lax.scan(
            scan_step, init_carry_agent, None, length=num_steps_average_reward
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
