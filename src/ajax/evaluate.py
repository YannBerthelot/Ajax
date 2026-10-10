from typing import Any, Callable, Optional, TypeVar

import jax
import jax.numpy as jnp
from gymnax.environments.environment import EnvParams
from jax.tree_util import Partial as partial

from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.interaction import get_pi, reset, step
from ajax.environments.row_collector import PolicyFn, check_unnormalized_env
from ajax.environments.system_class import env_params_is_batched
from ajax.environments.utils import (
    agent_action_to_env,
    agent_episode_length,
    check_env_is_gymnax,
    check_env_is_playground,
    check_if_environment_has_continuous_actions,
    env_action_repeat,
    get_env_type,
    get_raw_env,
)
from ajax.state import EnvironmentConfig
from ajax.wrappers import NormalizationInfo

T = TypeVar("T")  # generic type for pytrees


def repeat_first_entry(tree: T, num_repeats: int) -> T:
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x[0], (num_repeats, *x.shape[1:])), tree
    )
    return jax.tree.map(lambda x: jnp.repeat(x[0:1], repeats=num_repeats, axis=0), tree)


def setup_environment(
    env,
    env_params,
    num_episodes,
    norm_info,
    gamma,  # noqa: ARG001 -- positional in downstream calls; eval never normalises rewards
    action_repeat: int = 1,
    episode_length: Optional[int] = None,
    clip_actions: bool = True,
):
    """The training env's stack, rebuilt for evaluation (gymnax or brax).

    Ajax's own layers come off (:func:`~ajax.environments.create.strip_ajax_wrappers`)
    and go back on as training composed them
    (:func:`~ajax.environments.create.add_ajax_wrappers`): the action clip
    where training clipped (``clip_actions=False`` drops it, for callers
    that map actions to the env's bounds themselves, see
    :func:`ajax.environments.utils.agent_action_to_env`), the observation
    normaliser frozen at ``norm_info`` (``None``: no normaliser) and applied
    to the observations only where training applied it, rewards never
    normalised. The task under them -- the user's wrappers, the
    flattening -- is kept as trained on (gymnax).

    brax / playground tasks are rebuilt from their id with ``num_episodes``
    parallel envs. ``episode_length`` (simulator steps) and
    ``action_repeat`` are the training env's; the defaults (``None`` -> the
    env's native episode length, falling back to 1000; repeat 1) are the
    historical behaviour. gymnax envs do not support ``action_repeat > 1``.
    """
    from ajax.environments.create import (
        _build_brax_env,
        _build_playground_env,
        add_ajax_wrappers,
        strip_ajax_wrappers,
    )

    env, layers = strip_ajax_wrappers(env)
    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    continuous = check_if_environment_has_continuous_actions(env, env_params)

    if mode == "brax":
        ajax_env_id = getattr(env, "_ajax_env_id", None)
        if ajax_env_id is None:
            raw = get_raw_env(env)
            ajax_env_id = type(raw).__name__.lower()
        # Use the env's NATIVE episode_length where possible. Hardcoding
        # 1000 means eval episodes run up to 1000 steps even when the
        # env's own truncation is 150 (mujoco_playground manip) -- a
        # silent eval/train mismatch in episode-budget semantics.
        # Prefer the underlying env's ``_config.episode_length`` when
        # exposed (the playground/_make_panda_builder convention); fall
        # back to 1000 for envs without that attribute.
        _native_ep = getattr(
            getattr(env, "_config", None),
            "episode_length",
            None,
        )
        # Walk inward to find the original mujoco_playground env
        # (peeling our wrapper stack).
        _inner = env
        while _native_ep is None and _inner is not None:
            _inner = getattr(_inner, "env", None)
            _native_ep = getattr(
                getattr(_inner, "_config", None),
                "episode_length",
                None,
            )
        eval_ep_len = int(_native_ep) if _native_ep is not None else 1000
        if episode_length is not None:
            eval_ep_len = int(episode_length)
        if check_env_is_playground(env):
            env = _build_playground_env(
                ajax_env_id,
                n_envs=num_episodes,
                episode_length=eval_ep_len,
                action_repeat=action_repeat,
                fresh_reset=getattr(env, "_ajax_fresh_reset", True),
            )
        else:
            env = _build_brax_env(
                ajax_env_id,
                n_envs=num_episodes,
                episode_length=eval_ep_len,
                action_repeat=action_repeat,
            )
    elif action_repeat > 1:
        raise NotImplementedError(
            "action_repeat > 1 is not supported on gymnax envs (see"
            " ajax.environments.create.build_env_from_id)."
        )

    normaliser: dict = {}
    if norm_info is not None:
        normaliser = {
            "normalize_obs": norm_info.obs is not None,
            "apply_obs_normalization": layers.get("apply_obs_normalization", True),
            "train": False,
            "norm_info": repeat_first_entry(norm_info, num_repeats=num_episodes),
        }
    clip = layers.get("clip", False) and clip_actions
    return add_ajax_wrappers(env, clip=clip, **normaliser), mode, continuous


def get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous):
    """Return a function mapping (obs, done, hidden) → (action, entropy, hidden).

    In recurrent mode the hidden state is an explicit input/output so the
    eval loop can thread it through its carry (the live carry on
    ``actor_state`` belongs to the training envs and has the wrong batch
    size here). Non-recurrent mode passes ``hidden`` through untouched.
    """

    def fn(obs: jax.Array, done: Optional[jax.Array], hidden):
        if actor_state is None:
            raise ValueError("Actor not initialized.")
        state = actor_state.replace(hidden_state=hidden) if recurrent else actor_state
        pi, new_state = get_pi(state, actor_state.params, obs, done, recurrent)
        action = pi.mean() if continuous else pi.mode()
        entropy = (
            pi.unsquashed_entropy() if isinstance(pi, SquashedNormal) else pi.entropy()
        )
        if recurrent:
            # drop the single-step time axis added by get_pi
            action = action.squeeze(0)
            entropy = entropy.squeeze(0)
            return action, entropy, new_state.hidden_state
        return action, entropy, hidden

    return fn


def step_environment(
    mode,
    env,
    env_params,
    recurrent,
    actor_state,
    continuous,
    expert_policy=None,
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
            actor_hidden,
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
            augment_obs_with_expert_action or augment_obs_with_expert_state
        ) and expert_policy is not None
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

        raw_actions, entropy, new_actor_hidden = (
            get_deterministic_action_and_entropy_fn(
                actor_state, recurrent, continuous
            )(obs_for_actor, done if recurrent else None, actor_hidden)
        )

        if pid_gain_policy:
            gains = _anchor_gains * jnp.exp(_gain_log_scale * raw_actions)
            actions, new_expert_state = expert_policy.step_with_gains(
                expert_state, obs, gains
            )
        elif eval_action_transform is not None:
            if expert_policy is not None:
                if expert_is_stateful:
                    expert_actions, new_expert_state = expert_policy(expert_state, obs)
                else:
                    expert_actions = expert_policy(obs)
                    new_expert_state = expert_state
            else:
                expert_actions = 0.0
                new_expert_state = expert_state
            actions = eval_action_transform(
                raw_actions, expert_actions, obs, agent_state
            )
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
            actions,
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
            new_actor_hidden,
        )

    return fn


def step_environment_expert(mode, env, env_params, expert_policy):
    """Step function for expert policy. expert_policy must be a FunctionalExpertPolicy."""

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
            actor_hidden,  # unused by the expert; passed through
        ) = carry
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
            actor_hidden,
        )

    return fn


def _infer_max_eval_steps(env, env_params) -> int:
    """Derive an upper bound on eval rollout length from the env.

    Gymnax envs expose `max_steps_in_episode` via env_params; brax/playground
    envs expose `episode_length` through the EpisodeWrapper (propagated by
    __getattr__ through outer wrappers). That length counts simulator steps;
    with an action repeat an episode lasts ceil(episode_length / repeat)
    env.step calls (the length itself when the repeat is 1).
    """
    if env_params is not None and hasattr(env_params, "max_steps_in_episode"):
        return int(env_params.max_steps_in_episode)
    if hasattr(env, "episode_length"):
        return -(-int(env.episode_length) // env_action_repeat(env))
    return 1000


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "env_params",
        "num_episodes",
        "env",
        "avg_reward_mode",
        "expert_policy",
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
    gamma: float = 0.99,
    norm_info: Optional[NormalizationInfo] = None,
    avg_reward_mode: bool = False,
    num_steps_average_reward: int = int(1e4),
    expert_policy: Optional[Callable] = None,
    train_frac: Optional[float] = None,
    eval_action_transform: Optional[Callable] = None,
    max_eval_steps: Optional[int] = None,
    agent_state=None,
    pid_gain_policy: bool = False,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
) -> jax.Array:
    # Setup. The rebuild repeats each action as often as the training env
    # does (1 for every env built without a repeat: the default rebuild)
    # and runs its time limit: brax / playground envs carry it on the
    # EpisodeWrapper, gymnax envs in env_params.
    env, mode, continuous = setup_environment(
        env,
        env_params,
        num_episodes,
        norm_info,
        gamma,
        action_repeat=env_action_repeat(env),
        episode_length=(
            getattr(env, "episode_length", None)
            if get_env_type(env) == "brax"
            else None
        ),
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
    if _expert_is_stateful:
        assert expert_policy is not None
        _init_agent_expert_state = expert_policy.init_state(num_episodes)
    else:
        _init_agent_expert_state = jnp.zeros(
            (1,)
        )  # dummy; unused when expert is stateless
    # Fresh actor memory for evaluation: same structure as the live carry,
    # but batch-sized to num_episodes and zeroed (every supported memory
    # cell has a zero initial carry). Without this the eval loop would
    # reuse the training envs' carry, whose batch size doesn't even match.
    if (
        recurrent
        and actor_state is not None
        and getattr(actor_state, "hidden_state", None) is not None
    ):
        _init_actor_hidden = jax.tree.map(
            lambda x: jnp.zeros((num_episodes,) + x.shape[1:], x.dtype),
            actor_state.hidden_state,
        )
    else:
        _init_actor_hidden = jnp.zeros((1,))  # dummy; unused when feedforward

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
        _init_actor_hidden,  # actor memory (used only when recurrent)
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
        _init_actor_hidden,  # actor memory (pass-through)
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
        int(max_eval_steps)
        if max_eval_steps is not None
        else _infer_max_eval_steps(env, env_params)
    )

    def _scan_body(carry, _):
        return step_fn(carry), None

    final_carry, _ = jax.lax.scan(
        _scan_body, init_carry_agent, None, length=steps_bound
    )
    rewards, _, _, _, _, entropy_sum, step_count, step_count_2, _, _, _ = final_carry

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
            return carry, carry[8]  # per-step reward slot

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


def evaluate_policy(
    env_args: EnvironmentConfig,
    policy_fn: PolicyFn,
    init_carry_fn: Callable[[int], Any],
    num_episodes: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mean return and mean length of one episode in each of ``num_episodes`` envs.

    The evaluation protocol of the agents that own their collection (the
    world-model agents, DESIGN §5.4), for a *stateful* policy with the row
    collector's protocol ``policy_fn(carry, obs, is_first, key) -> (action,
    carry, extras)`` (extras ignored):

    * the env is rebuilt through :func:`setup_environment` with
      ``num_episodes`` parallel envs and the *training* ``action_repeat``
      and episode length, without the ``[-1, 1]`` action clip: actions go
      through :func:`ajax.environments.utils.agent_action_to_env`, as in
      the row collector, so train and eval act on the env identically;
    * the envs are reset with ``key`` (the same reset-key derivation as
      :func:`evaluate`, so both see the same initial states for one key);
    * the policy starts from ``init_carry_fn(num_episodes)`` (a zero carry)
      with ``is_first`` set on the first step only;
    * every env runs exactly one episode: a scan over the episode length in
      agent steps (:func:`ajax.environments.utils.agent_episode_length`)
      with rewards and lengths masked after each env's first ``done``.

    Preconditions (they raise): the training env does not normalise
    observations or rewards (the rebuild is the raw env, see
    :func:`ajax.environments.row_collector.check_unnormalized_env`), and
    ``env_params`` describes one system (per-env, batched params cannot be
    rebuilt with ``num_episodes`` envs; evaluate a nominal system instead).
    """
    env, env_params = env_args.env, env_args.env_params
    check_unnormalized_env(env, "evaluate_policy")
    if env_params_is_batched(env_params):
        raise ValueError(
            "evaluate_policy evaluates one system: pass unbatched env_params"
            " (e.g. a system class's nominal params), not per-env params."
        )
    # brax / playground envs carry their (simulator-step) episode length on
    # the EpisodeWrapper; gymnax envs carry it in env_params.
    train_episode_length = (
        getattr(env, "episode_length", None) if get_env_type(env) == "brax" else None
    )
    env, mode, _ = setup_environment(
        env,
        env_params,
        num_episodes,
        norm_info=None,
        gamma=0.99,  # unused without norm_info
        action_repeat=env_args.action_repeat,
        episode_length=train_episode_length,
        clip_actions=False,
    )
    horizon = agent_episode_length(env, env_params, env_args.action_repeat)

    def env_keys(key):
        return jax.random.split(key, num_episodes) if mode == "gymnax" else key

    key, reset_key = jax.random.split(key)
    obs, env_state = reset(env_keys(reset_key), env, mode, env_params)

    def body(carry, _):
        obs, env_state, policy_carry, is_first, done, ret, length, key = carry
        key, policy_key, step_key = jax.random.split(key, 3)
        action, policy_carry, _ = policy_fn(policy_carry, obs, is_first, policy_key)
        obs, env_state, reward, terminated, truncated, _ = step(
            env_keys(step_key),
            env_state,
            agent_action_to_env(action, env, env_params),
            env,
            mode,
            env_params,
        )
        running = jnp.logical_not(done)
        ret = ret + jnp.where(running, reward, 0.0)
        length = length + running.astype(jnp.float32)
        done = done | (terminated > 0) | (truncated > 0)
        return (
            obs,
            env_state,
            policy_carry,
            jnp.zeros_like(is_first),
            done,
            ret,
            length,
            key,
        ), None

    init = (
        obs,
        env_state,
        init_carry_fn(num_episodes),
        jnp.ones(num_episodes, bool),
        jnp.zeros(num_episodes, bool),
        jnp.zeros(num_episodes, jnp.float32),
        jnp.zeros(num_episodes, jnp.float32),
        key,
    )
    final, _ = jax.lax.scan(body, init, None, length=horizon)
    return final[5].mean(), final[6].mean()
