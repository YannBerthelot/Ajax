import dataclasses
from typing import Any, Callable, NamedTuple, Optional, TypeVar

import jax
import jax.numpy as jnp
from gymnax.environments.environment import EnvParams
from jax.tree_util import Partial as partial

from ajax.environments.interaction import (
    augment_obs,
    flatten_expert_state,
    get_pi,
    get_raw_obs,
    reset,
    step,
)
from ajax.environments.row_collector import PolicyFn, check_unnormalized_env
from ajax.environments.system_class import env_params_is_batched
from ajax.environments.utils import (
    ExpertInAgentUnits,
    agent_action_to_env,
    agent_episode_length,
    check_env_is_playground,
    check_if_environment_has_continuous_actions,
    env_action_repeat,
    env_keys,
    get_env_type,
    get_raw_env,
    maybe_append_train_frac,
)
from ajax.networks.distributions import SquashedNormal
from ajax.networks.memory import tree_where_reset
from ajax.state import EnvironmentConfig
from ajax.utils import online_normalize
from ajax.wrappers import NormalizationInfo

T = TypeVar("T")  # generic type for pytrees

#: The steps the average-reward mode rolls out (the episodes restart).
_AVERAGE_REWARD_STEPS = 10_000


@dataclasses.dataclass(frozen=True)
class ValueEstimate:
    """An agent's critic, read for the value bias :func:`evaluate` returns:
    ``value(agent_state, obs, action)`` is its estimate (one per env) of the
    return from ``obs`` taking ``action``, of the rewards times
    ``reward_scale`` discounted by ``gamma``, as its targets are."""

    value: Callable[[Any, jax.Array, jax.Array], jax.Array]
    gamma: float
    reward_scale: float = 1.0


def normalised_reward(x: jax.Array, norm_info: Optional[Any]) -> jax.Array:
    """``x``, a reward or a sum of rewards in the env's units, in the
    agent's: divided by the scale the training env's reward normaliser
    holds (``norm_info.reward``, frozen), as collection divides each
    reward; unchanged without one."""
    stats = getattr(norm_info, "reward", None)
    if stats is None:
        return x
    return online_normalize(
        x, stats.count, stats.mean, stats.mean_2, train=False, shift=False
    )[0]


def _value_bias(
    values: jax.Array, rewards: jax.Array, running: jax.Array, gamma: float
) -> jax.Array:
    """The mean over the evaluated steps (``running``; each ``(steps,
    envs)``) of the value estimate minus the return realised from the step:
    its reward and the later ones, discounted by ``gamma``, to the episode's
    end."""

    def back(later: jax.Array, step: tuple) -> tuple[jax.Array, jax.Array]:
        reward, run = step
        realised = run * (reward + gamma * later)
        return realised, realised

    _, realised = jax.lax.scan(
        back, jnp.zeros_like(rewards[0]), (rewards, running), reverse=True
    )
    return jnp.sum(running * (values - realised)) / jnp.maximum(running.sum(), 1.0)


def repeat_first_entry(tree: T, num_repeats: int) -> T:
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x[0], (num_repeats, *x.shape[1:])), tree
    )


def setup_environment(
    env,
    env_params,
    num_episodes,
    norm_info,
    gamma,  # noqa: ARG001 -- positional in downstream calls; eval never normalises rewards
    action_repeat: int = 1,
    episode_length: Optional[int] = None,
):
    """The training env's stack, rebuilt for evaluation (gymnax or brax).

    Ajax's own layers come off (:func:`~ajax.environments.create.strip_ajax_wrappers`)
    and go back on as training composed them
    (:func:`~ajax.environments.create.add_ajax_wrappers`): the clip to a
    continuous action space's bounds, the observation normaliser frozen at
    ``norm_info`` (``None``: no normaliser) and applied to the observations
    only where training applied it, rewards never normalised. The task
    under them -- the user's wrappers, the flattening -- is kept as trained
    on (gymnax).

    brax / playground tasks are rebuilt from their id with ``num_episodes``
    parallel envs. ``episode_length`` (simulator steps) and
    ``action_repeat`` are the training env's; the defaults are ``None`` ->
    the env's native episode length, falling back to 1000, and repeat 1.
    gymnax envs do not support ``action_repeat > 1``.
    """
    from ajax.environments.create import (
        _build_brax_env,
        _build_playground_env,
        add_ajax_wrappers,
        strip_ajax_wrappers,
    )

    env, layers = strip_ajax_wrappers(env)
    mode = get_env_type(env)
    continuous = check_if_environment_has_continuous_actions(env, env_params)

    if mode == "brax":
        ajax_env_id = getattr(env, "_ajax_env_id", None)
        if ajax_env_id is None:
            raw = get_raw_env(env)
            ajax_env_id = type(raw).__name__.lower()
        # The env's native episode length (``_config.episode_length``, the
        # playground convention, found by peeling our wrappers), else 1000.
        # A fixed 1000 would run eval episodes past the env's own
        # truncation (150 on mujoco_playground manip), a silent eval/train
        # mismatch in episode budget.
        _native_ep = getattr(
            getattr(env, "_config", None),
            "episode_length",
            None,
        )
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
    return add_ajax_wrappers(env, **normaliser), mode, continuous


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


def _expert_obs(state, env, mode) -> jax.Array:
    """The observations an expert is given, as the collector gives them
    (:func:`~ajax.environments.interaction.get_raw_obs`): the environment's
    own, before any observation normalisation; an expert is defined in the
    environment's units."""
    return get_raw_obs(env_state=state, env=env, mode=mode).astype(jnp.float32)


class _EvalCarry(NamedTuple):
    """The evaluation scan's carry; per env unless noted."""

    rewards: jax.Array  # the return, accumulated until the env's first done
    rng: jax.Array
    obs: jax.Array
    done: jax.Array  # int8: the env's first done has come
    state: Any  # the env's
    entropy_sum: jax.Array  # (1,): summed over steps, meaned over running envs
    step_count: jax.Array  # (1,): the mean episode length so far
    last_reward: jax.Array  # this step's reward (the average-reward mode's)
    value: jax.Array  # this step's value estimate (the value bias's); else 0
    expert_state: Any  # a stateful expert's; a dummy otherwise
    actor_hidden: Any  # the actor's memory; a dummy when feedforward
    command: Any  # a command-conditioned actor's command; a dummy otherwise


def _init_eval_carry(
    num_episodes: int, key, obs, state, expert_state, actor_hidden=None, command=None
) -> _EvalCarry:
    return _EvalCarry(
        rewards=jnp.zeros(num_episodes),
        rng=key,
        obs=obs,
        done=jnp.zeros(num_episodes, dtype=jnp.int8),
        state=state,
        entropy_sum=jnp.zeros(1),
        step_count=jnp.zeros(1),
        last_reward=jnp.zeros(num_episodes),
        value=jnp.zeros(num_episodes),
        expert_state=expert_state,
        actor_hidden=jnp.zeros((1,)) if actor_hidden is None else actor_hidden,
        command=jnp.zeros((1,)) if command is None else command,
    )


def _init_expert_state(expert_policy, num_episodes: int):
    """A stateful expert's (one with ``init_state``) initial state; a dummy
    without one."""
    if hasattr(expert_policy, "init_state"):
        return expert_policy.init_state(num_episodes)
    return jnp.zeros((1,))


def _call_expert(expert_policy, expert_state, expert_obs):
    """``(action, state)``: a stateful expert called as ``expert_policy(state,
    obs)``, a stateless one as ``expert_policy(obs)``, its state kept."""
    if hasattr(expert_policy, "init_state"):
        return expert_policy(expert_state, expert_obs)
    return expert_policy(expert_obs), expert_state


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
    value_estimate: Optional[ValueEstimate] = None,
    command: Optional[Any] = None,
    norm_info: Optional[Any] = None,
):
    """Return a pure function stepping the evaluation (an :class:`_EvalCarry`).

    The expert acts in agent units (:class:`ExpertInAgentUnits`), as every
    action here; the executed one is mapped onto the env's bounds
    (:func:`agent_action_to_env`), as in collection.

    pid_gain_policy: when True, the actor output is interpreted as PID gain
    modulation: env_action = expert.step_with_gains(state, obs, anchor*exp(ln10*a)).

    augment_obs_with_expert_action: when True, the actor receives
    [env_obs, expert_action] as input. The expert is queried statefully
    (matching training) so the policy sees the same augmented obs at eval
    that it saw during training.

    value_estimate: the critic's estimate at the actor's input and the
    action collection stores, each step: the executed action in agent units,
    the actor's (the gains') under ``pid_gain_policy``.

    command: a command-conditioned actor's (see :func:`evaluate`): the
    actor reads ``command.actor_input(obs, c)``, and each step's reward, in
    the agent's units (``norm_info``, :func:`normalised_reward`), and end
    update ``c``.
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

    def fn(carry: _EvalCarry) -> _EvalCarry:
        obs, done, state = carry.obs, carry.done, carry.state
        expert_state = carry.expert_state
        rng, step_key = jax.random.split(carry.rng)
        step_keys = env_keys(step_key, mode, obs.shape[0])
        # The expert acts on the environment's own observations, as in
        # collection: not normalised, without the training-fraction column.
        expert_obs = (
            _expert_obs(state, env, mode) if expert_policy is not None else None
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
        if (
            augment_obs_with_expert_action or augment_obs_with_expert_state
        ) and expert_policy is not None:
            _aug_expert_action, _aug_new_expert_state = _call_expert(
                expert_policy, expert_state, expert_obs
            )
        else:
            _aug_expert_action, _aug_new_expert_state = None, None

        # The actor's input in the training layout (augment_obs): the
        # expert's flattened internal state (PID integrator etc.) BEFORE
        # this step's call, as the collector stores it (its expert_state at
        # obs t has not yet seen obs t), then the expert's action.
        obs_for_actor = augment_obs(
            obs,
            (
                flatten_expert_state(expert_state)
                if augment_obs_with_expert_state and expert_is_stateful
                else None
            ),
            _aug_expert_action if augment_obs_with_expert_action else None,
        )
        if command is not None:
            obs_for_actor = command.actor_input(obs_for_actor, carry.command)

        raw_actions, entropy, new_actor_hidden = (
            get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous)(
                obs_for_actor, done if recurrent else None, carry.actor_hidden
            )
        )

        if pid_gain_policy:
            gains = _anchor_gains * jnp.exp(_gain_log_scale * raw_actions)
            actions, new_expert_state = expert_policy.step_with_gains(
                expert_state, expert_obs, gains
            )
        elif eval_action_transform is not None:
            if expert_policy is not None:
                expert_actions, new_expert_state = _call_expert(
                    expert_policy, expert_state, expert_obs
                )
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
                expert_state if _aug_new_expert_state is None else _aug_new_expert_state
            )
        value = (
            carry.value
            if value_estimate is None
            else value_estimate.value(
                agent_state,
                obs_for_actor,
                raw_actions if pid_gain_policy else actions,
            )
        )
        obs, new_state, new_rewards, new_term, new_trunc, _ = step(
            step_keys,
            state,
            agent_action_to_env(actions, env, env_params),
            env,
            mode,
            env_params,
        )
        obs = maybe_append_train_frac(obs, train_frac)

        new_done = jnp.logical_or(new_term, new_trunc)
        still_running = 1 - done
        new_command = carry.command
        if command is not None:
            new_command = command.update(
                carry.command, normalised_reward(new_rewards, norm_info), new_done
            )

        # Reset the expert's internal state per-env when the episode just ended
        # (autoreset produces the fresh first obs of the next episode).
        if expert_is_stateful:
            new_expert_state = tree_where_reset(
                new_done, expert_policy.init_state(obs.shape[0]), new_expert_state
            )

        return _EvalCarry(
            rewards=carry.rewards + new_rewards * still_running,
            rng=rng,
            obs=obs,
            done=done | new_done,
            state=new_state,
            entropy_sum=carry.entropy_sum + (entropy.mean() * still_running).mean(),
            step_count=carry.step_count + still_running.mean(),
            last_reward=new_rewards,
            value=value,
            expert_state=new_expert_state,
            actor_hidden=new_actor_hidden,
            command=new_command,
        )

    return fn


def step_environment_expert(mode, env, env_params, expert_policy):
    """Step function for the expert's own rollout, called as the agent's
    rollout calls it (:func:`step_environment`): a stateful expert (one with
    ``init_state``) as ``expert_policy(state, obs) -> (action, state)``, a
    stateless one as ``expert_policy(obs) -> action``, in agent units and
    mapped onto the env's bounds."""

    def fn(carry: _EvalCarry) -> _EvalCarry:
        rng, step_key = jax.random.split(carry.rng)
        step_keys = (
            jax.random.split(step_key, carry.obs.shape[0])
            if mode == "gymnax" and carry.obs.ndim > 1
            else step_key
        )

        expert_obs = _expert_obs(carry.state, env, mode)  # as in step_environment
        actions, new_expert_state = _call_expert(
            expert_policy, carry.expert_state, expert_obs
        )
        obs, new_state, new_rewards, new_term, new_trunc, _ = step(
            step_keys,
            carry.state,
            agent_action_to_env(actions, env, env_params),
            env,
            mode,
            env_params,
        )
        new_done = jnp.logical_or(new_term, new_trunc)
        still_running = 1 - carry.done
        return carry._replace(
            rewards=carry.rewards + new_rewards * still_running,
            rng=rng,
            obs=obs,
            done=carry.done | new_done,
            state=new_state,
            step_count=carry.step_count + still_running.mean(),
            last_reward=new_rewards,
            expert_state=new_expert_state,
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


def _reset_eval_env(env, env_params, num_episodes, norm_info, rng, gamma=0.99):
    """The evaluation env, reset from ``rng``: ``(env, mode, continuous,
    key, obs, state)``, ``key`` the rollout's.

    The rebuild (:func:`setup_environment`) repeats each action as often as
    the training env does (1 for every env built without a repeat: the
    default rebuild) and runs its time limit: brax / playground envs carry
    it on the EpisodeWrapper, gymnax envs in env_params.
    """
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
    return env, mode, continuous, *_reset_envs(env, mode, env_params, num_episodes, rng)


def _reset_envs(env, mode, env_params, num_episodes, rng):
    """``(key, obs, state)``: the ``num_episodes`` envs reset from one half
    of ``rng`` (split per env on gymnax), ``key`` the other, the rollout's."""
    key, reset_key = jax.random.split(rng)
    reset_keys = env_keys(reset_key, mode, num_episodes)
    return (key, *reset(reset_keys, env, mode, env_params))


def _expert_return(
    mode, env, env_params, expert_policy, key, obs, state, num_episodes, length
):
    """The expert's mean return over ``num_episodes`` episodes from the
    reset ``(obs, state)``, ``length`` steps, each env's after its first
    ``done`` masked."""
    carry = _init_eval_carry(
        num_episodes, key, obs, state, _init_expert_state(expert_policy, num_episodes)
    )
    step_fn = step_environment_expert(mode, env, env_params, expert_policy)
    final, _ = jax.lax.scan(
        lambda carry, _: (step_fn(carry), None), carry, None, length=length
    )
    return jnp.nanmean(final.rewards)


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
        "eval_expert_policy",
        "value_estimate",
        "command",
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
    expert_policy: Optional[Callable] = None,
    train_frac: Optional[float] = None,
    eval_action_transform: Optional[Callable] = None,
    max_eval_steps: Optional[int] = None,
    agent_state=None,
    pid_gain_policy: bool = False,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
    eval_expert_policy: Optional[Callable] = None,
    value_estimate: Optional[ValueEstimate] = None,
    command: Optional[Any] = None,
) -> tuple[jax.Array | float, ...]:
    """Evaluate the actor's deterministic policy on ``num_episodes``
    episodes; with ``expert_policy`` (the training expert, which builds the
    actor's input) the expert's return too, that of ``eval_expert_policy``
    when given, on the same episodes (the same reset and key). The experts
    speak env units: they are read in agent units here
    (:class:`ExpertInAgentUnits`; an agent's are already).

    ``command`` evaluates a command-conditioned actor (UDRL's desired
    return and horizon): ``command.initial(agent_state, num_episodes)`` is
    each episode's first command, the actor reads ``command.actor_input(obs,
    c)``, and ``command.update(c, reward, done)`` follows each step, the
    reward in the agent's units, normalised as in training: the units of
    its targets.

    Returns the mean return, entropy, average reward (``avg_reward_mode``),
    value bias (``value_estimate``), episode length and expert's return,
    each NaN without what it needs. The value bias is the mean over the
    evaluated steps of the critic's estimate (:class:`ValueEstimate`) minus
    the discounted return realised from the step, in the critic's units:
    the rewards normalised as in training (:func:`normalised_reward`) and
    scaled. A recurrent critic's memory is not run here: NaN then.
    """
    expert_policy = ExpertInAgentUnits.wrap(expert_policy, env, env_params)
    eval_expert_policy = ExpertInAgentUnits.wrap(eval_expert_policy, env, env_params)
    env, mode, continuous, key, obs, state = _reset_eval_env(
        env, env_params, num_episodes, norm_info, rng, gamma
    )
    # Fresh actor memory for evaluation: same structure as the live carry,
    # but batch-sized to num_episodes and zeroed (every supported memory
    # cell has a zero initial carry). Without this the eval loop would
    # reuse the training envs' carry, whose batch size doesn't even match.
    actor_hidden = None  # feedforward: a dummy slot
    if (
        recurrent
        and actor_state is not None
        and getattr(actor_state, "hidden_state", None) is not None
    ):
        actor_hidden = jax.tree.map(
            lambda x: jnp.zeros((num_episodes,) + x.shape[1:], x.dtype),
            actor_state.hidden_state,
        )

    init_carry_agent = _init_eval_carry(
        num_episodes,
        key,
        maybe_append_train_frac(obs, train_frac),
        state,
        _init_expert_state(expert_policy, num_episodes),
        actor_hidden,
        None if command is None else command.initial(agent_state, num_episodes),
    )

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
        value_estimate=None if recurrent else value_estimate,
        command=command,
        norm_info=norm_info,
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
        stepped = step_fn(carry)
        evaluated = (1 - carry.done).astype(jnp.float32)
        return stepped, (stepped.value, stepped.last_reward, evaluated)

    final_carry, (values, step_rewards, evaluated) = jax.lax.scan(
        _scan_body, init_carry_agent, None, length=steps_bound
    )
    rewards, step_count = final_carry.rewards, final_carry.step_count

    bias: jax.Array | float = jnp.nan
    if value_estimate is not None and not recurrent:
        scale = value_estimate.reward_scale
        bias = _value_bias(
            values,
            scale * normalised_reward(step_rewards, norm_info),
            evaluated,
            value_estimate.gamma,
        )

    expert_reward: jax.Array | float = jnp.nan
    if expert_policy is not None or eval_expert_policy is not None:
        expert_reward = _expert_return(
            mode,
            env,
            env_params,
            eval_expert_policy or expert_policy,
            key,
            obs,
            state,
            num_episodes,
            steps_bound,
        )

    avg_reward = jnp.nan
    if avg_reward_mode:

        def scan_step(carry, _):
            carry = step_fn(carry)
            return carry, carry.last_reward

        _, rewards_over_time = jax.lax.scan(
            scan_step, init_carry_agent, None, length=_AVERAGE_REWARD_STEPS
        )
        avg_reward = rewards_over_time.mean(axis=0)

    avg_entropy = final_carry.entropy_sum / jnp.maximum(step_count, 1.0)

    return (
        rewards.mean(axis=-1),
        avg_entropy.mean(axis=-1),
        jnp.nanmean(avg_reward),
        jnp.asarray(bias, jnp.float32),
        step_count.mean(),
        expert_reward,
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
      and episode length; actions go through
      :func:`ajax.environments.utils.agent_action_to_env`, as in the row
      collector, so train and eval act on the env identically (the env's
      clip to its bounds is then a no-op);
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
    )
    horizon = agent_episode_length(env, env_params, env_args.action_repeat)

    key, obs, env_state = _reset_envs(env, mode, env_params, num_episodes, key)

    def body(carry, _):
        obs, env_state, policy_carry, is_first, done, ret, length, key = carry
        key, policy_key, step_key = jax.random.split(key, 3)
        action, policy_carry, _ = policy_fn(policy_carry, obs, is_first, policy_key)
        obs, env_state, reward, terminated, truncated, _ = step(
            env_keys(step_key, mode, num_episodes),
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
