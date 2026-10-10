from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol

import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.evaluate import evaluate
from ajax.state import BaseAgentState

if TYPE_CHECKING:
    from ajax.extensions.base import ExtensionStack


def compose_eval_metrics(
    user_fn: Optional[Callable],
    extension_stack: Optional["ExtensionStack"],
    total_timesteps: int,
) -> Optional[Callable]:
    """Compose a user ``extra_eval_metrics`` callable with the stack's
    :meth:`ExtensionStack.fold_eval_metrics`.

    Replaces the per-agent ``_wrap_extra_eval_metrics`` pattern. Returns
    ``None`` when both inputs are no-ops so ``evaluate_and_log``'s zero-
    overhead branch stays.

    The composed callable has signature ``(agent_state, rng) -> dict``,
    matching what :func:`evaluate_and_log` expects under the
    ``extra_eval_metrics`` static-arg.
    """
    has_stack = extension_stack is not None and bool(extension_stack.extensions)
    if user_fn is None and not has_stack:
        return None
    if user_fn is None:
        # Stack-only path.
        def stack_only(agent_state, rng):
            return extension_stack.fold_eval_metrics(
                agent_state, agent_state.collector_state.timestep, rng, total_timesteps
            )

        return stack_only
    if not has_stack:
        # User-only path — just return it directly.
        return user_fn

    def merged(agent_state, rng):
        out: dict = {}
        out.update(user_fn(agent_state, rng))
        out.update(
            extension_stack.fold_eval_metrics(
                agent_state, agent_state.collector_state.timestep, rng, total_timesteps
            )
        )
        return out

    return merged


def gated_log_callback(log_fn: Callable, flag: Any, metrics: dict, index: Any) -> None:
    """``jax.debug.callback`` that forwards to ``log_fn`` only when ``flag`` is set.

    The logging branch runs inside ``jax.lax.cond(flag, ...)``. Under ``vmap``
    with a *batched* predicate (the resume path batches the whole agent state
    across seeds, so anything derived from it is batched) JAX lowers the cond
    to a ``select``: both branches execute on every iteration and an
    unconditional ``debug.callback`` in the log branch fires every time. A
    fresh run keeps its step counters unbatched, which is why the bug only
    showed on resumed / curriculum training. Gating inside the callback makes
    the side effect correct in both lowerings (the callback batching rule
    invokes the Python function once per batch element with its own flag).
    """

    def _gated(flag, metrics, index):
        if bool(flag):
            log_fn(metrics, index)

    jax.debug.callback(_gated, flag, metrics, index)


def maybe_eval_and_log(
    agent_state: Any,
    aux: Any,
    index: Any,
    iteration: Any,
    *,
    metrics_fn: Callable[[Any, Any], dict],
    evaluate_fn: Callable[[Any, jax.Array], dict],
    extra_eval_metrics: Optional[Callable],
    log: bool,
    log_fn: Callable,
    log_frequency: Optional[int],
    per_update: int,
) -> tuple[Any, dict]:
    """Evaluate + log every ``log_frequency`` env steps, gated on the scan index.

    For agents that own their logging cadence (APG, the world-model
    agents). The gate is ``(iteration + 1) % every == 0`` with
    ``every = max(log_frequency // per_update, 1)`` scan iterations
    (``per_update`` = env steps per iteration, so ``log_frequency`` is
    rounded to a whole number of iterations). ``iteration`` is the scan
    input -- unbatched even when the agent state is batched across seeds
    (resume, curriculum) -- so the ``lax.cond`` stays a real cond and the
    evaluation only runs on the iterations that log. With an iteration
    offset (``build_resumable_train``) the cadence is absolute; without one
    it is relative to the start of this ``train`` call.

    On a logging iteration the metrics are ``metrics_fn(agent_state, aux)``
    (the agent's training metrics, e.g. ``timestep`` and losses) updated
    with ``evaluate_fn(agent_state, eval_key)`` and, when given,
    ``extra_eval_metrics(agent_state, extra_key)``, where ``eval_key,
    extra_key = split(agent_state.eval_rng)``; they are sent to ``log_fn``
    through :func:`gated_log_callback` and ``agent_state.n_logs`` is
    incremented. Otherwise the same structure is returned filled with NaN
    (``-1`` for integer leaves) and ``n_logs`` is unchanged. When logging is
    disabled (``log`` false or no ``log_frequency``) nothing is evaluated.

    Returns ``(agent_state, metrics)``.
    """
    enabled = log and bool(log_frequency)
    every = (
        max(int(log_frequency) // max(per_update, 1), 1)
        if enabled and log_frequency is not None
        else 1
    )
    flag = jnp.logical_and(enabled, (iteration + 1) % every == 0)

    def run(agent_state, aux, index):
        eval_key, extra_key = jax.random.split(agent_state.eval_rng)
        metrics = dict(metrics_fn(agent_state, aux))
        metrics.update(evaluate_fn(agent_state, eval_key))
        if extra_eval_metrics is not None:
            metrics.update(extra_eval_metrics(agent_state, extra_key))
        if log:
            # gated inside the callback: see gated_log_callback
            gated_log_callback(log_fn, flag, metrics, index)
        return metrics

    def skip(agent_state, aux, index):
        shapes = jax.eval_shape(run, agent_state, aux, index)
        return jax.tree.map(
            lambda s: (
                jnp.asarray(-1, s.dtype)
                if jnp.issubdtype(s.dtype, jnp.integer)
                else jnp.full(s.shape, jnp.nan, s.dtype)
            ),
            shapes,
        )

    if not enabled:
        return agent_state, skip(agent_state, aux, index)
    metrics = jax.lax.cond(flag, run, skip, agent_state, aux, index)
    agent_state = agent_state.replace(
        n_logs=jax.lax.select(flag, agent_state.n_logs + 1, agent_state.n_logs)
    )
    return agent_state, metrics


class AuxiliaryLogsProtocol(Protocol): ...


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return_dict = {}
    for key, val in d.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                if isinstance(subval, jax.Array):
                    # convert 0-d array to Python scalar
                    return_dict[f"{key}/{subkey}"] = (
                        subval[0] if jnp.ndim(subval) > 0 else subval
                    )
                else:
                    return_dict[f"{key}/{subkey}"] = subval
        else:
            if isinstance(val, jax.Array):
                return_dict[key] = val.item() if val.ndim == 0 else val
            else:
                return_dict[key] = val
    return return_dict


def _make_no_op(extra_eval_metrics=None):
    def no_op(agent_state, aux, *_args):
        fake_metrics_to_log = {
            "timestep": -1,  # must be int
            "Eval/episodic mean reward": jnp.nan,
            "Eval/episodic mean expert reward": jnp.nan,
            "Eval/expert bias": jnp.nan,
            "Eval/episodic entropy": jnp.nan,
            "Eval/mean average reward": jnp.nan,
            "Eval/mean episodic length": jnp.nan,
            "Eval/mean bias": jnp.nan,
            "Train/episodic mean reward": jnp.nan,
        }
        aux_keys = flatten_dict(to_state_dict(aux)).keys()
        fake_metrics_to_log.update(dict.fromkeys(aux_keys, jnp.nan))
        if extra_eval_metrics is not None:
            shape_tree = jax.eval_shape(
                extra_eval_metrics, agent_state, jax.random.PRNGKey(0)
            )
            extras_nan = jax.tree_util.tree_map(
                lambda s: jnp.full(s.shape, jnp.nan, s.dtype), shape_tree
            )
            fake_metrics_to_log.update(extras_nan)
        return fake_metrics_to_log

    return no_op


@partial(
    jax.jit,
    static_argnames=[
        "mode",
        "env_args",
        "num_episode_test",
        "recurrent",
        "log",
        "log_fn",
        "log_frequency",
        "total_timesteps",
        "avg_reward_mode",
        "expert_policy",
        "sweep",
        "eval_action_transform",
        "extra_eval_metrics",
        "pid_gain_policy",
        "augment_obs_with_expert_action",
        "augment_obs_with_expert_state",
    ],
)
def evaluate_and_log(
    agent_state: BaseAgentState,
    aux: AuxiliaryLogsProtocol,
    index: int,
    mode: str,
    env_args: int,
    num_episode_test: int,
    recurrent: bool,
    log: bool,
    log_fn: Callable,
    log_frequency: int,
    total_timesteps: int,
    avg_reward_mode: bool = False,
    expert_policy: Optional[Callable] = None,
    sweep: bool = False,
    train_frac: Optional[float] = None,
    eval_action_transform: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    pid_gain_policy: bool = False,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
):
    timestep = agent_state.collector_state.timestep

    log_flag = (
        timestep - (agent_state.n_logs * log_frequency) >= log_frequency
        if log
        else False
    )
    not_finished_flag = timestep <= total_timesteps if log_frequency else False

    if sweep:
        close_to_end_flag = timestep >= 0.8 * total_timesteps
    else:
        close_to_end_flag = True

    flag = jnp.logical_and(
        jnp.logical_and(log_flag, timestep > 1),
        not_finished_flag,
    )
    flag = jnp.logical_and(flag, close_to_end_flag)

    def run_and_log(
        agent_state: BaseAgentState, aux: AuxiliaryLogsProtocol, index: int
    ):
        # Deterministic evaluation: eval_rng is fixed at init_PPO and never
        # advanced across iterations, so each eval uses identical initial
        # conditions (per-seed).
        eval_key = agent_state.eval_rng
        # Key name MUST match what ``NormalizeVecObservation.update_state_*``
        # stores in info (see ``ajax/wrappers.py``). The wrapper writes
        # ``info["normalization_info"]``; previously this check looked
        # for ``"obs_normalization_info"`` which never matched -- so
        # ``norm_info`` was always ``None``, ``setup_environment``
        # rebuilt the eval env WITHOUT the normalizer, and the agent's
        # eval feed was RAW obs while training was NORMALISED. This
        # silently broke eval-vs-train alignment for any brax-stack env
        # using normalize_observations=True (PPO on mujoco_playground,
        # locomotion, etc.).
        obs_normalization = (
            "normalization_info" in agent_state.collector_state.env_state.info
            if mode == "brax"
            else "normalization_info" in dir(agent_state.collector_state.env_state)
        )
        (
            eval_rewards,
            eval_entropy,
            avg_avg_reward,
            avg_bias,
            step_count,
            expert_rewards,
        ) = evaluate(
            env_args.env,
            actor_state=agent_state.actor_state,
            num_episodes=num_episode_test,
            rng=eval_key,
            env_params=env_args.env_params,
            recurrent=recurrent,
            norm_info=(
                (
                    agent_state.collector_state.env_state.info["normalization_info"]
                    if mode == "brax"
                    else agent_state.collector_state.env_state.normalization_info
                )
                if obs_normalization
                else None
            ),
            avg_reward_mode=avg_reward_mode,
            expert_policy=expert_policy,
            train_frac=train_frac,
            eval_action_transform=eval_action_transform,
            agent_state=agent_state,
            pid_gain_policy=pid_gain_policy,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
        )
        metrics_to_log = {
            "timestep": timestep,
            "Eval/episodic mean reward": eval_rewards.mean(),
            "Eval/episodic mean expert reward": expert_rewards.mean(),
            "Eval/expert bias": eval_rewards.mean() - expert_rewards.mean(),
            "Eval/mean average reward": avg_avg_reward,
            "Eval/mean episodic length": step_count,
            "Eval/mean bias": avg_bias,
            "Eval/episodic entropy": eval_entropy,
            "Train/episodic mean reward": (
                agent_state.collector_state.episodic_mean_return
            ),
        }

        metrics_to_log.update(flatten_dict(to_state_dict(aux)))

        if extra_eval_metrics is not None:
            extra_key, _ = jax.random.split(eval_key)
            extras = extra_eval_metrics(agent_state, extra_key)
            metrics_to_log.update(extras)

        if log:
            gated_log_callback(log_fn, flag, metrics_to_log, index)

        return metrics_to_log

    no_op_branch = _make_no_op(extra_eval_metrics)
    metrics_to_log = jax.lax.cond(
        flag, run_and_log, no_op_branch, agent_state, aux, index
    )

    agent_state = agent_state.replace(
        n_logs=jax.lax.select(log_flag, agent_state.n_logs + 1, agent_state.n_logs)
    )

    del aux

    return agent_state, metrics_to_log
