from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.evaluate import evaluate
from ajax.state import BaseAgentState, EnvironmentConfig

if TYPE_CHECKING:
    from ajax.extensions.base import ExtensionStack


def compose_eval_metrics(
    extension_stack: "ExtensionStack", total_timesteps: int
) -> Optional[Callable]:
    """``(agent_state, rng) -> dict``: the stack's
    :meth:`ExtensionStack.fold_eval_metrics` at the collector's timestep, the
    ``extra_eval_metrics`` of :func:`evaluate_and_log` and
    :func:`maybe_eval_and_log`; ``None`` without extensions, so that their
    zero-overhead branch stays."""
    if not extension_stack:
        return None

    def eval_metrics(agent_state, rng):
        return extension_stack.fold_eval_metrics(
            agent_state, agent_state.collector_state.timestep, rng, total_timesteps
        )

    return eval_metrics


def unevaluated(shapes: Any, rows: tuple[int, ...] = ()) -> Any:
    """Metrics of the structure ``shapes`` (a :func:`jax.eval_shape` result)
    for no evaluation: -1 in the integer leaves (the timestep sentinel), NaN
    in the others; ``rows`` axes in front."""
    return jax.tree.map(
        lambda s: jnp.full(
            (*rows, *s.shape),
            -1 if jnp.issubdtype(s.dtype, jnp.integer) else jnp.nan,
            s.dtype,
        ),
        shapes,
    )


def gated_log_callback(log_fn: Callable, flag: Any, metrics: dict, index: Any) -> None:
    """``jax.debug.callback`` that forwards to ``log_fn`` only when ``flag`` is set.

    The logging branch runs inside ``jax.lax.cond(flag, ...)``. Under ``vmap``
    with a *batched* predicate (anything derived from a state batched across
    seeds) JAX lowers the cond to a ``select``: both branches execute on
    every iteration and an unconditional ``debug.callback`` in the log branch
    fires every time. Fresh and resumed runs keep the counters the gate reads
    unbatched (:func:`ajax.agents.base.shared_counters`); gating inside the
    callback keeps the side effect correct in both lowerings (the callback
    batching rule invokes the Python function once per batch element with
    its own flag).
    """

    def _gated(flag, metrics, index):
        if bool(flag):
            log_fn(metrics, index)

    jax.debug.callback(_gated, flag, metrics, index)


def evaluation_keys(agent_state: Any) -> tuple[jax.Array, jax.Array]:
    """``(eval_key, extra_key)``: the keys of the policy's evaluation and of
    the extensions' ``eval_metrics`` at the evaluation ``agent_state.n_logs``
    counts up to, ``split(fold_in(eval_rng, n_logs))``. Each evaluation
    draws fresh initial states and noise, deterministically per seed and
    apart from training (``eval_rng`` is split apart from ``rng``)."""
    eval_key, extra_key = jax.random.split(
        jax.random.fold_in(agent_state.eval_rng, agent_state.n_logs)
    )
    return eval_key, extra_key


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
    every: Optional[int],
) -> tuple[Any, Optional[dict]]:
    """Evaluate + log every ``every`` scan iterations, gated on the scan index.

    For agents with their own evaluation
    (:meth:`ajax.agents.loop.TrainLoop.evaluate_every`). The gate is
    ``(iteration + 1) % every == 0``. ``iteration`` is the scan
    input -- unbatched even when the agent state is batched across seeds
    (resume, curriculum) -- so the ``lax.cond`` stays a real cond and the
    evaluation only runs on the iterations that log. With an iteration
    offset (``build_resumable_train``) the cadence is absolute; without one
    it is relative to the start of this ``train`` call.

    On a logging iteration the metrics are ``metrics_fn(agent_state, aux)``
    (the agent's training metrics, e.g. ``timestep`` and losses) updated
    with ``evaluate_fn(agent_state, eval_key)`` and, when given,
    ``extra_eval_metrics(agent_state, extra_key)`` (:func:`evaluation_keys`:
    fresh at each evaluation); they are sent to ``log_fn``
    through :func:`gated_log_callback` and ``agent_state.n_logs`` is
    incremented. Otherwise the same structure is returned filled with NaN
    (``-1`` for integer leaves) and ``n_logs`` is unchanged; never so
    without ``every``. Without logging (``log`` false) the evaluation is not
    even traced (tracing alone rebuilds the evaluation env) and the metrics
    are ``None``.

    Returns ``(agent_state, metrics)``.
    """
    if not log:
        return agent_state, None
    enabled = bool(every)
    flag = jnp.logical_and(enabled, (iteration + 1) % (every or 1) == 0)

    def run(agent_state, aux, index):
        eval_key, extra_key = evaluation_keys(agent_state)
        metrics = dict(metrics_fn(agent_state, aux))
        metrics.update(evaluate_fn(agent_state, eval_key))
        if extra_eval_metrics is not None:
            metrics.update(extra_eval_metrics(agent_state, extra_key))
        # gated inside the callback: see gated_log_callback
        gated_log_callback(log_fn, flag, metrics, index)
        return metrics

    def skip(agent_state, aux, index):
        return unevaluated(jax.eval_shape(run, agent_state, aux, index))

    if not enabled:
        return agent_state, skip(agent_state, aux, index)
    metrics = jax.lax.cond(flag, run, skip, agent_state, aux, index)
    agent_state = agent_state.replace(
        n_logs=jax.lax.select(flag, agent_state.n_logs + 1, agent_state.n_logs)
    )
    return agent_state, metrics


def eval_norm_info(agent_state: Any, mode: str) -> Optional[Any]:
    """The observation statistics the evaluation env normalises with: those
    the training env's normaliser holds now, ``None`` without one.

    The key is the one ``NormalizeVecObservation.update_state_*`` stores
    (``ajax/wrappers.py``). Looking for another one (as this once did:
    ``"obs_normalization_info"``) evaluates on raw observations an agent
    trained on normalised ones.
    """
    env_state = agent_state.collector_state.env_state
    if mode == "brax":
        return env_state.info.get("normalization_info")
    return getattr(env_state, "normalization_info", None)


def _metric_value(name: str, value: Any) -> Any:
    """A logged array is one value: shape ``()``, the shape ``_make_no_op``
    fills the skipped logging branch with. A wider array raises (at trace
    time under jit) instead of being cut to its first element."""
    if not isinstance(value, jax.Array):
        return value
    if value.size != 1:
        raise ValueError(
            f"Logged metric {name!r} must hold one value, got shape {value.shape}"
        )
    return value.reshape(())


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Name each metric ``key`` or, one level down, ``key/subkey``."""
    flat = {}
    for key, val in d.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                name = f"{key}/{subkey}"
                flat[name] = _metric_value(name, subval)
        else:
            flat[key] = _metric_value(key, val)
    return flat


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
        "eval_expert_policy",
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
    aux: Any,
    index: int,
    mode: str,
    env_args: EnvironmentConfig,
    num_episode_test: int,
    recurrent: bool,
    log: bool,
    log_fn: Callable,
    log_frequency: int,
    total_timesteps: int,
    avg_reward_mode: bool = False,
    expert_policy: Optional[Callable] = None,
    eval_expert_policy: Optional[Callable] = None,
    sweep: bool = False,
    eval_action_transform: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    pid_gain_policy: bool = False,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
) -> tuple[BaseAgentState, Optional[dict]]:
    """Evaluate and log every ``log_frequency`` collected steps, each
    evaluation from fresh keys (:func:`evaluation_keys`); returns the
    state (one more ``n_logs`` per evaluation) and the logged metrics, or
    the same structure filled with NaN (``-1``: the timestep) between
    evaluations. Without logging (``log`` false) the evaluation is not even
    traced (tracing alone rebuilds the evaluation env) and the metrics are
    ``None``."""
    if not log:
        return agent_state, None
    timestep = agent_state.collector_state.timestep

    log_flag = timestep - (agent_state.n_logs * log_frequency) >= log_frequency
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

    def run_and_log(agent_state: BaseAgentState, aux: Any, index: int):
        eval_key, extra_key = evaluation_keys(agent_state)
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
            norm_info=eval_norm_info(agent_state, mode),
            avg_reward_mode=avg_reward_mode,
            expert_policy=expert_policy,
            eval_expert_policy=eval_expert_policy,
            # The training-fraction column the collector appends (None
            # unless the agent conditions on it).
            train_frac=agent_state.collector_state.train_time_fraction,
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
            extras = extra_eval_metrics(agent_state, extra_key)
            metrics_to_log.update(extras)

        gated_log_callback(log_fn, flag, metrics_to_log, index)

        return metrics_to_log

    no_op_branch = _make_no_op(extra_eval_metrics)
    metrics_to_log = jax.lax.cond(
        flag, run_and_log, no_op_branch, agent_state, aux, index
    )

    agent_state = agent_state.replace(
        n_logs=jax.lax.select(log_flag, agent_state.n_logs + 1, agent_state.n_logs)
    )

    return agent_state, metrics_to_log
