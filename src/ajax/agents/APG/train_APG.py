"""Training loop for APG: analytic policy gradient through the simulator.

One update = sample a batch of systems from the system class, run the
deterministic controller in closed loop for ``horizon`` steps with
gradients flowing through policy, memory carry and environment
(:func:`ajax.environments.differentiable.closed_loop_rollout`), and take
one optimizer step on ``-mean_i sum_k r_k``. On a model-reference
tracking task (:mod:`ajax.environments.model_reference`) that objective
is exactly the closed-loop matching cost of Busetto et al. 2024, eq. (9),
and ``sqrt(loss / horizon)`` is their M-RMSE.

Evaluation re-samples ``num_episode_test`` fresh systems (and, through
the env's reset, fresh references and initial conditions) and reports the
deterministic closed-loop return and M-RMSE, i.e. the paper's validation
metric, at the logging frequency.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.APG.networks import Controller, PIDHeadConfig
from ajax.agents.APG.state import APGConfig, APGState
from ajax.environments.differentiable import closed_loop_rollout
from ajax.environments.interaction import init_collector_state
from ajax.environments.system_class import SystemClass, broadcast_env_params
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.log import compose_eval_metrics
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.networks.memory import zeros_carry_like
from ajax.networks.utils import get_adam_tx
from ajax.perf_utils import build_resumable_train
from ajax.schedule import warmup_cosine_schedule
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)


@struct.dataclass
class APGAuxiliaries:
    loss: jax.Array  # objective actually minimised (incl. extension terms)
    matching_loss: jax.Array  # -mean_i sum_k r_k, the closed-loop cost
    m_rmse: jax.Array  # sqrt(matching_loss / horizon)
    timestep: jax.Array


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def build_controller(
    env_args: EnvironmentConfig,
    network_args: NetworkConfig,
    pid: Optional[PIDHeadConfig],
    squash: bool,
) -> Controller:
    return Controller(
        input_architecture=tuple(network_args.actor_architecture),
        action_dim=get_action_dim(env_args.env, env_args.env_params),
        memory=network_args.memory,
        pid=pid,
        squash=squash,
    )


def init_APG(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    pid: Optional[PIDHeadConfig],
    squash: bool,
    window_size: int = 10,
) -> APGState:
    rng, params_key, carry_key, collector_key = jax.random.split(key, 4)
    controller = build_controller(env_args, network_args, pid, squash)
    obs_shape, _ = get_state_action_shapes(env_args.env)
    if controller.stateful:
        carry = controller.initialize_carry(carry_key, env_args.n_envs)
        params = controller.init(
            params_key,
            jnp.zeros((1, env_args.n_envs, *obs_shape)),
            hidden_state=carry,
            done=jnp.zeros((1, env_args.n_envs), bool),
        )
    else:
        carry = None
        params = controller.init(params_key, jnp.zeros((1, *obs_shape)))
    actor_state = LoadedTrainState.create(
        params=params,
        tx=get_adam_tx(**to_state_dict(actor_optimizer_args)),
        apply_fn=controller.apply,
        hidden_state=carry,
        recurrent=controller.stateful,
    )
    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode="gymnax", window_size=window_size
    )
    return APGState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=None,
        collector_state=collector_state,
    )


def reset_optimizer(agent_state: APGState, optimizer_args: OptimizerConfig) -> APGState:
    """Fresh optimizer (config + zeroed moments, step 0), same parameters.

    Used between curriculum stages so each stage runs its own learning-
    rate schedule from the top instead of inheriting the previous stage's
    step counter and decayed rate.
    """
    tx = get_adam_tx(**to_state_dict(optimizer_args))
    actor_state = agent_state.actor_state
    actor_state = actor_state.replace(
        tx=tx, opt_state=tx.init(actor_state.params), step=0
    )
    return agent_state.replace(actor_state=actor_state)


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


def make_policy_step(
    actor_state: LoadedTrainState, params: Any, stateful: bool
) -> Callable:
    """``(carry, obs, resets) -> (action, carry)`` for closed_loop_rollout.

    ``stateful`` is the static (trace-time) counterpart of
    ``actor_state.recurrent``, which is a pytree leaf inside jit.
    """
    if stateful:

        def policy_step(carry, obs, resets):
            pi, carry = actor_state.apply(
                params, obs[None], hidden_state=carry, done=resets[None]
            )
            return pi.mean()[0], carry

    else:

        def policy_step(carry, obs, resets):
            del resets
            return actor_state.apply(params, obs).mean(), carry

    return policy_step


def sample_systems(
    system_class: Optional[SystemClass], rng: jax.Array, n: int, nominal: Any
) -> Any:
    if system_class is None:
        return broadcast_env_params(nominal, n)
    return system_class.sample(rng, n)


def rollout_returns(
    agent_state: APGState,
    params: Any,
    rng: jax.Array,
    env_args: EnvironmentConfig,
    system_class: Optional[SystemClass],
    horizon: int,
    n: int,
    stateful: bool,
):
    """Closed-loop returns ``(n,)`` of the controller on ``n`` fresh systems."""
    sys_key, roll_key = jax.random.split(rng)
    env_params = sample_systems(system_class, sys_key, n, env_args.env_params)
    carry = (
        zeros_carry_like(agent_state.actor_state.hidden_state, n) if stateful else None
    )
    rollout, _ = closed_loop_rollout(
        make_policy_step(agent_state.actor_state, params, stateful),
        carry,
        roll_key,
        env_args.env,
        env_params,
        horizon,
    )
    return rollout.reward.sum(0), rollout


# ---------------------------------------------------------------------------
# Evaluation + logging
# ---------------------------------------------------------------------------


def evaluate_apg(
    agent_state: APGState,
    rng: jax.Array,
    env_args: EnvironmentConfig,
    system_class: Optional[SystemClass],
    horizon: int,
    num_episode_test: int,
    stateful: bool,
) -> dict:
    returns, _ = rollout_returns(
        agent_state,
        agent_state.actor_state.params,
        rng,
        env_args,
        system_class,
        horizon,
        num_episode_test,
        stateful,
    )
    return {
        "Eval/episodic mean reward": returns.mean(),
        "Eval/m_rmse": jnp.sqrt(jnp.maximum(-returns.mean(), 0.0) / horizon),
    }


def _maybe_log(
    agent_state: APGState,
    aux: APGAuxiliaries,
    index: Any,
    *,
    evaluate_fn: Callable,
    extra_eval_metrics: Optional[Callable],
    log: bool,
    log_fn: Callable,
    log_frequency: Optional[int],
    total_timesteps: int,
) -> Tuple[APGState, dict]:
    """Evaluate + log every ``log_frequency`` env steps (same gating as
    :func:`ajax.log.evaluate_and_log`)."""

    def run(agent_state, aux, index):
        eval_key, extra_key = jax.random.split(agent_state.eval_rng)
        metrics = {
            "timestep": aux.timestep,
            "Train/loss": aux.loss,
            "Train/matching_loss": aux.matching_loss,
            "Train/m_rmse": aux.m_rmse,
        }
        metrics.update(evaluate_fn(agent_state, eval_key))
        if extra_eval_metrics is not None:
            metrics.update(extra_eval_metrics(agent_state, extra_key))
        if log:
            jax.debug.callback(log_fn, metrics, index)
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

    if not log or not log_frequency:
        return agent_state, skip(agent_state, aux, index)
    timestep = aux.timestep
    log_flag = timestep - agent_state.n_logs * log_frequency >= log_frequency
    flag = jnp.logical_and(log_flag, timestep <= total_timesteps)
    metrics = jax.lax.cond(flag, run, skip, agent_state, aux, index)
    agent_state = agent_state.replace(
        n_logs=jax.lax.select(log_flag, agent_state.n_logs + 1, agent_state.n_logs)
    )
    return agent_state, metrics


# ---------------------------------------------------------------------------
# One update
# ---------------------------------------------------------------------------


def training_iteration(
    agent_state: APGState,
    _: Any,
    *,
    env_args: EnvironmentConfig,
    agent_config: APGConfig,
    system_class: Optional[SystemClass],
    extension_stack: Optional[ExtensionStack],
    total_timesteps: int,
    index: Any,
    log_kwargs: dict,
    stateful: bool,
) -> Tuple[APGState, APGAuxiliaries]:
    rng, roll_key, ext_key, post_key = jax.random.split(agent_state.rng, 4)
    agent_state = agent_state.replace(rng=rng)
    horizon = agent_config.horizon
    timestep = agent_state.collector_state.timestep

    def loss_fn(params):
        returns, rollout = rollout_returns(
            agent_state,
            params,
            roll_key,
            env_args,
            system_class,
            horizon,
            env_args.n_envs,
            stateful,
        )
        matching_loss = -returns.mean()
        loss = matching_loss
        if extension_stack is not None:
            loss = loss + extension_stack.fold_actor_loss(
                agent_state,
                {"rollout": rollout, "actor_params": params, "returns": returns},
                timestep,
                ext_key,
                total_timesteps,
            )
        return loss, matching_loss

    (loss, matching_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        agent_state.actor_state.params
    )
    actor_state = agent_state.actor_state.apply_gradients(grads=grads)
    new_timestep = timestep + env_args.n_envs * horizon
    agent_state = agent_state.replace(
        actor_state=actor_state,
        collector_state=agent_state.collector_state.replace(timestep=new_timestep),
        n_updates=agent_state.n_updates + 1,
    )
    if extension_stack is not None:
        agent_state = extension_stack.fold_post_update(
            agent_state, new_timestep, post_key, total_timesteps
        )
    aux = APGAuxiliaries(
        loss=loss,
        matching_loss=matching_loss,
        m_rmse=jnp.sqrt(jnp.maximum(matching_loss, 0.0) / horizon),
        timestep=new_timestep,
    )
    agent_state, _ = _maybe_log(agent_state, aux, index, **log_kwargs)
    return agent_state, aux


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: APGConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    system_class: Optional[SystemClass] = None,
    pid: Optional[PIDHeadConfig] = None,
    squash: bool = True,
    lr_schedule: Optional[str] = None,
    warmup_steps: int = 0,
    lr_end_fraction: float = 0.1,
    reset_optimizer_on_resume: bool = True,
    extra_eval_metrics: Optional[Callable] = None,
    extensions: Sequence = (),
    **_unused: Any,
):
    del critic_optimizer_args  # no critic
    per_update = env_args.n_envs * agent_config.horizon
    num_updates = max(total_timesteps // per_update, 1)
    if lr_schedule == "warmup_cosine":
        peak = actor_optimizer_args.learning_rate
        if not isinstance(peak, (int, float)):
            raise ValueError("lr_schedule='warmup_cosine' needs a float learning_rate")
        actor_optimizer_args = actor_optimizer_args.replace(
            learning_rate=warmup_cosine_schedule(
                float(peak),
                warmup_steps=min(warmup_steps, num_updates),
                total_steps=num_updates,
                end_value_fraction=lr_end_fraction,
            )
        )
    elif lr_schedule is not None:
        raise ValueError(f"Unknown lr_schedule {lr_schedule!r}; use 'warmup_cosine'")

    extension_stack = ExtensionStack(extensions) if extensions else None
    stateful = build_controller(env_args, network_args, pid, squash).stateful
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)
    if log:
        start_async_logging()
    log_kwargs = {
        "evaluate_fn": partial(
            evaluate_apg,
            env_args=env_args,
            system_class=system_class,
            horizon=agent_config.horizon,
            num_episode_test=num_episode_test,
            stateful=stateful,
        ),
        "extra_eval_metrics": compose_eval_metrics(
            extra_eval_metrics, extension_stack, total_timesteps
        ),
        "log": log,
        "log_fn": log_fn,
        "log_frequency": (
            logging_config.log_frequency if logging_config is not None else None
        ),
        "total_timesteps": total_timesteps,
    }

    def init_fn(key, index):
        agent_state = init_APG(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            network_args=network_args,
            pid=pid,
            squash=squash,
        )
        return agent_state.replace(index=index)

    def init_transform(agent_state, key):
        if extension_stack is None:
            return agent_state
        ext_key, pre_key = jax.random.split(key)
        agent_state = extension_stack.fold_init_states(agent_state, ext_key)
        return extension_stack.fold_pretrain(
            agent_state, jnp.asarray(0), pre_key, total_timesteps
        )

    def resume_transform(agent_state, key):
        del key
        if not reset_optimizer_on_resume:
            return agent_state
        return reset_optimizer(agent_state, actor_optimizer_args)

    def make_scan_fn(_agent_state, _resume, _key, index):
        return partial(
            training_iteration,
            env_args=env_args,
            agent_config=agent_config,
            system_class=system_class,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
            index=index,
            log_kwargs=log_kwargs,
            stateful=stateful,
        )

    return build_resumable_train(
        init_fn=init_fn,
        make_scan_fn=make_scan_fn,
        num_updates=num_updates,
        init_transform=init_transform,
        resume_transform=resume_transform,
    )
