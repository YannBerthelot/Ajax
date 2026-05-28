"""PQN (Gallici et al., 2024): Parallelised Q-Network.

A simplified deep Q-learning algorithm for vectorised environments:
  * No replay buffer -- learns from on-policy rollouts of ``n_steps``
    across ``n_envs`` parallel environments (PPO-style collection).
  * No target network -- TD targets use the current network; stability
    comes from LayerNorm in the Q-network (see :class:`PQNNetwork`).
  * Q(lambda) returns as the regression target (``max_a Q`` bootstrap).
  * epsilon-greedy exploration, multi-epoch minibatched TD updates.

PQN reuses the value-based machinery from DQN (``GreedyQPolicy``,
``predict_q``, the epsilon-greedy action pipeline, the gathered-Q TD
loss and its aux dataclasses) and the minibatch shuffler from PPO. Its
own code is the LayerNorm network, the Q(lambda) target, and this
on-policy training loop.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.agents.DQN.networks import get_initialized_q_network, predict_q
from ajax.agents.DQN.train_DQN import (
    AuxiliaryLogs,
    make_epsilon_greedy_pipeline,
    mse_td_loss,
    q_loss_fn,
)
from ajax.agents.PPO.utils import get_minibatches_from_batch
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.PQN.state import PQNConfig, PQNState
from ajax.agents.PQN.utils import compute_q_lambda_targets
from ajax.environments.interaction import collect_experience, init_collector_state
from ajax.environments.utils import check_env_is_gymnax, get_action_dim
from ajax.extensions.base import ExtensionStack
from ajax.log import compose_eval_metrics, evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.perf_utils import build_resumable_train
from ajax.state import (
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    zeros_like_abstract_pytree,
)

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_PQN(
    key: jax.Array,
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    n_actions: int,
    window_size: int = 10,
) -> PQNState:
    rng, init_key, collector_key = jax.random.split(key, num=3)

    q_state = get_initialized_q_network(
        key=init_key,
        env_config=env_args,
        optimizer_config=optimizer_args,
        network_config=network_args,
        n_actions=n_actions,
        q_network_cls=PQNNetwork,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    # No buffer: PQN is on-policy and learns directly from each rollout.
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        window_size=window_size,
    )
    # The single Q-network lives in actor_state (so the shared eval loop
    # works); critic_state mirrors it at init and is never updated.
    return PQNState(
        rng=rng,
        eval_rng=rng,
        actor_state=q_state,
        critic_state=q_state,
        collector_state=collector_state,
    )


# ---------------------------------------------------------------------------
# Training iteration (collect rollout + Q(lambda) targets + epoch updates)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "env_args",
        "mode",
        "recurrent",
        "agent_config",
        "total_timesteps",
        "lstm_hidden_size",
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "action_pipeline",
        "eval_action_transform",
        "td_loss_fn",
        "extra_eval_metrics",
        "extension_stack",
    ],
)
def training_iteration(
    agent_state: PQNState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    agent_config: PQNConfig,
    total_timesteps: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: Optional[int] = 1000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    td_loss_fn: Callable = mse_td_loss,
    extra_eval_metrics: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
) -> Tuple[PQNState, Any]:
    # 1. Collect an on-policy rollout of n_steps across the parallel envs.
    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        action_pipeline=action_pipeline,
    )
    agent_state, transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=agent_config.n_steps
    )

    # Gap A: expose the freshly collected ``(T, n_envs, ...)`` rollout
    # on ``agent_state.last_rollout`` for downstream measurement
    # extensions. Off by default — see :attr:`BaseAgentState.last_rollout`.
    if getattr(agent_config, "expose_recent_rollout", False):
        agent_state = agent_state.replace(last_rollout=transition)

    # 2. Q(lambda) targets. No target network -- bootstrap on the current
    #    network's max-Q at the next states.
    next_q = predict_q(
        agent_state.actor_state,
        agent_state.actor_state.params,
        transition.next_obs,
    )
    next_q_max = jnp.max(next_q, axis=-1, keepdims=True)
    targets = compute_q_lambda_targets(
        transition.reward * agent_config.reward_scale,
        next_q_max,
        transition.terminated,
        transition.truncated,
        agent_config.gamma,
        agent_config.q_lambda,
    )
    # Extension fold: allow on_target to reshape the Q(lambda) regression
    # target before stop_gradient (e.g. a residual-of-residual reweighting,
    # bias correction, or auxiliary penalty operand).
    if extension_stack is not None:
        _tgt_rng, _tgt_seed = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=_tgt_rng)
        _tgt_batch = {
            "observations": transition.obs,
            "actions": transition.action,
            "next_observations": transition.next_obs,
            "rewards": transition.reward,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "gamma": agent_config.gamma,
        }
        targets = extension_stack.fold_on_target(
            agent_state,
            _tgt_batch,
            targets,
            agent_state.collector_state.timestep,
            _tgt_seed,
            total_timesteps,
        )
    targets = jax.lax.stop_gradient(targets)

    batch = (transition.obs, transition.action, targets)

    # 3. Multi-epoch minibatched TD updates over the rollout.
    rng, epoch_rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    epoch_keys = jax.random.split(epoch_rng, agent_config.n_epochs)

    def epoch_body(agent_state, epoch_key):
        # Re-shuffle the rollout into fresh minibatches each epoch.
        minibatches = get_minibatches_from_batch(
            batch, epoch_key, agent_config.num_minibatches
        )

        def mb_body(agent_state, minibatch):
            obs_mb, action_mb, target_mb = minibatch

            def _q_loss(params, q_state, obs, act, tgt):
                loss, core_aux = q_loss_fn(params, q_state, obs, act, tgt, td_loss_fn)
                # Additive extension critic-loss term, summed over the
                # stack. Empty stack ⇒ 0.0 ⇒ identical to the core loss.
                if extension_stack is not None:
                    _cl_batch = {
                        "observations": obs,
                        "actions": act,
                        "targets": tgt,
                        "q_state": q_state,
                    }
                    loss = loss + extension_stack.fold_critic_loss(
                        agent_state,
                        _cl_batch,
                        agent_state.collector_state.timestep,
                        agent_state.rng,
                        total_timesteps,
                    )
                return loss, core_aux

            (_, value_aux), grads = jax.value_and_grad(_q_loss, has_aux=True)(
                agent_state.actor_state.params,
                agent_state.actor_state,
                obs_mb,
                action_mb,
                target_mb,
            )
            q_state = agent_state.actor_state.apply_gradients(grads=grads)
            return agent_state.replace(actor_state=q_state), value_aux

        return jax.lax.scan(mb_body, agent_state, minibatches)

    agent_state, value_aux = jax.lax.scan(epoch_body, agent_state, epoch_keys)
    # value_aux is (n_epochs, num_minibatches, ...) -> mean to a (1,) scalar.
    value_aux = jax.tree.map(lambda x: x.mean().reshape((1,)), value_aux)
    aux = AuxiliaryLogs(value=value_aux)
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)

    # Extension post_update hook (state-threading + φ-refresh-style state
    # mutation). Empty stack ⇒ identity.
    if extension_stack is not None:
        _pu_rng, _pu_seed = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=_pu_rng)
        agent_state = extension_stack.fold_post_update(
            agent_state,
            agent_state.collector_state.timestep,
            _pu_seed,
            total_timesteps,
        )

    # 4. Evaluate + log. Merge stack.eval_metrics into the user's
    # extra_eval_metrics callable (both run; both contribute to the log
    # dict).
    _merged_extra_eval = compose_eval_metrics(
        extra_eval_metrics, extension_stack, total_timesteps
    )
    agent_state, metrics_to_log = evaluate_and_log(
        agent_state,
        aux,
        index,
        mode,
        env_args,
        num_episode_test,
        recurrent,
        lstm_hidden_size,
        log,
        verbose,
        log_fn,
        log_frequency,
        total_timesteps,
        eval_action_transform=eval_action_transform,
        extra_eval_metrics=_merged_extra_eval,
    )
    return agent_state, metrics_to_log


# ---------------------------------------------------------------------------
# Training factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: PQNConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.5,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    td_loss_fn: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    extensions: Sequence = (),
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    recurrent = network_args.lstm_hidden_size is not None
    n_actions = get_action_dim(env_args.env, env_args.env_params)
    td_loss_fn = td_loss_fn if td_loss_fn is not None else mse_td_loss

    if action_pipeline is None:
        action_pipeline = make_epsilon_greedy_pipeline(
            env_args=env_args,
            n_actions=n_actions,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_frac=epsilon_decay_frac,
            total_timesteps=total_timesteps,
        )

    # One iteration consumes n_envs * n_steps environment steps.
    num_updates = total_timesteps // (env_args.n_envs * agent_config.n_steps) + 1

    extension_stack = ExtensionStack(extensions) if extensions else None

    def init_fn(key, index):
        agent_state = init_PQN(
            key=key,
            env_args=env_args,
            optimizer_args=critic_optimizer_args,
            network_args=network_args,
            n_actions=n_actions,
        )
        # Initialise per-extension state tuple (one entry per Extension;
        # stateless extensions hold ``()``). Skipped on resume — the
        # resumed state already carries ``ext_state``.
        if extension_stack is not None:
            _ext_key, _pre_key = jax.random.split(key)
            agent_state = extension_stack.fold_init_states(agent_state, _ext_key)
            # One-shot pretrain phase (fresh-init only). Empty stack /
            # extensions that don't override pretrain ⇒ identity.
            agent_state = extension_stack.fold_pretrain(
                agent_state, jnp.asarray(0), _pre_key, total_timesteps
            )
        # Gap A: pre-allocate the ``last_rollout`` placeholder so the
        # scan-carry pytree structure is stable from iteration zero.
        if getattr(agent_config, "expose_recent_rollout", False):
            _trace_scan = partial(
                collect_experience,
                recurrent=recurrent,
                mode=mode,
                env_args=env_args,
                action_pipeline=action_pipeline,
            )
            _, _trans_abs = jax.eval_shape(
                lambda st: jax.lax.scan(
                    _trace_scan, st, xs=None, length=agent_config.n_steps
                ),
                agent_state,
            )
            agent_state = agent_state.replace(
                last_rollout=zeros_like_abstract_pytree(_trans_abs)
            )
        return agent_state

    def make_scan_fn(_agent_state, _resume_from_state, _key, index):
        return partial(
            training_iteration,
            recurrent=recurrent,
            agent_config=agent_config,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            lstm_hidden_size=network_args.lstm_hidden_size,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            action_pipeline=action_pipeline,
            eval_action_transform=eval_action_transform,
            td_loss_fn=td_loss_fn,
            extra_eval_metrics=extra_eval_metrics,
            extension_stack=extension_stack,
        )

    return build_resumable_train(
        init_fn=init_fn,
        make_scan_fn=make_scan_fn,
        num_updates=num_updates,
    )
