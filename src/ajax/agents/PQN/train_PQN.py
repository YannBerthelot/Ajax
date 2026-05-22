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
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.perf_utils import train_jit
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig

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
            (_, value_aux), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
                agent_state.actor_state.params,
                agent_state.actor_state,
                obs_mb,
                action_mb,
                target_mb,
                td_loss_fn,
            )
            q_state = agent_state.actor_state.apply_gradients(grads=grads)
            return agent_state.replace(actor_state=q_state), value_aux

        return jax.lax.scan(mb_body, agent_state, minibatches)

    agent_state, value_aux = jax.lax.scan(epoch_body, agent_state, epoch_keys)
    # value_aux is (n_epochs, num_minibatches, ...) -> mean to a (1,) scalar.
    value_aux = jax.tree.map(lambda x: x.mean().reshape((1,)), value_aux)
    aux = AuxiliaryLogs(value=value_aux)
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)

    # 4. Evaluate + log.
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
        extra_eval_metrics=extra_eval_metrics,
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

    @train_jit
    def train(key, index: Optional[int] = None):
        agent_state = init_PQN(
            key=key,
            env_args=env_args,
            optimizer_args=critic_optimizer_args,
            network_args=network_args,
            n_actions=n_actions,
        )

        # One iteration consumes n_envs * n_steps environment steps.
        num_updates = total_timesteps // (env_args.n_envs * agent_config.n_steps) + 1

        training_iteration_scan_fn = partial(
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
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train
