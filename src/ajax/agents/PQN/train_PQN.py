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

from ajax.agents.DQN.networks import predict_q
from ajax.agents.DQN.train_DQN import (
    AuxiliaryLogs,
    init_DQN,
    make_epsilon_greedy_pipeline,
    mse_td_loss,
    q_gradient_step,
)
from ajax.agents.loop import TrainLoop
from ajax.agents.PPO.utils import get_minibatches_from_batch
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.PQN.state import PQNConfig, PQNState
from ajax.agents.PQN.utils import compute_q_lambda_targets
from ajax.environments.utils import get_action_dim
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import (
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)

# ---------------------------------------------------------------------------
# The update: Q(lambda) targets, then minibatched TD epochs
# ---------------------------------------------------------------------------


def update_agent(
    agent_state: PQNState,
    rollout: Transition,
    agent_config: PQNConfig,
    td_loss_fn: Callable,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[PQNState, AuxiliaryLogs]:
    """One PQN update on an ``(n_steps, n_envs)`` rollout: Q(lambda)
    targets bootstrapped on the current network (no target network), then
    ``n_epochs`` epochs of TD steps, the rollout reshuffled each epoch."""
    next_q = predict_q(
        agent_state.actor_state,
        agent_state.actor_state.params,
        rollout.next_obs,
    )
    next_q_max = jnp.max(next_q, axis=-1, keepdims=True)
    targets = compute_q_lambda_targets(
        rollout.reward * agent_config.reward_scale,
        next_q_max,
        rollout.terminated,
        rollout.truncated,
        agent_config.gamma,
        agent_config.q_lambda,
    )
    # The extensions reshape the target before it is held constant.
    if extension_stack:
        _tgt_rng, _tgt_seed = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=_tgt_rng)
        _tgt_batch = {
            "observations": rollout.obs,
            "actions": rollout.action,
            "next_observations": rollout.next_obs,
            "rewards": rollout.reward,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
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
    batch = (rollout.obs, rollout.action, targets)

    rng, epoch_rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    epoch_keys = jax.random.split(epoch_rng, agent_config.n_epochs)

    def epoch_body(agent_state: PQNState, epoch_key: jax.Array) -> Tuple[Any, Any]:
        minibatches = get_minibatches_from_batch(
            batch, epoch_key, agent_config.num_minibatches
        )

        def mb_body(agent_state: PQNState, minibatch: tuple) -> Tuple[Any, Any]:
            q_state, value_aux = q_gradient_step(
                agent_state, *minibatch, td_loss_fn, extension_stack, total_timesteps
            )
            return agent_state.replace(actor_state=q_state), value_aux

        return jax.lax.scan(mb_body, agent_state, minibatches)

    agent_state, value_aux = jax.lax.scan(epoch_body, agent_state, epoch_keys)
    # (n_epochs, num_minibatches, ...) -> one (1,) value per metric.
    value_aux = jax.tree.map(lambda x: x.mean().reshape((1,)), value_aux)
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, AuxiliaryLogs(value=value_aux)


# ---------------------------------------------------------------------------
# Training factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    # base.py passes every agent both optimiser configs; one Q-network here.
    actor_optimizer_args: OptimizerConfig,  # noqa: ARG001
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: PQNConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.5,
    td_loss_fn: Optional[Callable] = None,
    extensions: Sequence = (),
):
    """PQN's train function: an ``n_steps`` epsilon-greedy rollout per env,
    then one update, per iteration."""
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
    )
    n_actions = get_action_dim(env_args.env, env_args.env_params)
    td_loss_fn = td_loss_fn if td_loss_fn is not None else mse_td_loss

    def init(key: jax.Array, _pretrain_key: jax.Array) -> PQNState:
        return init_DQN(
            key,
            env_args,
            critic_optimizer_args,
            network_args,
            n_actions,
            q_network_cls=PQNNetwork,
            state_cls=PQNState,
        )

    def update(
        agent_state: PQNState, rollout: Transition, _start: PQNState
    ) -> Tuple[PQNState, AuxiliaryLogs]:
        return update_agent(
            agent_state, rollout, agent_config, td_loss_fn, loop.stack, total_timesteps
        )

    action_pipeline = make_epsilon_greedy_pipeline(
        n_actions=n_actions,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_frac=epsilon_decay_frac,
        total_timesteps=total_timesteps,
    )
    return loop.on_policy(
        init,
        update,
        agent_config.n_steps,
        expose_rollout=agent_config.expose_recent_rollout,
        collect_kwargs={"action_pipeline": action_pipeline},
    )
