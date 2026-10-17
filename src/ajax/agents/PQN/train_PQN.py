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

from ajax.agents.DQN.networks import predict_q, q_network_value
from ajax.agents.DQN.train_DQN import (
    AuxiliaryLogs,
    init_DQN,
    make_epsilon_greedy_pipeline,
    mse_td_loss,
    q_gradient_step,
)
from ajax.agents.loop import TrainLoop
from ajax.agents.PPO.sequences import split_carries
from ajax.agents.PPO.utils import get_minibatches_from_batch, split_fragments
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.PQN.state import PQNConfig, PQNState
from ajax.agents.PQN.utils import compute_q_lambda_targets
from ajax.agents.recurrent import unsupported_recurrent_options
from ajax.environments.interaction import get_pi_sequence
from ajax.environments.utils import get_action_dim
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import (
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


def update_agent(
    agent_state: PQNState,
    rollout: Transition,
    agent_config: PQNConfig,
    td_loss_fn: Callable,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    start: Optional[PQNState] = None,
) -> Tuple[PQNState, AuxiliaryLogs]:
    """One PQN update on an ``(n_steps, n_envs)`` rollout: Q(lambda)
    targets bootstrapped on the current network (no target network), then
    ``n_epochs`` epochs of TD steps, the rollout reshuffled each epoch.

    With memory (``start``: the state the rollout was collected from), the
    Q-network reads each env's whole rollout from its start carry: the
    targets from one pass, the TD steps on minibatches of whole rollouts
    (split over the envs, one minibatch when they do not split evenly)."""
    if start is None:
        next_q = predict_q(
            agent_state.actor_state,
            agent_state.actor_state.params,
            rollout.next_obs,
        )
        next_q_max = jnp.max(next_q, axis=-1, keepdims=True)
    else:
        resets, next_q_max = _recurrent_next_q(agent_state, rollout, start)
    targets = compute_q_lambda_targets(
        rollout.reward * agent_config.reward_scale,
        next_q_max,
        rollout.terminated,
        rollout.truncated,
        agent_config.gamma,
        agent_config.q_lambda,
    )
    # The extensions reshape the target before it is held constant, then add
    # their loss terms, each on a key of its own (drawn only with extensions):
    # one per epoch and minibatch.
    loss_keys = None
    if extension_stack:
        rng, ext_key = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        target_key, loss_key = jax.random.split(ext_key)
        loss_keys = jax.random.split(
            loss_key, (agent_config.n_epochs, agent_config.num_minibatches)
        )
        _tgt_batch = {
            "observations": rollout.obs,
            "actions": rollout.action,
            "next_observations": rollout.next_obs,
            "rewards": rollout.reward,
            # Where the target cut the bootstrap, as the replay agents pass.
            "dones": rollout.terminated,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "gamma": agent_config.gamma,
            "reward_scale": agent_config.reward_scale,
        }
        targets = extension_stack.fold_on_target(
            agent_state,
            _tgt_batch,
            targets,
            agent_state.collector_state.timestep,
            target_key,
            total_timesteps,
        )
    targets = jax.lax.stop_gradient(targets)

    rng, epoch_rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    epoch_keys = jax.random.split(epoch_rng, agent_config.n_epochs)
    n_steps, n_envs = rollout.obs.shape[:2]
    k = agent_config.num_minibatches
    if start is None:
        batch: Any = (rollout.obs, rollout.action, targets)
    else:
        batch = {
            "obs": rollout.obs,
            "action": rollout.action,
            "targets": targets,
            "resets": resets,
        }
        k = k if n_envs % k == 0 else 1
        carry = jax.lax.stop_gradient(start.actor_state.hidden_state)

    def minibatches(key: jax.Array) -> Any:
        if start is None:
            return get_minibatches_from_batch(batch, key, k)
        perm = jax.random.permutation(key, n_envs)
        return split_fragments(batch, perm, k, n_steps) | {
            "carry": jax.tree.map(lambda x: split_carries(x[None], perm, k), carry)
        }

    def epoch_body(agent_state: PQNState, keys: tuple) -> Tuple[Any, Any]:
        epoch_key, epoch_loss_keys = keys
        parts = minibatches(epoch_key)

        def mb_body(agent_state: PQNState, xs: tuple) -> Tuple[Any, Any]:
            mb, loss_key = xs
            q_values = None
            if start is None:
                observations, actions, targets = mb
            else:
                observations, actions, targets = mb["obs"], mb["action"], mb["targets"]
                q_state = agent_state.actor_state

                def q_values(params: Any) -> jax.Array:
                    pi, _ = get_pi_sequence(
                        q_state, params, observations, mb["resets"], mb["carry"]
                    )
                    return pi.q_values

            q_state, value_aux = q_gradient_step(
                agent_state,
                observations,
                actions,
                targets,
                td_loss_fn,
                extension_stack,
                loss_key,
                total_timesteps,
                q_values,
            )
            return agent_state.replace(actor_state=q_state), value_aux

        return jax.lax.scan(mb_body, agent_state, (parts, epoch_loss_keys))

    agent_state, value_aux = jax.lax.scan(
        epoch_body, agent_state, (epoch_keys, loss_keys)
    )
    # (n_epochs, num_minibatches, ...) -> one (1,) value per metric.
    value_aux = jax.tree.map(lambda x: x.mean().reshape((1,)), value_aux)
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, AuxiliaryLogs(value=value_aux)


def _recurrent_next_q(
    agent_state: PQNState, rollout: Transition, start: PQNState
) -> tuple[jax.Array, jax.Array]:
    """A recurrent Q-network's reset flags and ``max_a Q(s')`` over the
    rollout, step by step from the start carry: after each step it reads
    ``s'`` (the episode's final observation where it ended) from the carry
    that consumed ``s``, as PPO's recurrent values do."""
    q_state = agent_state.actor_state
    initial_done = jnp.logical_or(
        start.collector_state.last_terminated, start.collector_state.last_truncated
    ).astype(bool)
    dones = jnp.logical_or(rollout.terminated, rollout.truncated).squeeze(-1)
    resets = jnp.concatenate([initial_done[None], dones[:-1].astype(bool)], axis=0)
    going_on = jnp.zeros((1, resets.shape[1]), bool)

    def step(carry: Any, x: tuple) -> tuple[Any, jax.Array]:
        obs, next_obs, reset = x
        params = q_state.params
        _, carry = get_pi_sequence(q_state, params, obs[None], reset[None], carry)
        pi, _ = get_pi_sequence(q_state, params, next_obs[None], going_on, carry)
        return carry, pi.q_values[0].max(-1, keepdims=True)

    start_carry = jax.lax.stop_gradient(start.actor_state.hidden_state)
    _, next_q_max = jax.lax.scan(
        step, start_carry, (rollout.obs, rollout.next_obs, resets)
    )
    return resets, jax.lax.stop_gradient(next_q_max)


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
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options("PQN", extensions=(tuple(extensions) or None))
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

    def init(key: jax.Array) -> PQNState:
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
        agent_state: PQNState, rollout: Transition, start: PQNState
    ) -> Tuple[PQNState, AuxiliaryLogs]:
        return update_agent(
            agent_state,
            rollout,
            agent_config,
            td_loss_fn,
            loop.stack,
            total_timesteps,
            start if recurrent else None,
        )

    action_pipeline = make_epsilon_greedy_pipeline(
        n_actions=n_actions,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_frac=epsilon_decay_frac,
        total_timesteps=total_timesteps,
        recurrent=recurrent,
    )
    return loop.on_policy(
        init,
        update,
        agent_config.n_steps,
        recurrent=recurrent,
        expose_rollout=agent_config.expose_recent_rollout,
        collect_kwargs={"action_pipeline": action_pipeline},
        eval_kwargs={
            "value_estimate": ValueEstimate(
                q_network_value, agent_config.gamma, agent_config.reward_scale
            )
        },
    )
