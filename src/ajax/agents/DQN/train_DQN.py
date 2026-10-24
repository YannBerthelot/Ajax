"""DQN (Mnih et al., 2015): Deep Q-Network for discrete action spaces.

Value-based, off-policy. A single Q-network ``Q(s) -> R^{n_actions}`` is
trained to minimise the temporal-difference error against a slowly
refreshed target network:

    y    = r + gamma * (1 - done) * max_a' Q_target(s', a')
    loss = Huber_1(Q(s, a), y)

Differences from the SAC/TD3 template in this codebase:
  * No actor. The greedy policy is ``argmax_a Q(s, a)``; the Q-network is
    stored in ``DQNState.actor_state`` so the shared evaluation loop
    (hard-wired to ``agent_state.actor_state``) works unchanged.
  * Exploration is epsilon-greedy, applied at collection time via a
    custom ``action_pipeline`` -- the vanilla collector path samples a
    continuous action in [-1, 1] and cannot be reused for discrete envs.
  * The target network is refreshed every ``target_update_interval``
    gradient steps via ``soft_update(tau)``. The classic hard update is
    ``tau=1.0``; a Polyak update is ``tau<1`` with interval ``1`` --
    one mechanism, no boolean flag.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.critics import ValueAuxiliaries, huber_1_td_loss, q_gradient_step
from ajax.agents.DQN.state import DQNConfig, DQNState
from ajax.agents.loop import TrainLoop, extension_key, when
from ajax.buffers.dataset import TransitionDataset
from ajax.buffers.utils import get_batch_from_buffer
from ajax.deprecation import moved_names
from ajax.environments.interaction import (
    epsilon_greedy,
    explore_pipeline,
    init_collector_state,
    reward_statistics,
)
from ajax.environments.utils import get_action_dim
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.q_networks import (
    get_initialized_q_network,
    predict_q,
    q_network_value,
)
from ajax.perf_utils import final_aux_scan
from ajax.schedule import linear
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import BufferType


@struct.dataclass
class PolicyAuxiliaries:
    epsilon: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    # Nested one level deep so ajax.log.flatten_dict names the metrics
    # ``value/<field>``, as the other agents' auxiliaries are named.
    value: ValueAuxiliaries
    policy: PolicyAuxiliaries
    extensions: dict  # the extensions' training metrics


def init_DQN(
    key: jax.Array,
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    n_actions: int,
    buffer: Optional[BufferType] = None,
    q_network_cls: Optional[type] = None,
) -> DQNState:
    """The Q-network and the collector (writing to ``buffer``, if any)."""
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)

    q_state = get_initialized_q_network(
        key=init_key,
        env_config=env_args,
        optimizer_config=optimizer_args,
        network_config=network_args,
        n_actions=n_actions,
        q_network_cls=q_network_cls,
    )
    mode = env_args.mode
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
    )
    # The single Q-network lives in actor_state (so the shared eval loop
    # works); critic_state mirrors it at init and is never updated.
    return DQNState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=q_state,
        critic_state=q_state,
        collector_state=collector_state,
    )


def compute_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
) -> jax.Array:
    """Vanilla DQN target: y = r + gamma * (1 - d) * max_a' Q_target(s', a')."""
    rewards = rewards * reward_scale
    q_next = predict_q(q_state, q_state.target_params, next_observations)
    max_q_next = jnp.max(q_next, axis=-1, keepdims=True)
    target = rewards + gamma * (1.0 - dones) * max_q_next
    return jax.lax.stop_gradient(target)


def compute_double_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
) -> jax.Array:
    """Double DQN target (van Hasselt et al., 2016).

    Decouples action *selection* (online network) from action
    *evaluation* (target network), removing the maximisation bias of the
    vanilla ``max_a' Q_target`` target:

        a*  = argmax_a' Q_online(s', a')
        y   = r + gamma * (1 - d) * Q_target(s', a*)

    Drop-in for ``compute_dqn_td_target``; select via
    ``DQN(..., td_target_fn=compute_double_dqn_td_target)``.
    """
    rewards = rewards * reward_scale
    q_next_online = predict_q(q_state, q_state.params, next_observations)
    next_actions = jnp.argmax(q_next_online, axis=-1, keepdims=True)
    q_next_target = predict_q(q_state, q_state.target_params, next_observations)
    q_next = jnp.take_along_axis(q_next_target, next_actions, axis=-1)
    target = rewards + gamma * (1.0 - dones) * q_next
    return jax.lax.stop_gradient(target)


@partial(
    jax.jit,
    static_argnames=[
        "buffer",
        "gamma",
        "tau",
        "target_update_interval",
        "reward_scale",
        "td_target_fn",
        "td_loss_fn",
        "extension_stack",
        "total_timesteps",
        "epsilon_schedule",
    ],
)
def update_agent(
    agent_state: DQNState,
    _: Any,
    buffer: BufferType,
    gamma: float,
    tau: float,
    target_update_interval: int,
    reward_scale: float,
    td_target_fn: Callable,
    td_loss_fn: Callable,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    epsilon_schedule: Callable[[Any], jax.Array],
) -> Tuple[DQNState, AuxiliaryLogs]:
    """One gradient step on a replay batch, and the target network's
    update every ``target_update_interval`` steps; also logs
    ``epsilon_schedule`` at the collector's timestep."""
    agent_state, sample_key = agent_state.next_keys()
    collector = agent_state.collector_state
    batch = get_batch_from_buffer(
        buffer,
        collector.buffer_state,
        sample_key,
        reward_statistics(collector.env_state),
    )
    observations, actions, rewards = batch.obs, batch.action, batch.reward
    terminated, next_observations = batch.terminated, batch.next_obs
    # A truncated row bootstraps on its final observation (bootstrap_obs).
    dones = terminated.astype(jnp.float32)

    target_q = td_target_fn(
        q_state=agent_state.actor_state,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        reward_scale=reward_scale,
    )

    # The extensions reshape the target, then add their loss terms, each on
    # a key of its own.
    target_key, loss_key = jax.random.split(extension_key(sample_key))
    target_batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "dones": dones,
        "gamma": gamma,
        "reward_scale": reward_scale,
    }
    target_q, target_metrics = extension_stack.fold_on_target(
        agent_state,
        target_batch,
        target_q,
        agent_state.collector_state.timestep,
        target_key,
        total_timesteps,
    )

    q_state, value_aux, loss_metrics = q_gradient_step(
        agent_state,
        observations,
        actions,
        target_q,
        td_loss_fn,
        extension_stack,
        loss_key,
        total_timesteps,
    )

    do_target_update = (agent_state.n_updates % target_update_interval) == 0
    q_state = when(do_target_update, lambda s: s.soft_update(tau), q_state)

    agent_state = agent_state.replace(
        actor_state=q_state,
        n_updates=agent_state.n_updates + 1,
    )
    epsilon = epsilon_schedule(agent_state.collector_state.timestep)
    return agent_state, AuxiliaryLogs(
        value=value_aux,
        policy=PolicyAuxiliaries(epsilon=epsilon),
        extensions=merge_metrics(target_metrics, loss_metrics),
    )


def make_train(
    env_args: EnvironmentConfig,
    # base.py passes every agent both optimiser configs; one Q-network here.
    actor_optimizer_args: OptimizerConfig,  # noqa: ARG001
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: DQNConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.5,
    td_target_fn: Optional[Callable] = None,
    td_loss_fn: Optional[Callable] = None,
    q_network_cls: Optional[type] = None,
    offline_data: Optional[TransitionDataset] = None,
    extensions: Sequence = (),
):
    """DQN's train function: one step per env, then (from ``learning_starts``)
    ``n_gradient_steps`` updates, per iteration."""
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        offline_data=offline_data,
    )
    n_actions = get_action_dim(env_args.env, env_args.env_params)
    td_target_fn = td_target_fn if td_target_fn is not None else compute_dqn_td_target
    td_loss_fn = td_loss_fn if td_loss_fn is not None else huber_1_td_loss
    # Linear over the first epsilon_decay_frac of the horizon, then flat.
    epsilon_schedule = linear(
        epsilon_start, epsilon_end, epsilon_decay_frac * total_timesteps
    )

    def init(key: jax.Array) -> DQNState:
        return init_DQN(
            key,
            env_args,
            critic_optimizer_args,
            network_args,
            n_actions,
            buffer=buffer,
            q_network_cls=q_network_cls,
        )

    def update(agent_state: DQNState, _transition: Any) -> Any:
        # The step just collected is in the buffer: DQN samples it from there.
        one_update = partial(
            update_agent,
            buffer=buffer,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
            target_update_interval=agent_config.target_update_interval,
            reward_scale=agent_config.reward_scale,
            td_target_fn=td_target_fn,
            td_loss_fn=td_loss_fn,
            extension_stack=loop.extensions,
            total_timesteps=total_timesteps,
            epsilon_schedule=epsilon_schedule,
        )
        # Carry-only scan: only the final-step aux is materialised.
        return final_aux_scan(
            one_update, agent_state, length=agent_config.n_gradient_steps
        )

    action_pipeline = explore_pipeline(
        epsilon_greedy(epsilon_schedule, n_actions), False, n_actions
    )
    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        collect_kwargs={"buffer": buffer, "action_pipeline": action_pipeline},
        eval_kwargs={
            "value_estimate": ValueEstimate(
                q_network_value, agent_config.gamma, agent_config.reward_scale
            )
        },
    )


__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        ("make_huber_td_loss", "mse_td_loss", "q_loss_fn"), "ajax.agents.critics"
    ),
)
