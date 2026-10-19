"""IQL (Kostrikov, Nair and Levine, ICLR 2022): offline RL that never
queries the critic at actions outside the data.

One update on a replay batch, in the order of the reference
(ikostrikov/implicit_q_learning@09d7002, learner.py ``_update_jit``)::

    q = min_i Q_target_i(s, a)                              # the rows' actions
    V    <- argmin  E[ |expectile - 1(q - V(s) < 0)| (q - V(s))^2 ]   critic.py
    actor <- argmax E[ min(exp(temperature (q - V(s))), 100) log pi(a | s) ]
                                                            # actor.py, the new V
    Q    <- argmin  E[ (Q(s, a) - r - gamma (1 - d) V(s'))^2 ]
    Q_target <- tau Q + (1 - tau) Q_target

On the shared off-policy loop it trains offline (``train_offline``, the
paper's setting) and then fine-tunes online with the same update, the
offline rows kept in its replay (Sec. 5.3). Deviation: the critics' loss is
their mean squared error, half the reference's sum over the two (Adam's
step is the same).
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.IQL.state import IQLConfig, IQLState
from ajax.agents.loop import TrainLoop, critic_step, gradient_step
from ajax.agents.recurrent import q_values, sample_replay
from ajax.agents.SAC.core import critic_loss_fn
from ajax.buffers.dataset import TransitionDataset
from ajax.environments.interaction import get_pi, init_collector_state
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import (
    action_value,
    get_initialized_actor_critic,
    predict_value,
)
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig, Transition
from ajax.types import BufferType


@struct.dataclass
class Auxiliaries:
    value_loss: jax.Array
    actor_loss: jax.Array
    critic_loss: jax.Array
    advantage: jax.Array  # mean q - V(s) on the batch
    weight: jax.Array  # mean advantage weight


@struct.dataclass
class AuxiliaryLogs:
    value: Auxiliaries
    extensions: dict  # the extensions' training metrics


def expectile_loss(diff: jax.Array, expectile: float) -> jax.Array:
    """``|expectile - 1(diff < 0)| diff^2`` (critic.py ``loss``)."""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff**2


def init_IQL(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    value_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    store_raw_next_obs: bool = False,
) -> IQLState:
    """A Gaussian actor, two Q critics and one state value V(s); the replay
    keeps the env's own next observations for an extension reading them."""
    rng, init_key, value_key, collector_key, eval_rng = jax.random.split(key, 5)
    actor_state, critic_state = get_initialized_actor_critic(
        init_key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
        continuous=True,
        action_value=True,
        num_critics=2,
    )
    _, value_state = get_initialized_actor_critic(
        value_key,
        env_args,
        actor_optimizer_args,
        value_optimizer_args,
        network_args,
        continuous=True,
        num_critics=1,
    )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=env_args.mode,
        buffer=buffer,
        include_raw_next_obs=store_raw_next_obs,
    )
    return IQLState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        value_state=value_state,
        collector_state=collector_state,
    )


def update_agent(
    agent_state: IQLState,
    buffer: BufferType,
    config: IQLConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[IQLState, AuxiliaryLogs]:
    """One IQL update (the module docstring)."""
    sample_key, critic_key, actor_key, rng = jax.random.split(agent_state.rng, 4)
    batch, _ = sample_replay(agent_state, buffer, sample_key)
    critic = agent_state.critic_state
    q = jnp.min(q_values(critic, critic.target_params, batch.obs, batch.action), axis=0)

    def value_loss(params: Any) -> Tuple[jax.Array, None]:
        v = predict_value(agent_state.value_state, params, batch.obs)[0]
        return jnp.mean(expectile_loss(q - v, config.expectile)), None

    value_state, _, _ = gradient_step(agent_state.value_state, value_loss)
    v_loss = value_loss(value_state.params)[0]
    v = predict_value(value_state, value_state.params, batch.obs)[0]
    weight = jnp.minimum(jnp.exp(config.temperature * (q - v)), 100.0)
    actor = agent_state.actor_state

    def actor_loss(params: Any) -> Tuple[jax.Array, Any]:
        pi, _ = get_pi(actor, params, batch.obs)
        log_probs = pi.log_prob(batch.action).sum(-1, keepdims=True)
        loss = -jnp.mean(weight * log_probs)
        actor_batch = {
            "observations": batch.obs,
            "raw_observations": batch.raw_obs,
            "pi_mean": pi.mean(),
            "actor_params": params,
            "actor_state": actor,
            "actions": batch.action,
            "critic_state": critic,
        }
        extra, metrics = extension_stack.fold_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            actor_key,
            total_timesteps,
        )
        return loss + extra, (loss, metrics)

    actor_state, (a_loss, actor_metrics), _ = gradient_step(actor, actor_loss)
    next_v = predict_value(value_state, value_state.params, batch.next_obs)[0]
    dones = batch.terminated
    target_q = (
        config.reward_scale * batch.reward + config.gamma * (1.0 - dones) * next_v
    )

    def q_loss(params: Any, target_q: jax.Array) -> Tuple[jax.Array, jax.Array]:
        loss, _ = critic_loss_fn(params, critic, batch.obs, batch.action, target_q)
        return loss, loss

    critic_state, c_loss, _, critic_metrics = critic_step(
        agent_state,
        batch,
        target_q,
        q_loss,
        extension_stack,
        critic_key,
        total_timesteps,
        rewards=batch.reward,
        dones=dones,
        gamma=config.gamma,
        reward_scale=config.reward_scale,
    )
    agent_state = agent_state.replace(
        rng=rng,
        actor_state=actor_state,
        critic_state=critic_state.soft_update(tau=config.tau),
        value_state=value_state,
        n_updates=agent_state.n_updates + 1,
    )
    return agent_state, AuxiliaryLogs(
        value=Auxiliaries(
            value_loss=v_loss,
            actor_loss=a_loss,
            critic_loss=c_loss,
            advantage=jnp.mean(q - v),
            weight=jnp.mean(weight),
        ),
        extensions=merge_metrics(critic_metrics, actor_metrics),
    )


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: IQLConfig,
    total_timesteps: int,
    num_episode_test: int,
    value_optimizer_args: OptimizerConfig,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    offline_data: Optional[TransitionDataset] = None,
    extensions: Sequence = (),
) -> Callable:
    """IQL's train function: offline, one update per iteration; online, one
    step per env, then (from ``learning_starts``) one update."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "buffer": buffer,
    }
    extensions = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
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

    def init(key: jax.Array) -> IQLState:
        return init_IQL(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            value_optimizer_args,
            network_args,
            buffer,
            store_raw_next_obs="raw_next_obs" in loop.stack.buffer_fields(),
        )

    def update(agent_state: IQLState, _transition: Optional[Transition]) -> Any:
        return update_agent(
            agent_state, buffer, agent_config, loop.stack, total_timesteps
        )

    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        collect_kwargs={"buffer": buffer},
        eval_kwargs={
            "value_estimate": ValueEstimate(
                action_value, agent_config.gamma, agent_config.reward_scale
            )
        },
    )
