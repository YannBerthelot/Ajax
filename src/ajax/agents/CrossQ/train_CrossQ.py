"""CrossQ (Bhatt et al., ICLR 2024): SAC without target networks.

Batch Renormalization in the actor and the critics, and one training pass of
the critics over the rows ``(s, a)`` and their next states ``(s', a')``
together, so that both share the batch statistics: the bootstrap reads that
pass's second half, held constant. ``a'`` comes from the actor reading its
running statistics, and so does every critic read outside the critic loss.
The actor and the temperature step every ``policy_delay``-th update. All as
the authors' code (adityab/CrossQ@a318a26, ``sbx/sac/sac.py:195-400``).
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.serialization import to_state_dict

from ajax.agents.critics import CoreCriticAux, q_regression_loss
from ajax.agents.CrossQ.state import CrossQConfig, CrossQState
from ajax.agents.loop import (
    TrainLoop,
    actor_step,
    fold_critic_loss,
    fold_on_target,
    gradient_step,
    when,
)
from ajax.agents.recurrent import actor_dist_train, sample_replay
from ajax.agents.SAC import core
from ajax.agents.SAC.core import TemperatureAuxiliaries
from ajax.buffers.dataset import TransitionDataset
from ajax.controller import executed_fn
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, holds_expert, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import (
    action_value,
    action_value_input,
    predict_value_train,
)
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: CoreCriticAux
    extensions: dict  # the extensions' training metrics


def init_CrossQ(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    num_critics: int,
    collector_extras: Optional[dict] = None,
) -> CrossQState:
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
    actor_state, critic_state, collector_state = core.init_soft_actor_critic(
        init_key,
        collector_key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
        buffer,
        num_critics=num_critics,
        collector_extras=collector_extras,
    )
    return CrossQState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        # No target networks: nothing reads a copy of the critics.
        critic_state=critic_state.replace(target_params=None),
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
    )


def update_critics(
    agent_state: CrossQState,
    batch: Transition,
    config: CrossQConfig,
    stack: ExtensionStack,
    total_timesteps: int,
    executed: Optional[Callable] = None,
) -> tuple[CrossQState, CoreCriticAux, dict]:
    """One critic step on one training pass over ``(s, a)`` and ``(s',
    a')``, its running statistics kept (``sbx/sac/sac.py:251-342``); ``a'``
    is executed (``executed``, a residual policy's)."""
    agent_state, key, next_key = agent_state.next_keys(2)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    next_actions, next_log_probs = core.sample_next_actions(
        agent_state.actor_state,
        batch.next_obs,
        next_key,
        executed=executed,
        next_a_expert=batch.next_a_expert,
    )
    critic_state = agent_state.critic_state
    rows = jnp.concatenate(
        [
            action_value_input(batch.obs, batch.action),
            action_value_input(batch.next_obs, next_actions),
        ]
    )
    target_args: dict[str, Any] = {
        "rewards": batch.reward,
        "dones": batch.terminated,
        "gamma": config.gamma,
        "reward_scale": config.reward_scale,
    }

    def loss_fn(params: Any) -> tuple[jax.Array, Any]:
        q, batch_stats = predict_value_train(critic_state, params, rows)
        q_now, q_next = jnp.split(q, 2, axis=1)
        target = core.soft_bellman_target(
            jnp.min(q_next, axis=0),
            next_log_probs,
            batch.reward,
            batch.terminated,
            config.gamma,
            alpha,
            config.reward_scale,
        )
        target, target_metrics = fold_on_target(
            stack, agent_state, batch, target, key, total_timesteps, **target_args
        )
        loss, aux = q_regression_loss(q_now, target)
        extra, loss_metrics = fold_critic_loss(
            stack, agent_state, batch, params, target, key, total_timesteps
        )
        metrics = merge_metrics(target_metrics, loss_metrics)
        return loss + extra, (aux, batch_stats, metrics)

    critic_state, (aux, batch_stats, metrics), _ = gradient_step(critic_state, loss_fn)
    critic_state = critic_state.replace(batch_stats=batch_stats)
    return agent_state.replace(critic_state=critic_state), aux, metrics


def update_actor_and_temperature(
    agent_state: CrossQState,
    batch: Transition,
    config: CrossQConfig,
    stack: ExtensionStack,
    total_timesteps: int,
    executed: Optional[Callable] = None,
) -> tuple[CrossQState, tuple]:
    """The actor step in training, the critics read with their running
    statistics (at the executed actions, a residual policy's), then the
    temperature's on the same samples (``sbx/sac/sac.py:355-400``)."""
    agent_state, key = agent_state.next_keys()
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    actor_state, critic_state = agent_state.actor_state, agent_state.critic_state
    transform = None
    if executed is not None:
        score = executed

        def transform(actions: jax.Array) -> jax.Array:
            return score(actions, batch.a_expert)

    def policy_loss(params: Any) -> tuple[jax.Array, Any, jax.Array]:
        pi, batch_stats = actor_dist_train(actor_state, params, batch.obs)
        loss, (log_probs, q, pi) = core.soft_policy_objective(
            pi, critic_state, batch.obs, alpha, key, action_transform=transform
        )
        aux = PolicyAuxiliaries(
            policy_loss=loss, log_pi=log_probs.mean(), q_min=q.mean()
        )
        return loss, (aux, log_probs, batch_stats), pi.mean()

    actor_state, (aux, log_probs, batch_stats), _, metrics = actor_step(
        agent_state,
        policy_loss,
        stack,
        key,
        total_timesteps,
        observations=batch.obs,
        raw_observations=batch.raw_obs,
        a_expert=batch.a_expert,
        actions=batch.action,
        critic_state=critic_state,
    )
    actor_state = actor_state.replace(batch_stats=batch_stats)
    # The authors' temperature loss: alpha, not log alpha, times the gap.
    agent_state, temperature = core.update_temperature(
        agent_state.replace(actor_state=actor_state),
        jax.lax.stop_gradient(log_probs),
        jnp.asarray(config.target_entropy),
        core.alpha_temperature_loss_fn,
    )
    return agent_state, (aux, temperature, metrics)


def update_agent(
    agent_state: CrossQState,
    buffer: BufferType,
    config: CrossQConfig,
    stack: ExtensionStack,
    total_timesteps: int,
    executed: Optional[Callable] = None,
) -> tuple[CrossQState, AuxiliaryLogs]:
    """One update: a critic step on a fresh replay batch, then, on the
    ``policy_delay``-th, an actor and a temperature step on the same batch
    (the authors' ``(n_updates + i + 1) % policy_delay``,
    ``sbx/sac/sac.py:201``)."""
    agent_state, sample_key = agent_state.next_keys()
    batch, _ = sample_replay(agent_state, buffer, sample_key)
    agent_state, value, value_metrics = update_critics(
        agent_state, batch, config, stack, total_timesteps, executed
    )
    due = (agent_state.n_updates + 1) % config.policy_delay == 0
    agent_state, (policy, temperature, policy_metrics) = when(
        due,
        lambda s: update_actor_and_temperature(
            s, batch, config, stack, total_timesteps, executed
        ),
        agent_state,
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, AuxiliaryLogs(
        temperature=temperature,
        policy=policy,
        value=value,
        extensions=merge_metrics(value_metrics, policy_metrics),
    )


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: CrossQConfig,
    alpha_args: AlphaConfig,
    num_critics: int,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    expert_policy: Optional[Callable] = None,
    offline_data: Optional[TransitionDataset] = None,
    extensions: Sequence = (),
):
    """CrossQ's train function: one step per env, then (from
    ``learning_starts``) one update, per iteration."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "expert_policy": expert_policy,
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
        expert_policy=expert_policy,
        offline_data=offline_data,
    )

    def init(key: jax.Array) -> CrossQState:
        stored = holds_expert(expert_policy, loop.stack)
        return init_CrossQ(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            buffer,
            num_critics,
            collector_extras={
                "include_raw_obs": stored,
                "include_raw_next_obs": "raw_next_obs" in loop.stack.buffer_fields(),
                "include_return_to_go": "return_to_go" in loop.stack.buffer_fields(),
                "include_is_expert": stored,
                "expert_policy": expert_policy,
            },
        )

    def update(agent_state: CrossQState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: CrossQ samples it there.
        return update_agent(
            agent_state,
            buffer,
            agent_config,
            loop.stack,
            total_timesteps,
            executed_fn(env_args),
        )

    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        collect_kwargs={"buffer": buffer, "store_hidden": False},
        eval_kwargs={
            "expert_policy": expert_policy,
            "value_estimate": ValueEstimate(
                action_value, agent_config.gamma, agent_config.reward_scale
            ),
        },
    )
