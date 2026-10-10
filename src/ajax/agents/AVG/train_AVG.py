"""AVG (Vasan et al., 2024): Action Value Gradient, Algorithm 1.

Incremental deep RL: one environment step, then one update on that single
transition, with no replay buffer and no target network. The critic's TD
error is divided by a running estimate of its scale (from the statistics of
the entropy-regularised reward, the discount and the episode returns), the
entropy coefficient ``alpha`` stays at its initial value, and the actor's
gradient flows through the action actually taken.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents.AVG.state import AVGConfig, AVGState, NormalizationInfo
from ajax.agents.AVG.utils import compute_td_error_scaling
from ajax.agents.loop import TrainLoop, critic_step, gradient_step
from ajax.agents.recurrent import q_values
from ajax.agents.SAC import core
from ajax.agents.SAC.core import TemperatureAuxiliaries
from ajax.environments.interaction import get_pi
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDActorConfig
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred: jax.Array
    target_q: jax.Array
    log_probs: jax.Array
    scaling_coef: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def init_AVG(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    num_critics: int = 1,
    window_size: int = 10,
    pid_actor_config: Optional[PIDActorConfig] = None,
) -> AVGState:
    rng, init_key, collector_key = jax.random.split(key, num=3)
    actor_state, critic_state, collector_state = core.init_soft_actor_critic(
        init_key,
        collector_key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
        num_critics=num_critics,
        window_size=window_size,
        pid_actor_config=pid_actor_config,
    )
    # One running statistic per scalar (reward, discount, return), shared
    # by every env.
    init_norm_info = NormalizationInfo(
        value=jnp.zeros((env_args.n_envs, 1)),
        count=jnp.zeros((1,)),
        mean=jnp.zeros((1, 1)),
        mean_2=jnp.zeros((1, 1)),
    )
    return AVGState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
        reward=init_norm_info,
        gamma=init_norm_info,
        G_return=init_norm_info,
        scaling_coef=jnp.ones((1, 1)),
    )


def update_AVG_values(
    agent_state: AVGState, rollout: Transition, agent_config: AVGConfig
) -> AVGState:
    """Fold the step just taken into the statistics behind the TD-error
    scale: its entropy-regularised reward and discount and, where an
    episode ended, its return."""
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    r_ent = rollout.reward - alpha * jax.lax.stop_gradient(rollout.log_prob).sum(
        -1, keepdims=True
    )
    reward = agent_state.reward.replace(value=r_ent)
    gamma = agent_state.gamma.replace(
        value=agent_config.gamma * (1 - rollout.terminated)
    )
    new_G = agent_state.G_return.value + r_ent
    ended = rollout.terminated.astype(bool)
    # The statistics read the returns of the episodes that ended (NaN marks
    # the others); those restart at 0, the others accumulate.
    scaling_coef, reward, gamma, G_return = compute_td_error_scaling(
        reward,
        gamma,
        G_return=agent_state.G_return.replace(value=jnp.where(ended, new_G, jnp.nan)),
    )
    G_return = G_return.replace(value=jnp.where(ended, 0.0, new_G))
    return agent_state.replace(
        reward=reward, gamma=gamma, G_return=G_return, scaling_coef=scaling_coef
    )


def compute_avg_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    critic_params: FrozenDict,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    reward_scale: float,
) -> Tuple[jax.Array, jax.Array]:
    """AVG bellman target (no target network, uses current critic_params)."""
    rewards = rewards * reward_scale
    sample_key, _ = jax.random.split(rng)
    next_actions, next_log_probs = core.sample_next_actions(
        actor_state, next_observations, sample_key
    )
    q_target = jnp.min(
        q_values(critic_states, critic_params, next_observations, next_actions),
        axis=0,
    )
    target = jax.lax.stop_gradient(
        rewards + (1.0 - dones) * gamma * (q_target - alpha * next_log_probs),
    )
    return target, next_log_probs


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    next_log_probs: jax.Array,
    scaling_coef: jax.Array,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """The squared TD error divided by its running scale."""
    q_pred = jnp.min(
        q_values(critic_states, critic_params, observations, actions), axis=0
    )
    assert target_q.shape == q_pred.shape, f"{target_q.shape} != {q_pred.shape}"
    scaled_delta = (q_pred - target_q) / scaling_coef
    total_loss = jnp.mean(scaled_delta**2)
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q_pred=q_pred.mean().flatten(),
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
        scaling_coef=scaling_coef.mean().flatten(),
    )


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    raw_actions: jax.Array,
    alpha: jax.Array,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """``alpha log pi(a|s) - Q(s, a)`` at the action taken.

    The action is reparameterised with the noise ``eps`` of its stored
    pre-squash sample ``u``: ``a = tanh(mu(s) + sigma(s) eps)``, ``eps = (u
    - mu(s)) / sigma(s)``. The actor has not changed since it sampled ``u``,
    so ``a`` is the action executed and the gradient flows through it, as
    in AVG's update.
    """
    pi, _ = get_pi(actor_state, actor_params, observations)
    loc, scale = pi.distribution.loc, pi.distribution.scale
    eps = jax.lax.stop_gradient((raw_actions - loc) / scale)
    raw = loc + scale * eps
    log_probs = pi.log_prob_from_raw(raw)
    q_pred = jnp.min(
        q_values(critic_states, critic_states.params, observations, jnp.tanh(raw)),
        axis=0,
    )
    assert log_probs.shape == q_pred.shape, f"{log_probs.shape} != {q_pred.shape}"
    loss = (alpha * log_probs - q_pred).mean()
    return loss, PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_min=q_pred.mean()
    )


def update_value_functions(
    agent_state: AVGState,
    transition: Transition,
    agent_config: AVGConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[AVGState, ValueAuxiliaries]:
    """The critic step on the scaled TD error of ``transition``."""
    key, rng = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    critic_state = agent_state.critic_state
    dones = jnp.logical_or(transition.terminated, transition.truncated)
    target_q, next_log_probs = compute_avg_td_target(
        agent_state.actor_state,
        critic_state,
        critic_state.params,
        key,
        transition.next_obs,
        dones,
        transition.reward,
        agent_config.gamma,
        alpha,
        agent_config.reward_scale,
    )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        return value_loss_function(
            params,
            critic_state,
            transition.obs,
            transition.action,
            target_q,
            next_log_probs,
            agent_state.scaling_coef,
        )

    critic_state, aux = critic_step(
        agent_state,
        transition,
        target_q,
        value_loss,
        extension_stack,
        key,
        total_timesteps,
        rewards=transition.reward,
        dones=dones,
        gamma=agent_config.gamma,
        reward_scale=agent_config.reward_scale,
    )
    return agent_state.replace(rng=rng, critic_state=critic_state), aux


def update_policy(
    agent_state: AVGState,
    observations: jax.Array,
    raw_actions: jax.Array,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[AVGState, PolicyAuxiliaries]:
    """The actor step through the action taken."""
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    actor_state = agent_state.actor_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, PolicyAuxiliaries]:
        loss, aux = policy_loss_function(
            params,
            actor_state,
            agent_state.critic_state,
            observations,
            raw_actions,
            alpha,
        )
        actor_batch = {
            "observations": observations,
            "actor_params": params,
            "actor_state": actor_state,
        }
        extra = extension_stack.fold_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            agent_state.rng,
            total_timesteps,
        )
        return loss + extra, aux

    actor_state, aux = gradient_step(actor_state, loss_fn)
    return agent_state.replace(actor_state=actor_state), aux


def update_agent(
    agent_state: AVGState,
    transition: Transition,
    agent_config: AVGConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[AVGState, AuxiliaryLogs]:
    """One update on the step just taken: a critic and an actor step, both
    from the current parameters (the actor's loss sees the critic before
    its step). ``alpha`` is not updated."""
    critic_updated, aux_value = update_value_functions(
        agent_state, transition, agent_config, extension_stack, total_timesteps
    )
    policy_updated, aux_policy = update_policy(
        agent_state,
        transition.obs,
        transition.raw_action,
        extension_stack,
        total_timesteps,
    )
    agent_state = critic_updated.replace(actor_state=policy_updated.actor_state)
    log_alpha = agent_state.alpha.params["log_alpha"]
    aux = AuxiliaryLogs(
        temperature=TemperatureAuxiliaries(
            alpha=jnp.exp(log_alpha),
            log_alpha=log_alpha,
            effective_target_entropy=jnp.nan,  # alpha is not tuned
        ),
        policy=aux_policy,
        value=aux_value,
    )
    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: AVGConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    expert_policy: Optional[Callable] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    extensions: Sequence = (),
):
    """AVG's train function: one step per env, the TD-error statistics, then
    (from ``learning_starts``) one update on that step, per iteration."""
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )

    def init(key: jax.Array, _pretrain_key: jax.Array) -> AVGState:
        return init_AVG(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            num_critics=agent_config.num_critics,
            pid_actor_config=pid_actor_config,
        )

    def after_collect(agent_state: AVGState, transition: Transition) -> AVGState:
        return update_AVG_values(agent_state, transition, agent_config)

    def update(agent_state: AVGState, transition: Transition) -> Any:
        return update_agent(
            agent_state, transition, agent_config, loop.stack, total_timesteps
        )

    return loop.off_policy(
        init,
        update,
        AuxiliaryLogs,
        agent_config.learning_starts,
        expose_rollout=agent_config.expose_recent_rollout,
        after_collect=after_collect,
        eval_kwargs={"expert_policy": expert_policy},
    )
