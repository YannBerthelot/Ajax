"""TD3 (Fujimoto et al., 2018): Twin Delayed DDPG, Algorithm 1.

Three changes vs DDPG:
1. Twin critics, target = min over the two target Qs (overestimation bias).
2. Target policy smoothing: target action = clip(mu_target(s') + clip(N, -c, c), -1, 1).
3. Delayed policy + target updates every `policy_delay` critic steps.

The actor reuses Ajax's stochastic SquashedNormal head and is treated
deterministically by taking pi.mean() (== tanh(mu)) for both target
and behaviour. Exploration noise is added at action time.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.cloning import CloningConfig, pretrain_on_expert
from ajax.agents.loop import TrainLoop, gradient_step
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    bootstrap_cuts,
    q_values,
    sample_replay,
    stored_actor_carry_dim,
    unsupported_recurrent_options,
)
from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.state import TD3Config, TD3State
from ajax.environments.interaction import (
    ActionPipelineResult,
    get_pi,
    init_collector_state,
)
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDActorConfig
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    q_mean: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_min: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def make_default_action_pipeline(recurrent: bool, exploration_noise: float):
    """TD3's behaviour policy: ``clip(mu(s) + N(0, exploration_noise), -1, 1)``,
    uniform actions during the warm-up."""

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        del raw_obs
        collector = agent_state.collector_state
        done = jnp.logical_or(collector.last_terminated, collector.last_truncated)
        pi, actor_state = get_pi(
            agent_state.actor_state,
            agent_state.actor_state.params,
            collector.last_obs,
            done,
            recurrent,
        )
        # A recurrent actor runs a one-step sequence: drop its time axis.
        mean_action = pi.mean().squeeze(0) if recurrent else pi.mean()
        noise = jax.random.normal(action_key, mean_action.shape) * exploration_noise
        policy_action = jnp.clip(mean_action + noise, -1.0, 1.0)
        uniform_action = jax.random.uniform(
            mix_key, minval=-1.0, maxval=1.0, shape=policy_action.shape
        )
        zeros = jnp.zeros(mean_action.shape[:-1] + (1,))
        return ActionPipelineResult(
            env_action=jax.lax.cond(
                uniform, lambda: uniform_action, lambda: policy_action
            ),
            policy_action=policy_action,
            log_probs=zeros,
            is_expert_flag=zeros,
            in_value_box=zeros,
            entry_bonus=zeros,
            rng=rng,
            new_actor_hidden=actor_state.hidden_state if recurrent else None,
        )

    return pipeline


def init_TD3(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    num_critics: int = 2,
    window_size: int = 10,
    stored_state: bool = False,
    pid_actor_config: Optional[PIDActorConfig] = None,
    expert_policy: Optional[Callable] = None,
) -> TD3State:
    rng, init_key, collector_key = jax.random.split(key, num=3)
    actor_state, critic_state = get_initialized_td3_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        num_critics=num_critics,
        pid_actor_config=pid_actor_config,
    )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode="gymnax" if check_env_is_gymnax(env_args.env) else "brax",
        buffer=buffer,
        window_size=window_size,
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
    )
    # A stateful expert (PID integrator, CPG phase) threads its state
    # through collection; seed it so the scan carry keeps one shape.
    if expert_policy is not None and hasattr(expert_policy, "init_state"):
        collector_state = collector_state.replace(
            expert_state=expert_policy.init_state(env_args.n_envs)
        )
    return TD3State(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
    )


def compute_td3_td_target(
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    target_policy_noise: float,
    target_noise_clip: float,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
) -> jax.Array:
    """y = r + gamma * (1-d) * min_i Q_target_i(s', clip(mu_target(s') + clip(N, -c, c), -1, 1))."""
    rewards = rewards * reward_scale
    next_action = actor_dist(
        actor_state,
        actor_state.target_params,
        next_observations,
        carries,
        bootstrap=True,
        target_actor=True,
    ).mean()
    noise = jax.random.normal(rng, next_action.shape) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    q_targets = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_action,
        carries,
        bootstrap=True,
    )
    target = rewards + gamma * (1.0 - dones) * jnp.min(q_targets, axis=0)
    return jax.lax.stop_gradient(target)


def value_loss_function(
    critic_params: FrozenDict,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """Both critics regress on the one target: ``sum_i (Q_i(s, a) - y)^2``."""
    q_preds = q_values(critic_state, critic_params, observations, actions, carries)
    loss = jnp.mean((q_preds - target_q) ** 2)
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        target_q=target_q.mean().flatten(),
        q_pred_min=jnp.min(q_preds, axis=0).mean().flatten(),
    )


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, jax.Array]]:
    """TD3 actor loss ``-Q1(s, mu(s))``; also returns ``mu(s)`` for extensions."""
    actions = actor_dist(actor_state, actor_params, observations, carries).mean()
    q_first = q_values(
        critic_state, critic_state.params, observations, actions, carries
    )[0]  # TD3 uses Q1 only for the actor objective.
    loss = -q_first.mean()
    return loss, (PolicyAuxiliaries(policy_loss=loss, q_mean=q_first.mean()), actions)


def update_value_functions(
    agent_state: TD3State,
    batch: Transition,
    agent_config: TD3Config,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[TD3State, ValueAuxiliaries]:
    """The critic step on the smoothed, clipped double-Q target."""
    key, rng = jax.random.split(agent_state.rng)
    step = agent_state.collector_state.timestep
    dones = bootstrap_cuts(batch, carries)
    target_q = compute_td3_td_target(
        agent_state.actor_state,
        agent_state.critic_state,
        key,
        batch.next_obs,
        dones,
        batch.reward,
        agent_config.gamma,
        agent_config.target_policy_noise,
        agent_config.target_noise_clip,
        agent_config.reward_scale,
        carries,
    )
    target_batch = {
        "observations": batch.obs,
        "actions": batch.action,
        "next_observations": batch.next_obs,
        "rewards": batch.reward,
        "dones": dones,
        "gamma": agent_config.gamma,
        "reward_scale": agent_config.reward_scale,
    }
    target_q = jax.lax.stop_gradient(
        extension_stack.fold_on_target(
            agent_state, target_batch, target_q, step, key, total_timesteps
        )
    )
    critic_state = agent_state.critic_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, ValueAuxiliaries]:
        loss, aux = value_loss_function(
            params, critic_state, batch.obs, batch.action, target_q, carries
        )
        loss_batch = {
            "observations": batch.obs,
            "actions": batch.action,
            "targets": target_q,
            "critic_params": params,
            "critic_state": critic_state,
        }
        extra = extension_stack.fold_critic_loss(
            agent_state, loss_batch, step, key, total_timesteps
        )
        return loss + extra, aux

    critic_state, aux = gradient_step(critic_state, loss_fn)
    return agent_state.replace(rng=rng, critic_state=critic_state), aux


def update_policy(
    agent_state: TD3State,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[TD3State, PolicyAuxiliaries]:
    """The deterministic policy-gradient step on ``-Q1(s, mu(s))``."""
    actor_state = agent_state.actor_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, PolicyAuxiliaries]:
        loss, (aux, pi_mean) = policy_loss_function(
            params, actor_state, agent_state.critic_state, observations, carries
        )
        actor_batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi_mean,
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


def update_target_networks(agent_state: TD3State, tau: float) -> TD3State:
    """Polyak-average both target networks (actor and critics)."""
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau),
        actor_state=agent_state.actor_state.soft_update(tau=tau),
    )


def update_agent(
    agent_state: TD3State,
    buffer: BufferType,
    recurrent: bool,
    agent_config: TD3Config,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[TD3State, AuxiliaryLogs]:
    """One update: a critic step on a replay batch and, every
    ``policy_delay`` updates, an actor step and the target updates."""
    sample_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    # Sequence replay burns the TARGET actor's carry in as well: TD3's
    # bootstrap action comes from it.
    batch, carries = sample_replay(
        agent_state,
        buffer,
        sample_key,
        recurrent,
        agent_config.burn_in,
        agent_config.stored_state,
        burn_target_actor=True,
    )
    agent_state, value_aux = update_value_functions(
        agent_state, batch, agent_config, extension_stack, total_timesteps, carries
    )

    def policy_and_targets(agent_state: TD3State) -> Tuple[TD3State, Any]:
        agent_state, policy_aux = update_policy(
            agent_state,
            batch.obs,
            batch.raw_obs,
            extension_stack,
            total_timesteps,
            carries,
        )
        return update_target_networks(agent_state, agent_config.tau), policy_aux

    def skip_policy(agent_state: TD3State) -> Tuple[TD3State, Any]:
        zero = jnp.zeros(())
        return agent_state, PolicyAuxiliaries(policy_loss=zero, q_mean=zero)

    policy_delay = agent_config.policy_delay
    do_policy = (agent_state.n_updates % policy_delay) == 0
    agent_state, policy_aux = jax.lax.cond(
        do_policy, policy_and_targets, skip_policy, agent_state
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, AuxiliaryLogs(policy=policy_aux, value=value_aux)


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: TD3Config,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    action_pipeline: Optional[Callable] = None,
    extensions: Sequence = (),
):
    """TD3's train function: one step per env, then (from ``learning_starts``)
    one update, per iteration."""
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options(
            "TD3",
            expert_policy=expert_policy,
            extensions=(tuple(extensions) or None),
            # TD3 always builds a default CloningConfig; only actual
            # pre-training (pre_train_n_steps > 0) conflicts with memory.
            cloning_pretrain=(
                cloning_args
                if cloning_args is not None and cloning_args.pre_train_n_steps > 0
                else None
            ),
            pid_actor_config=pid_actor_config,
        )
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )
    if action_pipeline is None:
        action_pipeline = make_default_action_pipeline(
            recurrent, agent_config.exploration_noise
        )

    def init(key: jax.Array, pretrain_key: jax.Array) -> TD3State:
        agent_state = init_TD3(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            buffer,
            num_critics=agent_config.num_critics,
            stored_state=agent_config.stored_state,
            pid_actor_config=pid_actor_config,
            expert_policy=expert_policy,
        )
        return pretrain_on_expert(
            agent_state,
            pretrain_key,
            cloning_args,
            expert_policy,
            env_args,
            actor_optimizer_args,
        )

    def update(agent_state: TD3State, _transition: Transition) -> Any:
        # The step just collected is in the buffer: TD3 samples it from there.
        return update_agent(
            agent_state, buffer, recurrent, agent_config, loop.stack, total_timesteps
        )

    return loop.off_policy(
        init,
        update,
        AuxiliaryLogs,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "action_pipeline": action_pipeline,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={"expert_policy": expert_policy},
    )
