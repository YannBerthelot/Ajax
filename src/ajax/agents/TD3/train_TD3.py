"""TD3 (Fujimoto et al., 2018): Twin Delayed DDPG, Algorithm 1.

Three changes vs DDPG:
1. Twin critics, target = min over the two target Qs (overestimation bias).
2. Target policy smoothing: target action = clip(mu_target(s') + clip(N, -c, c), -1, 1).
3. Delayed policy + target updates every `policy_delay` critic steps.

The actor is deterministic (networks.py: a Deterministic wrapper of
tanh(mu)), and pi.mean() gives its action for both target and
behaviour. Exploration noise is added at action time.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.loop import TrainLoop, actor_step, critic_step, when
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    q_values,
    sample_replay,
    stored_actor_carry_dim,
    unsupported_recurrent_options,
)
from ajax.agents.SAC.core import critic_loss_fn
from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.state import TD3Config, TD3State
from ajax.buffers.dataset import TransitionDataset
from ajax.controller import executed_fn
from ajax.environments.interaction import (
    Explore,
    explore_pipeline,
    init_collector_state,
)
from ajax.environments.utils import critic_action_units
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, holds_expert, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import action_value
from ajax.schedule import Schedule, at
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
    # Before clipping; the loss builds these metrics before the gradient
    # exists, the step fills it in.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_min: jax.Array
    q_gap: jax.Array  # mean max - min over the critics: |Q1 - Q2| for two
    target_noise_clip_frac: jax.Array  # share of the target noise clipped
    grad_norm: jax.Array | float = jnp.nan  # as the policy's


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


def gaussian_exploration(exploration_noise: Schedule) -> Explore:
    """TD3's behaviour policy: ``clip(mu(s) + N(0, exploration_noise), -1, 1)``,
    the noise's scale a schedule of the timestep."""

    def explore(pi: Any, key: jax.Array, step: jax.Array) -> jax.Array:
        mean_action = pi.mean()
        sigma = at(exploration_noise, step)
        noise = jax.random.normal(key, mean_action.shape) * sigma
        return jnp.clip(mean_action + noise, -1.0, 1.0)

    return explore


def make_default_action_pipeline(
    recurrent: bool, exploration_noise: Schedule, transform: Any = None
):
    """TD3's action pipeline: :func:`gaussian_exploration`, through the run's
    residual ``transform`` where it has one, uniform actions during the
    warm-up."""
    return explore_pipeline(
        gaussian_exploration(exploration_noise), recurrent, transform=transform
    )


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
    expert_policy: Optional[Callable] = None,
    store_expert_data: bool = False,
    store_raw_next_obs: bool = False,
    store_return_to_go: bool = False,
    store_next_action: bool = False,
    critic_action_units: Optional[tuple] = None,
) -> TD3State:
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
    actor_state, critic_state = get_initialized_td3_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        num_critics=num_critics,
    )
    critic_state = critic_state.replace(action_units=critic_action_units)
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=env_args.mode,
        buffer=buffer,
        window_size=window_size,
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
        include_raw_obs=store_expert_data,
        include_raw_next_obs=store_raw_next_obs,
        include_return_to_go=store_return_to_go,
        include_next_action=store_next_action,
        include_is_expert=store_expert_data,
        expert_policy=expert_policy,
    )
    return TD3State(
        rng=rng,
        eval_rng=eval_rng,
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
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """y = r + gamma * (1-d) * min_i Q_target_i(s', clip(mu_target(s') + clip(N, -c, c), -1, 1)),
    the share of the noise samples the clip to ``[-c, c]`` changed, and
    those smoothed next actions, executed (``executed``, a residual policy's,
    from the expert's stored next actions)."""
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
    clip_frac = (jnp.abs(noise) > target_noise_clip).mean()
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)
    if executed is not None:
        next_action = executed(next_action, next_a_expert)
    q_targets = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_action,
        carries,
        bootstrap=True,
    )
    target = rewards + gamma * (1.0 - dones) * jnp.min(q_targets, axis=0)
    return jax.lax.stop_gradient(target), clip_frac, next_action


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    a_expert: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, jax.Array]]:
    """TD3 actor loss ``-Q1(s, mu(s))``, at the executed action with a
    residual policy (``executed``); also returns ``mu(s)`` for extensions."""
    actions = actor_dist(actor_state, actor_params, observations, carries).mean()
    scored = actions if executed is None else executed(actions, a_expert)
    q_first = q_values(
        critic_state, critic_state.params, observations, scored, carries
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
    executed: Optional[Callable] = None,
) -> Tuple[TD3State, ValueAuxiliaries, dict]:
    """The critic step on the smoothed, clipped double-Q target."""
    agent_state, key = agent_state.next_keys()
    # A truncated row bootstraps on its final observation (bootstrap_obs).
    dones = batch.terminated
    target_q, noise_clip_frac, next_actions = compute_td3_td_target(
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
        executed,
        batch.next_a_expert,
    )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        """Every critic regresses on the one target; their squared errors are
        summed, as sfujim/TD3 sums its two critics' MSEs (TD3.py train):
        critic_loss_fn averages them, and averaging would halve the gradient,
        which Adam's eps then weighs twice as much."""
        loss, aux = critic_loss_fn(
            params, agent_state.critic_state, batch.obs, batch.action, target_q, carries
        )
        loss = loss * agent_config.num_critics
        return loss, ValueAuxiliaries(
            critic_loss=loss,
            target_q=target_q.mean().flatten(),
            q_pred_min=aux.q_pred_min,
            q_gap=aux.q_gap,
            target_noise_clip_frac=noise_clip_frac,
        )

    critic_state, aux, grad_norm, metrics = critic_step(
        agent_state,
        batch,
        target_q,
        value_loss,
        extension_stack,
        key,
        total_timesteps,
        rewards=batch.reward,
        dones=dones,
        gamma=agent_config.gamma,
        reward_scale=agent_config.reward_scale,
        # The smoothed next actions the target scored, for a target term
        # on them (TargetBC).
        target_extras={"next_actions": next_actions},
    )
    aux = aux.replace(grad_norm=grad_norm)
    return agent_state.replace(critic_state=critic_state), aux, metrics


def update_policy(
    agent_state: TD3State,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    a_expert: Optional[jax.Array] = None,
    actions: Optional[jax.Array] = None,
    executed: Optional[Callable] = None,
) -> Tuple[TD3State, PolicyAuxiliaries, dict]:
    """The deterministic policy-gradient step on ``-Q1(s, mu(s))``;
    ``a_expert``, the expert's stored actions at the rows (None without an
    expert), and ``actions``, the rows' own, reach the extensions. Also
    returns the extensions' metrics."""
    agent_state, key = agent_state.next_keys()

    def policy_loss(params: FrozenDict) -> Tuple[jax.Array, Any, jax.Array]:
        loss, (aux, pi_mean) = policy_loss_function(
            params,
            agent_state.actor_state,
            agent_state.critic_state,
            observations,
            carries,
            executed,
            a_expert,
        )
        return loss, aux, pi_mean

    actor_state, aux, grad_norm, metrics = actor_step(
        agent_state,
        policy_loss,
        extension_stack,
        key,
        total_timesteps,
        observations=observations,
        raw_observations=raw_observations,
        a_expert=a_expert,
        actions=actions,
        critic_state=agent_state.critic_state,
    )
    aux = aux.replace(grad_norm=grad_norm)
    return agent_state.replace(actor_state=actor_state), aux, metrics


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
    executed: Optional[Callable] = None,
) -> Tuple[TD3State, AuxiliaryLogs]:
    """One update: a critic step on a replay batch and, every
    ``policy_delay`` updates, an actor step and the target updates."""
    agent_state, sample_key = agent_state.next_keys()
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
    agent_state, value_aux, value_metrics = update_value_functions(
        agent_state,
        batch,
        agent_config,
        extension_stack,
        total_timesteps,
        carries,
        executed,
    )

    def policy_and_targets(agent_state: TD3State) -> Tuple[TD3State, Any]:
        agent_state, policy_aux, policy_metrics = update_policy(
            agent_state,
            batch.obs,
            batch.raw_obs,
            extension_stack,
            total_timesteps,
            carries,
            a_expert=batch.a_expert,
            actions=batch.action,
            executed=executed,
        )
        agent_state = update_target_networks(agent_state, agent_config.tau)
        return agent_state, (policy_aux, policy_metrics)

    # sfujim/TD3 counts the update first (total_it += 1), then steps the actor
    # when the count is a multiple of the delay: its first actor step is the
    # policy_delay-th update, not the first. Skipped, its metrics are zeros.
    do_policy = ((agent_state.n_updates + 1) % agent_config.policy_delay) == 0
    agent_state, (policy_aux, policy_metrics) = when(
        do_policy, policy_and_targets, agent_state, fill=0.0
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, AuxiliaryLogs(
        policy=policy_aux,
        value=value_aux,
        extensions=merge_metrics(value_metrics, policy_metrics),
    )


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
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    expert_policy: Optional[Callable] = None,
    action_pipeline: Optional[Callable] = None,
    offline_data: Optional[TransitionDataset] = None,
    extensions: Sequence = (),
):
    """TD3's train function: one step per env, then (from ``learning_starts``)
    one update, per iteration."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "expert_policy": expert_policy,
        "buffer": buffer,
    }
    extensions = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options(
            "TD3",
            expert_policy=expert_policy,
            extensions=(tuple(extensions) or None),
        )
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
    if action_pipeline is None:
        action_pipeline = make_default_action_pipeline(
            recurrent,
            agent_config.exploration_noise,
            next(iter(env_args.transforms), None),
        )

    # sfujim/TD3's critics read max_action * tanh, the env's units: so do
    # TD3's (computed once here, from the concrete env params).
    units = critic_action_units(env_args.env, env_args.env_params)

    def init(key: jax.Array) -> TD3State:
        return init_TD3(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            buffer,
            num_critics=agent_config.num_critics,
            stored_state=agent_config.stored_state,
            expert_policy=expert_policy,
            store_expert_data=holds_expert(expert_policy, loop.stack),
            store_raw_next_obs="raw_next_obs" in loop.stack.buffer_fields(),
            store_return_to_go="return_to_go" in loop.stack.buffer_fields(),
            store_next_action="next_action" in loop.stack.buffer_fields(),
            critic_action_units=units,
        )

    def update(agent_state: TD3State, _transition: Transition) -> Any:
        # The step just collected is in the buffer: TD3 samples it from there.
        return update_agent(
            agent_state,
            buffer,
            recurrent,
            agent_config,
            loop.stack,
            total_timesteps,
            executed_fn(env_args),
        )

    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "action_pipeline": action_pipeline,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={
            "expert_policy": expert_policy,
            "value_estimate": ValueEstimate(
                action_value, agent_config.gamma, agent_config.reward_scale
            ),
        },
    )
