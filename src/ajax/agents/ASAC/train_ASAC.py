"""ASAC (Adamczyk et al., 2025): Average-Reward Soft Actor-Critic.

SAC for the average-reward criterion: the critics learn the differential
soft Q, bootstrapped without discount on ``r - theta``, where ``theta``
tracks the entropy-regularised reward rate. Every target is shifted by the
target critics' value at the origin ``Q(0, 0)`` (their reference point), and
terminations are charged a penalty learned from the non-terminal rewards
(``p_0`` scales it) so episodic tasks fit the continuing criterion.
"""

from collections.abc import Sequence
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents.ASAC.state import ASACConfig, ASACState
from ajax.agents.ASAC.utils import (
    compute_episode_termination_penalty,
    get_episode_termination_penalized_rewards,
)
from ajax.agents.loop import TrainLoop, gradient_step
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    q_values,
    sample_replay,
    stored_actor_carry_dim,
)
from ajax.agents.SAC.train_SAC import (
    TemperatureAuxiliaries,
    create_alpha_train_state,
)
from ajax.agents.SAC.train_SAC import update_temperature as sac_temperature_step
from ajax.environments.interaction import init_collector_state
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.memory import zeros_carry_like
from ajax.networks.networks import (
    get_initialized_actor_critic,
    predict_value,
    predict_value_sequence,
)
from ajax.state import (
    AlphaConfig,
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
    log_pi: jax.Array
    q_min: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q1_pred: jax.Array
    q2_pred: jax.Array
    target_q: jax.Array
    log_probs: jax.Array


@struct.dataclass
class ThetaAuxiliaries:
    theta: jax.Array
    episode_termination_penalty: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    theta: ThetaAuxiliaries


def init_ASAC(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    window_size: int = 10,
    stored_state: bool = False,
    pid_actor_config: Optional[PIDActorConfig] = None,
) -> ASACState:
    rng, init_key, collector_key = jax.random.split(key, num=3)
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
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
    return ASACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
        episode_termination_penalty=jnp.zeros(()),
        theta=0.0,
    )


def compute_asac_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    theta: float,
    alpha: jax.Array,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, jax.Array]:
    """``r - theta + min_i Q_target_i(s', a') - Q_target(0, 0) - alpha log pi(a'|s')``.

    Returns the stop_gradient'd target and the next actions' log-probs.
    ``dones`` is unused: the differential target bootstraps through every
    end, terminations being charged their penalty in the rewards instead.
    """
    rewards = rewards * reward_scale
    next_pi = actor_dist(
        actor_state, actor_state.params, next_observations, carries, bootstrap=True
    )
    sample_key, _ = jax.random.split(rng)
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)
    q_targets = q_values(
        critic_states,
        critic_states.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    # The origin shift Q(0, 0): one zero input; in sequence mode with a
    # fresh zero carry (a batch of 1).
    zero_x = jnp.zeros((1, next_observations.shape[-1] + next_actions.shape[-1]))
    if carries is None:
        shift_value = predict_value(critic_states, critic_states.target_params, zero_x)
    else:
        shift_value, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_states.target_params,
            x=zero_x[None],
            resets=jnp.zeros((1, 1), dtype=bool),
            initial_hidden=zeros_carry_like(
                carries.target_critic_hidden, 1, batch_axis=1
            ),
        )
    shifted_q_targets = q_targets - jnp.mean(shift_value)
    q1_target, q2_target = jnp.split(shifted_q_targets, 2, axis=0)
    min_q_target = jnp.minimum(q1_target, q2_target).squeeze(0)
    target = jax.lax.stop_gradient(rewards - theta + (min_q_target - alpha * log_probs))
    return target, log_probs


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    next_log_probs: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """Both critics regress on the one differential target."""
    q_preds = q_values(critic_states, critic_params, observations, actions, carries)
    assert target_q.shape == q_preds.shape[1:], f"{target_q.shape} != {q_preds.shape}"
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    loss_q1 = 0.5 * jnp.mean((q1_pred.squeeze(0) - target_q) ** 2)
    loss_q2 = 0.5 * jnp.mean((q2_pred.squeeze(0) - target_q) ** 2)
    total_loss = loss_q1 + loss_q2
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q1_pred=q1_pred.mean().flatten(),
        q2_pred=q2_pred.mean().flatten(),
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
    )


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    alpha: jax.Array,
    rng: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, jax.Array]]:
    """``alpha log pi(a|s) - min_i Q_i(s, a)``, ``a ~ pi(.|s)``; also returns
    the policy mean for the actor-loss extensions.

    In sequence mode the critic carry was burned in on the BUFFER actions;
    evaluating fresh policy actions from it is the standard stored /
    burned-state approximation.
    """
    pi = actor_dist(actor_state, actor_params, observations, carries)
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
    q_preds = q_values(
        critic_states, critic_states.params, observations, actions, carries
    )
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    q_min = jnp.minimum(q1_pred, q2_pred).squeeze(0)
    log_probs = log_probs.sum(-1, keepdims=True)
    assert log_probs.shape == q_min.shape, f"{log_probs.shape} != {q_min.shape}"
    total_loss = (alpha * log_probs - q_min).mean()
    aux = PolicyAuxiliaries(
        policy_loss=total_loss, log_pi=log_probs.mean(), q_min=q_min.mean()
    )
    return total_loss, (aux, pi.mean())


def update_value_functions(
    agent_state: ASACState,
    batch: Transition,
    rewards: jax.Array,
    reward_scale: float,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[ASACState, ValueAuxiliaries]:
    """The critic step on the differential target of the penalised ``rewards``."""
    key, rng = jax.random.split(agent_state.rng)
    step = agent_state.collector_state.timestep
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    dones = jnp.logical_or(batch.terminated, batch.truncated)
    target_q, next_log_probs = compute_asac_td_target(
        agent_state.actor_state,
        agent_state.critic_state,
        key,
        batch.next_obs,
        dones,
        rewards,
        agent_state.theta,
        alpha,
        reward_scale,
        carries,
    )
    # ASAC is average-reward: no gamma, None tells the extensions so.
    target_batch = {
        "observations": batch.obs,
        "actions": batch.action,
        "next_observations": batch.next_obs,
        "rewards": rewards,
        "dones": dones,
        "gamma": None,
        "reward_scale": reward_scale,
    }
    target_q = jax.lax.stop_gradient(
        extension_stack.fold_on_target(
            agent_state, target_batch, target_q, step, key, total_timesteps
        )
    )
    critic_state = agent_state.critic_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, ValueAuxiliaries]:
        loss, aux = value_loss_function(
            params,
            critic_state,
            batch.obs,
            batch.action,
            target_q,
            next_log_probs,
            carries,
        )
        loss_batch = {
            "observations": batch.obs,
            "actions": batch.action,
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
    agent_state: ASACState,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[ASACState, PolicyAuxiliaries]:
    """The actor step."""
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    actor_state = agent_state.actor_state

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, PolicyAuxiliaries]:
        loss, (aux, pi_mean) = policy_loss_function(
            params,
            actor_state,
            agent_state.critic_state,
            observations,
            alpha,
            policy_key,
            carries,
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
            policy_key,
            total_timesteps,
        )
        return loss + extra, aux

    actor_state, aux = gradient_step(actor_state, loss_fn)
    return agent_state.replace(rng=rng, actor_state=actor_state), aux


def update_temperature(
    agent_state: ASACState,
    observations: jax.Array,
    target_entropy: float,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[ASACState, TemperatureAuxiliaries]:
    """SAC's temperature step, on fresh samples of the updated policy."""
    rng, sample_key = jax.random.split(agent_state.rng)
    actor_state = agent_state.actor_state
    pi = actor_dist(actor_state, actor_state.params, observations, carries)
    _, log_probs = pi.sample_and_log_prob(seed=sample_key)
    return sac_temperature_step(  # type: ignore[return-value]
        agent_state.replace(rng=rng),  # type: ignore[arg-type]
        log_probs,
        jnp.asarray(target_entropy),
    )


def update_theta(
    agent_state: ASACState,
    tau: float,
    rewards: jax.Array,
    observations: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> ASACState:
    """Track the entropy-regularised reward rate: an EMA of
    ``mean(r - alpha log pi(a|s))`` over the batch, ``a ~ pi(.|s)``."""
    action_key, rng = jax.random.split(agent_state.rng)
    pi = actor_dist(
        agent_state.actor_state, agent_state.actor_state.params, observations, carries
    )
    _, log_probs = pi.sample_and_log_prob(seed=action_key)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    new_theta = jnp.mean(rewards - alpha * log_probs.sum(-1, keepdims=True))
    theta = agent_state.theta * (1 - tau) + tau * new_theta
    return agent_state.replace(theta=theta, rng=rng)


def update_agent(
    agent_state: ASACState,
    buffer: BufferType,
    recurrent: bool,
    agent_config: ASACConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[ASACState, AuxiliaryLogs]:
    """One update on a replay batch: the termination penalty, then the
    critic, actor, temperature, ``theta`` and target steps."""
    tau = agent_config.tau
    sample_key, rng = jax.random.split(agent_state.rng)
    transition, carries = sample_replay(
        agent_state,
        buffer,
        sample_key,
        recurrent,
        agent_config.burn_in,
        agent_config.stored_state,
    )
    episode_termination_penalty = compute_episode_termination_penalty(
        agent_state.episode_termination_penalty,
        transition.reward,
        transition.terminated,
        agent_config.p_0,
        tau,
    )
    rewards = get_episode_termination_penalized_rewards(
        episode_termination_penalty, transition.reward, transition.terminated
    )
    agent_state = agent_state.replace(
        rng=rng, episode_termination_penalty=episode_termination_penalty
    )
    agent_state, aux_value = update_value_functions(
        agent_state,
        transition,
        rewards,
        agent_config.reward_scale,
        extension_stack,
        total_timesteps,
        carries,
    )
    agent_state, aux_policy = update_policy(
        agent_state,
        transition.obs,
        transition.raw_obs,
        extension_stack,
        total_timesteps,
        carries,
    )
    # The dual-gradient step towards ``target_entropy`` (target entropy
    # per dimension times the action dimension).
    agent_state, aux_temperature = update_temperature(
        agent_state,
        observations=transition.obs,
        target_entropy=agent_config.target_entropy,
        carries=carries,
    )
    agent_state = update_theta(agent_state, tau, rewards, transition.obs, carries)
    agent_state = agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau)
    )
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=aux_value,
        theta=ThetaAuxiliaries(
            theta=agent_state.theta,
            episode_termination_penalty=episode_termination_penalty,
        ),
    )
    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: ASACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    extensions: Sequence = (),
):
    """ASAC's train function: one step per env, then (from
    ``learning_starts``) one update, per iteration."""
    recurrent = network_args.memory is not None
    if recurrent and extensions:
        raise NotImplementedError("Recurrent ASAC does not support extensions yet.")
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )

    def init(key: jax.Array, _pretrain_key: jax.Array) -> ASACState:
        return init_ASAC(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            buffer,
            stored_state=agent_config.stored_state,
            pid_actor_config=pid_actor_config,
        )

    def update(agent_state: ASACState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: ASAC samples it from there.
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
            "store_hidden": recurrent and agent_config.stored_state,
        },
    )
