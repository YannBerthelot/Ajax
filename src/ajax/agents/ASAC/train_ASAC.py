"""ASAC (Adamczyk et al., 2025): Average-Reward Soft Actor-Critic.

SAC for the average-reward criterion: the critics learn the differential
soft Q, bootstrapped without discount on ``r - theta``, where ``theta``
tracks the entropy-regularised reward rate. Every target is shifted by the
target critics' value at the origin ``Q(0, 0)`` (their reference point), and
a termination drops the next value and is charged a penalty learned from the
rewards (``p_0`` scales it) so episodic tasks fit the continuing criterion.
"""

import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple, Union

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
from ajax.agents.loop import TrainLoop, critic_step
from ajax.agents.recurrent import (
    RecurrentCarries,
    q_values,
    sample_replay,
    unsupported_recurrent_options,
)
from ajax.agents.SAC import core
from ajax.agents.SAC.core import TemperatureAuxiliaries
from ajax.agents.steps import update_target_networks
from ajax.buffers.dataset import TransitionDataset
from ajax.controller import executed_fn
from ajax.extensions.base import ExtensionStack, holds_expert, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.memory import zeros_carry_like
from ajax.networks.networks import (
    action_value_input,
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
    theta: Union[float, jax.Array]
    episode_termination_penalty: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    theta: ThetaAuxiliaries
    extensions: dict  # the extensions' training metrics


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
    store_expert_data: bool = False,
    expert_policy: Optional[Callable] = None,
    store_raw_next_obs: bool = False,
) -> ASACState:
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
    actor_state, critic_state, collector_state = core.init_soft_actor_critic(
        init_key,
        collector_key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
        buffer,
        num_critics=2,
        window_size=window_size,
        stored_state=stored_state,
        collector_extras={
            "include_raw_obs": store_expert_data,
            "include_raw_next_obs": store_raw_next_obs,
            "include_is_expert": store_expert_data,
            "expert_policy": expert_policy,
        },
    )
    return ASACState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
        episode_termination_penalty=jnp.zeros(()),
        theta=0.0,
    )


def uniform_log_prob(action_dim: int) -> float:
    """log(1 / nA), the authors' reference point for the entropy term
    (``darer/ASAC.py:54``, ``:182``, ``:191``): 0 for one action dimension."""
    return -math.log(action_dim)


def compute_asac_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    theta: float | jax.Array,
    alpha: jax.Array,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """``r - theta + (1 - done) (min_i Q_target_i(s', a') - alpha (log pi(a'|s')
    - log(1 / nA))) - Q_target(0, 0)``, ``r`` with a termination's penalty.

    As the authors' code (BobMak/asac@4b68862, ``darer/ASAC.py:179-205``): a
    termination drops the next value and is charged the penalty; the origin
    shift applies to every row. ``a'`` is executed (``executed``, a residual
    policy's). Returns the stop_gradient'd target and the next actions'
    log-probs.
    """
    rewards = rewards * reward_scale
    sample_key, _ = jax.random.split(rng)
    next_actions, log_probs = core.sample_next_actions(
        actor_state, next_observations, sample_key, carries, executed, next_a_expert
    )
    q_targets = q_values(
        critic_states,
        critic_states.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    # The origin shift Q(0, 0): one zero input; in sequence mode at an
    # episode start (a batch of 1, a fresh carry, no previous action).
    zero_obs = jnp.zeros((1, next_observations.shape[-1]))
    zero_action = jnp.zeros((1, next_actions.shape[-1]))
    assert critic_states.target_params is not None
    if carries is None:
        zero_x = action_value_input(zero_obs, zero_action)
        shift_value = predict_value(critic_states, critic_states.target_params, zero_x)
    else:
        shift_value, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_states.target_params,
            x=action_value_input(zero_obs, zero_action, zero_action)[None],
            resets=jnp.zeros((1, 1), dtype=bool),
            initial_hidden=zeros_carry_like(
                carries.target_critic_hidden, 1, batch_axis=1
            ),
        )
    q1_target, q2_target = jnp.split(q_targets, 2, axis=0)
    min_q_target = jnp.minimum(q1_target, q2_target).squeeze(0)
    logpi0 = uniform_log_prob(next_actions.shape[-1])
    next_value = (1.0 - dones) * (min_q_target - alpha * (log_probs - logpi0))
    target = rewards - theta + next_value - jnp.mean(shift_value)
    return jax.lax.stop_gradient(target), log_probs


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


def update_value_functions(
    agent_state: ASACState,
    batch: Transition,
    rewards: jax.Array,
    reward_scale: float,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable] = None,
) -> Tuple[ASACState, ValueAuxiliaries, dict]:
    """The critic step on the differential target of the penalised ``rewards``."""
    agent_state, key = agent_state.next_keys()
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    dones = batch.terminated
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
        executed,
        batch.next_a_expert,
    )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        return value_loss_function(
            params,
            agent_state.critic_state,
            batch.obs,
            batch.action,
            target_q,
            next_log_probs,
            carries,
        )

    # ASAC is average-reward: no gamma, None tells the extensions so.
    critic_state, aux, _, metrics = critic_step(
        agent_state,
        batch,
        target_q,
        value_loss,
        extension_stack,
        key,
        total_timesteps,
        rewards=rewards,
        dones=dones,
        gamma=None,
        reward_scale=reward_scale,
    )
    return agent_state.replace(critic_state=critic_state), aux, metrics


def update_policy(
    agent_state: ASACState,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    a_expert: Optional[jax.Array] = None,
    actions: Optional[jax.Array] = None,
    executed: Optional[Callable] = None,
) -> Tuple[ASACState, PolicyAuxiliaries, jax.Array, dict]:
    """The actor step on ``alpha log pi(a|s) - min_i Q_i(s, a)``; also
    returns its samples' log-probs (the temperature and ``theta`` read them)
    and the extensions' metrics."""
    agent_state, (loss, log_probs, q_min), metrics = core.soft_actor_step(
        agent_state,
        observations,
        raw_observations,
        extension_stack,
        total_timesteps,
        carries,
        a_expert=a_expert,
        actions=actions,
        executed=executed,
    )
    aux = PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_min=q_min.mean()
    )
    return agent_state, aux, log_probs, metrics


def update_theta(
    agent_state: ASACState,
    tau: float,
    rewards: jax.Array,
    log_probs: jax.Array,
    action_dim: int,
) -> ASACState:
    """Track the entropy-regularised reward rate: an EMA of ``mean(r -
    alpha (log pi(a|s) - log(1 / nA)))`` over the batch, ``a ~ pi(.|s)``."""
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    log_pi = log_probs.sum(-1, keepdims=True) - uniform_log_prob(action_dim)
    new_theta = jnp.mean(rewards - alpha * log_pi)
    theta = agent_state.theta * (1 - tau) + tau * new_theta
    return agent_state.replace(theta=theta)


def update_agent(
    agent_state: ASACState,
    buffer: BufferType,
    recurrent: bool,
    agent_config: ASACConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    fixed_alpha: bool,
    executed: Optional[Callable] = None,
) -> Tuple[ASACState, AuxiliaryLogs]:
    """One update on a replay batch: the termination penalty, then the
    critic, actor, temperature, ``theta`` and target steps."""
    tau = agent_config.tau
    agent_state, sample_key = agent_state.next_keys()
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
        agent_config.tau_theta,
    )
    rewards = get_episode_termination_penalized_rewards(
        episode_termination_penalty, transition.reward, transition.terminated
    )
    agent_state = agent_state.replace(
        episode_termination_penalty=episode_termination_penalty
    )
    agent_state, aux_value, value_metrics = update_value_functions(
        agent_state,
        transition,
        rewards,
        agent_config.reward_scale,
        extension_stack,
        total_timesteps,
        carries,
        executed,
    )
    # The temperature and theta read the actor's own sample, drawn before its
    # step, as the authors' code does (darer/ASAC.py:146-213).
    agent_state, aux_policy, log_probs, policy_metrics = update_policy(
        agent_state,
        transition.obs,
        transition.raw_obs,
        extension_stack,
        total_timesteps,
        carries,
        a_expert=transition.a_expert,
        actions=transition.action,
        executed=executed,
    )
    # The dual-gradient step towards ``target_entropy`` (target entropy
    # per dimension times the action dimension).
    # ``fixed_alpha`` drops the step but keeps its logged auxiliaries, as
    # SAC's does.
    tempered, aux_temperature = core.update_temperature(
        agent_state, log_probs, jnp.asarray(agent_config.target_entropy)
    )
    if not fixed_alpha:
        agent_state = tempered
    # In the target's units: the reward rate theta is subtracted from the
    # scaled rewards there. It averages the rewards without the penalty, as
    # the authors' code (darer/ASAC.py:191).
    agent_state = update_theta(
        agent_state,
        agent_config.tau_theta,
        transition.reward * agent_config.reward_scale,
        log_probs,
        transition.action.shape[-1],
    )
    agent_state = update_target_networks(agent_state, tau)
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=aux_value,
        theta=ThetaAuxiliaries(
            theta=agent_state.theta,
            episode_termination_penalty=episode_termination_penalty,
        ),
        extensions=merge_metrics(value_metrics, policy_metrics),
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
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    expert_policy: Optional[Callable] = None,
    offline_data: Optional[TransitionDataset] = None,
    extensions: Sequence = (),
    fixed_alpha: bool = True,
):
    """ASAC's train function: one step per env, then (from
    ``learning_starts``) one update, per iteration."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "expert_policy": expert_policy,
        "buffer": buffer,
    }
    extensions = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options("ASAC", expert_policy=expert_policy)
    if recurrent and extensions:
        raise NotImplementedError("Recurrent ASAC does not support extensions yet.")
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

    def init(key: jax.Array) -> ASACState:
        return init_ASAC(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            buffer,
            stored_state=agent_config.stored_state,
            store_expert_data=holds_expert(expert_policy, loop.extensions),
            expert_policy=expert_policy,
            store_raw_next_obs="raw_next_obs" in loop.extensions.buffer_fields(),
        )

    def update(agent_state: ASACState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: ASAC samples it from there.
        return update_agent(
            agent_state,
            buffer,
            recurrent,
            agent_config,
            loop.extensions,
            total_timesteps,
            fixed_alpha,
            executed_fn(env_args),
        )

    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={"expert_policy": expert_policy},
    )
