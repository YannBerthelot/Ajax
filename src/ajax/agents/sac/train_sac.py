import os
from collections.abc import Sequence
from dataclasses import fields
from math import floor
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.sac.state import SACConfig, SACState
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    init_collector_state,
    should_use_uniform_sampling,
)
from ajax.environments.utils import check_env_is_gymnax, get_state_action_shapes
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.networks.networks import (
    get_adam_tx,
    get_initialized_actor_critic,
    predict_value,
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

PROFILER_PATH = "./tensorboard"


def get_alpha_from_params(params: FrozenDict) -> float:
    return jnp.exp(params["log_alpha"])


@struct.dataclass
class TemperatureAuxiliaries:
    alpha_loss: jax.Array
    alpha: jax.Array
    log_alpha: jax.Array


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
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    """
    Initialize the train state for the temperature parameter (alpha).

    Args:
        learning_rate (float): Learning rate for alpha optimizer.
        alpha_init (float): Initial value for alpha.

    Returns:
        TrainState: Initialized train state for alpha.
    """
    log_alpha = jnp.log(alpha_init)
    params = FrozenDict({"log_alpha": log_alpha})
    tx = get_adam_tx(learning_rate)
    return TrainState.create(
        apply_fn=get_alpha_from_params,  # Optional
        params=params,
        tx=tx,
    )


def init_sac(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    window_size: int = 10,
) -> SACState:
    """
    Initialize the SAC agent's state, including actor, critic, alpha, and collector states.

    Args:
        key (jax.Array): Random number generator key.
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        buffer (BufferType): Replay buffer.

    Returns:
        SACState: Initialized SAC agent state.
    """
    (
        rng,
        init_key,
        collector_key,
    ) = jax.random.split(key, num=3)

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
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
    )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    return SACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
    )


@partial(jax.jit, static_argnames=["recurrent", "gamma", "reward_scale"])
def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    actor_state: LoadedTrainState,
    actions: jax.Array,
    observations: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    recurrent: bool,
    reward_scale: float = 5.0,  # Add reward scaling factor here
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """
    Compute the value loss for the critic networks.

    Args:
        critic_params (FrozenDict): Parameters of the critic networks.
        critic_states (LoadedTrainState): Critic train states.
        rng (jax.Array): Random number generator key.
        actor_state (LoadedTrainState): Actor train state.
        actions (jax.Array): Actions taken.
        observations (jax.Array): Current observations.
        next_observations (jax.Array): Next observations.
        dones (jax.Array): Done flags.
        rewards (jax.Array): Rewards received.
        gamma (float): Discount factor.
        alpha (jax.Array): Temperature parameter.
        recurrent (bool): Whether the model is recurrent.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[jax.Array, Dict[str, jax.Array]]: Loss and auxiliary metrics.
    """
    # Apply the reward scaling here
    rewards = rewards * reward_scale

    # Sample next actions from policy π(a|s_{t+1})

    next_pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=next_observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=sample_key)

    # Predict Q-values from critics
    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_params,
        x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
    )
    # Target Q-values using target networks
    assert (
        critic_states.target_params is not None
    ), "Target parameters are not set in critic states."
    q_targets = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.target_params,
        x=jnp.concatenate((next_observations, next_actions), axis=-1),
    )

    # Unpack and unsqueeze if needed
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    q1_target, q2_target = jnp.split(q_targets, 2, axis=0)

    # Bellman target and losses
    min_q_target = jnp.minimum(q1_target, q2_target).squeeze(0)
    log_probs = log_probs.sum(-1, keepdims=True)

    target_q = jax.lax.stop_gradient(
        rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs),
    )

    assert target_q.shape == q_preds.shape[1:], f"{target_q.shape} != {q_preds.shape}"
    assert min_q_target.shape == log_probs.shape

    loss_q1 = 0.5 * jnp.mean((q1_pred.squeeze(0) - target_q) ** 2)
    loss_q2 = 0.5 * jnp.mean((q2_pred.squeeze(0) - target_q) ** 2)
    total_loss = loss_q1 + loss_q2
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q1_pred=q1_pred.mean().flatten(),
        q2_pred=q2_pred.mean().flatten(),
        target_q=target_q.mean().flatten(),
        log_probs=log_probs.mean().flatten(),
    )


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    alpha: jax.Array,
    rng: jax.random.PRNGKey,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """
    Compute the policy loss for the actor network.

    Args:
        actor_params (FrozenDict): Parameters of the actor network.
        actor_state (LoadedTrainState): Actor train state.
        critic_states (LoadedTrainState): Critic train states.
        observations (jax.Array): Current observations.
        dones (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.
        alpha (jax.Array): Temperature parameter.
        rng (jax.random.PRNGKey): Random number generator key.

    Returns:
        Tuple[jax.Array, Dict[str, jax.Array]]: Loss and auxiliary metrics.
    """
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)

    # Predict Q-values from critics
    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.params,
        x=jnp.hstack((observations, actions)),
    )

    # Unpack and unsqueeze if needed
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    q_min = jnp.minimum(q1_pred, q2_pred).squeeze(0)

    log_probs = log_probs.sum(-1, keepdims=True)

    assert log_probs.shape == q_min.shape, f"{log_probs.shape} != {q_min.shape}"
    loss = (alpha * log_probs - q_min).mean()

    return loss, PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_min=q_min.mean()
    )


@partial(
    jax.jit,
    static_argnames=["target_entropy"],
)
def temperature_loss_function(
    log_alpha_params: FrozenDict,
    corrected_log_probs: jax.Array,
    target_entropy: float,
) -> Tuple[jax.Array, TemperatureAuxiliaries]:
    """
    Compute the loss for the temperature parameter (alpha).

    Args:
        log_alpha_params (FrozenDict): Logarithm of alpha parameters.
        corrected_log_probs (jax.Array): Log probabilities of actions.
        target_entropy (float): Target entropy value.

    Returns:
        Tuple[jax.Array, Dict[str, Any]]: Loss and auxiliary metrics.
    """
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    loss = (
        -1.0
        * (
            log_alpha * jax.lax.stop_gradient(corrected_log_probs + target_entropy)
        ).mean()
    )

    return loss, TemperatureAuxiliaries(
        alpha_loss=loss, alpha=alpha, log_alpha=log_alpha
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma", "reward_scale"],
)
def update_value_functions(
    agent_state: SACState,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float = 1.0,  # Add reward scaling factor here
) -> Tuple[SACState, Dict[str, Any]]:
    """
    Update the critic networks using the value loss.

    Args:
        agent_state (SACState): Current SAC agent state.
        observations (jax.Array): Current observations.
        actions (jax.Array): Actions taken.
        next_observations (jax.Array): Next observations.
        dones (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.
        rewards (jax.Array): Rewards received.
        gamma (float): Discount factor.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[SACState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    value_loss_key, rng = jax.random.split(agent_state.rng)
    value_and_grad_fn = jax.value_and_grad(value_loss_function, has_aux=True)
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    (loss, aux), grads = value_and_grad_fn(
        agent_state.critic_state.params,
        agent_state.critic_state,
        value_loss_key,
        agent_state.actor_state,
        actions,
        observations,
        next_observations,
        dones,
        rewards,
        gamma,
        alpha,
        recurrent,
        reward_scale,
    )

    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        rng=rng,
        critic_state=updated_critic_state,
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
) -> Tuple[SACState, Dict[str, Any]]:
    """
    Update the actor network using the policy loss.

    Args:
        agent_state (SACState): Current SAC agent state.
        observations (jax.Array): Current observations.
        done (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.

    Returns:
        Tuple[SACState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    rng, policy_key = jax.random.split(agent_state.rng)
    value_and_grad_fn = jax.value_and_grad(policy_loss_function, has_aux=True)
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    (loss, aux), grads = value_and_grad_fn(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        done,
        recurrent,
        alpha,
        policy_key,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        rng=rng,
        actor_state=updated_actor_state,
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=["target_entropy", "recurrent"],
)
def update_temperature(
    agent_state: SACState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    target_entropy: float,
    recurrent: bool,
) -> Tuple[SACState, Dict[str, Any]]:
    """
    Update the temperature parameter (alpha) using the alpha loss.

    Args:
        agent_state (SACState): Current SAC agent state.
        observations (jax.Array): Current observations.
        dones (Optional[jax.Array]): Done flags.
        target_entropy (float): Target entropy value.
        recurrent (bool): Whether the model is recurrent.

    Returns:
        Tuple[SACState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    loss_fn = jax.value_and_grad(temperature_loss_function, has_aux=True)

    pi, _ = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=observations,
        done=dones,
        recurrent=recurrent,
    )
    rng, sample_key = jax.random.split(agent_state.rng)
    _, log_probs = pi.sample_and_log_prob(seed=sample_key)

    (loss, aux), grads = loss_fn(
        agent_state.alpha.params,
        log_probs.sum(-1),
        target_entropy,
    )

    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        rng=rng,
        alpha=new_alpha_state,
    )
    return agent_state, jax.lax.stop_gradient(aux)


@partial(
    jax.jit,
    static_argnames=["tau"],
)
def update_target_networks(
    agent_state: SACState,
    tau: float,
) -> SACState:
    """
    Perform a soft update of the target networks.

    Args:
        agent_state (SACState): Current SAC agent state.
        tau (float): Soft update coefficient.

    Returns:
        SACState: Updated agent state.
    """
    new_critic_state = agent_state.critic_state.soft_update(tau=tau)
    return agent_state.replace(
        critic_state=new_critic_state,
    )


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "buffer",
        "gamma",
        "tau",
        "action_dim",
        "num_critic_updates",
        "reward_scale",
        "target_update_frequency",
        "transition_mix_fraction",
    ],
)
def update_agent(
    agent_state: SACState,
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    tau: float,
    num_critic_updates: int = 1,
    target_update_frequency: int = 1,
    reward_scale: float = 5.0,
    additional_transition: Optional[Any] = None,
    transition_mix_fraction: float = 1.0,  # part of original buffer sample to keep TODO : add control over this hyperparameter
) -> Tuple[SACState, AuxiliaryLogs]:
    """
    Update the SAC agent, including critic, actor, and temperature updates.

    Args:
        agent_state (SACState): Current SAC agent state.
        _ (Any): Placeholder for scan compatibility.
        buffer (BufferType): Replay buffer.
        recurrent (bool): Whether the model is recurrent.
        gamma (float): Discount factor.
        action_dim (int): Action dimensionality.
        tau (float): Soft update coefficient.
        num_critic_updates (int): Number of critic updates per step.
        target_update_frequency (int): Frequency of target network updates.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[SACState, None]: Updated agent state.
    """
    # Sample buffer

    sample_key, rng = jax.random.split(agent_state.rng)
    if buffer is not None and agent_state.collector_state.buffer_state is not None:
        observations, terminated, truncated, next_observations, rewards, actions = (
            get_batch_from_buffer(
                buffer,
                agent_state.collector_state.buffer_state,
                sample_key,
            )
        )
        original_transition = Transition(
            observations, actions, rewards, terminated, truncated, next_observations
        )

        if additional_transition is not None and transition_mix_fraction < 1.0:
            assert transition_mix_fraction >= 0 and transition_mix_fraction <= 1.0, (
                "transition_mix_fraction should be between 0 and 1, got",
                transition_mix_fraction,
            )
            len_original = len(observations)
            n_samples_from_original = floor(transition_mix_fraction * len_original)
            n_samples_from_transition = len_original - n_samples_from_original
            print(
                f"samples from buffer:{n_samples_from_original} online"
                f" samples:{n_samples_from_transition}"
            )
            additional_transition = jax.tree.map(
                lambda x: jax.random.choice(
                    sample_key, x, shape=(n_samples_from_transition,)
                ),
                additional_transition,
            )

            transition = jax.tree.map(
                lambda x, y: (
                    None
                    if (x is None or y is None)
                    else jnp.concatenate([x[:n_samples_from_original], y], axis=0)
                ),
                original_transition,
                additional_transition,
                is_leaf=lambda x: x is None,
            )
        else:
            transition = original_transition
    else:
        # If no buffer is provided, use the collector state to get the latest transition
        transition = additional_transition
        terminated = additional_transition.terminated
        truncated = additional_transition.truncated

    agent_state = agent_state.replace(rng=rng)
    dones = jnp.logical_or(transition.terminated, transition.truncated)

    # Update Q functions
    def critic_update_step(carry, _):
        agent_state = carry
        agent_state, aux_value = update_value_functions(
            observations=transition.obs,
            actions=transition.action,
            next_observations=transition.next_obs,
            rewards=transition.reward,
            dones=dones,
            agent_state=agent_state,
            recurrent=recurrent,
            gamma=gamma,
            reward_scale=reward_scale,
        )

        return agent_state, aux_value

    agent_state, aux_value = jax.lax.scan(
        critic_update_step,
        agent_state,
        None,
        length=num_critic_updates,
    )

    # Update policy
    agent_state, aux_policy = update_policy(
        observations=transition.obs,
        done=dones,
        agent_state=agent_state,
        recurrent=recurrent,
    )

    # Adjust temperature
    target_entropy = -action_dim
    agent_state, aux_temperature = update_temperature(
        agent_state,
        observations=transition.obs,
        target_entropy=target_entropy,
        recurrent=recurrent,
        dones=dones,
    )

    # Update target networks
    # TODO : Only update every update_target_network steps
    agent_state = update_target_networks(agent_state, tau=tau)
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=ValueAuxiliaries(
            **{key: val.flatten() for key, val in to_state_dict(aux_value).items()}
        ),
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=[
        "env_args",
        "mode",
        "recurrent",
        "buffer",
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "action_dim",
        "lstm_hidden_size",
        "agent_config",
        "horizon",
        "total_timesteps",
        "n_epochs",
        "transition_mix_fraction",
    ],
)
def training_iteration(
    agent_state: SACState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_config: SACConfig,
    action_dim: int,
    total_timesteps: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    horizon: int = 10000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    n_epochs: int = 1,
    transition_mix_fraction: float = 1.0,
) -> tuple[SACState, None]:
    """
    Perform one training iteration, including experience collection and agent updates.

    Args:
        agent_state (SACState): Current SAC agent state.
        _ (Any): Placeholder for scan compatibility.
        env_args (EnvironmentConfig): Environment configuration.
        mode (str): Environment mode ("gymnax" or "brax").
        recurrent (bool): Whether the model is recurrent.
        buffer (BufferType): Replay buffer.
        agent_config (SACConfig): SAC agent configuration.
        action_dim (int): Action dimensionality.
        lstm_hidden_size (Optional[int]): LSTM hidden size for recurrent models.
        log_frequency (int): Frequency of logging and evaluation.
        num_episode_test (int): Number of episodes for evaluation.

    Returns:
        Tuple[SACState, None]: Updated agent state.
    """
    # collector_state = agent_state.collector_state

    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
    )
    agent_state, transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=1
    )
    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        update_scan_fn = partial(
            update_agent,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            action_dim=action_dim,
            tau=agent_config.tau,
            reward_scale=agent_config.reward_scale,
            additional_transition=(
                jax.tree.map(lambda x: x.squeeze(0), transition)
                if transition_mix_fraction < 1.0
                else None
            ),
            transition_mix_fraction=transition_mix_fraction,
        )
        agent_state, aux = jax.lax.scan(
            update_scan_fn, agent_state, xs=None, length=n_epochs
        )
        aux = jax.tree.map(
            lambda x: x[-1].reshape((1,)), aux
        )  # keep only the final state across epochs
        aux = aux.replace(
            value=ValueAuxiliaries(
                **{key: val.flatten() for key, val in to_state_dict(aux.value).items()}
            )
        )
        return agent_state, aux

    def fill_with_nan(dataclass):
        """
        Recursively fills all fields of a dataclass with jnp.nan.
        """
        nan = jnp.ones(1) * jnp.nan
        dict = {}
        for field in fields(dataclass):
            sub_dataclass = field.type
            if hasattr(
                sub_dataclass, "__dataclass_fields__"
            ):  # Check if the field is another dataclass
                dict[field.name] = fill_with_nan(sub_dataclass)
            else:
                dict[field.name] = nan
        return dataclass(**dict)

    def skip_update(agent_state):
        return agent_state, fill_with_nan(AuxiliaryLogs)

    agent_state, aux = jax.lax.cond(
        timestep >= agent_config.learning_starts,
        do_update,
        skip_update,
        operand=agent_state,
    )

    agent_state, metrics_to_log = evaluate_and_log(
        agent_state,
        aux,
        index,
        mode,
        env_args,
        num_episode_test,
        recurrent,
        lstm_hidden_size,
        log,
        verbose,
        log_fn,
        log_frequency,
        total_timesteps,
    )
    return agent_state, metrics_to_log


def profile_memory(timestep):
    jax.profiler.save_device_memory_profile(f"memory{timestep}.prof")


def safe_get_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve an environment variable.

    Args:
        var_name (str): The name of the environment variable.
        default (Optional[str]): Default value if the variable is not set.

    Returns:
        Optional[str]: The value of the environment variable or default.
    """
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: SACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    sweep: bool = False,
):
    """
    Create the training function for the SAC agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        buffer (BufferType): Replay buffer.
        agent_config (SACConfig): SAC agent configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        total_timesteps (int): Total timesteps for training.
        num_episode_test (int): Number of episodes for evaluation during training.

    Returns:
        Callable: JIT-compiled training function.
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None and not sweep
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Start async logging if logging is enabled
    if logging_config is not None and not sweep:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        """Train the SAC agent."""
        agent_state = init_sac(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
        )

        num_updates = total_timesteps // env_args.n_envs
        _, action_shape = get_state_action_shapes(env_args.env)

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_config=agent_config,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
        )

        agent_state, eval_rewards = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )

        # Stop async logging if it was started
        # if logging_config is not None:
        #     stop_async_logging()
        window_size = int(0.1 * total_timesteps)

        return agent_state, jnp.nanmean(
            eval_rewards["Eval/episodic mean reward"][-window_size:]
        )

    return train
