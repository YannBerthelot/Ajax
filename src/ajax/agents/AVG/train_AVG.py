import os
from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.AVG.state import AVGConfig, AVGState, NormalizationInfo
from ajax.agents.AVG.utils import compute_td_error_scaling
from ajax.environments.interaction import (
    Transition,
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
)

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
    q_pred: jax.Array
    target_q: jax.Array
    log_probs: jax.Array
    scaling_coef: jax.Array


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


# @partial(
#     jax.jit,
#     static_argnames=[
#         "num_critics",
#         "window_size",
#         "alpha_args",
#         "network_args",
#         "actor_optimizer_args",
#         "critic_optimizer_args",
#         "env_args",
#     ],
# )
def init_AVG(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    num_critics: int = 1,
    window_size: int = 10,
) -> AVGState:
    """
    Initialize the SAC agent's state, including actor, critic, alpha, and collector states.

    Args:
        key (jax.Array): Random number generator key.
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        alpha_args (AlphaConfig): Alpha configuration.

    Returns:
        AVGState: Initialized SAC agent state.
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
        num_critics=num_critics,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"

    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode=mode, window_size=window_size
    )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    init_val = jnp.zeros(
        (env_args.n_envs, 1)
    )  # a single dim as we are normalizing scalars (gamma, reward, G_return) (1 scalar for all envs)

    init_norm_info = NormalizationInfo(
        value=init_val,
        count=jnp.zeros((1,)),
        mean=jnp.zeros((1, 1)),
        mean_2=jnp.zeros((1, 1)),
    )

    return AVGState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
        reward=init_norm_info,
        gamma=init_norm_info,
        G_return=init_norm_info,
        scaling_coef=jnp.ones((1, 1)),
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
    scaling_coef: jax.Array,
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
    # Predict Q-values from critics
    q_pred = jnp.min(
        predict_value(
            critic_state=critic_states,
            critic_params=critic_params,
            x=jnp.concatenate((observations, actions), axis=-1),
        ),
        axis=0,
    )

    next_pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=next_observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    next_actions, next_log_probs = next_pi.sample_and_log_prob(seed=sample_key)

    # Target Q-values using target networks
    q_target = jnp.min(
        predict_value(
            critic_state=critic_states,
            critic_params=critic_params,
            x=jnp.concatenate((next_observations, next_actions), axis=-1),
        ),
        axis=0,
    )

    # Bellman target and losses
    next_log_probs = next_log_probs.sum(-1, keepdims=True)

    target_q = jax.lax.stop_gradient(
        rewards + (1.0 - dones) * gamma * (q_target - alpha * next_log_probs),
    )

    assert target_q.shape == q_pred.shape, f"{target_q.shape} != {q_pred.shape}"
    assert q_target.shape == next_log_probs.shape

    delta = q_pred - target_q
    scaled_delta = delta / scaling_coef
    assert scaled_delta.shape == delta.shape, f"{scaled_delta.shape} != {delta.shape}"
    total_loss = jnp.mean(scaled_delta**2)
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q_pred=q_pred.mean().flatten(),
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
        scaling_coef=scaling_coef.mean().flatten(),
    )


EPS = 1e-12


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
    actions, log_probs = pi.sample_and_log_prob(seed=rng)
    # Predict Q-values from critics
    q_pred = jnp.min(
        predict_value(
            critic_state=critic_states,
            critic_params=critic_states.params,
            x=jnp.hstack((observations, actions)),
        ),
        axis=0,
    )

    # q_pred: shape(n_envs,1) , squeeze is to remove the leading dimension of ensemble-critic

    log_probs = log_probs.sum(
        -1, keepdims=True
    )  # Shape (B,n_actions) to shape (n_envs,1)

    assert log_probs.shape == q_pred.shape, f"{log_probs.shape} != {q_pred.shape}"
    loss = (alpha * log_probs - q_pred).mean()

    return loss, PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_min=q_pred.mean()
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
    agent_state: AVGState,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float = 1.0,  # Add reward scaling factor here
) -> Tuple[AVGState, Dict[str, Any]]:
    """
    Update the critic networks using the value loss.

    Args:
        agent_state (AVGState): Current SAC agent state.
        observations (jax.Array): Current observations.
        actions (jax.Array): Actions taken.
        next_observations (jax.Array): Next observations.
        dones (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.
        rewards (jax.Array): Rewards received.
        gamma (float): Discount factor.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[AVGState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    value_loss_key, _ = jax.random.split(agent_state.rng, 2)
    value_and_grad_fn = jax.value_and_grad(value_loss_function, has_aux=True)
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    # Call the value loss function with reward scaling applied
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
        agent_state.scaling_coef,
        reward_scale,  # Pass reward scaling factor here
    )
    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        # rng=rng, need to keep the same RNG to keep the same action sampled, \
        # next actions are still used with a different seed than action
        critic_state=updated_critic_state,
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def update_policy(
    agent_state: AVGState,
    actions: jax.Array,
    log_probs: jax.Array,
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
) -> Tuple[AVGState, Dict[str, Any]]:
    """
    Update the actor network using the policy loss.

    Args:
        agent_state (AVGState): Current SAC agent state.
        observations (jax.Array): Current observations.
        done (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.

    Returns:
        Tuple[AVGState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    rng, policy_key, _ = jax.random.split(agent_state.rng, 3)

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
    agent_state: AVGState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    target_entropy: float,
    recurrent: bool,
) -> Tuple[AVGState, Dict[str, Any]]:
    """
    Update the temperature parameter (alpha) using the alpha loss.

    Args:
        agent_state (AVGState): Current SAC agent state.
        observations (jax.Array): Current observations.
        dones (Optional[jax.Array]): Done flags.
        target_entropy (float): Target entropy value.
        recurrent (bool): Whether the model is recurrent.

    Returns:
        Tuple[AVGState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
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
    static_argnames=[
        "recurrent",
        "gamma",
        "action_dim",
        "reward_scale",
        "num_critic_updates",
    ],
)
def update_agent(
    agent_state: AVGState,
    _: Any,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    num_critic_updates: int = 1,
    reward_scale: float = 5.0,
) -> Tuple[AVGState, AuxiliaryLogs]:
    """
    Update the SAC agent, including critic, actor, and temperature updates.

    Args:
        agent_state (AVGState): Current SAC agent state.
        _ (Any): Placeholder for scan compatibility.
        recurrent (bool): Whether the model is recurrent.
        gamma (float): Discount factor.
        action_dim (int): Action dimensionality.
        tau (float): Soft update coefficient.
        num_critic_updates (int): Number of critic updates per step.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[AVGState, None]: Updated agent state.
    """
    transition = agent_state.collector_state.rollout
    done = jnp.logical_or(transition.terminated, transition.truncated)  # type: ignore[union-attr]

    # Update Q functions
    def critic_update_step(carry, _):
        agent_state = carry

        agent_state, aux_value = update_value_functions(
            observations=transition.obs,
            actions=transition.action,
            next_observations=transition.next_obs,
            dones=done,
            agent_state=agent_state,
            recurrent=recurrent,
            rewards=transition.reward,
            gamma=gamma,
            reward_scale=reward_scale,
        )

        return agent_state, aux_value

    agent_state_critic, aux_value = jax.lax.scan(
        critic_update_step,
        agent_state,
        None,
        length=num_critic_updates,
    )

    # Update policy
    agent_state_policy, aux_policy = update_policy(
        agent_state=agent_state,
        actions=transition.action,  # type: ignore[union-attr]
        log_probs=transition.log_prob,  # type: ignore[union-attr]
        observations=transition.obs,  # type: ignore[union-attr]
        done=done,
        recurrent=recurrent,
    )
    collector_state = agent_state.collector_state.replace(
        num_update=agent_state.collector_state.num_update + 1
    )
    agent_state = agent_state.replace(
        actor_state=agent_state_policy.actor_state,
        critic_state=agent_state_critic.critic_state,
        rng=agent_state_policy.rng,
        collector_state=collector_state,
    )

    aux = AuxiliaryLogs(
        temperature=TemperatureAuxiliaries(
            jnp.nan,
            jnp.exp(agent_state.alpha.params["log_alpha"]),
            agent_state.alpha.params["log_alpha"],
        ),
        policy=aux_policy,
        value=ValueAuxiliaries(
            **{key: val.flatten() for key, val in to_state_dict(aux_value).items()}
        ),
    )
    return agent_state, aux


def flatten_dict(dict: Dict) -> Dict:
    return_dict = {}
    for key, val in dict.items():
        if isinstance(val, Dict):
            for subkey, subval in val.items():
                return_dict[f"{key}/{subkey}"] = subval
        else:
            return_dict[key] = val
    return return_dict


def prepare_metrics(aux):
    log_metrics = flatten_dict(to_state_dict(aux))
    return {key: val for (key, val) in log_metrics.items() if not (jnp.isnan(val))}


def no_op(x, *args):
    return x


def no_op_none(*args, **kwargs):
    return None


def squeeze_dim_0(x):
    return x.squeeze(0)


def get_nan(x):
    return jnp.nan * x


def update_AVG_values(
    agent_state: AVGState, rollout: Transition, agent_config: AVGConfig
) -> AVGState:
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    r_ent = rollout.reward - alpha * jax.lax.stop_gradient(rollout.log_prob).sum(
        -1, keepdims=True
    )

    reward = agent_state.reward.replace(value=r_ent)

    gamma = agent_state.gamma.replace(
        value=agent_config.gamma * (1 - rollout.terminated)
    )

    new_G = agent_state.G_return.value + r_ent

    temp_G_value = jax.vmap(jax.lax.cond, in_axes=(0, None, None, 0))(
        rollout.terminated.astype("int8").squeeze(-1), no_op, get_nan, new_G
    )  # set G_return.value to nan if not terminal, for compute_td_error_scaling

    scaling_coef, reward, gamma, G_return = compute_td_error_scaling(
        reward, gamma, G_return=agent_state.G_return.replace(value=temp_G_value)
    )

    G_value = jax.vmap(jax.lax.cond, in_axes=(0, None, None, 0))(
        rollout.terminated.astype("int8").squeeze(-1), jnp.zeros_like, no_op, new_G
    )  # either revert to new_G (from nan) to keep on accumulating, or reset to 0
    G_return = G_return.replace(value=G_value)
    agent_state = agent_state.replace(
        reward=reward, gamma=gamma, G_return=G_return, scaling_coef=scaling_coef
    )

    return agent_state


@partial(
    jax.jit, static_argnames=["run_and_log", "no_op_none", "log_frequency", "env_args"]
)
def log_function(
    agent_state,
    log_frequency,
    timestep,
    total_timesteps,
    aux,
    run_and_log,
    no_op_none,
    env_args,
    index,
):
    _, eval_rng = jax.random.split(agent_state.eval_rng)
    agent_state = agent_state.replace(eval_rng=eval_rng)
    log_flag = timestep - (agent_state.n_logs * log_frequency) >= log_frequency

    agent_state = agent_state.replace(
        n_logs=jax.lax.select(log_flag, agent_state.n_logs + 1, agent_state.n_logs)
    )
    flag = jnp.logical_or(
        jnp.logical_and(log_flag, timestep > 1),
        timestep >= (total_timesteps - env_args.n_envs),
    )

    jax.lax.cond(flag, run_and_log, no_op_none, agent_state, aux, index)
    del aux
    return agent_state


@partial(
    jax.jit,
    static_argnames=[
        "env_args",
        "mode",
        "recurrent",
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "action_dim",
        "lstm_hidden_size",
        "agent_config",
        "total_timesteps",
    ],
)
def training_iteration(
    agent_state: AVGState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    agent_config: AVGConfig,
    action_dim: int,
    total_timesteps: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
) -> tuple[AVGState, None]:
    """
    Perform one training iteration, including experience collection and agent updates.

    Args:
        agent_state (AVGState): Current SAC agent state.
        _ (Any): Placeholder for scan compatibility.
        env_args (EnvironmentConfig): Environment configuration.
        mode (str): Environment mode ("gymnax" or "brax").
        recurrent (bool): Whether the model is recurrent.
        agent_config (SACConfig): SAC agent configuration.
        action_dim (int): Action dimensionality.
        lstm_hidden_size (Optional[int]): LSTM hidden size for recurrent models.
        log_frequency (int): Frequency of logging and evaluation.
        num_episode_test (int): Number of episodes for evaluation.

    Returns:
        Tuple[AVGState, None]: Updated agent state.
    """

    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        uniform=uniform,
    )
    rng = agent_state.rng
    agent_state, rollout = jax.lax.scan(collect_scan_fn, agent_state, xs=None, length=1)

    rollout = jax.tree.map(
        squeeze_dim_0, rollout
    )  # Remove first dim as we only have one transition
    # jax.debug.print("before {x}", x=timestep)
    collector_state = agent_state.collector_state.replace(rollout=rollout)
    agent_state = agent_state.replace(collector_state=collector_state)
    agent_state = update_AVG_values(agent_state, rollout, agent_config)
    timestep = agent_state.collector_state.timestep
    # jax.debug.print("after {x}", x=timestep)
    agent_state = agent_state.replace(rng=rng)

    def do_update(agent_state):
        update_scan_fn = partial(
            update_agent,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            action_dim=action_dim,
            reward_scale=agent_config.reward_scale,
        )
        agent_state, aux = jax.lax.scan(update_scan_fn, agent_state, xs=None, length=1)
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

    jax.clear_caches()
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
    agent_config: AVGConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
):
    """
    Create the training function for the SAC agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        agent_config (SACConfig): SAC agent configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        total_timesteps (int): Total timesteps for training.
        num_episode_test (int): Number of episodes for evaluation during training.

    Returns:
        Callable: JIT-compiled training function.
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Start async logging if logging is enabled
    if logging_config is not None:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        """Train the SAC agent."""

        agent_state = init_AVG(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            num_critics=agent_config.num_critics,
        )

        num_updates = total_timesteps  # // env_args.n_envs
        _, action_shape = get_state_action_shapes(env_args.env)

        training_iteration_scan_fn = partial(
            training_iteration,
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
        )

        agent_state, _ = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )

        # Stop async logging if it was started
        # if logging_config is not None:
        #     stop_async_logging()

        return agent_state

    return train
