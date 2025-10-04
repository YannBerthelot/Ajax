from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.APO.state import APOConfig, APOState
from ajax.agents.APO.utils import _compute_gae
from ajax.agents.cloning import (
    CloningConfig,
    get_cloning_args,
    get_pre_trained_agent,
)
from ajax.agents.PPO.train_PPO_pre_train import PolicyAuxiliaries, update_policy
from ajax.agents.PPO.utils import get_minibatches_from_batch
from ajax.environments.interaction import (
    collect_experience,
    init_collector_state,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
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
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.utils import get_one

PROFILER_PATH = "./tensorboard"

DEBUG = False


def get_alpha_from_params(params: FrozenDict) -> float:
    return jnp.exp(params["log_alpha"])


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: float
    predictions: float
    targets: float


@struct.dataclass
class AuxiliaryLogs:
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


def init_APO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    window_size: int = 10,
) -> APOState:
    """
    Initialize the APO agent's state, including actor, critic, alpha, and collector states.

    Args:
        key (jax.Array): Random number generator key.
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        buffer (BufferType): Replay buffer.

    Returns:
        APOState: Initialized APO agent state.
    """
    (
        rng,
        init_key,
        collector_key,
    ) = jax.random.split(key, num=3)

    continuous = check_if_environment_has_continuous_actions(
        env_args.env, env_params=env_args.env_params
    )
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=continuous,
        action_value=False,
        squash=False,
        num_critics=1,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        window_size=window_size,
    )

    return APOState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
        average_reward=0.0,
        b=0.0,
    )


@partial(jax.jit, static_argnames=["recurrent", "nu"])
def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    dones: jax.Array,
    recurrent: bool,
    nu: float,
    b: float,
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

    # Predict V-values from critics
    v_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_params,
        x=observations,
    ).squeeze(
        0
    )  # squeeze to stay consistent with ensemble_critic that adds a leading dimension even for a single critic.

    loss = 0.5 * jnp.mean(((v_preds - nu * b) - value_targets) ** 2)  # classic MSE

    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "nu"],
)
def update_value_functions(
    agent_state: APOState,
    observations: jax.Array,
    value_targets: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    nu: float,
    b: float,
) -> Tuple[APOState, Dict[str, Any]]:
    """
    Update the critic networks using the value loss.

    Args:
        agent_state (APOState): Current APO agent state.
        observations (jax.Array): Current observations.
        actions (jax.Array): Actions taken.
        next_observations (jax.Array): Next observations.
        dones (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.
        rewards (jax.Array): Rewards received.
        gamma (float): Discount factor.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[APOState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    value_and_grad_fn = jax.value_and_grad(value_loss_function, has_aux=True)

    (loss, aux), grads = value_and_grad_fn(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        value_targets,
        dones,
        recurrent,
        nu,
        b,
    )
    # jax.debug.print("Critic loss: {loss_val}", loss_val=loss)
    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        critic_state=updated_critic_state,
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "agent_config",
        "expert_policy",
        "imitation_coef",
        "distance_to_stable",
        "imitation_coef_offset",
    ],
)
def update_agent(
    agent_state: APOState,
    _: Any,
    shuffled_batch: tuple[jax.Array],
    agent_config: APOConfig,
    recurrent: bool,
    b: float,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = get_one,
    imitation_coef_offset: float = 0.0,
) -> Tuple[APOState, AuxiliaryLogs]:
    """
    Update the APO agent, including critic, actor, and temperature updates.

    Args:
        agent_state (APOState): Current APO agent state.
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
        Tuple[APOState, None]: Updated agent state.
    """
    # Sample buffer

    (
        observations,
        actions,
        terminated,
        truncated,
        value_targets,
        gae,
        log_probs,
        raw_observations,
    ) = shuffled_batch

    assert (
        observations.shape[:-1] == actions.shape[:-1]
    ), (  # FIXME : investigate the shape mismatch due to shuffling in batch and shapes shenanigans
        f"Shape mismatch between observations {observations.shape} and actions"
        f" {actions.shape}"
    )

    dones = jnp.logical_or(terminated, truncated)

    # Update critic/V-function
    agent_state, aux_value = update_value_functions(
        agent_state=agent_state,
        observations=observations,
        value_targets=value_targets,
        dones=dones,
        recurrent=recurrent,
        nu=agent_config.nu,
        b=b,
    )

    # Update policy
    if callable(agent_config.clip_range):
        clip_coef = agent_config.clip_range(agent_state.collector_state.timestep)
    else:
        clip_coef = agent_config.clip_range

    agent_state, aux_policy = update_policy(
        agent_state=agent_state,
        observations=observations,
        actions=actions,
        gae=gae,
        log_probs=log_probs,
        done=dones,
        recurrent=recurrent,
        ent_coef=agent_config.ent_coef,
        clip_coef=clip_coef,
        advantage_normalization=agent_config.normalize_advantage,
        raw_observations=raw_observations,
        expert_policy=expert_policy,
        imitation_coef=imitation_coef,
        distance_to_stable=distance_to_stable,
        imitation_coef_offset=imitation_coef_offset,
    )

    aux = AuxiliaryLogs(
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
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "lstm_hidden_size",
        "agent_config",
        "horizon",
        "total_timesteps",
        "n_steps",
        "expert_policy",
        "imitation_coef",
        "distance_to_stable",
        "imitation_coef_offset",
    ],
)
def training_iteration(
    agent_state: APOState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    agent_config: APOConfig,
    total_timesteps: int,
    n_steps: int,
    total_n_updates: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    horizon: int = 10000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 1e-3,
    distance_to_stable: Callable = get_one,
    imitation_coef_offset: float = 1e-3,
) -> tuple[APOState, None]:
    """
    Perform one training iteration, including experience collection and agent updates.

    Args:
        agent_state (APOState): Current APO agent state.
        _ (Any): Placeholder for scan compatibility.
        env_args (EnvironmentConfig): Environment configuration.
        mode (str): Environment mode ("gymnax" or "brax").
        recurrent (bool): Whether the model is recurrent.
        buffer (BufferType): Replay buffer.
        agent_config (APOConfig): APO agent configuration.
        action_dim (int): Action dimensionality.
        lstm_hidden_size (Optional[int]): LSTM hidden size for recurrent models.
        log_frequency (int): Frequency of logging and evaluation.
        num_episode_test (int): Number of episodes for evaluation.

    Returns:
        Tuple[APOState, None]: Updated agent state.
    """

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
    )
    agent_state, transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=n_steps
    )

    values = predict_value(
        critic_state=agent_state.critic_state,
        critic_params=agent_state.critic_state.params,
        x=transition.obs,
    ).squeeze(0)
    last_value = (
        predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=transition.next_obs[-1:],
        )
        .squeeze(0)
        .squeeze(0)  # don't need the first dimension for a single transition
    )
    # dones = jnp.vstack([first_done.reshape(1, -1, 1), transition.terminated[:-1]])
    dones = transition.terminated

    average_reward = (
        1 - agent_config.alpha
    ) * agent_state.average_reward + agent_config.alpha * jnp.mean(transition.reward)

    b = (1 - agent_config.alpha) * agent_state.b + agent_config.alpha * jnp.mean(values)
    agent_state = agent_state.replace(average_reward=average_reward, b=b)

    gae, value_targets = _compute_gae(
        values=values,
        last_value=last_value,
        rewards=transition.reward,
        dones=dones,
        gae_lambda=agent_config.gae_lambda,
        average_reward=average_reward,
    )

    batch = (
        transition.obs,
        (
            jnp.expand_dims(transition.action, axis=-1)
            if jnp.ndim(transition.action)
            < 3  # discrete case without trailing dimension
            else transition.action
        ),
        transition.terminated,
        transition.truncated,
        value_targets,
        gae,
        (
            jnp.expand_dims(transition.log_prob, axis=-1)
            if jnp.ndim(transition.log_prob)
            < 3  # discrete case without trailing dimension
            else transition.log_prob.sum(-1, keepdims=True)
        ),
        transition.raw_obs,
    )

    shuffle_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    assert (
        max(agent_config.batch_size, agent_config.n_steps)
        % min(agent_config.batch_size, agent_config.n_steps)
        == 0
    ), (
        "can't evenly break n_steps into batch size chunks,"
        f" n_steps={agent_config.n_steps} batch_size={agent_config.batch_size}"
    )
    num_minibatches = max(agent_config.batch_size, agent_config.n_steps) // min(
        agent_config.batch_size, agent_config.n_steps
    )
    shuffled_batch = get_minibatches_from_batch(
        batch, rng=shuffle_key, num_minibatches=num_minibatches
    )

    timestep = agent_state.collector_state.timestep
    imitation_coef = (
        imitation_coef(timestep) if callable(imitation_coef) else imitation_coef
    )

    def do_update(
        agent_state: APOState, num_epochs: int
    ) -> tuple[APOState, AuxiliaryLogs]:
        update_scan_fn = partial(
            update_agent,
            shuffled_batch=shuffled_batch,
            recurrent=recurrent,
            agent_config=agent_config,
            b=b,
            expert_policy=expert_policy,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset,
        )
        agent_state, aux = jax.lax.scan(
            update_scan_fn, agent_state, xs=None, length=num_epochs
        )
        aux = aux.replace(
            value=ValueAuxiliaries(
                **{key: val.flatten() for key, val in to_state_dict(aux.value).items()}
            )
        )
        aux = jax.tree_util.tree_map(
            lambda x: x.mean(), aux
        )  # need to aggregate over the n-epochs
        return (
            agent_state.replace(n_updates=agent_state.n_updates + 1),
            aux,
        )  # aux should be the one from the last epoch

    agent_state, aux = do_update(agent_state, num_epochs=agent_config.n_epochs)

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
        avg_reward_mode=True,
    )

    jax.clear_caches()
    # gc.collect()
    return agent_state, metrics_to_log


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: APOConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
):
    """
    Create the training function for the APO agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        buffer (BufferType): Replay buffer.
        agent_config (APOConfig): APO agent configuration.
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
        """Train the APO agent."""
        init_key, expert_key = jax.random.split(key)
        agent_state = init_APO(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
        )

        # pre-train agent

        (
            imitation_coef,
            imitation_coef_offset,
            distance_to_stable,
            pre_train_n_steps,
        ) = get_cloning_args(cloning_args, total_timesteps)
        if pre_train_n_steps > 0:
            agent_state = get_pre_trained_agent(
                agent_state,
                expert_policy,
                expert_key,
                env_args,
                cloning_args,
                mode,
                agent_config,
                actor_optimizer_args,
                critic_optimizer_args,
            )
        num_updates = (total_timesteps // (env_args.n_envs * agent_config.n_steps)) + 1

        training_iteration_scan_fn = partial(
            training_iteration,
            recurrent=network_args.lstm_hidden_size is not None,
            agent_config=agent_config,
            n_steps=agent_config.n_steps,
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
            total_n_updates=num_updates,
            expert_policy=expert_policy,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )

        return agent_state, out

    return train
