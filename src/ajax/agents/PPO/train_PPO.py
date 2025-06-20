import os
from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.utils import _compute_gae, get_minibatches_from_batch
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    init_collector_state,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
from ajax.evaluate import evaluate
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

PROFILER_PATH = "./tensorboard"


def get_alpha_from_params(params: FrozenDict) -> float:
    return jnp.exp(params["log_alpha"])


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: float
    log_probs: float
    old_log_probs: float
    clip_fraction: float
    entropy: float


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


def init_PPO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    window_size: int = 10,
) -> PPOState:
    """
    Initialize the PPO agent's state, including actor, critic, alpha, and collector states.

    Args:
        key (jax.Array): Random number generator key.
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        buffer (BufferType): Replay buffer.

    Returns:
        PPOState: Initialized PPO agent state.
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
        continuous=check_if_environment_has_continuous_actions(
            env_args.env, env_params=env_args.env_params
        ),
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

    return PPOState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
    )


@partial(jax.jit, static_argnames=["recurrent"])
def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    dones: jax.Array,
    recurrent: bool,
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

    loss = 0.5 * jnp.mean((v_preds - value_targets) ** 2)  # classic MSE

    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "clip_coef", "ent_coef", "advantage_normalization"],
)
def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
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
    new_log_probs = (
        jnp.expand_dims(pi.log_prob(actions.squeeze()), axis=-1)
        if jnp.ndim(actions.squeeze()) < 3
        else pi.log_prob(actions).sum(-1, keepdims=True)
    )
    assert new_log_probs.shape == log_probs.shape, (
        f"Shape mismatch between new_log_probs {new_log_probs.shape} and log_probs"
        f" {log_probs.shape}"
    )
    # .sum(
    #     -1, keepdims=True
    # )  # sum as the log probs are per-action-dim, squeeze to handle discrete actions without trailing dimension
    ratio = jnp.exp(
        new_log_probs - log_probs
    )  # log_probs are per-action-dim, so we sum them to get the total log prob
    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

    assert (
        ratio.shape[0] == gae.shape[0]
    ), f"Mismatch between ratio shape ({ratio.shape}) and gae shape ({gae.shape})"
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_coef,
            1.0 + clip_coef,
        )
        * gae
    )

    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

    # CALCULATE AUXILIARIES
    clip_fraction = (jnp.abs(ratio - 1) > clip_coef).mean()
    entropy = pi.entropy().mean()

    total_loss = loss_actor - ent_coef * entropy

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
    )


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def update_value_functions(
    agent_state: PPOState,
    observations: jax.Array,
    value_targets: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
) -> Tuple[PPOState, Dict[str, Any]]:
    """
    Update the critic networks using the value loss.

    Args:
        agent_state (PPOState): Current PPO agent state.
        observations (jax.Array): Current observations.
        actions (jax.Array): Actions taken.
        next_observations (jax.Array): Next observations.
        dones (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.
        rewards (jax.Array): Rewards received.
        gamma (float): Discount factor.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[PPOState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """
    value_and_grad_fn = jax.value_and_grad(value_loss_function, has_aux=True)

    (loss, aux), grads = value_and_grad_fn(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        value_targets,
        dones,
        recurrent,
    )
    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        critic_state=updated_critic_state,
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=["recurrent", "clip_coef", "ent_coef", "advantage_normalization"],
)
def update_policy(
    agent_state: PPOState,
    observations: jax.Array,
    actions: jax.Array,
    gae: jax.Array,
    log_probs: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
) -> Tuple[PPOState, Dict[str, Any]]:
    """
    Update the actor network using the policy loss.

    Args:
        agent_state (PPOState): Current PPO agent state.
        observations (jax.Array): Current observations.
        done (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.

    Returns:
        Tuple[PPOState, Dict[str, Any]]: Updated agent state and auxiliary metrics.
    """

    value_and_grad_fn = jax.value_and_grad(policy_loss_function, has_aux=True)
    (loss, aux), grads = value_and_grad_fn(
        agent_state.actor_state.params,
        agent_state.actor_state,
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        gae=gae,
        dones=done,
        recurrent=recurrent,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        advantage_normalization=advantage_normalization,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        actor_state=updated_actor_state,
    )
    return agent_state, aux


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "agent_config",
    ],
)
def update_agent(
    agent_state: PPOState,
    _: Any,
    shuffled_batch: tuple[jax.Array],
    agent_config: PPOConfig,
    recurrent: bool,
) -> Tuple[PPOState, AuxiliaryLogs]:
    """
    Update the PPO agent, including critic, actor, and temperature updates.

    Args:
        agent_state (PPOState): Current PPO agent state.
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
        Tuple[PPOState, None]: Updated agent state.
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
    )

    aux = AuxiliaryLogs(
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


def no_op(agent_state, *args):
    return None


def no_op_none(agent_state, index, timestep):
    pass


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
    ],
)
def training_iteration(
    agent_state: PPOState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    agent_config: PPOConfig,
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
) -> tuple[PPOState, None]:
    """
    Perform one training iteration, including experience collection and agent updates.

    Args:
        agent_state (PPOState): Current PPO agent state.
        _ (Any): Placeholder for scan compatibility.
        env_args (EnvironmentConfig): Environment configuration.
        mode (str): Environment mode ("gymnax" or "brax").
        recurrent (bool): Whether the model is recurrent.
        buffer (BufferType): Replay buffer.
        agent_config (PPOConfig): PPO agent configuration.
        action_dim (int): Action dimensionality.
        lstm_hidden_size (Optional[int]): LSTM hidden size for recurrent models.
        log_frequency (int): Frequency of logging and evaluation.
        num_episode_test (int): Number of episodes for evaluation.

    Returns:
        Tuple[PPOState, None]: Updated agent state.
    """
    # collector_state = agent_state.collector_state

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
    )
    agent_state, transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=n_steps
    )  # transition = s_t, a_t, r_{s_t -> s_{t+1}}, s_{t+1}, d_{s_t -> s_{t+1}}

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

    gae, value_targets = _compute_gae(
        values=values,
        last_value=last_value,
        rewards=transition.reward,
        dones=dones,
        gamma=agent_config.gamma,
        gae_lambda=agent_config.gae_lambda,
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

    def do_update(
        agent_state: PPOState, num_epochs: int
    ) -> tuple[PPOState, AuxiliaryLogs]:
        update_scan_fn = partial(
            update_agent,
            shuffled_batch=shuffled_batch,
            recurrent=recurrent,
            agent_config=agent_config,
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

    def run_and_log(agent_state: PPOState, aux: AuxiliaryLogs, index: int):
        eval_key = agent_state.eval_rng
        normalization = (
            "normalization_info" in agent_state.collector_state.env_state.info
            if mode == "brax"
            else "normalization_info" in dir(agent_state.collector_state.env_state)
        )
        env_norm_info = (
            (
                agent_state.collector_state.env_state.info["normalization_info"]
                if mode == "brax"
                else agent_state.collector_state.env_state.normalization_info
            )
            if normalization
            else None
        )
        eval_rewards, eval_entropy = evaluate(
            env_args.env,
            actor_state=agent_state.actor_state,
            num_episodes=num_episode_test,
            rng=eval_key,
            env_params=env_args.env_params,
            recurrent=recurrent,
            lstm_hidden_size=lstm_hidden_size,
            norm_info=env_norm_info,
        )
        mean_return = agent_state.collector_state.episodic_mean_return
        if env_norm_info is not None:
            if env_norm_info.reward is not None:
                mean_return = env_args.env.unnormalize_reward(
                    agent_state.collector_state.episodic_mean_return,
                    env_norm_info.reward,
                )
        metrics_to_log = {
            "timestep": timestep,
            "Eval/episodic mean reward": eval_rewards,
            "Eval/episodic entropy": eval_entropy,
            "Train/episodic mean reward": mean_return,
        }
        metrics_to_log.update(flatten_dict(to_state_dict(aux)))
        jax.debug.callback(log_fn, metrics_to_log, index)

        if verbose:
            jax.debug.print(
                (
                    "[Eval] Step={timestep_val}, Reward={rewards_val},"
                    " Entropy={entropy_val}"
                ),
                timestep_val=timestep,
                rewards_val=eval_rewards,
                entropy_val=eval_entropy,
            )

    if log:
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
        jax.lax.cond(flag, run_and_log, no_op, agent_state, aux, index)
        del aux

    jax.clear_caches()
    # gc.collect()
    return agent_state, None


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
    agent_config: PPOConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
):
    """
    Create the training function for the PPO agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        buffer (BufferType): Replay buffer.
        agent_config (PPOConfig): PPO agent configuration.
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
        """Train the PPO agent."""
        agent_state = init_PPO(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
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
