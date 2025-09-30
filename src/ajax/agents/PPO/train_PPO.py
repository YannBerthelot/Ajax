import os
from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.utils import _compute_gae, get_minibatches_from_batch
from ajax.agents.sac.utils import SquashedNormal
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
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

DEBUG = False


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

    return PPOState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
    )


# @partial(jax.jit, static_argnames=["recurrent"])
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
    # jax.debug.print(
    #     "v_preds:{v_preds}, value_targets:{value_targets}",
    #     v_preds=v_preds.mean(),
    #     value_targets=value_targets.mean(),
    # )

    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


# @partial(
#     jax.jit,
#     static_argnames=["recurrent", "advantage_normalization"],
# )
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

    # Need to deal with various shapes depending on brax vs gymnax and discrete vs continuous

    if isinstance(pi, distrax.Categorical):
        new_log_probs = jnp.expand_dims(
            pi.log_prob(actions.squeeze(-1)), -1
        )  # .sum(-1, keepdims=True)
    else:
        new_log_probs = pi.log_prob(actions).sum(-1, keepdims=True)
    if DEBUG:
        assert new_log_probs.shape == log_probs.shape, (
            f"Shape mismatch between new_log_probs {new_log_probs.shape} and log_probs"
            f" {log_probs.shape}"
        )

    ratio = jnp.exp(new_log_probs - log_probs)

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    if DEBUG:
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

    entropy = (
        pi.unsquashed_entropy().mean()
        if isinstance(pi, SquashedNormal)
        else pi.entropy().mean()
    )

    total_loss = loss_actor - ent_coef * entropy

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
    )


VALUE_AND_GRAD_FN = jax.value_and_grad(value_loss_function, has_aux=True)
POLICY_AND_GRAD_FN = jax.value_and_grad(policy_loss_function, has_aux=True)


# @partial(
#     jax.jit,
#     static_argnames=["recurrent"],
# )
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

    (loss, aux), grads = VALUE_AND_GRAD_FN(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        value_targets,
        dones,
        recurrent,
    )
    # jax.debug.print("Critic loss: {loss_val}", loss_val=loss)
    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    agent_state = agent_state.replace(
        critic_state=updated_critic_state,
    )
    return agent_state, aux


def check_no_nan(x, id):
    assert not jnp.isnan(x).any(), f"NaN detected {id}"


# @partial(
#     jax.jit,
#     static_argnames=["recurrent", "advantage_normalization"],
# )
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

    if DEBUG:
        jax.debug.callback(check_no_nan, log_probs, 1)
        jax.debug.callback(check_no_nan, actions, 2)
        jax.debug.callback(check_no_nan, observations, 3)
        jax.debug.callback(check_no_nan, gae, 4)
    (loss, aux), grads = POLICY_AND_GRAD_FN(
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

    if DEBUG:
        jax.debug.callback(check_no_nan, loss, 5)

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

    (
        observations,
        actions,
        terminated,
        truncated,
        value_targets,
        gae,
        log_probs,
    ) = shuffled_batch
    if DEBUG:
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
    clip_coef = (
        agent_config.clip_range(agent_state.collector_state.timestep)
        if callable(agent_config.clip_range)
        else agent_config.clip_range
    )

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
    next_values = predict_value(
        critic_state=agent_state.critic_state,
        critic_params=agent_state.critic_state.params,
        x=transition.next_obs,
    ).squeeze(0)

    gae, value_targets = _compute_gae(
        values=values,
        next_values=next_values,
        rewards=transition.reward,
        terminateds=transition.terminated,
        truncateds=transition.truncated,
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
    if DEBUG:
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

    def do_update(
        agent_state: PPOState, num_epochs: int
    ) -> tuple[PPOState, AuxiliaryLogs]:
        def body_fn(agent_state, _):
            agent_state, aux = update_agent(
                agent_state,
                None,
                shuffled_batch=shuffled_batch,
                recurrent=recurrent,
                agent_config=agent_config,
            )
            return agent_state, aux  # overwrite aux each time

        agent_state, aux = jax.lax.scan(
            f=body_fn, init=agent_state, xs=None, length=num_epochs
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
    )

    # jax.clear_caches()
    # gc.collect()
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

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )

        return agent_state, out

    return train
