import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training import train_state
from jax.tree_util import Partial as partial

from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.train_PPO import init_PPO, update_value_functions
from ajax.agents.PPO.utils import _compute_gae, get_minibatches_from_batch
from ajax.agents.sac.utils import SquashedNormal
from ajax.environments.interaction import (
    collect_experience,
    collect_experience_from_expert_policy,
    get_pi,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
)
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.networks.networks import (
    predict_value,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)

DEBUG = False


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: float
    log_probs: float
    old_log_probs: float
    clip_fraction: float
    entropy: float
    imitation_coef: float


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: float
    predictions: float
    targets: float


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "advantage_normalization",
        "expert_policy",
        "distance_to_stable",
        "imitation_coef_offset",
    ],
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
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.01,
    distance_to_stable: Optional[Callable] = None,
    imitation_coef_offset: float = 1e-3,
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

    ratio = jnp.exp(
        new_log_probs - log_probs
    )  # log_probs are per-action-dim, so we sum them to get the total log prob

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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

    imitation_loss = (
        -pi.log_prob(expert_policy(raw_observations))
        if expert_policy is not None
        else jnp.zeros(1)
    )

    EPS = 1e-6
    if distance_to_stable is not None:
        distance = (
            (1 / (distance_to_stable(observations) + EPS)) + imitation_coef_offset
        )  # small offset to prevent it going too low while avoiding max (which is conditional on the actual value) for performance
        distance = jnp.expand_dims(distance, -1)
    else:
        distance = 1
    total_loss = (
        loss_actor
        - ent_coef * entropy
        + (imitation_coef * distance * imitation_loss).mean()
    )

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
        imitation_coef=imitation_coef,
    )


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "advantage_normalization",
        "expert_policy",
        "distance_to_stable",
        "imitation_coef_offset",
    ],
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
    raw_observations: jax.Array,
    expert_policy: Callable,
    imitation_coef: float,
    distance_to_stable: Callable,
    imitation_coef_offset: float,
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
    pi, _ = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=observations,
        done=done,
        recurrent=recurrent,
    )

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
        raw_observations=raw_observations,
        expert_policy=expert_policy,
        imitation_coef=imitation_coef,
        distance_to_stable=distance_to_stable,
        imitation_coef_offset=imitation_coef_offset,
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
        "expert_policy",
        "distance_to_stable",
        "imitation_coef_offset",
    ],
)
def update_agent(
    agent_state: PPOState,
    _: Any,
    shuffled_batch: tuple[jax.Array],
    agent_config: PPOConfig,
    recurrent: bool,
    expert_policy: Callable,
    imitation_coef: float,
    distance_to_stable: Callable,
    imitation_coef_offset: float,
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
        raw_observations,
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
        "expert_policy",
        "imitation_coef",
        "distance_to_stable",
        "imitation_coef_offset",
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
    num_episode_test: int = 100,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 1e-3,
    distance_to_stable: Optional[Callable] = None,
    imitation_coef_offset: float = 1e-3,
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
        transition.raw_obs,
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

    timestep = agent_state.collector_state.timestep
    imitation_coef = (
        imitation_coef(timestep) if callable(imitation_coef) else imitation_coef
    )

    def do_update(
        agent_state: PPOState, num_epochs: int
    ) -> tuple[PPOState, AuxiliaryLogs]:
        update_scan_fn = partial(
            update_agent,
            shuffled_batch=shuffled_batch,
            recurrent=recurrent,
            agent_config=agent_config,
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


def batchify(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Reshape x into (num_batches, batch_size, ...) padding last batch if needed."""
    n = x.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    pad = n_batches * batch_size - n
    if pad > 0:
        x = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
    return x.reshape(n_batches, batch_size, *x.shape[1:])


@partial(
    jax.jit,
    static_argnames=[
        "gamma",
        "actor_lr",
        "actor_epochs",
        "actor_batch_size",
        "critic_lr",
        "critic_epochs",
        "critic_batch_size",
    ],
)
def pre_train(
    rng: jax.Array,
    actor_state: train_state.TrainState,
    critic_state: train_state.TrainState,
    dataset: Sequence,  # Sequence[Transition]
    gamma: float = 0.99,
    # Actor hyperparameters
    actor_lr: float = 1e-3,
    actor_epochs: int = 10,
    actor_batch_size: int = 64,
    # Critic hyperparameters
    critic_lr: float = 1e-3,
    critic_epochs: int = 10,
    critic_batch_size: int = 64,
) -> Tuple[train_state.TrainState, train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Pre-train actor (behavioral cloning) and critic (TD(0)) from a dataset of transitions.
    Returns trained states and metrics dict with per-epoch actor/critic losses.
    """

    # Flatten dataset
    # obs = jnp.concatenate([t.obs for t in dataset], axis=0)
    # actions = jnp.concatenate([t.action for t in dataset], axis=0)
    # rewards = jnp.concatenate([t.reward for t in dataset], axis=0)
    # terminated = jnp.concatenate([t.terminated for t in dataset], axis=0)
    # next_obs = jnp.concatenate([t.next_obs for t in dataset], axis=0)

    obs = dataset.obs
    actions = dataset.action
    rewards = dataset.reward
    terminated = dataset.terminated
    next_obs = dataset.next_obs

    metrics = {
        "actor_loss": jnp.zeros((actor_epochs,)),
        "critic_loss": jnp.zeros((critic_epochs,)),
    }

    # --------------------------
    # Actor pre-training
    # --------------------------
    bc_actor_state = train_state.TrainState.create(
        apply_fn=actor_state.apply_fn,
        params=actor_state.params,
        tx=optax.adam(actor_lr),
    )

    def actor_loss_fn(params, batch_obs, batch_actions):
        pred_actions = bc_actor_state.apply_fn(
            params, batch_obs
        ).mean()  # deterministic prediction
        return jnp.mean((pred_actions - batch_actions) ** 2)

    def actor_train_step(state, batch_obs, batch_actions):
        loss, grads = jax.value_and_grad(actor_loss_fn)(
            state.params, batch_obs, batch_actions
        )
        return state.apply_gradients(grads=grads), loss

    def actor_epoch_step(carry, rng_epoch):
        state = carry
        perm = jax.random.permutation(rng_epoch, obs.shape[0])
        obs_shuffled = obs[perm]
        actions_shuffled = actions[perm]

        obs_batches = batchify(obs_shuffled, actor_batch_size)
        act_batches = batchify(actions_shuffled, actor_batch_size)

        def batch_step(carry, batch):
            state = carry
            b_obs, b_act = batch
            new_state, loss = actor_train_step(state, b_obs, b_act)
            return new_state, loss

        state, batch_losses = jax.lax.scan(
            batch_step, state, (obs_batches, act_batches)
        )
        return state, jnp.mean(batch_losses)

    rng, rng_actor = jax.random.split(rng)
    rng_epochs = jax.random.split(rng_actor, actor_epochs)
    bc_actor_state, actor_losses = jax.lax.scan(
        actor_epoch_step, bc_actor_state, rng_epochs
    )
    metrics["actor_loss"] = actor_losses

    # --------------------------
    # Critic pre-training
    # --------------------------
    bc_critic_state = train_state.TrainState.create(
        apply_fn=critic_state.apply_fn,
        params=critic_state.params,
        tx=optax.adam(critic_lr),
    )

    def critic_loss_fn(
        params, batch_obs, batch_rewards, batch_next_obs, batch_terminated
    ):
        v_pred = bc_critic_state.apply_fn(params, batch_obs).squeeze(-1)
        v_next = bc_critic_state.apply_fn(params, batch_next_obs).squeeze(-1)
        td_target = batch_rewards.squeeze(-1) + gamma * v_next * (
            1.0 - batch_terminated.squeeze(-1)
        )
        return jnp.mean((v_pred - td_target) ** 2)

    def critic_train_step(
        state, batch_obs, batch_rewards, batch_next_obs, batch_terminated
    ):
        loss, grads = jax.value_and_grad(critic_loss_fn)(
            state.params, batch_obs, batch_rewards, batch_next_obs, batch_terminated
        )
        return state.apply_gradients(grads=grads), loss

    def critic_epoch_step(carry, rng_epoch):
        state = carry
        perm = jax.random.permutation(rng_epoch, obs.shape[0])
        obs_shuffled = obs[perm]
        rewards_shuffled = rewards[perm]
        next_obs_shuffled = next_obs[perm]
        terminated_shuffled = terminated[perm]

        obs_batches = batchify(obs_shuffled, critic_batch_size)
        rew_batches = batchify(rewards_shuffled, critic_batch_size)
        next_obs_batches = batchify(next_obs_shuffled, critic_batch_size)
        term_batches = batchify(terminated_shuffled, critic_batch_size)

        def batch_step(carry, batch):
            state = carry
            b_obs, b_rew, b_next_obs, b_term = batch
            new_state, loss = critic_train_step(state, b_obs, b_rew, b_next_obs, b_term)
            return new_state, loss

        state, batch_losses = jax.lax.scan(
            batch_step,
            state,
            (obs_batches, rew_batches, next_obs_batches, term_batches),
        )
        return state, jnp.mean(batch_losses)

    rng, rng_critic = jax.random.split(rng)
    rng_epochs = jax.random.split(rng_critic, critic_epochs)
    bc_critic_state, critic_losses = jax.lax.scan(
        critic_epoch_step, bc_critic_state, rng_epochs
    )
    metrics["critic_loss"] = critic_losses

    # Update original states with trained parameters
    actor_state = actor_state.replace(params=bc_actor_state.params)
    critic_state = critic_state.replace(params=bc_critic_state.params)

    return actor_state, critic_state, metrics


@struct.dataclass
class CloningConfig:
    actor_epochs: int = 10
    critic_epochs: int = 10
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    actor_batch_size: int = 64
    critic_batch_size: int = 64
    pre_train_n_steps: int = int(1e5)
    imitation_coef: float = 1e-3
    distance_to_stable: Optional[Callable] = None
    imitation_coef_offset: float = 1e-3


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    cloning_args: CloningConfig,
    expert_policy: Callable,
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

    imitation_coef = cloning_args.imitation_coef

    def imitation_coef_schedule(init_val):
        def imitation_coef(t, total_timesteps):
            return (1 - (t / total_timesteps)) * init_val

        return imitation_coef

    # if "auto" not in str(imitation_coef):
    if "lin" in str(imitation_coef):
        imitation_coef = (
            imitation_coef_schedule(float(imitation_coef.split("_")[1]))
            if isinstance(imitation_coef, str)
            else imitation_coef
        )
        imitation_coef = (
            partial(imitation_coef, total_timesteps=total_timesteps)
            if callable(imitation_coef)
            else imitation_coef
        )
    if "auto" in str(imitation_coef):
        imitation_coef = float(imitation_coef.split("_")[1])

    def train(key, index: Optional[int] = None):
        """Train the PPO agent."""
        init_key, expert_key = jax.random.split(key)
        agent_state = init_PPO(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
        )

        # pre-train agent
        if expert_policy is not None and cloning_args.pre_train_n_steps > 0:
            # dataset is examples of observations and actions taken
            dataset = collect_experience_from_expert_policy(
                expert_policy,
                rng=expert_key,
                env_args=env_args,
                mode=mode,
                n_timesteps=cloning_args.pre_train_n_steps,
            )
            jax.clear_caches()
            actor_state, critic_state, metrics = pre_train(
                rng=expert_key,
                actor_state=agent_state.actor_state,
                critic_state=agent_state.critic_state,
                dataset=dataset,
                gamma=agent_config.gamma,
                actor_lr=actor_optimizer_args.learning_rate,
                critic_lr=critic_optimizer_args.learning_rate,
                actor_epochs=cloning_args.actor_epochs,
                critic_epochs=cloning_args.critic_epochs,
                actor_batch_size=cloning_args.actor_batch_size,
                critic_batch_size=cloning_args.critic_batch_size,
            )
            jax.debug.print("{metrics}", metrics=metrics)
            agent_state = agent_state.replace(
                actor_state=actor_state, critic_state=critic_state
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
            distance_to_stable=cloning_args.distance_to_stable,
            imitation_coef_offset=cloning_args.imitation_coef_offset,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )

        return agent_state, out

    return train
