"""DQN (Mnih et al., 2015): Deep Q-Network for discrete action spaces.

Value-based, off-policy. A single Q-network ``Q(s) -> R^{n_actions}`` is
trained to minimise the temporal-difference error against a slowly
refreshed target network:

    y    = r + gamma * (1 - done) * max_a' Q_target(s', a')
    loss = MSE(Q(s, a), y)

Differences from the SAC/TD3 template in this codebase:
  * No actor. The greedy policy is ``argmax_a Q(s, a)``; the Q-network is
    stored in ``DQNState.actor_state`` so the shared evaluation loop
    (hard-wired to ``agent_state.actor_state``) works unchanged.
  * Exploration is epsilon-greedy, applied at collection time via a
    custom ``action_pipeline`` -- the vanilla collector path samples a
    continuous action in [-1, 1] and cannot be reused for discrete envs.
  * The target network is refreshed every ``target_update_interval``
    gradient steps via ``soft_update(tau)``. The classic hard update is
    ``tau=1.0``; a Polyak update is ``tau<1`` with interval ``1`` --
    one mechanism, no boolean flag.
"""

from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.DQN.networks import get_initialized_q_network, predict_q
from ajax.agents.DQN.state import DQNConfig, DQNState
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    init_collector_state,
    should_use_uniform_sampling,
)
from ajax.environments.utils import check_env_is_gymnax, get_action_dim
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.perf_utils import final_aux_scan, train_jit
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import BufferType

# ---------------------------------------------------------------------------
# Auxiliary dataclass (for logging)
# ---------------------------------------------------------------------------


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_mean: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    # Nested one level deep on purpose: ajax.log.flatten_dict only
    # collapses (1,)-shaped leaves to scalars for nested aux dataclasses,
    # which is what ajax.log._make_no_op fills the skipped branch with.
    value: ValueAuxiliaries


def _fill_with_nan(dataclass):
    """Build an all-NaN instance of a (possibly nested) aux dataclass."""
    nan = jnp.ones(1) * jnp.nan
    d = {}
    for field in fields(dataclass):
        sub = field.type
        if hasattr(sub, "__dataclass_fields__"):
            d[field.name] = _fill_with_nan(sub)
        else:
            d[field.name] = nan
    return dataclass(**d)


# ---------------------------------------------------------------------------
# Epsilon-greedy action pipeline
# ---------------------------------------------------------------------------


class DQNActionPipelineResult(NamedTuple):
    """Matches the structure ``collect_experience`` expects from a pipeline."""

    env_action: jax.Array
    policy_action: jax.Array
    log_probs: jax.Array
    is_expert_flag: jax.Array
    in_value_box: jax.Array
    entry_bonus: jax.Array
    rng: jax.Array
    buffer_action: Optional[jax.Array] = None
    new_expert_state: Optional[Any] = None


def make_epsilon_greedy_pipeline(
    env_args: EnvironmentConfig,
    n_actions: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_frac: float,
    total_timesteps: int,
) -> Callable:
    """Build the default DQN ``action_pipeline``: linear-decay epsilon-greedy.

    Epsilon decays linearly from ``epsilon_start`` to ``epsilon_end`` over
    the first ``epsilon_decay_frac`` of training, then stays flat. During
    the ``learning_starts`` warmup (``uniform`` is set) actions are fully
    random regardless of epsilon.
    """

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        del raw_obs, mix_key
        obs = agent_state.collector_state.last_obs
        q_values = predict_q(
            agent_state.actor_state, agent_state.actor_state.params, obs
        )
        greedy_action = jnp.argmax(q_values, axis=-1).astype(jnp.int32)

        # Linear epsilon schedule on the training fraction.
        timestep = agent_state.collector_state.timestep
        train_frac = timestep / total_timesteps
        decay = jnp.clip(train_frac / epsilon_decay_frac, 0.0, 1.0)
        epsilon = epsilon_start + decay * (epsilon_end - epsilon_start)

        explore_key, random_key = jax.random.split(action_key)
        random_action = jax.random.randint(
            random_key, shape=greedy_action.shape, minval=0, maxval=n_actions
        ).astype(jnp.int32)
        explore = jax.random.uniform(explore_key, shape=greedy_action.shape) < epsilon
        eps_greedy_action = jnp.where(explore, random_action, greedy_action)

        # Warmup: ignore the policy entirely and act uniformly at random.
        env_action = jax.lax.cond(
            uniform, lambda: random_action, lambda: eps_greedy_action
        )
        # The env expects (n_envs,) integer actions; the buffer schema
        # stores discrete actions as (n_envs, 1).
        buffer_action = env_action[:, None]
        n_envs = env_args.n_envs
        return DQNActionPipelineResult(
            env_action=env_action,
            policy_action=buffer_action,
            log_probs=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            is_expert_flag=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            in_value_box=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            entry_bonus=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            rng=rng,
            buffer_action=buffer_action,
        )

    return pipeline


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_DQN(
    key: jax.Array,
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    n_actions: int,
    window_size: int = 10,
    q_network_cls: Optional[type] = None,
) -> DQNState:
    rng, init_key, collector_key = jax.random.split(key, num=3)

    q_state = get_initialized_q_network(
        key=init_key,
        env_config=env_args,
        optimizer_config=optimizer_args,
        network_config=network_args,
        n_actions=n_actions,
        q_network_cls=q_network_cls,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
    )
    # The single Q-network lives in actor_state (so the shared eval loop
    # works); critic_state mirrors it at init and is never updated.
    return DQNState(
        rng=rng,
        eval_rng=rng,
        actor_state=q_state,
        critic_state=q_state,
        collector_state=collector_state,
    )


# ---------------------------------------------------------------------------
# TD target + Q loss
# ---------------------------------------------------------------------------


def compute_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
) -> jax.Array:
    """Vanilla DQN target: y = r + gamma * (1 - d) * max_a' Q_target(s', a')."""
    rewards = rewards * reward_scale
    q_next = predict_q(q_state, q_state.target_params, next_observations)
    max_q_next = jnp.max(q_next, axis=-1, keepdims=True)
    target = rewards + gamma * (1.0 - dones) * max_q_next
    return jax.lax.stop_gradient(target)


def compute_double_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
) -> jax.Array:
    """Double DQN target (van Hasselt et al., 2016).

    Decouples action *selection* (online network) from action
    *evaluation* (target network), removing the maximisation bias of the
    vanilla ``max_a' Q_target`` target:

        a*  = argmax_a' Q_online(s', a')
        y   = r + gamma * (1 - d) * Q_target(s', a*)

    Drop-in for ``compute_dqn_td_target``; select via
    ``DQN(..., td_target_fn=compute_double_dqn_td_target)``.
    """
    rewards = rewards * reward_scale
    q_next_online = predict_q(q_state, q_state.params, next_observations)
    next_actions = jnp.argmax(q_next_online, axis=-1, keepdims=True)
    q_next_target = predict_q(q_state, q_state.target_params, next_observations)
    q_next = jnp.take_along_axis(q_next_target, next_actions, axis=-1)
    target = rewards + gamma * (1.0 - dones) * q_next
    return jax.lax.stop_gradient(target)


def mse_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
    """Mean-squared TD error -- the DQN default."""
    return jnp.mean((q_taken - target_q) ** 2)


def make_huber_td_loss(delta: float = 1.0) -> Callable:
    """Build a Huber TD-loss callable.

    Huber loss is quadratic within ``delta`` of zero error and linear
    beyond it -- the smooth analogue of the original DQN's TD-error
    clipping, less sensitive to outlier targets. Select via
    ``DQN(..., td_loss_fn=make_huber_td_loss(1.0))``.
    """

    def huber_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
        return jnp.mean(optax.huber_loss(q_taken, target_q, delta=delta))

    return huber_td_loss


def q_loss_fn(
    q_params,
    q_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    loss_fn: Callable,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """TD loss between Q(s, a) at the taken action and a pre-computed target.

    ``loss_fn(q_taken, target_q) -> scalar`` selects the error norm
    (``mse_td_loss`` by default, ``make_huber_td_loss(...)`` for Huber).
    """
    q_all = predict_q(q_state, q_params, observations)
    # actions is (batch, 1) int32 -> gather one Q-value per row.
    q_taken = jnp.take_along_axis(q_all, actions, axis=-1)
    loss = loss_fn(q_taken, target_q)
    aux = ValueAuxiliaries(
        critic_loss=loss.flatten(),
        target_q=target_q.mean().flatten(),
        q_pred_mean=q_taken.mean().flatten(),
    )
    return loss, aux


# ---------------------------------------------------------------------------
# Per-iteration agent update (one Q step + periodic target refresh)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "buffer",
        "gamma",
        "tau",
        "target_update_interval",
        "reward_scale",
        "td_target_fn",
        "td_loss_fn",
    ],
)
def update_agent(
    agent_state: DQNState,
    _: Any,
    buffer: BufferType,
    gamma: float,
    tau: float,
    target_update_interval: int,
    reward_scale: float,
    td_target_fn: Callable,
    td_loss_fn: Callable,
) -> Tuple[DQNState, AuxiliaryLogs]:
    sample_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    (
        observations,
        terminated,
        truncated,
        next_observations,
        rewards,
        actions,
        _,
        _,
    ) = get_batch_from_buffer(
        buffer,
        agent_state.collector_state.buffer_state,
        sample_key,
    )
    dones = jnp.logical_or(terminated, truncated).astype(jnp.float32)

    target_q = td_target_fn(
        q_state=agent_state.actor_state,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        reward_scale=reward_scale,
    )

    (_, value_aux), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
        agent_state.actor_state.params,
        agent_state.actor_state,
        observations,
        actions,
        target_q,
        td_loss_fn,
    )
    q_state = agent_state.actor_state.apply_gradients(grads=grads)

    # Periodic target refresh: hard update is tau=1.0, Polyak is tau<1.
    do_target_update = (agent_state.n_updates % target_update_interval) == 0
    q_state = jax.lax.cond(
        do_target_update,
        lambda s: s.soft_update(tau),
        lambda s: s,
        q_state,
    )

    agent_state = agent_state.replace(
        actor_state=q_state,
        n_updates=agent_state.n_updates + 1,
    )
    return agent_state, AuxiliaryLogs(value=value_aux)


# ---------------------------------------------------------------------------
# Training iteration (collect 1 step + maybe update + log)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "env_args",
        "mode",
        "recurrent",
        "buffer",
        "agent_config",
        "total_timesteps",
        "lstm_hidden_size",
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "action_pipeline",
        "eval_action_transform",
        "td_target_fn",
        "td_loss_fn",
    ],
)
def training_iteration(
    agent_state: DQNState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_config: DQNConfig,
    total_timesteps: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: Optional[int] = 1000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    td_target_fn: Callable = compute_dqn_td_target,
    td_loss_fn: Callable = mse_td_loss,
) -> Tuple[DQNState, Any]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
        action_pipeline=action_pipeline,
    )
    agent_state, _transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=1
    )
    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        update_scan_fn = partial(
            update_agent,
            buffer=buffer,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
            target_update_interval=agent_config.target_update_interval,
            reward_scale=agent_config.reward_scale,
            td_target_fn=td_target_fn,
            td_loss_fn=td_loss_fn,
        )
        # Carry-only scan: only the final-step aux is materialised.
        agent_state, aux = final_aux_scan(
            update_scan_fn, agent_state, length=agent_config.n_gradient_steps
        )
        aux = jax.tree.map(lambda x: x.reshape((1,)), aux)
        return agent_state, aux

    def skip_update(agent_state):
        return agent_state, _fill_with_nan(AuxiliaryLogs)

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
        eval_action_transform=eval_action_transform,
    )
    return agent_state, metrics_to_log


# ---------------------------------------------------------------------------
# Training factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: DQNConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.5,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    td_target_fn: Optional[Callable] = None,
    td_loss_fn: Optional[Callable] = None,
    q_network_cls: Optional[type] = None,
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    recurrent = network_args.lstm_hidden_size is not None
    n_actions = get_action_dim(env_args.env, env_args.env_params)

    # Resolve composable hooks to their vanilla-DQN defaults when unset.
    td_target_fn = td_target_fn if td_target_fn is not None else compute_dqn_td_target
    td_loss_fn = td_loss_fn if td_loss_fn is not None else mse_td_loss

    if action_pipeline is None:
        action_pipeline = make_epsilon_greedy_pipeline(
            env_args=env_args,
            n_actions=n_actions,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_frac=epsilon_decay_frac,
            total_timesteps=total_timesteps,
        )

    @train_jit
    def train(key, index: Optional[int] = None):
        agent_state = init_DQN(
            key=key,
            env_args=env_args,
            optimizer_args=critic_optimizer_args,
            network_args=network_args,
            buffer=buffer,
            n_actions=n_actions,
            q_network_cls=q_network_cls,
        )

        num_updates = total_timesteps // env_args.n_envs

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=recurrent,
            agent_config=agent_config,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            lstm_hidden_size=network_args.lstm_hidden_size,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            action_pipeline=action_pipeline,
            eval_action_transform=eval_action_transform,
            td_target_fn=td_target_fn,
            td_loss_fn=td_loss_fn,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train
