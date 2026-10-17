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
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.DQN.networks import (
    get_initialized_q_network,
    predict_q,
    q_network_value,
)
from ajax.agents.DQN.state import DQNConfig, DQNState
from ajax.agents.loop import TrainLoop, gradient_step
from ajax.agents.recurrent import (
    actor_dist,
    sample_replay,
    stored_actor_carry_dim,
    unsupported_recurrent_options,
)
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    ActionPipelineResult,
    get_pi,
    init_collector_state,
    reward_statistics,
)
from ajax.environments.utils import get_action_dim
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.perf_utils import final_aux_scan
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import BufferType


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_mean: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    # Nested one level deep so ajax.log.flatten_dict names the metrics
    # ``value/<field>``, as the other agents' auxiliaries are named.
    value: ValueAuxiliaries


def make_epsilon_greedy_pipeline(
    n_actions: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_frac: float,
    total_timesteps: int,
    recurrent: bool = False,
) -> Callable:
    """Build the default DQN ``action_pipeline``: linear-decay epsilon-greedy.

    Epsilon decays linearly from ``epsilon_start`` to ``epsilon_end`` over
    the first ``epsilon_decay_frac`` of training, then stays flat. During
    the ``learning_starts`` warmup (``uniform`` is set) actions are fully
    random regardless of epsilon.
    """

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        del raw_obs, mix_key
        collector = agent_state.collector_state
        new_hidden = None
        if recurrent:
            # A recurrent Q-network runs a one-step sequence on its live
            # carry, which acting advances.
            done = jnp.logical_or(collector.last_terminated, collector.last_truncated)
            pi, q_state = get_pi(
                agent_state.actor_state,
                agent_state.actor_state.params,
                collector.last_obs,
                done,
                recurrent,
            )
            q_values, new_hidden = pi.q_values.squeeze(0), q_state.hidden_state
        else:
            q_values = predict_q(
                agent_state.actor_state,
                agent_state.actor_state.params,
                collector.last_obs,
            )
        greedy_action = jnp.argmax(q_values, axis=-1).astype(jnp.int32)

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

        env_action = jax.lax.cond(
            uniform, lambda: random_action, lambda: eps_greedy_action
        )
        # The env expects (n_envs,) integer actions; the buffer schema
        # stores discrete actions as (n_envs, 1).
        buffer_action = env_action[:, None]
        n_envs = greedy_action.shape[0]
        return ActionPipelineResult(
            env_action=env_action,
            policy_action=buffer_action,
            log_probs=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            is_expert_flag=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            in_value_box=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            entry_bonus=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            rng=rng,
            buffer_action=buffer_action,
            new_actor_hidden=new_hidden,
        )

    return pipeline


def init_DQN(
    key: jax.Array,
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    n_actions: int,
    buffer: Optional[BufferType] = None,
    q_network_cls: Optional[type] = None,
    state_cls: type = DQNState,
    stored_state: bool = False,
) -> Any:
    """The Q-network and the collector (writing to ``buffer``, if any) in a
    ``state_cls``: DQN's, or a descendant's (PQN: no buffer)."""
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)

    q_state = get_initialized_q_network(
        key=init_key,
        env_config=env_args,
        optimizer_config=optimizer_args,
        network_config=network_args,
        n_actions=n_actions,
        q_network_cls=q_network_cls,
    )
    mode = env_args.mode
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
    )
    # The single Q-network lives in actor_state (so the shared eval loop
    # works); critic_state mirrors it at init and is never updated.
    return state_cls(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=q_state,
        critic_state=q_state,
        collector_state=collector_state,
    )


def compute_dqn_td_target(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float,
    next_q_values: Optional[Callable[[bool], jax.Array]] = None,
) -> jax.Array:
    """Vanilla DQN target: y = r + gamma * (1 - d) * max_a' Q_target(s', a').
    ``next_q_values(target)`` reads the Q-values at s' (a recurrent
    network's, on its replayed sequence); by default the network on
    ``next_observations``."""
    rewards = rewards * reward_scale
    q_next = _next_q(q_state, next_observations, next_q_values)(True)
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
    next_q_values: Optional[Callable[[bool], jax.Array]] = None,
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
    q_at_next = _next_q(q_state, next_observations, next_q_values)
    next_actions = jnp.argmax(q_at_next(False), axis=-1, keepdims=True)
    q_next_target = q_at_next(True)
    q_next = jnp.take_along_axis(q_next_target, next_actions, axis=-1)
    target = rewards + gamma * (1.0 - dones) * q_next
    return jax.lax.stop_gradient(target)


def _next_q(
    q_state: LoadedTrainState,
    next_observations: jax.Array,
    next_q_values: Optional[Callable[[bool], jax.Array]],
) -> Callable[[bool], jax.Array]:
    """``target -> Q(s')``: ``next_q_values``, or the network (its target
    parameters when ``target``) on ``next_observations``."""
    if next_q_values is not None:
        return next_q_values

    def q(target: bool) -> jax.Array:
        params = q_state.target_params if target else q_state.params
        return predict_q(q_state, params, next_observations)

    return q


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
    q_values: Optional[Callable[[Any], jax.Array]] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """TD loss between Q(s, a) at the taken action and a pre-computed target.

    ``loss_fn(q_taken, target_q) -> scalar`` selects the error norm
    (``mse_td_loss`` by default, ``make_huber_td_loss(...)`` for Huber).
    ``q_values(params)`` reads the Q-values of ``observations`` (a recurrent
    network's, over a sequence from its carry); by default the network on
    them.
    """
    if q_values is None:
        q_all = predict_q(q_state, q_params, observations)
    else:
        q_all = q_values(q_params)
    # actions is (batch, 1) int32 -> gather one Q-value per row.
    q_taken = jnp.take_along_axis(q_all, actions, axis=-1)
    loss = loss_fn(q_taken, target_q)
    aux = ValueAuxiliaries(
        critic_loss=loss.flatten(),
        target_q=target_q.mean().flatten(),
        q_pred_mean=q_taken.mean().flatten(),
    )
    return loss, aux


def q_gradient_step(
    agent_state: Any,
    observations: jax.Array,
    actions: jax.Array,
    targets: jax.Array,
    td_loss_fn: Callable,
    extension_stack: ExtensionStack,
    key: Optional[jax.Array],
    total_timesteps: int,
    q_values: Optional[Callable[[Any], jax.Array]] = None,
) -> Tuple[LoadedTrainState, ValueAuxiliaries]:
    """One step of the Q-network (``agent_state.actor_state``) down the TD
    loss plus the extensions' critic-loss terms, folded on ``key`` (drawn
    for them alone; ``None`` without extensions). Returns the stepped
    Q-network and the loss's metrics."""
    q_state = agent_state.actor_state

    def loss_fn(params: Any) -> Tuple[jax.Array, ValueAuxiliaries]:
        loss, aux = q_loss_fn(
            params, q_state, observations, actions, targets, td_loss_fn, q_values
        )
        if extension_stack:
            assert key is not None
            batch = {
                "observations": observations,
                "actions": actions,
                "targets": targets,
                # At the parameters being differentiated, so the term has a
                # gradient.
                "q_state": q_state.replace(params=params),
            }
            loss = loss + extension_stack.fold_critic_loss(
                agent_state,
                batch,
                agent_state.collector_state.timestep,
                key,
                total_timesteps,
            )
        return loss, aux

    return gradient_step(q_state, loss_fn)


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
        "extension_stack",
        "total_timesteps",
        "recurrent",
        "burn_in",
        "stored_state",
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
    extension_stack: ExtensionStack,
    total_timesteps: int,
    recurrent: bool = False,
    burn_in: int = 0,
    stored_state: bool = False,
) -> Tuple[DQNState, AuxiliaryLogs]:
    """One update on a replay batch, or with memory on replayed sequences
    (R2D2: Kapturowski et al. 2019): the Q-network's carry burned in, or
    stored, for each sequence, its target network's burned in with the
    target parameters."""
    sample_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    collector = agent_state.collector_state
    carries, next_q = None, {}
    if recurrent:
        batch, carries = sample_replay(
            agent_state,
            buffer,
            sample_key,
            recurrent=True,
            burn_in=burn_in,
            stored_state=stored_state,
            burn_target_actor=True,
            burn_critic=False,
        )
    else:
        batch = get_batch_from_buffer(
            buffer,
            collector.buffer_state,
            sample_key,
            reward_statistics(collector.env_state),
        )
    observations, actions, rewards = batch.obs, batch.action, batch.reward
    terminated, next_observations = batch.terminated, batch.next_obs
    # A truncated row bootstraps on its final observation (bootstrap_obs).
    dones = terminated.astype(jnp.float32)
    if carries is not None:
        q_state = agent_state.actor_state

        def next_q_values(target: bool) -> jax.Array:
            params = q_state.target_params if target else q_state.params
            return actor_dist(
                q_state,
                params,
                next_observations,
                carries,
                bootstrap=True,
                target_actor=target,
            ).q_values

        next_q = {"next_q_values": next_q_values}

    target_q = td_target_fn(
        q_state=agent_state.actor_state,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        reward_scale=reward_scale,
        **next_q,
    )

    # The extensions reshape the target, then add their loss terms, each on
    # a key of its own (drawn only with extensions).
    loss_key = None
    if extension_stack:
        ext_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        target_key, loss_key = jax.random.split(ext_key)
        _tgt_batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
            "dones": dones,
            "gamma": gamma,
            "reward_scale": reward_scale,
        }
        target_q = extension_stack.fold_on_target(
            agent_state,
            _tgt_batch,
            target_q,
            agent_state.collector_state.timestep,
            target_key,
            total_timesteps,
        )

    q_state, value_aux = q_gradient_step(
        agent_state,
        observations,
        actions,
        target_q,
        td_loss_fn,
        extension_stack,
        loss_key,
        total_timesteps,
        None
        if carries is None
        else (lambda p: actor_dist(q_state, p, observations, carries).q_values),
    )

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


def make_train(
    env_args: EnvironmentConfig,
    # base.py passes every agent both optimiser configs; one Q-network here.
    actor_optimizer_args: OptimizerConfig,  # noqa: ARG001
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: DQNConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.5,
    td_target_fn: Optional[Callable] = None,
    td_loss_fn: Optional[Callable] = None,
    q_network_cls: Optional[type] = None,
    extensions: Sequence = (),
):
    """DQN's train function: one step per env, then (from ``learning_starts``)
    ``n_gradient_steps`` updates, per iteration."""
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options("DQN", extensions=(tuple(extensions) or None))
        if td_target_fn not in (
            None,
            compute_dqn_td_target,
            compute_double_dqn_td_target,
        ):
            raise NotImplementedError(
                "Recurrent DQN reads Q(s') on replayed sequences: it takes the"
                " built-in TD targets only (compute_dqn_td_target,"
                " compute_double_dqn_td_target)."
            )
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
    )
    n_actions = get_action_dim(env_args.env, env_args.env_params)
    td_target_fn = td_target_fn if td_target_fn is not None else compute_dqn_td_target
    td_loss_fn = td_loss_fn if td_loss_fn is not None else mse_td_loss

    def init(key: jax.Array) -> DQNState:
        return init_DQN(
            key,
            env_args,
            critic_optimizer_args,
            network_args,
            n_actions,
            buffer=buffer,
            q_network_cls=q_network_cls,
            stored_state=agent_config.stored_state,
        )

    def update(agent_state: DQNState, _transition: Any) -> Any:
        # The step just collected is in the buffer: DQN samples it from there.
        one_update = partial(
            update_agent,
            buffer=buffer,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
            target_update_interval=agent_config.target_update_interval,
            reward_scale=agent_config.reward_scale,
            td_target_fn=td_target_fn,
            td_loss_fn=td_loss_fn,
            extension_stack=loop.stack,
            total_timesteps=total_timesteps,
            recurrent=recurrent,
            burn_in=agent_config.burn_in,
            stored_state=agent_config.stored_state,
        )
        # Carry-only scan: only the final-step aux is materialised.
        return final_aux_scan(
            one_update, agent_state, length=agent_config.n_gradient_steps
        )

    action_pipeline = make_epsilon_greedy_pipeline(
        n_actions=n_actions,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_frac=epsilon_decay_frac,
        total_timesteps=total_timesteps,
        recurrent=recurrent,
    )
    return loop.off_policy(
        init,
        update,
        AuxiliaryLogs,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "action_pipeline": action_pipeline,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={
            "value_estimate": ValueEstimate(
                q_network_value, agent_config.gamma, agent_config.reward_scale
            )
        },
    )
