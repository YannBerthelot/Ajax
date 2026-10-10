"""TD3 (Fujimoto et al., 2018): Twin Delayed DDPG.

Three changes vs DDPG:
1. Twin critics, target = min over the two target Qs (overestimation bias).
2. Target policy smoothing: target action = clip(mu_target(s') + clip(N, -c, c), -1, 1).
3. Delayed policy + target updates every `policy_delay` critic steps.

The actor reuses Ajax's stochastic SquashedNormal head and is treated
deterministically by taking pi.mean() (== tanh(mu)) for both target
and behaviour. Exploration noise is added at action time.
"""

from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from jax.tree_util import Partial as partial

from ajax.agents.cloning import CloningConfig, get_pre_trained_agent
from ajax.agents.recurrent import (
    RecurrentCarries,
    sample_and_burnin_sequences,
    unsupported_recurrent_options,
)
from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.state import TD3Config, TD3State
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    get_pi_sequence,
    init_collector_state,
    should_use_uniform_sampling,
)
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import ExtensionStack
from ajax.log import compose_eval_metrics, evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.memory import flat_carry_dim
from ajax.networks.networks import predict_value, predict_value_sequence
from ajax.perf_utils import train_jit
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType

# ---------------------------------------------------------------------------
# Auxiliary dataclasses (for logging)
# ---------------------------------------------------------------------------


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    q_mean: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_min: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


# ---------------------------------------------------------------------------
# Action pipeline result (matches SAC's structure for collect_experience)
# ---------------------------------------------------------------------------


class TD3ActionPipelineResult(NamedTuple):
    env_action: jax.Array
    policy_action: jax.Array
    log_probs: jax.Array
    is_expert_flag: jax.Array
    in_value_box: jax.Array
    entry_bonus: jax.Array
    rng: jax.Array
    new_expert_state: Optional[Any] = None
    buffer_action: Optional[jax.Array] = None
    # Advanced actor carry (recurrent mode). collect_experience applies it
    # so the policy's memory keeps moving during collection.
    new_actor_hidden: Optional[Any] = None


def _deterministic_action(actor_state, obs, done, recurrent):
    """Mean of the SquashedNormal actor = tanh(mu(s)). Deterministic policy.

    Returns (action, new_actor_state): recurrent actors advance their
    carry on every forward pass and the caller must keep it."""
    pi, new_actor_state = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=obs,
        done=done,
        recurrent=recurrent,
    )
    action = pi.mean()
    if recurrent:
        action = action.squeeze(0)  # drop single-step time axis
    return action, new_actor_state


def make_default_action_pipeline(env_args, recurrent: bool, exploration_noise: float):
    """Default TD3 action selection: deterministic policy + Gaussian exploration noise."""

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        del raw_obs
        obs = agent_state.collector_state.last_obs
        done = jnp.logical_or(
            agent_state.collector_state.last_terminated,
            agent_state.collector_state.last_truncated,
        )
        mean_action, new_actor_state = _deterministic_action(
            agent_state.actor_state, obs, done, recurrent
        )
        noise = jax.random.normal(action_key, mean_action.shape) * exploration_noise
        policy_action = jnp.clip(mean_action + noise, -1.0, 1.0)
        log_probs = jnp.zeros(mean_action.shape[:-1] + (1,))
        uniform_action = jax.random.uniform(
            mix_key, minval=-1.0, maxval=1.0, shape=policy_action.shape
        )
        env_action = jax.lax.cond(
            uniform, lambda: uniform_action, lambda: policy_action
        )
        n_envs = env_args.n_envs
        return TD3ActionPipelineResult(
            env_action=env_action,
            policy_action=policy_action,
            log_probs=log_probs,
            is_expert_flag=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            in_value_box=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            entry_bonus=jnp.zeros((n_envs, 1), dtype=jnp.float32),
            rng=rng,
            new_actor_hidden=(new_actor_state.hidden_state if recurrent else None),
        )

    return pipeline


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_TD3(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    num_critics: int = 2,
    window_size: int = 10,
    stored_state: bool = False,
    pid_actor_config: Optional[PIDActorConfig] = None,
    expert_policy: Optional[Callable] = None,
) -> TD3State:
    rng, init_key, collector_key = jax.random.split(key, num=3)

    actor_state, critic_state = get_initialized_td3_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        num_critics=num_critics,
        pid_actor_config=pid_actor_config,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    _actor_carry_dim = 0
    if stored_state and network_args.memory is not None:
        _actor_carry_dim = flat_carry_dim(network_args.memory)
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
        actor_carry_dim=_actor_carry_dim,
    )
    # Seed batched expert state for stateful experts (PID integrator, CPG
    # phase). Mirrors SAC's pattern in init_SAC; required so the scan body's
    # carry input/output shapes agree when the action_pipeline returns a
    # non-None ``new_expert_state``.
    if expert_policy is not None and hasattr(expert_policy, "init_state"):
        collector_state = collector_state.replace(
            expert_state=expert_policy.init_state(env_args.n_envs)
        )
    return TD3State(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
    )


# ---------------------------------------------------------------------------
# Critic update — target policy smoothing + twin clipped target
# ---------------------------------------------------------------------------


def compute_td3_td_target(
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    recurrent: bool,
    target_policy_noise: float,
    target_noise_clip: float,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
) -> jax.Array:
    """y = r + gamma * (1-d) * min_i Q_target_i(s', clip(mu_target(s') + clip(N, -c, c), -1, 1))."""
    rewards = rewards * reward_scale

    # Target action via target params, deterministic mean
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        pi_target, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_state.target_params,
            obs=next_observations,
            resets=carries.next_resets,
            initial_hidden=carries.target_actor_next_hidden,
        )
    else:
        pi_target, _ = get_pi(
            actor_state=actor_state,
            actor_params=actor_state.target_params,
            obs=next_observations,
            done=dones,
            recurrent=recurrent,
        )
    next_action = pi_target.mean()

    noise = jax.random.normal(rng, next_action.shape) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_action = jnp.clip(next_action + noise, -1.0, 1.0)

    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        q_targets, _ = predict_value_sequence(
            critic_state=critic_state,
            critic_params=critic_state.target_params,
            x=jnp.concatenate((next_observations, next_action), axis=-1),
            resets=carries.next_resets,
            initial_hidden=carries.target_critic_hidden,
        )
    else:
        q_targets = predict_value(
            critic_state=critic_state,
            critic_params=critic_state.target_params,
            x=jnp.concatenate((next_observations, next_action), axis=-1),
        )
    min_q_target = jnp.min(q_targets, axis=0)

    target = rewards + gamma * (1.0 - dones) * min_q_target
    return jax.lax.stop_gradient(target)


def value_loss_function(
    critic_params: FrozenDict,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    recurrent: bool,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        q_preds, _ = predict_value_sequence(
            critic_state=critic_state,
            critic_params=critic_params,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
            resets=carries.resets,
            initial_hidden=carries.critic_hidden,
        )
    else:
        q_preds = predict_value(
            critic_state=critic_state,
            critic_params=critic_params,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
        )
    loss = jnp.mean((q_preds - target_q) ** 2)
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        target_q=target_q.mean().flatten(),
        q_pred_min=jnp.min(q_preds, axis=0).mean().flatten(),
    )


def update_value_functions(
    agent_state: TD3State,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    target_policy_noise: float,
    target_noise_clip: float,
    reward_scale: float,
    extension_stack: Optional[ExtensionStack] = None,
    total_timesteps: int = 1,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[TD3State, ValueAuxiliaries]:
    value_loss_key, rng = jax.random.split(agent_state.rng)

    target_q = compute_td3_td_target(
        actor_state=agent_state.actor_state,
        critic_state=agent_state.critic_state,
        rng=value_loss_key,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        recurrent=recurrent,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        reward_scale=reward_scale,
        carries=carries,
    )

    # Extension fold: TD-target shaping (analogous to SAC's ``stack.on_target``).
    # Empty stack ⇒ identity.
    if extension_stack is not None:
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
            value_loss_key,
            total_timesteps,
        )
        target_q = jax.lax.stop_gradient(target_q)

    def _critic_loss(params, c_state, obs, act, tgt):
        loss, core_aux = value_loss_function(
            params, c_state, obs, act, tgt, recurrent, carries=carries
        )
        if extension_stack is not None:
            _cl_batch = {
                "observations": obs,
                "actions": act,
                "targets": tgt,
                "critic_params": params,
                "critic_state": c_state,
            }
            loss = loss + extension_stack.fold_critic_loss(
                agent_state,
                _cl_batch,
                agent_state.collector_state.timestep,
                value_loss_key,
                total_timesteps,
            )
        return loss, core_aux

    (loss, aux), grads = jax.value_and_grad(_critic_loss, has_aux=True)(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        actions,
        target_q,
    )
    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, critic_state=updated_critic_state), aux


# ---------------------------------------------------------------------------
# Policy update (delayed)
# ---------------------------------------------------------------------------


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, jax.Array]]:
    """TD3 actor loss ``-Q1(s, mu(s))``; also returns ``mu(s)`` for extensions."""
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        pi, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=observations,
            resets=carries.resets,
            initial_hidden=carries.actor_hidden,
        )
    else:
        pi, _ = get_pi(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=observations,
            done=dones,
            recurrent=recurrent,
        )
    actions = pi.mean()  # deterministic

    # TD3 uses Q1 only for the actor objective.
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        q_preds, _ = predict_value_sequence(
            critic_state=critic_state,
            critic_params=critic_state.params,
            x=jnp.concatenate([observations, actions], axis=-1),
            resets=carries.resets,
            initial_hidden=carries.critic_hidden,
        )
    else:
        q_preds = predict_value(
            critic_state=critic_state,
            critic_params=critic_state.params,
            x=jnp.concatenate([observations, actions], axis=-1),
        )
    q_first = q_preds[0]
    loss = -q_first.mean()
    return loss, (PolicyAuxiliaries(policy_loss=loss, q_mean=q_first.mean()), actions)


def update_policy(
    agent_state: TD3State,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    raw_observations: Optional[jax.Array] = None,
    extension_stack: Optional[ExtensionStack] = None,
    total_timesteps: int = 1,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[TD3State, PolicyAuxiliaries]:
    def _actor_loss(params):
        loss, (core_aux, pi_mean) = policy_loss_function(
            params,
            agent_state.actor_state,
            agent_state.critic_state,
            observations,
            dones,
            recurrent,
            carries,
        )
        if extension_stack is not None:
            _al_batch = {
                "observations": observations,
                "raw_observations": raw_observations,
                "pi_mean": pi_mean,
                "actor_params": params,
                "actor_state": agent_state.actor_state,
            }
            loss = loss + extension_stack.fold_actor_loss(
                agent_state,
                _al_batch,
                agent_state.collector_state.timestep,
                agent_state.rng,
                total_timesteps,
            )
        return loss, core_aux

    (loss, aux), grads = jax.value_and_grad(_actor_loss, has_aux=True)(
        agent_state.actor_state.params,
    )
    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)
    return agent_state.replace(actor_state=updated_actor_state), aux


# ---------------------------------------------------------------------------
# Target network soft update — applies to BOTH actor and critic for TD3
# ---------------------------------------------------------------------------


def update_target_networks(agent_state: TD3State, tau: float) -> TD3State:
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau),
        actor_state=agent_state.actor_state.soft_update(tau=tau),
    )


# ---------------------------------------------------------------------------
# Per-iteration agent update (one critic + maybe-policy + maybe-target step)
# ---------------------------------------------------------------------------


def update_agent(
    agent_state: TD3State,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    tau: float,
    policy_delay: int,
    target_policy_noise: float,
    target_noise_clip: float,
    reward_scale: float,
    extension_stack: Optional[ExtensionStack] = None,
    total_timesteps: int = 1,
    burn_in: int = 8,
    stored_state: bool = False,
) -> Tuple[TD3State, AuxiliaryLogs]:
    sample_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    carries = None
    if recurrent:
        # Sequence replay with burned-in carries (R2D2-style). TD3's
        # bootstrap action comes from the TARGET actor, so its carry is
        # burned in as well (burn_target_actor=True).
        transition, carries = sample_and_burnin_sequences(
            agent_state,
            buffer,
            sample_key,
            burn_in,
            burn_target_actor=True,
            stored_state=stored_state,
        )
        raw_observations = None
    else:
        (
            observations,
            terminated,
            truncated,
            next_observations,
            rewards,
            actions,
            raw_observations,
            _,
        ) = get_batch_from_buffer(
            buffer,
            agent_state.collector_state.buffer_state,
            sample_key,
        )
        transition = Transition(
            observations, actions, rewards, terminated, truncated, next_observations
        )
    dones = jnp.logical_or(transition.terminated, transition.truncated)

    # Critic step (always)
    agent_state, value_aux = update_value_functions(
        agent_state=agent_state,
        observations=transition.obs,
        actions=transition.action,
        next_observations=transition.next_obs,
        dones=dones,
        recurrent=recurrent,
        rewards=transition.reward,
        gamma=gamma,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        reward_scale=reward_scale,
        extension_stack=extension_stack,
        total_timesteps=total_timesteps,
        carries=carries,
    )

    # Delayed policy + target update
    do_policy = (agent_state.n_updates % policy_delay) == 0

    def policy_and_targets(agent_state):
        agent_state, policy_aux = update_policy(
            agent_state=agent_state,
            observations=transition.obs,
            dones=dones,
            recurrent=recurrent,
            raw_observations=raw_observations,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
            carries=carries,
        )
        agent_state = update_target_networks(agent_state, tau=tau)
        return agent_state, policy_aux

    def skip_policy(agent_state):
        zero = jnp.zeros(())
        return agent_state, PolicyAuxiliaries(policy_loss=zero, q_mean=zero)

    agent_state, policy_aux = jax.lax.cond(
        do_policy, policy_and_targets, skip_policy, operand=agent_state
    )

    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)
    return agent_state, AuxiliaryLogs(policy=policy_aux, value=value_aux)


# ---------------------------------------------------------------------------
# Training iteration (collect 1 step + maybe update + log)
# ---------------------------------------------------------------------------


def training_iteration(
    agent_state: TD3State,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_config: TD3Config,
    total_timesteps: int,
    log_frequency: int = 1000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    expert_policy: Optional[Callable] = None,
    action_pipeline: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
) -> tuple[TD3State, Any]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        store_hidden=recurrent and agent_config.stored_state,
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
        agent_state, aux = update_agent(
            agent_state,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
            policy_delay=agent_config.policy_delay,
            target_policy_noise=agent_config.target_policy_noise,
            target_noise_clip=agent_config.target_noise_clip,
            reward_scale=agent_config.reward_scale,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
            burn_in=agent_config.burn_in,
            stored_state=agent_config.stored_state,
        )
        # One (1,)-shaped leaf per metric: the metric-flattening contract.
        aux = jax.tree.map(lambda x: x.reshape((1,)), aux)

        # Extension post_update — folded once per training_iteration, after
        # the gradient step. Empty stack ⇒ identity.
        if extension_stack is not None:
            _pu_rng, _pu_rng2 = jax.random.split(agent_state.rng)
            agent_state = agent_state.replace(rng=_pu_rng2)
            agent_state = extension_stack.fold_post_update(
                agent_state,
                agent_state.collector_state.timestep,
                _pu_rng,
                total_timesteps,
            )
        return agent_state, aux

    def fill_with_nan(dataclass):
        nan = jnp.ones(1) * jnp.nan
        d = {}
        for field in fields(dataclass):
            sub = field.type
            if hasattr(sub, "__dataclass_fields__"):
                d[field.name] = fill_with_nan(sub)
            else:
                d[field.name] = nan
        return dataclass(**d)

    def skip_update(agent_state):
        return agent_state, fill_with_nan(AuxiliaryLogs)

    agent_state, aux = jax.lax.cond(
        timestep >= agent_config.learning_starts,
        do_update,
        skip_update,
        operand=agent_state,
    )

    _extra_eval = compose_eval_metrics(None, extension_stack, total_timesteps)
    agent_state, metrics_to_log = evaluate_and_log(
        agent_state,
        aux,
        index,
        mode,
        env_args,
        num_episode_test,
        recurrent,
        log,
        log_fn,
        log_frequency,
        total_timesteps,
        expert_policy=expert_policy,
        extra_eval_metrics=_extra_eval,
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
    agent_config: TD3Config,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    action_pipeline: Optional[Callable] = None,
    extensions: Sequence = (),
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    pre_train_n_steps = cloning_args.pre_train_n_steps if cloning_args else 0
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options(
            "TD3",
            expert_policy=expert_policy,
            extensions=(tuple(extensions) or None),
            # TD3 always builds a default CloningConfig; only actual
            # pre-training (pre_train_n_steps > 0) conflicts with memory.
            cloning_pretrain=(cloning_args if pre_train_n_steps > 0 else None),
            pid_actor_config=pid_actor_config,
        )
    if action_pipeline is None:
        action_pipeline = make_default_action_pipeline(
            env_args=env_args,
            recurrent=recurrent,
            exploration_noise=agent_config.exploration_noise,
        )

    extension_stack = ExtensionStack(extensions) if extensions else None

    @train_jit
    def train(key, index: Optional[int] = None):
        init_key, expert_key = jax.random.split(key)
        agent_state = init_TD3(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            stored_state=agent_config.stored_state,
            buffer=buffer,
            num_critics=agent_config.num_critics,
            pid_actor_config=pid_actor_config,
            expert_policy=expert_policy,
        )

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

        if extension_stack is not None:
            _ext_key, _pre_key = jax.random.split(expert_key)
            agent_state = extension_stack.fold_init_states(agent_state, _ext_key)
            agent_state = extension_stack.fold_pretrain(
                agent_state, jnp.asarray(0), _pre_key, total_timesteps
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
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            expert_policy=expert_policy,
            action_pipeline=action_pipeline,
            extension_stack=extension_stack,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train
