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
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.cloning import (
    CloningConfig,
    get_cloning_args,
    get_pre_trained_agent,
)
from ajax.agents.TD3.state import TD3Config, TD3State
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
from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.networks import predict_value
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType
from ajax.utils import get_one


# ---------------------------------------------------------------------------
# Auxiliary dataclasses (for logging)
# ---------------------------------------------------------------------------


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    q_mean: jax.Array
    imitation_loss: jax.Array
    raw_loss: jax.Array


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


def _deterministic_action(actor_state, obs, done, recurrent):
    """Mean of the SquashedNormal actor = tanh(mu(s)). Deterministic policy."""
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=obs,
        done=done,
        recurrent=recurrent,
    )
    return pi.mean()


def make_default_action_pipeline(env_args, recurrent: bool, exploration_noise: float):
    """Default TD3 action selection: deterministic policy + Gaussian exploration noise."""

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        del raw_obs
        obs = agent_state.collector_state.last_obs
        done = jnp.logical_or(
            agent_state.collector_state.last_terminated,
            agent_state.collector_state.last_truncated,
        )
        mean_action = _deterministic_action(
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
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
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
) -> jax.Array:
    """y = r + gamma * (1-d) * min_i Q_target_i(s', clip(mu_target(s') + clip(N, -c, c), -1, 1))."""
    rewards = rewards * reward_scale

    # Target action via target params, deterministic mean
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

    q_targets = predict_value(
        critic_state=critic_state,
        critic_params=critic_state.target_params,
        x=jnp.concatenate((next_observations, next_action), axis=-1),
    )
    min_q_target = jnp.min(q_targets, axis=0)

    target = rewards + gamma * (1.0 - dones) * min_q_target
    return jax.lax.stop_gradient(target)


@partial(jax.jit, static_argnames=["recurrent"])
def value_loss_function(
    critic_params: FrozenDict,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    recurrent: bool,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    del recurrent
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
    target_modifier: Optional[Callable] = None,
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
    )

    if target_modifier is not None:
        q_preds_for_modifier = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
        )
        target_q, _, _ = target_modifier(
            target_q,
            agent_state,
            observations,
            actions,
            next_observations,
            dones,
            gamma,
            value_loss_key,
            q_preds_for_modifier,
        )
        target_q = jax.lax.stop_gradient(target_q)

    (loss, aux), grads = jax.value_and_grad(value_loss_function, has_aux=True)(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        actions,
        target_q,
        recurrent,
    )
    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, critic_state=updated_critic_state), aux


# ---------------------------------------------------------------------------
# Policy update (delayed)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "expert_policy",
        "distance_to_stable",
        "imitation_coef_offset",
        "obs_preprocessor",
        "policy_action_transform",
    ],
)
def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = get_one,
    imitation_coef_offset: float = 0.0,
    a_expert_precomputed: Optional[jax.Array] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    obs_for_actor = (
        obs_preprocessor(observations) if obs_preprocessor is not None else observations
    )
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=obs_for_actor,
        done=dones,
        recurrent=recurrent,
    )
    actions = pi.mean()  # deterministic

    _raw_obs = raw_observations if raw_observations is not None else observations
    q_input_actions = (
        policy_action_transform(actions, _raw_obs, a_expert_precomputed)
        if policy_action_transform is not None
        else actions
    )

    # TD3 uses Q1 only for the actor objective.
    q_preds = predict_value(
        critic_state=critic_state,
        critic_params=critic_state.params,
        x=jnp.concatenate([observations, q_input_actions], axis=-1),
    )
    q_first = q_preds[0]
    raw_loss = -q_first.mean()

    # Inline behavior-cloning loss: ||a_pi - a_expert||^2 / 4 (same scaling as
    # ajax.agents.cloning.compute_imitation_score for non-Categorical pi).
    # Skipped at trace time when no expert is set, so the graph stays clean.
    if expert_policy is not None:
        expert_action = jax.lax.stop_gradient(expert_policy(_raw_obs))
        imitation_loss = jnp.mean(jnp.square(actions - expert_action) / 4.0)
    else:
        imitation_loss = jnp.zeros(())

    loss = raw_loss + imitation_coef * imitation_loss
    return loss, PolicyAuxiliaries(
        policy_loss=loss,
        q_mean=q_first.mean(),
        imitation_loss=imitation_loss,
        raw_loss=raw_loss,
    )


def update_policy(
    agent_state: TD3State,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = get_one,
    imitation_coef_offset: float = 0.0,
    a_expert_precomputed: Optional[jax.Array] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
) -> Tuple[TD3State, PolicyAuxiliaries]:
    (loss, aux), grads = jax.value_and_grad(policy_loss_function, has_aux=True)(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        dones,
        recurrent,
        raw_observations,
        expert_policy,
        imitation_coef,
        distance_to_stable,
        imitation_coef_offset,
        a_expert_precomputed,
        obs_preprocessor,
        policy_action_transform,
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


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "buffer",
        "gamma",
        "tau",
        "policy_delay",
        "target_policy_noise",
        "target_noise_clip",
        "reward_scale",
        "expert_policy",
        "imitation_coef",
        "distance_to_stable",
        "imitation_coef_offset",
        "target_modifier",
        "obs_preprocessor",
        "policy_action_transform",
    ],
)
def update_agent(
    agent_state: TD3State,
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    tau: float,
    policy_delay: int,
    target_policy_noise: float,
    target_noise_clip: float,
    reward_scale: float,
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = get_one,
    imitation_coef_offset: float = 0.0,
    target_modifier: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
) -> Tuple[TD3State, AuxiliaryLogs]:
    sample_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

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

    a_expert_precomputed = None
    if expert_policy is not None and policy_action_transform is not None:
        _raw = raw_observations if raw_observations is not None else transition.obs
        a_expert_precomputed = jax.lax.stop_gradient(expert_policy(_raw))

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
        target_modifier=target_modifier,
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
            expert_policy=expert_policy,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset,
            a_expert_precomputed=a_expert_precomputed,
            obs_preprocessor=obs_preprocessor,
            policy_action_transform=policy_action_transform,
        )
        agent_state = update_target_networks(agent_state, tau=tau)
        return agent_state, policy_aux

    def skip_policy(agent_state):
        zero = jnp.zeros(())
        return agent_state, PolicyAuxiliaries(
            policy_loss=zero, q_mean=zero, imitation_loss=zero, raw_loss=zero
        )

    agent_state, policy_aux = jax.lax.cond(
        do_policy, policy_and_targets, skip_policy, operand=agent_state
    )

    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)

    aux = AuxiliaryLogs(
        policy=PolicyAuxiliaries(
            policy_loss=policy_aux.policy_loss.flatten(),
            q_mean=policy_aux.q_mean.flatten(),
            imitation_loss=policy_aux.imitation_loss.flatten(),
            raw_loss=policy_aux.raw_loss.flatten(),
        ),
        value=ValueAuxiliaries(
            critic_loss=value_aux.critic_loss.flatten(),
            target_q=value_aux.target_q.flatten(),
            q_pred_min=value_aux.q_pred_min.flatten(),
        ),
    )
    return agent_state, aux


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
        "expert_policy",
        "imitation_coef",
        "distance_to_stable",
        "imitation_coef_offset",
        "action_scale",
        "action_pipeline",
        "eval_action_transform",
        "target_modifier",
        "obs_preprocessor",
        "policy_action_transform",
    ],
)
def training_iteration(
    agent_state: TD3State,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_config: TD3Config,
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
    expert_policy: Optional[Callable] = None,
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = get_one,
    imitation_coef_offset: float = 0.0,
    action_scale: float = 1.0,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    target_modifier: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
) -> tuple[TD3State, Any]:
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
            recurrent=recurrent,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
            policy_delay=agent_config.policy_delay,
            target_policy_noise=agent_config.target_policy_noise,
            target_noise_clip=agent_config.target_noise_clip,
            reward_scale=agent_config.reward_scale,
            expert_policy=expert_policy,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset,
            target_modifier=target_modifier,
            obs_preprocessor=obs_preprocessor,
            policy_action_transform=policy_action_transform,
        )
        agent_state, aux = jax.lax.scan(
            update_scan_fn, agent_state, xs=None, length=n_epochs
        )
        aux = jax.tree.map(lambda x: x[-1].reshape((1,)), aux)
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
        expert_policy=expert_policy,
        action_scale=action_scale,
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
    agent_config: TD3Config,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    target_modifier: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    recurrent = network_args.lstm_hidden_size is not None
    if action_pipeline is None:
        action_pipeline = make_default_action_pipeline(
            env_args=env_args,
            recurrent=recurrent,
            exploration_noise=agent_config.exploration_noise,
        )

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        init_key, expert_key = jax.random.split(key)
        agent_state = init_TD3(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            buffer=buffer,
            num_critics=agent_config.num_critics,
            pid_actor_config=pid_actor_config,
            expert_policy=expert_policy,
        )

        cloning_parameters, pre_train_n_steps = get_cloning_args(
            cloning_args, total_timesteps
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

        num_updates = total_timesteps // env_args.n_envs
        _, action_shape = get_state_action_shapes(env_args.env)

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=recurrent,
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
            expert_policy=expert_policy,
            action_pipeline=action_pipeline,
            eval_action_transform=eval_action_transform,
            target_modifier=target_modifier,
            obs_preprocessor=obs_preprocessor,
            policy_action_transform=policy_action_transform,
            **cloning_parameters,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train
