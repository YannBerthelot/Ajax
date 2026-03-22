from collections.abc import Sequence
from dataclasses import fields
from math import floor
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.cloning import (
    CloningConfig,
    get_cloning_args,
    get_pre_trained_agent,
)
from ajax.agents.SAC.state import SACConfig, SACState
from ajax.agents.SAC.utils import SquashedNormal
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    collect_experience_from_expert_policy,
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

# ---------------------------------------------------------------------------
# Auxiliary dataclasses for logging
# ---------------------------------------------------------------------------


@struct.dataclass
class TemperatureAuxiliaries:
    alpha: jax.Array
    log_alpha: jax.Array


@struct.dataclass
class PolicyAuxiliaries:
    # Core loss
    raw_loss: jax.Array  # α·log π - Q: pure SAC gradient
    policy_loss: jax.Array  # raw_loss + AWBC term

    # Entropy diagnostics
    log_pi: jax.Array  # entropy proxy; should track target_entropy
    policy_std: jax.Array  # mean unsquashed std; lower = more deterministic

    # Q-value diagnostics
    q_min: jax.Array  # Q(s, π(s)): what policy optimizes
    q_expert: jax.Array  # Q(s, a_expert): expert value estimate

    # AWBC diagnostics
    awbc_coef: jax.Array  # λ(s): AWBC pull strength
    nll_expert: jax.Array  # -log π(a_expert): behavioral distance to expert
    above_expert_frac: jax.Array  # fraction of batch where policy beats expert


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array  # min over ensemble
    q_expert_mean: jax.Array  # critic's estimate of expert value
    q_gap: jax.Array  # q_expert - q_min: >0 = room to improve
    var_preds: jax.Array  # inter-critic variance


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


# ---------------------------------------------------------------------------
# Scalar alpha (temperature)
# ---------------------------------------------------------------------------


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    log_alpha = jnp.log(alpha_init)
    params = FrozenDict({"log_alpha": log_alpha})
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate),
    )
    return TrainState.create(
        apply_fn=lambda params: jnp.exp(params["log_alpha"]),
        params=params,
        tx=tx,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def maybe_append_train_frac(
    obs: jax.Array,
    train_frac: Optional[float],
) -> jax.Array:
    if train_frac is None:
        return obs
    new_col = jnp.full((obs.shape[0], 1), train_frac)
    return jnp.concatenate([obs, new_col], axis=-1)


# ---------------------------------------------------------------------------
# Expert data collection and buffer pre-population
# ---------------------------------------------------------------------------


def collect_and_store_expert_transitions(
    expert_policy: Callable,
    env_args: EnvironmentConfig,
    buffer: BufferType,
    buffer_state: Any,
    rng: jax.Array,
    n_steps: int,
    max_timesteps: Optional[int] = None,
) -> Any:
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_steps,
    )

    flat = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]) if x is not None else None,
        transitions,
        is_leaf=lambda x: x is None,
    )

    buffer_obs_dim = buffer_state.experience["obs"].shape[-1]
    expert_obs_dim = flat.obs.shape[-1]
    flat_obs = maybe_append_train_frac(
        flat.obs,
        train_frac=0.0 if buffer_obs_dim == expert_obs_dim + 1 else None,
    )
    flat_raw_obs = flat.raw_obs if flat.raw_obs is not None else flat.obs

    n_total = flat_obs.shape[0]

    def add_one(buffer_state, i):
        take = lambda x: jnp.take(x, i, axis=0, mode="clip")[None]
        _transition = {
            "obs": take(flat_obs),
            "action": take(flat.action),
            "reward": take(flat.reward),
            "terminated": take(flat.terminated),
            "truncated": take(flat.truncated),
            "raw_obs": take(flat_raw_obs),
        }
        return buffer.add(buffer_state, _transition), None

    buffer_state, _ = jax.lax.scan(add_one, buffer_state, jnp.arange(n_total))
    return buffer_state


# ---------------------------------------------------------------------------
# SAC initialization
# ---------------------------------------------------------------------------


def init_SAC(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    window_size: int = 10,
    expert_policy: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    max_timesteps: Optional[int] = None,
    num_critics: int = 4,
    expert_buffer_n_steps: int = 20_000,
) -> SACState:
    rng, init_key, collector_key, expert_key = jax.random.split(key, num=4)

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
        expert_policy=expert_policy,
        residual=False,
        fixed_alpha=False,
        max_timesteps=max_timesteps,
    )

    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
        max_timesteps=max_timesteps,
    )

    if (
        expert_policy is not None
        and buffer is not None
        and collector_state.buffer_state is not None
        and expert_buffer_n_steps > 0
    ):
        collector_state = collector_state.replace(
            buffer_state=collect_and_store_expert_transitions(
                expert_policy=expert_policy,
                env_args=env_args,
                buffer=buffer,
                buffer_state=collector_state.buffer_state,
                rng=expert_key,
                n_steps=expert_buffer_n_steps,
                max_timesteps=max_timesteps,
            )
        )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    return SACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
        lambda_param=1.0,
    )


# ---------------------------------------------------------------------------
# Critic pre-training
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma", "reward_scale", "n_steps", "buffer"],
)
def pretrain_critic(
    agent_state: SACState,
    recurrent: bool,
    gamma: float,
    reward_scale: float,
    buffer: BufferType,
    n_steps: int = 5_000,
) -> SACState:
    """Pre-train critic on expert buffer data so Q(s,a_expert) is meaningful at step 1."""

    def critic_pretrain_step(carry, _):
        agent_state = carry
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
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, sample_key
        )
        dones = jnp.logical_or(terminated, truncated)
        agent_state, _ = update_value_functions(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            dones=dones,
            agent_state=agent_state,
            recurrent=recurrent,
            gamma=gamma,
            reward_scale=reward_scale,
        )
        agent_state = update_target_networks(agent_state, tau=5e-4)
        return agent_state, None

    agent_state, _ = jax.lax.scan(
        critic_pretrain_step, agent_state, None, length=n_steps
    )
    return agent_state


# ---------------------------------------------------------------------------
# Actor pre-training via behavioral cloning
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["recurrent", "n_steps", "buffer", "expert_policy"],
)
def pretrain_actor(
    agent_state: SACState,
    recurrent: bool,
    buffer: BufferType,
    expert_policy: Callable,
    n_steps: int = 2_000,
) -> SACState:
    """
    Pre-train actor via BC so π(s) ≈ a_expert at initialization.
    After BC: awbc_coef ≈ 0, policy starts at expert level and explores freely.
    Only beneficial when num_critic_updates ≥ 4 to prevent overshoot.
    """

    def bc_step(carry, _):
        agent_state = carry
        sample_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        (observations, _, _, _, _, _, raw_observations) = get_batch_from_buffer(
            buffer,
            agent_state.collector_state.buffer_state,
            sample_key,
        )

        def bc_loss_fn(actor_params):
            pi, _ = get_pi(
                actor_state=agent_state.actor_state,
                actor_params=actor_params,
                obs=observations,
                done=None,
                recurrent=recurrent,
            )
            a_expert = jax.lax.stop_gradient(expert_policy(raw_observations))
            return (-pi.log_prob(a_expert).sum(-1)).mean()

        loss, grads = jax.value_and_grad(bc_loss_fn)(agent_state.actor_state.params)
        return (
            agent_state.replace(
                actor_state=agent_state.actor_state.apply_gradients(grads=grads)
            ),
            loss,
        )

    agent_state, _ = jax.lax.scan(bc_step, agent_state, None, length=n_steps)
    return agent_state


# ---------------------------------------------------------------------------
# Critic update
# ---------------------------------------------------------------------------


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
    reward_scale: float = 1.0,
    expert_q: Optional[jax.Array] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    rewards = rewards * reward_scale

    next_pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=next_observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_params,
        x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
    )
    var_preds = q_preds.var(axis=0, keepdims=True)

    assert critic_states.target_params is not None
    q_targets = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.target_params,
        x=jnp.concatenate((next_observations, next_actions), axis=-1),
    )
    min_q_target = jnp.min(q_targets, axis=0, keepdims=False)

    target_q = jax.lax.stop_gradient(
        rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)
    )
    assert target_q.shape == q_preds.shape[1:]

    total_loss = jnp.mean((q_preds - target_q) ** 2)
    q_pred_min = jnp.min(q_preds, axis=0)

    q_expert_mean = expert_q.mean().flatten() if expert_q is not None else jnp.zeros(1)
    q_gap = (
        (expert_q - q_pred_min).mean().flatten()
        if expert_q is not None
        else jnp.zeros(1)
    )

    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q_pred_min=q_pred_min.mean().flatten(),
        q_expert_mean=q_expert_mean,
        q_gap=q_gap,
        var_preds=var_preds.mean().flatten(),
    )


@partial(jax.jit, static_argnames=["recurrent", "gamma", "reward_scale"])
def update_value_functions(
    agent_state: SACState,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float = 1.0,
    expert_q: Optional[jax.Array] = None,
) -> Tuple[SACState, ValueAuxiliaries]:
    value_loss_key, rng = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    (loss, aux), grads = jax.value_and_grad(value_loss_function, has_aux=True)(
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
        expert_q,
    )

    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, critic_state=updated_critic_state), aux


# ---------------------------------------------------------------------------
# Policy update — SAC + asymmetric AWBC
# ---------------------------------------------------------------------------


def compute_awbc_coef(
    q_expert: jax.Array,
    q_min: jax.Array,
    loss_actor: jax.Array,
    above_expert_coef: float = 0.0,
) -> jax.Array:
    """
    AWBC coefficient λ(s).

    Symmetric (above_expert_coef=0.0, default):
        Below expert: λ = relu(Q_expert - Q_min) / |L_actor|
        Above expert: λ = 0   → pure SAC

    Asymmetric (above_expert_coef > 0):
        Below expert: same Q-driven coefficient
        Above expert: λ = above_expert_coef (small constant, prevents drift)

    Asymmetry rationale: with ~200 pts headroom on 10000-pt task, drifting
    away from above-expert behavior is expensive. The constant anchors the
    policy without blocking improvement.
    """
    advantage = q_expert - q_min
    loss_scale = jax.lax.stop_gradient(jnp.abs(loss_actor).mean() + 1e-6)
    coef_below = jax.nn.relu(advantage) / loss_scale
    coef_above = jnp.full_like(coef_below, above_expert_coef)
    coef = jnp.where(advantage > 0, coef_below, coef_above)
    return jax.lax.stop_gradient(coef)


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "expert_policy",
        "use_expert_guidance",
        "above_expert_coef",
    ],
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
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    above_expert_coef: float = 0.0,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """
    SAC + AWBC policy loss.

    above_expert_coef=0.0 → symmetric AWBC (pure SAC when above expert)
    above_expert_coef>0.0 → asymmetric AWBC (small constant when above expert)
    use_expert_guidance=False → pure SAC regardless
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
    log_probs = log_probs.sum(-1, keepdims=True)

    policy_std = (
        pi.unsquashed_stddev().mean()
        if isinstance(pi, SquashedNormal)
        else pi.stddev().mean()
    )

    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.params,
        x=jnp.hstack((observations, actions)),
    )
    q_min = jnp.min(q_preds, axis=0, keepdims=True).squeeze(0)

    assert log_probs.shape == q_min.shape
    loss_actor = alpha * log_probs - q_min

    if expert_policy is not None and use_expert_guidance:
        _raw_obs = (
            raw_observations if raw_observations is not None else observations[..., :-1]
        )
        a_expert = jax.lax.stop_gradient(expert_policy(_raw_obs))
        q_expert = jnp.max(
            predict_value(
                critic_state=critic_states,
                critic_params=critic_states.params,
                x=jnp.hstack((observations, a_expert)),
            ),
            axis=0,
            keepdims=True,
        )

        nll_expert = -pi.log_prob(a_expert).sum(-1, keepdims=True)
        lambda_s = compute_awbc_coef(
            q_expert, q_min, loss_actor, above_expert_coef=above_expert_coef
        )
        above_expert_frac = jnp.mean((q_min >= q_expert).astype(jnp.float32))

        total_loss = (loss_actor + lambda_s * nll_expert).mean()
        awbc_coef_logged = lambda_s.mean()
        nll_expert_logged = nll_expert.mean()
        q_expert_logged = q_expert.mean()
    else:
        total_loss = loss_actor.mean()
        awbc_coef_logged = jnp.zeros(1)
        nll_expert_logged = jnp.zeros(1)
        q_expert_logged = jnp.zeros(1)
        above_expert_frac = jnp.zeros(1)

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_pi=log_probs.mean(),
        policy_std=policy_std,
        q_min=q_min.mean(),
        q_expert=q_expert_logged,
        awbc_coef=awbc_coef_logged,
        nll_expert=nll_expert_logged,
        above_expert_frac=above_expert_frac,
        raw_loss=loss_actor.mean(),
    )


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "expert_policy",
        "use_expert_guidance",
        "above_expert_coef",
    ],
)
def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
    raw_observations: jax.Array,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    above_expert_coef: float = 0.0,
) -> Tuple[SACState, PolicyAuxiliaries]:
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    (loss, aux), grads = jax.value_and_grad(
        policy_loss_function, has_aux=True, argnums=0
    )(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        done,
        recurrent,
        alpha,
        policy_key,
        raw_observations=raw_observations,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        above_expert_coef=above_expert_coef,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, actor_state=updated_actor_state), aux


# ---------------------------------------------------------------------------
# Temperature update with adaptive target entropy
# ---------------------------------------------------------------------------


# @partial(jax.jit, static_argnames=["target_entropy"])
def temperature_loss_function(
    log_alpha_params: FrozenDict,
    corrected_log_probs: jax.Array,
    target_entropy: float,
) -> Tuple[jax.Array, TemperatureAuxiliaries]:
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (
        log_alpha * jax.lax.stop_gradient(-corrected_log_probs - target_entropy)
    ).mean()
    return loss, TemperatureAuxiliaries(alpha=alpha, log_alpha=log_alpha)


@partial(
    jax.jit,
    static_argnames=["target_entropy", "recurrent", "above_expert_entropy_scale"],
)
def update_temperature(
    agent_state: SACState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    target_entropy: float,
    recurrent: bool,
    above_expert: Optional[jax.Array] = None,
    above_expert_entropy_scale: float = 1.0,
) -> Tuple[SACState, TemperatureAuxiliaries]:
    """
    Adaptive target entropy: scaled down when policy beats the expert.
    above_expert_entropy_scale=1.0 → standard SAC (no adaptation)
    above_expert_entropy_scale<1.0 → lower target entropy when above expert
                                     (more deterministic = more precise approach)
    """
    pi, _ = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=observations,
        done=dones,
        recurrent=recurrent,
    )
    rng, sample_key = jax.random.split(agent_state.rng)
    _, log_probs = pi.sample_and_log_prob(seed=sample_key)

    effective_target = jnp.where(
        above_expert if above_expert is not None else jnp.array(False),
        target_entropy * above_expert_entropy_scale,
        target_entropy,
    )

    (loss, aux), grads = jax.value_and_grad(temperature_loss_function, has_aux=True)(
        agent_state.alpha.params,
        log_probs.sum(-1),
        effective_target,
    )

    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, alpha=new_alpha_state), jax.lax.stop_gradient(
        aux
    )


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["tau"])
def update_target_networks(agent_state: SACState, tau: float) -> SACState:
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau)
    )


# ---------------------------------------------------------------------------
# Agent update (one gradient step)
# ---------------------------------------------------------------------------


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
        "transition_mix_fraction",
        "expert_policy",
        "use_expert_guidance",
        "target_entropy",
        "policy_update_start",
        "alpha_update_start",
        "expert_mix_fraction",
        "above_expert_coef",
        "above_expert_entropy_scale",
    ],
)
def update_agent(
    agent_state: SACState,
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    target_entropy: float,
    tau: float,
    num_critic_updates: int = 1,
    reward_scale: float = 1.0,
    additional_transition: Optional[Any] = None,
    transition_mix_fraction: float = 1.0,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    policy_update_start: int = 20_000,
    alpha_update_start: int = 20_000,
    expert_mix_fraction: float = 0.1,
    above_expert_coef: float = 0.0,
    above_expert_entropy_scale: float = 1.0,
) -> Tuple[SACState, AuxiliaryLogs]:
    sample_key, expert_sample_key, rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)

    # --- Sample from buffer ---
    if buffer is not None and agent_state.collector_state.buffer_state is not None:
        (
            observations,
            terminated,
            truncated,
            next_observations,
            rewards,
            actions,
            raw_observations,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, sample_key
        )
        original_transition = Transition(
            observations,
            actions,
            rewards,
            terminated,
            truncated,
            next_observations,
            raw_obs=raw_observations,
        )

        if additional_transition is not None and transition_mix_fraction < 1.0:
            len_original = len(observations)
            n_from_buffer = floor(transition_mix_fraction * len_original)
            n_from_online = len_original - n_from_buffer
            additional_transition = jax.tree.map(
                lambda x: jax.random.choice(sample_key, x, shape=(n_from_online,)),
                additional_transition,
            )
            transition = jax.tree.map(
                lambda x, y: (
                    None
                    if (x is None or y is None)
                    else jnp.concatenate([x[:n_from_buffer], y], axis=0)
                ),
                original_transition,
                additional_transition,
                is_leaf=lambda x: x is None,
            )
        else:
            transition = original_transition

    elif additional_transition is not None:
        transition = additional_transition
    else:
        raise ValueError("Either buffer or additional_transition must be provided.")

    # --- Expert batch mixing ---
    if expert_mix_fraction > 0.0 and expert_policy is not None:
        (
            exp_obs,
            exp_terminated,
            exp_truncated,
            exp_next_obs,
            exp_rewards,
            exp_actions,
            exp_raw_obs,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, expert_sample_key
        )

        n_total = transition.obs.shape[0]
        n_expert = floor(expert_mix_fraction * n_total)
        n_online = n_total - n_expert

        def _cat(a, b):
            if a is None or b is None:
                return a
            return jnp.concatenate([a[:n_online], b[:n_expert]], axis=0)

        transition = Transition(
            obs=_cat(transition.obs, exp_obs),
            action=_cat(transition.action, exp_actions),
            reward=_cat(transition.reward, exp_rewards),
            terminated=_cat(transition.terminated, exp_terminated),
            truncated=_cat(transition.truncated, exp_truncated),
            next_obs=_cat(transition.next_obs, exp_next_obs),
            raw_obs=_cat(transition.raw_obs, exp_raw_obs),
        )

    dones = jnp.logical_or(transition.terminated, transition.truncated)

    # --- Pre-compute Q(s, a_expert) once for critic logging + AWBC ---
    expert_q = None
    if expert_policy is not None and use_expert_guidance:
        _raw = (
            transition.raw_obs
            if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        a_expert = jax.lax.stop_gradient(expert_policy(_raw))
        expert_q = jax.lax.stop_gradient(
            jnp.max(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.critic_state.params,
                    x=jnp.hstack((transition.obs, a_expert)),
                ),
                axis=0,
            )
        )

    # --- Critic updates ---
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
            expert_q=expert_q,
        )
        return agent_state, aux_value

    agent_state, aux_value_seq = jax.lax.scan(
        critic_update_step, agent_state, None, length=num_critic_updates
    )
    aux_value = jax.tree.map(lambda x: x[-1], aux_value_seq)

    # --- Policy update ---
    new_agent_state, aux_policy = update_policy(
        observations=transition.obs,
        done=dones,
        agent_state=agent_state,
        recurrent=recurrent,
        raw_observations=transition.raw_obs,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        above_expert_coef=above_expert_coef,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= policy_update_start,
        lambda: new_agent_state,
        lambda: agent_state,
    )

    # --- Temperature update with optional adaptive entropy ---
    above_expert_flag = (
        aux_value.q_pred_min.mean() >= aux_value.q_expert_mean.mean()
        if expert_q is not None
        else jnp.array(False)
    )
    new_agent_state_temp, aux_temperature = update_temperature(
        agent_state,
        observations=transition.obs,
        target_entropy=target_entropy,
        recurrent=recurrent,
        dones=dones,
        above_expert=above_expert_flag,
        above_expert_entropy_scale=above_expert_entropy_scale,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= alpha_update_start,
        lambda: new_agent_state_temp,
        lambda: agent_state,
    )

    agent_state = update_target_networks(agent_state, tau=tau)

    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=ValueAuxiliaries(
            **{k: v.flatten() for k, v in to_state_dict(aux_value).items()}
        ),
    )
    return agent_state, aux


# ---------------------------------------------------------------------------
# Training iteration
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
        "transition_mix_fraction",
        "expert_policy",
        "use_expert_guidance",
        "action_scale",
        "early_termination_condition",
        "num_critic_updates",
        "above_expert_coef",
        "above_expert_entropy_scale",
        "expert_mix_fraction",
        "distance_to_stable",
        "imitation_coef_offset",
        "imitation_coef",
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
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    action_scale: float = 1.0,
    early_termination_condition: Optional[Callable] = None,
    num_critic_updates: int = 1,
    above_expert_coef: float = 0.0,
    above_expert_entropy_scale: float = 1.0,
    expert_mix_fraction: float = 0.1,
    # API compat
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = lambda x: 1.0,
    imitation_coef_offset: float = 0.0,
) -> tuple[SACState, None]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
        expert_policy=expert_policy,
        action_scale=action_scale,
        always_expert=uniform,
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
            target_entropy=agent_config.target_entropy,
            tau=agent_config.tau,
            reward_scale=agent_config.reward_scale,
            additional_transition=(
                jax.tree.map(lambda x: x.squeeze(0), transition)
                if transition_mix_fraction < 1.0
                else None
            ),
            transition_mix_fraction=transition_mix_fraction,
            expert_policy=expert_policy,
            use_expert_guidance=use_expert_guidance,
            num_critic_updates=num_critic_updates,
            above_expert_coef=above_expert_coef,
            above_expert_entropy_scale=above_expert_entropy_scale,
            expert_mix_fraction=expert_mix_fraction,
        )
        agent_state, aux = jax.lax.scan(
            update_scan_fn, agent_state, xs=None, length=n_epochs
        )
        aux = jax.tree.map(lambda x: x[-1].reshape((1,)), aux)
        aux = aux.replace(
            value=ValueAuxiliaries(
                **{k: v.flatten() for k, v in to_state_dict(aux.value).items()}
            )
        )
        return agent_state, aux

    def fill_with_nan(dataclass):
        nan = jnp.ones(1) * jnp.nan
        result = {}
        for field in fields(dataclass):
            sub = field.type
            if hasattr(sub, "__dataclass_fields__"):
                result[field.name] = fill_with_nan(sub)
            else:
                result[field.name] = nan
        return dataclass(**result)

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
        early_termination_condition=early_termination_condition,
        train_frac=agent_state.collector_state.train_time_fraction,
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
    agent_config: SACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    early_termination_condition: Optional[Callable] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    num_critics: int = 4,
    expert_buffer_n_steps: int = 20_000,
    critic_pretrain_steps: int = 5_000,
    actor_pretrain_steps: int = 0,
    num_critic_updates: int = 1,
    above_expert_coef: float = 0.0,
    above_expert_entropy_scale: float = 1.0,
    expert_mix_fraction: float = 0.1,
):
    """
    SAC + AWBC training factory.

    Key parameters:
      num_critic_updates: critic gradient steps per env step (default 1, try 4)
      actor_pretrain_steps: BC pre-training steps (0 = disabled)
      critic_pretrain_steps: critic pre-training on expert buffer
      expert_buffer_n_steps: expert transitions pre-loaded into buffer
      above_expert_coef: asymmetric AWBC constant above expert (0.0 = symmetric)
      above_expert_entropy_scale: entropy target scale when above expert (1.0 = off)
      expert_mix_fraction: fraction of each batch from expert buffer
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        init_key, expert_key = jax.random.split(key)

        agent_state = init_SAC(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
            expert_policy=expert_policy,
            max_timesteps=total_timesteps,
            num_critics=num_critics,
            expert_buffer_n_steps=(
                expert_buffer_n_steps if expert_policy is not None else 0
            ),
        )

        if expert_policy is not None and critic_pretrain_steps > 0:
            agent_state = pretrain_critic(
                agent_state=agent_state,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                buffer=buffer,
                n_steps=critic_pretrain_steps,
            )

        if expert_policy is not None and actor_pretrain_steps > 0:
            agent_state = pretrain_actor(
                agent_state=agent_state,
                recurrent=network_args.lstm_hidden_size is not None,
                buffer=buffer,
                expert_policy=expert_policy,
                n_steps=actor_pretrain_steps,
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

        _valid_cloning_params = {
            k: v
            for k, v in cloning_parameters.items()
            if k
            in (
                "n_epochs",
                "transition_mix_fraction",
                "imitation_coef",
                "distance_to_stable",
                "imitation_coef_offset",
            )
        }

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
            expert_policy=expert_policy,
            use_expert_guidance=use_expert_guidance,
            early_termination_condition=early_termination_condition,
            num_critic_updates=num_critic_updates,
            above_expert_coef=above_expert_coef,
            above_expert_entropy_scale=above_expert_entropy_scale,
            expert_mix_fraction=expert_mix_fraction,
            **_valid_cloning_params,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train
