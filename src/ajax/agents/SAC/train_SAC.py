from collections.abc import Sequence
from dataclasses import fields
from math import floor
import time
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
from ajax.environments.utils import check_env_is_gymnax, get_action_dim, get_state_action_shapes
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.networks.networks import (
    get_adam_tx,
    get_initialized_actor_critic,
    get_initialized_critic,
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
    raw_loss: jax.Array           # α·log π - Q: pure SAC gradient
    policy_loss: jax.Array        # raw_loss + AWBC + value constraint terms

    # Entropy diagnostics
    log_pi: jax.Array             # entropy proxy; tracks target_entropy
    policy_std: jax.Array         # mean unsquashed std; lower = more deterministic

    # Q-value diagnostics
    q_min: jax.Array              # Q(s, π(s)): what policy optimises
    q_expert: jax.Array           # Q(s, a_expert): expert value estimate

    # AWBC diagnostics
    awbc_coef: jax.Array          # λ(s): AWBC pull strength (0 = pure SAC)
    l2_expert: jax.Array          # ||π(s) - a_expert||^2: L2 distance to expert action
    above_expert_frac: jax.Array  # fraction of batch where policy beats expert

    # Online decaying BC term
    bc_term: jax.Array            # decaying online BC loss magnitude (0 after warmup_frac)

    # Policy behavior KPIs (from raw_obs — tell us what the policy actually does)
    altitude_error: jax.Array     # mean |z - target| over batch
    z_dot_mean: jax.Array         # mean |z_dot| over batch: 0 = stable, high = aggressive


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array           # min over ensemble
    q_expert_mean: jax.Array        # critic's estimate of expert value
    q_gap: jax.Array                # q_expert - q_min: >0 = room to improve
    var_preds: jax.Array            # inter-critic variance
    alpha_blend: jax.Array          # current blend coefficient (1=pure expert, 0=pure Bellman)
    effective_threshold: jax.Array  # box threshold at current train_frac
    box_entry_rate: jax.Array       # fraction of batch inside value box


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


@struct.dataclass
class MCPretrainAux:
    """Diagnostics logged after MC critic pretraining."""
    initial_loss: jax.Array   # MSE at start of regression
    final_loss: jax.Array     # MSE at end of regression — should be much lower
    q_expert_mean: jax.Array  # mean Q(s, a_expert) on last batch after pretraining
    q_expert_min: jax.Array   # min  Q(s, a_expert) — sanity: should be negative far from target
    q_expert_max: jax.Array   # max  Q(s, a_expert) — should be near 0 (dense reward ≤ 0)
    v_min: jax.Array          # global min of Q(s, a_expert) over last batch
    v_max: jax.Array          # global max of Q(s, a_expert) over last batch


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
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    augment_obs_with_expert_action: bool = False,
) -> SACState:
    rng, init_key, collector_key, expert_key = jax.random.split(key, num=4)

    # When augment_obs_with_expert_action=True, the actor and critic receive
    # obs augmented with a_expert at runtime (action_dim extra dimensions).
    # We must initialise the networks with the matching inflated input size.
    if augment_obs_with_expert_action:
        _, action_shape = get_state_action_shapes(env_args.env)
        extra_obs_dim = action_shape[0]
    else:
        extra_obs_dim = 0

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
        extra_obs_dim=extra_obs_dim,
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
# Critic pre-training via Monte Carlo returns
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma", "reward_scale", "n_steps",
                    "expert_policy", "n_mc_steps", "n_mc_episodes",
                    "mode", "env_args", "max_timesteps", "batch_size",
                    "augment_obs_with_expert_action"],
)
def pretrain_critic_mc(
    agent_state: SACState,
    expert_critic_state: LoadedTrainState,
    expert_policy: Callable,
    mode: str,
    env_args: EnvironmentConfig,
    recurrent: bool,
    gamma: float,
    reward_scale: float,
    n_mc_steps: int = 10_000,
    n_mc_episodes: int = 500,
    n_steps: int = 5_000,
    batch_size: int = 256,
    max_timesteps: Optional[int] = None,
    augment_obs_with_expert_action: bool = False,
) -> Tuple[SACState, FrozenDict, jax.Array, jax.Array, "MCPretrainAux"]:
    """
    Pre-train critic using Monte Carlo returns from expert trajectories.

    Unlike Bellman pretraining (bootstraps from an untrained critic → biased),
    MC returns G_t = Σ γ^k r_{t+k} are unbiased estimates of V^expert(s).
    The critic starts with accurate Q(s, a_expert) near the target from step 1.

    Collection strategy: single call with n_mc_steps * n_mc_episodes // n_envs
    timesteps. The n_envs parallel environments reset to different (initial,
    target) altitude pairs on each episode boundary, giving the same state-space
    coverage as separate per-seed rollouts — without any mapping over traced keys
    (which fails inside the outer vmap over seeds).
    Total transitions ≈ n_mc_steps * n_mc_episodes regardless of n_envs.
    """
    # --- Single collection call: n_envs envs reset independently → diversity ---
    n_total_steps = max(1, (n_mc_steps * n_mc_episodes) // env_args.n_envs)
    all_transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=agent_state.rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_total_steps,
    )
    # Shape: (n_total_steps, n_envs, ...)

    # --- Compute MC returns with a single backward scan ---
    # Scan over time axis; n_envs are handled as a batch dim in the carry.
    rewards = all_transitions.reward * reward_scale  # (T, n_envs, 1)
    dones = jnp.logical_or(
        all_transitions.terminated, all_transitions.truncated
    ).astype(jnp.float32)                             # (T, n_envs, 1)

    def mc_scan(carry, x):
        reward, done = x
        mc_return = reward + gamma * carry * (1.0 - done)
        return mc_return, mc_return

    _, mc_returns = jax.lax.scan(
        mc_scan,
        jnp.zeros_like(rewards[0]),     # carry: (n_envs, 1)
        (rewards[::-1], dones[::-1]),   # scan backwards over time
    )
    mc_returns = mc_returns[::-1]       # (T, n_envs, 1)

    # --- Flatten time × envs into one dataset ---
    T, n_envs = rewards.shape[:2]
    obs_flat    = all_transitions.obs.reshape(T * n_envs, -1)
    action_flat = all_transitions.action.reshape(T * n_envs, -1)
    mc_flat     = mc_returns.reshape(T * n_envs, 1)

    # Append train_frac=0.0 if max_timesteps was set
    if max_timesteps is not None:
        obs_flat = jnp.concatenate(
            [obs_flat, jnp.zeros((obs_flat.shape[0], 1))], axis=-1
        )

    # Augment obs with expert action if enabled — must match network input dim
    # Layout: [env_obs | a_expert | train_frac]
    if augment_obs_with_expert_action:
        raw_flat = all_transitions.obs.reshape(T * n_envs, -1)
        a_expert_flat = jax.lax.stop_gradient(expert_policy(raw_flat))
        if max_timesteps is not None:
            # obs_flat is [env_obs | train_frac] — insert a_expert before train_frac
            obs_flat = jnp.concatenate(
                [obs_flat[..., :-1], a_expert_flat, obs_flat[..., -1:]], axis=-1
            )
        else:
            obs_flat = jnp.concatenate([obs_flat, a_expert_flat], axis=-1)

    # --- Batch into fixed-size chunks for regression ---
    n_total   = T * n_envs
    n_batches = n_total // batch_size
    obs_batched    = obs_flat[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    action_batched = action_flat[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    mc_batched     = mc_flat[:n_batches * batch_size].reshape(n_batches, batch_size, 1)

    # --- Supervised regression: Q(s, a_expert) → MC return ---
    # All steps operate on expert_critic_state only; agent_state is never touched.
    def mc_loss_fn(critic_params, obs, actions, targets):
        q_preds = predict_value(
            critic_state=expert_critic_state,
            critic_params=critic_params,
            x=jnp.concatenate([obs, actions], axis=-1),
        )
        return jnp.mean((q_preds - targets) ** 2)

    def regression_step(carry, batch):
        expert_critic_state, step = carry
        obs_b, action_b, mc_b = batch
        loss, grads = jax.value_and_grad(mc_loss_fn)(
            expert_critic_state.params, obs_b, action_b, mc_b,
        )
        new_expert_critic_state = expert_critic_state.apply_gradients(grads=grads)
        return (new_expert_critic_state, step + 1), loss

    # Initial loss for comparison
    initial_loss, _ = jax.value_and_grad(mc_loss_fn)(
        expert_critic_state.params,
        obs_batched[0], action_batched[0], mc_batched[0],
    )

    n_passes = max(1, n_steps // n_batches)
    batches  = (obs_batched, action_batched, mc_batched)

    def one_pass(carry, _):
        return jax.lax.scan(regression_step, carry, batches)

    (expert_critic_state, _), loss_history = jax.lax.scan(
        one_pass, (expert_critic_state, 0), None, length=n_passes
    )
    final_loss = loss_history[-1, -1]

    # Hard sync target network of expert critic only
    expert_critic_state = expert_critic_state.soft_update(tau=1.0)

    # Freeze expert params — agent_state is returned completely unchanged
    frozen_expert_params = jax.lax.stop_gradient(expert_critic_state.params)

    # Q-value diagnostics on the last batch
    q_preds_final = predict_value(
        critic_state=expert_critic_state,
        critic_params=frozen_expert_params,
        x=jnp.concatenate([obs_batched[-1], action_batched[-1]], axis=-1),
    )
    q_for_stats = jnp.min(q_preds_final, axis=0)  # min over ensemble → (batch, 1)
    v_min = q_for_stats.min()
    v_max = q_for_stats.max()

    return agent_state, frozen_expert_params, obs_batched, action_batched, MCPretrainAux(
        initial_loss=initial_loss,
        final_loss=final_loss,
        q_expert_mean=q_for_stats.mean(),
        q_expert_min=q_for_stats.min(),
        q_expert_max=q_for_stats.max(),
        v_min=v_min,
        v_max=v_max,
    )


# ---------------------------------------------------------------------------
# Online critic light pre-regression (weak supervised nudge toward φ*)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["n_steps", "lr_scale"])
def pretrain_critic_online_light(
    agent_state: SACState,
    obs_batched: jax.Array,      # reuse MC pretrain dataset
    action_batched: jax.Array,
    n_steps: int = 500,
    lr_scale: float = 0.1,
) -> SACState:
    """
    Weak supervised nudge of the online critic toward φ*.
    Goal: reduce seed-to-seed variance at initialization.
    NOT a full MC pretrain — just reduces starting point spread.
    Target network stays random and untouched.
    """
    def light_loss_fn(critic_params, obs, actions):
        # Target from frozen expert critic
        v_expert = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([obs, actions], axis=-1),
                ), axis=0,
            )
        )
        # Online critic prediction
        q_online = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=critic_params,
            x=jnp.concatenate([obs, actions], axis=-1),
        )
        return jnp.mean((q_online - v_expert) ** 2)

    def step(carry, batch):
        agent_state = carry
        obs_b, action_b = batch
        loss, grads = jax.value_and_grad(light_loss_fn)(
            agent_state.critic_state.params, obs_b, action_b
        )
        # Scale gradients down — weak signal only
        grads = jax.tree.map(lambda g: g * lr_scale, grads)
        new_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
        return agent_state.replace(critic_state=new_critic_state), loss

    agent_state, _ = jax.lax.scan(
        step, agent_state, (obs_batched[:n_steps], action_batched[:n_steps])
    )
    # Target network intentionally untouched
    return agent_state


# ---------------------------------------------------------------------------
# Actor pre-training via value-weighted behavioral cloning
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["n_steps", "recurrent"],
)
def pretrain_actor_weighted_bc(
    agent_state: SACState,
    obs_batched: jax.Array,     # (n_batches, batch_size, obs_dim) — from MC pretrain
    action_batched: jax.Array,  # (n_batches, batch_size, action_dim) — expert actions
    n_steps: int = 5_000,
    recurrent: bool = False,
) -> SACState:
    """
    Value-weighted behavioral cloning on the actor.

    Loss: mean_s [w(s) * ||μ_θ(s) - π*(s)||²]
    w(s) = clip((V*(s) - V_min) / (V_max - V_min), 0, 1)
    V*(s) = min_k Q_φ*(s, π*(s)) using frozen expert_critic_params

    Only actor_state is modified; all other agent_state fields are unchanged.
    μ_θ(s) = tanh(loc) — the mode of the SquashedNormal distribution.
    """
    expert_critic_params = agent_state.expert_critic_params
    v_min = agent_state.expert_v_min
    v_max = agent_state.expert_v_max
    n_batches = obs_batched.shape[0]

    def actor_bc_loss_fn(actor_params, obs, a_expert):
        pi, _ = get_pi(
            actor_state=agent_state.actor_state,
            actor_params=actor_params,
            obs=obs, done=None, recurrent=recurrent,
        )
        # Mode of SquashedNormal: tanh(loc)
        mu = jnp.tanh(pi.distribution.loc)

        # V*(s) = min_k Q_φ*(s, a_expert) using frozen expert critic
        v_star = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=expert_critic_params,
                    x=jnp.concatenate([obs, a_expert], axis=-1),
                ),
                axis=0,
            )
        )
        w = jnp.clip((v_star - v_min) / (v_max - v_min + 1e-8), 0.0, 1.0)
        w = w ** 2  # sharper concentration near setpoint
        l2 = jnp.sum((mu - a_expert) ** 2, axis=-1, keepdims=True)
        return jnp.mean(w * l2)

    def bc_step(carry, batch):
        actor_state = carry
        obs_b, action_b = batch
        loss, grads = jax.value_and_grad(actor_bc_loss_fn)(
            actor_state.params, obs_b, action_b,
        )
        return actor_state.apply_gradients(grads=grads), loss

    n_passes = max(1, n_steps // n_batches)
    batches = (obs_batched, action_batched)

    def one_pass(carry, _):
        return jax.lax.scan(bc_step, carry, batches)

    final_actor_state, _ = jax.lax.scan(
        one_pass, agent_state.actor_state, None, length=n_passes,
    )
    return agent_state.replace(actor_state=final_actor_state)


# ---------------------------------------------------------------------------
# Critic pre-training via Bellman (legacy fallback)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma", "reward_scale", "n_steps", "buffer"],
)
def pretrain_critic_bellman(
    agent_state: SACState,
    recurrent: bool,
    gamma: float,
    reward_scale: float,
    buffer: BufferType,
    n_steps: int = 5_000,
) -> SACState:
    """Bellman-bootstrapped critic pretraining on the expert buffer."""
    def critic_pretrain_step(carry, _):
        agent_state = carry
        sample_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        (
            observations, terminated, truncated, next_observations,
            rewards, actions, raw_observations,
        ) = get_batch_from_buffer(buffer, agent_state.collector_state.buffer_state, sample_key)
        dones = jnp.logical_or(terminated, truncated)
        agent_state, _ = update_value_functions(
            observations=observations, actions=actions,
            next_observations=next_observations, rewards=rewards,
            dones=dones, agent_state=agent_state, recurrent=recurrent,
            gamma=gamma, reward_scale=reward_scale,
        )
        agent_state = update_target_networks(agent_state, tau=5e-4)
        return agent_state, None

    agent_state, _ = jax.lax.scan(critic_pretrain_step, agent_state, None, length=n_steps)
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
    v_expert_next: Optional[jax.Array] = None,
    alpha_blend: Optional[jax.Array] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    rewards = rewards * reward_scale

    next_pi, _ = get_pi(
        actor_state=actor_state, actor_params=actor_state.params,
        obs=next_observations, done=dones, recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    q_preds = predict_value(
        critic_state=critic_states, critic_params=critic_params,
        x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
    )
    var_preds = q_preds.var(axis=0, keepdims=True)

    assert critic_states.target_params is not None
    q_targets = predict_value(
        critic_state=critic_states, critic_params=critic_states.target_params,
        x=jnp.concatenate((next_observations, next_actions), axis=-1),
    )
    min_q_target = jnp.min(q_targets, axis=0, keepdims=False)

    y_bellman = rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)

    # Blended Bellman target: (1-α_blend)*y_bellman + α_blend*V*(s')
    # α_blend decays 1→0 over critic_warmup_frac of training.
    # When v_expert_next is None (critic blend disabled), pure Bellman.
    if v_expert_next is not None:
        y = (1.0 - alpha_blend) * y_bellman + alpha_blend * v_expert_next
        alpha_blend_logged = alpha_blend.mean().flatten()
    else:
        y = y_bellman
        alpha_blend_logged = jnp.zeros(1)

    target_q = jax.lax.stop_gradient(y)
    assert target_q.shape == q_preds.shape[1:]

    total_loss = jnp.mean((q_preds - target_q) ** 2)
    q_pred_min = jnp.min(q_preds, axis=0)

    q_expert_mean = expert_q.mean().flatten() if expert_q is not None else jnp.zeros(1)
    q_gap = (expert_q - q_pred_min).mean().flatten() if expert_q is not None else jnp.zeros(1)

    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        q_pred_min=q_pred_min.mean().flatten(),
        q_expert_mean=q_expert_mean,
        q_gap=q_gap,
        var_preds=var_preds.mean().flatten(),
        alpha_blend=alpha_blend_logged,
        effective_threshold=jnp.zeros(1),
        box_entry_rate=jnp.zeros(1),
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
    v_expert_next: Optional[jax.Array] = None,
    alpha_blend: Optional[jax.Array] = None,
) -> Tuple[SACState, ValueAuxiliaries]:
    value_loss_key, rng = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    (loss, aux), grads = jax.value_and_grad(value_loss_function, has_aux=True)(
        agent_state.critic_state.params, agent_state.critic_state, value_loss_key,
        agent_state.actor_state, actions, observations, next_observations,
        dones, rewards, gamma, alpha, recurrent, reward_scale, expert_q,
        v_expert_next, alpha_blend,
    )

    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, critic_state=updated_critic_state), aux


# ---------------------------------------------------------------------------
# Policy update — SAC + AWBC + value constraint
# ---------------------------------------------------------------------------


def compute_awbc_coef(
    q_expert: jax.Array,
    q_min: jax.Array,
    normalize: bool = True,
    use_relu: bool = True,
    fixed_lambda: Optional[float] = None,
) -> jax.Array:
    """AWBC coefficient λ(s).

    Ablation axes (all controlled by static Python flags):
        fixed_lambda: bypass adaptive gating with a constant scalar
        use_relu:     False → raw (Q* - Q_π) difference, no self-annealing
        normalize:    False → skip |q_min| denominator
    """
    if fixed_lambda is not None:
        return jax.lax.stop_gradient(jnp.full_like(q_expert, fixed_lambda))
    gap = jax.nn.relu(q_expert - q_min) if use_relu else (q_expert - q_min)
    if normalize:
        loss_scale = jax.lax.stop_gradient(jnp.abs(q_min).mean() + 1e-6)
        return jax.lax.stop_gradient(gap / loss_scale)
    return jax.lax.stop_gradient(gap)


def augment_obs_if_needed(
    observations: jax.Array,
    raw_observations: jax.Array,
    expert_policy,
    augment: bool,
) -> jax.Array:
    """Append expert action to obs at runtime. Layout: [env_obs | a_expert | train_frac]."""
    if not augment or expert_policy is None:
        return observations
    a_expert = jax.lax.stop_gradient(expert_policy(raw_observations))
    return jnp.concatenate([observations[..., :-1], a_expert, observations[..., -1:]], axis=-1)


@partial(
    jax.jit,
    static_argnames=["recurrent", "expert_policy", "use_expert_guidance",
                    "altitude_obs_idx", "target_obs_idx",
                    "augment_obs_with_expert_action",
                    "proximity_scale",
                    "awbc_normalize", "awbc_use_relu", "fixed_awbc_lambda",
                    "detach_obs_aug_action", "critic_warmup_frac"],
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
    box_threshold: float = 500.0,
    proximity_scale: Optional[float] = None,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    a_expert_precomputed: Optional[jax.Array] = None,
    awbc_normalize: bool = True,
    awbc_use_relu: bool = True,
    fixed_awbc_lambda: Optional[float] = None,
    detach_obs_aug_action: bool = False,
    inside_box_mask: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    expert_critic_params: Optional[Any] = None,
    expert_v_min: Optional[jax.Array] = None,
    expert_v_max: Optional[jax.Array] = None,
    critic_warmup_frac: float = 0.15,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """
    SAC actor loss with optional expert guidance.
    1. AWBC: action imitation, self-anneals when policy >= expert
    2. Value constraint: value floor, action-agnostic
    3. Obs augmentation: expert action hint in obs
    """
    _raw_obs = raw_observations if raw_observations is not None else observations[..., :-1]
    # observations is already augmented when augment_obs_with_expert_action=True
    # (augmentation happens in update_agent before any network calls)

    # obs_aug_detached_action: stop-gradient through expert-action dims in obs
    # so the actor can read the hint but its parameters are not updated to exploit it.
    # Layout: [env_obs | a_expert | train_frac]; a_expert_precomputed.shape[-1] gives action_dim.
    if augment_obs_with_expert_action and detach_obs_aug_action and a_expert_precomputed is not None:
        _ad = a_expert_precomputed.shape[-1]
        obs_for_actor = jnp.concatenate([
            observations[..., :-(_ad + 1)],
            jax.lax.stop_gradient(observations[..., -(_ad + 1):-1]),
            observations[..., -1:],
        ], axis=-1)
    else:
        obs_for_actor = observations

    pi, _ = get_pi(actor_state=actor_state, actor_params=actor_params,
                obs=obs_for_actor, done=dones, recurrent=recurrent)
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    policy_std = (
        pi.unsquashed_stddev().mean() if isinstance(pi, SquashedNormal) else pi.stddev().mean()
    )
    q_preds = predict_value(
        critic_state=critic_states, critic_params=critic_states.params,
        x=jnp.concatenate([observations, actions], axis=-1),
    )
    q_min = jnp.min(q_preds, axis=0)
    assert log_probs.shape == q_min.shape
    loss_actor = alpha * log_probs - q_min

    # Behavior KPIs from raw obs
    # env obs layout: [x_dot(0), z(1), z_dot(2), theta(3), theta_dot(4),
    #                  gamma(5), target_altitude(6), power(7), stick(8)]
    altitude_error_val = jnp.abs(_raw_obs[..., altitude_obs_idx] - _raw_obs[..., target_obs_idx]).mean()
    z_dot_mean_val     = jnp.abs(_raw_obs[..., 2]).mean()

    needs_expert = expert_policy is not None and use_expert_guidance

    if needs_expert:
        a_expert = (
            a_expert_precomputed if a_expert_precomputed is not None
            else jax.lax.stop_gradient(expert_policy(_raw_obs))
        )
        q_expert = jnp.min(
            predict_value(
                critic_state=critic_states, critic_params=critic_states.params,
                x=jnp.concatenate([observations, a_expert], axis=-1),
            ), axis=0,
        )
        above_expert_frac = jnp.mean((q_min >= q_expert).astype(jnp.float32))
        q_expert_logged   = q_expert.mean()

        # 1. AWBC
        if use_expert_guidance:
            # SquashedNormal = Normal → tanh; mode of Normal is loc, so mode after squash is tanh(loc)
            l2_expert = jnp.sum((jnp.tanh(pi.distribution.loc) - a_expert) ** 2, axis=-1, keepdims=True)
            lambda_s   = compute_awbc_coef(
                q_expert, q_min,
                normalize=awbc_normalize, use_relu=awbc_use_relu,
                fixed_lambda=fixed_awbc_lambda,
            )
            if proximity_scale is not None:
                dist_norm        = jnp.abs(
                    _raw_obs[..., altitude_obs_idx:altitude_obs_idx + 1] -
                    _raw_obs[..., target_obs_idx:target_obs_idx + 1]
                ) / box_threshold
                proximity_weight = jnp.exp(-dist_norm / proximity_scale)
            else:
                proximity_weight = jnp.ones_like(l2_expert)
            if inside_box_mask is not None:
                awbc_term = (lambda_s * proximity_weight * l2_expert * (1.0 - inside_box_mask)).mean()
            else:
                awbc_term = (lambda_s * proximity_weight * l2_expert).mean()
            awbc_coef_logged  = lambda_s.mean()
            l2_expert_logged  = l2_expert.mean()
        else:
            awbc_term = awbc_coef_logged = l2_expert_logged = jnp.zeros(())

        # Online decaying BC term — value-weighted, warmup-decaying
        if (
            expert_policy is not None
            and critic_warmup_frac > 0.0
            and expert_critic_params is not None
        ):
            bc_weight = jnp.maximum(1.0 - train_frac / critic_warmup_frac, 0.0)
            v_star = jax.lax.stop_gradient(
                jnp.min(
                    predict_value(
                        critic_state=critic_states,
                        critic_params=expert_critic_params,
                        x=jnp.concatenate([observations, a_expert], axis=-1),
                    ), axis=0,
                )
            )
            v_weight = jnp.clip(
                (v_star - expert_v_min) / (expert_v_max - expert_v_min + 1e-6),
                0.0, 1.0,
            ) ** 2
            mu = jnp.tanh(pi.distribution.loc)
            bc_term = bc_weight * (v_weight * jnp.sum(
                (mu - jax.lax.stop_gradient(a_expert)) ** 2,
                axis=-1, keepdims=True,
            )).mean()
            total_loss = loss_actor.mean() + awbc_term + bc_term
        else:
            bc_term = jnp.zeros(())
            total_loss = loss_actor.mean() + awbc_term
    else:
        total_loss        = loss_actor.mean()
        awbc_coef_logged  = jnp.zeros(())
        l2_expert_logged  = jnp.zeros(())
        q_expert_logged   = jnp.zeros(())
        above_expert_frac = jnp.zeros(())
        bc_term           = jnp.zeros(())

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_pi=log_probs.mean(),
        policy_std=policy_std,
        q_min=q_min.mean(),
        q_expert=q_expert_logged,
        awbc_coef=awbc_coef_logged,
        l2_expert=l2_expert_logged,
        above_expert_frac=above_expert_frac,
        altitude_error=altitude_error_val,
        z_dot_mean=z_dot_mean_val,
        raw_loss=loss_actor.mean(),
        bc_term=bc_term,
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "expert_policy", "use_expert_guidance",
                    "altitude_obs_idx", "target_obs_idx",
                    "augment_obs_with_expert_action",
                    "proximity_scale",
                    "awbc_normalize", "awbc_use_relu", "fixed_awbc_lambda",
                    "detach_obs_aug_action", "critic_warmup_frac"],
)
def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
    raw_observations: jax.Array,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    box_threshold: float = 500.0,
    proximity_scale: Optional[float] = None,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    a_expert_precomputed: Optional[jax.Array] = None,
    awbc_normalize: bool = True,
    awbc_use_relu: bool = True,
    fixed_awbc_lambda: Optional[float] = None,
    detach_obs_aug_action: bool = False,
    inside_box_mask: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    critic_warmup_frac: float = 0.15,
) -> Tuple[SACState, PolicyAuxiliaries, jax.Array]:
    """Returns (new_state, aux, log_probs) — log_probs reused by update_temperature
    to avoid a redundant actor forward pass."""
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    (loss, aux), grads = jax.value_and_grad(policy_loss_function, has_aux=True, argnums=0)(
        agent_state.actor_state.params, agent_state.actor_state,
        agent_state.critic_state, observations, done, recurrent,
        alpha, policy_key, raw_observations=raw_observations,
        expert_policy=expert_policy, use_expert_guidance=use_expert_guidance,
        box_threshold=box_threshold,
        proximity_scale=proximity_scale,
        altitude_obs_idx=altitude_obs_idx,
        target_obs_idx=target_obs_idx,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        a_expert_precomputed=a_expert_precomputed,
        awbc_normalize=awbc_normalize,
        awbc_use_relu=awbc_use_relu,
        fixed_awbc_lambda=fixed_awbc_lambda,
        detach_obs_aug_action=detach_obs_aug_action,
        inside_box_mask=inside_box_mask,
        train_frac=train_frac,
        expert_critic_params=agent_state.expert_critic_params,
        expert_v_min=agent_state.expert_v_min,
        expert_v_max=agent_state.expert_v_max,
        critic_warmup_frac=critic_warmup_frac,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)

    # Compute log_probs using the key that update_temperature would have used
    # in the old code: split(rng)[1] where rng = split(original_agent_state.rng)[0].
    # This maintains the exact same numerical sequence as before the optimization.
    temp_rng, temp_sample_key = jax.random.split(rng)
    pi, _ = get_pi(
        actor_state=updated_actor_state, actor_params=updated_actor_state.params,
        obs=observations, done=done, recurrent=recurrent,
    )
    _, log_probs = pi.sample_and_log_prob(seed=temp_sample_key)
    # Return temp_rng so update_temperature can advance state identically to old code
    return (
        agent_state.replace(rng=temp_rng, actor_state=updated_actor_state),
        aux,
        jax.lax.stop_gradient(log_probs),
    )


# ---------------------------------------------------------------------------
# Temperature update with adaptive target entropy
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["target_entropy"])
def temperature_loss_function(
    log_alpha_params: FrozenDict,
    corrected_log_probs: jax.Array,
    target_entropy: float,
) -> Tuple[jax.Array, TemperatureAuxiliaries]:
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (log_alpha * jax.lax.stop_gradient(-corrected_log_probs - target_entropy)).mean()
    return loss, TemperatureAuxiliaries(alpha=alpha, log_alpha=log_alpha)


@partial(jax.jit, static_argnames=["target_entropy"])
def update_temperature(
    agent_state: SACState,
    log_probs: jax.Array,
    target_entropy: float,
) -> Tuple[SACState, TemperatureAuxiliaries]:
    """
    Standard SAC temperature update.
    log_probs passed from update_policy using the same RNG key the old
    update_temperature would have used — numerically equivalent to old code.
    target_entropy is a static Python float, never traced.
    """
    (loss, aux), grads = jax.value_and_grad(temperature_loss_function, has_aux=True)(
        agent_state.alpha.params, log_probs.sum(-1), target_entropy,
    )
    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    # agent_state.rng was already advanced correctly in update_policy — no split needed here
    return agent_state.replace(alpha=new_alpha_state), jax.lax.stop_gradient(aux)


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["tau"])
def update_target_networks(agent_state: SACState, tau: float) -> SACState:
    return agent_state.replace(critic_state=agent_state.critic_state.soft_update(tau=tau))


# ---------------------------------------------------------------------------
# Value-threshold box helper
# ---------------------------------------------------------------------------


def is_inside_box(
    obs: jax.Array,
    raw_obs: jax.Array,
    expert_policy: Callable,
    critic_state: Any,
    expert_critic_params: Any,
    threshold: jax.Array,
) -> jax.Array:
    """Returns bool mask: True where V_expert(s) > threshold."""
    a_exp = jax.lax.stop_gradient(expert_policy(raw_obs))
    v_expert = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=expert_critic_params,
            x=jnp.concatenate([obs, a_exp], axis=-1),
        ),
        axis=0,
    )
    return v_expert > threshold


# ---------------------------------------------------------------------------
# Agent update (one gradient step)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "recurrent", "buffer", "gamma", "tau", "action_dim",
        "num_critic_updates", "reward_scale", "transition_mix_fraction",
        "expert_policy", "use_expert_guidance", "target_entropy",
        "policy_update_start", "alpha_update_start", "expert_mix_fraction",
        "box_threshold", "proximity_scale", "altitude_obs_idx", "target_obs_idx",
        "augment_obs_with_expert_action",
        "awbc_normalize", "awbc_use_relu", "fixed_awbc_lambda",
        "detach_obs_aug_action",
        "use_critic_blend", "critic_warmup_frac", "use_box",
        "total_timesteps",
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
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    proximity_scale: Optional[float] = None,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    awbc_normalize: bool = True,
    awbc_use_relu: bool = True,
    fixed_awbc_lambda: Optional[float] = None,
    detach_obs_aug_action: bool = False,
    use_critic_blend: bool = False,
    critic_warmup_frac: float = 0.15,
    use_box: bool = False,
    box_v_min: float = 0.0,
    box_v_max: float = 0.0,
    total_timesteps: int = 1,
) -> Tuple[SACState, AuxiliaryLogs]:
    sample_key, expert_sample_key, rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)

    # --- Sample from buffer ---
    if buffer is not None and agent_state.collector_state.buffer_state is not None:
        (
            observations, terminated, truncated, next_observations,
            rewards, actions, raw_observations,
        ) = get_batch_from_buffer(buffer, agent_state.collector_state.buffer_state, sample_key)
        original_transition = Transition(
            observations, actions, rewards, terminated, truncated,
            next_observations, raw_obs=raw_observations,
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
                    None if (x is None or y is None)
                    else jnp.concatenate([x[:n_from_buffer], y], axis=0)
                ),
                original_transition, additional_transition,
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
            exp_obs, exp_terminated, exp_truncated, exp_next_obs,
            exp_rewards, exp_actions, exp_raw_obs,
        ) = get_batch_from_buffer(buffer, agent_state.collector_state.buffer_state, expert_sample_key)

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

    # --- Obs augmentation: append a_expert to obs and next_obs ---
    # Must happen before any network call (critic, actor, policy loss).
    # raw_obs gives the env observations without train_frac, which is what
    # expert_policy expects. For next_obs we strip the last dim (train_frac).
    if augment_obs_with_expert_action and expert_policy is not None:
        _raw = (
            transition.raw_obs if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        _raw_next = transition.next_obs[..., :-1]  # strip train_frac
        aug_obs      = augment_obs_if_needed(transition.obs,      _raw,      expert_policy, True)
        aug_next_obs = augment_obs_if_needed(transition.next_obs, _raw_next, expert_policy, True)
        transition = transition.replace(obs=aug_obs, next_obs=aug_next_obs)

    # --- Pre-compute Q(s, a_expert) and a_expert once for both critic logging + AWBC ---
    # Avoids computing expert_policy twice (once here, once inside policy_loss_function)
    expert_q = None
    a_expert_precomputed = None
    needs_expert = expert_policy is not None and (
        use_expert_guidance or use_box
    )
    if needs_expert:
        _raw = (
            transition.raw_obs if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        a_expert_precomputed = jax.lax.stop_gradient(expert_policy(_raw))
        # transition.obs is already augmented at this point if augment_obs_with_expert_action
        expert_q = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.critic_state.params,
                    x=jnp.concatenate([transition.obs, a_expert_precomputed], axis=-1),
                ),
                axis=0,
            )
        )

    # --- Critic blend: pre-compute V*(s') and α_blend for blended Bellman target ---
    # α_blend = max(1 - train_frac / critic_warmup_frac, 0)
    # Decays from 1 (pure expert target) to 0 (pure Bellman) over the warmup window.
    v_expert_next_blend = None
    alpha_blend_val = None
    if use_critic_blend and agent_state.expert_critic_params is not None and expert_policy is not None:
        next_raw = transition.next_obs[..., :-1]  # strip train_frac
        a_expert_next_blend = jax.lax.stop_gradient(expert_policy(next_raw))
        v_expert_next_blend = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([transition.next_obs, a_expert_next_blend], axis=-1),
                ),
                axis=0,
            )
        )
        train_frac_blend = agent_state.collector_state.timestep / total_timesteps
        alpha_blend_val = jnp.maximum(1.0 - train_frac_blend / critic_warmup_frac, 0.0)

    # --- Box mask for AWBC ---
    inside_box_mask = None
    if use_box and agent_state.expert_critic_params is not None and expert_policy is not None:
        _raw_box = (
            transition.raw_obs if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        a_box_upd = jax.lax.stop_gradient(expert_policy(_raw_box))
        v_box_upd = jnp.min(
            predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.expert_critic_params,
                x=jnp.concatenate([transition.obs, a_box_upd], axis=-1),
            ),
            axis=0,
        )
        train_frac = agent_state.collector_state.timestep / total_timesteps
        effective_threshold = box_v_min + (box_v_max - box_v_min) * train_frac
        inside_box_mask = (v_box_upd > effective_threshold).astype(jnp.float32)
        effective_threshold_logged = jnp.array(effective_threshold)
        box_entry_rate = inside_box_mask.mean()
    else:
        effective_threshold_logged = jnp.zeros(())
        box_entry_rate = jnp.zeros(())

    # --- Critic updates ---
    def critic_update_step(carry, _):
        agent_state = carry
        agent_state, aux_value = update_value_functions(
            observations=transition.obs, actions=transition.action,
            next_observations=transition.next_obs, rewards=transition.reward,
            dones=dones, agent_state=agent_state, recurrent=recurrent,
            gamma=gamma, reward_scale=reward_scale, expert_q=expert_q,
            v_expert_next=v_expert_next_blend, alpha_blend=alpha_blend_val,
        )
        return agent_state, aux_value

    agent_state, aux_value_seq = jax.lax.scan(
        critic_update_step, agent_state, None, length=num_critic_updates
    )
    aux_value = jax.tree.map(lambda x: x[-1], aux_value_seq)

    # --- Policy update — returns log_probs for temperature reuse ---
    train_frac = agent_state.collector_state.timestep / total_timesteps
    new_agent_state, aux_policy, policy_log_probs = update_policy(
        observations=transition.obs, done=dones, agent_state=agent_state,
        recurrent=recurrent, raw_observations=transition.raw_obs,
        expert_policy=expert_policy, use_expert_guidance=use_expert_guidance,
        box_threshold=box_threshold,
        proximity_scale=proximity_scale,
        altitude_obs_idx=altitude_obs_idx,
        target_obs_idx=target_obs_idx,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        a_expert_precomputed=a_expert_precomputed,
        awbc_normalize=awbc_normalize,
        awbc_use_relu=awbc_use_relu,
        fixed_awbc_lambda=fixed_awbc_lambda,
        detach_obs_aug_action=detach_obs_aug_action,
        inside_box_mask=inside_box_mask,
        train_frac=train_frac,
        critic_warmup_frac=critic_warmup_frac,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= policy_update_start,
        lambda: new_agent_state, lambda: agent_state,
    )

    # --- Temperature update — reuses log_probs, no redundant actor forward pass ---
    new_agent_state_temp, aux_temperature = update_temperature(
        agent_state, log_probs=policy_log_probs, target_entropy=target_entropy,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= alpha_update_start,
        lambda: new_agent_state_temp, lambda: agent_state,
    )

    agent_state = update_target_networks(agent_state, tau=tau)

    # Direct field access instead of to_state_dict — avoids serialization overhead
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=ValueAuxiliaries(
            critic_loss=aux_value.critic_loss.flatten(),
            q_pred_min=aux_value.q_pred_min.flatten(),
            q_expert_mean=aux_value.q_expert_mean.flatten(),
            q_gap=aux_value.q_gap.flatten(),
            var_preds=aux_value.var_preds.flatten(),
            alpha_blend=aux_value.alpha_blend.flatten(),
            effective_threshold=effective_threshold_logged,
            box_entry_rate=box_entry_rate,
        ),
    )
    return agent_state, aux


# ---------------------------------------------------------------------------
# Training iteration
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "env_args", "mode", "recurrent", "buffer", "log_frequency",
        "num_episode_test", "log_fn", "log", "verbose", "action_dim",
        "lstm_hidden_size", "agent_config", "horizon", "total_timesteps",
        "n_epochs", "transition_mix_fraction", "expert_policy",
        "eval_expert_policy",
        "use_expert_guidance", "action_scale", "early_termination_condition",
        "num_critic_updates", "expert_mix_fraction",
        "box_threshold", "proximity_scale", "altitude_obs_idx", "target_obs_idx",
        "augment_obs_with_expert_action",
        "distance_to_stable", "imitation_coef_offset", "imitation_coef",
        "awbc_normalize", "awbc_use_relu", "fixed_awbc_lambda",
        "detach_obs_aug_action",
        "policy_update_start", "alpha_update_start",
        "use_critic_blend", "critic_warmup_frac", "use_box",
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
    expert_policy: Optional[Callable] = None,       # used for training
    eval_expert_policy: Optional[Callable] = None,  # used for eval logging only
    use_expert_guidance: bool = True,
    action_scale: float = 1.0,
    early_termination_condition: Optional[Callable] = None,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    proximity_scale: Optional[float] = None,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    awbc_normalize: bool = True,
    awbc_use_relu: bool = True,
    fixed_awbc_lambda: Optional[float] = None,
    detach_obs_aug_action: bool = False,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    use_critic_blend: bool = False,
    critic_warmup_frac: float = 0.15,
    use_box: bool = False,
    box_v_min: float = 0.0,
    box_v_max: float = 0.0,
    # API compat
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = lambda x: 1.0,
    imitation_coef_offset: float = 0.0,
) -> tuple[SACState, None]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent, mode=mode, env_args=env_args, buffer=buffer,
        uniform=uniform, expert_policy=expert_policy,
        action_scale=action_scale,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        use_box=use_box, box_v_min=box_v_min, box_v_max=box_v_max,
        total_timesteps=total_timesteps,
    )

    agent_state, transition = collect_scan_fn(agent_state, None)
    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        update_scan_fn = partial(
            update_agent,
            buffer=buffer, recurrent=recurrent, gamma=agent_config.gamma,
            action_dim=action_dim, target_entropy=agent_config.target_entropy,
            tau=agent_config.tau, reward_scale=agent_config.reward_scale,
            additional_transition=(
                transition if transition_mix_fraction < 1.0 else None
            ),
            transition_mix_fraction=transition_mix_fraction,
            expert_policy=expert_policy,
            use_expert_guidance=use_expert_guidance,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            box_threshold=box_threshold,
            proximity_scale=proximity_scale,
            altitude_obs_idx=altitude_obs_idx,
            target_obs_idx=target_obs_idx,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            awbc_normalize=awbc_normalize,
            awbc_use_relu=awbc_use_relu,
            fixed_awbc_lambda=fixed_awbc_lambda,
            detach_obs_aug_action=detach_obs_aug_action,
            use_critic_blend=use_critic_blend,
            critic_warmup_frac=critic_warmup_frac,
            use_box=use_box,
            box_v_min=box_v_min,
            box_v_max=box_v_max,
            total_timesteps=total_timesteps,
        )
        agent_state, aux = jax.lax.scan(update_scan_fn, agent_state, xs=None, length=n_epochs)
        aux = jax.tree.map(lambda x: x[-1].reshape((1,)), aux)
        aux = aux.replace(
            value=ValueAuxiliaries(
                critic_loss=aux.value.critic_loss.flatten(),
                q_pred_min=aux.value.q_pred_min.flatten(),
                q_expert_mean=aux.value.q_expert_mean.flatten(),
                q_gap=aux.value.q_gap.flatten(),
                var_preds=aux.value.var_preds.flatten(),
                alpha_blend=aux.value.alpha_blend.flatten(),
                effective_threshold=aux.value.effective_threshold.flatten(),
                box_entry_rate=aux.value.box_entry_rate.flatten(),
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
        do_update, skip_update, operand=agent_state,
    )

    # For obs-augmented runs, wrap the actor's apply_fn so that evaluate_and_log
    # (and evaluate.py inside it) sees 12-dim obs transparently — no changes needed
    # to evaluate.py or log.py.
    if augment_obs_with_expert_action and expert_policy is not None:
        _orig_apply_fn = agent_state.actor_state.apply_fn
        def _augmented_apply_fn(params, obs, *args, **kwargs):
            _raw = obs[..., :-1]  # strip train_frac
            _a_exp = jax.lax.stop_gradient(expert_policy(_raw))
            _aug = jnp.concatenate([obs[..., :-1], _a_exp, obs[..., -1:]], axis=-1)
            return _orig_apply_fn(params, _aug, *args, **kwargs)
        _eval_agent_state = agent_state.replace(
            actor_state=agent_state.actor_state.replace(apply_fn=_augmented_apply_fn)
        )
    else:
        _eval_agent_state = agent_state

    _eval_agent_state, metrics_to_log = evaluate_and_log(
        _eval_agent_state, aux, index, mode, env_args, num_episode_test, recurrent,
        lstm_hidden_size, log, verbose, log_fn, log_frequency, total_timesteps,
        expert_policy=eval_expert_policy,
        action_scale=action_scale,
        early_termination_condition=early_termination_condition,
        train_frac=agent_state.collector_state.train_time_fraction,
    )
    # Keep the original agent_state (with original apply_fn) for training
    agent_state = agent_state.replace(
        eval_rng=_eval_agent_state.eval_rng,
        n_logs=_eval_agent_state.n_logs,
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
    eval_expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    early_termination_condition: Optional[Callable] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    proximity_scale: Optional[float] = None,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    # MC critic pretraining (replaces Bellman pretraining)
    use_mc_critic_pretrain: bool = False,
    mc_pretrain_n_mc_steps: int = 10_000,
    mc_pretrain_n_mc_episodes: int = 100,
    mc_pretrain_n_steps: int = 5_000,
    # Online critic light pre-regression (requires MC critic pretrain)
    use_online_critic_light_pretrain: bool = True,
    online_critic_pretrain_steps: int = 500,
    online_critic_pretrain_lr_scale: float = 0.1,
    # Bellman critic pretraining (legacy fallback, mutually exclusive with MC)
    use_bellman_critic_pretrain: bool = False,
    # Expert-guided policy loss terms
    augment_obs_with_expert_action: bool = False,
    # AWBC ablation flags
    awbc_normalize: bool = True,
    awbc_use_relu: bool = True,
    fixed_awbc_lambda: Optional[float] = None,
    detach_obs_aug_action: bool = False,
    # Train-fraction conditioning: append timestep/total_timesteps to obs
    use_train_frac: bool = False,
    # Update start thresholds
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    # Blended Bellman target (replaces potential-based shaping)
    use_critic_blend: bool = False,
    critic_warmup_frac: float = 0.15,
    # Value-threshold box (v_min/v_max inferred from MC pretraining)
    use_box: bool = False,
):
    """
    SAC + AWBC training factory.

    expert_policy:      used for training (warmup seeding, AWBC gradient, critic
                        pre-training). Pass None for true vanilla SAC.
    eval_expert_policy: used ONLY for eval logging (expert bias metric).
                        Always passed regardless of whether training uses expert.
                        Defaults to expert_policy if not set explicitly.
    """
    # If no separate eval policy provided, fall back to the training policy
    # (which may be None for vanilla SAC — in that case no expert bias logged)
    _eval_expert_policy = eval_expert_policy if eval_expert_policy is not None else expert_policy
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
            max_timesteps=total_timesteps if use_train_frac else None,
            num_critics=num_critics,
            expert_buffer_n_steps=(
                expert_buffer_n_steps if expert_policy is not None else 0
            ),
            augment_obs_with_expert_action=augment_obs_with_expert_action,
        )

        _box_v_min = jnp.array(0.0)
        _box_v_max = jnp.array(0.0)

        if expert_policy is not None and use_mc_critic_pretrain:
            expert_critic_state = get_initialized_critic(
                key=expert_key,
                env_config=env_args,
                critic_optimizer_config=critic_optimizer_args,
                network_config=network_args,
                num_critics=num_critics,
                max_timesteps=total_timesteps if use_train_frac else None,
                extra_obs_dim=(
                    get_action_dim(env_args.env, env_args.env_params)
                    if augment_obs_with_expert_action else 0
                ),
            )
            agent_state, frozen_expert_params, mc_obs_batched, mc_action_batched, mc_aux = pretrain_critic_mc(
                agent_state=agent_state,
                expert_critic_state=expert_critic_state,
                expert_policy=expert_policy,
                mode=mode,
                env_args=env_args,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                n_mc_steps=mc_pretrain_n_mc_steps,
                n_mc_episodes=mc_pretrain_n_mc_episodes,
                n_steps=mc_pretrain_n_steps,
                max_timesteps=total_timesteps if use_train_frac else None,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
            )
            agent_state = agent_state.replace(
                expert_critic_params=frozen_expert_params,
                expert_v_min=mc_aux.v_min,
                expert_v_max=mc_aux.v_max,
            )
            if use_box:
                _box_v_min = mc_aux.v_min
                _box_v_max = mc_aux.v_max
            jax.debug.print(
                "[MC pretrain] loss: {i:.4f} -> {f:.4f}  |  "
                "Q(s,a*) mean={qm:.1f}  min={qn:.1f}  max={qx:.1f}",
                i=mc_aux.initial_loss, f=mc_aux.final_loss,
                qm=mc_aux.q_expert_mean,
                qn=mc_aux.q_expert_min,
                qx=mc_aux.q_expert_max,
            )

            if use_online_critic_light_pretrain:
                agent_state = pretrain_critic_online_light(
                    agent_state,
                    mc_obs_batched,
                    mc_action_batched,
                    n_steps=online_critic_pretrain_steps,
                    lr_scale=online_critic_pretrain_lr_scale,
                )
                jax.debug.print(
                    "[Online critic light pretrain] done ({n} steps, lr_scale={s})",
                    n=online_critic_pretrain_steps, s=online_critic_pretrain_lr_scale,
                )

        if expert_policy is not None and use_bellman_critic_pretrain:
            agent_state = pretrain_critic_bellman(
                agent_state=agent_state,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                buffer=buffer,
                n_steps=mc_pretrain_n_steps,
            )
            jax.debug.print("[Bellman pretrain] done ({n} steps)", n=mc_pretrain_n_steps)

        cloning_parameters, pre_train_n_steps = get_cloning_args(cloning_args, total_timesteps)
        if pre_train_n_steps > 0:
            agent_state = get_pre_trained_agent(
                agent_state, expert_policy, expert_key, env_args, cloning_args,
                mode, agent_config, actor_optimizer_args, critic_optimizer_args,
            )

        num_updates = total_timesteps // env_args.n_envs
        _, action_shape = get_state_action_shapes(env_args.env)

        _valid_cloning_params = {
            k: v for k, v in cloning_parameters.items()
            if k in (
                "n_epochs", "transition_mix_fraction",
                "imitation_coef", "distance_to_stable", "imitation_coef_offset",
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
            eval_expert_policy=_eval_expert_policy,
            use_expert_guidance=use_expert_guidance,
            early_termination_condition=early_termination_condition,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            box_threshold=box_threshold,
            proximity_scale=proximity_scale,
            altitude_obs_idx=altitude_obs_idx,
            target_obs_idx=target_obs_idx,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            awbc_normalize=awbc_normalize,
            awbc_use_relu=awbc_use_relu,
            fixed_awbc_lambda=fixed_awbc_lambda,
            detach_obs_aug_action=detach_obs_aug_action,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            use_critic_blend=use_critic_blend,
            critic_warmup_frac=critic_warmup_frac,
            use_box=use_box,
            box_v_min=_box_v_min,
            box_v_max=_box_v_max,
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