"""Pretraining modules for SAC — composable, stackable.

Each pretrain step is a standalone function that takes an agent state
(and optional accumulated results) and returns the updated state.
Steps can be composed in any order; data dependencies flow through
PretrainResult.

Design:
  - pretrain_critic_mc: MC returns → expert critic → frozen params + v_min/v_max
  - pretrain_critic_online_light: weak nudge of online critic toward φ*
  - pretrain_actor_weighted_bc: value-weighted BC on the actor
  - pretrain_critic_bellman: legacy Bellman-bootstrapped pretraining
  - refresh_phi_star: periodic self-consistent φ* refresh (runtime)

All functions are pure: no hidden state, no feature flags.
"""
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience_from_expert_policy,
    get_pi,
)
from ajax.environments.utils import check_env_is_gymnax, maybe_append_train_frac
from ajax.networks.networks import predict_value
from ajax.state import EnvironmentConfig, LoadedTrainState
from ajax.types import BufferType


# ---------------------------------------------------------------------------
# Auxiliary dataclasses
# ---------------------------------------------------------------------------


@struct.dataclass
class MCPretrainAux:
    """Diagnostics logged after MC critic pretraining."""
    initial_loss: jax.Array
    final_loss: jax.Array
    q_expert_mean: jax.Array
    q_expert_min: jax.Array
    q_expert_max: jax.Array
    v_min: jax.Array
    v_max: jax.Array


@struct.dataclass
class PhiRefreshAuxiliaries:
    """φ* refresh diagnostics. All-zero when no refresh triggered this step."""
    loss_before: jax.Array
    loss_after: jax.Array
    expert_buffer_size: jax.Array


# ---------------------------------------------------------------------------
# MC critic pretraining
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma", "reward_scale", "n_steps",
                    "expert_policy", "n_mc_steps", "n_mc_episodes",
                    "mode", "env_args", "max_timesteps", "batch_size",
                    "augment_obs_with_expert_action"],
)
def pretrain_critic_mc(
    agent_state,
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
    preloaded_obs: Optional[jax.Array] = None,
    preloaded_action: Optional[jax.Array] = None,
    preloaded_mc: Optional[jax.Array] = None,
) -> Tuple[Any, FrozenDict, jax.Array, jax.Array, MCPretrainAux, LoadedTrainState]:
    """Pre-train critic using Monte Carlo returns from expert trajectories.

    Unlike Bellman pretraining (bootstraps from an untrained critic → biased),
    MC returns G_t = Σ γ^k r_{t+k} are unbiased estimates of V^expert(s).
    The critic starts with accurate Q(s, a_expert) near the target from step 1.

    Collection strategy: single call with n_mc_steps * n_mc_episodes // n_envs
    timesteps. The n_envs parallel environments reset to different (initial,
    target) altitude pairs on each episode boundary, giving the same state-space
    coverage as separate per-seed rollouts — without any mapping over traced keys
    (which fails inside the outer vmap over seeds).
    Total transitions ≈ n_mc_steps * n_mc_episodes regardless of n_envs.

    Returns: (agent_state, frozen_expert_params, obs_batched, action_batched,
              MCPretrainAux, expert_critic_state_trained)
    """
    if preloaded_obs is not None:
        obs_flat    = preloaded_obs
        action_flat = preloaded_action
        mc_flat     = preloaded_mc
        raw_obs_flat = preloaded_obs
    else:
        n_total_steps = max(1, (n_mc_steps * n_mc_episodes) // env_args.n_envs)
        all_transitions = collect_experience_from_expert_policy(
            expert_policy=expert_policy,
            rng=agent_state.rng,
            mode=mode,
            env_args=env_args,
            n_timesteps=n_total_steps,
        )

        rewards = all_transitions.reward * reward_scale
        dones = jnp.logical_or(
            all_transitions.terminated, all_transitions.truncated
        ).astype(jnp.float32)

        def mc_scan(carry, x):
            reward, done = x
            mc_return = reward + gamma * carry * (1.0 - done)
            return mc_return, mc_return

        _, mc_returns = jax.lax.scan(
            mc_scan,
            jnp.zeros_like(rewards[0]),
            (rewards[::-1], dones[::-1]),
        )
        mc_returns = mc_returns[::-1]

        T, n_envs = rewards.shape[:2]
        obs_flat    = all_transitions.obs.reshape(T * n_envs, -1)
        action_flat = all_transitions.action.reshape(T * n_envs, -1)
        mc_flat     = mc_returns.reshape(T * n_envs, 1)
        raw_obs_flat = obs_flat

    # Append train_frac=0.0 if max_timesteps was set
    if max_timesteps is not None:
        obs_flat = jnp.concatenate(
            [obs_flat, jnp.zeros((obs_flat.shape[0], 1))], axis=-1
        )

    # Augment obs with expert action if enabled
    if augment_obs_with_expert_action:
        a_expert_flat = jax.lax.stop_gradient(expert_policy(raw_obs_flat))
        if max_timesteps is not None:
            obs_flat = jnp.concatenate(
                [obs_flat[..., :-1], a_expert_flat, obs_flat[..., -1:]], axis=-1
            )
        else:
            obs_flat = jnp.concatenate([obs_flat, a_expert_flat], axis=-1)

    # Batch into fixed-size chunks for regression
    n_total   = obs_flat.shape[0]
    n_batches = n_total // batch_size
    obs_batched    = obs_flat[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    action_batched = action_flat[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    mc_batched     = mc_flat[:n_batches * batch_size].reshape(n_batches, batch_size, 1)

    # Supervised regression: Q(s, a_expert) → MC return
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

    frozen_expert_params = jax.lax.stop_gradient(expert_critic_state.params)

    # Q-value diagnostics on the last batch
    q_preds_final = predict_value(
        critic_state=expert_critic_state,
        critic_params=frozen_expert_params,
        x=jnp.concatenate([obs_batched[-1], action_batched[-1]], axis=-1),
    )
    q_for_stats = jnp.min(q_preds_final, axis=0)
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
    ), expert_critic_state


# ---------------------------------------------------------------------------
# Online critic light pre-regression
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["n_steps", "lr_scale"])
def pretrain_critic_online_light(
    agent_state,
    obs_batched: jax.Array,
    action_batched: jax.Array,
    n_steps: int = 500,
    lr_scale: float = 0.1,
):
    """Weak supervised nudge of the online critic toward φ*.

    Goal: reduce seed-to-seed variance at initialization.
    NOT a full MC pretrain — just reduces starting point spread.
    Target network stays random and untouched.
    """
    def light_loss_fn(critic_params, obs, actions):
        v_expert = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([obs, actions], axis=-1),
                ), axis=0,
            )
        )
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
        grads = jax.tree.map(lambda g: g * lr_scale, grads)
        new_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
        return agent_state.replace(critic_state=new_critic_state), loss

    agent_state, _ = jax.lax.scan(
        step, agent_state, (obs_batched[:n_steps], action_batched[:n_steps])
    )
    return agent_state


# ---------------------------------------------------------------------------
# Actor pre-training via value-weighted behavioral cloning
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["n_steps", "recurrent"])
def pretrain_actor_weighted_bc(
    agent_state,
    obs_batched: jax.Array,
    action_batched: jax.Array,
    n_steps: int = 5_000,
    recurrent: bool = False,
):
    """Value-weighted behavioral cloning on the actor.

    Loss: mean_s [w(s) * ||μ_θ(s) - π*(s)||²]
    w(s) = clip((V*(s) - V_min) / (V_max - V_min), 0, 1)²
    V*(s) = min_k Q_φ*(s, π*(s)) using frozen expert_critic_params

    Only actor_state is modified; all other agent_state fields are unchanged.
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
        mu = jnp.tanh(pi.distribution.loc)

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
        w = w ** 2
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
# Bellman critic pretraining (legacy fallback)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma", "reward_scale", "n_steps", "buffer",
                    "update_value_fn"],
)
def pretrain_critic_bellman(
    agent_state,
    recurrent: bool,
    gamma: float,
    reward_scale: float,
    buffer: BufferType,
    n_steps: int = 5_000,
    update_value_fn: Optional[Callable] = None,
    update_target_fn: Optional[Callable] = None,
):
    """Bellman-bootstrapped critic pretraining on the expert buffer.

    Requires update_value_fn and update_target_fn to be passed in,
    avoiding circular imports with train_SAC.
    """
    def critic_pretrain_step(carry, _):
        agent_state = carry
        sample_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        (
            observations, terminated, truncated, next_observations,
            rewards, actions, raw_observations, _,
        ) = get_batch_from_buffer(buffer, agent_state.collector_state.buffer_state, sample_key)
        dones = jnp.logical_or(terminated, truncated)
        agent_state, _ = update_value_fn(
            observations=observations, actions=actions,
            next_observations=next_observations, rewards=rewards,
            dones=dones, agent_state=agent_state, recurrent=recurrent,
            gamma=gamma, reward_scale=reward_scale,
        )
        agent_state = update_target_fn(agent_state, tau=5e-4)
        return agent_state, None

    agent_state, _ = jax.lax.scan(critic_pretrain_step, agent_state, None, length=n_steps)
    return agent_state


# ---------------------------------------------------------------------------
# Periodic self-consistent φ* refresh
# ---------------------------------------------------------------------------


def refresh_phi_star(
    agent_state,
    buffer: BufferType,
    phi_refresh_steps: int,
    gamma: float,
    reward_scale: float,
    expert_policy: Callable,
) -> Tuple[Any, PhiRefreshAuxiliaries]:
    """Periodic self-consistent φ* refresh using expert-flagged buffer transitions.

    Target: r + γ * min_k Q_φ*(s′, π*(s′))  — φ* supervises its own bootstraps.
    Non-expert transitions are masked out, so the gradient comes only from
    (s, a_expert, r, s') rows where EDGE fired the expert action.

    Returns updated agent state and PhiRefreshAuxiliaries.
    """
    buffer_state = agent_state.collector_state.buffer_state
    expert_critic_state = agent_state.expert_critic_state

    diag_key, refresh_key, new_rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=new_rng)

    # Diagnostic batch for loss_before / loss_after / expert_buffer_size
    obs_d, terminated_d, truncated_d, next_obs_d, rewards_d, actions_d, _, is_expert_d = (
        get_batch_from_buffer(buffer, buffer_state, diag_key)
    )
    expert_mask_d = is_expert_d[..., 0]
    expert_buffer_size = expert_mask_d.sum()
    rewards_d = rewards_d * reward_scale
    dones_d = jnp.logical_or(terminated_d, truncated_d).astype(jnp.float32)

    a_expert_d = jax.lax.stop_gradient(expert_policy(next_obs_d))
    q_next_d = predict_value(
        critic_state=expert_critic_state,
        critic_params=expert_critic_state.target_params,
        x=jnp.concatenate([next_obs_d, a_expert_d], axis=-1),
    )
    target_d = jax.lax.stop_gradient(
        rewards_d + gamma * (1.0 - dones_d) * jnp.min(q_next_d, axis=0)
    )

    def compute_diag_loss(params):
        q_preds = predict_value(
            critic_state=expert_critic_state,
            critic_params=params,
            x=jnp.concatenate([obs_d, actions_d], axis=-1),
        )
        mse_per = jnp.mean((q_preds - target_d) ** 2, axis=(0, 2))
        n_expert = expert_mask_d.sum() + 1e-6
        return (mse_per * expert_mask_d).sum() / n_expert

    loss_before = compute_diag_loss(expert_critic_state.params)

    # Gradient refresh steps
    def refresh_step(carry, _):
        expert_critic_state, step_key = carry
        sample_key, step_key = jax.random.split(step_key)

        obs, terminated, truncated, next_obs, rewards, actions, _, is_expert = (
            get_batch_from_buffer(buffer, buffer_state, sample_key)
        )

        expert_mask = is_expert[..., 0]
        rewards = rewards * reward_scale
        dones = jnp.logical_or(terminated, truncated).astype(jnp.float32)

        a_expert_next = jax.lax.stop_gradient(expert_policy(next_obs))
        q_next = predict_value(
            critic_state=expert_critic_state,
            critic_params=expert_critic_state.target_params,
            x=jnp.concatenate([next_obs, a_expert_next], axis=-1),
        )
        target = jax.lax.stop_gradient(rewards + gamma * (1.0 - dones) * jnp.min(q_next, axis=0))

        def loss_fn(params):
            q_preds = predict_value(
                critic_state=expert_critic_state,
                critic_params=params,
                x=jnp.concatenate([obs, actions], axis=-1),
            )
            mse_per = jnp.mean((q_preds - target) ** 2, axis=(0, 2))
            n_expert = expert_mask.sum() + 1e-6
            return (mse_per * expert_mask).sum() / n_expert

        _, grads = jax.value_and_grad(loss_fn)(expert_critic_state.params)
        new_expert_critic_state = expert_critic_state.apply_gradients(grads=grads)
        return (new_expert_critic_state, step_key), None

    (new_expert_critic_state, _), _ = jax.lax.scan(
        refresh_step, (expert_critic_state, refresh_key), None, length=phi_refresh_steps
    )

    loss_after = compute_diag_loss(new_expert_critic_state.params)

    new_expert_critic_state = new_expert_critic_state.soft_update(tau=1.0)
    frozen_params = jax.lax.stop_gradient(new_expert_critic_state.params)

    phi_refresh_aux = PhiRefreshAuxiliaries(
        loss_before=jnp.atleast_1d(loss_before),
        loss_after=jnp.atleast_1d(loss_after),
        expert_buffer_size=jnp.atleast_1d(expert_buffer_size),
    )
    return (
        agent_state.replace(
            expert_critic_state=new_expert_critic_state,
            expert_critic_params=frozen_params,
        ),
        phi_refresh_aux,
    )


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
    """Collect expert transitions and store them in the replay buffer."""
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
            "is_expert": take(jnp.ones_like(flat_obs[..., :1])),
        }
        return buffer.add(buffer_state, _transition), None

    buffer_state, _ = jax.lax.scan(add_one, buffer_state, jnp.arange(n_total))
    return buffer_state
