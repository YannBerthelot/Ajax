"""One-shot pre-training research features as composable :class:`Extension`s.

MCPretrain runs once, before the training loop (the ``pretrain`` phase,
skipped on resume); PhiRefresh refreshes its critic during training.

* :class:`MCPretrain`          — ``use_mc_critic_pretrain``: regress a
  frozen expert critic on Monte-Carlo returns, then (optionally) lightly
  nudge the online critic toward it.
* :class:`BellmanPretrain`     — ``use_bellman_critic_pretrain``: the
  legacy Bellman-bootstrapped critic pre-training on the expert buffer.
* :class:`PhiRefresh`          — ``use_phi_refresh``: periodic
  self-consistent refresh of the frozen expert critic during training
  (the ``post_update`` phase).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import predict_value
from ajax.state import EnvironmentConfig, LoadedTrainState
from ajax.types import BufferType


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


@partial(
    jax.jit,
    static_argnames=[
        "gamma",
        "reward_scale",
        "n_steps",
        "expert_policy",
        "n_mc_steps",
        "n_mc_episodes",
        "mode",
        "env_args",
        "max_timesteps",
        "batch_size",
        "augment_obs_with_expert_action",
    ],
)
def pretrain_critic_mc(
    agent_state,
    expert_critic_state: LoadedTrainState,
    expert_policy: Callable,
    mode: str,
    env_args: EnvironmentConfig,
    gamma: float,
    reward_scale: float,
    n_mc_steps: int = 10_000,
    n_mc_episodes: int = 500,
    n_steps: int = 5_000,
    batch_size: int = 256,
    max_timesteps: Optional[int] = None,
    augment_obs_with_expert_action: bool = False,
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
    obs_flat = all_transitions.obs.reshape(T * n_envs, -1)
    action_flat = all_transitions.action.reshape(T * n_envs, -1)
    mc_flat = mc_returns.reshape(T * n_envs, 1)
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
    n_total = obs_flat.shape[0]
    n_batches = n_total // batch_size
    obs_batched = obs_flat[: n_batches * batch_size].reshape(n_batches, batch_size, -1)
    action_batched = action_flat[: n_batches * batch_size].reshape(
        n_batches, batch_size, -1
    )
    mc_batched = mc_flat[: n_batches * batch_size].reshape(n_batches, batch_size, 1)

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
            expert_critic_state.params,
            obs_b,
            action_b,
            mc_b,
        )
        new_expert_critic_state = expert_critic_state.apply_gradients(grads=grads)
        return (new_expert_critic_state, step + 1), loss

    initial_loss, _ = jax.value_and_grad(mc_loss_fn)(
        expert_critic_state.params,
        obs_batched[0],
        action_batched[0],
        mc_batched[0],
    )

    n_passes = max(1, n_steps // n_batches)
    batches = (obs_batched, action_batched, mc_batched)

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

    return (
        agent_state,
        frozen_expert_params,
        obs_batched,
        action_batched,
        MCPretrainAux(
            initial_loss=initial_loss,
            final_loss=final_loss,
            q_expert_mean=q_for_stats.mean(),
            q_expert_min=q_for_stats.min(),
            q_expert_max=q_for_stats.max(),
            v_min=v_min,
            v_max=v_max,
        ),
        expert_critic_state,
    )


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
                ),
                axis=0,
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


def refresh_phi_star(
    agent_state,
    buffer: BufferType,
    phi_refresh_steps: int,
    gamma: float,
    reward_scale: float,
    expert_policy: Callable,
) -> Any:
    """Periodic self-consistent φ* refresh using expert-flagged buffer transitions.

    Target: r + γ * min_k Q_φ*(s′, π*(s′))  — φ* supervises its own bootstraps.
    Non-expert transitions are masked out, so the gradient comes only from
    (s, a_expert, r, s') rows where EDGE fired the expert action.
    """
    buffer_state = agent_state.collector_state.buffer_state
    expert_critic_state = agent_state.expert_critic_state

    # Three keys: the first once drew a diagnostic batch; splitting three
    # keeps the refresh's key and the agent's stream unchanged.
    _, refresh_key, new_rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=new_rng)

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
        target = jax.lax.stop_gradient(
            rewards + gamma * (1.0 - dones) * jnp.min(q_next, axis=0)
        )

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
    new_expert_critic_state = new_expert_critic_state.soft_update(tau=1.0)
    return agent_state.replace(
        expert_critic_state=new_expert_critic_state,
        expert_critic_params=jax.lax.stop_gradient(new_expert_critic_state.params),
    )


@dataclass(frozen=True)
class MCPretrain(Extension):
    """Monte-Carlo critic pre-training (``use_mc_critic_pretrain``).

    Regresses a frozen expert critic ``φ*`` on unbiased MC returns from
    expert rollouts, persists ``φ*`` + the value range ``v_min/v_max`` on
    the agent state, and — when ``use_online_light`` is set — applies a
    weak supervised nudge of the online critic toward ``φ*`` to reduce
    seed-to-seed variance at initialisation.

    Building ``φ*`` needs the agent's env, critic network and optimiser
    config, ``gamma`` and ``reward_scale``: the fields after the
    hyperparameters, which the agent fills through :meth:`bind_to_agent`
    (only SAC does). ``use_phi_refresh`` keeps ``φ*``'s optimiser state
    for a :class:`PhiRefresh` in the same stack.
    """

    expert_policy: Callable
    n_mc_steps: int = 10_000
    n_mc_episodes: int = 100
    n_steps: int = 5_000
    use_online_light: bool = True
    online_light_steps: int = 500
    online_light_lr_scale: float = 0.1
    # Agent context, bound by bind_to_agent.
    env_args: Any = None
    network_args: Any = None
    critic_optimizer_args: Any = None
    num_critics: int = 2
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    use_train_frac: bool = False
    augment_obs_with_expert_action: bool = False
    use_phi_refresh: bool = False
    name: str = "mc_pretrain"

    def bind_to_agent(self, **agent_context: Any) -> "MCPretrain":
        """Copy the agent's context (``env_args``, ``network_args``,
        ``critic_optimizer_args``, ``num_critics``, ``gamma``,
        ``reward_scale``, ``use_train_frac``,
        ``augment_obs_with_expert_action``) onto a new instance, and keep
        ``φ*``'s optimiser when the stack (``extensions``) holds a
        :class:`PhiRefresh`."""
        env_args = agent_context.get("env_args", None)
        if self.env_args is not None or env_args is None:
            return self
        return dataclasses.replace(
            self,
            env_args=env_args,
            network_args=agent_context.get("network_args"),
            critic_optimizer_args=agent_context.get("critic_optimizer_args"),
            num_critics=agent_context.get("num_critics", self.num_critics),
            gamma=agent_context.get("gamma"),
            reward_scale=agent_context.get("reward_scale"),
            use_train_frac=agent_context.get("use_train_frac", self.use_train_frac),
            augment_obs_with_expert_action=agent_context.get(
                "augment_obs_with_expert_action",
                self.augment_obs_with_expert_action,
            ),
            use_phi_refresh=any(
                isinstance(e, PhiRefresh) for e in agent_context.get("extensions", ())
            ),
        )

    def pretrain(
        self,
        agent_state: Any,
        ext_state: Any,
        ctx: ExtensionContext,
    ) -> tuple[Any, Any]:
        """Build the frozen expert critic ``φ*`` and persist it on ``agent_state``:
        a fresh critic regressed on unbiased MC returns sets
        ``expert_critic_params`` / ``expert_v_min`` / ``expert_v_max``,
        then (``use_online_light``) the online critic is nudged toward it.

        Raises ``ValueError`` when the agent did not bind its context.
        """
        context = ("env_args", "network_args", "critic_optimizer_args", "gamma")
        unbound = [f for f in (*context, "reward_scale") if getattr(self, f) is None]
        if unbound:
            raise ValueError(
                f"MCPretrain needs the agent context {unbound} to build phi*;"
                " the agent did not bind it (only SAC supports MCPretrain)."
            )

        # Local import to avoid a circular ``ajax.networks ↔
        # ajax.extensions`` import: networks.py doesn't depend on
        # extensions but the extension's pretrain phase does need the
        # critic builder.
        from ajax.environments.utils import get_action_dim
        from ajax.networks.networks import get_initialized_critic

        rng = ctx.rng
        max_timesteps = ctx.total_steps if self.use_train_frac else None
        expert_critic_state = get_initialized_critic(
            key=rng,
            env_config=self.env_args,
            critic_optimizer_config=self.critic_optimizer_args,
            network_config=self.network_args,
            num_critics=self.num_critics,
            max_timesteps=max_timesteps,
            extra_obs_dim=(
                get_action_dim(self.env_args.env, self.env_args.env_params)
                if self.augment_obs_with_expert_action
                else 0
            ),
        )
        (
            agent_state,
            frozen_expert_params,
            mc_obs_batched,
            mc_action_batched,
            mc_aux,
            expert_critic_state_trained,
        ) = pretrain_critic_mc(
            agent_state=agent_state,
            expert_critic_state=expert_critic_state,
            expert_policy=self.expert_policy,
            mode="gymnax" if check_env_is_gymnax(self.env_args.env) else "brax",
            env_args=self.env_args,
            gamma=self.gamma,
            reward_scale=self.reward_scale,
            n_mc_steps=self.n_mc_steps,
            n_mc_episodes=self.n_mc_episodes,
            n_steps=self.n_steps,
            max_timesteps=max_timesteps,
            augment_obs_with_expert_action=self.augment_obs_with_expert_action,
        )
        agent_state = agent_state.replace(
            expert_critic_params=frozen_expert_params,
            expert_v_min=mc_aux.v_min,
            expert_v_max=mc_aux.v_max,
            # Keep φ* optimizer state alive for periodic refresh (None when disabled)
            expert_critic_state=(
                expert_critic_state_trained if self.use_phi_refresh else None
            ),
        )
        jax.debug.print(
            "[MC pretrain] loss: {i:.4f} -> {f:.4f}  |  "
            "Q(s,a*) mean={qm:.1f}  min={qn:.1f}  max={qx:.1f}",
            i=mc_aux.initial_loss,
            f=mc_aux.final_loss,
            qm=mc_aux.q_expert_mean,
            qn=mc_aux.q_expert_min,
            qx=mc_aux.q_expert_max,
        )
        if self.use_online_light:
            agent_state = pretrain_critic_online_light(
                agent_state,
                mc_obs_batched,
                mc_action_batched,
                n_steps=self.online_light_steps,
                lr_scale=self.online_light_lr_scale,
            )
            jax.debug.print(
                "[Online critic light pretrain] done ({n} steps, lr_scale={s})",
                n=self.online_light_steps,
                s=self.online_light_lr_scale,
            )
        return agent_state, ext_state


@dataclass(frozen=True)
class BellmanPretrain(Extension):
    """Legacy Bellman-bootstrapped critic pre-training.

    Equivalent to ``use_bellman_critic_pretrain`` — mutually exclusive
    with :class:`MCPretrain`. Bootstraps from an untrained critic, so the
    pre-trained values are biased; kept for ablation parity.
    """

    expert_policy: Callable
    n_steps: int = 5_000
    name: str = "bellman_pretrain"


@dataclass(frozen=True)
class PhiRefresh(Extension):
    """Periodic self-consistent refresh of the frozen expert critic.

    Equivalent to ``use_phi_refresh``: every ``interval`` steps run
    ``steps`` self-consistent Bellman updates on expert-flagged buffer
    transitions, keeping ``φ*`` aligned with the live data distribution.
    Requires :class:`MCPretrain` (which creates the refreshable ``φ*``).

    ``buffer``, ``gamma`` and ``reward_scale`` are populated by the
    SAC training factory's auto-append shim from
    ``agent_config.gamma`` / ``agent_config.reward_scale`` / the agent
    ``buffer``. User-constructed ``PhiRefresh()`` instances leave them
    ``None``; the SAC factory copies the resolved values onto the
    instance with ``dataclasses.replace`` so the ``post_update`` math
    has everything it needs without threading per-step kwargs through
    the phase API.
    """

    expert_policy: Callable
    interval: int = 500
    steps: int = 20
    buffer: Any = None
    gamma: Optional[float] = None
    reward_scale: Optional[float] = None
    name: str = "phi_refresh"

    def bind_to_agent(self, **agent_context: Any) -> "PhiRefresh":
        """Populate ``buffer`` / ``gamma`` / ``reward_scale`` from the agent factory.

        Self-contained replacement for the legacy SAC-side
        ``_inject_policy_extensions_context`` helper: the agent factory
        calls ``stack.bind_to_agent(buffer=..., gamma=..., reward_scale=...,
        ...)`` and this method copies the kwargs it cares about onto a
        new frozen instance. Unrecognised kwargs are ignored, so the
        same call works uniformly across the whole stack.
        """
        buffer = agent_context.get("buffer", None)
        gamma = agent_context.get("gamma", None)
        reward_scale = agent_context.get("reward_scale", None)
        if self.buffer is not None or buffer is None:
            return self
        return dataclasses.replace(
            self, buffer=buffer, gamma=gamma, reward_scale=reward_scale
        )

    def post_update(
        self,
        agent_state: Any,
        ext_state: Any,
        ctx: ExtensionContext,
    ) -> tuple[Any, Any]:
        # ``post_update`` runs once per update iteration, after the
        # update. The interval gate below is byte-equivalent to the
        # pre-refactor ``make_runtime_maintenance`` callable: refresh
        # whenever ``timestep % interval == 0``; otherwise pass through
        # unchanged.
        del ctx
        if self.buffer is None or self.gamma is None or self.reward_scale is None:
            # Unconfigured PhiRefresh — no-op (the SAC factory's
            # auto-append shim is what populates these fields).
            return agent_state, ext_state
        interval = self.interval
        steps = self.steps
        gamma = self.gamma
        reward_scale = self.reward_scale
        expert_policy = self.expert_policy
        buffer = self.buffer

        def _do_refresh(s: Any) -> Any:
            return refresh_phi_star(
                s, buffer, steps, gamma, reward_scale, expert_policy
            )

        new_state = jax.lax.cond(
            agent_state.collector_state.timestep % interval == 0,
            _do_refresh,
            lambda s: s,
            operand=agent_state,
        )
        return new_state, ext_state


__all__ = ["MCPretrain", "BellmanPretrain", "PhiRefresh"]
