"""TD-target modifier research features as composable :class:`Extension`s.

These reshape the SAC Bellman target before it enters the critic loss
(the ``on_target`` phase). The four target-mod extensions each implement
their math directly on :meth:`Extension.on_target`; the SAC loop folds
them through ``stack.on_target(...)``. The pre-refactor flag-driven
``make_target_modifier`` builder used to assemble the same four pieces
into a single callable — that builder has been removed; the math lives
here.

* :class:`IBRL`               — ``ibrl_bootstrap``: add the positive gap
  ``γ(1-d)·max(Q_expert - Q_policy, 0)`` so the value function matches an
  argmax action-selection policy.
* :class:`LCBGatedBootstrap`  — ``lcb_gated_bootstrap``: soft-blend the
  policy and expert TD targets by an LCB-scored sigmoid gate.
* :class:`CriticBlend`        — ``use_critic_blend``: warmup-decaying
  blend of the Bellman target with the frozen-expert value estimate.
* :class:`MCVarianceCorrection` — ``mc_variance_threshold``: replace
  high-ensemble-variance Bellman targets with the MC-pretrained oracle.
* :class:`ValueBox`           — ``use_box``: value-threshold expert
  action override during collection (the ``action`` phase).

The ``batch`` argument to ``on_target`` is a dict carrying every input
each modifier needs, threaded through by ``update_value_functions`` in
``ajax.agents.SAC.sac``:

``observations``, ``actions``, ``next_observations``, ``dones``,
``rng_key``, ``q_preds``, ``gamma``, ``augment_obs_with_expert_action``,
``recurrent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_pi
from ajax.extensions.base import Extension, ExtensionContext
from ajax.modules.expert import (
    blend_modify_target,
    mc_correction_modify_target,
)
from ajax.modules.exploration import (
    box_action_override,
    box_compute_state,
    box_compute_threshold,
)
from ajax.networks.networks import predict_value


def _next_raw(batch: dict) -> jax.Array:
    """Strip the trailing expert-action dims if obs is augmented."""
    next_obs = batch["next_observations"]
    if batch.get("augment_obs_with_expert_action", False):
        return next_obs[..., :-1]
    return next_obs


@dataclass(frozen=True)
class IBRL(Extension):
    """IBRL bootstrap (``ibrl_bootstrap``).

    Adds ``γ(1-d)·max(min Q_target(s', a_expert') - min Q_target(s', a_π'),
    0)`` to the Bellman target so the value function is consistent with an
    argmax(Q_expert, Q_policy) action-selection rule.
    """

    expert_policy: Callable
    name: str = "ibrl"

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        gamma = batch["gamma"]
        rng = batch["rng_key"]
        recurrent = batch.get("recurrent", False)

        next_expert_actions = jax.lax.stop_gradient(
            self.expert_policy(_next_raw(batch))
        )
        q_targets_expert = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.target_params,
            x=jnp.concatenate((next_observations, next_expert_actions), axis=-1),
        )
        min_q_expert = jnp.min(q_targets_expert, axis=0, keepdims=False)

        next_pi, _ = get_pi(
            actor_state=agent_state.actor_state,
            actor_params=agent_state.actor_state.params,
            obs=next_observations,
            done=dones,
            recurrent=recurrent,
        )
        ibrl_key, _ = jax.random.split(rng)
        next_actions_ibrl, _ = next_pi.sample_and_log_prob(seed=ibrl_key)
        q_targets_policy = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.target_params,
            x=jnp.concatenate((next_observations, next_actions_ibrl), axis=-1),
        )
        min_q_policy = jnp.min(q_targets_policy, axis=0, keepdims=False)

        gap = jax.lax.stop_gradient(
            gamma * (1.0 - dones) * jnp.maximum(min_q_expert - min_q_policy, 0.0)
        )
        del ctx
        return target + gap


@dataclass(frozen=True)
class LCBGatedBootstrap(Extension):
    """LCB-gated bootstrap (``lcb_gated_bootstrap``).

    Scores the policy and expert next-actions by an LCB rule and
    soft-blends the two TD targets with ``p_expert =
    σ((score_e - score_p)/lcb_temperature)``. ``β`` anneals over training.
    """

    expert_policy: Callable
    lcb_beta_init: float = 1.0
    lcb_beta_decay_k: float = 2.0
    lcb_temperature: float = 1.0
    name: str = "lcb_gated_bootstrap"

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state
        # LCB-gated bootstrap: at s', score each candidate by
        #   score(a) = Q_min(s', a) - β · (Q_max(s', a) - Q_min(s', a))
        # then soft-blend the policy and expert TD targets by
        #   P_expert = σ((score_e - score_p) / lcb_temperature).
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        gamma = batch["gamma"]
        rng = batch["rng_key"]
        recurrent = batch.get("recurrent", False)

        next_expert_actions = jax.lax.stop_gradient(
            self.expert_policy(_next_raw(batch))
        )
        q_targets_expert = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.target_params,
            x=jnp.concatenate((next_observations, next_expert_actions), axis=-1),
        )
        next_pi_lcb, _ = get_pi(
            actor_state=agent_state.actor_state,
            actor_params=agent_state.actor_state.params,
            obs=next_observations,
            done=dones,
            recurrent=recurrent,
        )
        lcb_key, _ = jax.random.split(rng)
        next_actions_lcb, _ = next_pi_lcb.sample_and_log_prob(seed=lcb_key)
        q_targets_policy = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.target_params,
            x=jnp.concatenate((next_observations, next_actions_lcb), axis=-1),
        )
        q_min_e = jnp.min(q_targets_expert, axis=0, keepdims=False)
        q_max_e = jnp.max(q_targets_expert, axis=0, keepdims=False)
        q_min_p = jnp.min(q_targets_policy, axis=0, keepdims=False)
        q_max_p = jnp.max(q_targets_policy, axis=0, keepdims=False)
        # Anneal beta over training same as the action-selection gate. Use
        # the agent_state's collector_state.timestep / total_timesteps so
        # the rate matches the pre-refactor make_target_modifier exactly
        # (ctx.step / ctx.total_steps would also work but we keep the
        # historical numerics by reading the same fields).
        total_timesteps = max(int(ctx.total_steps), 1)
        train_frac = jnp.clip(
            agent_state.collector_state.timestep / total_timesteps,
            0.0,
            1.0,
        )
        beta_eff = self.lcb_beta_init * jnp.power(
            1.0 - train_frac, self.lcb_beta_decay_k
        )
        score_e = q_min_e - beta_eff * (q_max_e - q_min_e)
        score_p = q_min_p - beta_eff * (q_max_p - q_min_p)
        p_expert = jax.nn.sigmoid(
            (score_e - score_p) / jnp.maximum(self.lcb_temperature, 1e-6)
        )
        # Bellman target with the LCB-gated next action. We blend the
        # min-Q part only (entropy term stays on the policy branch since
        # the expert is deterministic — log_prob is undefined).
        min_q_lcb = (1.0 - p_expert) * q_min_p + p_expert * q_min_e
        gap_lcb = jax.lax.stop_gradient(gamma * (1.0 - dones) * (min_q_lcb - q_min_p))
        return target + gap_lcb


@dataclass(frozen=True)
class CriticBlend(Extension):
    """Warmup-decaying blend of the Bellman target with V_expert.

    Equivalent to ``use_critic_blend``: ``(1-α)·y_bellman + α·V_expert(s')``
    with ``α`` decaying linearly to 0 over ``critic_warmup_frac`` of
    training. Requires a frozen expert critic (MC pre-training).
    """

    expert_policy: Callable
    critic_warmup_frac: float = 0.15
    name: str = "critic_blend"

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state
        # No-op when the frozen expert critic has not been populated (no
        # MC pre-training in the stack). Matches the pre-refactor guard
        # `if has_blend and agent_state.expert_critic_params is not None`.
        if agent_state.expert_critic_params is None:
            return target

        next_observations = batch["next_observations"]
        # CriticBlend always strips the trailing expert-action dim
        # (matches the pre-refactor `next_raw = next_observations[..., :-1]`
        # path, which was unconditional in the blend branch).
        next_raw = next_observations[..., :-1]
        a_expert_next = jax.lax.stop_gradient(self.expert_policy(next_raw))
        v_expert_next = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([next_observations, a_expert_next], axis=-1),
                ),
                axis=0,
            )
        )
        total_timesteps = max(int(ctx.total_steps), 1)
        train_frac = agent_state.collector_state.timestep / total_timesteps
        alpha_blend_val = jnp.maximum(1.0 - train_frac / self.critic_warmup_frac, 0.0)
        target_new, _ = blend_modify_target(target, v_expert_next, alpha_blend_val)
        return jax.lax.stop_gradient(target_new)


@dataclass(frozen=True)
class MCVarianceCorrection(Extension):
    """Replace high-variance Bellman targets with the MC oracle.

    Equivalent to ``mc_variance_threshold``: where the inter-critic
    variance exceeds ``threshold`` the Bellman target is swapped for the
    MC-pretrained expert critic's estimate. Requires MC pre-training.
    """

    threshold: float
    name: str = "mc_variance_correction"

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state, ctx
        # Same guard as the legacy `has_mc and expert_critic_params is not
        # None` branch — silent no-op without MC pre-training.
        if agent_state.expert_critic_params is None:
            return target
        observations = batch["observations"]
        actions = batch["actions"]
        q_preds = batch["q_preds"]
        q_var = q_preds.var(axis=0)[..., 0]
        target_new, _ = mc_correction_modify_target(
            target,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            observations,
            actions,
            q_var,
            self.threshold,
        )
        return target_new


@dataclass(frozen=True)
class ValueBox(Extension):
    """Value-threshold expert-action override (``use_box``).

    During collection, override the policy action with the expert's
    whenever the frozen-expert value ``V_expert(s)`` exceeds a curriculum
    threshold that ramps from ``v_min`` to ``v_max`` over training.
    Requires MC pre-training (which supplies ``v_min`` / ``v_max``).

    The :meth:`action` phase owns the override math. It reads the
    pre-computed per-step quantities off the SAC action pipeline's batch
    dict (``policy_action``, ``expert_action``, the running
    ``post_warmup_action``, the box bounds ``box_v_min`` / ``box_v_max``
    and the previous step's ``last_in_box`` flag) and writes
    ``in_value_box`` / ``entry_bonus`` back into the dict so the pipeline
    can record them on the transition. The substitution itself is
    ``box_action_override`` from :mod:`ajax.modules.exploration` and is
    applied AFTER the warmup/post-warmup choice — matching the legacy
    ``make_action_pipeline`` ordering byte-for-byte.
    """

    expert_policy: Callable
    name: str = "value_box"

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Override with ``a_expert`` inside the value box.

        ``obs`` is the SAC action pipeline's per-step batch dict (see
        :mod:`ajax.extensions.exploration` for the convention). ``box_v_min``
        and ``box_v_max`` come from the MC-pretrain v_min/v_max written
        onto :class:`SACState`. Returns the action with the in-box rows
        replaced by the expert action; writes ``in_value_box`` and
        ``entry_bonus`` into ``obs`` for pipeline-side bookkeeping
        (buffer-write suppression, reward shaping, ``is_expert_flag``).
        """
        del ext_state, rng, ctx
        env_action = obs["env_action"]
        expert_action = obs["expert_action"]
        box_v_min = obs["box_v_min"]
        box_v_max = obs["box_v_max"]
        total_timesteps = obs["total_timesteps"]

        train_frac = agent_state.collector_state.timestep / total_timesteps
        threshold = box_compute_threshold(box_v_min, box_v_max, train_frac)
        last_obs = agent_state.collector_state.last_obs
        raw_obs = obs.get("raw_obs", None)
        raw_for_box = raw_obs if raw_obs is not None else last_obs[..., :-1]
        in_value_box, entry_bonus, _ = box_compute_state(
            last_obs,
            raw_for_box,
            self.expert_policy,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            threshold,
            agent_state.collector_state.last_in_box,
        )
        obs["in_value_box"] = in_value_box
        obs["entry_bonus"] = entry_bonus
        # ValueBox lives at the end of the action chain (legacy ordering:
        # AFTER the warmup vs post-warmup ``jax.lax.cond``). It rewrites
        # ``env_action`` directly.
        return box_action_override(env_action, expert_action, in_value_box)


__all__ = [
    "IBRL",
    "LCBGatedBootstrap",
    "CriticBlend",
    "MCVarianceCorrection",
    "ValueBox",
]
