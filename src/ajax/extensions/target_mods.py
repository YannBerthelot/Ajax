"""TD-target modifier research features as composable :class:`Extension`s.

These reshape the SAC Bellman target before it enters the critic loss
(the ``on_target`` phase):

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

The ``batch`` argument to ``on_target`` is the dict SAC's
``update_value_functions`` builds: ``observations``, ``actions``,
``next_observations``, ``dones``, ``rng_key``, ``q_preds``, ``gamma``,
``augment_obs_with_expert_action``, ``recurrent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional, Tuple

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_pi
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.exploration import lcb_beta, lcb_score
from ajax.networks.networks import predict_value


def blend_modify_target(
    target_q: jax.Array,
    v_expert_next: jax.Array,
    alpha_blend: jax.Array,
) -> jax.Array:
    """Blended Bellman: (1-alpha)*y_bellman + alpha*V*(s')."""
    return (1.0 - alpha_blend) * target_q + alpha_blend * v_expert_next


def mc_correction_modify_target(
    target_q: jax.Array,
    critic_state,
    critic_params_mc,
    observations: jax.Array,
    actions: jax.Array,
    q_var: jax.Array,
    mc_variance_threshold: float,
) -> jax.Array:
    """Replace high-variance Bellman targets with MC-pretrained oracle estimate."""
    uncertain_mask = q_var > mc_variance_threshold
    q_mc_target = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=critic_params_mc,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
        ),
        axis=0,
    )
    return jnp.where(
        uncertain_mask[..., None],
        jax.lax.stop_gradient(q_mc_target),
        target_q,
    )


def box_compute_threshold(
    v_min: jax.Array,
    v_max: jax.Array,
    train_frac: jax.Array,
) -> jax.Array:
    """Curriculum threshold: v_min + (v_max - v_min) * train_frac."""
    return v_min + (v_max - v_min) * train_frac


def box_compute_state(
    obs: jax.Array,
    raw_obs: jax.Array,
    expert_policy,
    critic_state,
    expert_critic_params,
    threshold: jax.Array,
    last_in_box: Optional[jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute box membership and entry bonus.

    Returns (in_box, entry_bonus, v_box).
    """
    a_exp = jax.lax.stop_gradient(expert_policy(raw_obs))
    v_box = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=expert_critic_params,
            x=jnp.concatenate([obs, a_exp], axis=-1),
        ),
        axis=0,
    )
    in_box = v_box > threshold

    if last_in_box is None:
        last_in_box = jnp.zeros_like(in_box)

    entry_bonus = jnp.where(
        (last_in_box < 0.5) & (in_box > 0.5),
        v_box,
        jnp.zeros_like(v_box),
    )
    return in_box, entry_bonus, v_box


def box_action_override(
    action: jax.Array,
    expert_action: jax.Array,
    in_box: jax.Array,
) -> jax.Array:
    """Override with expert action inside the value box."""
    return jnp.where(in_box, expert_action, action)


def _next_raw(batch: dict) -> jax.Array:
    """Strip the trailing expert-action dims if obs is augmented."""
    next_obs = batch["next_observations"]
    if batch.get("augment_obs_with_expert_action", False):
        return next_obs[..., :-1]
    return next_obs


def _next_target_q(
    expert_policy: Callable, agent_state: Any, batch: dict
) -> tuple[jax.Array, jax.Array]:
    """The target critics' ensemble at ``s'`` for the expert's action and
    for a policy sample: ``(Q_target(s', a_E), Q_target(s', a'))``."""
    next_observations = batch["next_observations"]
    critic_state = agent_state.critic_state

    def q(actions: jax.Array) -> jax.Array:
        x = jnp.concatenate((next_observations, actions), axis=-1)
        return predict_value(critic_state, critic_state.target_params, x)

    next_expert_actions = jax.lax.stop_gradient(expert_policy(_next_raw(batch)))
    next_pi, _ = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=next_observations,
        done=batch["dones"],
        recurrent=batch.get("recurrent", False),
    )
    key, _ = jax.random.split(batch["rng_key"])
    next_actions, _ = next_pi.sample_and_log_prob(seed=key)
    return q(next_expert_actions), q(next_actions)


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
        del ext_state, ctx
        dones, gamma = batch["dones"], batch["gamma"]
        q_expert, q_policy = _next_target_q(self.expert_policy, agent_state, batch)
        min_q_expert = jnp.min(q_expert, axis=0, keepdims=False)
        min_q_policy = jnp.min(q_policy, axis=0, keepdims=False)
        gap = jax.lax.stop_gradient(
            gamma * (1.0 - dones) * jnp.maximum(min_q_expert - min_q_policy, 0.0)
        )
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
        # Score each next action by its LCB (beta annealed as the EDGE
        # gate's, on the collector's timestep), then soft-blend the policy
        # and expert TD targets by
        #   P_expert = σ((score_e - score_p) / lcb_temperature).
        dones, gamma = batch["dones"], batch["gamma"]
        q_expert, q_policy = _next_target_q(self.expert_policy, agent_state, batch)
        beta = lcb_beta(
            agent_state.collector_state.timestep,
            ctx.total_steps,
            self.lcb_beta_init,
            self.lcb_beta_decay_k,
        )
        score_e, score_p = lcb_score(q_expert, beta), lcb_score(q_policy, beta)
        q_min_e = jnp.min(q_expert, axis=0, keepdims=False)
        q_min_p = jnp.min(q_policy, axis=0, keepdims=False)
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
        target_new = blend_modify_target(target, v_expert_next, alpha_blend_val)
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
        return mc_correction_modify_target(
            target,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            observations,
            actions,
            q_var,
            self.threshold,
        )


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
    ``post_warmup_action``), the box bounds off the agent state (so a
    resumed run keeps them) and writes ``in_value_box`` / ``entry_bonus``
    back into the dict so the pipeline can record them on the
    transition. The substitution itself is
    ``box_action_override`` above and is
    applied AFTER the warmup/post-warmup choice — matching the legacy
    ``make_action_pipeline`` ordering byte-for-byte.
    """

    expert_policy: Callable
    name: str = "value_box"
    action_slot: ClassVar[str] = "post_warmup"

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
        :mod:`ajax.extensions.exploration` for the convention). The box
        bounds are the MC-pretrain v_min/v_max stored on
        :class:`SACState`. Returns the action with the in-box rows
        replaced by the expert action; writes ``in_value_box`` and
        ``entry_bonus`` into ``obs`` for pipeline-side bookkeeping
        (buffer-write suppression, reward shaping, ``is_expert_flag``).
        """
        del ext_state, rng, ctx
        env_action = obs["env_action"]
        expert_action = obs["expert_action"]
        total_timesteps = obs["total_timesteps"]

        train_frac = agent_state.collector_state.timestep / total_timesteps
        threshold = box_compute_threshold(
            agent_state.expert_v_min, agent_state.expert_v_max, train_frac
        )
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
