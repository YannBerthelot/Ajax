"""TD-target modifier research features as composable :class:`Extension`s.

These reshape the SAC Bellman target before it enters the critic loss
(the ``on_target`` phase):

* :class:`IBRL`               — add the positive gap
  ``γ(1-d)·max(Q_expert - Q_policy, 0)`` so the value function matches an
  argmax action-selection policy.
* :class:`LCBGatedBootstrap`  — soft-blend the policy and expert TD
  targets by an LCB-scored sigmoid gate.
* :class:`CriticBlend`        — warmup-decaying blend of the Bellman target
  with the frozen expert critic's.
* :class:`MCVarianceCorrection` — replace high-ensemble-variance Bellman
  targets with the MC-pretrained oracle.
* :class:`ValueBox`           — value-threshold expert action override
  during collection (the ``action`` phase).

The ``batch`` argument to ``on_target`` is the dict SAC's
``update_value_functions`` builds: ``observations``, ``actions``,
``next_observations``, ``rewards`` (unscaled), ``reward_scale``, ``dones``,
``rng_key``, ``q_preds``, ``gamma``, ``raw_next_observations`` (the
environment's own next observations, which an extension's expert reads:
SAC keeps them when an extension carries an expert) and ``recurrent``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional, Tuple

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_pi
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.expert import expert_next_actions, reads_stored_expert_actions
from ajax.extensions.exploration import lcb_beta, lcb_score
from ajax.networks.networks import phi_star_critic_state, predict_value


def blend_modify_target(
    target_q: jax.Array,
    expert_target: jax.Array,
    alpha_blend: jax.Array,
) -> jax.Array:
    """Blended Bellman: (1-alpha)*y_bellman + alpha*y_expert."""
    return (1.0 - alpha_blend) * target_q + alpha_blend * expert_target


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
            critic_state=phi_star_critic_state(critic_state),
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
    v_min: jax.Array | float,
    v_max: jax.Array | float,
    train_frac: jax.Array,
) -> jax.Array:
    """Curriculum threshold: v_min + (v_max - v_min) * train_frac."""
    return v_min + (v_max - v_min) * train_frac


def box_entered(in_box: jax.Array, last_in_box: Optional[jax.Array]) -> jax.Array:
    """Where the step enters the value box: inside it now, not before."""
    if last_in_box is None:
        last_in_box = jnp.zeros_like(in_box)
    return (last_in_box < 0.5) & (in_box > 0.5)


def box_compute_state(
    obs: jax.Array,
    a_exp: jax.Array,
    critic_state,
    expert_critic_params,
    threshold: jax.Array,
    last_in_box: Optional[jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute box membership and entry bonus, the expert's value
    ``V(s) = min Q_phi*(s, a_exp)`` at its action ``a_exp``.

    Returns (in_box, entry_bonus, v_box).
    """
    v_box = jnp.min(
        predict_value(
            critic_state=phi_star_critic_state(critic_state),
            critic_params=expert_critic_params,
            x=jnp.concatenate([obs, a_exp], axis=-1),
        ),
        axis=0,
    )
    in_box = v_box > threshold
    entry_bonus = jnp.where(
        box_entered(in_box, last_in_box), v_box, jnp.zeros_like(v_box)
    )
    return in_box, entry_bonus, v_box


def box_action_override(
    action: jax.Array,
    expert_action: jax.Array,
    in_box: jax.Array,
) -> jax.Array:
    """Override with expert action inside the value box."""
    return jnp.where(in_box, expert_action, action)


def _next_target_q(
    ext: Any, agent_state: Any, batch: dict
) -> tuple[jax.Array, jax.Array]:
    """The target critics' ensemble at ``s'`` for the expert's action and
    for a policy sample: ``(Q_target(s', a_E), Q_target(s', a'))``."""
    next_observations = batch["next_observations"]
    critic_state = agent_state.critic_state

    def q(actions: jax.Array) -> jax.Array:
        x = jnp.concatenate((next_observations, actions), axis=-1)
        return predict_value(critic_state, critic_state.target_params, x)

    next_expert_actions = expert_next_actions(
        ext, batch["raw_next_observations"], batch["next_a_expert"]
    )
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
    """IBRL bootstrap (Hu, Mirchandani and Sadigh, *Imitation Bootstrapped
    Reinforcement Learning*, 2023).

    Adds ``γ(1-d)·max(min Q_target(s', a_expert') - min Q_target(s', a_π'),
    0)`` to the Bellman target so the value function is consistent with an
    argmax(Q_expert, Q_policy) action-selection rule.
    """

    expert_policy: Callable
    reads_stored_expert: bool = False  # bound: the agent's expert
    name: str = "ibrl"
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def bind_to_agent(self, **agent_context: Any) -> "IBRL":
        """Read the expert's stored actions when it is the agent's."""
        stored = reads_stored_expert_actions(self, agent_context)
        return dataclasses.replace(self, reads_stored_expert=stored)

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
        q_expert, q_policy = _next_target_q(self, agent_state, batch)
        min_q_expert = jnp.min(q_expert, axis=0, keepdims=False)
        min_q_policy = jnp.min(q_policy, axis=0, keepdims=False)
        gap = jax.lax.stop_gradient(
            gamma * (1.0 - dones) * jnp.maximum(min_q_expert - min_q_policy, 0.0)
        )
        return target + gap


@dataclass(frozen=True)
class LCBGatedBootstrap(Extension):
    """LCB-gated bootstrap.

    Scores the policy and expert next-actions by an LCB rule and
    soft-blends the two TD targets with ``p_expert =
    σ((score_e - score_p)/lcb_temperature)``. ``β`` anneals over training.
    """

    expert_policy: Callable
    lcb_beta_init: float = 1.0
    lcb_beta_decay_k: float = 2.0
    lcb_temperature: float = 1.0
    reads_stored_expert: bool = False  # bound: the agent's expert
    name: str = "lcb_gated_bootstrap"
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def bind_to_agent(self, **agent_context: Any) -> "LCBGatedBootstrap":
        """Read the expert's stored actions when it is the agent's."""
        stored = reads_stored_expert_actions(self, agent_context)
        return dataclasses.replace(self, reads_stored_expert=stored)

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state
        # beta anneals as the EDGE gate's, on the collector's timestep.
        dones, gamma = batch["dones"], batch["gamma"]
        q_expert, q_policy = _next_target_q(self, agent_state, batch)
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
    """Warmup-decaying blend of the Bellman target with the expert's.

    ``(1-α)·y_bellman + α·y_expert`` with ``y_expert = r + γ(1-d)·V_E(s')``,
    the one-step target on the frozen expert critic ``V_E(s') = φ*(s',
    a_E(s'))``, so the blend stays a Bellman target (``V_E(s')`` alone has
    no reward, discount or terminal mask). ``α`` decays linearly to 0 over
    ``critic_warmup_frac`` of training. Requires the frozen expert critic
    :class:`MCPretrain` fits (refused without it).
    """

    expert_policy: Callable
    critic_warmup_frac: float = 0.15
    reads_stored_expert: bool = False  # bound: the agent's expert
    name: str = "critic_blend"
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def reads_phi_star(self) -> bool:
        return True

    def bind_to_agent(self, **agent_context: Any) -> "CriticBlend":
        """Read the expert's stored actions when it is the agent's."""
        stored = reads_stored_expert_actions(self, agent_context)
        return dataclasses.replace(self, reads_stored_expert=stored)

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
        a_expert_next = expert_next_actions(
            self, batch["raw_next_observations"], batch["next_a_expert"]
        )
        v_expert_next = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=phi_star_critic_state(agent_state.critic_state),
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([next_observations, a_expert_next], axis=-1),
                ),
                axis=0,
            )
        )
        rewards = batch["reward_scale"] * batch["rewards"]
        expert_target = (
            rewards + batch["gamma"] * (1.0 - batch["dones"]) * v_expert_next
        )
        total_timesteps = max(int(ctx.total_steps), 1)
        train_frac = agent_state.collector_state.timestep / total_timesteps
        alpha_blend_val = jnp.maximum(1.0 - train_frac / self.critic_warmup_frac, 0.0)
        target_new = blend_modify_target(target, expert_target, alpha_blend_val)
        return jax.lax.stop_gradient(target_new)


@dataclass(frozen=True)
class MCVarianceCorrection(Extension):
    """Replace high-variance Bellman targets with the MC oracle.

    Where the inter-critic variance exceeds ``threshold`` the Bellman
    target is swapped for the MC-pretrained expert critic's estimate.
    Requires :class:`MCPretrain` (refused without it).
    """

    threshold: float
    name: str = "mc_variance_correction"

    def reads_phi_star(self) -> bool:
        return True

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state, ctx
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
    """Value-threshold expert-action override.

    During collection, override the policy action with the expert's
    whenever the frozen-expert value ``V_expert(s)`` exceeds a curriculum
    threshold that ramps from ``v_min`` to ``v_max`` over training.
    Requires :class:`MCPretrain` (which supplies phi*, ``v_min`` and
    ``v_max``; refused without it).

    The :meth:`action` phase reads the SAC action pipeline's per-step dict
    (``env_action``, ``expert_action``, the action of SAC's
    ``expert_policy``, at which it also measures the expert's value, and
    ``observations``, the critics' input, augmented when it is) and the box
    bounds off the agent state (so a resumed run keeps them), and writes
    ``in_value_box``, ``entry_bonus`` and ``box_entry`` back into the dict
    for the pipeline to record on the transition. An entry ends the agent's
    episode: the expert's value, paid as the bonus, replaces the bootstrap,
    whatever its sign. The override (:func:`box_action_override`) comes
    after the warmup / post-warmup choice (``action_slot``).
    """

    name: str = "value_box"
    action_slot: ClassVar[str] = "post_warmup"

    def reads_phi_star(self) -> bool:
        return True

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Override with ``a_expert`` inside the value box, its bounds the
        MC-pretrain v_min/v_max on :class:`SACState`; writes
        ``in_value_box``, ``entry_bonus`` and ``box_entry`` into ``obs`` for
        the pipeline's bookkeeping (reward shaping, the entry's
        termination, ``is_expert_flag``).
        """
        del ext_state, rng
        env_action = obs["env_action"]
        expert_action = obs["expert_action"]

        train_frac = ctx.step / ctx.total_steps
        threshold = box_compute_threshold(
            agent_state.expert_v_min, agent_state.expert_v_max, train_frac
        )
        in_value_box, entry_bonus, _ = box_compute_state(
            obs["observations"],
            expert_action,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            threshold,
            agent_state.collector_state.last_in_box,
        )
        obs["in_value_box"] = in_value_box
        obs["entry_bonus"] = entry_bonus
        obs["box_entry"] = box_entered(
            in_value_box, agent_state.collector_state.last_in_box
        ).astype(jnp.float32)
        return box_action_override(env_action, expert_action, in_value_box)


__all__ = [
    "IBRL",
    "CriticBlend",
    "LCBGatedBootstrap",
    "MCVarianceCorrection",
    "ValueBox",
]
