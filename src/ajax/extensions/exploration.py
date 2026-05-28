"""Exploration research features as composable :class:`Extension`s.

EDGE (Expert Decayed Guided Exploration) substitutes the expert action
for the policy action during collection, gated by a value comparison
that decays over training. The six gating variants of the pre-refactor
``train_SAC.py`` are exposed here as a single :class:`EDGEExploration`
extension parametrised by ``gate``. The gate math itself lives unchanged
in :mod:`ajax.modules.exploration`; this module is the thin Extension
adapter that calls those helpers from :meth:`EDGEExploration.action`.

* :class:`EDGEExploration` — one extension parametrised by ``gate``:
  ``"fixed"`` / ``"argmax"`` / ``"boltzmann"`` (value-gap gates) or
  ``"lcb"`` / ``"argmax_lcb"`` / ``"thompson"`` (quality-aware gates).

The SAC action pipeline pre-computes the per-step quantities every gate
needs (``policy_action``, ``expert_action``, ``obs_for_edge``, ...) and
threads them through the ``obs`` argument as a dict — mirroring the
``batch``-dict convention the ``on_target`` / ``actor_loss`` phases
already use to plumb call-site context into extension methods. The
Extension's job is to PICK an action (or return ``None`` to defer); the
SAC pipeline owns the surrounding bookkeeping (``is_expert_flag``,
``buffer_action``, expert-state threading). Gate randomness is threaded
through ``obs["gate_rng"]`` (read+overwritten in place) so the rng tree
matches the pre-refactor pipeline byte-for-byte — the framework's
per-extension ``rng`` split (``stack._keys``) would otherwise diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from ajax.extensions.base import Extension, ExtensionContext
from ajax.modules.exploration import (
    edge_argmax_gate,
    edge_boltzmann_gate,
    edge_compute_asym_scores,
    edge_compute_decay,
    edge_compute_lcb_scores,
    edge_compute_thompson_stats,
    edge_compute_value_gap,
    edge_fixed_gate,
    edge_lcb_argmax_gate,
    edge_lcb_gate,
    edge_thompson_gate,
)

_VALID_GATES = frozenset(
    {"fixed", "argmax", "boltzmann", "lcb", "argmax_lcb", "thompson"}
)


@dataclass(frozen=True)
class EDGEExploration(Extension):
    """Expert Decayed Guided Exploration during collection.

    ``gate`` selects the substitution rule (see module docstring). The
    remaining fields mirror the ``train_SAC.py`` flags one-for-one:

    * ``decay_frac``      — value-gap gates decay to 0 over this fraction.
    * ``tau``             — Boltzmann temperature on the value gap.
    * ``fixed_prob``      — substitution probability for the fixed gate.
    * ``lcb_beta_init`` / ``lcb_beta_decay_k`` — LCB pessimism schedule.
    * ``lcb_temperature`` — softmax temperature of the LCB / Thompson gate.
    * ``lcb_asymmetric``  — LCB on the expert arm, UCB on the policy arm.
    * ``epsilon_floor``   — minimum policy-pick probability (Thompson).
    """

    expert_policy: Callable
    gate: str = "fixed"
    decay_frac: float = 0.30
    tau: float = 1.0
    fixed_prob: float = 0.5
    lcb_beta_init: float = 1.0
    lcb_beta_decay_k: float = 2.0
    lcb_temperature: float = 1.0
    lcb_asymmetric: bool = False
    epsilon_floor: float = 0.0
    name: str = "edge_exploration"

    def __post_init__(self):
        if self.gate not in _VALID_GATES:
            raise ValueError(
                f"EDGEExploration.gate must be one of {sorted(_VALID_GATES)}, "
                f"got {self.gate!r}."
            )

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Apply the configured EDGE gate to substitute the policy action.

        ``obs`` is the SAC action pipeline's per-step batch dict (a
        Python ``dict``, not a tensor — duck-typed; mirrors the
        ``batch``-dict convention of the ``on_target`` / ``actor_loss``
        phases). The pipeline pre-computes every operand the six gate
        variants share: ``policy_action``, ``expert_action``,
        ``obs_for_edge`` (already augmented with the expert-action dims
        when ``ExpertObsAugmentation`` is active), ``edge_critic_params``
        (the frozen expert critic if MC pretraining ran, else the live
        critic params), the running ``post_warmup_action``, and the
        gate's mutable rng (``gate_rng``) — updated in-place so the
        pre-refactor rng tree is preserved byte-for-byte (the
        framework's per-extension ``rng`` split would otherwise diverge).

        Side effects: writes ``_edge_use_expert`` back into ``obs`` for
        the pipeline's ``is_expert_flag`` bookkeeping, and updates
        ``obs["gate_rng"]`` for any downstream gate.
        """
        del ext_state, rng  # see docstring on rng plumbing.
        policy_action = obs["policy_action"]
        expert_action = obs["expert_action"]
        obs_for_edge = obs["obs_for_edge"]
        edge_critic_params = obs["edge_critic_params"]
        critic_state = obs["critic_state"]
        post_warmup_action = obs["post_warmup_action"]
        gate_rng = obs["gate_rng"]

        decay = edge_compute_decay(
            agent_state.collector_state.timestep,
            ctx.total_steps,
            self.decay_frac,
        )

        gate = self.gate

        if gate == "thompson":
            mu_e, sigma_e, mu_p, sigma_p, _ = edge_compute_thompson_stats(
                obs_for_edge,
                policy_action,
                expert_action,
                critic_state,
                edge_critic_params,
            )
            use_expert_edge, gate_rng = edge_thompson_gate(
                mu_e,
                sigma_e,
                mu_p,
                sigma_p,
                gate_rng,
                self.lcb_temperature,
                epsilon_floor=self.epsilon_floor,
            )
        elif gate in ("lcb", "argmax_lcb"):
            # Quality-aware: LCB (or asymmetric LCB/UCB) scores + gate.
            total_timesteps = max(int(ctx.total_steps), 1)
            progress = jnp.clip(
                agent_state.collector_state.timestep / jnp.maximum(total_timesteps, 1),
                0.0,
                1.0,
            )
            beta_eff = self.lcb_beta_init * jnp.power(
                1.0 - progress, self.lcb_beta_decay_k
            )
            _scores_fn = (
                edge_compute_asym_scores
                if self.lcb_asymmetric
                else edge_compute_lcb_scores
            )
            (
                score_e,
                score_p,
                _q_policy,
                _mu_p,
                _mu_e,
                _sigma_p,
                _sigma_e,
            ) = _scores_fn(
                obs_for_edge,
                policy_action,
                expert_action,
                critic_state,
                edge_critic_params,
                beta_eff,
            )
            if gate == "argmax_lcb":
                use_expert_edge, gate_rng = edge_lcb_argmax_gate(
                    score_e, score_p, gate_rng
                )
            else:
                use_expert_edge, gate_rng = edge_lcb_gate(
                    score_e, score_p, gate_rng, self.lcb_temperature
                )
        else:
            gap, q_policy = edge_compute_value_gap(
                obs_for_edge,
                policy_action,
                expert_action,
                critic_state,
                edge_critic_params,
            )
            if gate == "argmax":
                use_expert_edge, gate_rng = edge_argmax_gate(gap, decay, gate_rng)
            elif gate == "boltzmann":
                use_expert_edge, gate_rng = edge_boltzmann_gate(
                    gap, decay, gate_rng, q_policy, self.tau
                )
            else:  # "fixed"
                use_expert_edge, gate_rng = edge_fixed_gate(
                    gap, decay, gate_rng, self.fixed_prob
                )

        # Thread the gate's updated rng back to the pipeline + record
        # the substitution mask so the pipeline can fold it into
        # ``is_expert_flag``. Dict-write side effects are explicit at
        # the call site (the pipeline reads these keys after every
        # ``stack.action`` call).
        obs["gate_rng"] = gate_rng
        obs["_edge_use_expert"] = use_expert_edge.astype(jnp.bool_)
        return jnp.where(use_expert_edge, expert_action, post_warmup_action)


__all__ = ["EDGEExploration"]
