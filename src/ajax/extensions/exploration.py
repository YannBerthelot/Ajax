"""Exploration research features as composable Extensions.

EDGE (Expert Decayed Guided Exploration) substitutes the expert action
for the policy action during collection, gated by a value comparison
that decays over training. EDGEExploration selects one of six gates with
gate: "fixed", "argmax" and "boltzmann" (value-gap gates) or "lcb",
"argmax_lcb" and "thompson" (quality-aware gates); the gate maths are the
functions below.

The action stage (ajax.environments.action_stage) hands the gate the
step's quantities (policy_action, expert_action, observations, ...) as a
dict through the obs argument, as the batch dict of the on_target and
actor_loss phases. The extension picks an action (or returns None to
defer); the stage owns the bookkeeping (is_expert_flag, the stored
action). Gate randomness is threaded through obs["gate_rng"] (read and
overwritten in place), the collector's own stream, not the framework's
per-extension key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Tuple

import jax
import jax.numpy as jnp

from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import phi_star_critic_state, predict_value
from ajax.schedule import progress


def score_action_pair(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
) -> Tuple[jax.Array, jax.Array]:
    """The critic ensemble's Q(s, a_expert) and Q(s, a_policy) at
    critic_params (the live critic's, phi*, or the target's), obs the
    critic's input."""

    def q(action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        return predict_value(
            critic_state=critic_state, critic_params=critic_params, x=x
        )

    return q(expert_action), q(policy_action)


def lcb_beta(
    step: jax.Array, total_steps: int, init: float, decay_k: float
) -> jax.Array:
    """The LCB pessimism ``init (1 - t / T)^decay_k``, annealed over the
    run (``t / T`` clipped to [0, 1], ``T`` at least 1)."""
    done = jnp.clip(progress(step, total_steps), 0.0, 1.0)
    return init * jnp.power(1.0 - done, decay_k)


def lcb_score(q: jax.Array, beta: jax.Array) -> jax.Array:
    """``min Q - beta (max Q - min Q)`` over the ensemble axis: the critics'
    disagreement as epistemic uncertainty, penalised."""
    q_min, q_max = jnp.min(q, axis=0), jnp.max(q, axis=0)
    return q_min - beta * (q_max - q_min)


def edge_compute_value_gap(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Q(s,pi*) - Q(s,pi).  Positive means expert is still better.

    Returns (gap, q_policy) — q_policy is reused by the Boltzmann gate.
    """
    q_e, q_p = score_action_pair(
        obs, policy_action, expert_action, critic_state, critic_params
    )
    q_policy = jnp.min(q_p, axis=0)
    return jnp.min(q_e, axis=0) - q_policy, q_policy


def edge_compute_decay(
    timestep: jax.Array,
    total_timesteps: int,
    decay_frac: float,
) -> jax.Array:
    """Linear decay: 1 -> 0 over decay_frac of training.

    decay_frac=0.0 means never decay (gate stays active for full training).
    """
    if decay_frac == 0.0:
        return jnp.ones(())
    train_frac = progress(timestep, total_timesteps)
    return jnp.maximum(1.0 - train_frac / decay_frac, 0.0)


def edge_argmax_gate(
    gap: jax.Array,
    decay: jax.Array,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Deterministic: use expert whenever gap > 0 and decay > 0.

    Returns (use_expert_mask, unchanged_rng).
    """
    return (gap > 0.0) & (decay > 0.0), rng


def edge_boltzmann_gate(
    gap: jax.Array,
    decay: jax.Array,
    rng: jax.Array,
    q_policy: jax.Array,
    tau: float,
) -> Tuple[jax.Array, jax.Array]:
    """Adaptive: p = decay * sigmoid(gap / (tau * |Q|)).

    Returns (use_expert_mask, updated_rng).
    """
    q_scale = jax.lax.stop_gradient(jnp.abs(q_policy).mean() + 1e-6)
    p_expert = decay * jax.nn.sigmoid(gap / (tau * q_scale))
    rng, key = jax.random.split(rng)
    return jax.random.uniform(key, shape=p_expert.shape) < p_expert, rng


def edge_fixed_gate(
    gap: jax.Array,  # noqa: ARG001 -- the gates' shared (gap, decay, rng) signature
    decay: jax.Array,
    rng: jax.Array,
    fixed_prob: float,
) -> Tuple[jax.Array, jax.Array]:
    """Fixed probability: p = decay * fixed_prob.

    Returns (use_expert_mask, updated_rng).
    """
    p = decay * fixed_prob
    rng, key = jax.random.split(rng)
    return jax.random.uniform(key, shape=p.shape) < p, rng


def edge_lcb_scores(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
    beta: jax.Array,
    asymmetric: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Each candidate's score ``(score_expert, score_policy)``: its
    :func:`lcb_score`, the critics' disagreement penalising uncertain
    (OOD) candidates; the gate then compares the two.

    ``asymmetric`` scores the policy arm optimistically instead (UCB,
    ``max Q + beta (max Q - min Q)``), so the gate hands control to the
    policy where its critic is uncertain, which is where fresh on-policy
    Bellman anchors help most; the expert arm keeps its LCB, staying
    conservative about following an unreliable expert. Symmetric LCB
    over-penalises the policy exactly where it explores beyond the
    expert's tube.
    """
    q_e, q_p = score_action_pair(
        obs, policy_action, expert_action, critic_state, critic_params
    )
    if asymmetric:
        q_min_p, q_max_p = jnp.min(q_p, axis=0), jnp.max(q_p, axis=0)
        return lcb_score(q_e, beta), q_max_p + beta * (q_max_p - q_min_p)
    return lcb_score(q_e, beta), lcb_score(q_p, beta)


def edge_lcb_argmax_gate(
    score_expert: jax.Array,
    score_policy: jax.Array,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Deterministic argmax gate over LCB scores.

    use_expert = score_e > score_p.

    Compared to ``edge_lcb_gate`` (softmax), this collapses the gate to
    a hard threshold while preserving the LCB scoring (mu - beta*sigma)
    so the (gating form, scoring rule) ablation can vary one axis at
    a time. Used by the ``argmax_lcb`` corner of the 2x2.
    """
    return score_expert > score_policy, rng


def edge_lcb_gate(
    score_expert: jax.Array,
    score_policy: jax.Array,
    rng: jax.Array,
    temperature: float,
) -> Tuple[jax.Array, jax.Array]:
    """Stochastic Boltzmann gate over LCB scores.

    p(use_expert) = sigmoid((score_e - score_p) / temperature).

    Hard threshold (argmax) creates a self-fulfilling expert preference:
    expert always wins → buffer all expert → critic never learns Q for
    policy actions → expert keeps winning. The stochastic version ensures
    policy occasionally gets sampled, breaking the inertia.

    Temperature → 0 recovers argmax; → ∞ recovers uniform mixing.
    """
    delta = (score_expert - score_policy) / jnp.maximum(temperature, 1e-6)
    p_expert = jax.nn.sigmoid(delta)
    rng, key = jax.random.split(rng)
    use_expert = jax.random.uniform(key, shape=p_expert.shape) < p_expert
    return use_expert, rng


def edge_compute_thompson_stats(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-action Q mean / std across critic ensemble: (mu_e, sigma_e,
    mu_p, sigma_p)."""
    q_e, q_p = score_action_pair(
        obs, policy_action, expert_action, critic_state, critic_params
    )
    mu_e = jnp.mean(q_e, axis=0)
    sigma_e = jnp.std(q_e, axis=0)
    mu_p = jnp.mean(q_p, axis=0)
    sigma_p = jnp.std(q_p, axis=0)
    return mu_e, sigma_e, mu_p, sigma_p


def edge_thompson_gate(
    mu_e: jax.Array,
    sigma_e: jax.Array,
    mu_p: jax.Array,
    sigma_p: jax.Array,
    rng: jax.Array,
    temperature: float = 1.0,
    epsilon_floor: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    """Thompson-sampling gate, with optional epsilon floor on policy picks.

    Treat each candidate's Q as Gaussian(mu, (temperature * sigma)^2),
    draw one sample per side, pick whichever is larger. Equivalent to
    p(use_expert) = Phi((mu_e - mu_p) / (T * sqrt(sigma_e^2 + sigma_p^2)))
    in expectation, but stochastic per-step (which is what we want for
    exploration: the gate itself injects noise rather than deferring to a
    fixed rule).

    Symmetric: high uncertainty on either side widens the gate, never
    biasing it. Confident estimates dominate. With both sigmas → 0 we
    recover deterministic argmax.

    epsilon_floor: with this probability, force the policy regardless of
    the Thompson outcome. Required when the expert dominates by many
    sigmas (Phi -> 1 -> Thompson never picks the policy -> the critic
    never sees policy actions in its updates -> Q for policy actions
    stays stale -> the gate can never flip even if the policy improves).
    Set carefully on brittle envs: too high crashes the agent, too low
    starves the policy of evaluation data. ~0.001-0.05 is a sane range.
    """
    rng, key_e, key_p, key_floor = jax.random.split(rng, 4)
    scale = jnp.maximum(temperature, 1e-6)
    q_tilde_e = mu_e + scale * sigma_e * jax.random.normal(key_e, mu_e.shape)
    q_tilde_p = mu_p + scale * sigma_p * jax.random.normal(key_p, mu_p.shape)
    use_expert_thompson = q_tilde_e > q_tilde_p
    if epsilon_floor > 0.0:
        force_policy = jax.random.uniform(key_floor, mu_e.shape) < epsilon_floor
        use_expert = jnp.where(
            force_policy, jnp.zeros_like(use_expert_thompson), use_expert_thompson
        )
    else:
        use_expert = use_expert_thompson
    return use_expert, rng


_VALID_GATES = frozenset(
    {"fixed", "argmax", "boltzmann", "lcb", "argmax_lcb", "thompson"}
)


@dataclass(frozen=True)
class EDGEExploration(Extension):
    """Expert Decayed Guided Exploration during collection.

    gate selects the substitution rule (see the module docstring). The
    other fields:

    * decay_frac: the value-gap gates decay to 0 over this fraction.
    * tau: the Boltzmann temperature on the value gap.
    * fixed_prob: the substitution probability of the fixed gate.
    * lcb_beta_init, lcb_beta_decay_k: the LCB pessimism schedule.
    * lcb_temperature: the softmax temperature of the LCB and Thompson gates.
    * lcb_asymmetric: LCB on the expert arm, UCB on the policy arm.
    * epsilon_floor: the minimum policy-pick probability (Thompson).

    The expert is the agent's expert_policy: the gate compares the action
    collection called it for with the policy's. Evaluations log the share
    of envs the expert drove on the last collection step
    (edge/live_expert_frac) and the gate's schedule: edge/beta for the LCB
    gates, edge/decay for the fixed, argmax and Boltzmann gates (the
    Thompson gate has none).
    """

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
    # Its gates score an action at the critic's input beside the observation,
    # and it has no published form for discrete actions (owner, 10-25).
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "continuous_actions"}
    )
    action_slot: ClassVar[str] = "pre_warmup"

    def needs(self) -> frozenset[str]:
        """The value gates compare the critic's values of the two actions;
        the LCB and Thompson gates read its ensemble."""
        needs = set(super().needs())
        if self.gate != "fixed":
            needs.add("q_critic")
        if self.gate in ("lcb", "argmax_lcb", "thompson"):
            needs.add("critic_ensemble")
        return frozenset(needs)

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
        context: ExtensionContext,
    ) -> jax.Array | None:
        """Apply the configured EDGE gate to substitute the policy action.

        obs is the action stage's dict (the module docstring):
        policy_action, expert_action, observations (the critics' input, its
        features included), post_warmup_action and gate_rng. The gate
        scores with the frozen expert critic phi* where MCPretrain fitted
        one, else with the agent's live critic.

        Side effects: writes expert_mask back into obs, which flags the
        steps the expert took (is_expert_flag), and updates obs["gate_rng"]
        for any downstream gate.
        """
        del ext_state, rng  # the gates draw from obs["gate_rng"]
        policy_action = obs["policy_action"]
        expert_action = obs["expert_action"]
        observations = obs["observations"]
        post_warmup_action = obs["post_warmup_action"]
        gate_rng = obs["gate_rng"]
        # phi* lives on SACState alone, where MCPretrain binds: getattr is a
        # deliberate workaround until it reaches every agent's state. phi*
        # is applied without the agent's statistics.
        phi_star = getattr(agent_state, "expert_critic_params", None)
        critic_state = agent_state.critic_state
        critic_params = critic_state.params
        if phi_star is not None:
            critic_state, critic_params = phi_star_critic_state(critic_state), phi_star

        gate = self.gate

        if gate == "thompson":
            mu_e, sigma_e, mu_p, sigma_p = edge_compute_thompson_stats(
                observations,
                policy_action,
                expert_action,
                critic_state,
                critic_params,
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
            beta = lcb_beta(
                agent_state.collector_state.timestep,
                context.total_steps,
                self.lcb_beta_init,
                self.lcb_beta_decay_k,
            )
            score_e, score_p = edge_lcb_scores(
                observations,
                policy_action,
                expert_action,
                critic_state,
                critic_params,
                beta,
                asymmetric=self.lcb_asymmetric,
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
            decay = edge_compute_decay(
                agent_state.collector_state.timestep,
                context.total_steps,
                self.decay_frac,
            )
            gap, q_policy = edge_compute_value_gap(
                observations,
                policy_action,
                expert_action,
                critic_state,
                critic_params,
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

        # The stage reads both keys after every action call.
        obs["gate_rng"] = gate_rng
        obs["expert_mask"] = use_expert_edge.astype(jnp.bool_)
        return jnp.where(use_expert_edge, expert_action, post_warmup_action)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        """The fraction of envs an expert drove on the last collection step,
        and the gate's schedule at the collector's timestep."""
        del ext_state, rng
        collector = agent_state.collector_state
        metrics = {"edge/live_expert_frac": collector.last_expert_frac}
        if self.gate in ("lcb", "argmax_lcb"):
            metrics["edge/beta"] = lcb_beta(
                collector.timestep,
                context.total_steps,
                self.lcb_beta_init,
                self.lcb_beta_decay_k,
            )
        elif self.gate != "thompson":
            metrics["edge/decay"] = edge_compute_decay(
                collector.timestep, context.total_steps, self.decay_frac
            )
        return metrics


__all__ = ["EDGEExploration"]
