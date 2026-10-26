"""TD-target modifier research features as composable Extensions.

These reshape the Bellman target before it enters the critic loss (the
on_target phase):

* IBRL: add gamma (1 - d) (Q_expert - Q_policy) where the expert's next
  action is selected, so the value function matches an argmax
  action-selection policy.
* LCBGatedBootstrap: soft-blend the policy and expert TD targets by an
  LCB-scored sigmoid gate.
* CriticBlend: a warmup-decaying blend of the Bellman target with the
  frozen expert critic's.
* MCVarianceCorrection: replace high-ensemble-variance Bellman targets
  with the MC-pretrained oracle.
* ValueBox: a value-threshold expert action override during collection
  (the action phase).

The batch on_target reads is the dict the shared critic step builds
(update_steps.apply_on_target): observations, actions, next_observations,
rewards (unscaled), reward_scale, dones, gamma, raw_next_observations (the
environment's own next observations, which an extension's expert reads:
an agent keeps them when an extension carries an expert), next_a_expert,
next_actions (the next actions the target scored) and critic_subset (the
target critics it took its minimum over, where a random subset: REDQ's).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional, Tuple

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_pi
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.expert import extension_expert_actions, reads_stored_expert_actions
from ajax.extensions.exploration import lcb_beta, lcb_score, score_action_pair
from ajax.networks.networks import phi_star_critic_state, predict_value
from ajax.schedule import progress


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


def discounted(name: str, gamma: Any) -> Any:
    """The target's discount ``gamma``, which a bootstrap bonus scales;
    refused on an average-reward agent (ASAC, APO: ``gamma`` None), whose
    target has no discount and for which the bonus has no form."""
    if gamma is None:
        raise ValueError(
            f"{name} adds a discounted bonus, gamma (1 - d) x [...], to a"
            " discounted Bellman target; this agent's target is"
            " average-reward (no gamma), for which the bonus has no form."
        )
    return gamma


def score_next_actions(
    ext: Any, agent_state: Any, batch: dict, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The target critics' ensemble at s' for the expert's action and for
    the policy's, (Q_target(s', a_E), Q_target(s', a')): a' is the next
    action the agent's target scored (TD3's smoothed target action, the
    sample SAC's, REDQ's and TQC's targets valued), as the authors' IBRL
    compares the expert with the action its target values
    (hengyuan-hu/ibrl rl/q_agent.py:_act_ibrl). A target without next
    actions (IQL's V(s')) has no original form: a sample of the online
    actor on key stands in."""
    next_observations = batch["next_observations"]
    critic_state = agent_state.critic_state

    def q(actions: jax.Array) -> jax.Array:
        x = jnp.concatenate((next_observations, actions), axis=-1)
        return predict_value(critic_state, critic_state.target_params, x)

    next_expert_actions = extension_expert_actions(
        ext, batch["raw_next_observations"], batch["next_a_expert"]
    )
    next_actions = batch["next_actions"]
    if next_actions is None:
        next_policy, _ = get_pi(
            actor_state=agent_state.actor_state,
            actor_params=agent_state.actor_state.params,
            obs=next_observations,
            done=batch["dones"],
            # A run with memory refuses every extension
            # (check_recurrent_options): the actor here is feedforward.
            recurrent=False,
        )
        next_actions, _ = next_policy.sample_and_log_prob(seed=key)
    return q(next_expert_actions), q(next_actions)


def take_minimum(values: jax.Array, critics: Optional[jax.Array]) -> jax.Array:
    """The minimum of values over its ensemble axis, over the critics indexed
    by critics (None: all of them)."""
    if critics is not None:
        values = values[critics]
    return jnp.min(values, axis=0)


def draw_selection_critics(
    size: Optional[int], num_critics: int, key: jax.Array
) -> Optional[jax.Array]:
    """The critics IBRL selects the expert's or the policy's action on: a
    fresh random subset of size critics, the size of the subset the base
    target takes its minimum over, drawn on key; None (all critics) where
    the target takes all of them (size None). The authors' state critic
    draws a random pair on each call, one to select and one to value
    (hengyuan-hu/ibrl rl/critic.py forward_k, rl/q_agent.py:288, :442), a
    quirk Ajax copies (UPSTREAM_BUGS 1); with no subset both are every
    critic, as in their two-critic pixel runs."""
    if size is None:
        return None
    return jax.random.choice(key, num_critics, shape=(size,), replace=False)


@dataclass(frozen=True)
class IBRL(Extension):
    """IBRL bootstrap (Hu, Mirchandani and Sadigh, Imitation Bootstrapped
    Reinforcement Learning, 2023), as its authors' code computes it
    (hengyuan-hu/ibrl 3c457da, rl/q_agent.py update_critic and _act_ibrl).

    The target bootstraps on the better of the expert's next action a_E and
    the policy's a', the next action the agent's target scored
    (score_next_actions): it gains gamma (1 - d) [a_E selected] (min
    Q_target(s', a_E) - min Q_target(s', a')), the minima over the critics
    the base target took its minimum over. With all of them, the bonus is
    max(min Q(s', a_E) - min Q(s', a'), 0). With a random subset of them
    (REDQ's), a_E is selected on a fresh subset of the same size
    (draw_selection_critics), as the authors select on one random pair and
    value on another, so the bonus can be negative. Selection is hard, the
    authors' "ibrl", not their "ibrl_soft".

    With hybrid_acting it also acts as their act_method "ibrl" (every IBRL
    configuration of theirs): it executes the better of the agent's action
    and the expert's (choose_action), in collection (the action stage, past
    the warm-up) and in evaluation (eval_action). The agent's is the one it
    would execute (TD3's noisy action, a sample; in evaluation its mean);
    each is scored by the target critics' minimum, over a fresh random
    subset where the target takes one (REDQ's), as the authors' forward_k
    draws a random pair per call. The replay flags the steps the expert's
    action took. The actor's loss ignores the choice, as theirs; their
    epsilon-greedy choice (ibrl_eps_greedy, 1 in every configuration) is
    left out.

    No original form exists for the rest, kept as Ajax's: SAC's, REDQ's and
    TQC's entropy term stays at a' when the expert is selected; TQC's bonus
    reads its quantiles' means, while its target pools truncated quantiles;
    a target without next actions (IQL's V(s')) scores a sample of the
    online actor instead.
    """

    expert_policy: Callable
    hybrid_acting: bool = False
    reads_stored_expert: bool = False  # bound: the agent's expert
    critic_subset_size: Optional[int] = None  # bound: REDQ's subset_size
    name: str = "ibrl"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"actor", "discounted", "q_critic", "target_critic"}
    )
    action_slot: ClassVar[str] = "pre_warmup"
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset(
            {
                "critic_subset",
                "dones",
                "next_a_expert",
                "next_actions",
                "raw_next_observations",
            }
        )
    }
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

    def implemented_phases(self) -> frozenset[str]:
        """Its acting phases only where it acts hybrid."""
        phases = super().implemented_phases()
        if self.hybrid_acting:
            return phases
        return phases.difference({"action", "eval_action"})

    def bind_to_agent(self, **agent_context: Any) -> "IBRL":
        """Read the expert's stored actions when it is the agent's, and act
        on the target's subset size (REDQ's; None: every critic)."""
        stored = reads_stored_expert_actions(self, agent_context)
        size = agent_context.get("subset_size")
        return dataclasses.replace(
            self, reads_stored_expert=stored, critic_subset_size=size
        )

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array:
        """Collection's choice, from the action the step would execute."""
        del ext_state, rng, context
        return self.choose_action(agent_state, obs, obs["post_warmup_action"])

    def eval_action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array:
        """Evaluation's choice, from the action the agent evaluates with."""
        del ext_state, rng, context
        return self.choose_action(agent_state, obs, obs["executed_action"])

    def choose_action(
        self, agent_state: Any, step: dict, action: jax.Array
    ) -> jax.Array:
        """The expert's action where its minimum over the target critics
        tops action's, else action (hengyuan-hu/ibrl rl/q_agent.py
        _act_ibrl); a tie keeps action, as their argmax does. Reads step
        (the stage's dict, or evaluation's) and writes into it where the
        expert's won (expert_mask) and the key a subset drew on (gate_rng)."""
        expert_action = extension_expert_actions(
            self, step["raw_obs"], step["expert_action"]
        )
        critic_state = agent_state.critic_state
        q_expert, q_policy = score_action_pair(
            step["observations"],
            action,
            expert_action,
            critic_state,
            critic_state.target_params,
        )
        critics = None
        if self.critic_subset_size is not None:
            gate_rng, key = jax.random.split(step["gate_rng"])
            step["gate_rng"] = gate_rng
            num_critics = q_expert.shape[0]
            critics = draw_selection_critics(self.critic_subset_size, num_critics, key)
        expert_won = take_minimum(q_expert, critics) > take_minimum(q_policy, critics)
        step["expert_mask"] = expert_won
        return jnp.where(expert_won, expert_action, action)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        """Where it acts hybrid, the share of envs whose last collection
        step executed an expert's action (ibrl/expert_frac)."""
        del ext_state, rng, context
        if not self.hybrid_acting:
            return {}
        return {"ibrl/expert_frac": agent_state.collector_state.last_expert_frac}

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        context: ExtensionContext,
    ) -> Any:
        del ext_state
        dones, gamma = batch["dones"], discounted("IBRL", batch["gamma"])
        action_key, selection_key = jax.random.split(context.rng)
        q_expert, q_policy = score_next_actions(self, agent_state, batch, action_key)
        value_critics = batch["critic_subset"]
        size = None if value_critics is None else value_critics.shape[0]
        selection_critics = draw_selection_critics(
            size, q_expert.shape[0], selection_key
        )
        expert_on_selection = take_minimum(q_expert, selection_critics)
        policy_on_selection = take_minimum(q_policy, selection_critics)
        expert_selected = expert_on_selection > policy_on_selection
        expert_value = take_minimum(q_expert, value_critics)
        advantage = expert_value - take_minimum(q_policy, value_critics)
        gap = jax.lax.stop_gradient(
            gamma * (1.0 - dones) * jnp.where(expert_selected, advantage, 0.0)
        )
        metrics = {
            "ibrl/expert_wins": jnp.mean(expert_selected),
            "ibrl/bonus": jnp.mean(gap),
        }
        return target + gap, metrics


@dataclass(frozen=True)
class LCBGatedBootstrap(Extension):
    """LCB-gated bootstrap.

    Scores the policy and expert next-actions by an LCB rule and
    soft-blends the two TD targets with ``p_expert =
    σ((score_e - score_p)/lcb_temperature)``. ``β`` anneals over training;
    evaluations log it (``lcb_bootstrap/beta``). The policy's next action is
    the one the agent's target scored, as IBRL's, and the two values blended
    are minima over the critics the base target took its minimum over (all
    of them, or REDQ's subset), so the blend moves the target from what it
    valued; the LCB scores read the whole ensemble.
    """

    expert_policy: Callable
    lcb_beta_init: float = 1.0
    lcb_beta_decay_k: float = 2.0
    lcb_temperature: float = 1.0
    reads_stored_expert: bool = False  # bound: the agent's expert
    name: str = "lcb_gated_bootstrap"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"actor", "critic_ensemble", "discounted", "q_critic", "target_critic"}
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset(
            {
                "critic_subset",
                "dones",
                "next_a_expert",
                "next_actions",
                "raw_next_observations",
            }
        )
    }
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def uses_expert_actions_as_policy_actions(self) -> bool:
        return True

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
        context: ExtensionContext,
    ) -> Any:
        del ext_state
        # beta anneals as the EDGE gate's, on the collector's timestep.
        dones, gamma = batch["dones"], discounted("LCBGatedBootstrap", batch["gamma"])
        action_key, _ = jax.random.split(context.rng)
        q_expert, q_policy = score_next_actions(self, agent_state, batch, action_key)
        beta = self.beta(agent_state.collector_state.timestep, context.total_steps)
        score_e, score_p = lcb_score(q_expert, beta), lcb_score(q_policy, beta)
        q_min_e = take_minimum(q_expert, batch["critic_subset"])
        q_min_p = take_minimum(q_policy, batch["critic_subset"])
        p_expert = jax.nn.sigmoid(
            (score_e - score_p) / jnp.maximum(self.lcb_temperature, 1e-6)
        )
        # Bellman target with the LCB-gated next action. We blend the
        # min-Q part only (entropy term stays on the policy branch since
        # the expert is deterministic — log_prob is undefined).
        min_q_lcb = (1.0 - p_expert) * q_min_p + p_expert * q_min_e
        gap_lcb = jax.lax.stop_gradient(gamma * (1.0 - dones) * (min_q_lcb - q_min_p))
        return target + gap_lcb, {"lcb_bootstrap/p_expert": jnp.mean(p_expert)}

    def beta(self, timestep: Any, total_steps: int) -> jax.Array:
        """The LCB pessimism ``β`` at ``timestep`` (:func:`lcb_beta`)."""
        return lcb_beta(
            timestep, total_steps, self.lcb_beta_init, self.lcb_beta_decay_k
        )

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del ext_state, rng
        timestep = agent_state.collector_state.timestep
        return {"lcb_bootstrap/beta": self.beta(timestep, context.total_steps)}


@dataclass(frozen=True)
class CriticBlend(Extension):
    """Warmup-decaying blend of the Bellman target with the expert's.

    ``(1-α)·y_bellman + α·y_expert`` with ``y_expert = r + γ(1-d)·V_E(s')``,
    the one-step target on the frozen expert critic ``V_E(s') = φ*(s',
    a_E(s'))``, so the blend stays a Bellman target (``V_E(s')`` alone has
    no reward, discount or terminal mask). ``α`` decays linearly to 0 over
    ``critic_warmup_frac`` of training; evaluations log it
    (``critic_blend/expert_weight``). Requires the frozen expert critic
    :class:`MCPretrain` fits (refused without it).
    """

    expert_policy: Callable
    critic_warmup_frac: float = 0.15
    reads_stored_expert: bool = False  # bound: the agent's expert
    name: str = "critic_blend"
    # The blend is of one target value: atoms blended toward it would shrink
    # a distribution's spread.
    requires: ClassVar[frozenset[str]] = frozenset(
        {"discounted", "phi_star", "q_critic", "scalar_target"}
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset(
            {"dones", "next_a_expert", "raw_next_observations", "reward_scale"}
        )
    }
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})
    reads_agent_expert_actions: ClassVar[bool] = True

    def experts(self) -> tuple:
        return (self.expert_policy,)

    def uses_expert_actions_as_policy_actions(self) -> bool:
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
        context: ExtensionContext,
    ) -> Any:
        del ext_state
        next_observations = batch["next_observations"]
        a_expert_next = extension_expert_actions(
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
        alpha_blend_val = self.expert_weight(
            agent_state.collector_state.timestep, context.total_steps
        )
        target_new = blend_modify_target(target, expert_target, alpha_blend_val)
        gap = jax.lax.stop_gradient(jnp.mean(expert_target - target))
        return jax.lax.stop_gradient(target_new), {"critic_blend/target_gap": gap}

    def expert_weight(self, timestep: Any, total_steps: int) -> jax.Array:
        """``α``: 1 at the start, falling linearly to 0 at
        ``critic_warmup_frac`` of the run."""
        train_frac = progress(timestep, total_steps)
        return jnp.maximum(1.0 - train_frac / self.critic_warmup_frac, 0.0)

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del ext_state, rng
        timestep = agent_state.collector_state.timestep
        return {
            "critic_blend/expert_weight": self.expert_weight(
                timestep, context.total_steps
            )
        }


@dataclass(frozen=True)
class MCVarianceCorrection(Extension):
    """Replace high-variance Bellman targets with the MC oracle.

    Where the inter-critic variance exceeds ``threshold`` the Bellman
    target is swapped for the MC-pretrained expert critic's estimate.
    Requires :class:`MCPretrain` (refused without it).
    """

    threshold: float
    name: str = "mc_variance_correction"
    # It swaps one target value for the oracle's: a distribution's would
    # collapse to that point.
    requires: ClassVar[frozenset[str]] = frozenset(
        {"critic_ensemble", "phi_star", "q_critic", "scalar_target"}
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset({"actions", "observations"})
    }

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        target: jax.Array,
        context: ExtensionContext,
    ) -> Any:
        del ext_state, context
        observations = batch["observations"]
        actions = batch["actions"]
        # The critics' predictions at the rows, from a pass of its own, as
        # KernelRepulsion computes its own: an agent hands on_target none.
        critic_state = agent_state.critic_state
        x = jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1)
        q_preds = predict_value(
            critic_state=critic_state, critic_params=critic_state.params, x=x
        )
        q_var = q_preds.var(axis=0)[..., 0]
        corrected = mc_correction_modify_target(
            target,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            observations,
            actions,
            q_var,
            self.threshold,
        )
        replaced = jnp.mean(q_var > self.threshold)
        return corrected, {"mc_variance/replaced_frac": replaced}


@dataclass(frozen=True)
class ValueBox(Extension):
    """Value-threshold expert-action override.

    During collection, override the policy action with the expert's
    whenever the frozen-expert value V_expert(s) exceeds a curriculum
    threshold that ramps from v_min to v_max over training. Requires
    MCPretrain (which supplies phi*, v_min and v_max; refused without it).

    The action phase, in the action stage's post-warm-up slot
    (ajax.environments.action_stage), reads the stage's dict (env_action;
    expert_action, the action of the agent's expert_policy, at which it
    also measures the expert's value; observations, the critics' input,
    augmented when it is) and the box bounds off the agent state (so a
    resumed run keeps them), and writes in_value_box, entry_bonus and
    box_entry back into the dict for the collector to record on the
    transition, and expert_mask, where the expert drove. An entry ends the
    agent's episode: the expert's value, paid as the bonus, replaces the
    bootstrap, whatever its sign. The override (box_action_override) comes
    after the warm-up's choice.

    Evaluations log the threshold (value_box/threshold), the share of envs
    inside the box and the share the expert drove on the last collection
    step (value_box/in_box_frac, value_box/expert_frac); the bounds are
    MCPretrain's (mc_pretrain/v_min, mc_pretrain/v_max).
    """

    name: str = "value_box"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"agent_expert", "phi_star", "q_critic"}
    )
    action_slot: ClassVar[str] = "post_warmup"

    def uses_expert_actions_as_policy_actions(self) -> bool:
        # It executes the expert's action where the policy's goes.
        return True

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array | None:
        """Override with a_expert inside the value box, its bounds
        MCPretrain's v_min and v_max on the agent's state; writes
        in_value_box, entry_bonus, box_entry and expert_mask into obs for
        the collector's bookkeeping (reward shaping, the entry's
        termination, is_expert_flag).
        """
        del ext_state, rng
        env_action = obs["env_action"]
        expert_action = obs["expert_action"]

        threshold = self.threshold(agent_state, context.step, context.total_steps)
        in_value_box, entry_bonus, _ = box_compute_state(
            obs["observations"],
            expert_action,
            agent_state.critic_state,
            agent_state.expert_critic_params,
            threshold,
            agent_state.collector_state.last_in_box,
        )
        obs["in_value_box"] = in_value_box
        obs["expert_mask"] = in_value_box
        obs["entry_bonus"] = entry_bonus
        obs["box_entry"] = box_entered(
            in_value_box, agent_state.collector_state.last_in_box
        ).astype(jnp.float32)
        return box_action_override(env_action, expert_action, in_value_box)

    def threshold(self, agent_state: Any, step: Any, total_steps: int) -> jax.Array:
        """The box's threshold at ``step``: MCPretrain's ``v_min`` rising
        linearly to its ``v_max`` over the run."""
        return box_compute_threshold(
            agent_state.expert_v_min,
            agent_state.expert_v_max,
            progress(step, total_steps),
        )

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> dict:
        del ext_state, rng
        collector = agent_state.collector_state
        return {
            "value_box/threshold": self.threshold(
                agent_state, context.step, context.total_steps
            ),
            "value_box/in_box_frac": jnp.mean(collector.last_in_box),
            "value_box/expert_frac": collector.last_expert_frac,
        }


__all__ = [
    "IBRL",
    "CriticBlend",
    "LCBGatedBootstrap",
    "MCVarianceCorrection",
    "ValueBox",
]
