"""What an agent has and what a module needs.

An agent declares the capabilities of its algorithm
(``ActorCritic.capabilities``); its run adds those its configuration gives
(``ActorCritic.run_capabilities``) and the modules of its stack those they
provide (MCPretrain's phi*). A module needs the capabilities its phases imply
(:data:`PHASE_NEEDS`) and those it declares (``Extension.needs``). It runs on
an agent whose run has them all, unless the agent's code does not pass them
on yet: it does not fold the module's phase, bind its context, hold its
batch keys or store its buffer fields.

A refusal names what is missing: :class:`MissingCapability` when the
algorithm or the run lacks a capability, :class:`NotWired` when the agent has
them all and its code does not pass them on yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: The capabilities an agent declares: what its algorithm has, whatever the
#: state of the code.
ALGORITHM_CAPABILITIES: Mapping[str, str] = {
    "actor": "a policy network trained by gradient",
    "q_critic": "a learned action value Q(s, a)",
    "v_critic": "a learned state value V(s)",
    "discounted": "it maximises a discounted return, not the average reward",
    "next_action_target": (
        "its target scores the critic at a next action a' (the policy's at s',"
        " or the greedy one), not a state value V(s')"
    ),
    "replay": "its updates sample stored past transitions",
    "one_step_target": (
        "its target bootstraps one step ahead on each transition, r + gamma"
        " V(s'), not on a multi-step or lambda-return"
    ),
    "off_policy": "its update stays correct on actions its policy did not choose",
    "critic_ensemble": "two critics or more (num_critics, where the agent has it)",
    "target_critic": "a slowly updated copy of its critic, which its bootstrap reads",
    "distributional_critic": (
        "its critic predicts quantiles of the return, and regresses on many"
        " target values per row"
    ),
    "policy_density": "a stochastic policy, with a density over its actions",
    "obs_networks": (
        "its policy and value networks read the observation, not a world"
        " model's latent state"
    ),
    "reward_learning": "it learns from the reward, not from an expert's labels alone",
    "warmup": (
        "it acts uniformly at random for its first steps (learning_starts;"
        " TD-MPC2's seed phase), and later learns from their stored transitions"
    ),
}

#: The capabilities a run gives, with what gives them.
RUN_CAPABILITIES: Mapping[str, str] = {
    "critic": "a q_critic or a v_critic",
    "scalar_target": "a critic that regresses on one target per row, not a distributional_critic",
    "continuous_actions": "an environment with continuous actions",
    "discrete_actions": "an environment with discrete actions",
    "agent_expert": "give the agent an expert_policy",
    "phi_star": "add MCPretrain(expert_policy=...), which fits phi*, to the extensions",
    "seeded_replay": "add ExpertPrefill or PriorData, which fill the replay before training",
    "warmup_steps": "set learning_starts above 0",
    "stored_steps": "a replay, or the last rollout an on-policy or incremental agent keeps",
}

#: The capabilities each phase needs; the others need none.
PHASE_NEEDS: Mapping[str, frozenset[str]] = {
    "on_target": frozenset({"critic"}),
    "critic_loss": frozenset({"critic"}),
    "actor_loss": frozenset({"actor"}),
    # The executed action becomes one the policy did not choose.
    "action": frozenset({"off_policy"}),
    "transform_action": frozenset({"continuous_actions"}),
    "actuate": frozenset({"continuous_actions"}),
    # A warm-up exists only on an agent that has one, run with steps in it.
    "choose_warmup_action": frozenset({"warmup", "warmup_steps"}),
    # Only a replay stores actions to learn from.
    "choose_replay_action": frozenset({"replay"}),
}


class MissingCapability(ValueError):
    """The algorithm or the run lacks capabilities a module needs: per module
    class name, those it lacks (``missing``)."""

    def __init__(self, message: str, missing: Mapping[str, frozenset[str]]):
        super().__init__(message)
        self.missing = dict(missing)


class NotWired(ValueError):
    """The agent has what a module needs, and its code does not pass it on
    yet: per module class name, what it does not (``unwired``)."""

    def __init__(self, message: str, unwired: Mapping[str, str]):
        super().__init__(message)
        self.unwired = dict(unwired)


def run_capabilities(
    algorithm: Iterable[str],
    *,
    continuous: bool,
    has_expert: bool,
    num_critics: int | None = None,
    learning_starts: int | None = None,
    keeps_rollout: bool = False,
) -> frozenset[str]:
    """A run's capabilities: its ``algorithm``'s, without ``critic_ensemble``
    when ``num_critics`` (the agent's option, where it has one) is below 2,
    and those its environment's actions, its expert, its warm-up steps
    (learning_starts, where the agent has one) and its replay or kept
    rollout (``keeps_rollout``) give."""
    caps = frozenset(algorithm)
    if num_critics is not None and num_critics < 2:
        caps = caps.difference({"critic_ensemble"})
    derived = {
        "critic": bool(caps.intersection({"q_critic", "v_critic"})),
        "scalar_target": bool(caps.intersection({"q_critic", "v_critic"}))
        and "distributional_critic" not in caps,
        "continuous_actions": continuous,
        "discrete_actions": not continuous,
        "agent_expert": has_expert,
        "warmup_steps": "warmup" in caps and bool(learning_starts),
        "stored_steps": "replay" in caps or keeps_rollout,
    }
    return caps.union({name for name, given in derived.items() if given})


def check_capabilities(
    agent_name: str,
    stack: Sequence[Any],
    has: Iterable[str],
    *,
    phases: Iterable[str],
    context: Iterable[str],
    pretrain_context: Iterable[str],
    buffer_fields: Iterable[str],
    batch_keys: Mapping[str, frozenset[str]],
    takes_expert_policy: bool,
) -> None:
    """Raise when a module of ``stack`` cannot run on ``agent_name``, whose
    run has the capabilities ``has`` and whose code folds ``phases``, binds
    the ``context`` keys, fills the ``pretrain_context`` fields of its
    pretraining's ExtensionContext, stores the ``buffer_fields``, passes the
    ``batch_keys`` (per phase), and which takes an ``expert_policy`` when
    ``takes_expert_policy``.

    :class:`MissingCapability` names each module and the capabilities it
    lacks; otherwise :class:`NotWired` names what the agent does not pass on.
    """
    has = frozenset(has).union(*(ext.provides for ext in stack))
    phases, context = frozenset(phases), frozenset(context)
    missing: dict[str, frozenset[str]] = {}
    unwired: dict[str, str] = {}
    for ext in stack:
        lacks = ext.needs().difference(has)
        gaps = []
        if "agent_expert" in lacks and not takes_expert_policy:
            gaps.append("takes no expert_policy")
            lacks = lacks.difference({"agent_expert"})
        if lacks:
            missing[type(ext).__name__] = lacks
            continue
        unfolded = ext.implemented_phases().difference(phases)
        unbound = ext.context.difference(context)
        unfilled = ext.pretrain_context.difference(frozenset(pretrain_context))
        unstored = ext.buffer_fields.difference(frozenset(buffer_fields))
        gaps += [
            *([f"folds no {sorted(unfolded)} phase"] if unfolded else []),
            *([f"binds no {sorted(unbound)}"] if unbound else []),
            *([f"gives its pretraining no {sorted(unfilled)}"] if unfilled else []),
            *([f"stores no {sorted(unstored)}"] if unstored else []),
            *(
                f"puts no {sorted(keys.difference(batch_keys.get(p, frozenset())))}"
                f" in its {p} batch"
                for p, keys in ext.reads().items()
                if keys.difference(batch_keys.get(p, frozenset()))
            ),
        ]
        if gaps:
            unwired[type(ext).__name__] = "; ".join(gaps)
    labels = {
        type(ext).__name__: f"{type(ext).__name__} ({ext.name!r})" for ext in stack
    }
    if missing:
        details = "; ".join(
            f"{labels[name]}, which needs "
            + ", ".join(f"{c} ({_meaning(c)})" for c in sorted(lacks))
            for name, lacks in missing.items()
        )
        raise MissingCapability(f"{agent_name} cannot run {details}.", missing)
    if unwired:
        details = "; ".join(
            f"{labels[name]}: {agent_name} {gaps} yet" for name, gaps in unwired.items()
        )
        raise NotWired(
            f"{agent_name} has what its extensions need, but does not pass it"
            f" on yet: {details}.",
            unwired,
        )


def _meaning(capability: str) -> str:
    return {**ALGORITHM_CAPABILITIES, **RUN_CAPABILITIES}[capability]
