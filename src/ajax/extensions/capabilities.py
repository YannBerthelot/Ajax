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
    "replay": "its updates sample stored past transitions",
    "off_policy": "its update stays correct on actions its policy did not choose",
    "critic_ensemble": "two critics or more (num_critics, where the agent has it)",
    "target_critic": "a slowly updated copy of its critic, which its bootstrap reads",
    "policy_density": "a stochastic policy, with a density over its actions",
    "obs_networks": (
        "its policy and value networks read the observation, not a world"
        " model's latent state"
    ),
}

#: The capabilities a run gives, with what gives them.
RUN_CAPABILITIES: Mapping[str, str] = {
    "critic": "a q_critic or a v_critic",
    "continuous_actions": "an environment with continuous actions",
    "discrete_actions": "an environment with discrete actions",
    "agent_expert": "give the agent an expert_policy",
    "phi_star": "add MCPretrain(expert_policy=...), which fits phi*, to the extensions",
}

#: The capabilities each phase needs; the others need none.
PHASE_NEEDS: Mapping[str, frozenset[str]] = {
    "on_target": frozenset({"critic"}),
    "critic_loss": frozenset({"critic"}),
    "actor_loss": frozenset({"actor"}),
    # The executed action becomes one the policy did not choose.
    "action": frozenset({"off_policy"}),
    "transform_action": frozenset({"continuous_actions"}),
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
) -> frozenset[str]:
    """A run's capabilities: its ``algorithm``'s, without ``critic_ensemble``
    when ``num_critics`` (the agent's option, where it has one) is below 2,
    and those its environment's actions and its expert give."""
    caps = frozenset(algorithm)
    if num_critics is not None and num_critics < 2:
        caps -= {"critic_ensemble"}
    derived = {
        "critic": bool(caps & {"q_critic", "v_critic"}),
        "continuous_actions": continuous,
        "discrete_actions": not continuous,
        "agent_expert": has_expert,
    }
    return caps | {name for name, given in derived.items() if given}


def batch_keys(agent_name: str, phase: str, table: Mapping) -> frozenset[str]:
    """The keys ``agent_name``'s ``phase`` batch holds (the rows of ``table``,
    ``PHASE_BATCH_KEYS``, for every agent and for this one)."""
    rows = table.get(phase, {})
    return frozenset(
        key
        for agents, keys in rows.items()
        if agents == "*" or agent_name in agents.split()
        for key in keys.split()
    )


def check_capabilities(
    agent_name: str,
    stack: Sequence[Any],
    has: Iterable[str],
    *,
    phases: Iterable[str],
    context: Iterable[str],
    pretrain_context: Iterable[str],
    buffer_fields: Iterable[str],
    batch_table: Mapping,
    takes_expert_policy: bool,
) -> None:
    """Raise when a module of ``stack`` cannot run on ``agent_name``, whose
    run has the capabilities ``has`` and whose code folds ``phases``, binds
    the ``context`` keys, fills the ``pretrain_context`` fields of its
    pretraining's ExtensionContext, stores the ``buffer_fields``, passes the batch
    keys ``batch_table`` lists, and which takes an ``expert_policy`` when
    ``takes_expert_policy``.

    :class:`MissingCapability` names each module and the capabilities it
    lacks; otherwise :class:`NotWired` names what the agent does not pass on.
    """
    has = frozenset(has).union(*(ext.provides for ext in stack))
    phases, context = frozenset(phases), frozenset(context)
    missing: dict[str, frozenset[str]] = {}
    unwired: dict[str, str] = {}
    for ext in stack:
        lacks = ext.needs() - has
        gaps = []
        if "agent_expert" in lacks and not takes_expert_policy:
            gaps.append("takes no expert_policy")
            lacks -= {"agent_expert"}
        if lacks:
            missing[type(ext).__name__] = lacks
            continue
        unfolded = ext.implemented_phases() - phases
        unbound = ext.context - context
        unfilled = ext.pretrain_context - frozenset(pretrain_context)
        unstored = ext.buffer_fields - frozenset(buffer_fields)
        gaps += [
            *([f"folds no {sorted(unfolded)} phase"] if unfolded else []),
            *([f"binds no {sorted(unbound)}"] if unbound else []),
            *([f"gives its pretraining no {sorted(unfilled)}"] if unfilled else []),
            *([f"stores no {sorted(unstored)}"] if unstored else []),
            *(
                f"puts no {sorted(keys - batch_keys(agent_name, p, batch_table))}"
                f" in its {p} batch"
                for p, keys in ext.reads().items()
                if keys - batch_keys(agent_name, p, batch_table)
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
