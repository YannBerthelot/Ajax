"""Rows a replay keeps for the whole run, in a ring of their own beside the
one collection writes (ajax.buffers.utils.TwoBufferState), so no later step
evicts them.

* :class:`RetainedDemonstrations` -- DDPGfD's demonstrations, kept forever.
* :class:`RLPD`                   -- RLPD's offline buffer, half of each batch.

The update ratio, the critics and their architecture stay the agent's own
(``num_critic_updates``, ``num_critics``, ``critic_architecture``), so these
modules run on any backbone. RLPD's original configuration, ten LayerNorm
critics, a subset of two and 20 critic steps per env step on fresh batches,
is REDQ's defaults with ``critic_architecture=("256", "layernorm", "relu",
"256", "layernorm", "relu")`` and ``extensions=(RLPD(...),)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp

from ajax.agents.loop import seed_replay
from ajax.buffers.dataset import TransitionDataset
from ajax.buffers.utils import empty_ring_like, with_retained_rows
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.pretrain import ExpertPrefill
from ajax.networks.networks import action_value_input, predict_value


@dataclass(frozen=True)
class RetainedDemonstrations(Extension):
    """Demonstrations in a ring of their own, sampled for the whole run.

    DDPGfD (Vecerik et al. 2017, sec. 3) "loads the demonstration
    transitions into the replay buffer before the training begins and keeps
    all transitions forever". The rows are an expert's, ``n_steps`` per env
    (ExpertPrefill's: with the run's observation features), the agent's or,
    so that no online step calls one, ``expert``'s; or those of ``dataset``
    (``n_steps`` 0), whose observations hold no features. They count once in
    the normalisers, in no timestep, and the replay's capacity leaves them out.

    ``share`` is the part of each batch they fill. None draws each row over
    all the rows held, so they fade as the agent's own rows arrive, as in a
    replay that evicts nothing of theirs. 0.5 is Rainbow-DemoRL's prefill
    (Bhatt et al. 2026), which with CloningPretrain before it in the stack is
    their BC start plus prefill.

    Departures from DDPGfD: the sampling is uniform in each ring, without its
    prioritised replay (its priorities for demonstrations set the share
    itself); no n-step loss and no L2 penalty, which stay the backbone's.
    """

    n_steps: int
    dataset: Optional[TransitionDataset] = None
    share: Optional[float] = None
    # In agent units: the agent's is bound by bind_to_agent, one given here
    # converted by ExtensionStack.with_experts.
    expert: Optional[Callable] = None
    name: str = "retained_demonstrations"
    requires: ClassVar[frozenset[str]] = frozenset({"off_policy", "replay"})
    provides: ClassVar[frozenset[str]] = frozenset({"seeded_replay"})
    context: ClassVar[frozenset[str]] = frozenset({"expert_policy"})
    pretrain_context: ClassVar[frozenset[str]] = frozenset({"buffer", "env_args"})
    expert_fields: ClassVar[frozenset[str]] = frozenset({"expert"})

    def __post_init__(self) -> None:
        if (self.dataset is None) != (self.n_steps > 0):
            raise ValueError(
                f"{type(self).__name__} keeps the rows of a dataset or of n_steps"
                " steps of the agent's expert: give exactly one."
            )
        if self.share is not None and not 0.0 <= self.share <= 1.0:
            raise ValueError(f"share is a part of a batch, got {self.share}.")

    def needs(self) -> frozenset[str]:
        """The agent's expert, unless a dataset or its own gives the rows."""
        if self.dataset is not None or self.expert is not None:
            return super().needs()
        return super().needs().union({"agent_expert"})

    def bind_to_agent(self, **agent_context: Any) -> Any:
        if self.expert is not None:
            return self
        return dataclasses.replace(self, expert=agent_context.get("expert_policy"))

    def pretrain(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, Any]:
        """The kept rows written to a ring of their own, as a replay's seeders
        write theirs, which then joins the replay."""
        collector = agent_state.collector_state
        online, env_args = collector.buffer_state, context.env_args
        rows = self.n_steps
        if self.dataset is not None:
            rows = len(self.dataset.reward) // env_args.n_envs
        # One row more: a ring filled exactly would report full and wrapped.
        ring = empty_ring_like(online, rows + 1)
        seeding = agent_state.replace(
            collector_state=collector.replace(buffer_state=ring)
        )
        if self.dataset is None:
            prefill = ExpertPrefill(self.n_steps, self.expert)
            seeded, _ = prefill.pretrain(seeding, (), context)
        else:
            seeded = seed_replay(seeding, context.buffer, self.dataset, env_args)
        collector = seeded.collector_state
        replay = with_retained_rows(online, collector.buffer_state, self.share)
        collector = collector.replace(buffer_state=replay)
        return seeded.replace(collector_state=collector), ext_state


def bootstrap_on_subset(
    target: jax.Array,
    next_values: jax.Array,
    subset: jax.Array,
    gamma: float,
    dones: jax.Array,
) -> jax.Array:
    """target, whose bootstrap took the minimum of the target critics'
    next_values (their axis first), as if it took the minimum over subset."""
    gap = jnp.min(next_values[subset], axis=0) - jnp.min(next_values, axis=0)
    return target + gamma * (1.0 - dones) * gap


@dataclass(frozen=True)
class RLPD(RetainedDemonstrations):
    """Online RL with offline data (Ball, Smith, Kostrikov and Levine, ICML
    2023; ikostrikov/rlpd@c90fd4b): an offline ring of logged rows, half of
    every batch from it, on critics with LayerNorm.

    The logged rows keep their rewards, and the agent's reward normaliser
    divides them as it divides its own. ``critic_subset`` is their random
    subset Z: each target's bootstrap takes the minimum of that many target
    critics, drawn once per target (Algorithm 1, line 15; None: all of them).
    ``critic_layer_norm`` refuses critics without LayerNorm; set it to False
    to run the ablation (their figure 7, value divergence on narrow data).

    Departures, all kept for the shared backbone, and all gone on REDQ (the
    module docstring): SAC's actor loss reads the critics' minimum, RLPD's
    their mean; SAC's critic steps of one update share one batch, RLPD's each
    draw theirs (rlpd/agents/sac/sac_learner.py update); and its target
    entropy is -dim(A), RLPD's -dim(A)/2. Ajax's warm-up acts uniformly up to
    learning_starts, as their code's start_training does and their Algorithm
    1 does not. The offline rows are not interleaved with the online ones:
    the batch's first half is theirs, which a mean over rows does not see.
    """

    share: Optional[float] = 0.5
    critic_subset: Optional[int] = None
    critic_layer_norm: bool = True
    name: str = "rlpd"
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset({"dones", "next_actions"})
    }

    def implemented_phases(self) -> frozenset[str]:
        """on_target only where it subsamples: otherwise the target is the
        agent's own, on any critic."""
        phases = super().implemented_phases()
        if self.critic_subset is None:
            return phases.difference({"on_target"})
        return phases

    def reads(self) -> Mapping[str, frozenset[str]]:
        """Only a subsampling target reads its batch."""
        return self.batch_keys if self.critic_subset is not None else {}

    def needs(self) -> frozenset[str]:
        """Subsampling needs a next action's values, one per target critic."""
        if self.critic_subset is None:
            return super().needs()
        wanted = {"critic_ensemble", "next_action_target", "q_critic", "target_critic"}
        return super().needs().union(wanted).union({"scalar_target"})

    def bind_to_agent(self, **agent_context: Any) -> "RLPD":
        """Check the critics the agent builds, where it hands them over."""
        network = agent_context.get("network_args")
        if self.critic_layer_norm and network is not None:
            if "layernorm" not in network.critic_architecture:
                raise ValueError(
                    "RLPD's critics have LayerNorm: pass critic_architecture="
                    '("256", "layernorm", "relu", "256", "layernorm", "relu"), or'
                    " critic_layer_norm=False to run without it."
                )
        critics = agent_context.get("num_critics")
        if self.critic_subset is not None:
            if agent_context.get("subset_size") is not None:
                raise ValueError(
                    "This agent's target already takes a random subset of its"
                    " critics (subset_size): leave critic_subset None."
                )
            if critics is not None and not 1 <= self.critic_subset <= critics:
                raise ValueError(
                    f"critic_subset {self.critic_subset} of {critics} critics."
                )
        return super().bind_to_agent(**agent_context)

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        target: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array:
        """The target, bootstrapping on a random subset of the target critics."""
        del ext_state
        if self.critic_subset is None:
            return target
        x = action_value_input(batch["next_observations"], batch["next_actions"])
        critic = agent_state.critic_state
        next_values = predict_value(critic, critic.target_params, x)
        critics = next_values.shape[0]
        subset = jax.random.choice(
            context.rng, critics, (self.critic_subset,), replace=False
        )
        return bootstrap_on_subset(
            target, next_values, subset, batch["gamma"], batch["dones"]
        )


__all__ = ["RLPD", "RetainedDemonstrations", "bootstrap_on_subset"]
