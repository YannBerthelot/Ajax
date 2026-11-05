"""One budget of environment steps for a run, whoever acted.

Arms given unequal experience compare unequally: an expert's prefill, a
cloning rollout and a recorded dataset all cost the plant steps, but sit
outside n_timesteps, which counts only the steps of collection. The ledger
charges each once, against one budget, with the expert's queries beside them.

Evaluation steps are not charged: no arm learns from them. An expert query is
a call on a state the plant visited; a stored action read again on a replayed
row is none (an extension's own expert, re-called on replayed rows, is not
counted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ajax.extensions.pretrain import (
    BellmanPretrain,
    CloningPretrain,
    ExpertPrefill,
    MCPretrain,
    PriorData,
)
from ajax.extensions.retained import RetainedDemonstrations

#: The steps every arm of the study shares.
BUDGET = 1_000_000


@dataclass(frozen=True)
class Charge:
    source: str
    env_steps: int
    expert_queries: int = 0


@dataclass(frozen=True)
class StepLedger:
    """A run's charges (run_ledger), each source once."""

    charges: tuple[Charge, ...] = ()

    def charge(
        self, source: str, env_steps: int, expert_queries: int = 0
    ) -> StepLedger:
        """This ledger and one charge more: data a harness brings itself,
        as a dataset train_offline trains on."""
        new = Charge(source, env_steps, expert_queries)
        return StepLedger((*self.charges, new))

    @property
    def env_steps(self) -> int:
        return sum(charge.env_steps for charge in self.charges)

    @property
    def expert_queries(self) -> int:
        return sum(charge.expert_queries for charge in self.charges)

    def assert_budget(self, budget: int = BUDGET) -> None:
        """Raise unless the run spent exactly budget: an arm under it had
        less experience, as one over it had more."""
        if self.env_steps != budget:
            raise ValueError(
                f"The run spent {self.env_steps} environment steps, not the"
                f" shared budget of {budget}: {self.charges}."
            )


def _pretraining_charge(extension: Any, n_envs: int) -> Charge:
    """The steps an extension's pretrain phase costs: every one of them the
    expert's, but a dataset's, whose is_expert rows it says. A pretraining
    the ledger does not know is refused, not left uncounted."""
    logged = isinstance(extension, (PriorData, RetainedDemonstrations))
    if logged and extension.dataset is not None:
        dataset = extension.dataset
        rows = len(dataset.reward)
        return Charge("recorded dataset", rows, int(np.sum(dataset.is_expert)))
    if isinstance(extension, ExpertPrefill):
        steps = extension.replay_rows(n_envs)
    elif isinstance(extension, (CloningPretrain, RetainedDemonstrations)):
        steps = extension.n_steps * n_envs
    elif isinstance(extension, MCPretrain):
        total = extension.n_mc_steps * extension.n_mc_episodes
        steps = max(1, total // n_envs) * n_envs  # as pretrain_critic_mc rolls out
    elif isinstance(extension, BellmanPretrain):
        steps = 0  # it updates on the replay alone
    else:
        raise ValueError(
            f"{type(extension).__name__} pretrains, but the ledger does not"
            " know the environment steps it costs."
        )
    return Charge(extension.name, steps, steps)


def pretraining_charges(agent: Any) -> tuple[Charge, ...]:
    """The steps the agent's pretraining costs, outside n_timesteps."""
    extensions = agent.extension_stack.extensions
    pretraining = [e for e in extensions if "pretrain" in e.implemented_phases()]
    return tuple(_pretraining_charge(e, agent.env_args.n_envs) for e in pretraining)


def online_steps(agent: Any, budget: int = BUDGET) -> int:
    """The n_timesteps that spends what the agent's pretraining leaves of
    budget, for a run to spend exactly budget."""
    left = budget - StepLedger(pretraining_charges(agent)).env_steps
    if left < 0:
        raise ValueError(f"The agent's pretraining alone spends over {budget} steps.")
    return left


def run_ledger(agent: Any, n_timesteps: int) -> StepLedger:
    """The ledger of agent's run of n_timesteps collected steps (a state's
    collector_state.timestep): its pretraining's charges, then the warm-up
    before the first update and the steps after it, on all of which an
    expert in the loop is called, once per step, whoever then acts."""
    stack = agent.extension_stack.extensions
    acting = [e for e in stack if "action" in e.implemented_phases() and e.experts()]
    calls = 1 if agent.expert is not None or acting else 0
    uniform = agent.uniform_steps()
    warmup = getattr(agent, "learning_starts", 0) if uniform is None else uniform
    warmup = min(warmup, n_timesteps)
    return (
        StepLedger(pretraining_charges(agent))
        .charge("warm-up", warmup, warmup * calls)
        .charge("online", n_timesteps - warmup, (n_timesteps - warmup) * calls)
    )
