"""Curriculum training for APG: successive stages resumed from one another.

Busetto et al. 2024 (Algorithm 2) train the contextual controller in
stages of increasing diversity: a single system with a fixed reference,
then the system class, then systems and references, then everything
including initial conditions. Each stage is an :class:`APG` agent built
on its own environment / system class; :func:`train_curriculum` chains
them through Ajax's resume path, carrying the learned parameters over
and (by default) restarting the optimizer and its schedule per stage.

The paper trains each stage to a loss tolerance; Ajax's training loop is
a fixed-length scan, so each stage gets an explicit ``n_timesteps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp

from ajax.agents.APG.APG import APG


@dataclass
class CurriculumStage:
    agent: APG
    n_timesteps: int
    name: str = ""


def _check_compatible(first: APG, other: APG, name: str) -> None:
    if other.env_args.n_envs != first.env_args.n_envs:
        raise ValueError(
            f"stage {name!r}: n_envs={other.env_args.n_envs} differs from the"
            f" first stage's {first.env_args.n_envs}; the resumed state is"
            " batched per env"
        )
    if other.network_args != first.network_args:
        raise ValueError(
            f"stage {name!r}: network configuration differs from the first"
            " stage's; parameters cannot be carried over"
        )
    if (other.pid, other.squash) != (first.pid, first.squash):
        raise ValueError(
            f"stage {name!r}: controller head (pid / squash) differs from the"
            " first stage's"
        )


def train_curriculum(
    stages: Sequence[CurriculumStage],
    seed: int | Sequence[int] = 42,
    num_episode_test: int = 10,
    logging_config: Optional[Any] = None,
    reset_optimizer: bool = True,
) -> list:
    """Train ``stages`` in order, each resumed from the previous one.

    Returns one ``(agent_state, aux)`` pair per stage (what ``train``
    returns). ``reset_optimizer`` re-initialises the optimizer and its
    learning-rate schedule at every stage boundary.
    """
    if not stages:
        raise ValueError("train_curriculum needs at least one stage")
    first = stages[0].agent
    for stage in stages[1:]:
        _check_compatible(first, stage.agent, stage.name)
    results: list = []
    state: Any = None
    for stage in stages:
        out: Any = stage.agent.train(
            seed=seed,
            n_timesteps=stage.n_timesteps,
            num_episode_test=num_episode_test,
            logging_config=logging_config,
            initial_state=state,
            reset_optimizer_on_resume=reset_optimizer,
        )
        # The resume path donates the incoming buffers; hand the next stage
        # a copy so every returned stage state stays readable.
        state = jax.tree.map(jnp.copy, out[0])
        results.append(out)
    return results


__all__ = ["CurriculumStage", "train_curriculum"]
