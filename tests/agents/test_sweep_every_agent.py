"""Every agent's sweep (ActorCritic.train's sweep, ajax.agents.sweep): each
field an agent declares sweepable, on each agent of the contract table.

Contract (train's docstring; docs/guide/training.md, "Hyperparameter
sweeps"; ActorCritic.sweepable):

- an agent's sweepable fields are those its classes declare, a mixin's
  included, less those it refuses, whose reason the refusal gives;
- a member changing one sweepable field trains otherwise than the member at
  the agent's own value: the field reaches the training;
- that member trains as a separate run built with the value does, up to
  float32 rounding;
- every library agent has a row here or a stated reason for none.

The values are half the agent's own: a learning rate, a discount, a
coefficient and a noise scale all stay valid halved.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import CrossQ
from tests.agents.test_agent_contract import CASES, Case
from tests.test_exports import library_agents

#: A separate run folds its Python numbers in double precision before XLA
#: sees them; a member's are float32 throughout (tests/agents/test_sweep.py,
#: which measured 6e-8 after about 100 updates of networks this size), and a
#: sweep's wider batch reorders float32 sums. Leaves of order 1 stay within
#: 1e-6. Larger ones (an optimiser's moments, ASAC's termination penalty near
#: 40) differ by one float32 step, which is 1.2e-7 of the value: hence the
#: relative part, eight such steps. A field the member did not receive
#: leaves 1e-3 or more.
ROUNDING = 1e-6
RELATIVE_ROUNDING = 1e-6

#: The values that replace halving where half changes nothing in so short a
#: run. PPO's clip_range: small enough to clip in the first updates
#: (tests/agents/test_sweep.py uses the same).
VALUES = {("PPO", "clip_range"): 0.001}

#: The library agents without a row, and why.
NO_ROW = {
    "APG": "differentiates through the simulator: no row in the contract table",
    "DreamerV3": "its own replay loop refuses a sweep (no sweepable field)",
    "TDMPC2": "its own replay loop refuses a sweep (no sweepable field)",
    "TDMPC2MultiTask": "trains from a dataset, with its own train",
}

SWEEPING = sorted(name for name, case in CASES.items() if case.cls.sweepable)


def build(case: Case, **fields: Any) -> Any:
    kwargs = dict(case.kwargs)
    kwargs.update(fields)
    return case.cls(case.envs[-1], **kwargs)


def floating_leaves(state: Any) -> dict[str, np.ndarray]:
    """Everything training writes, by its path in the state."""
    return {
        jax.tree_util.keystr(path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(state)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    }


def one_field_changes(agent: Any) -> tuple[list[str], list[dict[str, float]]]:
    """The agent's own values, then one configuration per sweepable field
    with that field halved (or at its VALUES entry)."""
    names = sorted(agent.sweepable_fields())
    base = {name: float(getattr(agent, name)) for name in names}
    configurations = [base]
    for name in names:
        changed = dict(base)
        changed[name] = VALUES.get((type(agent).__name__, name), base[name] / 2)
        configurations.append(changed)
    return names, configurations


def test_every_library_agent_has_a_row_or_a_reason() -> None:
    known = set(CASES).union(NO_ROW)
    assert {agent.__name__ for agent in library_agents()} <= known


def test_a_mixins_fields_join_the_agents_own_less_those_it_refuses() -> None:
    """SAC takes the temperature's three from EntropyTemperature beside its
    own; CrossQ, with the same mixin, refuses them and says why."""
    temperature = {"alpha_learning_rate", "alpha_init", "target_entropy_per_dim"}
    sac = CASES["SAC"].cls
    assert temperature < sac.sweepable and "tau" in sac.sweepable
    assert not temperature.intersection(CrossQ.sweepable)
    crossq = build(CASES["CrossQ"])
    with pytest.raises(ValueError, match="not explained yet"):
        crossq.train(seed=0, n_timesteps=16, sweep={"alpha_init": [0.5, 1.0]})


@pytest.mark.parametrize("name", SWEEPING)
def test_every_sweepable_field_reaches_the_agents_training(name: str) -> None:
    """Catches a field captured when the agent is built, which every member
    would train with at the agent's own value."""
    case = CASES[name]
    agent = build(case)
    names, configurations = one_field_changes(agent)
    state, _ = agent.train(
        seed=0, n_timesteps=case.n_timesteps, sweep=configurations, logging_config=None
    )
    leaves = list(floating_leaves(state).values())
    for row, field in enumerate(names, start=1):
        # equal_nan: NaN fills a metrics slot no update wrote yet, in every row.
        moved = [
            not np.allclose(leaf[row], leaf[0], atol=ROUNDING, equal_nan=True)
            for leaf in leaves
        ]
        assert any(moved), f"{name}'s {field} changed nothing"


@pytest.mark.slow
@pytest.mark.parametrize("name", SWEEPING)
def test_each_member_trains_as_its_separate_run(name: str) -> None:
    case = CASES[name]
    agent = build(case)
    names, configurations = one_field_changes(agent)
    state, _ = agent.train(
        seed=0, n_timesteps=case.n_timesteps, sweep=configurations, logging_config=None
    )
    swept = floating_leaves(state)
    for row, field in enumerate(names, start=1):
        separate, _ = build(case, **{field: configurations[row][field]}).train(
            seed=0, n_timesteps=case.n_timesteps, logging_config=None
        )
        for path, leaf in floating_leaves(separate).items():
            np.testing.assert_allclose(
                swept[path][row],
                leaf[0],
                atol=ROUNDING,
                rtol=RELATIVE_ROUNDING,
                err_msg=f"{name} {field} {path}",
            )
