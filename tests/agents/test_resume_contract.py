"""The resume contract: a run split in two, its second leg on a new agent
instance from the first leg's state, ends in the state one call reaches,
every leaf alike (floats to rtol 1e-5: programs compiled apart may round
apart). One row per agent of the contract table with the normalisation it
takes, one more with memory where it takes one, and APG with memory (its
plain resume: tests/probing/test_p4_split_run.py). The first leg's
checkpoint restores every leaf into the new instance's skeleton. The world
models' own tests hold theirs (test_p4_split_run.ELSEWHERE)."""

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ajax.agents.APG.APG import APG
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.networks.memory import MemoryConfig
from tests.agents.test_agent_contract import CASES
from tests.environments import paid_action
from tests.probing import readouts, runs

SEEDS = (0, 1)
#: Every normalisation an agent takes, on (AVG normalises observations always).
NORMALISATION = {
    "normalize_observations": True,
    "normalize_rewards": True,
    "reward_normalization_gamma": 0.9,
    "normalize_obs_running": True,
}
# A call's schedules run to the horizon it knows, the first leg's its own:
# exploration is held constant.
CONSTANT = {"epsilon_start": 0.3, "epsilon_end": 0.3}
MEMORY: dict[str, Any] = {"memory": MemoryConfig(kind="gru", hidden_size=8)}
# Sequence replay: windows of 2 + 4 + 1 steps, one per env before the first update.
MEMORY |= {"burn_in": 2, "sequence_length": 4, "learning_starts": 16}


def _row(name: str, memory: bool) -> tuple[Any, int, dict]:
    """The agent's builder, the uninterrupted run's budget, train's options."""
    if name == "APG":  # it restarts its optimizer on resume unless told not to
        env, params = paid_action.make()
        kw: dict[str, Any] = {"n_envs": 4, "horizon": 1, "memory": MEMORY["memory"]}
        keep = {"reset_optimizer_on_resume": False}
        return lambda: APG(env, env_params=params, pid=None, **kw), 64, keep
    case, taken = CASES[name], inspect.signature(CASES[name].cls).parameters
    options = CONSTANT | NORMALISATION | (MEMORY if memory else {})
    kw = case.kwargs | {k: v for k, v in options.items() if k in taken}
    return lambda: case.cls(env_id=case.envs[-1], n_envs=2, **kw), case.n_timesteps, {}


# Day one: a new instance's gymnax normaliser defines its own state class,
# which the resumed scan's carry refuses.
NEW_CLASS = pytest.mark.xfail(
    strict=True, raises=TypeError, reason="fix/resume-normalised-new-instance"
)
ROWS = [pytest.param(n, False, id=n, marks=NEW_CLASS) for n in sorted(CASES)]
ROWS += [
    pytest.param(n, True, id=f"{n}-memory", marks=NEW_CLASS)
    for n in sorted(CASES)
    if CASES[n].cls.supports_memory
]
ROWS += [pytest.param("APG", True, id="APG-memory")]


@pytest.mark.slow
@pytest.mark.parametrize("name, memory", ROWS)
def test_a_resumed_run_equals_the_uninterrupted_run(
    name: str, memory: bool, tmp_path: Path
) -> None:
    build, budget, kw = _row(name, memory)
    full = runs.train(build(), SEEDS, budget, **kw).state
    first = runs.train(build(), SEEDS, budget // 2, **kw).state
    # An on-policy call runs one rollout past its budget (TrainLoop.n_rollouts):
    # the second leg's budget leaves room for its own.
    ended, end = (int(np.asarray(s.collector_state.timestep)[0]) for s in (first, full))
    second = end - ended - (ended - budget // 2)
    agent, path = build(), str(tmp_path / "checkpoint.pkl")
    save_checkpoint(first, path)
    restored = restore_into(runs.train(agent, SEEDS, 0, **kw).state, path)
    assert not readouts.differing_leaves(first, restored), "the checkpoint"
    split = runs.train(agent, SEEDS, second, initial_state=first, **kw).state
    differing = readouts.differing_leaves(full, split)
    assert not differing, (f"first leg ended at {ended}", differing[:20])
