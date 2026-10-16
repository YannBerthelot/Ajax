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
from unittest import mock

import jax
import numpy as np
import pytest

import ajax
from ajax.agents.APG.APG import APG
from ajax.agents.DQN import train_DQN
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


ROWS = [pytest.param(n, False, id=n) for n in sorted(CASES)]
ROWS += [
    pytest.param(n, True, id=f"{n}-memory")
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


def test_every_agent_taking_normalisation_or_memory_has_its_row() -> None:
    """A new agent taking either joins test_agent_contract.CASES: its rows follow."""
    ids = {p.id for p in ROWS}
    for name in ajax.__all__:
        cls = getattr(ajax, name)
        normalises = NORMALISATION.keys() & inspect.signature(cls).parameters.keys()
        assert not normalises or name in ids, name
        assert not getattr(cls, "supports_memory", None) or f"{name}-memory" in ids


def _dqn() -> Any:
    case = CASES["DQN"]
    return case.cls(env_id=case.envs[0], n_envs=2, **case.kwargs)


def _epsilon_trace(legs: tuple[int, ...], horizon: int) -> list[tuple[int, float]]:
    """(timestep, epsilon) at every collected step of a DQN run trained in
    ``legs`` calls, each told the run's horizon."""
    trace: list[tuple[int, float]] = []
    epsilon_at = train_DQN.linear_epsilon

    def recorded(timestep: Any, *args: Any) -> Any:
        epsilon = epsilon_at(timestep, *args)
        jax.debug.callback(
            lambda t, e: trace.append((int(t), float(e))), timestep, epsilon
        )
        return epsilon

    agent, state = _dqn(), None
    with mock.patch.object(train_DQN, "linear_epsilon", recorded):
        for steps in legs:
            state, _ = agent.train(
                seed=0, n_timesteps=steps, initial_state=state, total_timesteps=horizon
            )
            jax.effects_barrier()
    return sorted(trace)


def test_a_run_split_in_two_follows_the_one_call_epsilon_schedule() -> None:
    """The two legs of a run told its horizon see, step for step, the epsilon
    one call sees: from 1.0 down to its floor 0.05 at half the run."""
    budget = CASES["DQN"].n_timesteps
    one = _epsilon_trace((budget,), budget)
    assert [t for t, _ in one] == list(range(0, budget, 2))
    assert one[0][1] == 1.0 and one[-1][1] == pytest.approx(0.05)
    assert _epsilon_trace((budget // 2, budget // 2), budget) == one


#: What a call records on the agent: the world models' replay, which they
#: size for the call.
RECORDED = ("replay_rows_per_env", "replay_capacity", "replay_bytes_per_seed")
REFUSING = {
    "DQN": _dqn,
    "DreamerV3": lambda: ajax.DreamerV3("CartPole-v1", n_envs=2),
    "TDMPC2": lambda: ajax.TDMPC2("Pendulum-v1", n_envs=2),
}


@pytest.mark.parametrize("name", sorted(REFUSING))
def test_a_horizon_before_the_calls_end_is_refused(name: str) -> None:
    """Before the call records anything on the agent."""
    agent = REFUSING[name]()

    def records() -> list:
        return [(getattr(agent, k, None), agent.config.get(k)) for k in RECORDED]

    before = records()
    with pytest.raises(ValueError, match="total_timesteps=32 ends before"):
        agent.train(seed=0, n_timesteps=64, total_timesteps=32)
    assert records() == before
