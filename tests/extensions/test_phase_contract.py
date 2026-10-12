"""The contract every agent keeps with its extensions' phases, read off one
tiny run per agent with a recorder implementing each phase it folds of
``on_target``, ``critic_loss``, ``actor_loss`` and ``post_update``:

(a) each batch holds the keys ``PHASE_BATCH_KEYS`` documents;
(b) each fold gets a key no other fold gets, and never the one the agent
    carries on to draw its own;
(c) ``critic_loss`` gets as its targets what ``on_target`` returned;
(d) a loss term reaches the optimizer of the network it names, and that
    one only: ``SLOPE * sum(theta)`` on every parameter moves each by one
    common amount (Adam steps by the gradient's sign), the critic's term
    down and the actor's (``-SLOPE * sum(theta)``) up.

The agents are test_agent_contract's table and APG; a new agent joins by
its row there, its phases read from ``supported_extension_phases``.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

from ajax.agents.APG.APG import APG
from ajax.extensions.base import PHASE_BATCH_KEYS, Extension
from tests.agents.test_agent_contract import CASES

PHASES = ("on_target", "critic_loss", "actor_loss", "post_update")
SHIFT = 10.0  # what on_target adds to the target
SLOPE = 1e6  # a loss term's gradient on each parameter of its network
BATCH_KEYS: dict[str, set] = defaultdict(set)  # phase -> key sets, at trace time
ROWS: list[tuple] = []  # (phase, step, ctx.rng, agent's rng, *values) per fold


def _record(phase: str, agent_state: Any, batch: Any, ctx: Any, *values: Any) -> None:
    if batch is not None:
        BATCH_KEYS[phase].add(frozenset(batch))
    row = lambda *a: ROWS.append((phase, *map(np.asarray, a)))  # noqa: E731
    jax.debug.callback(row, ctx.step, ctx.rng, agent_state.rng, *values)


def _total(params: Any) -> Any:
    return sum(leaf.sum() for leaf in jax.tree.leaves(params))


def on_target(self, agent_state, ext_state, batch, target, ctx):
    target = target + SHIFT
    _record("on_target", agent_state, batch, ctx, target)
    return target


def critic_loss(self, agent_state, ext_state, batch, ctx):
    targets = [batch["targets"]] if "targets" in batch else []
    _record("critic_loss", agent_state, batch, ctx, *targets)
    q_state = batch.get("q_state")
    params = batch.get("critic_params") if q_state is None else q_state.params
    return SLOPE * _total(params)


def actor_loss(self, agent_state, ext_state, batch, ctx):
    _record("actor_loss", agent_state, batch, ctx)
    return -SLOPE * _total(batch.get("actor_params"))


def post_update(self, agent_state, ext_state, ctx):
    _record("post_update", agent_state, None, ctx)
    return agent_state, ext_state


class Recorder(Extension):
    """Keeps the networks it starts with, for (d); the phases are added per
    agent (an agent refuses a phase it does not fold)."""

    name = "recorder"

    def init_state(self, agent_state, rng):
        critic = getattr(agent_state, "critic_state", None)
        return agent_state.actor_state.params, getattr(critic, "params", None)


#: Overrides of the table's runs: SAC's actor steps from its first update;
#: the rollout agents fold one minibatch of one epoch, so the targets of
#: an update's critic_loss folds are its on_target output, shuffled.
OVERRIDES: dict[str, dict] = {
    "SAC": {"policy_update_start": 8},
    "PPO": {"batch_size": 16},
    "APO": {"batch_size": 16},
    "PQN": {"n_epochs": 1, "num_minibatches": 1},
}
AGENTS = [*sorted(CASES), "APG"]


@functools.cache
def _run(name: str) -> SimpleNamespace:
    cls: Any
    if name == "APG":
        cls, n_timesteps, env_id = APG, 64, "Pendulum-v1"
        kwargs: dict = {"n_envs": 4, "horizon": 2, "pid": None}
    else:
        case = CASES[name]
        cls, n_timesteps = case.cls, case.n_timesteps
        env_id = "Pendulum-v1" if "Pendulum-v1" in case.envs else "CartPole-v1"
        kwargs = case.kwargs | OVERRIDES.get(name, {})
    phases = frozenset(PHASES) & cls.supported_extension_phases
    recorder = type("Recorder", (Recorder,), {p: globals()[p] for p in phases})()
    BATCH_KEYS.clear()
    ROWS.clear()
    agent = cls(env_id, **kwargs, extensions=(recorder,))
    state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
    jax.effects_barrier()
    return SimpleNamespace(
        phases=phases, batch_keys=dict(BATCH_KEYS), rows=list(ROWS), state=state
    )


@pytest.mark.parametrize("name", AGENTS)
def test_each_batch_holds_the_documented_keys(name: str) -> None:
    run = _run(name)
    assert set(run.batch_keys) == run.phases - {"post_update"}
    for phase, key_sets in run.batch_keys.items():
        rows = PHASE_BATCH_KEYS[phase].items()
        ours = [
            keys for agents, keys in rows if agents == "*" or name in agents.split()
        ]
        documented = set(" ".join(ours).split())
        for keys in key_sets:
            assert not documented - keys, (phase, sorted(keys))


SHARED_KEYS = pytest.mark.xfail(
    strict=True,
    reason="fix/extension-keys: on_target and critic_loss share the TD target's"
    " key (TD3 and AVG fold actor_loss on the agent's own)",
)
_SHARED = ("ASAC", "AVG", "REDQ", "SAC", "TD3")


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=SHARED_KEYS) if n in _SHARED else n for n in AGENTS]
)
def test_each_fold_gets_a_key_of_its_own(name: str) -> None:
    """Across phases, folds, epochs and minibatches."""
    run = _run(name)
    assert {row[0] for row in run.rows} == run.phases  # every phase was folded
    keys, own = (
        np.concatenate([np.reshape(row[i], (-1, 2)) for row in run.rows])
        for i in (2, 3)
    )
    shared = len(keys) - len(np.unique(keys, axis=0))
    assert not shared, f"{shared} of {len(keys)} folds got a key another fold got"
    assert not (keys == own).all(axis=1).any(), "a fold got the agent's own key"


@pytest.mark.parametrize("name", [n for n in AGENTS if n not in ("APG", "UDRL")])
def test_critic_loss_gets_the_target_on_target_returns(name: str) -> None:
    """As a multiset over each update (a step of the collector)."""
    by_step: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for phase, step, _, _, *values in _run(name).rows:
        if phase in ("on_target", "critic_loss"):
            pair = by_step[int(np.ravel(step)[0])]
            pair[phase == "critic_loss"].extend(np.ravel(v) for v in values)
    assert by_step
    for step, (returned, got) in by_step.items():
        assert returned and got, step
        returned, got = (np.sort(np.concatenate(x)) for x in (returned, got))
        np.testing.assert_allclose(got, returned, rtol=1e-6, err_msg=str(step))


def _flat(params: Any) -> np.ndarray:
    return np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(params)])


@pytest.mark.parametrize("name", AGENTS)
def test_each_loss_term_moves_its_own_network(name: str) -> None:
    run = _run(name)
    (actor, critic), state = run.state.ext_state[0], run.state
    networks = {"actor_loss": (actor, state.actor_state.params, 1.0)}
    if "critic_loss" in run.phases:  # DQN's and PQN's Q-network is the actor
        q_net = any("q_state" in keys for keys in run.batch_keys["critic_loss"])
        after = state.actor_state.params if q_net else state.critic_state.params
        networks["critic_loss"] = (actor if q_net else critic, after, -1.0)
    for phase in run.phases & set(networks):
        before, after, sign = networks[phase]
        moved = _flat(after) - _flat(before)
        common = moved.mean()
        assert np.sign(common) == sign, (phase, common)
        assert np.abs(moved - common).max() <= 1e-3 * abs(common), phase
