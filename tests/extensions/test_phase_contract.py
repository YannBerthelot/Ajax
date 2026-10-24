"""The contract every agent keeps with its extensions' phases, read off one
tiny run per agent with a recorder implementing each phase it folds of
``on_target``, ``critic_loss``, ``actor_loss`` and ``post_update``:

(a) each batch holds the keys the agent declares (``phase_batch_keys``)
    and those every agent passes (``COMMON_BATCH_KEYS``);
(b) each fold gets a key no other fold gets, and none the agent draws on
    itself (a key it hands ``jax.random``) or carries on;
(c) ``critic_loss`` gets as its targets what ``on_target`` returned;
(d) a loss term reaches the optimizer of the network it names, and that
    one only: ``SLOPE * sum(theta)`` on every parameter moves each by one
    common amount (Adam steps by the gradient's sign), the critic's term
    down and the actor's (``-SLOPE * sum(theta)``) up;
(e) the metrics a phase returns beside its value reach the evaluations.

The agents are every agent of ``ajax`` folding one of these phases (read
from ``supported_extension_phases``): test_agent_contract's table, APG and
the world models, which fold post_update alone; one without a run here
fails.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax
from ajax.agents.APG.APG import APG
from ajax.extensions.base import COMMON_BATCH_KEYS, Extension
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY
from tests.agents.TDMPC2.test_TDMPC2MultiTask import make_dataset
from tests.agents.TDMPC2.toy_envs import ConstantRewardEnv
from tests.agents.test_agent_contract import CASES

PHASES = ("on_target", "critic_loss", "actor_loss", "post_update")
SHIFT = 10.0  # what on_target adds to the target
SLOPE = 1e6  # a loss term's gradient on each parameter of its network
BATCH_KEYS: dict[str, set] = defaultdict(set)  # phase -> key sets, at trace time
ROWS: list[tuple] = []  # (phase, step, context.rng, agent's rng, *values) per fold
DRAWN: list[np.ndarray] = []  # every key the run hands jax.random's functions
#: jax.random's functions the agents (and distrax, flax) draw or derive keys with.
DRAWS = "split fold_in normal uniform randint permutation choice bernoulli"
DRAWS += " categorical gumbel bits truncated_normal exponential"


def _record(
    phase: str, agent_state: Any, batch: Any, context: Any, *values: Any
) -> None:
    if batch is not None:
        BATCH_KEYS[phase].add(frozenset(batch))
    row = lambda *a: ROWS.append((phase, *map(np.asarray, a)))  # noqa: E731
    jax.debug.callback(row, context.step, context.rng, agent_state.rng, *values)


def _spy(draw: Any) -> Any:
    def spied(key: Any, *args: Any, **kwargs: Any) -> Any:
        jax.debug.callback(lambda k: DRAWN.append(np.asarray(k)), key)
        return draw(key, *args, **kwargs)

    return spied


def _total(params: Any) -> Any:
    return sum(leaf.sum() for leaf in jax.tree.leaves(params))


def on_target(self, agent_state, ext_state, batch, target, context):
    target = target + SHIFT
    _record("on_target", agent_state, batch, context, target)
    return target


def critic_loss(self, agent_state, ext_state, batch, context):
    targets = [batch["targets"]] if "targets" in batch else []
    _record("critic_loss", agent_state, batch, context, *targets)
    q_state = batch.get("q_state")
    params = batch.get("critic_params") if q_state is None else q_state.params
    return SLOPE * _total(params)


def actor_loss(self, agent_state, ext_state, batch, context):
    _record("actor_loss", agent_state, batch, context)
    return -SLOPE * _total(batch.get("actor_params"))


def post_update(self, agent_state, ext_state, context):
    _record("post_update", agent_state, None, context)
    return agent_state, ext_state


class Recorder(Extension):
    """Keeps the networks it starts with, for (d); the phases are added per
    agent (an agent refuses a phase it does not fold)."""

    name = "recorder"

    def init_state(self, agent_state, rng):
        critic = getattr(agent_state, "critic_state", None)
        return agent_state.actor_state.params, getattr(critic, "params", None)


#: Overrides of the table's runs: SAC's, TD3's and CrossQ's actors step on every
#: update; the rollout agents fold one minibatch of one epoch, so the targets of
#: an update's critic_loss folds are its on_target output, shuffled.
OVERRIDES: dict[str, dict] = {
    "SAC": {"policy_update_start": 8},
    "TD3": {"policy_delay": 1},
    "CrossQ": {"policy_delay": 1},
    "PPO": {"batch_size": 16},
    "APO": {"batch_size": 16},
    "PQN": {"n_epochs": 1, "num_minibatches": 1},
}
#: The world models' runs (what they train on, options, n_timesteps): their
#: agent tests' tiny sizes and a few updates.
WORLD_MODELS: dict[str, Any] = {
    "DreamerV3": lambda: ("Pendulum-v1", {"n_envs": 1, **DREAMER_TINY}, 16),
    "TDMPC2": lambda: (ConstantRewardEnv(10), {"seed_steps": 5, **TDMPC2_TINY}, 20),
    "TDMPC2MultiTask": lambda: (make_dataset(), {"task_dim": 4, **TDMPC2_TINY}, 4),
}


def _phases(name: str) -> frozenset:
    return frozenset(PHASES).intersection(
        getattr(ajax, name).supported_extension_phases
    )


AGENTS = [name for name in ajax.__all__ if _phases(name)]
#: The agents folding a phase that hands a batch.
BATCHED = [name for name in AGENTS if _phases(name).difference({"post_update"})]


def test_every_agent_folding_a_phase_has_a_run() -> None:
    """A clear failure, not a KeyError inside _run, for a new agent."""
    for name in AGENTS:
        has_run = name == "APG" or name in WORLD_MODELS or name in CASES
        assert has_run, f"{name} folds {sorted(_phases(name))}: add its run"


def _setup(name: str) -> tuple[Any, Any, dict, int]:
    """The agent's class, environment, options and timesteps."""
    cls: Any
    if name == "APG":
        cls, n_timesteps, env_id = APG, 64, "Pendulum-v1"
        kwargs: dict = {"n_envs": 4, "horizon": 2, "pid": None}
    elif name in WORLD_MODELS:
        cls, (env_id, kwargs, n_timesteps) = getattr(ajax, name), WORLD_MODELS[name]()
    else:  # a new agent's run goes in CASES, WORLD_MODELS or here
        case = CASES[name]
        cls, n_timesteps = case.cls, case.n_timesteps
        env_id = "Pendulum-v1" if "Pendulum-v1" in case.envs else "CartPole-v1"
        kwargs = {**case.kwargs, **OVERRIDES.get(name, {})}
    return cls, env_id, kwargs, n_timesteps


@functools.cache
def _run(name: str) -> SimpleNamespace:
    cls, env_id, kwargs, n_timesteps = _setup(name)
    phases = frozenset(PHASES).intersection(cls.supported_extension_phases)
    recorder = type("Recorder", (Recorder,), {p: globals()[p] for p in phases})()
    BATCH_KEYS.clear()
    ROWS.clear()
    DRAWN.clear()
    spies = {name: _spy(getattr(jax.random, name)) for name in DRAWS.split()}
    with mock.patch.multiple(jax.random, **spies):
        jax.clear_caches()  # a helper traced before would draw unseen
        agent = cls(env_id, **kwargs, extensions=(recorder,))
        state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
        jax.effects_barrier()
    jax.clear_caches()  # no trace keeps a spy past the run
    drawn = {tuple(k) for d in DRAWN for k in np.reshape(d, (-1, 2)).tolist()}
    return SimpleNamespace(
        phases=phases,
        batch_keys=dict(BATCH_KEYS),
        rows=list(ROWS),
        drawn=drawn,
        state=state,
    )


@pytest.mark.parametrize("name", BATCHED)
def test_each_batch_holds_the_documented_keys(name: str) -> None:
    run = _run(name)
    assert set(run.batch_keys) == run.phases.difference({"post_update"})
    for phase, key_sets in run.batch_keys.items():
        declared = _setup(name)[0].phase_batch_keys.get(phase, frozenset())
        documented = COMMON_BATCH_KEYS.get(phase, frozenset()).union(declared)
        for keys in key_sets:
            assert not documented.difference(keys), (phase, sorted(keys))


@pytest.mark.parametrize("name", AGENTS)
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
    drawn = sum(key in run.drawn for key in map(tuple, keys.tolist()))
    assert not drawn, f"the agent draws on {drawn} of {len(keys)} fold keys itself"


@pytest.mark.parametrize("name", [n for n in AGENTS if "on_target" in _phases(n)])
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
        want, have = (np.sort(np.concatenate(x)) for x in (returned, got))
        np.testing.assert_allclose(have, want, rtol=1e-6, err_msg=str(step))


#: Agents whose actor loss runs the rollout it differentiates (APG: Busetto
#: et al., 2024), so it sees the env-step count before that rollout.
ROLLOUT_IN_THE_ACTOR_LOSS = {"APG"}


@pytest.mark.parametrize("name", [n for n in AGENTS if "post_update" in _phases(n)])
def test_every_loss_term_sees_the_collectors_step(name: str) -> None:
    """``context.step`` is the collector's env-step count at the update: the steps
    the loss terms see are among those ``post_update`` sees (one rollout
    earlier for an actor loss that runs the rollout)."""
    steps: dict[str, set] = defaultdict(set)
    for phase, step, *_ in _run(name).rows:
        steps[phase].update(np.ravel(step).tolist())
    after = steps["post_update"]
    for phase in ("on_target", "critic_loss", "actor_loss"):
        seen = steps.get(phase, set())
        if phase == "actor_loss" and name in ROLLOUT_IN_THE_ACTOR_LOSS:
            seen = {s + min(after) for s in seen}
        assert seen <= after, (phase, steps)


def _flat(params: Any) -> np.ndarray:
    return np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(params)])


@pytest.mark.parametrize("name", BATCHED)
def test_each_loss_term_moves_its_own_network(name: str) -> None:
    run = _run(name)
    (actor, critic), state = run.state.ext_state[0], run.state
    networks = {"actor_loss": (actor, state.actor_state.params, 1.0)}
    if "critic_loss" in run.phases:  # DQN's and PQN's Q-network is the actor
        q_net = any("q_state" in keys for keys in run.batch_keys["critic_loss"])
        after = state.actor_state.params if q_net else state.critic_state.params
        networks["critic_loss"] = (actor if q_net else critic, after, -1.0)
    for phase in run.phases.intersection(networks):
        before, after, sign = networks[phase]
        moved = _flat(after) - _flat(before)
        common = moved.mean()
        assert np.sign(common) == sign, (phase, common)
        assert np.abs(moved - common).max() <= 1e-3 * abs(common), phase


#: The constant each phase logs beside its value, for (e).
LOGGED = {"on_target": 1.0, "critic_loss": 2.0, "actor_loss": 3.0}


def _logging(phase: str) -> Any:
    """``phase`` leaving its value as it is and logging its constant."""

    def logged(self: Any, agent_state: Any, ext_state: Any, *args: Any) -> Any:
        value = args[1] if phase == "on_target" else 0.0  # (batch, target, context)
        return value, {f"logger/{phase}": jnp.asarray(LOGGED[phase])}

    return logged


#: PPO's time-preserving minibatches fold on_target in each minibatch.
UNROLL = {"unroll_length": 8, "num_minibatches": 2}


@pytest.mark.parametrize(
    "name, options",
    [(name, {}) for name in BATCHED] + [("PPO", UNROLL)],
    ids=[*BATCHED, "PPO-unroll"],
)
def test_each_phase_logs_its_metrics_with_the_updates(name: str, options: dict) -> None:
    """Every evaluation logs each folded phase's constant, as the last
    update's or the update's mean."""
    cls, env_id, kwargs, n_timesteps = _setup(name)
    kwargs = {**kwargs, **options}
    phases = frozenset(LOGGED).intersection(cls.supported_extension_phases)
    logger = type("Logger", (Extension,), {p: _logging(p) for p in phases})()
    agent = cls(env_id, **kwargs, extensions=(logger,))
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent.train(
        seed=0, n_timesteps=n_timesteps, num_episode_test=1, logging_config=logging
    )
    if name == "APG":  # its aux, every update's, holds what each evaluated
        out = out.evaluations
    evaluated = np.asarray(out["timestep"]).reshape(-1) >= 0
    assert evaluated.any()
    for phase in phases:
        logged = np.asarray(out[f"logger/{phase}"]).reshape(-1)[evaluated]
        assert (logged == LOGGED[phase]).all(), (phase, logged)
