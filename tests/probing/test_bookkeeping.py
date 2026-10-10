"""Bookkeeping: what an agent counts, logs, stores, draws and executes around
training, read on tiny MDPs whose answers are known exactly (P2 counter, P7
UDRL commands, Q14 command transform, Q3 task wrapper, Q4 expert warm-up, Q5
declared box, Q8 frozen-actor ratio). Most checks are exact equalities on
seeds trained in one program; the last test judges every case.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections import Counter
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax.agents.UDRL.train_UDRL as udrl
from ajax.evaluate import evaluate, setup_environment
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.wrappers import InitialStateWrapper

from . import agents, envs, runs
from . import readouts as R
from .verdict import STAGE_1, Case, Query, check, params, xfail, xparam

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "a8c5ecad49aa"  # verdict.digest(CASES): every answer, pinned
L = 7


@functools.cache
def counter(
    actions: int = 0, terminate: bool = True, record: tuple = (32, 64)
) -> envs.Spec:
    """Reward 1 per step (plus a reset wrapper's flag), every episode L
    steps; the time limit, L + 1 (L when episodes only truncate), is also
    the evaluation's scan length, so it stays small."""

    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        return st.replace(s=st.s + 1), 1.0 + st.flag, terminate & (st.s + 1 >= L)

    def obs(st: envs.State) -> jax.Array:
        return envs.one_hot(jnp.minimum(st.s, L), L + 1)

    limit = L + 1 if terminate else L
    return envs.Spec(
        "counter",
        transition,
        obs,
        obs_dim=L + 1,
        actions=actions,
        limit=limit,
        record=record,
    )


# --- P2: budget, returns, log events, optimiser steps, keys, seeds ----------
# Readings tagged (a) final timestep, (b) returns, (c) log events and their
# values, (d) optimiser steps, (e) repeated 8-step windows of the recorded
# step-key draws (u) and executed actions (a), (f) distinct seeds.

LOG_EVERY, BUDGET, SEEDS = 140, {1: 1400, 4: 2800}, tuple(range(8))
TRAIN, EVAL, LENGTH = runs.TRAIN_KEYS
P2_AGENTS = ("SAC", "ASAC", "REDQ", "TD3", "AVG", "PPO", "APO", "DQN", "PQN")
ROLLOUT = {"PPO": 32, "APO": 32, "PQN": 16}


@functools.cache
def p2_run(agent: str, n_envs: int, actions: int = -1, terminate: bool = True, **kw):
    """The bookkeeping preset on the recording counter (Discrete(2) for DQN
    and PQN), logging every 140 steps."""
    actions = (2 if agent in ("DQN", "PQN") else 0) if actions < 0 else actions
    env, params = counter(actions, terminate).make()
    model = agents.make(agent, env, params, preset="bookkeeping", n_envs=n_envs, **kw)
    return runs.contract_readings(runs.train(model, SEEDS, BUDGET[n_envs], LOG_EVERY))


def iterations(agent: str, n_envs: int) -> list[int]:
    return runs.iterations(agent, n_envs, BUDGET[n_envs], ROLLOUT.get(agent, 1))


def log_timesteps(agent: str, n_envs: int) -> list[int]:
    """Replay of evaluate_and_log's gate (log.py:276-293, 411-413): the
    rollout agents' overshoot iteration is never logged."""
    n_logs, out = 0, []
    for t in iterations(agent, n_envs):
        due = t - n_logs * LOG_EVERY >= LOG_EVERY
        if due and 1 < t <= BUDGET[n_envs]:
            out.append(t)
        n_logs += due
    return out


def steps(agent: str, n_envs: int, starts: int) -> dict[str, int]:
    """Each optimiser's final ``.step`` from the first post-collection
    timestep >= ``starts``; AVG never updates its temperature
    (train_AVG.update_agent), DQN and PQN never their critic_state copy."""
    its = iterations(agent, n_envs)
    u, rollout = sum(t >= starts for t in its), len(its) * 2 * 2
    if agent == "SAC":
        actor, alpha = (sum(t >= max(starts, s) for t in its) for s in (200, 300))
        return {"critic": u, "actor": actor, "alpha": alpha}
    return {
        "TD3": {"critic": u, "actor": math.ceil(u / 3)},
        "REDQ": {"critic": 3 * u, "actor": u, "alpha": u},
        "ASAC": {"critic": u, "actor": u, "alpha": u},
        "AVG": {"critic": u, "actor": u, "alpha": 0},
        "DQN": {"actor": u, "critic": 0},
        "PQN": {"actor": rollout, "critic": 0},
        "PPO": {"actor": rollout, "critic": rollout},
        "APO": {"actor": rollout, "critic": rollout},
    }[agent]


def repeats(record: np.ndarray, saturated: bool) -> dict[str, float]:
    """Share of 8-step windows of ``record`` (seeds, envs, steps) seen more
    than once within an env, across envs and across seeds; ``saturated``
    drops windows holding +-1 (clipped actions repeat without key reuse)."""
    win = np.lib.stride_tricks.sliding_window_view(record, 8, axis=-1)
    keep = ~(np.abs(win) >= 1.0).any(-1) if saturated else np.ones(win.shape[:3], bool)

    def share(axes: tuple[int, int, int]) -> float:
        w, k = win.transpose(*axes, 3), keep.transpose(axes)
        flags: list[bool] = []
        for group, kept in zip(w.reshape(-1, *w.shape[2:]), k.reshape(-1, k.shape[2])):
            seen = Counter(x.tobytes() for x, ok in zip(group, kept) if ok)
            flags += [seen[x.tobytes()] > 1 for x, ok in zip(group, kept) if ok]
        return float(np.mean(flags)) if flags else 0.0

    out = {"within env": share((0, 1, 2))}
    if win.shape[1] > 1:
        out["across envs"] = share((0, 2, 1))
    if win.shape[0] > 1:
        out["across seeds"] = share((1, 2, 0))
    return out


def errors(agent: str, n_envs: int, r: dict, starts: int) -> list[tuple[str, str]]:
    """Every reading off its exact answer, as (tag, message)."""
    out, final = [], iterations(agent, n_envs)[-1]
    if not (r["timestep"] == final).all():
        out.append(("a", f"final timestep {r['timestep']}, right {final}"))
    if not ((r["mean return"] == L).all() and (r["returns"] == L).all()):
        out.append(("b", f"returns {Counter(r['returns'].tolist())}"))
    if any(list(e[e >= 0]) != log_timesteps(agent, n_envs) for e in r["events"]):
        out.append(("c", f"log events per seed {(r['events'] >= 0).sum(1)}"))
    for k in (TRAIN, EVAL, LENGTH):
        if (r[k] != L).any():
            out.append(("c", f"'{k}' {sorted(set(r[k]))}"))
    for name, want in steps(agent, n_envs, starts).items():
        if not np.all(r.get(f"steps {name}", -1) == want):
            out.append(("d", f"{name} steps {r.get(f'steps {name}')}, right {want}"))
    for f in ("first_u", "last_u", "first_a", "last_a"):
        if f.endswith("_a") and agent in ("DQN", "PQN"):
            continue  # discrete actions repeat by chance
        for where, x in repeats(r[f], f.endswith("_a")).items():
            if x:
                out.append((f"e:{f}:{where}", f"{f} windows repeated {where}: {x:.3f}"))
    if len(set(r["checksums"].tolist())) != len(SEEDS):
        out.append(("f", f"{len(set(r['checksums'].tolist()))} distinct seeds"))
    return out


def test_p2_oracle_reproduces_the_replayed_counts() -> None:
    """The replayed counts: final timestep 1400/2800, 1408/2816 for rollout
    agents; 10/20 and 9/19 log events; AVG from timestep 100, 676 updates."""
    for agent, n, final, events in (
        ("SAC", 1, 1400, 10),
        ("SAC", 4, 2800, 20),
        ("PPO", 1, 1408, 9),
        ("PPO", 4, 2816, 19),
        ("APO", 4, 2816, 19),
        ("PQN", 1, 1408, 9),
        ("PQN", 4, 2816, 19),
    ):
        got = iterations(agent, n)[-1], len(log_timesteps(agent, n))
        assert got == (final, events), agent
    assert steps("AVG", 4, 100) == {"critic": 676, "actor": 676, "alpha": 0}


P2_CELLS = [(a, n, True) for a in P2_AGENTS for n in (1, 4) if (a, n) != ("AVG", 4)]


@pytest.mark.parametrize(
    ("agent", "n_envs", "terminate"),
    [pytest.param(*c, id=f"{c[0]}-n{c[1]}") for c in P2_CELLS]
    + [pytest.param(a, 4, False, id=f"{a}-n4-truncation") for a in ("SAC", "PPO")],
)
def test_p2_bookkeeping(agent: str, n_envs: int, terminate: bool) -> None:
    """Budget spent, returns 7, one log event per 140 steps with Train = Eval
    = 7 and length 7, optimiser steps, fresh keys, 8 distinct seeds; also
    when episodes only truncate (a truncation closes a return too)."""
    starts = 0 if agent == "AVG" else 100
    found = errors(agent, n_envs, p2_run(agent, n_envs, terminate=terminate), starts)
    assert not found, found


def _warm_up(tag: str) -> bool:
    return tag.startswith("e:first_") and tag.endswith("within env")


AVG_PARTS = {
    "budget": lambda tag: tag in ("a", "d"),
    "warm-up keys": _warm_up,
    "rest": lambda tag: tag not in ("a", "d") and not _warm_up(tag),
}


@pytest.mark.parametrize("part", list(AVG_PARTS))
def test_p2_avg_with_parallel_envs(part: str) -> None:
    """AVG at 4 envs from timestep 100: budget, warm-up keys, the rest."""
    found = errors("AVG", 4, p2_run("AVG", 4, learning_starts=100), 100)
    assert not [e for e in found if AVG_PARTS[part](e[0])], found


@xfail(
    "with normalize_rewards=True the Train return is logged in normalised units: no gamma reaches the normaliser (base.py:128-137, create.py:385), a constant reward has std sqrt(1e-8) (utils.py:24-29), and the collector adds the normalised reward to the Train window (interaction.py:905-909); right Train 7, today 70000 (Eval reads 7)"
)
@pytest.mark.parametrize("agent", ["SAC", "PPO", "DQN", "PQN"])
def test_p2_train_return_in_raw_units_with_normalized_rewards(agent: str) -> None:
    r = p2_run(agent, 1, normalize_rewards=True)
    keys = (TRAIN, EVAL, "mean return", "returns")
    wrong = {k: set(r[k].round(4)) for k in keys if not np.allclose(r[k], L, 1e-5, 0)}
    assert not wrong, wrong


@xfail(
    "PPO's _force_reset replaces the env state, last_obs and rng but not episodic_return_state.cumulative_reward (train_PPO.py:1282-1323): the cut episode's 32 mod 7 = 4 steps carry over; right every return 7, today 16 of 80 are 11 (mean 7.8)"
)
def test_p2_ppo_forced_reset_starts_each_return_at_zero() -> None:
    """A reset after every 32-step rollout (num_evals 45: reset_every 1)."""
    r = p2_run("PPO", 1, num_resets_per_eval=1, num_evals=45)
    exact = (r["returns"] == L).all() and (r["mean return"] == L).all()
    assert exact, Counter(r["returns"].tolist())


@xfail(
    "prepare_env wraps discrete envs in ClipAction(-1, 1) whenever it normalises (create.py:375-386), after the int32 cast (interaction.py:204-205): action 2 runs as 1; right each action on 1/3 of the uniform steps, today action 2 on 0.0 and action 1 on 0.654"
)
@pytest.mark.parametrize("agent", ["DQN", "PQN"])
def test_p2_discrete_actions_are_executed_as_chosen(agent: str) -> None:
    """Uniform exploration on Discrete(3), observations normalised: each
    executed action's share within 0.1 of 1/3 (0.03 off unnormalised)."""
    kw = {"epsilon_start": 1.0, "epsilon_end": 1.0}
    if agent == "DQN":
        kw = {"learning_starts": 200}
    r = p2_run(agent, 4, 3, normalize_observations=True, **kw)
    fields = ("first_a", "last_a") if agent == "PQN" else ("first_a",)
    executed = np.concatenate([r[f].reshape(-1) for f in fields])
    shares = [float((executed == a).mean()) for a in range(3)]
    assert all(abs(s - 1 / 3) <= 0.1 for s in shares), shares


def p2_worker(agent: str, folder: str) -> dict:
    env, params = counter().make()
    model = agents.make(agent, env, params, preset="bookkeeping")
    return runs.worker(model, SEEDS, BUDGET[1], LOG_EVERY, folder)


@pytest.mark.parametrize("agent", ["SAC", "AVG"])
def test_p2_logging_worker_writes_every_event(agent: str, tmp_path: Any) -> None:
    """Through the real logging process, in a fresh interpreter with a
    timeout: 8 runs of 10 events, Train = Eval = 7, length 7."""
    worker = "tests.probing.test_bookkeeping:p2_worker"
    logged = runs.in_subprocess(worker, [agent, str(tmp_path)], 300)
    want = [L] * len(log_timesteps(agent, 1))
    wrong = {run: v for run, v in logged.items() if any(x != want for x in v.values())}
    assert len(want) == 10 and len(logged) == len(SEEDS) and not wrong, wrong


# --- P7: UDRL's hindsight commands on episodes of two segments --------------
# The chain: constant observation, reward = action in {0, 1}, truncated after
# 8 steps; 4-step segments, so every episode spans two buffer slots. Commands
# scaled 1/8 (the defaults put (0, 1) and (1, 1) 0.02 apart); learning rate
# 1e-4, at which none of 96 seeds committed to one action before its buffer
# held the other. A label (h, h) was earned by h ones, (0, h) by h zeros.

EP, SEG = 8, 4


def _chain(st: envs.State, a: jax.Array, key: Any) -> tuple:
    return st.replace(s=st.s + 1), a.astype(jnp.float32), False


CHAIN = envs.Spec("udrl_chain", _chain, lambda st: jnp.zeros(1), actions=2, limit=EP)
P7_UDRL: dict[str, Any] = {"n_envs": 1, "n_steps": SEG, "n_updates_per_iter": SEG}
P7_UDRL |= {"actor_learning_rate": 1e-4, "actor_architecture": agents.NET}
P7_UDRL |= {"command_scale_r": 1 / EP, "command_scale_h": 1 / EP}
FIXED = {"command_target_tau": 0.0, "command_return_init": EP / 2}
FIXED |= {"command_horizon_init": float(EP)}


def udrl_agent(**kw: Any) -> Any:
    return agents.make("UDRL", *CHAIN.make(), preset=None, **{**P7_UDRL, **kw})


def stream(buffer: Any, i: int) -> tuple[np.ndarray, ...]:
    """Seed i's FIFO buffer as one stream, oldest first: the raw command
    (T, 2), the action, the reward and the done flag (one env)."""
    cap, fill, w = buffer.obs.shape[1], buffer.fill_count[i], buffer.write_idx[i]
    order = [(w + j) % cap for j in range(cap)] if fill == cap else range(fill)
    leaves = (buffer.obs, buffer.actions, buffer.rewards, buffer.dones)
    x = [np.asarray(v[i])[np.asarray(order)] for v in leaves]
    obs, *rest = [v.reshape(-1, v.shape[-1]) for v in x]
    return obs[:, -2:], *(v[:, 0] for v in rest)


def episodes(done: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) step of every whole episode the stream holds."""
    return [(int(e) - EP + 1, int(e)) for e in np.flatnonzero(done > 0) if e >= EP - 1]


@xfail(
    "UDRL's top-K command statistics restart their running sums at every segment (UDRL/buffer.py:194), so each 8-step episode counts as its last 4-step fragment; right command_target_horizon 8.0, today 4.0"
)
def test_p7_command_target_describes_whole_episodes() -> None:
    """32 segments into 16 slots, top 4 of 8 episodes: target horizon 8, target
    return the top-4 whole-episode mean, every start commanded horizon 8."""
    s = runs.train(udrl_agent(buffer_capacity=16, command_topk=4), (0, 1), 128).state
    for i in range(2):
        cmd, _, reward, done = stream(s.buffer, i)
        whole, ends = episodes(done), np.flatnonzero(done > 0)
        top = sorted((reward[a : b + 1].sum() for a, b in whole), reverse=True)
        right = float(np.mean(top[:4])) if whole else np.nan
        starts = sorted({float(cmd[a, 1]) for a, _ in whole})
        every = len(ends) and (np.diff(ends) == EP).all() and ends[0] < EP
        every = every and len(done) - 1 - ends[-1] < EP  # dones every 8 steps
        got = (float(s.command_target_horizon[i]), float(s.command_target_return[i]))
        ok = got[0] == EP and abs(got[1] - right) <= 1e-5 and starts == [EP]
        assert ok and every, f"(horizon, return target) {got}, right (8, {right})"


def test_p7_stored_commands_follow_the_hindsight_recurrence() -> None:
    """Within each buffered episode, across its segment boundary too, the
    stored command decays by what the step earned: d_r' = d_r - r, d_h' =
    max(d_h - 1, 1); every episode starts at (4, 8)."""
    s = runs.train(udrl_agent(buffer_capacity=16, **FIXED), (0, 1), 128).state
    for i in range(2):
        cmd, _, reward, done = stream(s.buffer, i)
        same = done[:-1] == 0
        crossing = same & (np.arange(len(same)) % SEG == SEG - 1)
        starts = [a for a, _ in episodes(done)]
        if not (crossing.any() and (same & (reward[:-1] > 0)).any() and starts):
            raise RuntimeError(f"seed {i}: nothing to check in the buffer")
        d_r = (cmd[1:, 0] - (cmd[:-1, 0] - reward[:-1]))[same]
        d_h = (cmd[1:, 1] - np.maximum(cmd[:-1, 1] - 1.0, 1.0))[same]
        assert np.abs(d_r).max() <= 1e-5, f"d_r error {np.unique(d_r.round(3))}"
        assert np.abs(d_h).max() <= 1e-5, f"d_h error {np.unique(d_h.round(3))}"
        assert np.abs(cmd[starts] - (EP / 2, EP)).max() <= 1e-5, cmd[starts]


LABELS = ((0.0, 1.0), (1.0, 1.0), (0.0, 2.0), (2.0, 2.0))
P7_COMMANDS = {f"P(a=1 | d_r={r:g}, d_h={h:g})": (r, h) for r, h in LABELS}


def _p7_read(run: runs.Run) -> dict:
    """P(a=1) at each command, scaled as the collector scales it; raises
    unless each queried label occurs in every seed's final buffer."""
    for i, seed in enumerate(run.seeds):
        acts = stream(run.state.buffer, i)[1].reshape(-1, SEG)
        for r, h in P7_COMMANDS.values():
            # the windows today's sampler draws: t1 in 0..2, inside one slot
            if not sum((acts[:, t : t + int(h)].sum(1) == r).sum() for t in range(3)):
                raise RuntimeError(f"seed {seed}: no window labelled ({r}, {h})")

    def probs(n: R.Nets) -> dict:
        out = {}
        for k, c in P7_COMMANDS.items():
            x = udrl._scale_obs_command(jnp.array([[0.0, *c]]), 1 / EP, 1 / EP)
            out[k] = R.pi(n, x).probs.reshape(-1)[1]
        return out

    return R.per_seed(probs, R.nets(run.state))


CASES["p7-UDRL"] = Case(
    "p7-UDRL",
    tuple(
        Query(k, float(r == h), {"command ignored": 0.5})
        for k, (r, h) in P7_COMMANDS.items()
    ),
    runs.readings(lambda: udrl_agent(**FIXED), _p7_read),
    2048,
    (0.02, 0.02, 0.027, 0.06),
    note="1024 failed (0.078 on (2, 2)); cert 32/32; every label >= 25 windows",
)


# --- Q14: UDRL scales its command alike when acting and when training -------
# P7's chain, the command pinned at (3, 8) (tau 0), scales 0.04 and 0.01
# (unequal: a one-site fault is then not a pure change of magnitude). Every
# episode must return exactly 3: P(a=1 | d_r = 0) = 0 and P(a=1 | d_r = d_h)
# = 1 are exact hindsight labels and leave no other return reachable.

Q14 = {"command_target_tau": 0.0, "command_return_init": 3.0}
Q14 |= {"command_horizon_init": 8.0, "command_scale_r": 0.04, "command_scale_h": 0.01}
Q14 |= {"actor_learning_rate": 3e-4, "buffer_capacity": 64}
SHARE = "share of episodes returning exactly 3"


def _q14_read(run: runs.Run) -> dict:
    """Per seed, the share of the final buffer's whole episodes returning
    exactly 3; raises unless there are 32, each commanded (3, 8)."""
    share = []
    for i, seed in enumerate(run.seeds):
        cmd, _, reward, done = stream(run.state.buffer, i)
        spans = episodes(done)
        if len(spans) != 32 or any((cmd[a] != (3.0, 8.0)).any() for a, _ in spans):
            raise RuntimeError(f"seed {seed}: not 32 whole episodes commanded (3, 8)")
        share.append(np.mean([reward[a : b + 1].sum() == 3.0 for a, b in spans]))
    return {SHARE: np.array(share)}


Q14_WRONG = {"command ignored (at most)": 56 * (3 / 8) ** 3 * (5 / 8) ** 5}
Q14_WRONG |= {"acting site unscaled (static estimate)": 0.093}
CASES["q14-UDRL"] = Case(
    "q14-UDRL",
    (Query(SHARE, 1.0, Q14_WRONG),),
    runs.readings(lambda: udrl_agent(**Q14), _q14_read),
    4096,
    (2 / 32,),
    note="1024, 2048 failed; 2/32: at most 2 of 32 episodes off 3; cert 32/32",
)


def test_q14_actor_sees_the_same_command_when_acting_and_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One step acts on the raw command (3, 8) and training batches carry it
    too: get_pi must be fed the declared (0, 3 * 0.04, 8 * 0.01) at both
    sites, and the buffer keep the raw (0, 3, 8)."""
    seen, get_pi = [], udrl.get_pi
    raw = np.array([0.0, 3.0, 8.0], np.float32)

    def recording(actor_state: Any, actor_params: Any, obs: Any, **kw: Any) -> Any:
        jax.debug.callback(lambda o: seen.append(np.asarray(o)), obs)
        return get_pi(actor_state, actor_params, obs, **kw)

    def batch(key: Any, buffer: Any, size: int) -> tuple:
        obs = jnp.tile(jnp.asarray(raw), (size, 1))
        return obs, jnp.zeros((size, 1), jnp.int32), obs[:, -2], obs[:, -1]

    monkeypatch.setattr(udrl, "get_pi", recording)
    monkeypatch.setattr(udrl, "sample_training_batch", batch)
    one = {"n_steps": 1, "n_updates_per_iter": 1, "batch_size": 2}
    state = runs.train(udrl_agent(**Q14, **one), (0,), 1).state
    acting = [x.reshape(-1, 3)[0] for x in seen if x.shape[-2] == 1]
    training = [x.reshape(-1, 3) for x in seen if x.shape[-2] == 2]
    if len(acting) != 1 or not training:
        raise RuntimeError(f"recorded input shapes {[x.shape for x in seen]}")
    np.testing.assert_array_equal(np.asarray(state.buffer.obs)[0, 0, 0, 0], raw)
    fed = np.concatenate(training)
    assert (fed == acting[0]).all(), (acting[0], np.unique(fed, axis=0))
    declared = raw * np.array([1.0, 0.04, 0.01], np.float32)
    np.testing.assert_allclose(acting[0], declared, 1e-6)


# --- Q3: a task wrapper holds on every episode, in training and evaluation --
# The counter wrapped in InitialStateWrapper, whose init_state_fn raises the
# flag: every episode of the task returns 14 in 7 steps (APG: 32 per 16-step
# rollout) whatever the policy; 7 is the bare env. Two seeds, exact.

BYPASS = "InitialStateWrapper overrides reset only (wrappers.py:253-272) and gymnax's step auto-resets with the inner reset_env (gymnax environment.py:82), so only each env's first episode is wrapped; right 14 per episode (APG 32 per rollout), today 7 (UDRL's segments 10.5 then 7, APG 23)"


def _raise_flag(key: Any, state: envs.State, params: Any) -> envs.State:
    return state.replace(flag=jnp.ones_like(state.flag))


def task(actions: int = 0, terminate: bool = True) -> tuple:
    spec = dataclasses.replace(counter(actions, terminate, (0, 0)), gradients=True)
    env, params = spec.make()
    return InitialStateWrapper(env, _raise_flag), params


# agent: (actions, terminates, budget, log every, the probe's own overrides);
# UDRL's 14-step segment holds two whole episodes; TD-MPC2 only truncates.
Q3_AGENTS: dict[str, tuple] = {a: (0, True, 210, 70, {}) for a in P2_AGENTS[:8]}
Q3_AGENTS |= {"DQN": (2, True, 210, 70, {}), "PQN": (2, True, 210, 70, {})}
Q3_AGENTS |= {"UDRL": (0, True, 98, None, {"n_steps": 14})}
Q3_AGENTS |= {"DreamerV3": (0, True, 140, 70, {})}
Q3_AGENTS |= {"TDMPC2": (0, False, 98, 49, {"seed_steps": 1000})}
Q3_AGENTS |= {"APG": (0, True, 128, 32, {})}


@functools.cache
def q3_run(agent: str) -> dict[str, np.ndarray]:
    """The training returns the agent keeps (collector window, UDRL's segment
    means, APG's rollouts), the last Train mean, every Eval return, length."""
    actions, terminate, budget, every, kw = Q3_AGENTS[agent]
    model = agents.make(agent, *task(actions, terminate), preset="bookkeeping", **kw)
    run = runs.train(model, (0, 1), budget, every, num_episode_test=4)
    out = {k: run.values(k) for k in (EVAL, LENGTH, "Train/matching_loss")}
    out = {k: v[~np.isnan(v)] for k, v in out.items()} | {TRAIN: run.logged(TRAIN)}
    s, train = run.state, -out["Train/matching_loss"].reshape(2, -1)
    if agent == "UDRL":
        b = s.buffer
        paid, ends = (np.asarray(x).sum((2, 3, 4)) for x in (b.rewards, b.dones))
        train = (paid / ends)[:, : int(b.fill_count[0])]
    elif agent != "APG":
        window = s.collector_state.episodic_return_state
        if (np.asarray(window.count) != window.buffer.shape[1]).any():
            raise RuntimeError("the rolling window must be full: lengthen the run")
        train = np.asarray(window.buffer).reshape(2, -1)
    return out | {"train": train}


@xfail(BYPASS)
def test_q3_auto_reset_starts_the_next_episode_from_the_wrapper() -> None:
    """Three episodes stepped from the task's reset through gymnax's
    auto-reset each return 14."""
    env, params = task()
    keys = jax.random.split(jax.random.PRNGKey(0), 3 * L + 1)
    _, state = env.reset(keys[0], params)
    returns: list[float] = []
    total = 0.0
    for key in keys[1:]:
        _, state, reward, term, trunc, _ = env.step(key, state, jnp.zeros(1), params)
        total += float(reward)
        if term or trunc:
            returns, total = [*returns, total], 0.0
    assert returns == [2.0 * L] * 3, f"episode returns {returns}"


def test_q3_the_evaluation_env_is_the_task() -> None:
    """The evaluation env built from the task as every evaluation builds it:
    one episode in each of 4 envs returns 14 and terminates on step 7."""
    env, params = task()
    built, mode, _ = setup_environment(env, params, 4, norm_info=None, gamma=0.99)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    _, state = jax.vmap(built.reset, in_axes=(0, None))(keys, params)
    total, step = np.zeros(4), jax.vmap(built.step, (0, 0, None, None))
    for t in range(L):
        step_keys = jax.random.split(jax.random.fold_in(keys[0], t), 4)
        out = step(step_keys, state, jnp.zeros(1), params)
        state, total = out[1], total + np.asarray(out[2])
    assert mode == "gymnax" and np.all(out[3]) and (total == 2.0 * L).all(), total


def _q3_right(agent: str) -> float:
    return 32.0 if agent == "APG" else 2.0 * L


@pytest.mark.parametrize(
    "agent", [xparam(a, "" if a == "TDMPC2" else BYPASS) for a in Q3_AGENTS]
)
def test_q3_training_episodes_run_the_wrapped_task(agent: str) -> None:
    """Every training return the agent keeps and every logged Train mean is
    14 (APG: 32 per 16-step rollout)."""
    r, right = q3_run(agent), _q3_right(agent)
    mean = r[TRAIN][~np.isnan(r[TRAIN])]
    assert r["train"].size and (r["train"] == right).all() and (mean == right).all(), r


@pytest.mark.parametrize(
    "agent",
    [xparam(a, BYPASS if a == "APG" else "") for a in Q3_AGENTS if a != "UDRL"],
)
def test_q3_evaluation_runs_the_wrapped_task(agent: str) -> None:
    """Every logged evaluation returns 14 in 7 steps (APG evaluates its own
    16-step closed-loop rollout: 32, no length)."""
    r = q3_run(agent)
    assert r[EVAL].size and (r[EVAL] == _q3_right(agent)).all(), r
    assert (r[LENGTH] == L).all(), r


# --- Q4: SAC's expert warm-up, read row by row from the replay buffer --------
# A scalar counter (obs = the step k, 7 steps) and a counting expert (state =
# its calls since the episode started, action (-0.5, -0.5)) with the obs
# augmented by that state: row t of each env stores (k, k), k = t mod 7. The
# run is all warm-up: the expert on a share 0.7 of steps, else U[-1, 1)^2.


class CountingExpert:
    """``init_state(n)`` and ``(state, obs) -> (action, state + 1)``."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1), jnp.float32)

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return jnp.full((obs.shape[0], 2), -0.5, jnp.float32), state + 1.0


SCALAR = dataclasses.replace(counter(0, True, (0, 0)), obs=envs.index_obs, obs_dim=1)
SCALAR = dataclasses.replace(SCALAR, obs_box=(0.0, float(L)), box=(-1.0, 1.0, 2))
Q4_SAC: dict[str, Any] = {"n_envs": 2, "gamma": agents.GAMMA, "learning_starts": 10**9}
Q4_SAC |= {"actor_architecture": ("16", "relu"), "critic_architecture": ("16", "relu")}
Q4_SAC |= {"buffer_size": 32_064, "batch_size": 64, "expert_buffer_n_steps": 0}
Q4_SAC |= {"expert_mix_fraction": 0.0, "expert_fraction": 0.7}
Q4_SAC |= {"augment_obs_with_expert_state": True}
Q4_KEYS = ("obs", "action", "is_expert", "reward", "terminated", "truncated")
CELLS = [(e, d) for e in range(2) for d in range(2)]


def _q4_sac() -> Any:
    kw = Q4_SAC | {"expert_policy": CountingExpert()}
    return agents.make("SAC", *SCALAR.make(), preset=None, **kw)


def _q4_read(run: runs.Run) -> dict:
    """The buffer's rows (seeds, envs, rows, ...), the rolling mean return,
    and per seed the uniform rows' cell statistics and correlations."""
    r = R.replay_rows(run.state, Q4_KEYS)
    a, ex, seeds = r["action"], r["is_expert"][..., 0] > 0.5, range(len(run.seeds))
    r["mean return"] = np.asarray(run.state.collector_state.episodic_mean_return)
    for e, d in CELLS:
        q, cell = f"a[env {e}, dim {d}]", [a[s, e, ~ex[s, e], d] for s in seeds]
        r[f"mean {q}"] = np.array([c.mean() for c in cell])
        r[f"share {q} < 0.4"] = np.array([(c < 0.4).mean() for c in cell])
    both, r["expert share"] = ~ex[:, 0] & ~ex[:, 1], ex.mean((1, 2))
    for i in range(2):
        dims = [np.corrcoef(a[s, i, ~ex[s, i]].T)[0, 1] for s in seeds]
        across = [np.corrcoef(a[s, :, both[s], i].T)[0, 1] for s in seeds]
        r[f"corr(dim 0, dim 1) in env {i}"] = np.array(dims)
        r[f"corr(env 0, env 1) in dim {i}"] = np.array(across)
    return r


Q4_READ, Q4_BUDGET = runs.readings(_q4_sac, _q4_read), 32_000
SAME = "the decision's draw reused for this cell (today: env 0, dim 0)"
Q4_UNIFORM: list[Query] = []
for _e, _d in CELLS:
    Q4_UNIFORM += [Query(f"mean a[env {_e}, dim {_d}]", 0.0, {SAME: 0.7})]
    Q4_UNIFORM += [Query(f"share a[env {_e}, dim {_d}] < 0.4", 0.7, {SAME: 0.0})]
CASES["q4-SAC-uniform"] = Case(
    "q4-SAC-uniform",
    tuple(Q4_UNIFORM),
    Q4_READ,
    Q4_BUDGET,
    defect="one mix_key draws both the warm-up decision and the uniform action (agents/SAC/action_pipeline.py); right env 0 dim 0 mean 0 and share below 0.4 = 0.7, today 0.70 and 0.0",
)
ONE, SHARE_WRONG = {"one draw shared by the two cells": 1.0}, {"expert never used": 0.0}
CASES["q4-SAC-independence"] = Case(
    "q4-SAC-independence",
    (
        Query("expert share", 0.7, SHARE_WRONG | {"decision inverted": 1 - 0.7}),
        *(Query(f"corr(dim 0, dim 1) in env {e}", 0.0, ONE) for e in range(2)),
        *(Query(f"corr(env 0, env 1) in dim {d}", 0.0, ONE) for d in range(2)),
    ),
    Q4_READ,
    Q4_BUDGET,
    (0.02, 0.077, 0.074, 0.063, 0.068),
    note="ladder 2000-32000: 32000 the first rung within 0.1; cert 32/32",
)


def test_q4_rows_store_the_step_and_the_expert_state_before_the_call() -> None:
    """Rows inside episodes store (k, k) (the run's first row (0, 0)); reward
    1, terminated on k = 6 only, never truncated, return 7; the expert flag
    marks exactly the rows acting (-0.5, -0.5), the others in [-1, 1)."""
    r = Q4_READ(STAGE_1, Q4_BUDGET)
    t = np.arange(r["obs"].shape[2])
    inside, k = (t % L > 0) | (t == 0), (t % L).astype(np.float32)
    bad = (r["obs"][:, :, inside] != np.stack([k, k], -1)[inside]).any(-1)
    assert not bad.any(), f"{bad.sum()} rows inside episodes do not store (k, k)"
    ex = r["is_expert"][..., 0] > 0.5
    uniform = r["action"][~ex]
    np.testing.assert_array_equal(r["reward"][..., 0], 1.0)
    np.testing.assert_array_equal(r["truncated"][..., 0], 0.0)
    assert (r["terminated"][..., 0] == (k == 6)).all()
    np.testing.assert_array_equal(r["mean return"], float(L))
    np.testing.assert_array_equal(r["action"][ex], -0.5)
    assert ((uniform >= -1) & (uniform < 1)).all()
    assert not (uniform == -0.5).all(-1).any()


@xfail(
    "on a done step the next last_obs is the pre-reset final obs concatenated with the un-reset post-step expert state (environments/interaction.py:874, 933-941, 970-971); right (0, 0) on every episode-start row, today (7, 7)"
)
def test_q4_episode_start_rows_store_the_reset_obs_and_reset_expert_state() -> None:
    """The first row of every later episode stores (0, 0): the reset obs and
    the expert's reset state."""
    r = Q4_READ(STAGE_1, Q4_BUDGET)
    t = np.arange(r["obs"].shape[2])
    start = np.unique(r["obs"][:, :, (t % L == 0) & (t > 0)].reshape(-1, 2), axis=0)
    assert start.tolist() == [[0.0, 0.0]], f"episode-start rows store {start}"


# --- Q5: the actions that reach an env declaring Box(0, 2) -------------------
# The counter declaring Box(0, 2) (asymmetric: [-1, 1] and the box differ),
# its first 64 executed actions recorded: SAC's and TD3's uniform warm-up,
# PPO's first rollouts at sigma 1. Which contract is right (map to the box,
# reject it, keep [-1, 1]) is open: only what holds under all three is checked.
# SAC and TD3 read exactly 0 on 96 seeds (both cells draw the same warm-up).


def _paid(st: envs.State, a: jax.Array, key: Any) -> tuple:
    return st.replace(s=st.s + 1), a.reshape(-1)[0], st.s + 1 >= L


def q5_run(agent: str, seeds: tuple, budget: int, paid: bool = False, **kw: Any) -> Any:
    """``paid``: the reward is the executed action. PPO at log_std_init 0."""
    box = dataclasses.replace(counter(0, True, (64, 0)), box=(0.0, 2.0, 1))
    spec = dataclasses.replace(box, transition=_paid) if paid else box
    kw = ({"log_std_init": 0.0} if agent == "PPO" else {}) | kw
    model = agents.make(agent, *spec.make(), preset="bookkeeping", **kw)
    run = runs.train(model, seeds, budget)
    if (R.field(run.state, "clock") < 64).any():
        raise RuntimeError("the 64-step window is not filled")
    return run, R.record(run.state, "first_a")[:, 0]


W = {"ClipAction(-1, 1) only under a normalisation flag": 0.3173}
W |= {"collector maps to the box, training clip stays +-1": 0.5}
UNIT = Query("share outside [-1, 1], cell A minus cell B", 0.0, W)
W = {"training clip made bounds-aware, collector unmapped": 0.5}
BOX = Query("share outside [0, 2], cell A minus cell B", 0.0, W)
CLIPPED = "ClipAction(-1, 1) wraps the env only when a normalisation flag is set (environments/create.py:375-388) and PPO is unsquashed (PPO.py:78); right the same share of executed actions outside [-1, 1] with and without normalize_rewards (A - B = 0 +- 0.1), today A ~0.32 and B 0"


def _outside(x: np.ndarray) -> dict:
    return {UNIT.name: (abs(x) > 1).mean(1), BOX.name: ((x < 0) | (x > 2)).mean(1)}


def _q5_flag(agent: str) -> Callable:
    """Cell A (no flag) minus cell B (normalize_rewards), per share."""

    @functools.cache
    def read(seeds: tuple, budget: int) -> dict:
        cell = functools.partial(q5_run, agent, seeds, budget)
        a, b = (_outside(cell(normalize_rewards=f)[1]) for f in (False, True))
        return {k: a[k] - b[k] for k in a}

    return read


for _a in ("PPO", "SAC", "TD3"):
    _tol, _why = ((), CLIPPED) if _a == "PPO" else ((0.02, 0.02), "")
    CASES[f"q5-{_a}"] = Case(f"q5-{_a}", (UNIT, BOX), _q5_flag(_a), 64, _tol, _why)


def test_q5_ppo_trains_and_evaluates_the_same_action() -> None:
    """A constant PPO policy sending 1.5 (in the box, outside [-1, 1]) has
    the same action executed in training and in evaluate() (the env pays it)."""
    kw = {"log_std_init": -20.0, "mean_kernel_init": "0.0", "actor_bias_init": "1.5"}
    run, trained = q5_run("PPO", (0, 1), 32, True, actor_learning_rate=0.0, **kw)
    start, s, args = envs.one_hot(0, L + 1), run.state, run.agent.env_args
    sent = R.per_seed(lambda n: {"a": R.action(n, start, False)}, R.nets(s))["a"]
    if not np.allclose(sent, 1.5, atol=1e-6) or np.ptp(trained, 1).any():
        raise RuntimeError(f"the planted policy sends {sent}, not a constant 1.5")
    evaluated, rng = [], s.eval_rng
    for i in range(2):
        actor = jax.tree.map(lambda x, i=i: x[i], s.actor_state)
        out = evaluate(args.env, actor, 2, rng[i], args.env_params, gamma=agents.GAMMA)
        evaluated.append(float(np.asarray(out[0]).mean() / np.asarray(out[4])))
    np.testing.assert_allclose(evaluated, trained.mean(1), atol=1e-6)


# --- Q8: the likelihood-ratio identity of PPO and APO with a frozen actor ---
# Actor learning rate 0: the loss recomputes every stored log-prob, so at clip
# range 1e-4 every logged clip_fraction is exactly 0 (float noise stays under
# 1e-6) on 8 seeds and 10 logged rollouts; a collection/loss mismatch clips
# the share of samples it touches (0.15 to 1). The counter, 2-D Box or
# Discrete(2); the memory cells run sequence-pool minibatches.


@dataclasses.dataclass(frozen=True)
class ZeroActorLoss(Extension):
    """Adds 0 to the actor loss: PPO's actor loss then folds the stack's
    term (train_PPO.py, _stack_actor_loss)."""

    name: str = "zero_actor_loss"

    def actor_loss(self, agent_state: Any, ext_state: Any, batch: Any, ctx: Any) -> Any:
        return 0.0


WIDE = {"squash": True, "log_std_init": 1.0}  # std 2.72: an arctanh recompute shows
SPLIT = {"n_envs": 4, "num_minibatches": 2}
Q8_CELLS: dict[str, dict] = {"PPO-squash": WIDE, "PPO-gaussian": {}}
Q8_CELLS |= {"PPO-discrete": {"actions": 2}}
_MEMORY = {"actions": 2, "n_envs": 2, "num_minibatches": 2, "bptt_length": 8}
for _kind in ("gru", "lstm", "transformer", "mamba"):
    Q8_CELLS[f"PPO-{_kind}"] = _MEMORY | {"memory": MemoryConfig(_kind, 16)}
Q8_CELLS |= {"PPO-env-split": WIDE | SPLIT}
Q8_CELLS |= {"PPO-unroll": WIDE | SPLIT | {"unroll_length": 8}}
Q8_CELLS |= {"PPO-extension": WIDE | {"extensions": (ZeroActorLoss(),)}}
Q8_CELLS |= {"APO-discrete": {"actions": 2}, "APO-squash": {}}
Q8_CELLS |= {"APO-normalized": {"normalize_observations": True}}
Q8_CELLS |= {"PPO-normalized": WIDE | {"normalize_observations": True}}


@functools.cache
def q8_run(cell: str) -> runs.Run:
    """Ten rollouts of 32 steps per env, each logged."""
    kw, agent = dict(Q8_CELLS[cell]), cell.split("-")[0]
    two_d = dataclasses.replace(counter(0, True, (0, 0)), box=(-1.0, 1.0, 2))
    spec = counter(2, True, (0, 0)) if kw.pop("actions", 0) else two_d
    kw |= {"actor_learning_rate": 0.0, "clip_range": 1e-4}
    model = agents.make(agent, *spec.make(), preset="bookkeeping", **kw)
    per = kw.get("n_envs", 1) * 32
    return runs.train(model, STAGE_1, 10 * per, per)


NORMALIZED = pytest.mark.skip(
    reason="right answer awaits an owner decision: PPO's agent-side observation statistics move at every collection step (interaction.py:742-755) but the loss normalises with the end-of-rollout statistics (clip_fraction 0.56 in the first rollout, 0 from about the 13th)"
)
Q8_MARKS = {"PPO-squash": [], "PPO-normalized": [NORMALIZED]}


@pytest.mark.parametrize(
    "cell", [pytest.param(c, marks=Q8_MARKS.get(c, pytest.mark.slow)) for c in Q8_CELLS]
)
def test_q8_ratio_is_one_with_a_frozen_actor(cell: str) -> None:
    """Per seed 10 logged events, every logged clip_fraction exactly 0 and
    every logged policy loss and log-prob finite."""
    run = q8_run(cell)
    events = (run.timesteps() >= 0).sum(1)
    keys = ("clip_fraction", "policy_loss", "log_probs")
    clip, loss, log_p = (run.values(f"policy/{k}") for k in keys)
    assert (events == 10).all() and (clip == 0).all(), (events, np.unique(clip))
    assert np.isfinite(loss).all() and np.isfinite(log_p).all(), (loss, log_p)


# --- every judged case --------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
