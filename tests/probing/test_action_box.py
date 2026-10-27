"""Action box: a trained policy acts at either edge of, and inside, a box
that overhangs [-1, 1] unequally (``envs.action_box``: Box(-2, 3), contexts
paying a, -a and -(a - 2)^2). Each cell reads the deterministic action at
each context, in env units, and judges its centre, its reach and the inside
action against the agent family's oracle; the last test judges every case.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3 import networks as dreamer
from ajax.agents.TDMPC2.planner import plan_for_envs
from ajax.environments.utils import agent_action_to_env

from . import agents, envs, oracles, runs
from . import readouts as R
from .verdict import Case, Query, check, params

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "509c57f7da0c"  # verdict.digest(CASES): every answer, pinned
ENV, PARAMS = envs.action_box().make()
CONTEXTS = jnp.eye(3)  # up, down, inside

# --- the answers ---------------------------------------------------------------
# In agent units a = (a_env - 0.5) / 2.5 the contexts pay 2.5 a, -2.5 a and
# -(2.5 a - 1.5)^2, up to constants. Without an entropy term the optimum is
# the box's edges and a_env = 2: reach 2.5, inside 2. A max-entropy actor
# settles where E[r] + alpha H peaks (oracles.maxent_policy): SAC, REDQ and
# ASAC at the one temperature whose mean entropy over the three contexts is
# their target, -1 nat (alpha 0.281: reach 2.327, inside 2.027); AVG at its
# fixed alpha, 0.07 (2.457, 2.011). Wrong answers: a consumer clamping the
# actions to env [-1, 1] (centre 0, reach 1); a critic fed actions in other
# units than it was trained on (inside 0.6 or near 3). Actions never change
# what follows, so target-side faults (smoothing, next actions) only shift
# values by a constant: no cell here can see them.

PAY = (lambda a: 2.5 * a, lambda a: -2.5 * a, lambda a: -((2.5 * a - 1.5) ** 2))
CENTRE, REACH = "centre (a_up + a_down) / 2", "reach (a_up - a_down) / 2"
INSIDE, CLAMP = "a_inside", "actions clamped to env [-1, 1]"
ALPHA = {"none": None, "auto": oracles.shared_temperature(PAY, -1.0), "AVG": 0.07}
ALPHA |= {"ASAC": 0.2}  # ASAC fixes its temperature by default


def _queries(alpha: float | None) -> tuple[Query, ...]:
    def env(reward: Callable, best: float) -> float:  # its optimum, env units
        a = best if alpha is None else oracles.maxent_policy(reward, alpha)[0]
        return 0.5 + 2.5 * a

    wrong = {"env units read as agent units (a critic fed mapped actions)": 0.04}
    wrong |= {"agent units read as env units (unmapped, or the critic's)": 1.0}
    pay = (lambda a: -((6.25 * a - 0.25) ** 2), lambda a: -((a - 2.0) ** 2))
    wrong = {k: env(r, b) for (k, b), r in zip(wrong.items(), pay, strict=True)}
    return (
        Query(CENTRE, 0.5, {CLAMP: 0.0}),
        Query(REACH, env(PAY[0], 1.0) - 0.5, {CLAMP: 1.0}),
        Query(INSIDE, env(PAY[2], 0.6), wrong),
    )


# --- the readouts ----------------------------------------------------------------


def _judged(a: Any) -> dict[str, np.ndarray]:
    """Centre, reach and inside of (seeds, 3) env-unit actions."""
    up, down, inside = np.asarray(a).T
    return {CENTRE: (up + down) / 2, REACH: (up - down) / 2, INSIDE: inside}


def _to_env(a: jax.Array) -> jax.Array:
    """(n, 1) agent actions as every Ajax env step executes them."""
    return agent_action_to_env(a.reshape(-1, 1), ENV, PARAMS)[:, 0]


def _actor(run: runs.Run) -> dict:
    """The evaluation's action: the policy's mean (``pi.mean()``)."""
    n = R.nets(run.state, stats=R.env_stats(run))
    act = jax.vmap(
        lambda n: _to_env(jax.vmap(lambda x: R.action(n, x, False))(CONTEXTS))
    )
    return _judged(act(n))


def _tdmpc2(run: runs.Run) -> dict:
    """The evaluation planner's first decision (eval_mode, a zero warm
    start), averaged over 16 envs of each context: it picks an elite at
    random."""
    agent, s, obs = run.agent, run.state, jnp.repeat(CONTEXTS, 16, 0)
    kw = {"config": agent.model_config, "gamma": agent.gamma, "eval_mode": True}

    def one(wm: Any, pi: Any, key: jax.Array) -> jax.Array:
        carry = jnp.zeros((48, agent.model_config.horizon, 1))
        first = jnp.ones(48, bool)
        a = plan_for_envs(carry, obs, first, key, wm_params=wm, pi_params=pi, **kw)
        return _to_env(a[0]).reshape(3, 16).mean(1)

    keys = jax.random.split(R.KEY, len(run.seeds))
    wm, pi = s.world_model_state.params, s.actor_state.params
    return _judged(jax.vmap(one)(wm, pi, keys))


def _dreamer(run: runs.Run) -> dict:
    """The actor's mode after one observe step from a fresh carry."""
    c, s = run.agent.model_config, run.state

    def one(wm: Any, pi: Any) -> jax.Array:
        model, rssm = dreamer.WorldModel(c, 3), dreamer.RSSM(c)
        token = model.apply({"params": wm}, CONTEXTS, method="encode")
        first, noise = jnp.ones(3, bool), jnp.zeros((3, c.stoch, c.classes))
        step = (dreamer.initial_state(c, (3,)), token, jnp.zeros((3, 1)), first, noise)
        h = rssm.apply({"params": wm["rssm"]}, *step, method="observe_step")[0]
        policy = dreamer.Actor(c, 1, False).apply({"params": pi}, dreamer.features(*h))
        return _to_env(policy.mean)  # type: ignore[union-attr]

    return _judged(jax.vmap(one)(s.world_model_state.params, s.actor_state.params))


# --- the cells -------------------------------------------------------------------
# Every reading is in env units, the reach a half-width like the centre, so
# one ceiling, 0.25 env units, is the framework's 0.1 in agent units of 2.5.
# Overrides: learning rates 3e-4, and 1e-4 for APO and ASAC, where the
# presets' 1e-3 (AVG: its 6.3e-3 and 8.7e-3) jitter the inside action past
# it (ASAC 0.277 at 20,000 steps; AVG 0.289 on 16 seeds at 40,000; APO, even
# at 3e-4: NOTES); REDQ 5 critic updates a step, as P8; APG one step a
# rollout, as an action pays only its own step. PPO starts at sigma e^-1: at
# its default 1 the clipped Gaussian's best inside mean lies past the box's
# edge, and once sigma has shrunk every sample clips there, so nothing brings
# it back. TD-MPC2 takes terminations in its episodic mode and plans one
# step: a training slice (horizon + 1 rows) must fit in a one-step episode;
# at its horizon 3 (time limit 4) 16,000 rows left its readings where they
# started. The world models read through their own policies, on the tiny
# preset. DreamerV3's inside action has no known answer here: its samples are
# not squashed but clipped at the box, so while its std stays wide the best
# mean leans to the edge. Its edges do.
LR3 = {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4}
CELLS: dict[str, tuple] = {  # agent: family, budget, tolerances, overrides
    "SAC": ("auto", 10_000, (0.055, 0.051, 0.218), {}),
    "TD3": ("none", 20_000, (0.02, 0.02, 0.164), {}),
    "PPO": ("none", 25_000, (0.02, 0.02, 0.219), {"log_std_init": -1.0}),
    "REDQ": ("auto", 10_000, (0.057, 0.052, 0.232), {"num_critic_updates": 5}),
    "ASAC": ("ASAC", 20_000, (0.043, 0.074, 0.215), dict.fromkeys(LR3, 1e-4)),
    "AVG": ("AVG", 20_000, (0.026, 0.045, 0.239), LR3),
    "APO": ("none", 300_000, (0.02, 0.02, 0.145), dict.fromkeys(LR3, 1e-4)),
    "APG": ("none", 5000, (0.02, 0.02, 0.065), {"horizon": 1}),
    "DreamerV3": ("none", 32_000, (0.074, 0.076), {}),
    "TDMPC2": ("none", 4000, (0.02, 0.02, 0.248), {"episodic": True, "horizon": 1}),
}
NOTES = {  # beyond choose's pick: owner decisions, rungs ruled out and why
    "SAC": "5,000 too few: twice its worst reach error, 0.332, passed 0.25",
    "PPO": "owner: at PPO's reference log-std 0, 2 of the 64 ladder seeds (1018,"
    " 2007) end the inside mean past the box's edge, 3.0 from 25,000 to 100,000"
    " steps; that is PPO's behaviour, not a defect, so the cell starts at -1",
    "REDQ": "at its reference init 5,000 too few: twice its worst inside error 0.257",
    "AVG": "re-laddered at its reference initialisation on 20,000 and 40,000 only",
    "APO": "at 3e-4 100,000 qualifies but certifies 28/32, and the inside action"
    " drifts past it (0.60-1.68 at 150,000-300,000); at 1e-4, 200,000: 0.293",
    "DreamerV3": "owner: judged on the edges only; its inside action, 1.67 to"
    " 3.0 over 8 seeds at 48,000 steps, is backlogged. choose picks 16,000, but"
    " a seed's readings depend on the seeds trained with it: 1000-1007 trained"
    " 8 together, as the check trains them, read centre 0.261 on seed 1002"
    " (0.486 in the 32-seed ladder), twice its error 0.479 past 0.25; at 32,000"
    " the same 8 read twice the worst error 0.019",
}
READ, EDGES_ONLY = {"DreamerV3": _dreamer, "TDMPC2": _tdmpc2}, ("DreamerV3",)
for agent, (family, budget, tol, kw) in CELLS.items():
    spec = dataclasses.replace(envs.action_box(), gradients=agent == "APG")
    preset = "tiny" if agent in READ else "probe"
    build = functools.partial(agents.make, agent, *spec.make(), preset=preset, **kw)
    reads = runs.readings(build, READ.get(agent, _actor))
    queries = _queries(ALPHA[family])[: 2 if agent in EDGES_ONLY else 3]
    note = f"{NOTES[agent]}; cert 32/32" if agent in NOTES else "cert 32/32"
    CASES[f"box-{agent}"] = Case(
        f"box-{agent}", queries, reads, budget, tol, slow=True, ceiling=0.25, note=note
    )


def test_the_box_pays_by_context_in_env_units() -> None:
    """Up pays a, down -a, inside -(a - 2)^2, a clipped to [-2, 3]; every
    step terminates; 1024 resets draw each context about a third of the time."""
    keys = jax.random.split(R.KEY, 1024)
    obs, st = jax.vmap(ENV.reset, (0, None))(keys, PARAMS)
    context = np.asarray(obs.argmax(-1))
    step = jax.vmap(ENV.step, (0, 0, None, None))
    for a in (-3.0, -2.0, 0.5, 2.0, 3.0, 4.0):
        _, _, r, term, trunc, _ = step(keys, st, jnp.array([a]), PARAMS)
        x = min(max(a, -2.0), 3.0)
        np.testing.assert_allclose(r, np.array([x, -x, -((x - 2) ** 2)])[context])
        assert np.asarray(term).all() and not np.asarray(trunc).any()
    assert (np.abs(np.bincount(context, minlength=3) / 1024 - 1 / 3) < 0.05).all()


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
