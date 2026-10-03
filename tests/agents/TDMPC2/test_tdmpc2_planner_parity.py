"""TD-MPC2 planner parity against the real paper-era reference code.

The fixture ``fixtures/tdmpc2_plan.npz`` was recorded by running the
unmodified ``TDMPC2.act()`` (MPPI planning) of ``nicklashansen/tdmpc2@5f6fade``
for five consecutive decisions (``t0`` training, two training warm starts, an
evaluation-mode warm start, then ``t0`` training at an episode boundary, where
the agent still holds a non-zero warm start) with the paper's planning
hyperparameters (512 samples, 24 policy trajectories, 64 elites, 6
iterations), recording every random draw and, per iteration, the candidate
values, elites, scores, mean and std
(``docs/world_models/parity/tdmpc2_plan_fixtures.py``, whose docstring explains
the configuration and how to regenerate it). Its reward, Q and policy heads are
random, and every decision has policy trajectories among the elites of some
iteration, so their part in the MPPI update is compared too.

The reference's parameters are mapped onto Ajax's trees
(:mod:`.reference_params`, shared with the update parity test) and Ajax's
jitted :func:`ajax.agents.TDMPC2.planner.plan` is chained over the same
observations with the recorded draws, its own returned mean feeding the next
decision's warm start. Two further tests feed Ajax the reference's own
intermediate quantities one iteration at a time: the value estimate on the
reference's candidates, and the MPPI update and elite draw on the reference's
values.

Expected float32 differences (achieved on CPU with ``highest`` matmul
precision; print them with ``-s``): candidate values within 1.3e-5 (their
two-hot decode differs from the reference's at float32 rounding level,
deviation T24, which ``symexp`` scales by ``1 + |V|``); policy-trajectory
actions within 2.6e-6 (the prior sample ``tanh(mu + eps exp(log_std))``
scales the rounding of the policy head by ``|eps| exp(log_std)``, up to ~20);
actions, means, stds and the returned mean within 3e-7; scores within 6e-8.
The fixture keeps only decisions whose top-k boundary and elite draw are at
least ``2e-4 * max(1, |V|)`` from a tie (15x the value error), so the elite
sets and the drawn elite are compared exactly. The order of elites with nearly
tied values could differ across platforms without changing any output (the
mean, std and draw are invariant to it), so elite sets are compared, not their
order; the teacher-forced test, whose inputs are identical, compares the order
too.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2 import planner
from ajax.agents.TDMPC2.state import TDMPC2Config

from .reference_params import (
    ErrorReport,
    Fixture,
    ajax_config,
    as_jnp,
    load_fixture,
    reference_config,
    torch_to_ajax,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tdmpc2_plan.npz"
# (rtol, atol), a few times the errors of the module docstring, which may
# differ across platforms (macOS ARM / Linux x86). Values are O(1) to O(10);
# the fixture's tie margins, 2e-4 * max(1, |V|), stay 3x or more above this
# tolerance, so the elite sets can be compared exactly.
VALUE_TOL = (2e-5, 4e-5)
# Actions, means, stds in [-1, 1] / [0.05, 2].
ACTION_TOL = (0.0, 5e-6)
# Policy-prior samples: rounding amplified by |eps| exp(log_std) (docstring).
PI_ACTION_TOL = (0.0, 2e-5)
SCORE_TOL = (0.0, 1e-6)
PLANNING_FIELDS = (
    "iterations",
    "num_samples",
    "num_elites",
    "num_pi_trajs",
    "min_std",
    "max_std",
    "temperature",
)


@pytest.fixture(scope="module")
def fx() -> Fixture:
    return load_fixture(FIXTURE)


@pytest.fixture(scope="module")
def config(fx: Fixture) -> TDMPC2Config:
    return ajax_config(reference_config(fx))


@pytest.fixture(scope="module")
def gamma(fx: Fixture) -> float:
    """The reference's Python-float discount (float64 powers in planning)."""
    return float(fx["meta/discount"])


@pytest.fixture(scope="module")
def params(fx: Fixture, config: TDMPC2Config) -> tuple[Any, Any]:
    """World-model and policy parameters of the recorded decisions: the
    reference's initialisation with the fixture's random reward, Q and policy
    heads."""
    snapshot = dict(fx)
    for key, value in fx.items():
        if key.startswith("heads/"):
            snapshot["init/" + key[len("heads/") :]] = value
    wm, _, pi = torch_to_ajax(snapshot, "init/", config)
    return as_jnp(wm), as_jnp(pi)


def n_decisions(fx: Fixture) -> int:
    return int(fx["meta/n_decisions"])


def recorded_noise(fx: Fixture, d: int) -> planner.PlanNoise:
    """Decision ``d``'s draws (dropout is 0 in the fixture: the keys are unused)."""
    p = f"decision{d}/draws/"
    iterations = fx[p + "q_pair"].shape[0]
    return planner.PlanNoise(
        pi_eps=jnp.asarray(fx[p + "pi_eps"]),
        candidate_eps=jnp.asarray(fx[p + "candidate_eps"]),
        terminal_eps=jnp.asarray(fx[p + "terminal_eps"]),
        q_pair=jnp.asarray(fx[p + "q_pair"]),
        q_dropout=jax.random.split(jax.random.PRNGKey(0), iterations),
        elite_uniform=jnp.asarray(fx[p + "elite_uniform"], jnp.float32),
        action_eps=jnp.asarray(fx[p + "action_eps"]),
    )


def reference_candidates(fx: Fixture, d: int, i: int) -> np.ndarray:
    """Iteration ``i``'s candidates ``[H, N, A]`` from the reference's mean / std."""
    p = f"decision{d}/"
    mean = fx[p + "init_mean"] if i == 0 else fx[p + "mean"][i - 1]
    std = fx[p + "init_std"] if i == 0 else fx[p + "std"][i - 1]
    eps = fx[p + "draws/candidate_eps"][i]
    sampled = np.clip(mean[:, None] + std[:, None] * eps, -1.0, 1.0)
    return np.concatenate([fx[p + "pi_actions"], sampled], axis=1)


@functools.lru_cache(maxsize=None)
def jitted_plan(config: TDMPC2Config, gamma: float, eval_mode: bool) -> Any:
    return jax.jit(
        functools.partial(planner.plan, config=config, gamma=gamma, eval_mode=eval_mode)
    )


def test_fixture_metadata(fx, config, gamma):
    """5f6fade, Q dropout ignoring eval() (planning runs in eval mode), the
    paper's planning defaults, the float64 discount, and the coverage the
    decisions were chosen for (a regenerated fixture must keep it)."""
    assert str(fx["meta/reference_commit"]).startswith("5f6fade")
    assert bool(fx["meta/q_dropout_active_in_eval_mode"])
    paper = json.loads(str(fx["meta/paper_config"]))
    default = TDMPC2Config()
    for name in PLANNING_FIELDS:
        assert getattr(config, name) == paper[name] == getattr(default, name), name
    assert fx["meta/discount"].dtype == np.float64 and gamma == 0.9
    decisions = [f"decision{d}/" for d in range(n_decisions(fx))]
    t0 = [bool(fx[p + "t0"]) for p in decisions]
    evals = [bool(fx[p + "eval_mode"]) for p in decisions]
    assert t0[0] and not all(t0) and not evals[0] and any(evals)
    # t0 discards a real (non-zero) warm start somewhere, not only the
    # zeros of a fresh agent.
    assert any(
        t0[d] and np.any(fx[p + "prev_mean_in"] != 0) for d, p in enumerate(decisions)
    )
    # Policy trajectories are elites in some iteration of every decision.
    for p in decisions:
        pi_elites = (fx[p + "elite_idx"] < config.num_pi_trajs).sum(axis=1)
        np.testing.assert_array_equal(pi_elites, fx[p + "pi_elites"])
        assert pi_elites.any(), p


def test_plan_matches_reference_decisions(fx, config, gamma, params):
    """Ajax's jitted plan, chained over the fixture's decisions."""
    wm, pi = params
    report = ErrorReport("TD-MPC2 plan parity (chained decisions)")
    prev_mean = jnp.zeros((config.horizon, int(fx["decision0/action"].shape[0])))
    with jax.default_matmul_precision("highest"):
        for d in range(n_decisions(fx)):
            p = f"decision{d}/"
            if d > 0:  # the reference's warm start is its previous output
                np.testing.assert_array_equal(
                    fx[p + "prev_mean_in"], fx[f"decision{d - 1}/prev_mean"]
                )
            plan = jitted_plan(config, gamma, bool(fx[p + "eval_mode"]))
            action, prev_mean, info = plan(
                wm,
                pi,
                jnp.asarray(fx[p + "obs"]),
                prev_mean,
                jnp.asarray(fx[p + "t0"]),
                recorded_noise(fx, d),
            )
            report.check("init_mean", info.init_mean, fx[p + "init_mean"], ACTION_TOL)
            report.check(
                "pi_actions", info.pi_actions, fx[p + "pi_actions"], PI_ACTION_TOL
            )
            report.check("value", info.value, fx[p + "value"], VALUE_TOL)
            np.testing.assert_array_equal(
                np.sort(info.elite_idx, axis=-1), np.sort(fx[p + "elite_idx"], axis=-1)
            )
            report.check(
                "elite_value", info.elite_value, fx[p + "elite_value"], VALUE_TOL
            )
            report.check(
                "score",
                -np.sort(-np.asarray(info.score), axis=-1),
                -np.sort(-fx[p + "score"], axis=-1),
                SCORE_TOL,
            )
            report.check("mean", info.mean, fx[p + "mean"], ACTION_TOL)
            report.check("std", info.std, fx[p + "std"], ACTION_TOL)
            rank, ref_rank = int(info.elite_rank), int(fx[p + "elite_rank"])
            assert info.elite_idx[-1, rank] == fx[p + "elite_idx"][-1, ref_rank]
            report.check("action", action, fx[p + "action"], ACTION_TOL)
            report.check("prev_mean", prev_mean, fx[p + "prev_mean"], ACTION_TOL)
    report.print()


def test_value_estimate_on_reference_candidates(fx, config, gamma, params):
    """``estimate_value`` on the reference's own candidates of every iteration."""
    wm, pi = params
    report = ErrorReport("TD-MPC2 value estimate parity (reference candidates)")
    estimate = jax.jit(
        functools.partial(planner.estimate_value, config=config, gamma=gamma)
    )
    with jax.default_matmul_precision("highest"):
        for d in range(n_decisions(fx)):
            p = f"decision{d}/"
            noise = recorded_noise(fx, d)
            z = jnp.broadcast_to(
                jnp.asarray(fx[p + "z"]), (config.num_samples, config.latent_dim)
            )
            for i in range(fx[p + "value"].shape[0]):
                value = estimate(
                    wm,
                    pi,
                    z,
                    jnp.asarray(reference_candidates(fx, d, i)),
                    noise.terminal_eps[i],
                    noise.q_pair[i],
                    noise.q_dropout[i],
                )
                report.check("value", value, fx[p + "value"][i], VALUE_TOL)
    report.print()


def test_mppi_update_and_elite_draw_on_reference_values(fx, config):
    """``mppi_step`` and ``sample_elite`` fed the reference's values: identical
    inputs, so the elite order and the drawn rank are exact."""
    step = jax.jit(functools.partial(planner.mppi_step, config=config))
    for d in range(n_decisions(fx)):
        p = f"decision{d}/"
        for i in range(fx[p + "value"].shape[0]):
            out = step(
                jnp.asarray(fx[p + "value"][i]),
                jnp.asarray(reference_candidates(fx, d, i)),
            )
            np.testing.assert_array_equal(out.elite_idx, fx[p + "elite_idx"][i])
            np.testing.assert_array_equal(out.elite_value, fx[p + "elite_value"][i])
            np.testing.assert_allclose(out.score, fx[p + "score"][i], atol=1e-7)
            np.testing.assert_allclose(out.mean, fx[p + "mean"][i], atol=1e-6)
            np.testing.assert_allclose(out.std, fx[p + "std"][i], atol=1e-6)
        rank = planner.sample_elite(
            jnp.asarray(fx[p + "score"][-1]), jnp.asarray(fx[p + "draws/elite_uniform"])
        )
        assert int(rank) == int(fx[p + "elite_rank"])


def test_tie_order_matches_torch_at_initialisation(fx, config, gamma):
    """With the paper's zero-initialised reward and Q heads every candidate is
    tied (spec 3.23). ``lax.top_k`` keeps the first indices in order, as the
    reference's ``torch.topk`` did on the same tie (recorded), so the step-0
    plan of an untrained model elects the same 64 candidates: the 24 policy
    trajectories and the first 40 Gaussian samples."""
    tied = jnp.full((config.num_samples,), 0.37)
    _, idx = jax.lax.top_k(tied, config.num_elites)
    np.testing.assert_array_equal(idx, fx["ties/topk_all_tied"])

    wm, _, pi = (as_jnp(t) for t in torch_to_ajax(fx, "init/", config))
    obs_dim, action_dim = fx["decision0/obs"].shape[0], fx["decision0/action"].shape[0]
    noise = planner.draw_plan_noise(jax.random.PRNGKey(3), config, action_dim)
    _, _, info = jitted_plan(config, gamma, True)(
        wm,
        pi,
        jnp.zeros(obs_dim),
        jnp.zeros((config.horizon, action_dim)),
        True,
        noise,
    )
    assert np.all(info.value == info.value[:, :1])
    np.testing.assert_array_equal(info.elite_idx, fx["ties/init_elite_idx"])
