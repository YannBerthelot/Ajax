"""TD-MPC2 multi-task parity against the real paper-era reference code.

The fixtures ``fixtures/tdmpc2_multitask_{update,plan}.npz`` were recorded by
running the unmodified ``nicklashansen/tdmpc2@5f6fade`` agent with
``cfg.multitask = True`` on three synthetic tasks of observation dims (5, 2,
7) and action dims (2, 4, 1), padded to 7 and 4, with distinct per-task
discounts (0.95, 0.975, 0.99) and ``task_dim = 11``, the size of no other
axis (``docs/world_models/parity/tdmpc2_multitask_fixtures.py``, whose
docstring explains the design and how to regenerate them):

* four consecutive ``update()`` calls on mixed-task batches, with the
  embedding table recorded on entry, after the pre-step renorm, after the
  world-model Adam step and on exit; the max-norm write-back binds at both
  points (a row of norm 1.6 at update 0's first look-up, a row of norm 1.3
  absent from update 0 and renormed by update 1's, and post-step renorms in
  every update), and task 0's row moves by Adam momentum only in update 2,
  whose batch lacks it;
* five ``act(obs, t0, eval_mode, task)`` decisions: task 2 (one valid
  action dim of four; its row of norm 1.4 renormed by the look-up), task 1
  (all four dims valid) and task 0 (two valid dims).

The reference's parameters (embedding table included) are mapped onto Ajax's
trees and Ajax's jitted :func:`ajax.agents.TDMPC2.multitask.update` and
:func:`ajax.agents.TDMPC2.multitask.plan` are chained over the same inputs
with the recorded draws. Every logged loss term, both gradient norms, the
RunningScale, the TD targets, the policy samples and log-probabilities, the
policy loss' Q values and the embedding table after every update are
compared, then all parameters after the last update; every planning
statistic and action of every decision. The single-task parity tests' float32
error budget applies (see ``test_tdmpc2_parity.py`` and
``test_tdmpc2_planner_parity.py``); the errors achieved on CPU are printed
with ``-s``.
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

from ajax.agents.TDMPC2 import core, multitask, planner
from ajax.agents.TDMPC2.core import TASK_EMB
from ajax.agents.TDMPC2.multitask import TaskSet
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
from .test_tdmpc2_parity import LOGGED, PARAM_TOL, POST_STEP_Q_TOL, SCALAR_TOL
from .test_tdmpc2_planner_parity import (
    ACTION_TOL,
    PI_ACTION_TOL,
    SCORE_TOL,
    VALUE_TOL,
)

FIXTURES = Path(__file__).parent / "fixtures"
N_UPDATES = 4
# The embedding table after each update: renorms and Adam steps of O(1)
# rows (achieved: ~1e-8).
TABLE_TOL = (0.0, 2e-6)
# All parameters after the four updates: the single-task PARAM_TOL
# (achieved on CPU: 2.3e-6).


@pytest.fixture(scope="module")
def ufx() -> Fixture:
    return load_fixture(FIXTURES / "tdmpc2_multitask_update.npz")


@pytest.fixture(scope="module")
def pfx() -> Fixture:
    return load_fixture(FIXTURES / "tdmpc2_multitask_plan.npz")


def task_set(fx: Fixture) -> TaskSet:
    ref = reference_config(fx)
    return TaskSet.create(
        fx["meta/obs_dims"].tolist(),
        fx["meta/action_dims"].tolist(),
        fx["meta/episode_lengths"].tolist(),
        names=[str(n) for n in fx["meta/tasks"]],
        discount_denom=ref["discount_denom"],
        discount_min=ref["discount_min"],
        discount_max=ref["discount_max"],
    )


def test_task_set_reproduces_the_reference_tables(ufx, pfx):
    """Prefix masks and the float32 per-task discounts of the reference
    (``world_model.py:21-23``, ``tdmpc2.py:32-34``), at the paper's discount
    heuristic; the padded dims; the paper's planning iterations for the
    padded action dim."""
    tasks = task_set(ufx)
    np.testing.assert_array_equal(tasks.action_masks, ufx["meta/action_masks"])
    np.testing.assert_array_equal(tasks.discount_table, ufx["meta/discounts"])
    np.testing.assert_array_equal(tasks.discount_table, pfx["meta/discounts"])
    assert tasks.discounts == pytest.approx((0.95, 0.975, 0.99))
    assert (tasks.obs_dim, tasks.action_dim) == (7, 4)
    ref = reference_config(ufx)
    paper = json.loads(str(ufx["meta/paper_config"]))
    assert ref["discount_min"] == paper["discount_min"]
    assert ref["discount_max"] == paper["discount_max"]
    assert paper["task_dim"] == multitask.PAPER_TASK_DIM
    # task_dim is the size of no other axis of the fixtures (no embedding
    # axis can be confused with another by shape): every integer of the
    # configuration (iterations, batch, widths, ...), the tasks' dims and
    # count and the planner's derived sizes.
    sizes = {
        abs(v)
        for k, v in ref.items()
        if k != "task_dim" and isinstance(v, int) and not isinstance(v, bool)
    }
    sizes |= {*tasks.obs_dims, *tasks.action_dims, tasks.num_tasks}
    sizes |= {ref["horizon"] + 1, ref["num_samples"] - ref["num_pi_trajs"]}
    assert ref["task_dim"] not in sizes
    assert reference_config(pfx)["task_dim"] == ref["task_dim"]
    config = ajax_config(ref)
    assert config.planning_iterations(tasks.action_dim) == ref["iterations"]


def test_fixture_exercises_the_renorm_at_both_points(ufx):
    """The coverage the update fixture was built for (a regenerated fixture
    must keep it), and Ajax's renorm against torch's ``embedding_renorm_``
    on the recorded tables, both points of every update."""
    pre = ufx["meta/pre_step_renormed_rows"]
    post = ufx["meta/post_step_renormed_rows"]
    assert pre[0] == 1 and pre[1] == 1 and post.sum() > 0
    # Task 2 is absent from update 0: its row (norm > 1) is not renormed.
    assert 2 not in ufx["update0/batch/task"]
    assert np.linalg.norm(ufx["update0/task_emb/pre"][2]) > 1.0
    for k in range(N_UPDATES):
        p = f"update{k}/"
        ids = jnp.asarray(ufx[p + "batch/task"], jnp.int32)
        for before, after in (("in", "pre"), ("adam", "out")):
            renormed = core.renorm_task_embedding(
                {TASK_EMB: jnp.asarray(ufx[p + "task_emb/" + before])}, ids
            )[TASK_EMB]
            np.testing.assert_allclose(
                renormed, ufx[p + "task_emb/" + after], rtol=0, atol=1e-7
            )
    # Task 0 is absent from update 2: Adam momentum alone moves its row.
    assert 0 not in ufx["update2/batch/task"]
    assert not np.array_equal(
        ufx["update2/task_emb/in"][0], ufx["update2/task_emb/out"][0]
    )


def initial_state(fx: Fixture, config: TDMPC2Config, tasks: TaskSet) -> Any:
    """Ajax's update state holding the reference's initial parameters."""
    ref = reference_config(fx)
    state = multitask.create_update_state(
        jax.random.PRNGKey(0),
        config,
        tasks,
        task_dim=ref["task_dim"],
        learning_rate=ref["lr"],
        enc_lr_scale=ref["enc_lr_scale"],
    )
    wm, target_q, pi = (as_jnp(t) for t in torch_to_ajax(fx, "init/", config))
    for mapped, ajax in (
        (wm, state.world_model_state.params),
        (target_q, state.world_model_state.target_params),
        (pi, state.actor_state.params),
    ):
        assert jax.tree_util.tree_structure(mapped) == jax.tree_util.tree_structure(
            ajax
        )
        assert jax.tree_util.tree_map(jnp.shape, mapped) == jax.tree_util.tree_map(
            jnp.shape, ajax
        )
    wm_state, pi_state = state.world_model_state, state.actor_state
    return state.replace(
        world_model_state=wm_state.replace(
            params=wm, target_params=target_q, opt_state=wm_state.tx.init(wm)
        ),
        actor_state=pi_state.replace(params=pi, opt_state=pi_state.tx.init(pi)),
    )


def batch_and_noise(
    fx: Fixture, k: int
) -> tuple[core.TDMPC2Batch, jax.Array, core.UpdateNoise]:
    """Update ``k``'s batch, task ids and recorded draws (dropout is 0)."""
    p = f"update{k}/"
    batch = core.TDMPC2Batch(
        obs=jnp.asarray(fx[p + "batch/obs"]),
        action=jnp.asarray(fx[p + "batch/action"]),
        reward=jnp.asarray(fx[p + "batch/reward"][..., 0]),
    )
    key = jax.random.PRNGKey(0)
    noise = core.UpdateNoise(
        td_eps=jnp.asarray(fx[p + "draws/td_eps"]),
        td_pair=jnp.asarray(fx[p + "draws/td_pair"]),
        td_dropout=key,
        value_dropout=key,
        pi_eps=jnp.asarray(fx[p + "draws/pi_eps"]),
        pi_pair=jnp.asarray(fx[p + "draws/pi_pair"]),
        pi_dropout=key,
    )
    return batch, jnp.asarray(fx[p + "batch/task"], jnp.int32), noise


def test_multitask_update_matches_reference_over_four_updates(ufx):
    config = ajax_config(reference_config(ufx))
    tasks = task_set(ufx)
    state = initial_state(ufx, config, tasks)
    wm_apply = state.world_model_state.apply_fn
    pi_apply = state.actor_state.apply_fn
    step = jax.jit(
        lambda s, b, t, n: multitask.update(s, b, t, n, config=config, tasks=tasks)
    )

    @jax.jit
    def pre_step(wm_params, target_q, pi_params, batch, task, noise):
        """The TD target and the rollout, on the pre-step renormed table."""
        ctx = tasks.context(task)
        pre = core.renorm_task_embedding(wm_params, task)
        td, next_z = core.td_target(
            wm_apply,
            pi_apply,
            pre,
            target_q,
            pi_params,
            batch.obs[1:],
            batch.reward,
            tasks.discount(task),
            noise.td_eps,
            noise.td_pair,
            None,
            config,
            ctx,
        )
        _, (_, zs) = core.world_model_loss(
            pre, wm_apply, batch, next_z, td, None, config, ctx
        )
        return td, zs

    @jax.jit
    def policy_side(post, pi_params, zs, task, noise):
        """The policy loss' samples and Q: pre-step latents, pre-update
        policy, the post-step (renormed) embedding and Q."""
        ctx = tasks.context(task)
        emb = core.task_embedding(post, ctx)
        sample = core.policy_sample(
            pi_apply, pi_params, zs, noise.pi_eps, config, emb, ctx.mask
        )
        logits = core.q_logits(wm_apply, post, zs, sample.action, task_emb=emb)
        q = core.reduce_q_pair(logits, noise.pi_pair, "avg", config.two_hot)
        return sample.action, sample.log_pi, q

    report = ErrorReport("TD-MPC2 multi-task update parity")
    with jax.default_matmul_precision("highest"):
        for k in range(N_UPDATES):
            p = f"update{k}/"
            batch, task, noise = batch_and_noise(ufx, k)
            wm, pi = state.world_model_state, state.actor_state
            report.check(
                "task_emb (entry)",
                wm.params[TASK_EMB],
                ufx[p + "task_emb/in"],
                TABLE_TOL,
            )
            td, zs = pre_step(
                wm.params, wm.target_params, pi.params, batch, task, noise
            )
            report.check("td_targets", td, ufx[p + "td_targets"][..., 0], SCALAR_TOL)

            state, aux = step(state, batch, task, noise)
            post = state.world_model_state.params
            report.check(
                "task_emb (exit)", post[TASK_EMB], ufx[p + "task_emb/out"], TABLE_TOL
            )
            action, log_pi, q = policy_side(post, pi.params, zs, task, noise)
            report.check("pi_actions", action, ufx[p + "pi_actions"], SCALAR_TOL)
            report.check(
                "pi_log_pis", log_pi, ufx[p + "pi_log_pis"][..., 0], SCALAR_TOL
            )
            invalid = np.asarray(tasks.context(task).mask) == 0
            assert np.all(np.asarray(action)[:, invalid] == 0)
            report.check("pi_q", q, ufx[p + "pi_q"][..., 0], POST_STEP_Q_TOL)
            for name in LOGGED:
                report.check(name, aux[name], ufx[p + name], SCALAR_TOL)
            report.check(
                "q_scale", state.q_scale.value, ufx[p + "pi_scale"], SCALAR_TOL
            )

    wm_ref, target_ref, pi_ref = torch_to_ajax(ufx, "final/", config)
    report.check("wm_params", state.world_model_state.params, wm_ref, PARAM_TOL)
    report.check(
        "target_q_params", state.world_model_state.target_params, target_ref, PARAM_TOL
    )
    report.check("pi_params", state.actor_state.params, pi_ref, PARAM_TOL)
    report.print()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def recorded_noise(fx: Fixture, d: int) -> planner.PlanNoise:
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


@functools.lru_cache(maxsize=None)
def jitted_plan(config: TDMPC2Config, tasks: TaskSet, eval_mode: bool) -> Any:
    return jax.jit(
        functools.partial(
            multitask.plan, config=config, tasks=tasks, eval_mode=eval_mode
        )
    )


def test_multitask_plan_matches_reference_decisions(pfx):
    """Ajax's jitted multi-task plan, chained over the fixture's decisions:
    masked candidates, means and stds, the per-task discount, the task
    embedding (renormed for the look-up of task 2's row, without writing it
    back: deviation T15) and the padded iteration rule."""
    config = ajax_config(reference_config(pfx))
    tasks = task_set(pfx)
    wm, _, pi = (as_jnp(t) for t in torch_to_ajax(pfx, "init/", config))
    report = ErrorReport("TD-MPC2 multi-task plan parity (chained decisions)")
    prev_mean = jnp.zeros((config.horizon, tasks.action_dim))
    n = int(pfx["meta/n_decisions"])
    with jax.default_matmul_precision("highest"):
        for d in range(n):
            p = f"decision{d}/"
            task = int(pfx[p + "task"])
            if d > 0:
                np.testing.assert_array_equal(
                    pfx[p + "prev_mean_in"], pfx[f"decision{d - 1}/prev_mean"]
                )
            plan = jitted_plan(config, tasks, bool(pfx[p + "eval_mode"]))
            action, prev_mean, info = plan(
                wm,
                pi,
                jnp.asarray(pfx[p + "obs"]),
                prev_mean,
                jnp.asarray(pfx[p + "t0"]),
                recorded_noise(pfx, d),
                task,
            )
            report.check("init_mean", info.init_mean, pfx[p + "init_mean"], ACTION_TOL)
            report.check(
                "pi_actions", info.pi_actions, pfx[p + "pi_actions"], PI_ACTION_TOL
            )
            report.check("value", info.value, pfx[p + "value"], VALUE_TOL)
            np.testing.assert_array_equal(
                np.sort(info.elite_idx, axis=-1), np.sort(pfx[p + "elite_idx"], axis=-1)
            )
            report.check(
                "elite_value", info.elite_value, pfx[p + "elite_value"], VALUE_TOL
            )
            report.check(
                "score",
                -np.sort(-np.asarray(info.score), axis=-1),
                -np.sort(-pfx[p + "score"], axis=-1),
                SCORE_TOL,
            )
            report.check("mean", info.mean, pfx[p + "mean"], ACTION_TOL)
            report.check("std", info.std, pfx[p + "std"], ACTION_TOL)
            rank, ref_rank = int(info.elite_rank), int(pfx[p + "elite_rank"])
            assert info.elite_idx[-1, rank] == pfx[p + "elite_idx"][-1, ref_rank]
            report.check("action", action, pfx[p + "action"], ACTION_TOL)
            report.check("prev_mean", prev_mean, pfx[p + "prev_mean"], ACTION_TOL)
            # Masked dims are exactly 0 everywhere (spec 3.9).
            invalid = tasks.action_masks[task] == 0
            for name, value in (
                ("pi_actions", info.pi_actions),
                ("mean", info.mean),
                ("std", info.std),
                ("action", action),
                ("prev_mean", prev_mean),
            ):
                assert np.all(np.asarray(value)[..., invalid] == 0), name
    report.print()
    # The reference wrote task 2's renormed row into its table at act();
    # Ajax's parameters are untouched (T15).
    before, after = pfx["decision0/task_emb/in"], pfx["decision0/task_emb/out"]
    assert np.linalg.norm(before[2]) > 1.0
    np.testing.assert_allclose(np.linalg.norm(after[2]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(wm[TASK_EMB], pfx["init/_task_emb.weight"])
