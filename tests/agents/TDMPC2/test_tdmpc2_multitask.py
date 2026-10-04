"""TD-MPC2 multi-task mechanisms (M8): task set, embedding, masks, discounts.

Ajax-side behaviour tests of :mod:`ajax.agents.TDMPC2.multitask` and of the
task hooks of :mod:`ajax.agents.TDMPC2.core` / :mod:`ajax.agents.TDMPC2.planner`
(``docs/world_models/DESIGN.md`` §7; tdmpc2_spec 1.21-1.24, 2.18, 2.20, 3.1,
3.9). The parity with the reference is ``test_tdmpc2_multitask_parity.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2 import core, multitask, planner
from ajax.agents.TDMPC2.core import TASK_EMB
from ajax.agents.TDMPC2.multitask import TaskSet
from ajax.agents.TDMPC2.state import TDMPC2Config

TINY = TDMPC2Config(
    latent_dim=16,
    enc_dim=32,
    mlp_dim=32,
    num_q=2,
    num_samples=32,
    num_elites=8,
    num_pi_trajs=4,
    iterations=2,
    batch_size=8,
)
# Three tasks of distinct obs / action dims and episode lengths; padded to 5
# and 4. Task 1 has a single valid action dim.
TASKS = TaskSet.create((3, 5, 2), (2, 1, 4), (100, 200, 500))
TASK_DIM = 6


def _state(key=0, learning_rate=3e-4):
    return multitask.create_update_state(
        jax.random.PRNGKey(key),
        TINY,
        TASKS,
        task_dim=TASK_DIM,
        learning_rate=learning_rate,
    )


def _with_table(state, table):
    wm = state.world_model_state
    params = {**wm.params, TASK_EMB: jnp.asarray(table, jnp.float32)}
    return state.replace(
        world_model_state=wm.replace(params=params, opt_state=wm.tx.init(params))
    )


def _with_random_heads(wm_params, key=11, std=0.3):
    """The reward and Q heads' final layers drawn ``N(0, std^2)``.

    The paper zero-initialises them, so an untrained model's planning values
    all tie and neither the discount nor the embedding can change a plan;
    random heads (as the parity fixtures' ``HEAD_STD``) make them matter.
    """
    k_reward, k_q = jax.random.split(jax.random.PRNGKey(key))
    reward, q = wm_params["reward"], wm_params["q"]
    reward_out, q_out = reward["out"], q["members"]["out"]
    return {
        **wm_params,
        "reward": {
            **reward,
            "out": {
                **reward_out,
                "kernel": std * jax.random.normal(k_reward, reward_out["kernel"].shape),
            },
        },
        "q": {
            **q,
            "members": {
                **q["members"],
                "out": {
                    **q_out,
                    "kernel": std * jax.random.normal(k_q, q_out["kernel"].shape),
                },
            },
        },
    }


def _rows_with_norms(norms, key=1):
    direction = jax.random.normal(jax.random.PRNGKey(key), (len(norms), TASK_DIM))
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    return direction * jnp.asarray(norms)[:, None]


def _batch(task, key=2):
    """A batch of the tasks ``task [B]``: padded obs, actions zero on the
    invalid dims (the dataset schema)."""
    task = jnp.asarray(task, jnp.int32)
    h, b = TINY.horizon, task.shape[0]
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(key), 3)
    obs_valid = (
        jnp.arange(TASKS.obs_dim)[None] < jnp.asarray(TASKS.obs_dims)[task][:, None]
    )
    return core.TDMPC2Batch(
        obs=jax.random.normal(k1, (h + 1, b, TASKS.obs_dim)) * obs_valid,
        action=jax.random.uniform(k2, (h, b, TASKS.action_dim), minval=-1, maxval=1)
        * TASKS.context(task).mask,
        reward=jax.random.normal(k3, (h, b)),
    )


# ---------------------------------------------------------------------------
# Task set
# ---------------------------------------------------------------------------


def test_task_set_tables():
    """Prefix masks, per-task discounts of the paper's heuristic, padded dims
    (``world_model.py:21-23``, ``tdmpc2.py:32-49``; spec 1.23, 2.20)."""
    np.testing.assert_array_equal(
        TASKS.action_masks,
        [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
    )
    assert TASKS.discounts == pytest.approx((0.95, 0.975, 0.99))
    assert TASKS.discount_table.dtype == np.float32
    assert (TASKS.num_tasks, TASKS.obs_dim, TASKS.action_dim) == (3, 5, 4)
    assert TASKS.names == ("task0", "task1", "task2")
    ctx = TASKS.context(jnp.array([2, 1]))
    np.testing.assert_array_equal(ctx.mask, TASKS.action_masks[[2, 1]])
    np.testing.assert_array_equal(
        TASKS.discount(jnp.array([1, 0])), np.float32([0.975, 0.95])
    )
    assert TASKS.context(1).ids.shape == ()  # one decision
    hash(TASKS)  # a static jit argument
    # T = 10: (2 - 1) / 2 clipped up to discount_min.
    custom = TaskSet.create((3,), (2,), (10,), discount_min=0.9, names=["walk"])
    assert custom.discounts == pytest.approx((0.9,)) and custom.names == ("walk",)
    with pytest.raises(ValueError, match="one obs dim"):
        TaskSet((3, 4), (1,), (10, 10), (0.9, 0.9), ("a", "b"))
    with pytest.raises(ValueError, match=">= 1"):
        TaskSet.create((3, 0), (1, 1), (10, 10))


def test_observations_are_zero_padded_at_the_end():
    obs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        multitask.pad_observation(obs, 4), [[1, 2, 0, 0], [3, 4, 0, 0]]
    )
    assert multitask.pad_observation(obs, 2) is obs
    with pytest.raises(ValueError, match="cannot pad"):
        multitask.pad_observation(obs, 1)


# ---------------------------------------------------------------------------
# Embedding table
# ---------------------------------------------------------------------------


def test_the_embedding_table_is_a_world_model_parameter():
    """``U(-0.02, 0.02)`` ``[num_tasks, task_dim]`` (``init.py:10-11``), in
    the world-model parameters and their Adam at the full learning rate
    (``tdmpc2.py:26``), not in target Q (``world_model.py:31``)."""
    state = _state()
    table = state.world_model_state.params[TASK_EMB]
    assert table.shape == (TASKS.num_tasks, TASK_DIM)
    assert float(jnp.abs(table).max()) <= 0.02 and float(jnp.abs(table).min()) > 0
    assert TASK_EMB not in state.world_model_state.target_params
    grads = jax.tree.map(jnp.ones_like, state.world_model_state.params)
    updates, _ = state.world_model_state.tx.update(
        grads, state.world_model_state.opt_state
    )
    np.testing.assert_allclose(updates[TASK_EMB], -3e-4, rtol=1e-4)
    # The single-task keys are unchanged by the table's draw.
    st = core.create_update_state(jax.random.PRNGKey(0), TINY, 5, 4, task_dim=TASK_DIM)
    for name in ("encoder", "dynamics", "reward", "q"):
        for a, b in zip(
            jax.tree.leaves(st.world_model_state.params[name]),
            jax.tree.leaves(state.world_model_state.params[name]),
        ):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="task_dim > 0"):
        core.create_update_state(jax.random.PRNGKey(0), TINY, 5, 4, num_tasks=3)


def test_multitask_parameter_count_matches_the_paper_at_5m():
    """Spec 1.24 (App. H): 80 tasks, padded S = 39, A = 6, task_dim 96 at 5M:
    5,389,930 parameters, the 7,680-parameter embedding table included."""
    config = TDMPC2Config.from_model_size(5)
    tasks = TaskSet.create((39,) * 80, (6,) * 80, (500,) * 80)
    shapes = jax.eval_shape(
        lambda k: multitask.create_update_state(k, config, tasks, task_dim=96),
        jax.random.PRNGKey(0),
    )

    def count(tree):
        return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(tree))

    wm = shapes.world_model_state.params
    assert count(wm[TASK_EMB]) == 7_680
    assert count(wm) + count(shapes.actor_state.params) == 5_389_930


def test_renorm_rescales_only_looked_up_rows_above_norm_one():
    """``torch.embedding_renorm_`` (spec 1.22): looked-up rows of norm > 1
    become ``w / (|w| + 1e-7)``; looked-up rows inside the ball and rows not
    looked up are unchanged; repeated ids are fine."""
    table = _rows_with_norms([3.0, 0.5, 2.0])
    out = core.renorm_task_embedding({TASK_EMB: table}, jnp.array([0, 1, 0, 1]))
    out = out[TASK_EMB]
    np.testing.assert_allclose(out[0], table[0] / (3.0 + 1e-7), rtol=1e-6)
    np.testing.assert_array_equal(out[1], table[1])
    np.testing.assert_array_equal(out[2], table[2])  # not looked up
    assert float(jnp.linalg.norm(out[0])) <= 1.0
    # A scalar id (one planning decision).
    single = core.renorm_task_embedding({TASK_EMB: table}, jnp.int32(2))[TASK_EMB]
    np.testing.assert_allclose(jnp.linalg.norm(single[2]), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(single[0], table[0])


def test_update_renorms_and_writes_back_before_the_target_and_after_the_step(
    monkeypatch,
):
    """The two look-up points of one update (``tdmpc2.py:232, 186``).

    Pre-step: the update of a table whose looked-up row has norm 3 is the
    update of the pre-renormed table. Post-step: with a large learning rate
    Adam pushes the looked-up rows above norm 1, and they leave the update
    renormed. A row the batch does not look up is never renormed (and its
    first Adam step is 0: no gradient, no moments).
    """
    seen: list = []
    renorm = core.renorm_task_embedding

    def spy(wm_params, ids):
        jax.debug.callback(
            lambda t: seen.append(np.asarray(t)), wm_params[TASK_EMB], ordered=True
        )
        return renorm(wm_params, ids)

    monkeypatch.setattr(core, "renorm_task_embedding", spy)
    task = jnp.array([0, 1, 0, 1, 1, 0, 0, 1], jnp.int32)
    table = _rows_with_norms([3.0, 0.5, 2.0])
    batch = _batch(task)
    noise = core.draw_update_noise(jax.random.PRNGKey(3), TINY, 8, TASKS.action_dim)
    step = jax.jit(
        lambda s: multitask.update(s, batch, task, noise, config=TINY, tasks=TASKS)
    )
    new, aux = step(_with_table(_state(learning_rate=0.5), table))
    jax.effects_barrier()
    assert len(seen) == 2  # the two write-back points
    np.testing.assert_array_equal(seen[0], table)
    after_adam = seen[1]
    assert np.all(np.linalg.norm(after_adam[:2], axis=-1) > 1.0)
    final = np.asarray(new.world_model_state.params[TASK_EMB])
    np.testing.assert_allclose(np.linalg.norm(final[:2], axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(final[2], table[2])

    pre = renorm({TASK_EMB: table}, task)[TASK_EMB]
    same, same_aux = step(_with_table(_state(learning_rate=0.5), pre))
    # (The pre-renormed row may be renormed again by a float32 ulp.)
    for name in aux:
        np.testing.assert_allclose(aux[name], same_aux[name], rtol=1e-5, err_msg=name)
    for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(same)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_plan_renorms_the_row_without_persisting_it():
    """Evaluation renorms the looked-up row for the decision, not in the
    parameters (deviation T15): planning with a row of norm 4 is planning
    with its renormed row, the renorm matters (planning from the latent of
    the raw row differs), and only the decision is returned."""
    state = _state()
    wm, pi = state.world_model_state.params, state.actor_state.params
    big = {**wm, TASK_EMB: _rows_with_norms([0.3, 0.6, 4.0])}
    renormed = core.renorm_task_embedding(big, jnp.int32(2))
    noise = planner.draw_plan_noise(jax.random.PRNGKey(4), TINY, TASKS.action_dim)
    prev = jnp.zeros((TINY.horizon, TASKS.action_dim))
    obs = jnp.array([0.3, -0.2, 0.0, 0.0, 0.0])

    @jax.jit
    def plan(w):
        return multitask.plan(
            w, pi, obs, prev, True, noise, 2, config=TINY, tasks=TASKS, eval_mode=True
        )

    action, _, info = plan(big)
    action_r, _, info_r = plan(renormed)
    np.testing.assert_array_equal(action, action_r)
    np.testing.assert_array_equal(info.pi_actions, info_r.pi_actions)

    @jax.jit
    def plan_raw(w):  # plan_from_latent reads the table as given
        ctx = TASKS.context(2)
        z = state.world_model_state.apply_fn(
            {"params": w}, obs[None], w[TASK_EMB][2], method="encode"
        )[0]
        return planner.plan_from_latent(
            w,
            pi,
            z,
            prev,
            True,
            noise,
            config=TINY,
            gamma=TASKS.discount(2),
            eval_mode=True,
            task=ctx,
        )

    # (Untrained reward and Q heads are zero: the policy trajectories carry
    # the embedding's effect.)
    np.testing.assert_array_equal(plan_raw(renormed)[2].pi_actions, info.pi_actions)
    assert not np.allclose(plan_raw(big)[2].pi_actions, info.pi_actions, atol=1e-4)


# ---------------------------------------------------------------------------
# Masks and the entropy's valid dims
# ---------------------------------------------------------------------------


def test_masked_policy_samples_and_the_valid_dims_in_the_entropy():
    """``world_model.py:136-148`` (spec 1.9, 1.23): mean, log-std and noise
    are masked before sampling; invalid dims are exactly 0; the Gaussian
    log-probability is scaled by the number of valid dims ``n``; the tanh
    correction runs over every dim (a masked dim adds ``log(1 + 1e-6)``)."""
    key = jax.random.split(jax.random.PRNGKey(5), 3)
    mean = jax.random.normal(key[0], (3, 4))
    raw = jax.random.normal(key[1], (3, 4))
    eps = jax.random.normal(key[2], (3, 4))
    mask = TASKS.context(jnp.array([0, 1, 2])).mask
    sample = core.squashed_gaussian(mean, raw, eps, TINY, mask)
    invalid = np.asarray(mask) == 0
    for name in ("action", "mean", "log_std"):
        assert np.all(np.asarray(getattr(sample, name))[invalid] == 0), name
    log_std = TINY.log_std_min + 0.5 * (TINY.log_std_max - TINY.log_std_min) * (
        np.tanh(np.asarray(raw)) + 1
    )
    m, e, ls = np.asarray(mask), np.asarray(eps) * mask, log_std * mask
    n = m.sum(-1)
    np.testing.assert_array_equal(n, [2, 1, 4])
    u = np.asarray(mean) * m + e * np.exp(ls)
    expected = n * (np.sum(-0.5 * e**2 - ls, -1) - 0.5 * np.log(2 * np.pi)) - np.sum(
        np.log(np.maximum(1 - np.tanh(u) ** 2, 0) + 1e-6), -1
    )
    np.testing.assert_allclose(sample.log_pi, expected, rtol=2e-6, atol=2e-6)
    # Without a mask: n = A (the single-task computation).
    full = core.squashed_gaussian(mean, raw, eps, TINY)
    np.testing.assert_allclose(
        full.log_pi,
        core.squashed_gaussian(mean, raw, eps, TINY, jnp.ones((3, 4))).log_pi,
        rtol=1e-6,
    )
    # The invalid outputs of the policy head get no gradient.
    grad = jax.grad(
        lambda mu, r: jnp.sum(core.squashed_gaussian(mu, r, eps, TINY, mask).log_pi)
        + jnp.sum(core.squashed_gaussian(mu, r, eps, TINY, mask).action),
        argnums=(0, 1),
    )(mean, raw)
    for g in grad:
        assert np.all(np.asarray(g)[invalid] == 0)


@pytest.mark.parametrize("eval_mode", [True, False], ids=["eval", "train"])
def test_planner_masks_candidates_means_stds_and_actions(eval_mode):
    """Spec 3.9: invalid dims are exactly 0 in the policy trajectories, the
    means, the stds (0 < min_std, masked after the clamp), the warm start
    and the executed action, and the noise drawn on them changes nothing
    (candidates are masked after sampling, every iteration)."""
    state = _state()
    wm, pi = state.world_model_state.params, state.actor_state.params
    noise = planner.draw_plan_noise(jax.random.PRNGKey(6), TINY, TASKS.action_dim)
    plan = jax.jit(
        lambda nz, prev, t0: multitask.plan(
            wm,
            pi,
            jnp.array([0.4, -0.1, 0.7, 0.0, 0.0]),
            prev,
            t0,
            nz,
            1,  # one valid dim of four
            config=TINY,
            tasks=TASKS,
            eval_mode=eval_mode,
        )
    )
    prev = jnp.zeros((TINY.horizon, TASKS.action_dim))
    action, mean, info = plan(noise, prev, True)
    action2, mean2, info2 = plan(noise, mean, False)  # a warm start
    loud = noise.replace(
        pi_eps=noise.pi_eps.at[..., 1:].set(50.0),
        candidate_eps=noise.candidate_eps.at[..., 1:].set(-50.0),
        terminal_eps=noise.terminal_eps.at[..., 1:].set(50.0),
        action_eps=noise.action_eps.at[1:].set(50.0),
    )
    action_loud, mean_loud, info_loud = plan(loud, prev, True)
    for out in (
        (action, action_loud),
        (mean, mean_loud),
        (info.value, info_loud.value),
        (info.std, info_loud.std),
    ):
        np.testing.assert_array_equal(*out)
    for value in (
        action,
        mean,
        info.mean,
        info.std,
        info.pi_actions,
        action2,
        mean2,
        info2.init_mean,
        info2.mean,
    ):
        assert np.all(np.asarray(value)[..., 1:] == 0)
    assert np.all(np.asarray(info.std)[..., 0] >= TINY.min_std)
    assert np.any(np.asarray(action)[0] != 0)


def test_iterations_follow_the_padded_action_dim():
    """``cfg.iterations += 2 * (cfg.action_dim >= 20)`` with the padded dim
    (``tdmpc2.py:31``; spec 3.1): a 2-dim task of a set padded to 20 plans
    with the +2 (a shape: traced, not run)."""
    tasks = TaskSet.create((3, 3), (2, 20), (100, 100))

    def decide(key):
        state = multitask.create_update_state(key, TINY, tasks, task_dim=TASK_DIM)
        noise = planner.draw_plan_noise(key, TINY, tasks.action_dim)
        return multitask.plan(
            state.world_model_state.params,
            state.actor_state.params,
            jnp.zeros(3),
            jnp.zeros((TINY.horizon, 20)),
            True,
            noise,
            0,
            config=TINY,
            tasks=tasks,
            eval_mode=True,
        )

    _, _, info = jax.eval_shape(decide, jax.random.PRNGKey(0))
    assert info.value.shape[0] == TINY.iterations + 2
    assert TINY.planning_iterations(TASKS.action_dim) == TINY.iterations


# ---------------------------------------------------------------------------
# Discounts, padding, gradient routing
# ---------------------------------------------------------------------------


def test_td_targets_use_each_samples_task_discount():
    """``y = r + discount[task] * Q`` per sample (``tdmpc2.py:215``); the
    update passes ``discount[task]`` (pinned by the multi-task parity
    fixture, whose batches mix tasks of distinct discounts)."""
    state = _state()
    wm, pi = state.world_model_state, state.actor_state
    task = jnp.array([0, 1, 2, 0, 2, 1, 1, 0], jnp.int32)
    batch = _batch(task)
    noise = core.draw_update_noise(jax.random.PRNGKey(8), TINY, 8, TASKS.action_dim)
    ctx = TASKS.context(task)

    @jax.jit
    def target(gamma):
        return core.td_target(
            wm.apply_fn,
            pi.apply_fn,
            wm.params,
            wm.target_params,
            pi.params,
            batch.obs[1:],
            batch.reward,
            gamma,
            noise.td_eps,
            noise.td_pair,
            None,
            TINY,
            ctx,
        )[0]

    gamma = TASKS.discount(task)
    np.testing.assert_array_equal(
        gamma, np.float32([0.95, 0.975, 0.99, 0.95, 0.99, 0.975, 0.975, 0.95])
    )
    q = target(jnp.ones(8)) - batch.reward
    np.testing.assert_allclose(
        target(gamma), batch.reward + gamma * q, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("task", [0, 1, 2])
def test_task_planner_policy_pads_observations_and_slices_actions(task):
    """The evaluation policy in task ``task``'s env (obs 3 / 5 / 2, action
    2 / 1 / 4): it pads the observation to 5, plans in the padded space with
    *that* task's embedding, mask and discount, and returns the action's
    valid dims (``MultitaskWrapper``). Random reward and Q heads make the
    plan depend on the discount (checked)."""
    state = _state()
    wm = _with_random_heads(state.world_model_state.params)
    pi = state.actor_state.params
    obs_dim, action_dim = TASKS.obs_dims[task], TASKS.action_dims[task]
    obs = jnp.linspace(-0.5, 0.5, 2 * obs_dim).reshape(2, obs_dim)
    carry = jnp.zeros((2, TINY.horizon, TASKS.action_dim))
    key = jax.random.PRNGKey(9)
    action, new_carry, _ = jax.jit(
        lambda c, o, f, k: multitask.task_planner_policy(
            c,
            o,
            f,
            k,
            wm_params=wm,
            pi_params=pi,
            config=TINY,
            tasks=TASKS,
            task=task,
            eval_mode=True,
        )
    )(carry, obs, jnp.ones(2, bool), key)
    assert action.shape == (2, action_dim) and new_carry.shape == carry.shape
    noise = jax.vmap(lambda k: planner.draw_plan_noise(k, TINY, 4))(
        jax.random.split(key, 2)
    )
    reference = jax.jit(
        lambda o, c, nz, gamma: planner.plan(
            wm,
            pi,
            o,
            c,
            True,
            nz,
            config=TINY,
            gamma=gamma,
            eval_mode=True,
            task=TASKS.context(task),
        )
    )
    gamma = jnp.float32((0.95, 0.975, 0.99)[task])
    # The policy (vmapped over 2 lanes) and the per-lane plan are different
    # programs, so float32 rounding through the MPPI iterations differs between
    # them: up to 1.6e-6 on Linux x86 (CI). 1e-5 keeps a 10x margin below the
    # 1e-4 by which another task's discount moves the plan (checked below).
    for i in range(2):
        inputs = (
            jnp.concatenate([obs[i], jnp.zeros(TASKS.obs_dim - obs_dim)]),
            carry[i],
            jax.tree.map(lambda x, i=i: x[i], noise),
        )
        a, m, _ = reference(*inputs, gamma)
        np.testing.assert_allclose(action[i], a[:action_dim], atol=1e-5)
        np.testing.assert_allclose(new_carry[i], m, atol=1e-5)
        assert np.all(np.asarray(m)[:, action_dim:] == 0)
        # Another task's discount plans differently.
        other = jnp.float32((0.975, 0.99, 0.95)[task])
        assert np.abs(np.asarray(reference(*inputs, other)[1]) - m).max() > 1e-4


def test_gradients_reach_the_looked_up_rows_from_the_world_model_only():
    """The world-model loss trains the rows its batch looks up (only those);
    the policy loss gives the table no gradient (5f6fade's
    ``track_q_grad(False)`` freezes ``_task_emb``; spec 2.18)."""
    state = _state()
    wm, pi = state.world_model_state, state.actor_state
    task = jnp.array([0, 2, 0, 2, 2, 0, 0, 2], jnp.int32)
    batch = _batch(task)
    ctx = TASKS.context(task)
    noise = core.draw_update_noise(jax.random.PRNGKey(10), TINY, 8, TASKS.action_dim)

    @jax.jit
    def gradients(wm_params):
        td, next_z = core.td_target(
            wm.apply_fn,
            pi.apply_fn,
            wm_params,
            wm.target_params,
            pi.params,
            batch.obs[1:],
            batch.reward,
            TASKS.discount(task),
            noise.td_eps,
            noise.td_pair,
            None,
            TINY,
            ctx,
        )
        wm_grads, (_, zs) = jax.grad(core.world_model_loss, has_aux=True)(
            wm_params, wm.apply_fn, batch, next_z, td, None, TINY, ctx
        )

        def pi_loss_wrt_wm(params):
            return core.policy_loss(
                pi.params,
                pi.apply_fn,
                wm.apply_fn,
                params,
                zs,
                state.q_scale,
                noise.pi_eps,
                noise.pi_pair,
                None,
                TINY,
                ctx,
            )[0]

        return wm_grads, jax.grad(pi_loss_wrt_wm)(wm_params)

    wm_grads, pi_grads = gradients(wm.params)
    rows = np.abs(np.asarray(wm_grads[TASK_EMB])).sum(-1)
    assert rows[0] > 0 and rows[2] > 0 and rows[1] == 0
    for leaf in jax.tree.leaves(pi_grads):
        assert np.all(np.asarray(leaf) == 0)
