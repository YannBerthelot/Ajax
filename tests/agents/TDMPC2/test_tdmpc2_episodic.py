"""TD-MPC2's episodic mode (``config.episodic``; deviation T29).

The latest code's support for terminating tasks
(``nicklashansen/tdmpc2@eece801``, as at ``e9f5932``): the termination head
and its loss, ``1 - terminated`` in the TD target, the termination-masked
value estimate of the planner, the variable-length episode replay, and a
training-level known answer on a terminating chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2 import core, planner
from ajax.agents.TDMPC2.networks import make_world_model
from ajax.agents.TDMPC2.state import TDMPC2Config

EPISODIC = TDMPC2Config(
    latent_dim=16,
    enc_dim=32,
    mlp_dim=32,
    num_q=5,
    dropout=0.0,
    num_samples=64,
    num_elites=8,
    num_pi_trajs=6,
    episodic=True,
    termination_coef=0.7,
)
PAPER_ERA = EPISODIC.replace(episodic=False)
OBS, ACT, B = 5, 2, 8
GAMMA = 0.9


def _random_heads(params: dict, key: jax.Array) -> dict:
    """Non-zero reward and Q heads, so values and targets differ by sample."""
    params = jax.tree_util.tree_map(lambda x: x, params)
    k1, k2 = jax.random.split(key)
    for head, k in ((params["reward"]["out"], k1), (params["q"]["members"]["out"], k2)):
        head["kernel"] = 0.3 * jax.random.normal(k, head["kernel"].shape)
    return params


def _with_termination_bias(params: dict, bias: float) -> dict:
    """A head that predicts the same logit ``bias`` everywhere."""
    params = jax.tree_util.tree_map(lambda x: x, params)
    out = params["termination"]["out"]
    out["kernel"] = jnp.zeros_like(out["kernel"])
    out["bias"] = jnp.full_like(out["bias"], bias)
    return params


@pytest.fixture(scope="module")
def state() -> core.TDMPC2UpdateState:
    state = core.create_update_state(jax.random.PRNGKey(0), EPISODIC, OBS, ACT)
    wm = state.world_model_state
    params = _random_heads(wm.params, jax.random.PRNGKey(1))
    wm = wm.replace(params=params, target_params=params["q"])
    return state.replace(world_model_state=wm)


@pytest.fixture(scope="module")
def batch() -> core.TDMPC2Batch:
    k = jax.random.split(jax.random.PRNGKey(2), 4)
    h = EPISODIC.horizon
    return core.TDMPC2Batch(
        obs=jax.random.normal(k[0], (h + 1, B, OBS)),
        action=jax.random.uniform(k[1], (h, B, ACT), minval=-1, maxval=1),
        reward=jax.random.normal(k[2], (h, B)),
        terminated=(jax.random.uniform(k[3], (h, B)) < 0.3).astype(jnp.float32),
    )


@pytest.fixture(scope="module")
def noise() -> core.UpdateNoise:
    return core.draw_update_noise(jax.random.PRNGKey(3), EPISODIC, B, ACT)


# ------------------------------------------------------------------- the head


def test_the_termination_head_exists_only_in_the_episodic_mode():
    """``mlp([z], 2 * [mlp_dim], 1)`` with a ``N(0, 0.02)`` output layer, built
    only when episodic; every other parameter is the paper-era model's."""
    x, a = jnp.zeros((1, OBS)), jnp.zeros((1, ACT))
    key = jax.random.PRNGKey(4)
    on = make_world_model(EPISODIC).init(key, x, a)["params"]
    off = make_world_model(PAPER_ERA).init(key, x, a)["params"]
    assert set(on) == set(off) | {"termination"}
    jax.tree.map(np.testing.assert_array_equal, {k: on[k] for k in off}, off)
    head = on["termination"]
    assert head["out"]["kernel"].shape == (EPISODIC.mlp_dim, 1)
    assert 0.005 < float(jnp.std(head["out"]["kernel"])) < 0.05  # not zeroed
    np.testing.assert_array_equal(head["out"]["bias"], 0.0)
    assert set(head["trunk"]) == set(on["reward"]["trunk"])  # 2 hidden layers
    z = jnp.ones((3, 4, EPISODIC.latent_dim)) / 8
    logits = make_world_model(EPISODIC).apply(
        {"params": on}, z, method="termination_logits"
    )
    assert logits.shape == (3, 4)


# ------------------------------------------------------------------- the loss


def test_the_termination_loss_is_the_bce_at_the_rollout_latents(state, batch):
    """``mean_{H,B} BCE(T(z_{t+1}), terminated_t)`` on the dynamics' latents
    ``z_1 .. z_H``, weighted by ``termination_coef`` into the total, which is
    otherwise the paper-era loss; only the head's loss reaches its weights."""
    wm = state.world_model_state
    args = {
        "wm_apply": wm.apply_fn,
        "batch": batch,
        "next_z": jnp.zeros((EPISODIC.horizon, B, EPISODIC.latent_dim)),
        "td_targets": batch.reward,
        "dropout_key": None,
    }
    total, (terms, zs) = core.world_model_loss(wm.params, config=EPISODIC, **args)
    base, (base_terms, _) = core.world_model_loss(wm.params, config=PAPER_ERA, **args)
    logits = wm.apply_fn({"params": wm.params}, zs[1:], method="termination_logits")
    p = jax.nn.sigmoid(logits)
    t = batch.terminated
    bce = -jnp.mean(t * jnp.log(p) + (1 - t) * jnp.log(1 - p))
    np.testing.assert_allclose(terms["termination_loss"], bce, rtol=1e-5)
    np.testing.assert_allclose(total, base + 0.7 * bce, rtol=1e-5)
    assert set(terms) == set(base_terms) | {
        "termination_loss",
        "termination_rate",
        "termination_f1",
    }
    np.testing.assert_allclose(terms["termination_rate"], jnp.mean(t[-1]))

    def head_loss(head):
        params = {**wm.params, "termination": head}
        return core.world_model_loss(params, config=EPISODIC, **args)[0]

    grads = jax.grad(head_loss)(wm.params["termination"])
    assert all(np.any(np.asarray(g)) for g in jax.tree.leaves(grads))


def test_termination_statistics_are_the_latest_codes():
    logits = jnp.array([2.0, -1.0, 3.0, -2.0, 0.5])
    target = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    stats = core.termination_statistics(logits, target)
    # tp = 2, fn = 1, fp = 1: precision = recall = 2 / 3.
    np.testing.assert_allclose(stats["termination_rate"], 0.6)
    np.testing.assert_allclose(stats["termination_f1"], 2 / 3, rtol=1e-6)


# ------------------------------------------------------------------ TD target


def test_the_td_target_drops_the_bootstrap_after_a_termination(state, batch, noise):
    """``y = r + gamma (1 - terminated) Q``: the reward alone where the next
    state is terminal, the paper-era target elsewhere."""
    wm, pi = state.world_model_state, state.actor_state

    def target(terminated):
        y, _ = core.td_target(
            wm_apply=wm.apply_fn,
            pi_apply=pi.apply_fn,
            wm_params=wm.params,
            target_q_params=wm.params["q"],
            pi_params=pi.params,
            next_obs=batch.obs[1:],
            reward=batch.reward,
            gamma=GAMMA,
            eps=noise.td_eps,
            pair=noise.td_pair,
            dropout_key=None,
            config=EPISODIC,
            terminated=terminated,
        )
        return y

    paper_era = target(None)
    assert not np.allclose(paper_era, batch.reward)
    np.testing.assert_array_equal(target(jnp.zeros_like(batch.reward)), paper_era)
    np.testing.assert_array_equal(target(jnp.ones_like(batch.reward)), batch.reward)
    t = batch.terminated
    np.testing.assert_array_equal(target(t), jnp.where(t == 1, batch.reward, paper_era))


def test_an_episodic_update_trains_the_head_and_masks_its_targets(state, batch, noise):
    """The update logs the termination terms, steps the head, and its TD
    targets read ``batch.terminated`` (a batch that terminates everywhere
    gives another value loss); the paper-era update ignores it."""
    step = jax.jit(
        lambda s, b, c: core.update(s, b, noise, config=c, gamma=GAMMA),
        static_argnums=2,
    )
    new, aux = step(state, batch, EPISODIC)
    assert {"termination_loss", "termination_rate", "termination_f1"} <= set(aux)
    head = state.world_model_state.params["termination"]
    moved = jax.tree.map(
        lambda a, b: bool(np.any(a != b)),
        new.world_model_state.params["termination"],
        head,
    )
    assert all(jax.tree.leaves(moved))
    everywhere = batch.replace(terminated=jnp.ones_like(batch.reward))
    _, ended = step(state, everywhere, EPISODIC)
    assert float(ended["value_loss"]) != float(aux["value_loss"])
    _, a = step(state, batch, PAPER_ERA)
    _, b = step(state, everywhere, PAPER_ERA)
    assert "termination_loss" not in a
    assert float(a["value_loss"]) == float(b["value_loss"])


def test_the_episodic_mode_is_single_task(state, batch):
    wm = state.world_model_state
    task = core.TaskContext(ids=jnp.zeros(B, jnp.int32), mask=jnp.ones((B, ACT)))
    with pytest.raises(ValueError, match="single-task"):
        core.world_model_loss(
            wm.params,
            wm_apply=wm.apply_fn,
            batch=batch,
            next_z=jnp.zeros((EPISODIC.horizon, B, EPISODIC.latent_dim)),
            td_targets=batch.reward,
            dropout_key=None,
            config=EPISODIC,
            task=task,
        )


# ------------------------------------------------------------------ planning


def _estimate(params, config, z, actions, noise):
    return planner.estimate_value(
        params[0],
        params[1],
        z,
        actions,
        noise.terminal_eps[0],
        noise.q_pair[0],
        noise.q_dropout[0],
        config=config,
        gamma=GAMMA,
    )


def test_the_value_estimate_stops_at_a_predicted_termination(state):
    """The latest ``_estimate_value``: ``G = sum_t gamma^t alive_t r_t +
    gamma^H alive_H Q`` with ``alive_{t+1} = alive_t [sigmoid(T(z_{t+1}))
    <= 0.5]``: the reward of the terminating step counts, later ones and
    the terminal Q do not. A head that never terminates gives the
    paper-era estimate; one that always does, the first reward."""
    wm, pi = state.world_model_state.params, state.actor_state.params
    n, h = EPISODIC.num_samples, EPISODIC.horizon
    k = jax.random.split(jax.random.PRNGKey(8), 3)
    z = jax.nn.softmax(jax.random.normal(k[0], (n, EPISODIC.latent_dim)))
    actions = jax.random.uniform(k[1], (h, n, ACT), minval=-1, maxval=1)
    noise = planner.draw_plan_noise(k[2], EPISODIC, ACT)
    apply = make_world_model(EPISODIC).apply

    never = _estimate(
        (_with_termination_bias(wm, -9.0), pi), EPISODIC, z, actions, noise
    )
    paper_era = _estimate((wm, pi), PAPER_ERA, z, actions, noise)
    np.testing.assert_array_equal(never, paper_era)
    always = _estimate(
        (_with_termination_bias(wm, 9.0), pi), EPISODIC, z, actions, noise
    )
    first = EPISODIC.two_hot.decode(
        apply({"params": wm}, z, actions[0], method="reward_logits")
    )
    np.testing.assert_allclose(always, first, rtol=1e-6)

    # A head that terminates some candidates at some steps, against the
    # latest code's loop written out step by step.
    mixed = jax.tree_util.tree_map(lambda x: x, wm)
    mixed["termination"]["out"]["kernel"] = 3.0 * jax.random.normal(
        jax.random.PRNGKey(9), mixed["termination"]["out"]["kernel"].shape
    )
    got = _estimate((mixed, pi), EPISODIC, z, actions, noise)
    full = _estimate((mixed, pi), PAPER_ERA, z, actions, noise)
    value, termination, latent, discount = 0.0, jnp.zeros(n), z, 1.0
    for t in range(h):
        reward = EPISODIC.two_hot.decode(
            apply({"params": mixed}, latent, actions[t], method="reward_logits")
        )
        latent = apply({"params": mixed}, latent, actions[t], method="next")
        value = value + discount * (1 - termination) * reward
        discount = discount * GAMMA
        p = jax.nn.sigmoid(
            apply({"params": mixed}, latent, method="termination_logits")
        )
        termination = jnp.minimum(termination + (p > 0.5), 1.0)
    alive = 1 - termination
    assert 0 < float(jnp.mean(alive)) < 1  # some candidates terminate
    # The terminal Q term is the paper-era estimate's, masked.
    terminal_q = (full - _rewards_only((mixed, pi), z, actions)) / discount
    np.testing.assert_allclose(
        got, value + discount * alive * terminal_q, rtol=1e-5, atol=1e-5
    )


def _rewards_only(params, z, actions):
    """The paper-era discounted reward sum, without the terminal Q."""
    apply = make_world_model(PAPER_ERA).apply
    value, discount = 0.0, 1.0
    for t in range(actions.shape[0]):
        reward = PAPER_ERA.two_hot.decode(
            apply({"params": params[0]}, z, actions[t], method="reward_logits")
        )
        z = apply({"params": params[0]}, z, actions[t], method="next")
        value, discount = value + discount * reward, discount * GAMMA
    return value
