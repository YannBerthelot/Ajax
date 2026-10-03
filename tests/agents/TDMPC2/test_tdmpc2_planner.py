"""TD-MPC2 MPPI planner: the properties of the paper-era ``plan()``.

The end-to-end numbers are pinned against the real reference in
``test_tdmpc2_planner_parity.py``; these tests pin the properties that fixture
cannot isolate or that it runs with fixed settings (dropout, temperature, std
bounds, action dimension, several environments). References are to
``nicklashansen/tdmpc2@5f6fade:tdmpc2/tdmpc2.py`` and
``docs/world_models/tdmpc2_spec.md`` §3.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2 import core, planner
from ajax.agents.TDMPC2.networks import make_policy_prior, make_world_model
from ajax.agents.TDMPC2.state import TDMPC2Config

# Tiny networks; fewer candidates than the paper (512 / 24 / 64) to keep the
# tests fast. The parity test runs the paper's numbers.
SMALL = TDMPC2Config(
    latent_dim=16,
    enc_dim=32,
    mlp_dim=32,
    num_q=5,
    dropout=0.0,
    num_samples=64,
    num_elites=8,
    num_pi_trajs=6,
)
OBS, ACT = 5, 2
GAMMA = 0.9
P = SMALL.num_pi_trajs
# Values of the jitted plan against an eager (or differently traced)
# re-computation: XLA fuses differently, and symexp scales float32 rounding by
# 1 + |V| (values here reach ~20).
EAGER_TOL = {"rtol": 2e-5, "atol": 2e-5}


def make_params(
    config: TDMPC2Config, action_dim: int, key: int = 0, random_policy: bool = False
) -> Any:
    """World-model and policy parameters with non-zero reward / Q heads
    (``N(0, 0.3^2)``), so candidate values differ; with ``random_policy`` the
    policy head too, so the policy trajectories differ from one another."""
    state = core.create_update_state(jax.random.PRNGKey(key), config, OBS, action_dim)
    wm = jax.tree_util.tree_map(lambda x: x, state.world_model_state.params)
    pi = jax.tree_util.tree_map(lambda x: x, state.actor_state.params)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(key + 1), 3)
    reward, q = wm["reward"]["out"], wm["q"]["members"]["out"]
    reward["kernel"] = 0.3 * jax.random.normal(k1, reward["kernel"].shape)
    q["kernel"] = 0.3 * jax.random.normal(k2, q["kernel"].shape)
    if random_policy:
        pi["out"]["kernel"] = 0.3 * jax.random.normal(k3, pi["out"]["kernel"].shape)
    return wm, pi


@pytest.fixture(scope="module")
def params() -> Any:
    return make_params(SMALL, ACT)


@pytest.fixture(scope="module")
def obs() -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(5), (OBS,))


@pytest.fixture(scope="module")
def noise() -> planner.PlanNoise:
    return planner.draw_plan_noise(jax.random.PRNGKey(7), SMALL, ACT)


@functools.lru_cache(maxsize=None)
def jitted_plan(config: TDMPC2Config, eval_mode: bool = False) -> Any:
    return jax.jit(
        functools.partial(planner.plan, config=config, gamma=GAMMA, eval_mode=eval_mode)
    )


def run(
    params: Any,
    obs: jax.Array,
    noise: planner.PlanNoise,
    *,
    config: TDMPC2Config = SMALL,
    prev_mean: Any = None,
    t0: bool = True,
    eval_mode: bool = False,
) -> tuple[jax.Array, jax.Array, planner.PlanInfo]:
    wm, pi = params
    if prev_mean is None:
        prev_mean = jnp.zeros((config.horizon, noise.action_eps.shape[-1]))
    return jitted_plan(config, eval_mode)(wm, pi, obs, prev_mean, t0, noise)


def encode(params: Any, obs: jax.Array, config: TDMPC2Config = SMALL) -> jax.Array:
    return make_world_model(config).apply(
        {"params": params[0]}, obs[None], method="encode"
    )[0]


def values_of(
    params: Any,
    z: jax.Array,
    actions: jax.Array,
    noise: planner.PlanNoise,
    i: int,
    config: TDMPC2Config = SMALL,
) -> jax.Array:
    """Iteration ``i``'s value estimate of ``actions [H, N, A]``."""
    zs = jnp.broadcast_to(z, (actions.shape[1], z.shape[-1]))
    return planner.estimate_value(
        params[0],
        params[1],
        zs,
        actions,
        noise.terminal_eps[i],
        noise.q_pair[i],
        noise.q_dropout[i],
        config=config,
        gamma=GAMMA,
    )


def candidates(
    info: planner.PlanInfo,
    noise: planner.PlanNoise,
    i: int = 0,
    config: TDMPC2Config = SMALL,
) -> jax.Array:
    """Iteration ``i``'s candidates ``[H, N, A]``: the policy trajectories,
    then the clamped Gaussian samples around the previous iteration's mean and
    std (the warm start and ``max_std`` for the first iteration)."""
    mean = info.init_mean if i == 0 else info.mean[i - 1]
    std = config.max_std if i == 0 else info.std[i - 1][:, None]
    sampled = jnp.clip(mean[:, None] + std * noise.candidate_eps[i], -1, 1)
    return jnp.concatenate([info.pi_actions, sampled], axis=1)


# ------------------------------------------------------------- configuration


def test_iterations_plus_two_for_large_action_spaces():
    """``tdmpc2.py:31``: +2 iterations when the action dimension is >= 20."""
    config = TDMPC2Config()
    iterations = [config.planning_iterations(a) for a in (1, 6, 19, 20, 21, 38)]
    assert iterations == [6] * 3 + [8] * 3  # dog: A = 38, humanoid: A = 21
    assert TDMPC2Config(iterations=4).planning_iterations(20) == 6
    noise = planner.draw_plan_noise(jax.random.PRNGKey(0), SMALL, 20)
    assert noise.candidate_eps.shape[0] == noise.q_pair.shape[0] == 8

    big = make_params(SMALL, 20)
    obs = jnp.ones(OBS)
    action, prev_mean, info = run(big, obs, noise)
    assert action.shape == (20,) and prev_mean.shape == (SMALL.horizon, 20)
    assert info.mean.shape == (8, SMALL.horizon, 20)


def test_plan_rejects_noise_drawn_for_another_setting(obs):
    """``plan`` runs one iteration per row of the draws, so it checks them
    against ``config.planning_iterations(A)``: noise of the right action
    dimension but 6 iterations cannot run an A = 20 decision (8 iterations)."""
    big = make_params(SMALL, 20)
    six = planner.draw_plan_noise(
        jax.random.PRNGKey(0), SMALL.replace(iterations=4), 20
    )
    assert six.candidate_eps.shape[0] == 6 and six.action_eps.shape == (20,)
    with pytest.raises(ValueError, match="8 iterations"):
        run(big, obs, six)
    for_two = planner.draw_plan_noise(jax.random.PRNGKey(0), SMALL, ACT)
    with pytest.raises(ValueError, match="action_dim=20"):
        run(big, obs, for_two, prev_mean=jnp.zeros((SMALL.horizon, 20)))
    for config in (SMALL.replace(num_samples=65), SMALL.replace(num_pi_trajs=5)):
        with pytest.raises(ValueError, match="noise does not match"):
            planner.check_plan_noise(for_two, config, ACT)
    planner.check_plan_noise(for_two, SMALL, ACT)


@pytest.mark.parametrize(
    "overrides",
    [
        {"iterations": 0},
        {"num_elites": 0},
        {"num_elites": 65, "num_samples": 64},
        {"num_pi_trajs": -1},
        {"num_pi_trajs": 600},
        {"min_std": 0.0},
        {"min_std": 3.0, "max_std": 2.0},
    ],
)
def test_config_rejects_invalid_planning_settings(overrides):
    with pytest.raises(ValueError):
        TDMPC2Config(**overrides)


def test_draw_plan_noise_shapes_and_ranges():
    draw = jax.jit(lambda k: planner.draw_plan_noise(k, SMALL, ACT))
    n = draw(jax.random.PRNGKey(0))
    h, s, i = SMALL.horizon, SMALL.num_samples, SMALL.iterations
    assert n.pi_eps.shape == (h, P, ACT)
    assert n.candidate_eps.shape == (i, h, s - P, ACT)
    assert n.terminal_eps.shape == (i, s, ACT)
    assert n.q_pair.shape == (i, 2) and n.q_pair.dtype == jnp.int32
    assert np.all(n.q_pair[:, 0] != n.q_pair[:, 1])
    assert np.all((n.q_pair >= 0) & (n.q_pair < SMALL.num_q))
    assert len({tuple(np.asarray(k)) for k in n.q_dropout}) == i
    assert n.elite_uniform.shape == () and 0.0 <= float(n.elite_uniform) < 1.0
    assert n.action_eps.shape == (ACT,)
    other = draw(jax.random.PRNGKey(1))
    for a, b in zip(jax.tree_util.tree_leaves(n), jax.tree_util.tree_leaves(other)):
        assert not np.array_equal(a, b)


def test_q_pairs_are_distinct_heads_over_many_draws():
    """The reference's ``np.random.choice(num_q, 2, replace=False)``: never
    the same head twice, every ordered pair reachable (one key gives only
    ``I`` pairs, which may be distinct by chance)."""
    draw = jax.vmap(lambda k: planner.draw_plan_noise(k, SMALL, ACT))
    pairs = np.asarray(draw(jax.random.split(jax.random.PRNGKey(0), 256)).q_pair)
    pairs = pairs.reshape(-1, 2)
    assert np.all(pairs[:, 0] != pairs[:, 1])
    assert len({tuple(p) for p in pairs}) == SMALL.num_q * (SMALL.num_q - 1)


def test_noise_fields_are_independent_streams(noise):
    """Each field has its own key. With jax's default (partitionable) threefry,
    normal draws from one key share their flat prefix across shapes, so a
    reused key would show as equal leading values in two fields."""
    flat = {
        name: np.ravel(getattr(noise, name))
        for name in ("pi_eps", "candidate_eps", "terminal_eps", "action_eps")
    }
    names = list(flat)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            k = min(flat[a].size, flat[b].size)
            assert not np.any(flat[a][:k] == flat[b][:k]), (a, b)


# ------------------------------------------------------------- warm start, std


def test_warm_start_shifts_the_mean_and_t0_resets_it(params, obs, noise):
    """``tdmpc2.py:130-133`` (spec 3.7): ``mean[:-1] = prev_mean[1:]``, last
    row 0; at ``t0`` the warm start is ignored."""
    prev = jax.random.uniform(jax.random.PRNGKey(2), (SMALL.horizon, ACT), minval=-1)
    _, _, warm = run(params, obs, noise, prev_mean=prev, t0=False)
    expected = jnp.concatenate([prev[1:], jnp.zeros((1, ACT))])
    np.testing.assert_array_equal(warm.init_mean, expected)
    _, _, reset = run(params, obs, noise, prev_mean=prev, t0=True)
    np.testing.assert_array_equal(reset.init_mean, jnp.zeros((SMALL.horizon, ACT)))
    _, _, zero = run(params, obs, noise, t0=True)
    np.testing.assert_array_equal(reset.mean, zero.mean)


def test_std_restarts_at_max_std_every_decision(params, obs, noise):
    """Only the mean is warm-started (spec 3.7): the first iteration of every
    decision samples with ``std = max_std``, whatever the previous decision
    converged to."""
    config = SMALL.replace(max_std=0.5)
    _, prev_mean, first = run(params, obs, noise, config=config)
    assert np.all(first.std[-1] < 0.5)  # the previous decision's std is lower
    _, _, info = run(params, obs, noise, config=config, prev_mean=prev_mean, t0=False)
    z = encode(params, obs, config)
    expected = values_of(
        params, z, candidates(info, noise, 0, config), noise, 0, config
    )
    np.testing.assert_allclose(info.value[0], expected, **EAGER_TOL)


def test_std_is_clamped_to_its_bounds(params, obs, noise):
    """``tdmpc2.py:158-159``: ``std = clip(std, min_std, max_std)``; both
    bounds bind with these settings."""
    config = SMALL.replace(min_std=0.3, max_std=0.4)
    _, _, info = run(params, obs, noise, config=config)
    std = np.asarray(info.std)
    assert np.all((std >= 0.3) & (std <= 0.4))
    assert np.any(std == np.float32(0.3)) and np.any(std == np.float32(0.4))


# ---------------------------------------------------------------- candidates


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_candidates_are_clamped_before_evaluation(params, obs, noise, dropout):
    """``tdmpc2.py:142-144`` (spec 3.8): every iteration evaluates
    ``clip(mean + std * eps, -1, 1)`` around the previous iteration's mean and
    std; many raw samples at ``std = 2`` leave ``[-1, 1]``. With Q dropout
    each iteration's terminal Q pass uses that iteration's own key."""
    config = SMALL.replace(dropout=dropout)
    _, _, info = run(params, obs, noise, config=config)
    z = encode(params, obs, config)
    for i in range(config.iterations):
        clamped = candidates(info, noise, i, config)
        assert np.all(np.abs(clamped) <= 1)
        np.testing.assert_allclose(
            info.value[i],
            values_of(params, z, clamped, noise, i, config),
            **EAGER_TOL,
        )
    raw = info.init_mean[:, None] + config.max_std * noise.candidate_eps[0]
    assert np.mean(np.abs(raw) > 1) > 0.3
    unclamped = jnp.concatenate([info.pi_actions, raw], axis=1)
    unclamped_value = values_of(params, z, unclamped, noise, 0, config)
    assert not np.allclose(unclamped_value, info.value[0])


def test_policy_trajectories_fill_the_first_columns_and_are_rescored(
    params, obs, noise
):
    """``tdmpc2.py:119-126, 135-136`` (spec 3.5, 3.6): ``P`` stochastic prior
    rollouts through the dynamics, the same in every iteration (never
    resampled) but re-scored with each iteration's terminal sample and Q pair."""
    _, _, info = run(params, obs, noise)
    wm, pi = params
    pi_apply = make_policy_prior(SMALL, ACT).apply
    wm_apply = make_world_model(SMALL).apply
    z = jnp.broadcast_to(encode(params, obs), (P, SMALL.latent_dim))
    expected = []
    for t in range(SMALL.horizon):
        a = core.policy_sample(pi_apply, pi, z, noise.pi_eps[t], SMALL).action
        expected.append(a)
        z = wm_apply({"params": wm}, z, a, method="next")
    np.testing.assert_allclose(info.pi_actions, jnp.stack(expected), atol=1e-6)

    z0 = jnp.broadcast_to(encode(params, obs), (P, SMALL.latent_dim))
    for i in range(SMALL.iterations):
        expected_i = planner.estimate_value(
            wm,
            pi,
            z0,
            info.pi_actions,
            noise.terminal_eps[i, :P],
            noise.q_pair[i],
            noise.q_dropout[i],
            config=SMALL,
            gamma=GAMMA,
        )
        np.testing.assert_allclose(info.value[i, :P], expected_i, **EAGER_TOL)
    assert not np.allclose(info.value[0, :P], info.value[1, :P])

    # With the same terminal draws in every iteration the scores repeat.
    same = noise.replace(
        terminal_eps=jnp.broadcast_to(noise.terminal_eps[0], noise.terminal_eps.shape),
        q_pair=jnp.broadcast_to(noise.q_pair[0], noise.q_pair.shape),
    )
    _, _, fixed = run(params, obs, same)
    np.testing.assert_array_equal(
        fixed.value[:, :P], fixed.value[:1, :P].repeat(SMALL.iterations, 0)
    )


def test_planning_without_policy_trajectories(params, obs):
    """``num_pi_trajs = 0``, the "Planning" actor of paper Fig. 9 (spec 3.4)."""
    config = SMALL.replace(num_pi_trajs=0)
    noise = planner.draw_plan_noise(jax.random.PRNGKey(0), config, ACT)
    assert noise.pi_eps.shape == (SMALL.horizon, 0, ACT)
    action, _, info = run(params, obs, noise, config=config)
    assert info.pi_actions.shape == (SMALL.horizon, 0, ACT)
    assert info.value.shape == (SMALL.iterations, SMALL.num_samples)
    assert np.all(np.abs(action) <= 1)


def test_policy_trajectories_compete_as_elites(obs):
    """Columns ``[:P]`` enter top-k, the weighted mean and the final draw like
    any candidate (spec 3.15-3.19): the executed elite may be a policy
    trajectory. Diverse policy trajectories filling 56 of the 64 columns are
    elites with weight in every iteration."""
    config = SMALL.replace(num_pi_trajs=56)
    params = make_params(config, ACT, random_policy=True)
    noise = planner.draw_plan_noise(jax.random.PRNGKey(7), config, ACT)
    _, _, info = run(params, obs, noise, config=config, eval_mode=True)
    p = config.num_pi_trajs
    for i in range(config.iterations):
        elites = candidates(info, noise, i, config)[:, info.elite_idx[i]]
        score = info.score[i]
        assert float(jnp.sum(score[info.elite_idx[i] < p])) > 0.1
        weighted = jnp.sum(score[None, :, None] * elites, axis=1) / (
            jnp.sum(score) + 1e-9
        )
        np.testing.assert_allclose(info.mean[i], weighted, atol=1e-6)

    # Put the elite uniform inside the CDF step of a policy elite of the last
    # iteration (numpy's inverse CDF, as sample_elite).
    last = config.iterations - 1
    cdf = np.cumsum(np.asarray(info.score[last], np.float64))
    cdf = np.concatenate([[0.0], cdf / cdf[-1]])
    ranks = [
        k
        for k in range(config.num_elites)
        if info.elite_idx[last, k] < p and cdf[k + 1] - cdf[k] > 1e-3
    ]
    k = ranks[0]
    drawn = noise.replace(elite_uniform=jnp.asarray((cdf[k] + cdf[k + 1]) / 2))
    action, _, other = run(params, obs, drawn, config=config, eval_mode=True)
    assert int(other.elite_rank) == k
    first = info.pi_actions[0, info.elite_idx[last, k]]
    np.testing.assert_allclose(action, first, atol=1e-6)


# ------------------------------------------------------------- MPPI statistics


def _reference_mppi_step(
    value: np.ndarray, actions: np.ndarray, config: TDMPC2Config
) -> tuple[np.ndarray, ...]:
    """``tdmpc2.py:149-159`` line for line in float32 numpy (torch ops noted)."""
    value = np.nan_to_num(value[:, None], nan=0.0)  # .nan_to_num_(0)
    # torch.topk(value.squeeze(1), E).indices; the values here have no ties.
    elite_idxs = np.argsort(-value[:, 0], kind="stable")[: config.num_elites]
    elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
    max_value = elite_value.max(0)
    score = np.exp(np.float32(config.temperature) * (elite_value - max_value))
    score /= score.sum(0)
    mean = np.sum(score[None] * elite_actions, axis=1) / (score.sum(0) + 1e-9)
    std = np.sqrt(
        np.sum(score[None] * (elite_actions - mean[:, None]) ** 2, axis=1)
        / (score.sum(0) + 1e-9)
    ).clip(config.min_std, config.max_std)
    return elite_idxs, score[:, 0], mean.astype(np.float32), std.astype(np.float32)


@pytest.mark.parametrize("bounds", [(0.05, 2.0), (0.3, 0.4)])
def test_mppi_step_is_the_reference_update(bounds):
    """Weighted mean and biased weighted std around the new mean, the score
    normalised before both divisions by ``sum(score) + 1e-9`` (spec 3.16, 3.17)."""
    config = SMALL.replace(min_std=bounds[0], max_std=bounds[1])
    rng = np.random.default_rng(0)
    value = (3.0 * rng.normal(size=SMALL.num_samples)).astype(np.float32)
    actions = rng.uniform(-1, 1, (SMALL.horizon, SMALL.num_samples, ACT))
    actions = actions.astype(np.float32)
    step = planner.mppi_step(jnp.asarray(value), jnp.asarray(actions), config)
    ref_idx, ref_score, ref_mean, ref_std = _reference_mppi_step(value, actions, config)
    np.testing.assert_array_equal(step.elite_idx, ref_idx)
    np.testing.assert_allclose(step.score, ref_score, rtol=1e-6)
    np.testing.assert_allclose(step.mean, ref_mean, atol=1e-6)
    np.testing.assert_allclose(step.std, ref_std, atol=1e-6)


def test_nan_to_num_semantics():
    """``value.nan_to_num_(0)`` (``tdmpc2.py:149``): NaN -> 0, +inf -> the
    largest finite float32, -inf -> the smallest. A +inf candidate becomes
    the only elite with weight; NaN ranks as 0."""
    value = jnp.array([1.0, jnp.nan, jnp.inf, -jnp.inf, -2.0, 0.5, 3.0, -1.0])
    actions = jnp.linspace(-1, 1, 8 * 2 * 3).reshape(3, 8, 2)
    config = SMALL.replace(num_samples=8, num_elites=4, num_pi_trajs=0)
    step = planner.mppi_step(value, actions, config)
    big = np.finfo(np.float32).max
    np.testing.assert_array_equal(
        step.value, [1.0, 0.0, big, -big, -2.0, 0.5, 3.0, -1.0]
    )
    np.testing.assert_array_equal(step.elite_idx, [2, 6, 0, 5])
    np.testing.assert_array_equal(step.elite_value, [big, 3.0, 1.0, 0.5])
    np.testing.assert_array_equal(step.score, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(step.mean, actions[:, 2], rtol=1e-6)
    six = planner.mppi_step(value, actions, config.replace(num_elites=6)).elite_idx
    assert 1 in np.asarray(six) and 3 not in np.asarray(six)


def test_large_temperature_is_the_argmax_limit(params, obs, noise):
    """With a very large temperature the score puts all weight on the best
    elite: the mean is its action sequence, the std clamps to ``min_std``,
    and the executed (eval-mode) action is its first action."""
    config = SMALL.replace(temperature=1e8)
    action, _, info = run(params, obs, noise, config=config, eval_mode=True)
    best = candidates(info, noise, 0, config)[:, info.elite_idx[0, 0]]
    np.testing.assert_allclose(info.mean[0], best, rtol=1e-6)
    np.testing.assert_array_equal(info.score[:, 0], np.ones(SMALL.iterations))
    np.testing.assert_allclose(info.std, config.min_std)
    assert int(info.elite_rank) == 0
    np.testing.assert_allclose(action, info.mean[-1][0], rtol=1e-6)


# ---------------------------------------------------------- final selection


def test_sample_elite_is_numpys_choice():
    """``np.random.choice(arange(E), p=score)`` (``tdmpc2.py:166``) replayed
    from the uniform it consumes, as the parity fixture records it."""
    rng = np.random.default_rng(0)
    for trial in range(50):
        score = rng.exponential(size=16).astype(np.float32)
        score[rng.integers(16)] = 0.0  # a zero score is never drawn
        score /= score.sum()
        legacy = np.random.RandomState(trial)
        uniform = np.random.RandomState(trial).random_sample()
        expected = legacy.choice(np.arange(16), p=score)
        rank = planner.sample_elite(jnp.asarray(score), jnp.asarray(uniform))
        assert int(rank) == expected and score[int(rank)] > 0
    # A uniform exactly on a CDF value goes past it, to the next non-zero
    # score: numpy's ``cdf.searchsorted(uniform, side='right')``, never a
    # zero score (continuous uniforms above never land there).
    score = jnp.array([0.5, 0.0, 0.5])
    assert int(planner.sample_elite(score, jnp.asarray(0.5))) == 2
    assert int(planner.sample_elite(jnp.array([0.0, 1.0]), jnp.asarray(0.0))) == 1


def test_sample_elite_follows_the_score_distribution():
    score = jnp.array([0.5, 0.0, 0.3, 0.15, 0.05])
    uniforms = jax.random.uniform(jax.random.PRNGKey(0), (40_000,))
    ranks = jax.vmap(planner.sample_elite, in_axes=(None, 0))(score, uniforms)
    freq = np.bincount(np.asarray(ranks), minlength=5) / uniforms.size
    np.testing.assert_allclose(freq, score, atol=0.01)
    assert freq[1] == 0.0


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_eval_mode_removes_only_the_final_noise(params, obs, noise, dropout):
    """``tdmpc2.py:167-171`` (spec 3.20, 3.21): same draws, same planning and
    returned mean; only ``std[0] * eps`` is added outside ``eval_mode``. The
    Q dropout stays on in ``eval_mode`` (5f6fade's ensemble ignores
    ``eval()``; ``eval_mode`` is only the exploration switch)."""
    config = SMALL.replace(dropout=dropout)
    prev = 0.5 * jnp.ones((SMALL.horizon, ACT))
    train_action, train_mean, train = run(
        params, obs, noise, config=config, prev_mean=prev, t0=False
    )
    eval_action, eval_mean, evaluation = run(
        params, obs, noise, config=config, prev_mean=prev, t0=False, eval_mode=True
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(train), jax.tree_util.tree_leaves(evaluation)
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(train_mean, eval_mean)
    np.testing.assert_array_equal(train_mean, train.mean[-1])
    std0 = train.std[-1][0]
    np.testing.assert_allclose(
        train_action, jnp.clip(eval_action + std0 * noise.action_eps, -1, 1), atol=1e-7
    )
    assert not np.allclose(train_action, eval_action)
    assert np.all(np.abs(eval_action) <= 1) and np.all(np.abs(train_action) <= 1)


def test_executed_action_is_the_drawn_elites_first_action(params, obs, noise):
    """``a = elite_actions[0, k]`` with ``k`` drawn from the last iteration's
    score, not the mean (spec 3.19)."""
    last = SMALL.iterations - 1
    ranks = set()
    for u in (0.0, 0.5, 0.999):
        moved = noise.replace(elite_uniform=jnp.asarray(u))
        action, _, info = run(params, obs, moved, eval_mode=True)
        k = int(info.elite_rank)
        assert k == int(planner.sample_elite(info.score[-1], jnp.asarray(u)))
        elite = candidates(info, moved, last)[:, info.elite_idx[last, k]]
        np.testing.assert_allclose(action, elite[0], atol=1e-6)
        assert not np.allclose(action, info.mean[-1][0])
        ranks.add(k)
    assert len(ranks) == 3


# --------------------------------------------------------------- Q dropout


def test_q_dropout_is_active_in_the_terminal_q_pass(obs, noise):
    """Paper era: Q dropout ignores ``eval()``, so planning's Q pass is
    stochastic (deviations.md §2, spec §0.5 item 10). The key changes only
    the terminal term ``gamma^H Q``; with ``dropout = 0`` it changes nothing."""
    config = SMALL.replace(dropout=0.3)
    wm, pi = make_params(config, ACT)
    z = jnp.broadcast_to(encode((wm, pi), obs, config), (8, config.latent_dim))
    actions = jax.random.uniform(jax.random.PRNGKey(4), (config.horizon, 8, ACT))

    def value(key, cfg=config):
        return planner.estimate_value(
            wm,
            pi,
            z,
            actions,
            noise.terminal_eps[0, :8],
            noise.q_pair[0],
            key,
            config=cfg,
            gamma=GAMMA,
        )

    k1, k2 = jax.random.split(jax.random.PRNGKey(11))
    np.testing.assert_array_equal(value(k1), value(k1))
    assert not np.allclose(value(k1), value(k2))

    wm_apply = make_world_model(config).apply
    pi_apply = make_policy_prior(config, ACT).apply
    z_h = z
    for t in range(config.horizon):
        z_h = wm_apply({"params": wm}, z_h, actions[t], method="next")
    a_h = core.policy_sample(pi_apply, pi, z_h, noise.terminal_eps[0, :8], config)

    def q(key):
        logits = core.q_pair_logits(
            config, wm, z_h, a_h.action, noise.q_pair[0], dropout_key=key
        )
        return jnp.mean(config.two_hot.decode(logits), axis=0)

    np.testing.assert_allclose(
        value(k1) - value(k2), GAMMA**3 * (q(k1) - q(k2)), rtol=1e-4, atol=1e-5
    )
    no_dropout = config.replace(dropout=0.0)
    np.testing.assert_array_equal(value(k1, no_dropout), value(k2, no_dropout))

    # Every iteration's Q pass is stochastic in both modes: eval_mode is the
    # exploration switch, not torch's eval(). (The policy columns are the same
    # candidates in both runs. Which key each iteration uses is pinned by
    # test_candidates_are_clamped_before_evaluation.)
    other = noise.replace(q_dropout=jax.random.split(k2, config.iterations))
    for eval_mode in (False, True):
        _, _, a = run((wm, pi), obs, noise, config=config, eval_mode=eval_mode)
        _, _, b = run((wm, pi), obs, other, config=config, eval_mode=eval_mode)
        for i in range(config.iterations):
            assert not np.allclose(a.value[i, :P], b.value[i, :P])


# ------------------------------------------------------ jit, vmap, interface


def test_plan_encodes_the_observation(params, obs, noise):
    """``act()`` encodes, then plans from the latent (``tdmpc2.py:84-89``)."""
    wm, pi = params
    prev = jnp.zeros((SMALL.horizon, ACT))
    a, m, info = run(params, obs, noise)
    b, n, other = planner.plan_from_latent(
        wm,
        pi,
        encode(params, obs),
        prev,
        True,
        noise,
        config=SMALL,
        gamma=GAMMA,
        eval_mode=False,
    )
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(m, n, atol=1e-6)
    np.testing.assert_array_equal(info.elite_idx, other.elite_idx)


def test_plan_is_reproducible_and_bounded(params, obs, noise):
    a1, m1, _ = run(params, obs, noise)
    a2, m2, _ = run(params, obs, noise)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(m1, m2)
    other = planner.draw_plan_noise(jax.random.PRNGKey(8), SMALL, ACT)
    a3, _, _ = run(params, obs, other)
    assert not np.allclose(a1, a3)
    huge = noise.replace(action_eps=1e3 * jnp.ones(ACT))
    a4, _, _ = run(params, obs, huge)
    np.testing.assert_array_equal(a4, jnp.ones(ACT))


def test_vmap_over_envs_equals_a_loop(params):
    """Per-env observation, ``t0``, warm start and draws (deviation T7)."""
    wm, pi = params
    n_envs = 3
    obs = jax.random.normal(jax.random.PRNGKey(1), (n_envs, OBS))
    prev = jax.random.uniform(
        jax.random.PRNGKey(2), (n_envs, SMALL.horizon, ACT), minval=-1
    )
    t0 = jnp.array([True, False, False])
    noise = jax.vmap(lambda k: planner.draw_plan_noise(k, SMALL, ACT))(
        jax.random.split(jax.random.PRNGKey(3), n_envs)
    )
    plan = functools.partial(planner.plan, config=SMALL, gamma=GAMMA, eval_mode=False)
    batched = jax.jit(jax.vmap(plan, in_axes=(None, None, 0, 0, 0, 0)))
    actions, means, infos = batched(wm, pi, obs, prev, t0, noise)
    assert actions.shape == (n_envs, ACT) and means.shape == prev.shape
    single = jax.jit(plan)
    for e in range(n_envs):
        env_noise = jax.tree_util.tree_map(lambda x, e=e: x[e], noise)
        action, mean, info = single(wm, pi, obs[e], prev[e], t0[e], env_noise)
        np.testing.assert_allclose(actions[e], action, atol=1e-6)
        np.testing.assert_allclose(means[e], mean, atol=1e-6)
        np.testing.assert_array_equal(infos.elite_idx[e], info.elite_idx)
        np.testing.assert_allclose(infos.init_mean[e], info.init_mean)
    np.testing.assert_array_equal(infos.init_mean[0], 0.0)
    np.testing.assert_array_equal(infos.init_mean[1, :-1], prev[1, 1:])


def test_discount_powers_follow_the_reference():
    """A Python discount is multiplied in float64 (``tdmpc2.py:102``); a traced
    one in float32 under ``jit``, with the same values to rounding."""
    powers = planner._discount_powers(0.9, 3)
    assert powers == [1.0, 0.9, 0.9 * 0.9, 0.9 * 0.9 * 0.9]
    traced = jax.jit(lambda g: jnp.stack(planner._discount_powers(g, 3)))(0.9)
    np.testing.assert_allclose(traced, np.float32(powers), rtol=1e-6)


def test_array_discount_matches_a_python_discount(params, obs, noise):
    wm, pi = params
    prev = jnp.zeros((SMALL.horizon, ACT))
    plan = jax.jit(
        lambda gamma: planner.plan(
            wm, pi, obs, prev, True, noise, config=SMALL, gamma=gamma, eval_mode=True
        )
    )
    a, _, info = plan(jnp.float32(GAMMA))
    b, _, other = run(params, obs, noise, eval_mode=True)
    np.testing.assert_allclose(info.value, other.value, **EAGER_TOL)
    np.testing.assert_allclose(a, b, atol=1e-5)
