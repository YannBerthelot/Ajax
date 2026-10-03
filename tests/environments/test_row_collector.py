"""Row collector (ajax.environments.row_collector): row conventions, both
reset modes on gymnax / brax / mujoco_playground, the random-action phase,
per-env params, the raw-env precondition, and the unbatched-cond structure
under the seed vmap."""

import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.environments.create import build_env_from_id, prepare_env
from ajax.environments.row_collector import collect_row, init_row_collector_state
from ajax.environments.system_class import broadcast_env_params
from ajax.environments.utils import agent_episode_length
from ajax.state import EnvironmentConfig


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)


def env_args_for(env_id, n_envs, **build_kwargs):
    env, env_params = build_env_from_id(env_id, n_envs=n_envs, **build_kwargs)
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=n_envs,
        continuous=env_id != "CartPole-v1",
        action_repeat=build_kwargs.get("action_repeat", 1),
    )


class Run(NamedTuple):
    state: Any  # final collector state
    rows: Any  # Row, leading axis n_ticks
    env_states: Any  # a copy of the env state taken before each tick
    last_terminated: Any  # the state's flags after each tick, [n_ticks, n_envs]
    last_truncated: Any


def run_collector(
    env_args,
    policy,
    n_ticks,
    *,
    reset_mode,
    episode_length=None,
    random_phase=None,
    timestep_unit="rows",
    carry=None,
    seed=0,
):
    """Scan ``n_ticks`` ticks of the collector."""
    state = init_row_collector_state(
        jax.random.PRNGKey(seed), env_args, policy_carry=carry
    )

    def body(state, tick):
        # Copied into fresh containers before the tick: if the tick mutated
        # the env state's containers in place (brax wrappers update
        # ``state.info`` in place), this record would not see it.
        before = jax.tree.map(lambda x: x, state.env_state)
        new_state, row = collect_row(
            state,
            tick,
            policy,
            env_args=env_args,
            reset_mode=reset_mode,
            episode_length=episode_length,
            random_phase=None if random_phase is None else random_phase(tick),
            timestep_unit=timestep_unit,
        )
        flags = (new_state.last_terminated, new_state.last_truncated)
        return new_state, (row, before, *flags)

    state, outputs = jax.jit(lambda s: jax.lax.scan(body, s, jnp.arange(n_ticks)))(
        state
    )
    return Run(state, *outputs)


def episodes(rows, env):
    """(first_tick, last_tick) of every completed episode of one env."""
    first = np.asarray(rows.is_first[:, env])
    last = np.asarray(rows.is_last[:, env])
    starts = np.flatnonzero(first)
    ends = np.flatnonzero(last)
    return list(zip(starts, ends))


def check_row_conventions(rows):
    """Invariants shared by both modes: a reset row follows every final row
    and nothing else; reward 0 on reset rows; action 0 on final rows;
    is_terminal only on final rows."""
    first = np.asarray(rows.is_first)
    last = np.asarray(rows.is_last)
    terminal = np.asarray(rows.is_terminal)
    assert first[0].all()
    np.testing.assert_array_equal(first[1:], last[:-1])
    assert not (first & last).any()
    assert (np.asarray(rows.reward)[first] == 0).all()
    assert (np.asarray(rows.action)[last] == 0).all()
    assert not (terminal & ~last).any()


def finished_returns(rows, state, env):
    """Returns of the finished episodes of one env, from its rows (plus the
    episode whose final row the state emits next)."""
    reward = np.asarray(rows.reward[:, env])
    returns = [reward[s + 1 : t + 1].sum() for s, t in episodes(rows, env)]
    if bool(state.is_last[env]):
        start = np.flatnonzero(np.asarray(rows.is_first[:, env]))[-1]
        returns.append(reward[start + 1 :].sum() + float(state.reward[env]))
    return returns


def check_house_return_mean(rows, state, n_envs, window=10):
    """``episodic_mean_return``: mean over envs of each env's rolling mean of
    its last ``window`` finished-episode returns."""
    per_env = [
        np.mean(finished_returns(rows, state, e)[-window:]) for e in range(n_envs)
    ]
    np.testing.assert_allclose(
        float(state.episodic_mean_return), np.mean(per_env), rtol=1e-5
    )


def check_last_flags(run):
    """``last_terminated`` / ``last_truncated`` after a tick describe the row
    the state emits next: terminal, and final but not terminal."""
    rows, state = run.rows, run.state
    terminal = np.concatenate(
        [np.asarray(rows.is_terminal[1:]), np.asarray(state.is_terminal)[None]]
    )
    last = np.concatenate(
        [np.asarray(rows.is_last[1:]), np.asarray(state.is_last)[None]]
    )
    np.testing.assert_array_equal(run.last_terminated, terminal.astype(np.float32))
    np.testing.assert_array_equal(
        run.last_truncated, (last & ~terminal).astype(np.float32)
    )


def _is_batch_shared_seed(path):
    """``info["rng"]``: the reset seed of Ajax's brax AutoResetWrapper, one
    per batch tiled per row, which advances when any env's episode ends
    (held or not). Playground's per-env keys live under another name."""
    return any(getattr(k, "key", None) == "rng" for k in path)


def held_state_is_bit_identical(env_states, rows, n_envs):
    """Across a final (held) row, the env's own state does not change at all."""
    last = np.asarray(rows.is_last)
    checked = 0
    leaves = [
        (path, np.asarray(leaf))
        for path, leaf in jax.tree_util.tree_leaves_with_path(env_states)
        if not _is_batch_shared_seed(path)
    ]
    for t, e in zip(*np.nonzero(last[:-1])):
        for _, leaf in leaves:
            if leaf.ndim > 1 and leaf.shape[1] == n_envs:
                np.testing.assert_array_equal(leaf[t + 1, e], leaf[t, e])
        checked += 1
    return checked


# ---------------------------------------------------------------------------
# gymnax
# ---------------------------------------------------------------------------


def push_right(carry, obs, is_first, key):
    """CartPole: always push right (falls in ~10 steps), count the calls."""
    del is_first, key
    return jnp.ones(obs.shape[0], jnp.int32), carry + 1, {"x": obs[:, 0]}


def test_dynamic_mode_on_terminating_gymnax_cartpole():
    n_envs, n_ticks = 3, 80
    env_args = env_args_for("CartPole-v1", n_envs)
    run = run_collector(
        env_args,
        push_right,
        n_ticks,
        reset_mode="dynamic",
        carry=jnp.zeros((), jnp.int32),
    )
    state, rows = run.state, run.rows
    check_row_conventions(rows)
    check_last_flags(run)
    obs = np.asarray(rows.obs)
    for e in range(n_envs):
        eps = episodes(rows, e)
        assert len(eps) >= 3
        for start, end in eps:
            # T steps -> T + 1 rows, every step rewarded 1 on CartPole
            assert (np.asarray(rows.reward[start + 1 : end + 1, e]) == 1).all()
            # a CartPole episode this short only ends by termination ...
            assert bool(rows.is_terminal[end, e])
            # ... and the final row shows the terminal obs, not the reset obs
            assert max(abs(obs[end, e, 0]) / 2.4, abs(obs[end, e, 2]) / 0.2095) >= 1
            assert np.abs(obs[start, e]).max() <= 0.05  # a fresh reset obs
        # consecutive episodes start from different (fresh) states
        starts = [obs[s, e] for s, _ in eps]
        assert all(not np.allclose(a, b) for a, b in zip(starts, starts[1:]))
        # the length of the last finished episode is tracked (its final row
        # may be the one the state emits next)
        start, end = eps[-1]
        if bool(state.is_last[e]):
            start, end = np.flatnonzero(np.asarray(rows.is_first[:, e]))[-1], n_ticks
        assert int(state.last_episode_length[e]) == end - start
    assert held_state_is_bit_identical(run.env_states, rows, n_envs) >= 3 * n_envs
    n_held = int(np.asarray(rows.is_last).sum())
    assert int(state.rows) == n_ticks * n_envs == int(state.timestep)
    assert int(state.env_steps) == n_ticks * n_envs - n_held
    assert int(state.n_offschedule_dones) == 0
    assert int(state.policy_carry) == n_ticks
    # the house rolling mean of the finished episodes' returns
    check_house_return_mean(rows, state, n_envs)
    # the step count of the current episode (0 once it has ended)
    for e in range(n_envs):
        start = np.flatnonzero(np.asarray(rows.is_first[:, e]))[-1]
        ended = bool(rows.is_last[-1, e]) or bool(state.is_last[e])
        expected = 0 if ended else n_ticks - start
        assert int(state.step_in_episode[e]) == expected


def count_since_first(carry, obs, is_first, key):
    """A stateful policy: per-env count of its calls since ``is_first``."""
    del key
    carry = jnp.where(is_first, 0, carry) + 1
    return jnp.ones(obs.shape[0], jnp.int32), carry, {"count": carry}


def test_a_stateful_policy_sees_is_first_on_every_reset_row():
    n_envs = 3
    env_args = env_args_for("CartPole-v1", n_envs)
    run = run_collector(
        env_args,
        count_since_first,
        60,
        reset_mode="dynamic",
        carry=jnp.zeros(n_envs, jnp.int32),
    )
    count = np.asarray(run.rows.extras["count"])
    for e in range(n_envs):
        eps = episodes(run.rows, e)
        assert len(eps) >= 3
        for start, end in eps:
            # reset at the first row, called on every row up to the final one
            np.testing.assert_array_equal(
                count[start : end + 1, e], np.arange(1, end - start + 2)
            )


def hold_half(carry, obs, is_first, key):
    del is_first, key
    return jnp.full((obs.shape[0], 1), 0.5), carry + 1, None


def test_static_mode_on_fixed_length_gymnax_pendulum():
    n_envs = 2
    env_args = env_args_for("Pendulum-v1", n_envs)
    T = agent_episode_length(env_args.env, env_args.env_params, 1)
    assert T == 200
    n_ticks = 2 * (T + 1) + 3
    run = run_collector(
        env_args,
        hold_half,
        n_ticks,
        reset_mode="static",
        episode_length=T,
        timestep_unit="env_steps",
        carry=jnp.zeros((), jnp.int32),
    )
    state, rows = run.state, run.rows
    check_row_conventions(rows)
    check_last_flags(run)
    first = np.asarray(rows.is_first).all(axis=1)
    last = np.asarray(rows.is_last).all(axis=1)
    np.testing.assert_array_equal(np.flatnonzero(first), [0, T + 1, 2 * (T + 1)])
    np.testing.assert_array_equal(np.flatnonzero(last), [T, 2 * T + 1])
    assert not np.asarray(rows.is_terminal).any()  # time limits only
    stepping = ~np.asarray(rows.is_first)
    assert (np.asarray(rows.reward)[stepping] < 0).all()
    np.testing.assert_array_equal(np.asarray(rows.action)[~last, :, 0], 0.5)
    assert int(state.n_offschedule_dones) == 0
    assert int(state.rows) == n_ticks * n_envs
    assert int(state.env_steps) == (n_ticks - 2) * n_envs == int(state.timestep)
    check_house_return_mean(rows, state, n_envs)
    np.testing.assert_array_equal(state.last_episode_length, T)
    np.testing.assert_array_equal(state.step_in_episode, 3)  # ticks 402..404
    # every episode starts from a fresh reset of its own: each env's
    # consecutive episodes differ, and so do the envs' initial states
    obs = np.asarray(rows.obs)
    starts = (0, T + 1, 2 * (T + 1))
    for e in range(n_envs):
        a, b, c = (obs[t, e] for t in starts)
        assert not np.allclose(a, b) and not np.allclose(b, c)
    for t in starts:
        assert not np.allclose(obs[t, 0], obs[t, 1])
    # the env receives the action mapped to Pendulum's [-2, 2] bounds:
    # replaying the first step with torque 1.0 reproduces row 1.
    env, params = env_args.env, env_args.env_params
    first_state = jax.tree.map(lambda x: x[0, 0], run.env_states)
    obs, *_ = env.step(jax.random.PRNGKey(0), first_state, jnp.array([1.0]), params)
    np.testing.assert_allclose(np.asarray(rows.obs[1, 0]), np.asarray(obs), atol=1e-6)


def test_per_env_params_map_each_action_to_its_own_systems_bounds():
    """Per-env (batched) gymnax params, as from a SystemClass: Pendulum with
    torque bounds [-1, 1], [-2, 2], [-3, 3] for envs 0, 1, 2."""
    n_envs, T = 3, 4
    env, params = build_env_from_id("Pendulum-v1")
    params = broadcast_env_params(params, n_envs).replace(
        max_torque=jnp.array([1.0, 2.0, 3.0])
    )
    env_args = EnvironmentConfig(
        env=env, env_params=params, n_envs=n_envs, continuous=True
    )
    run = run_collector(
        env_args,
        hold_half,
        2 * (T + 1),
        reset_mode="static",
        episode_length=T,
        carry=jnp.zeros((), jnp.int32),
    )
    check_row_conventions(run.rows)
    assert int(run.state.n_offschedule_dones) == 0
    # replaying env e's first step with torque 0.5 * max_torque[e]
    for e in range(n_envs):
        env_state = jax.tree.map(lambda x, e=e: x[0, e], run.env_states)
        env_params = jax.tree.map(lambda x, e=e: x[e], params)
        torque = jnp.array([0.5 * float(params.max_torque[e])])
        obs, *_ = env.step(jax.random.PRNGKey(0), env_state, torque, env_params)
        np.testing.assert_allclose(run.rows.obs[1, e], obs, atol=1e-6)


def test_static_mode_counts_offschedule_dones():
    """A terminating task in static mode: the env reports dones the schedule
    did not expect; they are counted (the caller raises on them)."""
    n_envs = 2
    env_args = env_args_for("CartPole-v1", n_envs)
    T = agent_episode_length(env_args.env, env_args.env_params, 1)
    run = run_collector(
        env_args,
        push_right,
        40,
        reset_mode="static",
        episode_length=T,
        carry=jnp.zeros((), jnp.int32),
    )
    assert not np.asarray(run.rows.is_last).any()  # schedule end not reached
    assert int(run.state.n_offschedule_dones) >= n_envs


def sentinel_policy(continuous):
    def policy(carry, obs, is_first, key):
        del is_first, key
        n = obs.shape[0]
        action = jnp.full((n, 1), 0.123) if continuous else jnp.ones(n, jnp.int32)
        return action, carry + 1, {"seen": jnp.ones(n)}

    return policy


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_random_phase_acts_uniformly_and_keeps_the_carry(env_id):
    continuous = env_id == "Pendulum-v1"
    n_envs, n_ticks, n_random = 4, 30, 20
    env_args = env_args_for(env_id, n_envs)
    run = run_collector(
        env_args,
        sentinel_policy(continuous),
        n_ticks,
        reset_mode="dynamic",
        random_phase=lambda tick: tick < n_random,
        carry=jnp.zeros((), jnp.int32),
    )
    state, rows = run.state, run.rows
    actions = np.asarray(rows.action)
    random_rows = actions[:n_random][~np.asarray(rows.is_last[:n_random])]
    if continuous:
        # ~ U[-1, 1] (not, e.g., U[0, 1)): in range, spread over both halves
        assert np.abs(random_rows).max() <= 1.0
        assert random_rows.min() < -0.5 and random_rows.max() > 0.5
        assert abs(random_rows.mean()) < 0.2
        assert not np.isclose(random_rows, 0.123).any()
        np.testing.assert_allclose(actions[n_random:], 0.123)
    else:
        assert set(np.unique(random_rows)) == {0, 1}
    # the policy (and its carry) only ran outside the random phase
    assert int(state.policy_carry) == n_ticks - n_random
    np.testing.assert_array_equal(np.asarray(rows.extras["seen"][:n_random]), 0)
    np.testing.assert_array_equal(np.asarray(rows.extras["seen"][n_random:]), 1)


def test_collector_vmaps_over_seeds():
    n_envs = 2
    env_args = env_args_for("CartPole-v1", n_envs)

    def run(seed):
        state = init_row_collector_state(seed, env_args, policy_carry=jnp.int32(0))

        def body(state, tick):
            return collect_row(
                state, tick, push_right, env_args=env_args, reset_mode="dynamic"
            )

        return jax.lax.scan(body, state, jnp.arange(30))[1]

    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    batched = jax.jit(jax.vmap(run))(keys)
    for i in range(2):
        single = jax.jit(run)(keys[i])
        np.testing.assert_array_equal(batched.is_last[i], single.is_last)
        np.testing.assert_allclose(batched.obs[i], single.obs, atol=1e-6)


def test_static_hold_and_random_phase_stay_conds_under_seed_vmap():
    """The tick (hence the hold and random-phase predicates) is unbatched, so
    both stay real ``cond``s when the state is batched across seeds: the
    fresh reset and the policy run only on the ticks that need them."""
    env_args = env_args_for("Pendulum-v1", 2)
    T = 200
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    states = jax.vmap(
        lambda k: init_row_collector_state(k, env_args, policy_carry=jnp.int32(0))
    )(keys)

    def tick_fn(state, tick):
        return collect_row(
            state,
            tick,
            hold_half,
            env_args=env_args,
            reset_mode="static",
            episode_length=T,
            random_phase=tick < 3,
        )

    jaxpr = jax.make_jaxpr(jax.vmap(tick_fn, in_axes=(0, None)))(states, jnp.int32(5))
    conds = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "cond"]
    assert len(conds) == 2


def test_collect_row_validates_its_configuration():
    env_args = env_args_for("CartPole-v1", 1)
    state = init_row_collector_state(jax.random.PRNGKey(0), env_args, jnp.int32(0))
    with pytest.raises(ValueError, match="reset_mode"):
        collect_row(state, 0, push_right, env_args=env_args, reset_mode="lockstep")
    with pytest.raises(ValueError, match="episode_length"):
        collect_row(state, 0, push_right, env_args=env_args, reset_mode="static")
    with pytest.raises(ValueError, match="timestep_unit"):
        collect_row(
            state,
            0,
            push_right,
            env_args=env_args,
            reset_mode="dynamic",
            timestep_unit="frames",
        )


@pytest.mark.parametrize(
    "env_id, build_kwargs",
    [("CartPole-v1", {}), ("fast", {"episode_length": 10})],
)
def test_the_collector_refuses_a_normalising_env(env_id, build_kwargs):
    """Rows and the house return hold raw rewards and observations."""
    env, env_params, _, continuous = prepare_env(
        env_id,
        n_envs=2,
        normalize_obs=True,
        normalize_reward=True,
        gamma=0.99,
        **build_kwargs,
    )
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=2, continuous=continuous
    )
    with pytest.raises(ValueError, match="normalisation"):
        init_row_collector_state(jax.random.PRNGKey(0), env_args)


# ---------------------------------------------------------------------------
# brax / mujoco_playground
# ---------------------------------------------------------------------------


def uniform_policy(carry, obs, is_first, key):
    del is_first
    action = jax.random.uniform(key, (obs.shape[0], 1), minval=-1.0, maxval=1.0)
    return action, carry, None


def test_dynamic_mode_on_brax_inverted_pendulum():
    n_envs, episode_length, n_ticks = 4, 20, 300
    env_args = env_args_for("inverted_pendulum", n_envs, episode_length=episode_length)
    run = run_collector(env_args, uniform_policy, n_ticks, reset_mode="dynamic")
    state, rows = run.state, run.rows
    check_row_conventions(rows)
    check_last_flags(run)
    obs = np.asarray(rows.obs)
    n_episodes = 0
    for e in range(n_envs):
        eps = episodes(rows, e)
        for start, end in eps:
            # an end before the time limit is a termination; a time-limit end
            # is a truncation (unless the pole fell on that very step)
            terminal = bool(rows.is_terminal[end, e])
            assert terminal or end - start == episode_length
            if end - start < episode_length:
                assert terminal
            if terminal:
                assert abs(obs[end, e, 1]) > 0.2  # the pole fell: terminal obs
        # every episode of an env starts from a fresh initial state: a held
        # env must not roll back the batch's reset seed (which would redraw
        # initial states already used)
        starts = np.stack([obs[s, e] for s, _ in eps])
        assert len(np.unique(starts, axis=0)) == len(starts)
        n_episodes += len(eps)
    assert n_episodes >= 10 * n_envs
    # the batch-shared reset seed stays one seed for the whole batch
    seed_rows = np.asarray(run.env_states.info["rng"])
    np.testing.assert_array_equal(
        seed_rows, np.broadcast_to(seed_rows[:, :1], seed_rows.shape)
    )
    assert (
        held_state_is_bit_identical(run.env_states, rows, n_envs) >= n_episodes - n_envs
    )
    assert int(state.n_offschedule_dones) == 0


def test_dynamic_mode_time_limit_ends_are_not_terminal():
    """brax "fast" never terminates: every episode ends at the time limit,
    a final row that is not terminal, T + 1 rows per episode."""
    n_envs, T = 2, 4
    env_args = env_args_for("fast", n_envs, episode_length=T)
    run = run_collector(env_args, uniform_policy, 3 * (T + 1) + 1, reset_mode="dynamic")
    check_row_conventions(run.rows)
    check_last_flags(run)
    for e in range(n_envs):
        eps = episodes(run.rows, e)
        assert len(eps) == 3
        for start, end in eps:
            assert end - start == T
            assert bool(run.rows.is_last[end, e])
            assert not bool(run.rows.is_terminal[end, e])
    assert np.asarray(run.last_truncated).sum() == 3 * n_envs


@requires_playground
def test_static_mode_on_playground_gives_fresh_episodes():
    """With ``fresh_reset=False`` playground's auto-reset returns a cached
    first state; the static mode still re-randomises every episode with its
    own fresh reset on the held tick."""
    n_envs, action_repeat = 2, 2
    env_args = env_args_for(
        "CartpoleBalance",
        n_envs,
        episode_length=6,
        action_repeat=action_repeat,
        fresh_reset=False,
    )
    T = agent_episode_length(env_args.env, None, action_repeat)
    assert T == 3
    run = run_collector(
        env_args,
        uniform_policy,
        3 * (T + 1),
        reset_mode="static",
        episode_length=T,
        timestep_unit="env_steps",
    )
    state, rows = run.state, run.rows
    check_row_conventions(rows)
    check_last_flags(run)
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(rows.is_first).all(axis=1)), [0, 4, 8]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(rows.is_last).all(axis=1)), [3, 7, 11]
    )
    assert not np.asarray(rows.is_terminal).any()
    assert int(state.n_offschedule_dones) == 0
    assert int(state.env_steps) == 3 * T * n_envs == int(state.timestep)
    obs = np.asarray(rows.obs)
    for e in range(n_envs):
        a, b, c = obs[0, e], obs[4, e], obs[8, e]
        assert not np.allclose(a, b) and not np.allclose(b, c)
        # final rows show the terminal obs, not the cached first obs
        assert not np.allclose(obs[3, e], a)
    # one agent step = 2 simulator steps of reward <= 1 each, summed
    assert np.asarray(rows.reward).max() > 1.0


@requires_playground
def test_dynamic_mode_on_playground_warns_about_the_cached_reset():
    env_args = env_args_for("CartpoleBalance", 1, episode_length=4, fresh_reset=False)
    state = init_row_collector_state(jax.random.PRNGKey(0), env_args)
    with pytest.warns(UserWarning, match="cached first state"):
        jax.eval_shape(
            lambda s: collect_row(
                s, 0, uniform_policy, env_args=env_args, reset_mode="dynamic"
            ),
            state,
        )


@requires_playground
def test_dynamic_mode_on_a_fresh_reset_playground_env_does_not_warn():
    env_args = env_args_for("CartpoleBalance", 1, episode_length=4)
    state = init_row_collector_state(jax.random.PRNGKey(0), env_args)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        jax.eval_shape(
            lambda s: collect_row(
                s, 0, uniform_policy, env_args=env_args, reset_mode="dynamic"
            ),
            state,
        )
