"""Transition datasets: whole episodes recorded from an expert or read from a
replay, saved and loaded, and added back to a replay.

On the identity task (tests/environments/identity_env.py) a row reads its
own place back: the observation is ``[t, c]``, ``t`` the step in the episode,
which terminates at ``t = 3`` when ``c > 0`` and is truncated at ``t = 5``.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from ajax import TD3
from ajax.buffers import dataset as ds
from ajax.buffers.utils import get_buffer, init_buffer
from tests.environments.identity_env import LIMIT, TERMINAL, IdentityEnv


def check_episodes(data: ds.TransitionDataset) -> None:
    """Each episode starts at t = 0, steps t by one and ends where the
    identity task ends it; inside one, next_obs is the next row's obs."""
    t, c = data.obs[:, 0], data.obs[:, 1]
    starts = np.r_[0, np.nonzero(np.diff(data.episode))[0] + 1]
    np.testing.assert_array_equal(t, np.arange(len(t)) - starts[data.episode])
    np.testing.assert_array_equal(data.terminated, (c > 0) & (t + 1 == TERMINAL))
    np.testing.assert_array_equal(data.truncated, (c <= 0) & (t + 1 == LIMIT))
    inside = ~(data.terminated | data.truncated)[:-1]
    np.testing.assert_array_equal(data.next_obs[:-1][inside], data.obs[1:][inside])


@pytest.fixture(scope="module")
def recorded() -> ds.TransitionDataset:
    """Two envs, 12 steps each, of the expert acting c (env units)."""
    env_args = TD3(IdentityEnv(), n_envs=2).env_args
    return ds.record(lambda obs: obs[1:], env_args, jax.random.PRNGKey(0), 12, 0.5)


def test_logged_rows_get_their_episodes_and_returns() -> None:
    ends, no = np.array([0, 0, 1, 0, 1], bool), np.zeros(5, bool)
    obs, action = np.zeros((5, 2)), np.zeros((5, 1))
    data = ds.TransitionDataset(obs, action, [1, 1, 1, 2, 2], ends, no, obs, gamma=0.5)
    np.testing.assert_array_equal(data.return_to_go, [1.75, 1.5, 1.0, 3.0, 2.0])
    np.testing.assert_array_equal(data.episode, [0, 0, 0, 1, 1])
    assert data.n_episodes == 2 and not data.is_expert.any()
    assert np.isnan(data.a_expert).all() and data.terminated.dtype == bool
    with pytest.raises(ValueError, match="whole episodes"):
        ds.TransitionDataset(obs, action, data.reward, ~ends, no, obs)
    with pytest.raises(ValueError, match="action has 4 rows"):
        ds.TransitionDataset(obs, action[:4], data.reward, ends, no, obs)


def test_record_keeps_each_envs_whole_episodes(recorded) -> None:
    check_episodes(recorded)
    # Each env's cut last episode is left out: 12 steps hold 2 to 4 whole.
    assert 4 <= recorded.n_episodes <= 8 and recorded.is_expert.all()
    c = recorded.obs[:, 1:]
    np.testing.assert_allclose(recorded.action, c / 2, atol=1e-6)  # agent units
    np.testing.assert_array_equal(recorded.a_expert, recorded.action)
    np.testing.assert_allclose(recorded.reward, c[:, 0], atol=1e-6)


def test_save_and_load_round_trip(recorded, tmp_path) -> None:
    path = str(tmp_path / "sub" / "data.npz")
    ds.save(recorded, path)
    back = ds.load(path)
    for f in ds.FIELDS:
        np.testing.assert_array_equal(getattr(back, f), getattr(recorded, f), f)
    assert (back.gamma, back.env) == (recorded.gamma, recorded.env)
    np.savez(path, meta=np.asarray('{"format": "other", "version": 1}'))
    with pytest.raises(ValueError, match="not a ajax.transition_dataset v1"):
        ds.load(path)


def replay(n_envs: int = 1, size: int = 64, **fields):
    buffer = get_buffer(size, 4, n_envs)
    return buffer, init_buffer(
        buffer, TD3(IdentityEnv(), n_envs=n_envs).env_args, **fields
    )


@pytest.mark.parametrize("raw", [True, False])
def test_fill_then_export_gives_the_dataset_back(recorded, raw) -> None:
    keep = {
        "include_raw_obs": raw,
        "include_is_expert": raw,
        "include_raw_next_obs": raw,
    }
    buffer, empty = replay(**keep)
    filled = ds.fill_replay(buffer, empty, recorded)
    state = SimpleNamespace(collector_state=SimpleNamespace(buffer_state=filled))
    back = ds.export(state, gamma=0.5)
    for f in ("obs", "action", "reward", "terminated", "truncated", "return_to_go"):
        np.testing.assert_array_equal(getattr(back, f), getattr(recorded, f), f)
    np.testing.assert_array_equal(back.is_expert, recorded.is_expert & raw)
    # Without raw_next_obs the replay keeps the reset observation after a
    # termination, as collection does: here the next episode's first.
    after = np.roll(recorded.obs, -1, 0)
    reset = np.where(recorded.terminated[:, None], after, recorded.next_obs)
    np.testing.assert_array_equal(back.next_obs, recorded.next_obs if raw else reset)


def test_export_reads_a_wrapped_replay_oldest_first() -> None:
    agent = TD3(
        IdentityEnv(), n_envs=2, buffer_size=20, batch_size=4, learning_starts=10**6
    )
    state, _ = agent.train(seed=[0, 1], n_timesteps=60)  # 30 rows per env: wrapped
    data = ds.export(state, seed_index=1)
    check_episodes(data)
    assert len(data.reward) <= 20 and data.n_episodes >= 2
    with pytest.raises(ValueError, match="seed_index"):
        ds.export(state)


def test_fill_refuses_a_replay_it_cannot_fill(recorded) -> None:
    n = len(recorded.reward)
    odd = next(k for k in (2, 3, 4, 5, 7) if n % k)
    for args, fields, match in (
        ({}, {"include_expert_fields": True}, "next_a_expert"),
        ({"n_envs": odd}, {}, "do not divide"),
        ({"size": n - 1}, {}, "a replay of"),
        ({}, {"add_train_frac": True}, r"obs rows \(3,\), dataset \(2,\)"),
    ):
        buffer, empty = replay(**args, **fields)
        with pytest.raises(ValueError, match=match):
            ds.fill_replay(buffer, empty, recorded)
