"""Tests for ajax.checkpoint: save/load and resume-training round-trip."""

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.checkpoint import (
    checkpoint_exists,
    restore_into,
    save_checkpoint,
)


def test_save_and_load_simple_pytree(tmp_path):
    """Basic pytree round-trip via skeleton."""
    state = {
        "a": jnp.array([1.0, 2.0, 3.0]),
        "b": {"c": jnp.array(7), "d": jnp.zeros((4, 4))},
        "e": 42,  # non-array leaf
    }
    path = str(tmp_path / "ckpt.pkl")

    assert not checkpoint_exists(path)
    save_checkpoint(state, path)
    assert checkpoint_exists(path)

    # Skeleton has the same structure; data will be overwritten by restore.
    skeleton = {
        "a": jnp.zeros((3,)),
        "b": {"c": jnp.array(0), "d": jnp.zeros((4, 4))},
        "e": 0,
    }
    loaded = restore_into(skeleton, path)
    assert jnp.allclose(loaded["a"], state["a"])
    assert jnp.allclose(loaded["b"]["c"], state["b"]["c"])
    assert jnp.allclose(loaded["b"]["d"], state["b"]["d"])


def test_save_overwrites_existing(tmp_path):
    path = str(tmp_path / "ckpt.pkl")
    save_checkpoint({"x": jnp.array(1.0)}, path)
    save_checkpoint({"x": jnp.array(2.0)}, path)
    skeleton = {"x": jnp.array(0.0)}
    loaded = restore_into(skeleton, path)
    assert jnp.allclose(loaded["x"], jnp.array(2.0))


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        restore_into({"x": jnp.array(0.0)}, str(tmp_path / "nonexistent.pkl"))


def test_flax_struct_dataclass_roundtrip(tmp_path):
    """Ensure flax.struct.dataclass types survive the pickle round-trip.

    Uses a real Ajax struct (Transition) so the test doesn't depend on a
    locally-defined class that pickle can't import back.
    """
    from ajax.state import Transition

    state = Transition(
        obs=jnp.ones((4, 3)),
        action=jnp.zeros((4, 2)),
        reward=jnp.array([1.0, 2.0, 3.0, 4.0]),
        terminated=jnp.zeros((4,), dtype=bool),
        truncated=jnp.zeros((4,), dtype=bool),
        next_obs=jnp.ones((4, 3)),
    )
    path = str(tmp_path / "struct.pkl")
    save_checkpoint(state, path)
    skeleton = Transition(
        obs=jnp.zeros((4, 3)),
        action=jnp.zeros((4, 2)),
        reward=jnp.zeros((4,)),
        terminated=jnp.zeros((4,), dtype=bool),
        truncated=jnp.zeros((4,), dtype=bool),
        next_obs=jnp.zeros((4, 3)),
    )
    loaded = restore_into(skeleton, path)
    assert isinstance(loaded, Transition)
    assert jnp.allclose(loaded.obs, state.obs)
    assert jnp.allclose(loaded.action, state.action)
    assert jnp.allclose(loaded.reward, state.reward)


def test_atomic_save_tmp_cleanup(tmp_path):
    """After a successful save, the .tmp side file must not remain."""
    path = str(tmp_path / "ckpt.pkl")
    save_checkpoint({"x": jnp.array(1.0)}, path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


@pytest.fixture(scope="module")
def sac_run():
    """One tiny SAC run, shared: 40 transitions of 2 envs, past its learning
    starts (critic, policy, temperature), so every gradient update runs.
    ``train``'s output is kept on the host: a state passed as
    ``initial_state`` may be donated."""
    from ajax.agents.SAC.SAC import SAC

    net = ("8", "relu")
    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=10,
        policy_update_start=10,
        alpha_update_start=10,
        batch_size=4,
        buffer_size=40,
    )
    out = agent.train(seed=[0], n_timesteps=40, num_episode_test=1)
    return SimpleNamespace(agent=agent, out=jax.device_get(out))


def test_a_replay_buffer_saved_without_next_obs_restores_with_the_next_rows(
    sac_run, tmp_path
):
    """A checkpoint from before the replay buffer stored ``next_obs`` resumes
    with each row's successor there, what its replay bootstrapped on then."""
    state = sac_run.out[0]
    buffer = state.collector_state.buffer_state
    rows = {k: v for k, v in buffer.experience.items() if k != "next_obs"}
    old = state.collector_state.replace(buffer_state=buffer.replace(experience=rows))
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state.replace(collector_state=old), path)
    got = restore_into(state, path).collector_state.buffer_state.experience
    obs = np.asarray(rows["obs"])
    np.testing.assert_array_equal(got["next_obs"][..., :-1, :], obs[..., 1:, :])
    np.testing.assert_array_equal(got["next_obs"][..., -1, :], obs[..., 0, :])


def test_agent_train_returns_tuple_shape(sac_run):
    """Regression: ``agent.train(...)`` returns a ``(state, metrics)`` 2-tuple,
    not a bare ``BaseAgentState``. The resume path in ``agents/base.py`` must
    accept either form; tests elsewhere rely on this shape. If Ajax ever
    changes this contract silently, downstream code (and the orchestrator's
    build_agent_and_train wrapper) will break in subtle ways.
    """
    out = sac_run.out
    assert isinstance(
        out, tuple
    ), f"agent.train() was expected to return a 2-tuple; got {type(out)}"
    assert len(out) == 2, f"expected (state, metrics); got length {len(out)}"
    state, _metrics = out
    assert state.actor_state is not None


def test_sac_resumes_from_a_restored_checkpoint_of_its_train_tuple(sac_run, tmp_path):
    """End-to-end, and the critical test of ``agent.train(initial_state=...)``
    and the ``resume_from_state=True`` branch of ``make_train``: the
    ``(state, metrics)`` tuple ``train`` returned is saved, restored into
    itself unchanged, and passed whole as ``initial_state`` (regression:
    the vmap in ``base.py`` scanned ``collector_state`` on a tuple). The
    resumed run continues the timestep and keeps updating: the critic's
    step count goes on and the policy's parameters move, finite and of
    their shapes. (A resumed run draws fresh keys, so
    it is not compared with an uninterrupted one.)"""
    agent, first = sac_run.agent, sac_run.out
    path = str(tmp_path / "sac.pkl")
    save_checkpoint(first, path)
    restored = restore_into(first, path)
    jax.tree.map(np.testing.assert_array_equal, restored, first)
    resumed, _ = agent.train(
        seed=[0], n_timesteps=40, num_episode_test=1, initial_state=restored
    )
    np.testing.assert_array_equal(
        resumed.collector_state.timestep, 2 * first[0].collector_state.timestep
    )
    assert np.all(resumed.critic_state.step > first[0].critic_state.step)
    before = jax.tree.leaves(first[0].actor_state.params)
    after = jax.tree.leaves(resumed.actor_state.params)
    assert len(after) == len(before)
    for ours, theirs in zip(after, before, strict=True):
        assert np.shape(ours) == np.shape(theirs)
        assert np.all(np.isfinite(ours)), "NaN in resumed params"
    assert any(not np.array_equal(a, b) for a, b in zip(after, before, strict=True))
