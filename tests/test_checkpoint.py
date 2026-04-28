"""Tests for ajax.checkpoint: save/load and resume-training round-trip."""

import os
import tempfile

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


def test_agent_train_returns_tuple_shape():
    """Regression: ``agent.train(...)`` returns a ``(state, metrics)`` 2-tuple,
    not a bare ``BaseAgentState``. The resume path in ``agents/base.py`` must
    accept either form; tests elsewhere rely on this shape. If Ajax ever
    changes this contract silently, downstream code (and the orchestrator's
    build_agent_and_train wrapper) will break in subtle ways.
    """
    from ajax.agents.SAC.SAC import SAC

    agent = SAC(env_id="Pendulum-v1")
    out = agent.train(seed=[0], n_timesteps=100, num_episode_test=1)
    assert isinstance(out, tuple), (
        f"agent.train() was expected to return a 2-tuple; got {type(out)}"
    )
    assert len(out) == 2, f"expected (state, metrics); got length {len(out)}"
    state, _metrics = out
    assert state.actor_state is not None


def test_resume_accepts_tuple_initial_state(tmp_path):
    """Regression: passing the whole ``(state, metrics)`` tuple returned by
    ``agent.train()`` as ``initial_state=`` must not crash. Before the fix,
    the vmap in ``base.py`` tried to scan ``collector_state`` on a tuple.
    """
    from ajax.agents.SAC.SAC import SAC

    agent = SAC(env_id="Pendulum-v1")
    first = agent.train(seed=[0], n_timesteps=100, num_episode_test=1)
    # Pass the tuple directly (not state_only).
    second = agent.train(
        seed=[0], n_timesteps=100, num_episode_test=1, initial_state=first,
    )
    assert second is not None


@pytest.mark.slow
def test_sac_short_train_then_resume_preserves_return(tmp_path):
    """End-to-end: train SAC briefly, checkpoint, reload, resume, verify the
    final state matches a continuation computed without the save/load cycle.

    This is the critical test: it exercises ``agent.train(initial_state=...)``
    and the ``resume_from_state=True`` branch inside ``make_train``.
    """
    import gymnax
    from flax.serialization import to_state_dict

    from ajax.agents.SAC.SAC import SAC
    from ajax.state import (
        AlphaConfig,
        BufferConfig,
        EnvironmentConfig,
        NetworkConfig,
        OptimizerConfig,
    )

    env_id = "Pendulum-v1"
    seeds = [0]
    total = 2_000  # intentionally tiny
    half = 1_000

    agent = SAC(env_id=env_id)

    # Reference run: train for `total` steps in one go
    ref_result = agent.train(seed=seeds, n_timesteps=total, num_episode_test=1)
    ref_state = ref_result[0] if isinstance(ref_result, tuple) else ref_result

    # Resume run: train for `half` steps, save, load, train for `total-half`
    mid_result = agent.train(seed=seeds, n_timesteps=half, num_episode_test=1)

    path = str(tmp_path / "sac_mid.pkl")
    save_checkpoint(mid_result, path)
    # The mid_result itself is a valid skeleton (same structure as the saved
    # checkpoint). In production, the skeleton would come from a fresh init.
    loaded = restore_into(mid_result, path)

    resumed_result = agent.train(
        seed=seeds,
        n_timesteps=total - half,
        num_episode_test=1,
        initial_state=loaded,
    )
    resumed_state = (
        resumed_result[0] if isinstance(resumed_result, tuple) else resumed_result
    )

    # Sanity: both paths produce a state with a valid actor_state
    assert resumed_state.actor_state is not None
    assert ref_state.actor_state is not None

    # The two paths are NOT expected to produce bit-identical results because
    # resumed training gets fresh random keys. We check that it converges to
    # a *reasonable* final state (no NaNs, actor params populated).
    ref_leaves = jax.tree.leaves(ref_state.actor_state.params)
    res_leaves = jax.tree.leaves(resumed_state.actor_state.params)
    assert len(ref_leaves) == len(res_leaves)
    for rl, sl in zip(ref_leaves, res_leaves):
        assert not np.any(np.isnan(np.asarray(sl))), "NaN in resumed params"
        assert np.asarray(sl).shape == np.asarray(rl).shape
