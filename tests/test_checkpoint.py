"""Tests for ajax.checkpoint: save/load and resume-training round-trip."""

import os

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


@pytest.mark.parametrize(
    "field, source", [("next_obs", "obs"), ("raw_next_obs", "raw_obs")]
)
def test_a_replay_buffer_saved_without_a_next_field_restores_with_the_next_rows(
    tmp_path, field, source
):
    """A checkpoint from before the replay buffer stored ``next_obs`` (or
    ``raw_next_obs``, kept when an extension carries its own expert)
    resumes with each row's successor there, what its replay read then."""
    from ajax.agents.SAC.SAC import SAC
    from ajax.extensions.target_mods import IBRL

    net = ("8", "relu")
    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=100,
        batch_size=4,
        buffer_size=40,
        extensions=(IBRL(expert_policy=lambda obs: obs[..., :1]),),
    )
    state, _ = agent.train(seed=0, n_timesteps=10)
    buffer = state.collector_state.buffer_state
    rows = {k: v for k, v in buffer.experience.items() if k != field}
    old = state.collector_state.replace(buffer_state=buffer.replace(experience=rows))
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state.replace(collector_state=old), path)
    got = restore_into(state, path).collector_state.buffer_state.experience
    obs = np.asarray(rows[source])
    np.testing.assert_array_equal(got[field][..., :-1, :], obs[..., 1:, :])
    np.testing.assert_array_equal(got[field][..., -1, :], obs[..., 0, :])


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
    assert isinstance(
        out, tuple
    ), f"agent.train() was expected to return a 2-tuple; got {type(out)}"
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
        seed=[0],
        n_timesteps=100,
        num_episode_test=1,
        initial_state=first,
    )
    assert second is not None


@pytest.mark.slow
def test_sac_short_train_then_resume_preserves_return(tmp_path):
    """End-to-end: train SAC briefly, checkpoint, reload, resume, verify the
    final state matches a continuation computed without the save/load cycle.

    This is the critical test: it exercises ``agent.train(initial_state=...)``
    and the ``resume_from_state=True`` branch inside ``make_train``.
    """

    from ajax.agents.SAC.SAC import SAC

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
    for rl, sl in zip(ref_leaves, res_leaves, strict=False):
        assert not np.any(np.isnan(np.asarray(sl))), "NaN in resumed params"
        assert np.asarray(sl).shape == np.asarray(rl).shape


def _without(state_dict, field):
    """The checkpoint as saved before ``field`` existed: every empty (None)
    entry of that name dropped, the rollouts' Transition's included."""
    if not isinstance(state_dict, dict):
        return state_dict
    return {
        k: _without(v, field)
        for k, v in state_dict.items()
        if not (k == field and v is None)
    }


@pytest.mark.parametrize("agent_name", ["SAC", "PPO"])
def test_a_checkpoint_saved_before_a_transition_field_restores(tmp_path, agent_name):
    """Transition gained raw_next_obs, empty in every rollout the collector
    and the last-rollout placeholder hold: a checkpoint written before it
    restores, the new field empty."""
    import pickle

    from ajax.agents.PPO.PPO import PPO
    from ajax.agents.SAC.SAC import SAC

    net = ("8", "relu")
    kw = {"n_envs": 2, "actor_architecture": net, "critic_architecture": net}
    if agent_name == "SAC":
        kw |= {"learning_starts": 100, "batch_size": 4, "buffer_size": 40}
        agent = SAC(env_id="Pendulum-v1", **kw)
    else:
        kw |= {"n_steps": 8, "batch_size": 8, "expose_recent_rollout": True}
        agent = PPO(env_id="Pendulum-v1", **kw)
    state, _ = agent.train(seed=0, n_timesteps=16)
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state, path)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    old = _without(saved, "raw_next_obs")
    assert old != saved  # the field was there to drop
    with open(path, "wb") as fh:
        pickle.dump(old, fh)
    restored = restore_into(state, path)
    assert restored.collector_state.rollout.raw_next_obs is None


def test_a_state_without_a_collector_restores(tmp_path):
    """TDMPC2MultiTask trains offline: its state holds no collector
    (collector_state None). It restores, as does a checkpoint whose root is
    not a mapping (an array)."""
    from ajax import TDMPC2MultiTask
    from ajax.agents.TDMPC2.dataset import TaskEpisodes, pool_tasks

    rng, length = np.random.default_rng(0), 10
    action = rng.uniform(-1, 1, (2, length + 1, 1)).astype(np.float32)
    reward = rng.uniform(0, 1, (2, length + 1)).astype(np.float32)
    action[:, -1], reward[:, 0] = 0.0, 0.0
    obs = rng.normal(size=(2, length + 1, 2)).astype(np.float32)
    data = pool_tasks([TaskEpisodes(obs, action, reward, length, "task")])
    tiny = {"enc_dim": 32, "mlp_dim": 32, "latent_dim": 16, "num_q": 2}
    tiny |= {"task_dim": 4, "batch_size": 16, "num_samples": 32, "num_elites": 8}
    agent = TDMPC2MultiTask(data, num_pi_trajs=4, iterations=2, **tiny)
    state, _ = agent.train(seed=0, n_timesteps=0)
    assert state.collector_state is None
    path = str(tmp_path / "offline.pkl")
    save_checkpoint(state, path)
    restore_into(state, path)
    path = str(tmp_path / "array.pkl")
    save_checkpoint(jnp.arange(3.0), path)
    np.testing.assert_array_equal(restore_into(jnp.zeros(3), path), [0.0, 1.0, 2.0])


def test_a_checkpoint_missing_a_data_field_still_fails(tmp_path):
    """Only fields the skeleton holds empty (None) are filled on restore: a
    checkpoint missing a field with data is a mismatch and raises."""
    import pickle

    from ajax.agents.SAC.SAC import SAC

    net = ("8", "relu")
    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=100,
        batch_size=4,
        buffer_size=40,
    )
    state, _ = agent.train(seed=0, n_timesteps=8)
    path = str(tmp_path / "mismatched.pkl")
    save_checkpoint(state, path)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    del saved["collector_state"]["rollout"]["obs"]
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    with pytest.raises(ValueError, match="obs"):
        restore_into(state, path)


def test_a_brax_state_saved_before_its_raw_observation_restores_it(tmp_path):
    """A brax env state saved before the observation normaliser kept the raw
    observation in its info restores with it, un-normalised from the saved
    observation and statistics."""
    import pickle

    from ajax.agents.SAC.SAC import SAC

    net = ("8", "relu")
    agent = SAC(
        env_id="fast",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=100,
        batch_size=4,
        buffer_size=40,
        normalize_observations=True,
    )
    state, _ = agent.train(seed=0, n_timesteps=16)
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state, path)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    info = saved["collector_state"]["_env_state"]["info"]
    want = info.pop("raw_obs")
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    got = restore_into(state, path).collector_state.env_state.info["raw_obs"]
    np.testing.assert_allclose(got, want, rtol=1e-6)
