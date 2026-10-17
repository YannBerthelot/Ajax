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


def test_a_replay_buffer_saved_without_raw_next_obs_restores_with_the_next_rows(
    tmp_path,
):
    """A checkpoint from before the replay buffer stored ``raw_next_obs``
    (kept when an extension carries its own expert) resumes with each row's
    raw successor there, what its replay read then."""
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
    rows = {k: v for k, v in buffer.experience.items() if k != "raw_next_obs"}
    old = state.collector_state.replace(buffer_state=buffer.replace(experience=rows))
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state.replace(collector_state=old), path)
    got = restore_into(state, path).collector_state.buffer_state.experience
    raw = np.asarray(rows["raw_obs"])
    np.testing.assert_array_equal(got["raw_next_obs"][..., :-1, :], raw[..., 1:, :])
    np.testing.assert_array_equal(got["raw_next_obs"][..., -1, :], raw[..., 0, :])


def test_a_replay_buffer_saved_with_fields_no_reader_needs_resumes(sac_run, tmp_path):
    """A checkpoint from before the replay stored raw_obs and is_expert only
    for an expert restores with them, and the resumed run keeps writing
    every field its buffer holds."""
    agent, (state, metrics) = sac_run.agent, sac_run.out
    buffer = state.collector_state.buffer_state
    assert "raw_obs" not in buffer.experience  # no expert, no reader
    rows = dict(buffer.experience)
    rows["raw_obs"] = np.ones_like(rows["obs"])
    rows["is_expert"] = np.zeros_like(rows["reward"])
    old = state.collector_state.replace(buffer_state=buffer.replace(experience=rows))
    path = str(tmp_path / "old.pkl")
    save_checkpoint((state.replace(collector_state=old), metrics), path)
    restored = restore_into(sac_run.out, path)
    assert {"raw_obs", "is_expert"} <= set(
        restored[0].collector_state.buffer_state.experience
    )
    resumed, _ = agent.train(
        seed=[0], n_timesteps=40, num_episode_test=1, initial_state=restored
    )
    raw = np.asarray(resumed.collector_state.buffer_state.experience["raw_obs"])
    assert np.isfinite(raw).all() and (raw != 1.0).any()  # new rows written


def test_agent_train_returns_tuple_shape(sac_run):
    """Regression: ``agent.train(...)`` returns a ``(state, metrics)`` 2-tuple,
    not a bare ``BaseAgentState``. The resume path in ``agents/base.py`` must
    accept either form; tests elsewhere rely on this shape. If Ajax ever
    changes this contract silently, downstream code (and the orchestrator's
    build_agent_and_train wrapper) will break in subtle ways.
    """
    out = sac_run.out
    assert isinstance(out, tuple), (
        f"agent.train() was expected to return a 2-tuple; got {type(out)}"
    )
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


def test_a_memory_checkpoint_restores_only_into_its_own_final_norm(tmp_path):
    """A transformer network saved before its core ended with a LayerNorm
    lacks ln_f (mamba's is named alike): restoring it into the new network
    raises an error naming final_norm=False, and it restores into the
    network built so. A checkpoint with ln_f restoring into that network
    raises too: flax would drop the norm silently."""
    import dataclasses

    from ajax.agents.PPO.PPO import PPO
    from ajax.networks.memory import MemoryConfig

    memory = MemoryConfig(kind="transformer", hidden_size=8, num_heads=2, window=4)
    net = ("8", "relu")
    kw = {"n_envs": 2, "n_steps": 8, "batch_size": 8}
    kw |= {"actor_architecture": net, "critic_architecture": net}

    def skeleton(memory):
        agent = PPO(env_id="CartPole-v1", memory=memory, **kw)
        return agent.train(seed=0, n_timesteps=0)[0]

    old = skeleton(dataclasses.replace(memory, final_norm=False))
    path = str(tmp_path / "old.pkl")
    save_checkpoint(old, path)
    with pytest.raises(ValueError, match=r"final_norm=False"):
        restore_into(skeleton(memory), path)
    restored = restore_into(old, path)
    same = jax.tree.map(np.array_equal, restored.actor_state, old.actor_state)
    assert all(jax.tree.leaves(same))

    new_path = str(tmp_path / "new.pkl")
    save_checkpoint(skeleton(memory), new_path)
    with pytest.raises(ValueError, match=r"final_norm=True"):
        restore_into(old, new_path)


def test_a_ppo_checkpoint_restores_only_into_its_own_log_std_head(tmp_path):
    """A PPO saved with the Dense log-std head, its default before, lacks
    log_std_param: restoring it into the default network raises an error
    naming the flags that rebuild its own, into which it restores. A default
    checkpoint restoring into the Dense head raises too: flax would drop
    the parameter silently."""
    from ajax.agents.PPO.PPO import PPO

    net = ("8", "relu")
    kw = {"n_envs": 2, "n_steps": 8, "batch_size": 8}
    kw |= {"actor_architecture": net, "critic_architecture": net}

    def skeleton(**head):
        return PPO(env_id="Pendulum-v1", **kw, **head).train(seed=0, n_timesteps=0)[0]

    old = skeleton(log_std_state_independent=False, log_std_init=-1.0)
    path = str(tmp_path / "old.pkl")
    save_checkpoint(old, path)
    flags = r"log_std_state_independent=False and\s+log_std_init=-1\.0"
    with pytest.raises(ValueError, match=flags):
        restore_into(skeleton(), path)
    restored = restore_into(old, path)
    same = jax.tree.map(np.array_equal, restored.actor_state, old.actor_state)
    assert all(jax.tree.leaves(same))

    new_path = str(tmp_path / "new.pkl")
    save_checkpoint(skeleton(), new_path)
    with pytest.raises(ValueError, match=r"log_std_state_independent=True"):
        restore_into(old, new_path)


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


def test_a_checkpoint_missing_a_data_field_still_fails(sac_run, tmp_path):
    """Only fields the skeleton holds empty (None) are filled on restore: a
    checkpoint missing a field with data is a mismatch and raises."""
    import pickle

    state = sac_run.out[0]
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
    observation and statistics. PPO normalises through the env (a replay
    agent normalises observations itself)."""
    import pickle

    from ajax.agents.PPO.PPO import PPO

    net = ("8", "relu")
    agent = PPO(
        env_id="fast",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        n_steps=8,
        batch_size=8,
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


def _without_format(path: str) -> None:
    """Rewrite the checkpoint at ``path`` as one written before checkpoints
    stated their format."""
    import pickle

    from ajax.checkpoint import FORMAT_KEY

    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    assert saved.pop(FORMAT_KEY) == 1
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)


@pytest.mark.parametrize("normalised", ["rewards", "observations"])
def test_a_replay_checkpoint_from_before_raw_replay_is_refused(tmp_path, normalised):
    """A replay agent's checkpoint from before its replay kept the env's own
    observations and rewards holds rows normalised with statistics that
    are gone: refused, saying why, when the run normalised them. Without
    normalisation an old checkpoint restores, and a new one always does."""
    from ajax.agents.SAC.SAC import SAC

    net = ("8", "relu")
    kw = {"n_envs": 2, "actor_architecture": net, "critic_architecture": net}
    kw |= {"learning_starts": 100, "batch_size": 4, "buffer_size": 40}
    agent = SAC(env_id="Pendulum-v1", normalize_rewards=normalised == "rewards", **kw)
    state, _ = agent.train(seed=0, n_timesteps=16)
    path = str(tmp_path / "new.pkl")
    save_checkpoint(state, path)
    restore_into(state, path)
    if normalised == "observations":
        # The env's observation statistics, as an old SAC run saved them.
        import pickle

        with open(path, "rb") as fh:
            saved = pickle.load(fh)
        env = saved["collector_state"]["_env_state"]
        saved["collector_state"]["_env_state"] = {
            "normalization_info": {"obs": {"mean": np.zeros(3)}},
            **env,
        }
        with open(path, "wb") as fh:
            pickle.dump(saved, fh)
    _without_format(path)
    with pytest.raises(ValueError, match="before a replay agent kept"):
        restore_into(state, path)
    plain, _ = SAC(env_id="Pendulum-v1", **kw).train(seed=0, n_timesteps=16)
    save_checkpoint(plain, path)
    _without_format(path)
    restore_into(plain, path)
