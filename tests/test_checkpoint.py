"""Tests for ajax.checkpoint: save/load and resume-training round-trip."""

import os
from types import SimpleNamespace
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

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
        kw.update({"learning_starts": 100, "batch_size": 4, "buffer_size": 40})
        agent = SAC(env_id="Pendulum-v1", **kw)
    else:
        kw.update({"n_steps": 8, "batch_size": 8, "expose_recent_rollout": True})
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
    kw.update({"actor_architecture": net, "critic_architecture": net})

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
    kw.update({"actor_architecture": net, "critic_architecture": net})

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


@pytest.mark.parametrize("agent_name", ["PPO", "APO"])
def test_an_optimiser_state_saved_behind_its_own_clip_restores(tmp_path, agent_name):
    """PPO's and APO's networks each clipped their own gradient
    (get_adam_tx's chain, whose clip holds an empty state) until each took
    its reference code's one clip over both. A checkpoint saved then
    restores into today's unclipped Adams, every moment and step count
    kept."""
    import optax

    import ajax

    agent_cls = getattr(ajax, agent_name)
    agent = agent_cls(env_id="Pendulum-v1", n_envs=2, n_steps=8, batch_size=8)
    skeleton = agent.train(seed=0, n_timesteps=0)[0]
    moved = {  # moments and counts the skeleton does not hold
        name: jax.tree.map(lambda x: x + 1, getattr(skeleton, name).opt_state)
        for name in ("actor_state", "critic_state")
    }
    old = skeleton.replace(
        **{
            name: getattr(skeleton, name).replace(opt_state=(optax.EmptyState(), s))
            for name, s in moved.items()
        }
    )
    path = str(tmp_path / "clipped.pkl")
    save_checkpoint(old, path)
    restored = restore_into(skeleton, path)
    for name, opt_state in moved.items():
        same = jax.tree.map(
            np.array_equal, getattr(restored, name).opt_state, opt_state
        )
        assert all(jax.tree.leaves(same)), name


def test_another_agents_optimiser_saved_behind_a_clip_is_refused(sac_run, tmp_path):
    """The clip shim is PPO's and APO's, whose clip moved over both
    networks: an SAC checkpoint whose actor optimiser state sits behind a
    clip's empty state is refused, as any other optimiser layout is."""
    import optax

    state = sac_run.out[0]
    actor = state.actor_state
    old = state.replace(
        actor_state=actor.replace(opt_state=(optax.EmptyState(), actor.opt_state))
    )
    path = str(tmp_path / "clipped.pkl")
    save_checkpoint(old, path)
    with pytest.raises(ValueError):
        restore_into(state, path)


def test_an_apo_checkpoint_restores_only_into_its_own_hidden_widths(tmp_path):
    """APO's hidden layers were two of 128 units before its reference code's
    two of 64. Its checkpoint saved then is refused by the default network
    (the leaf-shape check, test_a_checkpoint_of_another_shape_is_refused),
    the error naming the architecture that rebuilds its own, into which it
    restores."""
    from ajax.agents.APO.APO import APO

    kw = {"n_envs": 2, "n_steps": 8, "batch_size": 8}
    wide = ("128", "tanh", "128", "tanh")

    def skeleton(**net):
        return APO(env_id="Pendulum-v1", **kw, **net).train(seed=0, n_timesteps=0)[0]

    old = skeleton(actor_architecture=wide, critic_architecture=wide)
    path = str(tmp_path / "old.pkl")
    save_checkpoint(old, path)
    shapes = r"\['layers_0'\]\['bias'\] is \(1, 128\) in the checkpoint, \(1, 64\)"
    with pytest.raises(ValueError, match=r"does not fit.*" + shapes) as refused:
        restore_into(skeleton(), path)
    assert 'actor_architecture=("128", "tanh", "128", "tanh")' in str(refused.value)
    restored = restore_into(old, path)
    same = jax.tree.map(np.array_equal, restored.actor_state, old.actor_state)
    assert all(jax.tree.leaves(same))


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
    tiny.update({"task_dim": 4, "batch_size": 16, "num_samples": 32, "num_elites": 8})
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


@struct.dataclass
class _WithExtensions:
    params: jax.Array
    ext_state: tuple = ()


def test_a_checkpoint_saved_before_a_stateless_extension_joined_restores(tmp_path):
    """A stateless extension appended to a stack (SAC's expert prefill, as
    ExpertPrefill) has an empty entry in ext_state: a checkpoint saved
    without it restores. An entry holding data stays required."""
    path = str(tmp_path / "before.pkl")
    save_checkpoint(_WithExtensions(jnp.ones(2)), path)
    restored = restore_into(_WithExtensions(jnp.zeros(2), ext_state=((),)), path)
    assert restored.ext_state == ((),)
    np.testing.assert_array_equal(restored.params, 1.0)
    with pytest.raises(ValueError, match="ext_state"):
        restore_into(_WithExtensions(jnp.zeros(2), ext_state=(jnp.zeros(1),)), path)


@struct.dataclass
class _Slim:
    """A state class that deleted its field ``gone``."""

    removed_fields: ClassVar[frozenset[str]] = frozenset({"gone"})
    kept: jax.Array


class _Pair(NamedTuple):
    slim: _Slim
    other: jax.Array


@struct.dataclass
class _Holder:
    """A state class that deleted ``old``, holding classes that deleted
    theirs: directly, in a NamedTuple, a list and a dict."""

    removed_fields: ClassVar[frozenset[str]] = frozenset({"old"})
    slim: _Slim
    pair: _Pair
    slims: list
    by_name: dict


def test_a_checkpoint_holding_fields_their_classes_removed_restores(tmp_path):
    """The fields a state class lists in ``removed_fields`` are dropped on
    restore, wherever the class sits, and the rest restores into a skeleton
    of zeros. A field its class does not list still fails: ``gone`` on the
    holder, which lists only ``old``."""
    import pickle

    rng = np.random.default_rng(0)

    def slim():
        return _Slim(kept=jnp.asarray(rng.normal(size=2)))

    state = _Holder(
        slim=slim(),
        pair=_Pair(slim(), jnp.asarray(rng.normal(size=1))),
        slims=[slim()],
        by_name={"a": slim()},
    )
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state, path)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    saved["old"] = np.ones(3)
    for held in (saved["slim"], saved["pair"]["slim"], saved["slims"]["0"]):
        held["gone"] = np.ones(3)
    saved["by_name"]["a"]["gone"] = np.ones(3)
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    skeleton = jax.tree.map(jnp.zeros_like, state)
    restored = restore_into(skeleton, path)
    jax.tree.map(np.testing.assert_array_equal, restored, state)

    saved["gone"] = np.ones(3)
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    with pytest.raises(ValueError, match="Unknown field.*gone.*_Holder"):
        restore_into(skeleton, path)


@struct.dataclass
class _Grown:
    """A state class that gained its field ``new``."""

    added_fields: ClassVar[frozenset[str]] = frozenset({"new"})
    kept: jax.Array
    new: jax.Array


@struct.dataclass
class _Results:
    """A state class all of whose fields are new: an extension's state that
    took the place of the stateless ``()``."""

    added_fields: ClassVar[frozenset[str]] = frozenset({"first", "last"})
    first: jax.Array
    last: jax.Array


def test_a_checkpoint_saved_before_fields_were_added_reads_them_as_nan(tmp_path):
    """The fields a state class lists in ``added_fields`` restore as NaN, in
    the skeleton's shape, from a checkpoint saved without them, here inside
    a tuple of extension states. A missing field the class does not list
    still fails: ``kept``, when ``_Grown`` was saved as ``()``."""
    kept = jnp.asarray([1.0, 2.0])
    full = _Results(first=jnp.zeros(2), last=jnp.zeros(2))
    skeleton = {"ext_state": (_Grown(kept=jnp.zeros(2), new=jnp.zeros(2)), full)}
    path = str(tmp_path / "old.pkl")
    save_checkpoint({"ext_state": ({"kept": kept}, ())}, path)
    grown, results = restore_into(skeleton, path)["ext_state"]
    np.testing.assert_array_equal(grown.kept, kept)
    for new in (grown.new, results.first, results.last):
        assert new.shape == (2,) and np.isnan(new).all()

    save_checkpoint({"ext_state": ((), ())}, path)
    with pytest.raises(ValueError, match="Missing field kept"):
        restore_into(skeleton, path)


@struct.dataclass
class _Logging:
    """A state class that gained two fields holding trees, as
    BaseAgentState.last_actor_metrics: one filled, one held as None."""

    added_fields: ClassVar[frozenset[str]] = frozenset({"metrics", "unset"})
    kept: jax.Array
    metrics: dict
    unset: object = None


def test_an_added_tree_reads_as_nan_and_an_added_none_stays_none(tmp_path):
    """A tree listed in added_fields restores in the skeleton's structure,
    NaN (-1 in an integer leaf); one the skeleton holds as None stays None."""
    metrics = {"loss": jnp.zeros(()), "step": jnp.zeros((), jnp.int32)}
    skeleton = _Logging(kept=jnp.zeros(2), metrics=metrics)
    path = str(tmp_path / "old.pkl")
    save_checkpoint({"kept": jnp.ones(2)}, path)
    restored = restore_into(skeleton, path)
    np.testing.assert_array_equal(restored.kept, jnp.ones(2))
    assert np.isnan(restored.metrics["loss"]) and int(restored.metrics["step"]) == -1
    assert restored.unset is None


def test_a_checkpoint_of_another_agent_still_fails(sac_run, tmp_path):
    """No class lists another agent's fields: SAC's state, holding the
    expert critic REDQ's lacks, does not restore into REDQ's."""
    from ajax.agents.REDQ.REDQ import REDQ

    net = ("8", "relu")
    redq = REDQ(
        env_id="Pendulum-v1",
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=10,
        batch_size=4,
        buffer_size=40,
    )
    skeleton, _ = redq.train(seed=[0], n_timesteps=0)
    path = str(tmp_path / "sac.pkl")
    save_checkpoint(sac_run.out[0], path)
    with pytest.raises(ValueError, match="Unknown field.*expert_critic"):
        restore_into(skeleton, path)


def _ppo_normalising(env_id: str):
    from ajax.agents.PPO.PPO import PPO

    net = ("8", "relu")
    return PPO(
        env_id=env_id,
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        n_steps=8,
        batch_size=8,
        normalize_observations=True,
    )


def _saved(state, tmp_path) -> tuple[str, dict]:
    import pickle

    path = str(tmp_path / "old.pkl")
    save_checkpoint(state, path)
    with open(path, "rb") as fh:
        return path, pickle.load(fh)


def _resave(path: str, saved: dict) -> None:
    import pickle

    with open(path, "wb") as fh:
        pickle.dump(saved, fh)


def test_a_gymnax_on_policy_checkpoint_with_env_statistics_restores_them_merged(
    tmp_path,
):
    """An on-policy checkpoint from when each gymnax env's normaliser kept
    its own observation statistics restores them into the collector as one
    set: the statistics of every env's observations together."""
    from ajax.environments.obs_norm import init_agent_obs_norm, update_obs_norm

    state, _ = _ppo_normalising("Pendulum-v1").train(seed=0, n_timesteps=16)
    path, saved = _saved(state, tmp_path)
    rows = np.arange(15, dtype=np.float32).reshape(5, 3)
    per_env = [
        update_obs_norm(jnp.asarray(r), init_agent_obs_norm(3))[1]
        for r in (rows[:2], rows[2:])
    ]
    # One set per env, under the run's seed axis.
    stacked = jax.tree.map(lambda *x: np.stack(x)[None], *per_env)
    collector = saved["collector_state"]
    collector["obs_norm_info"] = None
    obs_info = {
        k: np.asarray(getattr(stacked, k)) for k in ("var", "count", "mean", "mean_2")
    }
    obs_info.update({"returns": None})
    collector["_env_state"]["normalization_info"] = {"obs": obs_info, "reward": None}
    _resave(path, saved)
    got = restore_into(state, path).collector_state.obs_norm_info
    want = update_obs_norm(jnp.asarray(rows), init_agent_obs_norm(3))[1]
    for field in ("count", "mean", "mean_2", "var"):
        want_field = np.asarray(getattr(want, field))[None]
        np.testing.assert_allclose(getattr(got, field), want_field, rtol=1e-5)


@pytest.mark.parametrize("with_raw", [True, False], ids=["raw-in-info", "before-raw"])
def test_a_brax_on_policy_checkpoint_with_env_statistics_restores_them(
    tmp_path, with_raw
):
    """An on-policy brax checkpoint from when the env's normaliser kept the
    batch's observation statistics (every row the same) and the normalised
    observation restores the statistics into the collector and the raw
    observation into the env state: its info's, else un-normalised."""
    from ajax.environments.obs_norm import init_agent_obs_norm, update_obs_norm
    from ajax.utils import online_normalize

    state, _ = _ppo_normalising("fast").train(seed=0, n_timesteps=16)
    path, saved = _saved(state, tmp_path)
    env = saved["collector_state"]["_env_state"]
    raw = np.asarray(env["obs"])  # (seed, env, obs)
    stats = update_obs_norm(jnp.asarray(raw[0]), init_agent_obs_norm(raw.shape[-1]))[1]
    rows = {
        k: np.repeat(np.asarray(getattr(stats, k)), 2, axis=0)[None]
        for k in ("var", "count", "mean", "mean_2")
    }
    env["obs"] = np.asarray(
        online_normalize(raw, rows["count"], rows["mean"], rows["mean_2"], train=False)[
            0
        ]
    )
    env["info"]["normalization_info"] = {
        "obs": {**rows, "returns": None},
        "reward": None,
    }
    if with_raw:
        env["info"]["raw_obs"] = raw
    saved["collector_state"]["obs_norm_info"] = None
    _resave(path, saved)
    restored = restore_into(state, path).collector_state
    for field in ("count", "mean", "mean_2", "var"):
        want_field = np.asarray(getattr(stats, field))[None]
        np.testing.assert_allclose(
            getattr(restored.obs_norm_info, field), want_field, rtol=1e-6
        )
    np.testing.assert_allclose(restored.env_state.obs, raw, rtol=1e-5, atol=1e-5)
    assert "normalization_info" not in restored.env_state.info


def test_a_checkpoint_of_another_shape_is_refused(sac_run, tmp_path):
    """A checkpoint saved by an agent of another size (here wider critic
    layers) is refused, naming the leaf, instead of grafted onto the
    skeleton."""
    state = sac_run.out[0]
    path = str(tmp_path / "wider.pkl")
    params = state.critic_state.params
    wider = jax.tree.map(lambda x: np.zeros(x.shape[:-1] + (x.shape[-1] + 1,)), params)
    save_checkpoint(
        state.replace(critic_state=state.critic_state.replace(params=wider)), path
    )
    with pytest.raises(ValueError, match=r"does not fit.*critic_state.*params") as e:
        restore_into(state, path)
    assert "APO" not in str(e.value)  # the APO hint is for an APO only


def _without_format(path: str) -> None:
    """Rewrite the checkpoint at ``path`` as one written before checkpoints
    stated their format."""
    import pickle

    from ajax.checkpoint import FORMAT, FORMAT_KEY

    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    assert saved.pop(FORMAT_KEY) == FORMAT
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
    kw.update({"learning_starts": 100, "batch_size": 4, "buffer_size": 40})
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


def test_a_checkpoint_with_a_longer_replay_restores(sac_run, tmp_path):
    """A replay outgrows its skeleton legitimately (the world models size
    theirs to the run, which a resumed run's skeleton does not cover yet):
    only the networks and optimizer states must fit."""
    state = sac_run.out[0]
    buffer = state.collector_state.buffer_state
    rows = {
        k: np.concatenate([v, v], axis=2) if np.ndim(v) > 2 else v
        for k, v in buffer.experience.items()
    }
    longer = state.collector_state.replace(buffer_state=buffer.replace(experience=rows))
    path = str(tmp_path / "longer.pkl")
    save_checkpoint(state.replace(collector_state=longer), path)
    restored = restore_into(state, path).collector_state.buffer_state.experience
    assert restored["obs"].shape[2] == 2 * buffer.experience["obs"].shape[2]


def _udrl(**kw):
    from ajax.agents.UDRL.UDRL import UDRL

    net = ("8", "relu")
    sizes = {"n_envs": 2, "n_steps": 8, "batch_size": 8, "n_updates_per_iter": 1}
    return UDRL(env_id="Pendulum-v1", actor_architecture=net, **sizes, **kw)


def _resaved(path: str, change) -> dict:
    """The checkpoint at ``path`` as ``change`` leaves its pickled dict."""
    import pickle

    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    change(saved)
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    return saved


@pytest.mark.parametrize("copy", [True, False], ids=["raw_obs", "none"])
def test_a_udrl_buffer_saved_with_a_raw_obs_field_restores(tmp_path, copy):
    """On 2026-10-17 UDRL's segment buffer held the env's own observations
    twice, in obs and, for a run with an expert, in raw_obs (None without
    one): a checkpoint written then restores, the copy dropped."""
    state, _ = _udrl().train(seed=0, n_timesteps=16)
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state, path)

    def with_field(saved: dict) -> None:
        obs = saved["buffer"]["obs"]
        saved["buffer"]["raw_obs"] = obs[..., :-2] if copy else None

    saved = _resaved(path, with_field)
    restored = restore_into(state, path).buffer
    np.testing.assert_array_equal(restored.obs, saved["buffer"]["obs"])


def test_a_udrl_buffer_the_env_normalised_is_refused(tmp_path):
    """A UDRL checkpoint saved while the env normalised its observations (its
    env state holds their statistics) keeps each row at the statistics of
    its collection: refused, saying why, since UDRL now normalises the raw
    rows it reads."""
    state, _ = _udrl(normalize_observations=True).train(seed=0, n_timesteps=16)
    path = str(tmp_path / "old.pkl")
    save_checkpoint(state, path)
    stats = {"count": 1.0, "mean": 0.0, "mean_2": 1.0}

    def env_normalised(saved: dict) -> None:
        saved["collector_state"]["_env_state"]["normalization_info"] = {"obs": stats}

    _resaved(path, env_normalised)
    with pytest.raises(ValueError, match="environment normalised"):
        restore_into(state, path)


def test_a_checkpoint_from_before_the_collections_own_stream_restarts_it(tmp_path):
    """A state saved before collection had its own stream of keys (format 1)
    held, as its collector's key, the carry of its last collection, an
    ancestor of the updates' key: it restores with that stream restarted on
    the key derived from the updates' ("collector"). A checkpoint of the
    current format keeps the key it saved."""
    import pickle

    from ajax.agents.SAC.SAC import SAC
    from ajax.checkpoint import FORMAT_KEY
    from ajax.random_keys import derive_key

    net = ("8", "relu")
    kw = {"n_envs": 2, "actor_architecture": net, "critic_architecture": net}
    kw.update({"learning_starts": 100, "batch_size": 4, "buffer_size": 40})
    state, _ = SAC(env_id="Pendulum-v1", **kw).train(seed=[0, 1], n_timesteps=16)
    path = str(tmp_path / "state.pkl")
    save_checkpoint((state, jnp.zeros(3)), path)  # a state saved with its logs
    kept = restore_into((state, jnp.zeros(3)), path)[0]
    np.testing.assert_array_equal(kept.collector_state.rng, state.collector_state.rng)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    saved[FORMAT_KEY] = 1
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    restored = restore_into((state, jnp.zeros(3)), path)[0]
    derived = jax.vmap(lambda key: derive_key(key, "collector"))(state.rng)
    np.testing.assert_array_equal(restored.collector_state.rng, derived)
    np.testing.assert_array_equal(restored.rng, state.rng)


def test_a_row_collector_keeps_its_own_stream_from_an_older_checkpoint(tmp_path):
    """The world models' row collectors had a stream of their own before
    format 2: an older checkpoint keeps it."""
    import pickle

    from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
    from ajax.checkpoint import FORMAT_KEY
    from tests.agents.TDMPC2.common import TINY_AGENT

    state, _ = TDMPC2(**TINY_AGENT, env_id="Pendulum-v1").train(seed=0, n_timesteps=0)
    path = str(tmp_path / "state.pkl")
    save_checkpoint(state, path)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    saved[FORMAT_KEY] = 1
    with open(path, "wb") as fh:
        pickle.dump(saved, fh)
    restored = restore_into(state, path)
    np.testing.assert_array_equal(
        restored.collector_state.rng, state.collector_state.rng
    )
