"""A flat replay sized to its run (ReplayAgent.replay_length, ajax.buffers.ring).

Contract (the owner's plan of 2026-10-27, section 4, and buffer_size's
docstring): a run's replay holds, per env, the rows the run writes plus
one, at most buffer_size // n_envs and at least the rows it holds before it
can sample, so it never reports full before buffer_size // n_envs rows per
env. The sizing changes no number: the rows written, and every other leaf,
equal those of the same run on a replay of another length. A resumed run
grows the ring to the uninterrupted run's, so two calls give that run's
state. The rows written before training (ExpertPrefill, PriorData, an
offline dataset) count. A 0-step skeleton keeps the smallest ring and
restores a longer one.

Equality is exact: the runs take the same steps on the same rows, the
ring's length moving only where the unwritten rows sit.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import DQN, SAC, TD3
from ajax.checkpoint import load_checkpoint, restore_into, save_checkpoint
from ajax.extensions.pretrain import ExpertPrefill, PriorData
from ajax.networks.memory import MemoryConfig
from tests.agents.test_offline import ROWS, episodes

N_ENVS, TICKS, HUGE = 2, 32, 10**6
NET = ("16", "relu")
AGENTS: dict[str, tuple[Any, dict[str, Any]]] = {
    "SAC": (SAC, {"env_id": "Pendulum-v1", "actor_architecture": NET}),
    "DQN": (DQN, {"env_id": "CartPole-v1", "architecture": NET}),
    "TD3-gru": (
        TD3,
        {
            "env_id": "Pendulum-v1",
            "actor_architecture": NET,
            "memory": MemoryConfig(kind="gru", hidden_size=8),
            "burn_in": 2,
            "sequence_length": 4,
            "learning_starts": 16,  # 8 steps an env: more than a window of 7
        },
    ),
}


def build(name: str, buffer_size: int, **options: Any) -> Any:
    cls, kwargs = AGENTS[name]
    base = {"n_envs": N_ENVS, "learning_starts": 8, "batch_size": 8}
    return cls(**{**base, **kwargs, "buffer_size": buffer_size, **options})


def train(agent: Any, ticks: int, **options: Any) -> Any:
    """One seed's state after ticks ticks, on the host: a state passed as
    initial_state may be donated."""
    state, _ = agent.train(
        seed=[0], n_timesteps=N_ENVS * ticks, num_episode_test=1, **options
    )
    return jax.device_get(state)


def ring(state: Any) -> int:
    return np.shape(state.collector_state.buffer_state.experience["reward"])[2]


def written(state: Any) -> int:
    return int(state.collector_state.buffer_state.current_index[0])


def leaves(state: Any, replay: bool = True) -> dict[str, np.ndarray]:
    """Every leaf of state (the replay's too, unless replay is False), PRNG
    keys as their data."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        name = jax.tree_util.keystr(path)
        if not replay and ".buffer_state." in name:
            continue
        if jnp.issubdtype(np.asarray(leaf).dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        out[name] = np.asarray(leaf)
    return out


def rows_written(state: Any) -> dict[str, np.ndarray]:
    """Seed 0's replay fields, its written rows alone, oldest first: those
    before current_index, or a full ring's from current_index round."""
    replay = state.collector_state.buffer_state
    index, full = int(replay.current_index[0]), bool(replay.is_full[0])
    if full:
        return {k: np.roll(v[0], -index, axis=1) for k, v in replay.experience.items()}
    return {k: v[0][:, :index] for k, v in replay.experience.items()}


def assert_equal(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for key, values in want.items():
        np.testing.assert_array_equal(got[key], values, err_msg=key)


@pytest.fixture(scope="module")
def sac() -> dict[str, Any]:
    """SAC over 2 * TICKS ticks in one call and in two, and the first call
    again on a replay its capacity bounds to TICKS rows per env."""
    agent = build("SAC", HUGE)
    total = {"total_timesteps": 2 * N_ENVS * TICKS}
    half = train(agent, TICKS, **total)
    return {
        "agent": agent,
        "whole": train(agent, 2 * TICKS),
        "half": half,
        "resumed": train(agent, TICKS, initial_state=half, **total),
        "bound": train(build("SAC", N_ENVS * TICKS), TICKS, **total),
    }


@pytest.mark.parametrize("name", ["SAC", "DQN", "TD3-gru"])
def test_a_run_writes_and_reads_the_rows_of_a_replay_of_another_length(
    name: str, sac: dict
) -> None:
    """A run on its sized ring, TICKS + 1 rows per env, and on the ring its
    capacity bounds to TICKS, which the run fills."""
    if name == "SAC":
        sized, bound = sac["half"], sac["bound"]
    else:
        sized = train(build(name, HUGE), TICKS)
        bound = train(build(name, N_ENVS * TICKS), TICKS)
    assert (ring(sized), ring(bound)) == (TICKS + 1, TICKS)
    assert_equal(rows_written(sized), rows_written(bound))
    assert_equal(leaves(sized, replay=False), leaves(bound, replay=False))


def test_a_resumed_run_grows_its_ring_into_the_uninterrupted_runs(sac: dict) -> None:
    assert ring(sac["half"]) == TICKS + 1
    assert ring(sac["resumed"]) == ring(sac["whole"]) == 2 * TICKS + 1
    got, want = leaves(sac["resumed"]), leaves(sac["whole"])
    assert got.keys() == want.keys()
    for key, values in want.items():
        if key.startswith(".critic_state."):
            # A resumed call runs another program, whose float32 sums XLA may
            # order otherwise (on main too): the critic drifts by under 1e-6
            # here, far less than a batch of other rows would move it.
            np.testing.assert_allclose(got[key], values, atol=1e-5, err_msg=key)
        else:  # the replay, ring included, the actor and the counters
            np.testing.assert_array_equal(got[key], values, err_msg=key)


def test_a_ring_is_full_only_at_its_capacity() -> None:
    agent = build("SAC", N_ENVS * 20, batch_size=16)  # 20 rows an env, 9 to sample
    capacity, smallest = 20, 9
    for rows in range(3 * capacity):
        length = agent.replay_length(rows)
        assert smallest <= length <= capacity
        # A ring of length rows per env reports full once it holds length.
        assert rows < length or length == capacity


def pendulum_expert(obs: jax.Array) -> jax.Array:
    return jnp.zeros((*obs.shape[:-1], 1))


def test_the_rows_written_before_training_count_in_the_ring() -> None:
    ticks, prefill, per_env = 8, 6, ROWS // N_ENVS
    data = episodes(3, discrete=False)  # Pendulum's observations
    expert = {"expert_policy": pendulum_expert}
    agent = build("SAC", HUGE, extensions=(ExpertPrefill(n_steps=prefill),), **expert)
    state = train(agent, ticks)
    assert (ring(state), written(state)) == (prefill + ticks + 1, prefill + ticks)
    state = train(build("SAC", HUGE, extensions=(PriorData(data),)), ticks)
    assert (ring(state), written(state)) == (per_env + ticks + 1, per_env + ticks)
    agent = build("SAC", HUGE)
    offline, _ = agent.train_offline(data, n_updates=4, seed=[0], num_episode_test=1)
    assert (ring(offline), written(offline)) == (per_env + 1, per_env)
    online = train(agent, ticks, initial_state=jax.device_get(offline))
    assert (ring(online), written(online)) == (per_env + ticks + 1, per_env + ticks)


def test_a_skeleton_restores_and_resumes_a_longer_ring(sac: dict, tmp_path) -> None:
    agent, half = sac["agent"], sac["half"]
    path = str(tmp_path / "half.pkl")
    save_checkpoint(half, path)
    skeleton, _ = agent.train(seed=[0], n_timesteps=0)
    assert ring(skeleton) == 8 // N_ENVS + 1 < ring(half)  # the rows to sample
    assert_equal(leaves(restore_into(skeleton, path)), leaves(half))
    loaded = load_checkpoint(agent, path)
    assert_equal(leaves(loaded), leaves(half))
    total = {"total_timesteps": 2 * N_ENVS * TICKS}
    resumed = train(agent, TICKS, initial_state=loaded, **total)
    assert_equal(leaves(resumed), leaves(sac["resumed"]))
