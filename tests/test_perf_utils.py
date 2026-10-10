"""perf_utils: the resume iteration offset and the shared input of
build_resumable_train / ActorCritic.train."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.base import ActorCritic, shared_counters
from ajax.perf_utils import build_resumable_train

# ---------------------------------------------------------------------------
# Resume offset
# ---------------------------------------------------------------------------


def make_toy_train(num_updates, on_index=None):
    """A make_train whose scan body records the iteration index it sees.

    ``on_index=(k, record)`` adds a schedule gated on the index: a
    ``lax.cond`` on ``i == k`` whose branch appends ``i`` to ``record``.
    """

    def init_fn(key, index):
        del key, index
        return {"ticks": jnp.zeros((), jnp.int32)}

    def body(state, i):
        if on_index is not None:
            at, record = on_index
            jax.lax.cond(
                i == at,
                lambda: jax.debug.callback(lambda v: record.append(int(v)), i),
                lambda: None,
            )
        return {"ticks": state["ticks"] + 1}, i

    return build_resumable_train(
        init_fn=init_fn, make_scan_fn=lambda *_: body, num_updates=num_updates
    )


def test_resumed_scan_sees_absolute_iteration_indices():
    train = make_toy_train(4)
    state, seen = train(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(seen, np.arange(4))
    state, seen = train(
        jax.random.PRNGKey(0),
        initial_state=state,
        resume_from_state=True,
        iteration_offset=jnp.int32(4),
    )
    np.testing.assert_array_equal(seen, np.arange(4, 8))
    assert int(state["ticks"]) == 8
    # without an offset a resumed scan restarts at 0 (historical behaviour)
    _, seen = train(jax.random.PRNGKey(0), initial_state=state, resume_from_state=True)
    np.testing.assert_array_equal(seen, np.arange(4))


def test_offset_is_unbatched_under_the_seed_vmap():
    train = make_toy_train(3)
    states = {"ticks": jnp.array([3, 3], jnp.int32)}
    _, seen = jax.vmap(
        lambda key, state, offset: train(
            key, initial_state=state, resume_from_state=True, iteration_offset=offset
        ),
        in_axes=(0, 0, None),
    )(jax.random.split(jax.random.PRNGKey(0), 2), states, jnp.int32(3))
    np.testing.assert_array_equal(seen, [[3, 4, 5], [3, 4, 5]])


class ToyAgent(ActorCritic):
    """An agent whose schedule is absolute: it resumes at its tick counter."""

    def __init__(self, absolute=True, on_index=None):
        super().__init__(env_id="CartPole-v1", n_envs=1)
        self.absolute = absolute
        self.on_index = on_index

    def get_make_train(self):
        def make_train(total_timesteps, **_):
            return make_toy_train(total_timesteps, self.on_index)

        return make_train

    def resume_iteration_offset(self, initial_state):
        if not self.absolute:
            return super().resume_iteration_offset(initial_state)
        ticks = np.asarray(initial_state["ticks"])
        assert (ticks == ticks[0]).all(), "seeds out of step"
        return int(ticks[0])


def test_actor_critic_train_passes_the_resume_offset():
    agent = ToyAgent()
    state, seen = agent.train(seed=[0, 1], n_timesteps=3)
    np.testing.assert_array_equal(seen, [[0, 1, 2]] * 2)
    state, seen = agent.train(seed=[0, 1], n_timesteps=2, initial_state=state)
    np.testing.assert_array_equal(seen, [[3, 4]] * 2)
    np.testing.assert_array_equal(state["ticks"], [5, 5])


def test_actor_critic_keeps_the_resume_offset_unbatched_under_the_seed_vmap():
    """A schedule gated on the absolute index stays a real ``cond`` on the
    resume path (seeds vmapped, agent state batched): its branch runs once,
    at index 3. A batched offset would turn it into a select running the
    branch on every iteration of every seed."""
    record = []
    agent = ToyAgent(on_index=(3, record))
    state, _ = agent.train(seed=[0, 1], n_timesteps=2)
    agent.train(seed=[0, 1], n_timesteps=3, initial_state=state)
    jax.effects_barrier()
    assert record == [3]


def test_a_resume_reads_the_counters_its_seeds_share():
    """The resumed timestep and logged evaluations, unbatched; none for a
    state without them; seeds standing at different points are refused."""

    def state(timesteps):
        collector = SimpleNamespace(timestep=jnp.array(timesteps, jnp.int32))
        return SimpleNamespace(collector_state=collector, n_logs=jnp.array([2, 2]))

    counters = shared_counters(state([5, 5]))
    assert {k: int(v) for k, v in counters.items()} == {"timestep": 5, "n_logs": 2}
    assert all(v.shape == () for v in counters.values())
    assert shared_counters({"ticks": jnp.array([3, 3])}) == {}
    with pytest.raises(ValueError, match="timestep differ"):
        shared_counters(state([5, 6]))


def test_default_resume_offset_keeps_existing_agents_unchanged():
    agent = ToyAgent(absolute=False)
    assert agent.resume_iteration_offset({"ticks": jnp.array([7])}) == 0
    state, _ = agent.train(seed=0, n_timesteps=3)
    _, seen = agent.train(seed=0, n_timesteps=2, initial_state=state)
    np.testing.assert_array_equal(seen, [[0, 1]])


# ---------------------------------------------------------------------------
# build_resumable_train: the shared input
# ---------------------------------------------------------------------------


def make_shared_train(num_updates):
    """A scan that reads ``shared["w"][i]`` and carries its running sum."""

    def init_fn(key, index):
        return {"total": jnp.zeros(())}

    def make_scan_fn(agent_state, resume_from_state, key, index, *, shared):
        def body(state, i):
            w = shared["w"][i]
            return {"total": state["total"] + w}, w

        return body

    return build_resumable_train(
        init_fn=init_fn, make_scan_fn=make_scan_fn, num_updates=num_updates
    )


def test_shared_input_reaches_the_body_as_one_argument_of_the_seed_vmap():
    """``shared`` reaches ``make_scan_fn`` (as ``shared=``), unbatched under
    the seed vmap (``in_axes=None``): the program takes it as one argument,
    neither a per-seed copy nor a constant."""
    train = make_shared_train(3)
    shared = {"w": jnp.arange(1.0, 6.0)}
    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    def run(key, data):
        return train(key, shared=data)

    states, seen = jax.vmap(run, in_axes=(0, None))(keys, shared)
    np.testing.assert_array_equal(seen, [[1.0, 2.0, 3.0]] * 2)
    np.testing.assert_array_equal(states["total"], [6.0, 6.0])
    text = jax.jit(jax.vmap(run, in_axes=(0, None))).lower(keys, shared).as_text()
    assert "tensor<5xf32>" in text and "tensor<2x5xf32>" not in text
    # Resumed with the same input: the scan continues from the carried sum.
    states, _ = jax.vmap(
        lambda key, state, data: train(
            key, initial_state=state, resume_from_state=True, shared=data
        ),
        in_axes=(0, 0, None),
    )(keys, states, shared)
    np.testing.assert_array_equal(states["total"], [12.0, 12.0])


def test_builders_without_a_shared_input_are_unchanged():
    """Without ``shared`` the builder is called with its four positional
    arguments, as before the slot existed."""
    train = make_toy_train(2)
    _, seen = train(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(seen, [0, 1])
