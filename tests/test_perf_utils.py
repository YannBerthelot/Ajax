"""perf_utils: the resume iteration offset and the shared input of
build_resumable_train / ActorCritic.train, build_chunked_train's evaluations
between chunks, and final_aux_scan's last aux."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax.agents.base import ActorCritic, shared_counters
from ajax.perf_utils import build_chunked_train, build_resumable_train, final_aux_scan

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
    """The resumed timestep, logged evaluations and update count, unbatched;
    none for a state without them; seeds standing at different points are
    refused, unless the update counts of an agent whose seeds may update
    unequally differ: those stay batched."""

    def state(timesteps, n_updates=(4, 4)):
        collector = SimpleNamespace(timestep=jnp.array(timesteps, jnp.int32))
        return SimpleNamespace(
            collector_state=collector,
            n_logs=jnp.array([2, 2]),
            n_updates=jnp.array(n_updates),
        )

    counters = shared_counters(state([5, 5]))
    want = {"timestep": 5, "n_logs": 2, "n_updates": 4}
    assert {k: int(v) for k, v in counters.items()} == want
    assert all(v.shape == () for v in counters.values())
    unequal = state([5, 5], n_updates=(4, 3))
    assert "n_updates" not in shared_counters(unequal, per_seed_updates=True)
    assert shared_counters({"ticks": jnp.array([3, 3])}) == {}
    with pytest.raises(ValueError, match="timestep differ"):
        shared_counters(state([5, 6]))
    with pytest.raises(ValueError, match="n_updates differ"):
        shared_counters(unequal)


@struct.dataclass
class Counted:
    n_updates: jax.Array


class CountingAgent(ActorCritic):
    """A scan whose step ``at`` (counted by ``n_updates``) calls back."""

    def __init__(self, at, record):
        super().__init__(env_id="CartPole-v1", n_envs=1)
        self.at, self.record = at, record

    def get_make_train(self):
        def body(state, i):
            jax.lax.cond(
                state.n_updates == self.at,
                lambda: jax.debug.callback(self.record.append, i),
                lambda: None,
            )
            return state.replace(n_updates=state.n_updates + 1), i

        def make_train(total_timesteps, **_):
            return build_resumable_train(
                init_fn=lambda *_: Counted(jnp.zeros((), jnp.int32)),
                make_scan_fn=lambda *_: body,
                num_updates=total_timesteps,
            )

        return make_train


def test_a_resumed_update_schedule_stays_a_cond_under_the_seed_vmap():
    """A schedule gated on the update count (TD3's policy delay, DQN's
    target refresh) runs its branch once on the resume path, at update 3:
    a batched count would turn it into a select running the branch on
    every iteration."""
    record: list = []
    agent = CountingAgent(at=3, record=record)
    state, _ = agent.train(seed=[0, 1], n_timesteps=2)
    agent.train(seed=[0, 1], n_timesteps=3, initial_state=state)
    jax.effects_barrier()
    assert len(record) == 1


class PerSeedCountingAgent(CountingAgent):
    """A CountingAgent whose seeds may update unequally, declared per
    instance as TD-MPC2's episodic mode does."""

    def __init__(self, at, record):
        super().__init__(at, record)
        self.per_seed_update_counts = True


def test_a_resume_keeps_unequal_update_counts_only_where_declared():
    """Seeds resumed at update counts 2 and 1: an agent declaring per-seed
    counts continues each seed's own (to 5 and 4); any other agent refuses
    the state."""

    def unequal():  # a fresh state each call: a resume donates its own
        return Counted(jnp.array([2, 1], jnp.int32))

    state, _ = PerSeedCountingAgent(at=-1, record=[]).train(
        seed=[0, 1], n_timesteps=3, initial_state=unequal()
    )
    np.testing.assert_array_equal(state.n_updates, [5, 4])
    with pytest.raises(ValueError, match="n_updates differ"):
        CountingAgent(at=-1, record=[]).train(
            seed=[0, 1], n_timesteps=3, initial_state=unequal()
        )


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


# ---------------------------------------------------------------------------
# build_chunked_train: evaluations between chunks of ticks
# ---------------------------------------------------------------------------
#
# Contract (build_chunked_train's docstring): the ticks run in order on one
# state, as one scan would; log_fn runs after exactly the ticks due marks,
# on the state after that tick and with that tick's aux; ticks are absolute
# on a resume. The toy state sums the ticks it has run, and each tick's aux
# is ten times the tick, so every expected value below is worked out by hand.


def make_chunked_toy(num_updates, every=3):
    def init_fn(key, index):
        del key, index
        return {"sum": jnp.zeros((), jnp.int32)}

    def tick_fn(state, index, tick):
        del index
        return {"sum": state["sum"] + tick}, 10 * tick

    def log_fn(state, aux, out, index, tick):
        del index
        rows, count = out
        row = jnp.stack([tick, state["sum"], aux])
        return state, (rows.at[count].set(row), count + 1)

    def carry_out(state, index):
        del state, index
        return jnp.full((4, 3), -1, jnp.int32), jnp.zeros((), jnp.int32)

    return build_chunked_train(
        init_fn=init_fn,
        tick_fn=tick_fn,
        log_fn=log_fn,
        due=lambda ticks: (ticks + 1) % every == 0,
        num_updates=num_updates,
        carry_out=carry_out,
    )


def test_chunks_evaluate_after_each_due_tick_on_that_ticks_state():
    """Catches a chunk boundary off by one tick, an evaluation reading the
    chunk's first aux instead of its last, and a final chunk that does not
    end on a due tick being dropped (the sum would fall short of 0+...+7)."""
    train = make_chunked_toy(8)
    state, (rows, count) = jax.vmap(train)(
        jax.random.split(jax.random.PRNGKey(0), 2), jnp.arange(2)
    )
    # Ticks 2 and 5 are due ((t + 1) % 3 == 0); the sums after them are
    # 0+1+2 = 3 and 0+...+5 = 15, the aux 10 t; tick 7 ends the call.
    expected = [[2, 3, 20], [5, 15, 50], [-1, -1, -1], [-1, -1, -1]]
    np.testing.assert_array_equal(rows, [expected, expected])
    np.testing.assert_array_equal(count, [2, 2])
    np.testing.assert_array_equal(state["sum"], [28, 28])  # 0+1+...+7


def test_a_resumed_chunked_run_evaluates_on_absolute_ticks():
    """Catches a resume whose chunks restart at tick 0: it would evaluate
    after ticks 2 and 5 of the call, not after absolute ticks 5 and 8."""
    train = make_chunked_toy(5)
    state = {"sum": jnp.int32(6)}  # ticks 0..3 ran: 0+1+2+3
    state, (rows, count) = train(
        jax.random.PRNGKey(0),
        initial_state=state,
        resume_from_state=True,
        iteration_offset=jnp.int32(4),
    )
    # Ticks 4..8 run; 5 and 8 are due: sums 6+4+5 = 15 and 15+6+7+8 = 36.
    np.testing.assert_array_equal(rows[:2], [[5, 15, 50], [8, 36, 80]])
    assert int(count) == 2
    assert int(state["sum"]) == 36


# ---------------------------------------------------------------------------
# final_aux_scan
# ---------------------------------------------------------------------------


def running_sum(total: jax.Array, x: jax.Array) -> tuple[jax.Array, dict]:
    """The carry adds x; the aux is the new total times 10 and the x read."""
    total = total + x
    return total, {"scaled": total * 10.0, "x": x}


@pytest.mark.parametrize(
    "length, total, scaled, x",
    # xs = 1, 2, ..., length: the total is length (length + 1) / 2 and the
    # last step reads x = length.
    [(1, 1.0, 10.0, 1.0), (2, 3.0, 30.0, 2.0), (5, 15.0, 150.0, 5.0)],
)
def test_final_aux_scan_returns_the_carry_and_the_last_steps_aux(
    length: int, total: float, scaled: float, x: float
) -> None:
    """The bugs: the second-to-last step's aux, or the last step skipped or
    run twice (the steps before it run apart from it)."""
    xs = jnp.arange(1.0, length + 1.0)
    carry, aux = jax.jit(final_aux_scan, static_argnums=(0, 2))(
        running_sum, jnp.asarray(0.0), length, xs
    )
    assert float(carry) == total
    assert (float(aux["scaled"]), float(aux["x"])) == (scaled, x)


def test_final_aux_scan_without_inputs_counts_its_steps() -> None:
    def count(steps: jax.Array, x: None) -> tuple[jax.Array, jax.Array]:
        assert x is None
        return steps + 1, steps * 100

    carry, aux = final_aux_scan(count, jnp.asarray(0), 4)
    # Four steps; the last starts from 3.
    assert (int(carry), int(aux)) == (4, 300)
