"""perf_utils: final_aux_fori and the resume iteration offset of
build_resumable_train / ActorCritic.train."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.base import ActorCritic
from ajax.perf_utils import build_resumable_train, final_aux_fori

# ---------------------------------------------------------------------------
# final_aux_fori
# ---------------------------------------------------------------------------


def make_body(scale, traces):
    def body(i, carry):
        traces.append(1)
        carry = carry * scale + jnp.sin(carry)
        return carry, {"total": carry.sum(), "i": i}

    return body


def reference(carry, scale, n):
    aux = {"total": np.float32(0.0), "i": np.int32(0)}
    for i in range(n):
        carry = carry * scale + np.sin(carry)
        aux = {"total": carry.sum(), "i": i}
    return carry, aux


@pytest.mark.parametrize("n", [0, 1, 5])
def test_final_aux_fori_runs_exactly_n_times_and_keeps_the_last_aux(n):
    traces = []
    x = jnp.linspace(-1.0, 1.0, 6).reshape(2, 3)
    run = jax.jit(lambda x, n: final_aux_fori(make_body(0.5, traces), x, n))
    carry, aux = run(x, n)
    ref_carry, ref_aux = reference(np.asarray(x), 0.5, n)
    np.testing.assert_allclose(carry, ref_carry, rtol=1e-6)
    np.testing.assert_allclose(aux["total"], ref_aux["total"], rtol=1e-6)
    assert int(aux["i"]) == ref_aux["i"]  # zeros when n == 0
    assert aux["i"].dtype == jnp.int32
    assert len(traces) == 1  # one trace of the body


def test_final_aux_fori_is_one_unbatched_while_under_the_seed_vmap():
    """``n`` from unbatched values -> a single while whose predicate is not
    batched, even though the carry (and a closed-over value) are per seed:
    no select-masked loop to the maximum trip count."""
    traces = []

    def per_seed(x, scale, n):
        return final_aux_fori(make_body(scale, traces), x, n)

    seeds_x = jnp.ones((4, 2, 3))
    scales = jnp.linspace(0.1, 0.4, 4)
    jaxpr = jax.make_jaxpr(jax.vmap(per_seed, in_axes=(0, 0, None)))(
        seeds_x, scales, 3
    ).jaxpr
    loops = [e for e in jaxpr.eqns if e.primitive.name == "while"]
    assert len(loops) == 1
    cond_jaxpr = loops[0].params["cond_jaxpr"].jaxpr
    assert cond_jaxpr.outvars[0].aval.shape == ()
    assert [e.primitive.name for e in cond_jaxpr.eqns] == ["lt"]
    assert len(traces) == 1
    # and the batched result is the per-seed result
    carry, aux = jax.vmap(per_seed, in_axes=(0, 0, None))(seeds_x, scales, 3)
    for s in range(4):
        ref_carry, ref_aux = reference(np.ones((2, 3), np.float32), float(scales[s]), 3)
        np.testing.assert_allclose(carry[s], ref_carry, rtol=1e-6)
        assert int(aux["i"][s]) == 2


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

    return build_resumable_train(init_fn=init_fn, scan_fn=body, num_updates=num_updates)


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


def test_default_resume_offset_keeps_existing_agents_unchanged():
    agent = ToyAgent(absolute=False)
    assert agent.resume_iteration_offset({"ticks": jnp.array([7])}) == 0
    state, _ = agent.train(seed=0, n_timesteps=3)
    _, seen = agent.train(seed=0, n_timesteps=2, initial_state=state)
    np.testing.assert_array_equal(seen, [[0, 1]])
