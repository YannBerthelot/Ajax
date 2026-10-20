"""The conformance seams (tests/conformance/kit.py): the tape fixes every random draw of a step, it
tapes permutations as indices, it leaves jax.random as it found it, and the step and scan captures
return callables that run on any state."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax
from tests.conformance import kit


def _is_key(path) -> bool:
    name = jax.tree_util.keystr(path).lower()
    return "rng" in name or "key" in name


def with_keys_of(state, donor):
    """``state``'s values with ``donor``'s PRNG keys."""
    return jax.tree_util.tree_map_with_path(
        lambda p, x, y: y if _is_key(p) else x, state, donor
    )


def values(state) -> dict:
    return {
        jax.tree_util.keystr(p): np.asarray(x)
        for p, x in jax.tree_util.tree_leaves_with_path(state)
        if not _is_key(p)
    }


def run(fn, state, context):
    with context as session:
        out = jax.jit(fn)(state)
        jax.block_until_ready(out)
    return out, session


AGENTS = {
    # (agent, budget of one iteration, the largest relative difference allowed: the record and replay
    # programs fuse differently, which moves SAC's update by rounding; PQN's step comes out bit for bit)
    "PQN": (
        lambda: ajax.PQN(
            "CartPole-v1", n_envs=4, n_steps=8, num_minibatches=2, n_epochs=1
        ),
        32,
        0.0,
    ),
    "SAC": (
        lambda: ajax.SAC(
            "Pendulum-v1", learning_starts=0, batch_size=8, buffer_size=64
        ),
        1,
        1e-5,
    ),
}


@pytest.mark.parametrize("name", AGENTS)
def test_the_tape_alone_fixes_every_draw_of_a_step(name):
    make, budget, rtol = AGENTS[name]
    step = kit.capture_step(make(), budget)
    one = lambda s: step.iteration(s, 0, 0)[0]  # noqa: E731
    b = step.fresh(jax.random.PRNGKey(1))
    a = with_keys_of(b, step.fresh(jax.random.PRNGKey(0)))  # b's state, other keys
    expected = jax.jit(one)(b)
    _, rec_a = run(one, a, kit.recording())
    _, rec_b = run(one, b, kit.recording())
    keys_a, draws_b = kit.by_role(rec_a.log), kit.by_role(rec_b.log)
    assert keys_a.keys() == draws_b.keys()
    assert all(len(keys_a[r]) == len(draws_b[r]) for r in keys_a)
    table = kit.key_table(keys_a, {r: [v for _, v in d] for r, d in draws_b.items()})
    got, replay = run(one, a, kit.replaying(table))
    assert not replay.misses
    want, have = values(expected), values(got)
    for leaf, w in want.items():
        # Bookkeeping leaves hold NaN until an episode ends: equal NaNs are equal.
        scale = float(np.nanmax(np.abs(w), initial=0.0)) if w.size else 0.0
        np.testing.assert_allclose(
            have[leaf], w, rtol=0, atol=rtol * scale, equal_nan=True, err_msg=leaf
        )
    # The keys matter: untaped, a's step is not b's.
    plain = values(jax.jit(one)(a))
    assert any(not np.array_equal(plain[k], want[k]) for k in want)


def test_a_permutation_of_an_array_is_taped_as_its_indices():
    x = jnp.arange(12.0).reshape(6, 2)
    key = jax.random.PRNGKey(3)
    f = lambda k: jax.random.permutation(k, x)  # noqa: E731
    with kit.recording() as rec:
        out = jax.jit(f)(key)
    [(role, k, v)] = rec.log
    assert role[0] == "permutation" and role[2:] == ((6,), "int32")
    assert sorted(v.tolist()) == list(range(6))
    np.testing.assert_array_equal(out, np.asarray(x)[v])
    chosen = np.array([5, 4, 3, 2, 1, 0], np.int32)
    with kit.replaying({role: (k[None], chosen[None])}) as rep:
        out = jax.jit(f)(key)
    assert not rep.misses
    np.testing.assert_array_equal(out, np.asarray(x)[chosen])


def test_a_float64_run_replays_a_float32_tape():
    def f(k):
        a, b = jax.random.split(k)
        return jax.random.normal(a, (3,)), jax.random.permutation(b, 5)

    key = jax.random.PRNGKey(4)
    with kit.recording() as rec:
        want = jax.jit(f)(key)
    draws = kit.by_role(rec.log)
    table = kit.key_table(draws, {r: [v for _, v in d] for r, d in draws.items()})
    with jax.enable_x64(True), kit.replaying(table) as rep:
        got = jax.jit(f)(key)
    assert not rep.misses and not rep.unreplayed
    assert (got[0].dtype, got[1].dtype) == (jnp.float64, jnp.int64)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)


def test_the_tape_restores_jax_random():
    before = jax.random.normal
    with kit.recording():
        assert jax.random.normal is not before
    assert jax.random.normal is before


def _toy(x):
    def body(carry, _):
        return carry + 1, carry

    return jax.lax.scan(body, x, None, length=3)


def test_a_captured_scan_body_runs_on_any_carry_and_stashing_keeps_its_outputs():
    body, init = kit.capture_scan("body", lambda: _toy(jnp.zeros(())))
    assert float(init) == 0.0
    carry, out = body(jnp.asarray(5.0), None)
    assert (float(carry), float(out)) == (6.0, 5.0)

    def traced(x):
        with kit.stashing("body") as stash:
            _toy(x)
        return stash["body"]

    carry, stacked = jax.jit(traced)(jnp.zeros(()))
    assert float(carry) == 3.0
    np.testing.assert_array_equal(stacked, [0.0, 1.0, 2.0])
    with pytest.raises(RuntimeError, match="no lax.scan"):
        kit.capture_scan("absent", lambda: _toy(jnp.zeros(())))


def test_a_substituted_scan_returns_the_given_outputs_and_the_step_goes_on_from_them():
    def step(x):
        carry, stacked = _toy(x)
        return carry * 10 + stacked.sum()

    given = (jnp.asarray(7.0), jnp.asarray([1.0, 1.0, 1.0]))
    with kit.substituting({"body": given}):
        assert float(jax.jit(step)(jnp.zeros(()))) == 73.0
    with kit.substituting({"other": given}):
        assert float(jax.jit(step)(jnp.zeros(()))) == 33.0
