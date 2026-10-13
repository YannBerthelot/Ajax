"""Unit tests for the pluggable memory cells (ajax.networks.memory).

The load-bearing invariants:
1. step-vs-sequence equivalence: feeding (T, B, F) at once must equal
   feeding T slices of (1, B, F) threading the carry. Collection/eval
   (T=1) and training (full T) share correctness through this property.
2. reset semantics: after a mid-sequence reset the outputs must equal a
   run started from a fresh carry, and gradients must not flow across
   the reset boundary.
3. a transformer carrying projected keys and values (``kv_cache``) computes
   what re-projecting its cached inputs computes.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.networks.memory import (
    MEMORY_KINDS,
    MemoryCell,
    MemoryConfig,
    flat_carry_dim,
    flatten_carry,
    init_carry,
    parse_memory_config,
    unflatten_carry,
    zeros_carry_like,
)

KINDS = MEMORY_KINDS
# every kind, plus the transformer carrying keys and values: (kind, overrides)
VARIANTS: list[tuple[str, dict[str, Any]]] = [(kind, {}) for kind in KINDS]
VARIANTS += [("transformer", {"kv_cache": True})]
T, B, F = 7, 4, 5


def make_cell(kind, **overrides):
    """A cell of ``kind`` (width 8, ``overrides`` on the config), its
    parameters, initial carry, a ``(T, B, F)`` input and no resets."""
    # window < T so the transformer's sliding window is actually exercised
    config = MemoryConfig(**{"kind": kind, "hidden_size": 8, "window": 5, **overrides})
    cell = MemoryCell(config)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, F))
    resets = jnp.zeros((T, B), dtype=bool)
    carry = init_carry(config, rng, B)
    params = cell.init(rng, carry, x, resets)
    return cell, params, carry, x, resets


# ---------------------------------------------------------------- config


def test_config_rejects_unknown_kind():
    with pytest.raises(ValueError):
        MemoryConfig(kind="hopfield")


def test_config_validates_kind_specific_knobs():
    with pytest.raises(ValueError, match="divisible"):
        MemoryConfig(kind="transformer", hidden_size=10, num_heads=4)
    with pytest.raises(ValueError, match="window"):
        MemoryConfig(kind="transformer", window=0)
    with pytest.raises(ValueError, match="d_state"):
        MemoryConfig(kind="mamba", d_conv=0)


def test_parse_memory_config_accepts_dict_and_none():
    assert parse_memory_config(None) is None
    cfg = parse_memory_config({"kind": "lstm", "hidden_size": 16})
    assert cfg == MemoryConfig(kind="lstm", hidden_size=16)


def test_config_is_hashable():
    hash(MemoryConfig())  # required for jit-static args and module fields


# ---------------------------------------------------------------- shapes


@pytest.mark.parametrize("kind, overrides", VARIANTS)
@pytest.mark.parametrize("num_layers", [1, 2])
def test_output_and_carry_shapes(kind, overrides, num_layers):
    cell, params, carry, x, resets = make_cell(kind, num_layers=num_layers, **overrides)
    new_carry, y = cell.apply(params, carry, x, resets)
    assert y.shape == (T, B, 8)
    assert isinstance(new_carry, tuple) and len(new_carry) == num_layers
    # carry structure must be stable (required by lax.scan carries)
    assert jax.tree.structure(new_carry) == jax.tree.structure(carry)
    for out_leaf, in_leaf in zip(
        jax.tree.leaves(new_carry), jax.tree.leaves(carry), strict=False
    ):
        assert out_leaf.shape == in_leaf.shape


@pytest.mark.parametrize("kind, overrides", VARIANTS)
def test_initial_carry_is_zeros(kind, overrides):
    # zeros_carry_like (used for eval and replayed sequences) relies on this
    config = MemoryConfig(kind=kind, hidden_size=8, **overrides)
    carry = init_carry(config, jax.random.PRNGKey(0), B)
    for leaf in jax.tree.leaves(carry):
        assert jnp.all(leaf == 0)


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_zeros_carry_like_resizes_batch_axis(batch_axis):
    config = MemoryConfig(kind="lstm", hidden_size=8)
    carry = init_carry(config, jax.random.PRNGKey(0), B)
    if batch_axis == 1:  # simulate an ensemble-stacked carry (num, B, H)
        carry = jax.tree.map(lambda x: jnp.stack([x, x]), carry)
    resized = zeros_carry_like(carry, 11, batch_axis=batch_axis)
    for leaf in jax.tree.leaves(resized):
        assert leaf.shape[batch_axis] == 11
        assert jnp.all(leaf == 0)


# ------------------------------------------------- step/sequence equivalence


@pytest.mark.parametrize("kind, overrides", VARIANTS)
@pytest.mark.parametrize("num_layers", [1, 2])
def test_step_equals_sequence(kind, overrides, num_layers):
    cell, params, carry, x, _ = make_cell(kind, num_layers=num_layers, **overrides)
    resets = jnp.zeros((T, B), dtype=bool).at[3, :2].set(True)

    seq_carry, y_seq = cell.apply(params, carry, x, resets)

    step_carry = carry
    ys = []
    for t in range(T):
        step_carry, y_t = cell.apply(
            params, step_carry, x[t : t + 1], resets[t : t + 1]
        )
        ys.append(y_t)
    y_steps = jnp.concatenate(ys, axis=0)

    assert jnp.allclose(y_seq, y_steps, atol=1e-6)
    for a, b in zip(
        jax.tree.leaves(seq_carry), jax.tree.leaves(step_carry), strict=False
    ):
        assert jnp.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------- resets


@pytest.mark.parametrize("kind, overrides", VARIANTS)
def test_reset_equals_fresh_start(kind, overrides):
    cell, params, carry, x, _ = make_cell(kind, **overrides)
    t_reset = 4
    resets = jnp.zeros((T, B), dtype=bool).at[t_reset].set(True)

    _, y_full = cell.apply(params, carry, x, resets)
    fresh_carry = init_carry(cell.config, jax.random.PRNGKey(0), B)
    _, y_fresh = cell.apply(
        params, fresh_carry, x[t_reset:], jnp.zeros((T - t_reset, B), dtype=bool)
    )
    assert jnp.allclose(y_full[t_reset:], y_fresh, atol=1e-6)


@pytest.mark.parametrize("kind, overrides", VARIANTS)
def test_no_gradient_across_reset(kind, overrides):
    cell, params, carry, x, _ = make_cell(kind, **overrides)
    t_reset = 3
    resets = jnp.zeros((T, B), dtype=bool).at[t_reset].set(True)

    def out_at_end(x_in):
        _, y = cell.apply(params, carry, x_in, resets)
        return y[-1].sum()

    grads = jax.grad(out_at_end)(x)
    # inputs before the reset must not influence outputs after it
    assert jnp.all(grads[:t_reset] == 0)
    # but BPTT must flow within the post-reset episode
    assert jnp.any(grads[t_reset:-1] != 0)


@pytest.mark.parametrize("kind", KINDS)
def test_gradient_flows_through_time_without_reset(kind):
    # window >= T so the transformer's full receptive field covers step 0
    cell, params, carry, x, resets = make_cell(kind, window=T + 1)

    def out_at_end(x_in):
        _, y = cell.apply(params, carry, x_in, resets)
        return y[-1].sum()

    grads = jax.grad(out_at_end)(x)
    assert jnp.any(grads[0] != 0)


def test_transformer_kv_cache_is_the_input_cache_computation():
    """Caching each layer's projected keys and values gives what
    re-projecting the cached layer inputs gives (the input cache is the
    oracle): outputs and gradients, stepping and in chunks, over a rolling
    window, two layers and a mid-run reset, to a few float32 eps of their
    scale (rounding only; never last-ulp, which differs across platforms)."""
    n_steps, eps, key = 12, float(jnp.finfo(jnp.float32).eps), jax.random.PRNGKey(0)
    oracle = MemoryConfig(
        kind="transformer", hidden_size=8, window=5, num_layers=2, num_heads=2
    )
    cached = dataclasses.replace(oracle, kv_cache=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (n_steps, B, F))
    resets = jnp.zeros((n_steps, B), dtype=bool).at[6, :2].set(True)
    params, kv_params = (
        MemoryCell(c).init(key, init_carry(c, key, B), x[:1], resets[:1])
        for c in (oracle, cached)
    )
    assert jax.tree.all(jax.tree.map(jnp.array_equal, params, kv_params))

    def grads_and_outputs(config, chunk):
        def loss(params, x):
            cell, carry = MemoryCell(config), init_carry(config, key, B)
            ys = []
            for t in range(0, n_steps, chunk):
                sl = slice(t, t + chunk)
                carry, y = cell.apply(params, carry, x[sl], resets[sl])
                ys.append(y)
            return (jnp.concatenate(ys) ** 2).mean(), jnp.concatenate(ys)

        return jax.jit(jax.grad(loss, argnums=(0, 1), has_aux=True))(params, x)

    def assert_close(want, got):
        # One scale for the whole tree: the key bias's gradient is rounding
        # noise (softmax ignores a shift shared by every key), so no
        # per-leaf relative bound.
        scale = max(float(jnp.abs(leaf).max()) for leaf in jax.tree.leaves(want))
        for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
            assert jnp.allclose(a, b, rtol=0, atol=8 * eps * scale)

    for chunk in (1, 3):
        (grads, y), (kv_grads, kv_y) = (
            grads_and_outputs(c, chunk) for c in (oracle, cached)
        )
        assert_close(y, kv_y)
        assert_close(grads, kv_grads)


def test_transformer_sliding_window_bounds_memory():
    """The carry footprint is O(window), independent of sequence length,
    and inputs beyond the window cannot influence the output."""
    config = MemoryConfig(kind="transformer", hidden_size=8, window=3)
    cell = MemoryCell(config)
    rng = jax.random.PRNGKey(0)
    carry = init_carry(config, rng, B)
    for t_len in (2, 11):
        x = jax.random.normal(rng, (t_len, B, F))
        resets = jnp.zeros((t_len, B), dtype=bool)
        params = cell.init(rng, carry, x, resets)
        new_carry, _ = cell.apply(params, carry, x, resets)
        assert jax.tree.structure(new_carry) == jax.tree.structure(carry)
        for out_leaf, in_leaf in zip(
            jax.tree.leaves(new_carry), jax.tree.leaves(carry), strict=False
        ):
            assert out_leaf.shape == in_leaf.shape  # O(window), not O(T)

    # out-of-window inputs are invisible: perturbing x[0] must not change
    # the output at t >= window
    x = jax.random.normal(rng, (6, B, F))
    resets = jnp.zeros((6, B), dtype=bool)
    params = cell.init(rng, carry, x, resets)
    _, y1 = cell.apply(params, carry, x, resets)
    x2 = x.at[0].add(100.0)
    _, y2 = cell.apply(params, carry, x2, resets)
    assert jnp.allclose(y1[3:], y2[3:], atol=1e-5)
    assert not jnp.allclose(y1[0], y2[0], atol=1e-5)


# ---------------------------------------------------------------- jit


@pytest.mark.parametrize("kind, overrides", VARIANTS)
def test_jit_and_gradient_checkpoint(kind, overrides):
    cell, params, carry, x, resets = make_cell(
        kind, gradient_checkpoint=True, window=MemoryConfig().window, **overrides
    )

    @jax.jit
    def run(c, x_in):
        return cell.apply(params, c, x_in, resets)

    new_carry, y = run(carry, x)
    assert y.shape == (T, B, 8)
    grads = jax.grad(lambda x_in: run(carry, x_in)[1].sum())(x)
    assert jnp.all(jnp.isfinite(grads))


@pytest.mark.parametrize("kind, overrides", VARIANTS)
def test_flatten_carry_roundtrip(kind, overrides):
    """flatten_carry/unflatten_carry must be exact inverses for every kind
    (stored-state replay writes flattened carries into the buffer)."""
    config = MemoryConfig(kind=kind, hidden_size=8, window=5, num_layers=2, **overrides)
    carry = init_carry(config, jax.random.PRNGKey(0), B)
    # fill with distinguishable values (bools stay bool)
    carry = jax.tree.map(
        lambda x: (
            jnp.ones(x.shape, bool)
            if x.dtype == bool
            else jnp.arange(x.size, dtype=x.dtype).reshape(x.shape)
        ),
        carry,
    )
    flat = flatten_carry(carry)
    assert flat.shape == (B, flat_carry_dim(config) * 1)
    back = unflatten_carry(flat, zeros_carry_like(carry, B))
    for a, b in zip(jax.tree.leaves(carry), jax.tree.leaves(back), strict=False):
        assert a.dtype == b.dtype
        assert jnp.array_equal(a, b)
