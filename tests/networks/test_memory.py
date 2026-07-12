"""Unit tests for the pluggable memory cells (ajax.networks.memory).

The load-bearing invariants:
1. step-vs-sequence equivalence: feeding (T, B, F) at once must equal
   feeding T slices of (1, B, F) threading the carry. Collection/eval
   (T=1) and training (full T) share correctness through this property.
2. reset semantics: after a mid-sequence reset the outputs must equal a
   run started from a fresh carry, and gradients must not flow across
   the reset boundary.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.networks.memory import (
    MemoryCell,
    MemoryConfig,
    init_carry,
    parse_memory_config,
    resolve_memory_config,
    zeros_carry_like,
)

KINDS = ["gru", "lstm", "transformer", "mamba"]
T, B, F = 7, 4, 5


def make_cell(kind, num_layers=1, hidden_size=8):
    # window < T so the transformer's sliding window is actually exercised
    config = MemoryConfig(
        kind=kind, hidden_size=hidden_size, num_layers=num_layers, window=5
    )
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


def test_resolve_memory_config_legacy_lstm_hidden_size_maps_to_gru():
    cfg = resolve_memory_config(None, lstm_hidden_size=32)
    assert cfg == MemoryConfig(kind="gru", hidden_size=32)
    # explicit memory wins over the legacy field
    explicit = MemoryConfig(kind="lstm", hidden_size=8)
    assert resolve_memory_config(explicit, lstm_hidden_size=32) is explicit


def test_config_is_hashable():
    hash(MemoryConfig())  # required for jit-static args and module fields


# ---------------------------------------------------------------- shapes


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_layers", [1, 2])
def test_output_and_carry_shapes(kind, num_layers):
    cell, params, carry, x, resets = make_cell(kind, num_layers=num_layers)
    new_carry, y = cell.apply(params, carry, x, resets)
    assert y.shape == (T, B, 8)
    assert isinstance(new_carry, tuple) and len(new_carry) == num_layers
    # carry structure must be stable (required by lax.scan carries)
    assert jax.tree.structure(new_carry) == jax.tree.structure(carry)
    for out_leaf, in_leaf in zip(jax.tree.leaves(new_carry), jax.tree.leaves(carry)):
        assert out_leaf.shape == in_leaf.shape


@pytest.mark.parametrize("kind", KINDS)
def test_initial_carry_is_zeros(kind):
    # zeros_carry_like (used for eval and replayed sequences) relies on this
    config = MemoryConfig(kind=kind, hidden_size=8)
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


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_layers", [1, 2])
def test_step_equals_sequence(kind, num_layers):
    cell, params, carry, x, _ = make_cell(kind, num_layers=num_layers)
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
    for a, b in zip(jax.tree.leaves(seq_carry), jax.tree.leaves(step_carry)):
        assert jnp.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------- resets


@pytest.mark.parametrize("kind", KINDS)
def test_reset_equals_fresh_start(kind):
    cell, params, carry, x, _ = make_cell(kind)
    t_reset = 4
    resets = jnp.zeros((T, B), dtype=bool).at[t_reset].set(True)

    _, y_full = cell.apply(params, carry, x, resets)
    fresh_carry = init_carry(cell.config, jax.random.PRNGKey(0), B)
    _, y_fresh = cell.apply(
        params, fresh_carry, x[t_reset:], jnp.zeros((T - t_reset, B), dtype=bool)
    )
    assert jnp.allclose(y_full[t_reset:], y_fresh, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_no_gradient_across_reset(kind):
    cell, params, carry, x, _ = make_cell(kind)
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
    config = MemoryConfig(kind=kind, hidden_size=8, window=T + 1)
    cell = MemoryCell(config)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(1), (T, B, F))
    resets = jnp.zeros((T, B), dtype=bool)
    carry = init_carry(config, rng, B)
    params = cell.init(rng, carry, x, resets)

    def out_at_end(x_in):
        _, y = cell.apply(params, carry, x_in, resets)
        return y[-1].sum()

    grads = jax.grad(out_at_end)(x)
    assert jnp.any(grads[0] != 0)


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
            jax.tree.leaves(new_carry), jax.tree.leaves(carry)
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


@pytest.mark.parametrize("kind", KINDS)
def test_jit_and_gradient_checkpoint(kind):
    config = MemoryConfig(kind=kind, hidden_size=8, gradient_checkpoint=True)
    cell = MemoryCell(config)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (T, B, F))
    resets = jnp.zeros((T, B), dtype=bool)
    carry = init_carry(config, rng, B)
    params = cell.init(rng, carry, x, resets)

    @jax.jit
    def run(c, x_in):
        return cell.apply(params, c, x_in, resets)

    new_carry, y = run(carry, x)
    assert y.shape == (T, B, 8)
    grads = jax.grad(lambda x_in: run(carry, x_in)[1].sum())(x)
    assert jnp.all(jnp.isfinite(grads))
