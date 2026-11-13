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
import optax
import pytest

from ajax.networks.memory import (
    MEMORY_INITS,
    MEMORY_KINDS,
    MemoryCell,
    MemoryConfig,
    flat_carry_dim,
    flatten_carry,
    init_carry,
    parse_memory_config,
    unflatten_carry,
    with_default_init,
    zeros_carry_like,
)

KINDS = MEMORY_KINDS
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
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        MemoryConfig(kind="transformer", layer_norm_epsilon=0.0)
    with pytest.raises(ValueError, match="mlp_ratio"):
        MemoryConfig(kind="transformer", mlp_ratio=0)


def test_config_refuses_an_init_of_another_kind():
    for kind, inits in MEMORY_INITS.items():
        for init in (None, *inits):
            MemoryConfig(kind=kind, init=init)
    with pytest.raises(ValueError, match="does not apply"):
        MemoryConfig(kind="gru", init="gpt2")
    with pytest.raises(ValueError, match="does not apply"):
        MemoryConfig(kind="mamba", init="stacked_orthogonal")


def test_with_default_init_fills_only_an_unset_init_of_a_named_kind():
    inits = {"gru": "stacked_orthogonal"}
    assert with_default_init(None, inits) is None
    gru = with_default_init(MemoryConfig("gru"), inits)
    assert gru == MemoryConfig("gru", init="stacked_orthogonal")
    assert with_default_init(MemoryConfig("gru", init="flax"), inits).init == "flax"
    assert with_default_init(MemoryConfig("lstm"), inits).init is None


def test_parse_memory_config_accepts_dict_and_none():
    assert parse_memory_config(None) is None
    cfg = parse_memory_config({"kind": "lstm", "hidden_size": 16})
    assert cfg == MemoryConfig(kind="lstm", hidden_size=16)


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
    for out_leaf, in_leaf in zip(
        jax.tree.leaves(new_carry), jax.tree.leaves(carry), strict=False
    ):
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
    for a, b in zip(
        jax.tree.leaves(seq_carry), jax.tree.leaves(step_carry), strict=False
    ):
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
    cell, params, carry, x, resets = make_cell(kind, window=T + 1)

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


# ------------------------------------------------- key/value cache (APG)
# The default cell, caching inputs, is the reference: a carry built under the
# parameters that read it must give the same outputs and gradients either way.


def _with_key_value_twin(num_layers):
    """A transformer cell, its twin caching keys and values on the same
    parameters with its own fresh carry, a (T, B, F) input, and resets at
    the start and in the middle of one env's sequence."""
    cell, params, carry, x, _ = make_cell("transformer", num_layers=num_layers)
    twin = MemoryCell(cell.config, cache_keys_values=True)
    twin_carry = init_carry(cell.config, jax.random.PRNGKey(0), B, True)
    resets = jnp.zeros((T, B), dtype=bool).at[0].set(True).at[3, 1].set(True)
    return cell, twin, params, carry, twin_carry, x, resets


@pytest.mark.parametrize("num_layers", [1, 2])
def test_caching_keys_and_values_gives_the_outputs_of_caching_inputs(num_layers):
    """Catches a key/value cache that slides its window by the wrong amount,
    keeps a row across a reset or attends to a stale one: stepped one row at
    a time, the twin must give the default cell's outputs over the whole
    sequence (window 5 < T = 7, so rows also leave the window)."""
    cell, twin, params, carry, twin_carry, x, resets = _with_key_value_twin(num_layers)
    _, expected = cell.apply(params, carry, x, resets)
    outputs = []
    for t in range(T):
        twin_carry, y = twin.apply(params, twin_carry, x[t : t + 1], resets[t : t + 1])
        outputs.append(y[0])
    assert jnp.allclose(jnp.stack(outputs), expected, atol=1e-5)


def test_caching_keys_and_values_keeps_the_gradient_through_the_carry():
    """Catches cached keys and values cut from the gradient, or computed
    from other parameters than the step's: APG differentiates its return
    through the carry, so a stepped rollout's gradient with respect to the
    parameters must be the default cell's."""
    cell, twin, params, carry, twin_carry, x, resets = _with_key_value_twin(2)

    def stepped_loss(network, initial_carry):
        def loss(params):
            state, total = initial_carry, 0.0
            for t in range(T):
                state, y = network.apply(params, state, x[t : t + 1], resets[t : t + 1])
                total = total + jnp.sum(jnp.sin(y + t))
            return total

        return loss

    expected = jax.grad(stepped_loss(cell, carry))(params)
    gradient = jax.grad(stepped_loss(twin, twin_carry))(params)
    gap = optax.global_norm(jax.tree.map(jnp.subtract, gradient, expected))
    assert gap <= 1e-5 * optax.global_norm(expected)


@pytest.mark.parametrize("kind", ["gru", "lstm", "mamba"])
def test_only_a_transformer_caches_keys_and_values(kind):
    """Catches the option ignored on a memory that has no keys: asked of
    another kind, the carry and the cell refuse it."""
    config = MemoryConfig(kind=kind, hidden_size=8)
    with pytest.raises(ValueError, match="transformer"):
        init_carry(config, jax.random.PRNGKey(0), B, cache_keys_values=True)
    carry = init_carry(config, jax.random.PRNGKey(0), B)
    x, resets = jnp.zeros((T, B, F)), jnp.zeros((T, B), dtype=bool)
    with pytest.raises(ValueError, match="transformer"):
        MemoryCell(config, cache_keys_values=True).init(
            jax.random.PRNGKey(0), carry, x, resets
        )


# ------------------------------------------------------------ final norm


@pytest.mark.parametrize("kind", ["transformer", "mamba"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_output_ends_with_a_layer_norm(kind, num_layers):
    """The pre-norm stacks end with a final LayerNorm (GPT-2's ln_f, Mamba's
    norm_f): each step's output is its pre-norm stream normalised over the
    features, then scaled and shifted by ln_f's parameters. Without it
    (final_norm=False) the stream comes out raw and ln_f is absent."""
    cell, params, carry, x, resets = make_cell(kind, num_layers=num_layers)
    core = params["params"]["core"]
    assert set(core["ln_f"]) == {"scale", "bias"}
    scale = jax.random.uniform(jax.random.PRNGKey(2), (8,), minval=0.5, maxval=2)
    bias = jax.random.normal(jax.random.PRNGKey(3), (8,))
    params = {"params": {"core": {**core, "ln_f": {"scale": scale, "bias": bias}}}}
    _, y = cell.apply(params, carry, 50.0 * x, resets)
    z = (y - bias) / scale
    assert jnp.allclose(z.mean(-1), 0.0, atol=1e-4)
    assert jnp.allclose(z.var(-1), 1.0, atol=1e-3)

    raw, raw_params, *_ = make_cell(kind, num_layers=num_layers, final_norm=False)
    assert "ln_f" not in raw_params["params"]["core"]
    _, y_raw = raw.apply(raw_params, carry, 50.0 * x, resets)
    assert not jnp.allclose(y_raw.var(-1), 1.0, atol=0.1)


def test_a_transformer_without_bias_keeps_only_its_embeddings():
    """bias=False is nanoGPT's config.bias: no bias in the attention and MLP
    layers or the LayerNorms, ln_f included; the input embedding keeps one,
    as nanoGPT's wte does. mlp_ratio sets the MLP's width (GPT-2's 4)."""
    _, params, *_ = make_cell("transformer", bias=False, mlp_ratio=4, num_layers=2)
    core = params["params"]["core"]
    for name in ("q_0", "k_1", "v_0", "attn_out_1", "mlp_up_0", "mlp_down_1"):
        assert set(core[name]) == {"kernel"}, name
    for name in ("ln_attn_0", "ln_mlp_1", "ln_f"):
        assert set(core[name]) == {"scale"}, name
    assert set(core["embed"]) == {"kernel", "bias"}
    assert core["mlp_up_0"]["kernel"].shape == (8, 32)
    assert core["mlp_down_0"]["kernel"].shape == (32, 8)


# ---------------------------------------------------------------- init


def _core(kind, init, **overrides):
    _, params, *_ = make_cell(kind, init=init, hidden_size=64, **overrides)
    return params["params"]["core"]


@pytest.mark.parametrize("kind", KINDS)
def test_init_none_is_flax(kind):
    flax_init = jax.tree.leaves(_core(kind, "flax"))
    for a, b in zip(jax.tree.leaves(_core(kind, None)), flax_init, strict=True):
        assert (a == b).all()


@pytest.mark.parametrize("kind, n_gates", [("gru", 3), ("lstm", 4)])
def test_stacked_orthogonal_gives_each_gate_torchs_block(kind, n_gates):
    """Ni et al.'s GRU and LSTM: every kernel, input (F=5 wide) and
    recurrent (H=64), has std 1/sqrt(n_gates H), torch's per-gate share of
    an orthogonal stacked weight; biases 0. Flax's recurrent kernels are
    orthogonal per gate, std 1/sqrt(H)."""
    for init, want in (("stacked_orthogonal", n_gates * 64), ("flax", None)):
        cell = _core(kind, init)[f"{kind}_0"]
        for gate, dense in cell.items():
            if gate.startswith("h") or want:
                std = 1 / jnp.sqrt(want or 64)
                assert jnp.std(dense["kernel"]) == pytest.approx(std, rel=0.15)
            if "bias" in dense:
                assert (dense["bias"] == 0).all()


@pytest.mark.parametrize("init", ["gpt2", "nanogpt"])
def test_transformer_inits_follow_gpt2(init):
    """normal(0.02) on every Dense, biases 0; "nanogpt" scales the residual
    projections to 0.02/sqrt(2 num_layers), here 0.01."""
    core = _core("transformer", init, num_layers=2)
    for name, layer in core.items():
        # rel_bias_* is a bare array, and JAX 0.11 refuses `in` on an array.
        if not isinstance(layer, dict) or "kernel" not in layer:
            continue
        residual = name.startswith(("attn_out", "mlp_down"))
        std = 0.01 if init == "nanogpt" and residual else 0.02
        assert jnp.std(layer["kernel"]) == pytest.approx(std, rel=0.15), name
        assert (layer["bias"] == 0).all(), name


def test_mamba_init_draws_its_step_sizes_log_uniform():
    """softplus(dt bias) spans [1e-3, 0.1] log-uniformly over the 128
    channels; flax's is -3 everywhere."""
    dt = jax.nn.softplus(_core("mamba", "mamba")["dt_proj_0"]["bias"])
    assert dt.min() >= 1e-3 * (1 - 1e-5) and dt.max() <= 0.1 * (1 + 1e-5)
    assert jnp.median(jnp.log10(dt)) == pytest.approx(-2.0, abs=0.25)
    assert (_core("mamba", None)["dt_proj_0"]["bias"] == -3.0).all()


@pytest.mark.parametrize("kind", KINDS)
def test_ends_normalised_names_the_kinds_with_a_final_norm(kind):
    ends = kind in ("transformer", "mamba")
    assert MemoryConfig(kind=kind).ends_normalised == ends
    assert not MemoryConfig(kind=kind, final_norm=False).ends_normalised


# ---------------------------------------------------------------- jit


@pytest.mark.parametrize("kind", KINDS)
def test_jit_and_gradient_checkpoint(kind):
    cell, params, carry, x, resets = make_cell(
        kind, gradient_checkpoint=True, window=MemoryConfig().window
    )

    @jax.jit
    def run(c, x_in):
        return cell.apply(params, c, x_in, resets)

    new_carry, y = run(carry, x)
    assert y.shape == (T, B, 8)
    grads = jax.grad(lambda x_in: run(carry, x_in)[1].sum())(x)
    assert jnp.all(jnp.isfinite(grads))


@pytest.mark.parametrize("kind", KINDS)
def test_flatten_carry_roundtrip(kind):
    """flatten_carry/unflatten_carry must be exact inverses for every kind
    (stored-state replay writes flattened carries into the buffer)."""
    config = MemoryConfig(kind=kind, hidden_size=8, window=5, num_layers=2)
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
