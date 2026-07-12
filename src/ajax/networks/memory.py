"""Pluggable memory (recurrent) modules for actor/critic networks.

Every memory architecture unifies behind one interface, operating on
time-major sequences ``(T, B, features)`` with per-step reset flags
``(T, B)``:

- ``initialize_carry(rng, batch_size) -> carry`` returns the recurrent
  state as a pytree (always a tuple with one entry per layer).
- ``__call__(carry, x, resets) -> (new_carry, y)`` runs the cell over the
  whole sequence, isolating episodes wherever ``resets[t]`` is set, and
  returns the final carry plus the per-step outputs ``(T, B, hidden)``.

Step-wise inference (collection, evaluation) is the same code path with
``T=1``, so training and acting can never diverge. The step-vs-sequence
equivalence tests in ``tests/networks/test_memory.py`` are the acceptance
harness for every kind.

Carries are zero at initialization for every supported kind; callers rely
on this to build fresh carries with ``zeros_carry_like`` (e.g. at eval
time or for replayed sequences).

Kinds and their compute/memory profile:

- ``"gru"`` / ``"lstm"``: ``nn.scan`` over the classic cells — sequential
  in T, carry is O(hidden) per env.
- ``"transformer"``: sliding-window causal self-attention. Training runs
  FULL PARALLEL attention over the chunk (no scan) with an episode-segment
  mask; acting keeps a rolling cache of the last ``window - 1`` embeddings
  per layer, so per-step memory is O(window * hidden) and constant in
  episode length.
- ``"mamba"``: selective state-space (S6-style) block. Training runs a
  parallel ``associative_scan`` over T; the carry is the SSM state plus a
  small causal-conv tail — RNN-sized memory with transformer-like training
  parallelism.
"""

import functools
from dataclasses import dataclass
from typing import Any, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

Carry = Any

_RNN_CELL_CLASSES = {
    "gru": nn.GRUCell,
    "lstm": nn.OptimizedLSTMCell,
}
MEMORY_KINDS = ("gru", "lstm", "transformer", "mamba")


@dataclass(frozen=True)
class MemoryConfig:
    """Single-hyperparameter surface for adding memory to any network.

    Frozen (hashable) so it can be a Flax module field and a jit-static
    argument. ``kind`` selects the architecture; everything else has a
    sane default. Kind-specific knobs are ignored by the other kinds.
    """

    kind: str = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    # Recompute activations in the backward pass instead of storing them;
    # trades compute for memory on long sequences.
    gradient_checkpoint: bool = False
    # transformer: attention span (a step attends to itself and the
    # window-1 preceding steps of the same episode) and head count.
    window: int = 32
    num_heads: int = 4
    # mamba: SSM state size, causal-conv width, inner-width multiplier.
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    def __post_init__(self):
        if self.kind not in MEMORY_KINDS:
            raise ValueError(
                f"Unknown memory kind '{self.kind}', expected one of {MEMORY_KINDS}"
            )
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.kind == "transformer":
            if self.window < 1:
                raise ValueError(f"window must be >= 1, got {self.window}")
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_heads, got"
                    f" {self.hidden_size} and {self.num_heads}"
                )
        if self.kind == "mamba":
            if self.d_state < 1 or self.d_conv < 1 or self.expand < 1:
                raise ValueError(
                    "d_state, d_conv and expand must be >= 1, got"
                    f" {self.d_state}, {self.d_conv}, {self.expand}"
                )


def parse_memory_config(
    memory: Optional[Union[MemoryConfig, dict]],
) -> Optional[MemoryConfig]:
    """Coerce user input (None | MemoryConfig | dict) into a MemoryConfig."""
    if memory is None or isinstance(memory, MemoryConfig):
        return memory
    if isinstance(memory, dict):
        return MemoryConfig(**memory)
    raise TypeError(
        f"memory must be None, a MemoryConfig or a dict, got {type(memory)}"
    )


def resolve_memory_config(
    memory: Optional[Union[MemoryConfig, dict]],
    lstm_hidden_size: Optional[int] = None,
) -> Optional[MemoryConfig]:
    """Resolve the memory config, honouring the legacy ``lstm_hidden_size``.

    The historical ``lstm_hidden_size`` hyperparameter actually built a GRU
    (see the original ``ScannedRNN``); the mapping preserves that behaviour.
    ``memory`` wins when both are provided.
    """
    memory = parse_memory_config(memory)
    if memory is not None:
        return memory
    if lstm_hidden_size is not None:
        return MemoryConfig(kind="gru", hidden_size=lstm_hidden_size)
    return None


def init_carry(config: MemoryConfig, rng: jax.Array, batch_size: int) -> Carry:
    """Fresh (all-zero) carry for ``config``: a tuple with one entry per
    layer. Shapes are derived from the config alone — no params needed."""
    B, H = batch_size, config.hidden_size
    if config.kind in _RNN_CELL_CLASSES:
        cell = _RNN_CELL_CLASSES[config.kind](features=H, parent=None)
        return tuple(
            cell.initialize_carry(rng, (B, H)) for _ in range(config.num_layers)
        )
    if config.kind == "transformer":
        cache_len = config.window - 1
        return tuple(
            (
                jnp.zeros((B, cache_len, H)),  # cached layer inputs
                jnp.zeros((B, cache_len), dtype=bool),  # validity flags
            )
            for _ in range(config.num_layers)
        )
    # mamba
    d_inner = config.expand * H
    return tuple(
        (
            jnp.zeros((B, config.d_conv - 1, d_inner)),  # causal-conv tail
            jnp.zeros((B, d_inner, config.d_state)),  # SSM state
        )
        for _ in range(config.num_layers)
    )


def zeros_carry_like(carry: Carry, batch_size: int, batch_axis: int = 0) -> Carry:
    """Fresh carry with the batch dimension resized to ``batch_size``.

    Valid because every supported cell initializes its carry to zeros.
    ``batch_axis`` is 0 for a single network's carry and 1 for an
    ensemble's stacked carry (leading ``(num_critics,)`` axis).
    """

    def zeros(leaf):
        shape = list(leaf.shape)
        shape[batch_axis] = batch_size
        return jnp.zeros(tuple(shape), leaf.dtype)

    return jax.tree.map(zeros, carry)


def _mask_reset(resets: jax.Array, fresh: Carry, carry: Carry) -> Carry:
    """Replace ``carry`` with ``fresh`` for every batch entry where
    ``resets`` is set. ``resets`` has shape (B,) (or (B, 1))."""
    resets = resets.reshape((resets.shape[0],)).astype(bool)

    def select(fresh_leaf, carry_leaf):
        mask = resets.reshape((-1,) + (1,) * (carry_leaf.ndim - 1))
        return jnp.where(mask, fresh_leaf, carry_leaf)

    return jax.tree.map(select, fresh, carry)


def _segment_ids(resets: jax.Array) -> jax.Array:
    """Episode segment ids along time: seg[t] counts the resets at or
    before t, so seg==0 means 'still the episode the carry belongs to'."""
    return jnp.cumsum(resets.astype(jnp.int32), axis=0)


class _RecurrentCore(nn.Module):
    """GRU/LSTM: reset-aware ``nn.scan`` over the classic cells."""

    config: MemoryConfig

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x, resets):
        fresh = init_carry(self.config, jax.random.PRNGKey(0), x.shape[0])
        carry = _mask_reset(resets, fresh, carry)

        cell_cls = _RNN_CELL_CLASSES[self.config.kind]
        if self.config.gradient_checkpoint:
            cell_cls = nn.remat(cell_cls)

        new_carry = []
        hidden = x
        for layer in range(self.config.num_layers):
            cell = cell_cls(
                features=self.config.hidden_size,
                name=f"{self.config.kind}_{layer}",
            )
            layer_carry, hidden = cell(carry[layer], hidden)
            new_carry.append(layer_carry)
        return tuple(new_carry), hidden


class _TransformerCore(nn.Module):
    """Sliding-window causal self-attention (GTrXL-flavoured, pre-LN).

    Per layer the carry holds the last ``window - 1`` layer INPUTS and
    their validity flags; a step attends to itself plus up to window-1
    predecessors from the same episode. Sequence mode concatenates the
    cache in front of the chunk and runs one parallel attention with a
    (causal ∧ windowed ∧ same-episode) mask — no scan — so step-by-step
    and full-sequence execution are numerically identical. A learned
    relative-distance bias replaces absolute positions, which keeps the
    computation shift-invariant (required for streaming equivalence).
    """

    config: MemoryConfig

    @nn.compact
    def __call__(self, carry, x, resets):
        cfg = self.config
        T, B = x.shape[:2]
        H, W, n_heads = cfg.hidden_size, cfg.window, cfg.num_heads
        head_dim = H // n_heads
        cache_len = W - 1

        resets = resets.reshape((T, B)).astype(bool)
        seg = _segment_ids(resets)  # (T, B)
        seg_bt = jnp.swapaxes(seg, 0, 1)  # (B, T)

        hidden = nn.Dense(H, name="embed")(x)  # (T, B, H)

        # Query position t / key position p masks, key stream = cache ++ chunk.
        t_idx = jnp.arange(T)
        # sequence-key part: causal, windowed, same episode
        s_idx = jnp.arange(T)
        dist_seq = t_idx[:, None] - s_idx[None, :]  # (T, T)
        seq_ok = (dist_seq >= 0) & (dist_seq < W)  # causal + window
        same_ep = seg_bt[:, :, None] == seg_bt[:, None, :]  # (B, T, T)
        mask_seq = seq_ok[None] & same_ep  # (B, T, T)
        # cache-key part: position p = j - cache_len, so dist = t - j + cache_len
        j_idx = jnp.arange(cache_len)
        dist_cache = t_idx[:, None] + cache_len - j_idx[None, :]  # (T, C)
        cache_in_window = dist_cache < W  # dist >= 1 always
        no_reset_yet = seg_bt == 0  # (B, T): cache is same-episode only then

        new_carry = []
        for layer in range(cfg.num_layers):
            cache, valid = carry[layer]  # (B, C, H), (B, C)
            h_bt = jnp.swapaxes(hidden, 0, 1)  # (B, T, H) — layer input
            stream = jnp.concatenate([cache, h_bt], axis=1)  # (B, C+T, H)

            normed = nn.LayerNorm(name=f"ln_attn_{layer}")(stream)
            q = nn.Dense(H, name=f"q_{layer}")(normed[:, cache_len:])
            k = nn.Dense(H, name=f"k_{layer}")(normed)
            v = nn.Dense(H, name=f"v_{layer}")(normed)
            q = q.reshape(B, T, n_heads, head_dim)
            k = k.reshape(B, cache_len + T, n_heads, head_dim)
            v = v.reshape(B, cache_len + T, n_heads, head_dim)

            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
            # learned relative-distance bias, dist in [0, W-1]
            rel_bias = self.param(
                f"rel_bias_{layer}", nn.initializers.zeros, (n_heads, W)
            )
            dist_all = jnp.concatenate([dist_cache, dist_seq], axis=1)  # (T, C+T)
            dist_clipped = jnp.clip(dist_all, 0, W - 1)
            logits = logits + rel_bias[:, dist_clipped][None]  # (B, h, T, C+T)

            mask_cache = (
                valid[:, None, :]  # (B, 1, C)
                & no_reset_yet[:, :, None]  # (B, T, 1)
                & cache_in_window[None]  # (1, T, C)
            )
            mask = jnp.concatenate([mask_cache, mask_seq], axis=-1)  # (B, T, C+T)
            logits = jnp.where(mask[:, None], logits, -jnp.inf)
            # a query always attends at least itself, so no all-masked rows
            attn = jax.nn.softmax(logits, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, T, H)
            h_bt = h_bt + nn.Dense(H, name=f"attn_out_{layer}")(out)

            mlp_in = nn.LayerNorm(name=f"ln_mlp_{layer}")(h_bt)
            mlp = nn.Dense(2 * H, name=f"mlp_up_{layer}")(mlp_in)
            mlp = nn.Dense(H, name=f"mlp_down_{layer}")(nn.gelu(mlp))
            h_bt = h_bt + mlp

            # New cache: last C entries of the input stream, valid only if
            # they belong to the episode running at the END of the chunk
            # (future queries re-check 'no reset since' themselves).
            last_seg = seg_bt[:, -1:]  # (B, 1)
            valid_cache_part = valid & (last_seg == 0)  # (B, C)
            valid_seq_part = seg_bt == last_seg  # (B, T)
            stream_valid = jnp.concatenate(
                [valid_cache_part, valid_seq_part], axis=1
            )  # (B, C+T)
            new_carry.append(
                (
                    stream[:, stream.shape[1] - cache_len :],
                    stream_valid[:, stream_valid.shape[1] - cache_len :],
                )
            )
            hidden = jnp.swapaxes(h_bt, 0, 1)  # (T, B, H) for the next layer

        return tuple(new_carry), hidden


class _MambaCore(nn.Module):
    """Selective state-space (S6-style) block with parallel training.

    Per layer: LN → in-projection to (u, gate) → reset-aware causal
    depthwise conv → input-dependent (Δ, B, C) selective SSM evaluated
    with ``jax.lax.associative_scan`` over time → gate → out-projection,
    with a residual connection. Resets zero the state-transition Ā at
    episode starts, which cuts both the carried state and all cross-episode
    history in one place — valid for the scan and the step path alike.
    """

    config: MemoryConfig

    @nn.compact
    def __call__(self, carry, x, resets):
        cfg = self.config
        T, B = x.shape[:2]
        H = cfg.hidden_size
        d_inner, d_state, d_conv = cfg.expand * H, cfg.d_state, cfg.d_conv

        resets = resets.reshape((T, B)).astype(bool)
        seg = _segment_ids(resets)  # (T, B)

        hidden = nn.Dense(H, name="embed")(x)  # (T, B, H)

        new_carry = []
        for layer in range(cfg.num_layers):
            conv_tail, ssm_state = carry[layer]
            inp = hidden
            z = nn.LayerNorm(name=f"ln_{layer}")(inp)
            u, gate = jnp.split(
                nn.Dense(2 * d_inner, name=f"in_proj_{layer}")(z), 2, axis=-1
            )  # each (T, B, d_inner)

            # Reset-aware causal depthwise conv, unrolled over the (small)
            # kernel. The stored tail is pre-masked to the episode running
            # at storage time, so here only within-chunk episode identity
            # (and 'no reset yet' for tail taps) needs checking.
            conv_w = self.param(
                f"conv_w_{layer}",
                nn.initializers.lecun_normal(),
                (d_conv, d_inner),
            )
            conv_b = self.param(f"conv_b_{layer}", nn.initializers.zeros, (d_inner,))
            tail = jnp.moveaxis(conv_tail, 1, 0)  # (d_conv-1, B, d_inner)
            ext = jnp.concatenate([tail, u], axis=0)  # (d_conv-1+T, B, d)
            seg_ext = jnp.concatenate(
                [jnp.zeros((d_conv - 1, B), seg.dtype), seg], axis=0
            )
            conv_out = jnp.zeros_like(u) + conv_b
            for i in range(d_conv):
                start = d_conv - 1 - i
                x_shift = jax.lax.dynamic_slice_in_dim(ext, start, T, axis=0)
                seg_shift = jax.lax.dynamic_slice_in_dim(seg_ext, start, T, axis=0)
                tap_ok = (seg == seg_shift).astype(u.dtype)[..., None]
                conv_out = conv_out + conv_w[i] * x_shift * tap_ok
            u = nn.silu(conv_out)

            # Selective SSM with input-dependent discretization.
            A_log = self.param(
                f"A_log_{layer}",
                lambda key, shape: jnp.broadcast_to(
                    jnp.log(jnp.arange(1, d_state + 1, dtype=jnp.float32)),
                    shape,
                ),
                (d_inner, d_state),
            )
            D = self.param(f"D_{layer}", nn.initializers.ones, (d_inner,))
            delta = nn.softplus(
                nn.Dense(
                    d_inner,
                    name=f"dt_proj_{layer}",
                    bias_init=nn.initializers.constant(-3.0),
                )(u)
            )  # (T, B, d_inner); softplus(-3) ≈ 0.049 initial step size
            B_mat = nn.Dense(d_state, name=f"B_proj_{layer}")(u)  # (T, B, N)
            C_mat = nn.Dense(d_state, name=f"C_proj_{layer}")(u)  # (T, B, N)

            A_bar = jnp.exp(
                -delta[..., None] * jnp.exp(A_log)[None, None]
            )  # (T, B, d_inner, N)
            # Zero the transition at episode starts: kills the carried
            # state and any within-chunk history crossing the boundary.
            A_bar = jnp.where(resets[..., None, None], 0.0, A_bar)
            Bx = (
                delta[..., None] * B_mat[:, :, None, :] * u[..., None]
            )  # (T, B, d_inner, N)

            def _combine(left, right):
                a_l, b_l = left
                a_r, b_r = right
                return a_r * a_l, a_r * b_l + b_r

            a_cum, h_scan = jax.lax.associative_scan(_combine, (A_bar, Bx), axis=0)
            h = h_scan + a_cum * ssm_state[None]  # (T, B, d_inner, N)

            y = jnp.einsum("tbdn,tbn->tbd", h, C_mat) + D * u
            y = y * nn.silu(gate)
            out = nn.Dense(H, name=f"out_proj_{layer}")(y)
            hidden = inp + out

            # New carry: final SSM state, and the last d_conv-1 conv inputs
            # pre-masked to the episode running at the end of the chunk.
            tail_len = d_conv - 1
            new_tail = jax.lax.dynamic_slice_in_dim(
                ext, ext.shape[0] - tail_len, tail_len, axis=0
            )
            seg_tail = jax.lax.dynamic_slice_in_dim(
                seg_ext, seg_ext.shape[0] - tail_len, tail_len, axis=0
            )
            tail_ok = (seg_tail == seg[-1][None]).astype(u.dtype)[..., None]
            new_conv_tail = jnp.moveaxis(new_tail * tail_ok, 0, 1)
            new_carry.append((new_conv_tail, h[-1]))

        return tuple(new_carry), hidden


_CORE_CLASSES = {
    "gru": _RecurrentCore,
    "lstm": _RecurrentCore,
    "transformer": _TransformerCore,
    "mamba": _MambaCore,
}


class MemoryCell(nn.Module):
    """Reset-aware memory block; dispatches to the core selected by
    ``config.kind``. See the module docstring for the interface contract.
    """

    config: MemoryConfig

    def setup(self):
        core_cls = _CORE_CLASSES[self.config.kind]
        if self.config.gradient_checkpoint and core_cls is not _RecurrentCore:
            # RNN cores remat per-step inside their scan instead.
            core_cls = nn.remat(core_cls)
        self.core = core_cls(self.config, name="core")

    def __call__(self, carry, x, resets):
        return self.core(carry, x, resets)

    def initialize_carry(self, rng: jax.Array, batch_size: int) -> Carry:
        """Fresh (zero) carry: a tuple with one entry per layer."""
        return init_carry(self.config, rng, batch_size)
