"""Literal translations of the reference code, used as test oracles.

These functions are deliberately NOT idiomatic Ajax code: each one
reproduces an upstream function line for line, so that a test comparing
Ajax's modular implementation against it checks fidelity to the code that
produced the papers' numbers (``docs/world_models/DESIGN.md`` §10).

Sources (both MIT-licensed; copyright notices below):

* DreamerV3, ``danijar/dreamerv3`` at the paper-era commit
  ``2411f7d136832378c0291c587cdbf2fca6506873``. The code is already JAX; it
  is copied with the ninjax module state (``nj.Variable``, ``self.get``)
  replaced by explicit arguments and with only the branches the papers'
  configurations use. Copyright (c) 2023 Danijar Hafner.
* TD-MPC2, ``nicklashansen/tdmpc2`` at the paper-era commit
  ``5f6fadec0fec78304b4b53e8171d348b58cac486``. The torch code is ported
  to float32 numpy line for line (torch operations noted where the
  translation is not obvious). Copyright (c) Nicklas Hansen (2023).

Each function cites ``path:line`` at the pinned commit. Only the shared
blocks of milestone M1 are here; agent-level oracles live next to each
agent's tests.

MIT License (both projects): Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the
following conditions: The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

f32 = jnp.float32
i32 = jnp.int32


# ============================================================== DreamerV3
# danijar/dreamerv3@2411f7d


# dreamerv3/jaxutils.py:76-77
def d_symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


# dreamerv3/jaxutils.py:80-81
def d_symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


# dreamerv3/nets.py:462-466 (Dist.inner, dist == 'symexp_twohot', odd bins);
# ``n`` is ``out.shape[-1]`` after the padding logit is dropped (:432-438).
def d_symexp_twohot_bins(n=255):
    half = jnp.linspace(-20, 0, (n - 1) // 2 + 1, dtype=f32)
    half = d_symexp(half)
    bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
    return bins


# dreamerv3/jaxutils.py:255-271 (TwoHotDist.log_prob, the target part;
# transfwd is the identity for 'symexp_twohot').
def d_twohot_target(x, bins):
    below = (bins <= x[..., None]).astype(i32).sum(-1) - 1
    above = len(bins) - (bins > x[..., None]).astype(i32).sum(-1)
    below = jnp.clip(below, 0, len(bins) - 1)
    above = jnp.clip(above, 0, len(bins) - 1)
    equal = below == above
    dist_to_below = jnp.where(equal, 1, jnp.abs(bins[below] - x))
    dist_to_above = jnp.where(equal, 1, jnp.abs(bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
        jax.nn.one_hot(below, len(bins)) * weight_below[..., None]
        + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
    )
    return target


# dreamerv3/jaxutils.py:255-274 (TwoHotDist.log_prob, dims=0); the loss is
# its negation.
def d_twohot_log_prob(logits, x, bins):
    target = d_twohot_target(x, bins)
    log_pred = logits - jax.scipy.special.logsumexp(logits, -1, keepdims=True)
    return (target * log_pred).sum(-1)


# dreamerv3/jaxutils.py:218, :226-243 (TwoHotDist.probs and .mean, odd n).
def d_twohot_mean(logits, bins):
    probs = jax.nn.softmax(logits)
    n = logits.shape[-1]
    m = (n - 1) // 2
    p1 = probs[..., :m]
    p2 = probs[..., m : m + 1]
    p3 = probs[..., m + 1 :]
    b1 = bins[..., :m]
    b2 = bins[..., m : m + 1]
    b3 = bins[..., m + 1 :]
    wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
    return wavg


class DMoments:
    """dreamerv3/jaxutils.py:301-395, ``Moments(impl='perc')``.

    Configured as ``retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo:
    5.0, perchi: 95.0}`` (dreamerv3/configs.yaml:141). Single-device branch
    of ``update`` (``per = jnp.percentile``, :343).
    """

    def __init__(self, rate=0.01, limit=1.0, perclo=5.0, perchi=95.0):
        self.rate, self.limit = rate, limit
        self.perclo, self.perchi = perclo, perchi
        self.low = jnp.zeros((), f32)  # :320
        self.high = jnp.zeros((), f32)  # :321

    def __call__(self, x, update=True):  # :329-331
        update and self.update(x)
        return self.stats()

    def update(self, x):  # :333-359, impl == 'perc'
        per = jnp.percentile
        x = jax.lax.stop_gradient(x.astype(f32))
        m = self.rate
        low, high = per(x, self.perclo), per(x, self.perchi)
        self.low = (1 - m) * self.low + m * low
        self.high = (1 - m) * self.high + m * high

    def stats(self):  # :382-386, impl == 'perc'
        offset = self.low
        span = self.high - self.low
        span = jnp.maximum(self.limit, span)
        return jax.lax.stop_gradient(offset), jax.lax.stop_gradient(span)


# dreamerv3/nets.py:753-758 (Norm._norm, impl == 'rms', eps = 1e-4 from
# :728). At 2411f7d the statistics are computed in the input dtype; the
# oracle is used with float32 inputs.
def d_rms_norm(x, scale, eps=1e-4):
    dtype = x.dtype
    x = f32(x) if x.dtype == jnp.float16 else x
    scale = scale.astype(x.dtype)
    mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + eps) * scale
    return (x * mult).astype(dtype)


# dreamerv3/nets.py:873-884 (Initializer._fans, block_fans=False).
def d_fans(shape):
    if len(shape) == 0:
        return (1, 1)
    elif len(shape) == 1:
        return (1, shape[0])
    elif len(shape) == 2:
        return shape
    else:
        space = int(np.prod(shape[:-2]))
        return (shape[-2] * space, shape[-1] * space)


# dreamerv3/nets.py:830-848, :870 (Initializer.__call__, dist == 'normal',
# fan == 'in', VARIANCE_FACTOR = 1, FORCE_STDDEV = 0); ``scale`` is the
# layer's outscale (Linear passes it at :608-609). ``nj.seed()`` -> ``key``.
def d_init_normal(key, shape, scale=1.0):
    fanin, _fanout = d_fans(shape)
    fan = fanin
    value = jax.random.truncated_normal(key, -2, 2, shape)
    value *= 1.1368 * np.sqrt(1.0 / fan)
    value = value.astype(f32)
    value *= scale
    return value


# dreamerv3/nets.py:387-392 (MLP.__call__ hidden layers) with each layer
# Linear(units, act='silu', norm='rms'): nets.py:613-618 (__call__),
# :620-635 (_layer: ``x @ kernel + bias``), Norm rms, get_act -> jax.nn.silu
# (:903-904). ``layers`` holds (kernel, bias, norm scale) per layer.
def d_mlp(x, layers: Sequence[Tuple[jax.Array, jax.Array, jax.Array]], bdims=2):
    feat = x
    x = feat.reshape([-1, feat.shape[-1]])
    for kernel, bias, scale in layers:
        x = x @ kernel.astype(x.dtype)
        x += bias.astype(x.dtype)
        x = d_rms_norm(x, scale)
        x = jax.nn.silu(x)
    x = x.reshape((*feat.shape[:bdims], -1))
    return x


# ================================================================ TD-MPC2
# nicklashansen/tdmpc2@5f6fade, torch -> float32 numpy.

T_NUM_BINS, T_VMIN, T_VMAX = 101, -10.0, 10.0  # tdmpc2/config.yaml:48-50


# tdmpc2/common/math.py:48-54
def t_symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))


# tdmpc2/common/math.py:57-63
def t_symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


# tdmpc2/common/math.py:66-78 (num_bins >= 2 branch); x: (N, 1) float32.
# ``scatter_`` writes 1 - offset at bin_idx, then offset at
# (bin_idx + 1) % num_bins.
def t_two_hot(x, num_bins=T_NUM_BINS, vmin=T_VMIN, vmax=T_VMAX):
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = np.clip(t_symlog(x), vmin, vmax).squeeze(1).astype(np.float32)
    bin_idx = np.floor((x - vmin) / bin_size).astype(np.int64)
    bin_offset = ((x - vmin) / bin_size - bin_idx.astype(np.float32))[:, None]
    soft_two_hot = np.zeros((x.shape[0], num_bins), np.float32)
    rows = np.arange(x.shape[0])
    soft_two_hot[rows, bin_idx] = (1 - bin_offset)[:, 0]
    soft_two_hot[rows, (bin_idx + 1) % num_bins] = bin_offset[:, 0]
    return soft_two_hot


# tdmpc2/common/math.py:84-95 (num_bins >= 2 branch): softmax, naive sum
# over torch.linspace bins, symexp; returns (N, 1).
def t_two_hot_inv(x, num_bins=T_NUM_BINS, vmin=T_VMIN, vmax=T_VMAX):
    dreg_bins = np.linspace(vmin, vmax, num_bins, dtype=np.float32)
    x = np.exp(x - x.max(-1, keepdims=True))
    x = x / x.sum(-1, keepdims=True)  # F.softmax(x, dim=-1)
    x = np.sum(x * dreg_bins, axis=-1, keepdims=True)
    return t_symexp(x)


# tdmpc2/common/math.py:5-9 (soft_ce); pred: (N, num_bins), target: (N, 1).
def t_soft_ce(pred, target):
    shifted = pred - pred.max(-1, keepdims=True)
    pred = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))  # log_softmax
    target = t_two_hot(target)
    return -(target * pred).sum(-1, keepdims=True)


class TRunningScale:
    """tdmpc2/common/scale.py:4-45 (RunningScale), with ``cfg.tau = 0.01``
    (tdmpc2/config.yaml:23)."""

    def __init__(self, tau=0.01):
        self._value = np.ones(1, np.float32)  # :9
        self._percentiles = np.array([5, 95], np.float32)  # :10
        self.tau = tau

    @property
    def value(self):  # :19-21
        return float(self._value[0])

    def _percentile(self, x):  # :23-35
        x_dtype, x_shape = x.dtype, x.shape
        x = x.reshape(x.shape[0], -1)
        in_sorted = np.sort(x, axis=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = np.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.astype(np.int64), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.astype(np.int64), :] * weight_ceiled[:, None]
        return (d0 + d1).reshape(-1, *x_shape[1:]).astype(x_dtype)

    def update(self, x):  # :37-40; torch lerp_ with weight < 0.5
        percentiles = self._percentile(x)
        value = np.maximum(percentiles[1] - percentiles[0], np.float32(1.0))
        self._value = self._value + np.float32(self.tau) * (value - self._value)

    def __call__(self, x, update=False):  # :42-45
        if update:
            self.update(x)
        return x * (1 / self.value)


# torch.nn.Mish, used as NormedLinear's default activation
# (tdmpc2/common/layers.py:90): x * tanh(softplus(x)).
def t_mish(x):
    return x * np.tanh(np.logaddexp(np.float32(0.0), x))


# torch.nn.LayerNorm(out_features) as built at tdmpc2/common/layers.py:92
# (eps 1e-5, biased variance, elementwise affine).
def t_layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + np.float32(eps)) * weight + bias


# tdmpc2/common/layers.py:96-100 (NormedLinear.forward). torch stores the
# weight as (out, in) and computes x @ W.T; ``kernel`` here is W.T. The
# dropout mask (1 = kept) replaces nn.Dropout(p) in train mode, which
# scales kept units by 1 / (1 - p).
def t_normed_linear(
    x,
    kernel,
    bias,
    ln_weight,
    ln_bias,
    dropout_p=0.0,
    dropout_mask: Optional[np.ndarray] = None,
):
    x = x @ kernel + bias
    if dropout_p and dropout_mask is not None:
        x = x * dropout_mask / np.float32(1 - dropout_p)
    return t_mish(t_layer_norm(x, ln_weight, ln_bias))


# tdmpc2/common/layers.py:110-122 (mlp), hidden NormedLinear layers only
# (dropout only in layer 0, :120). ``layers`` holds (kernel, bias, LN
# weight, LN bias) per layer.
def t_mlp_hidden(x, layers, dropout_p=0.0, dropout_mask=None):
    for i, (kernel, bias, ln_weight, ln_bias) in enumerate(layers):
        p = dropout_p * (i == 0)
        x = t_normed_linear(x, kernel, bias, ln_weight, ln_bias, p, dropout_mask)
    return x
