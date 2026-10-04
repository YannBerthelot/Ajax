# Contains code derived from danijar/dreamerv3 at commit 29eb964
# (29eb964e2918a3f4db04086f7f51b60388e97f3d): _symexp is jaxutils.symexp
# (dreamerv3/jaxutils.py:80-81), ref_bins the odd-n 'symexp_twohot' bins of
# dreamerv3/nets.py:467-471, and ref_decode the odd-n branch of
# TwoHotDist.mean (dreamerv3/jaxutils.py:233-243).
# Modified: rewritten as free functions with the signature of Ajax's
# TwoHot.decode (softmax of the logits, float32), for the identity transform
# only; install() patches them into Ajax at import time.
#
# Copyright (c) 2023 Danijar Hafner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The license text is also in LICENSE.dreamerv3 in this directory.
"""The reference's two-hot expectation, literally (29eb964 dreamerv3/jaxutils.py:222-243 with the
bins of nets.py:466-476 computed in float32 under jit), as a drop-in for ajax TwoHot.decode.
Used by run_ajax_refdecode.py (round 2) and, before that, by the investigation's probe
scripts (DIAGNOSIS.md), always as a monkeypatch at import time; never written into the
Ajax tree."""

import jax
import jax.numpy as jnp

f32 = jnp.float32


def _symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def ref_bins(n=255):
    assert n % 2 == 1
    half = jnp.linspace(-20, 0, (n - 1) // 2 + 1, dtype=f32)
    half = _symexp(half)
    return jnp.concatenate([half, -half[:-1][::-1]], 0)


def ref_decode(self, logits):
    assert self.transform == "identity" and self.limit == 20.0
    logits = jnp.asarray(logits, f32)
    probs = jax.nn.softmax(logits)
    bins = ref_bins(self.num_bins)
    n = logits.shape[-1]
    m = (n - 1) // 2
    p1, p2, p3 = probs[..., :m], probs[..., m : m + 1], probs[..., m + 1 :]
    b1, b2, b3 = bins[..., :m], bins[..., m : m + 1], bins[..., m + 1 :]
    return (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)


def install():
    from ajax.distributional import TwoHot

    TwoHot._ajax_decode = TwoHot.decode
    TwoHot.decode = ref_decode
