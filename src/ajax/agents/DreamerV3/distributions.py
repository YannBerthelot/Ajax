"""Output distributions of the DreamerV3 world model.

Three distributions, as in the paper-era code ``danijar/dreamerv3@2411f7d``
(the fidelity target; ``docs/world_models/deviations.md`` section 1), all in
float32:

* :class:`OneHot` -- the categorical latents: ``S`` independent categoricals
  over ``C`` classes with a 1 % uniform mixture, sampled as a straight-through
  one-hot (dreamerv3_spec 1.8; ``2411f7d:dreamerv3/nets.py:207-220``,
  ``jaxutils.py:103-121``).
* :func:`bernoulli_loss` -- the continue head, logistic regression on soft
  labels (dreamerv3_spec 1.10; tfp ``Bernoulli(logits).log_prob``).
* :func:`symlog_mse` -- the vector decoder, squared error in symlog space with
  the 2411f7d tolerance (dreamerv3_spec 1.11; ``jaxutils.py:179-207``).

The reward head's two-hot distribution is the shared
:class:`ajax.distributional.TwoHot` (``TwoHot.dreamerv3``).

Sampling takes its randomness as an explicit argument: :meth:`OneHot.sample`
consumes Gumbel noise drawn by :func:`draw_onehot_noise`, and returns
``argmax(log p + noise)``, which is how tfp's JAX backend samples a
categorical (``random_generators._categorical_jax``). Production code draws
the noise from a key; the parity tests pass the noise the reference drew.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ajax.distributional import symlog

#: 2411f7d ``TransformedMseDist(tol=1e-8)``: squared errors below it count 0.
SYMLOG_MSE_TOLERANCE = 1e-8


def draw_onehot_noise(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Standard Gumbel noise of ``shape = (..., S, C)`` for :meth:`OneHot.sample`.

    One draw per class, as tfp's JAX categorical sampler draws
    (``jax.random.gumbel`` over the logits' shape).
    """
    return jax.random.gumbel(key, shape, jnp.float32)


class OneHot(NamedTuple):
    """``S`` independent categoricals over ``C`` classes, summed over latents.

    ``logits [..., S, C]`` are the log-probabilities *after* the uniform
    mixture, as the paper-era code stores them (``2411f7d:nets.py:207-217``
    applies the mixture inside ``_logit`` and builds
    ``tfd.Independent(OneHotDist(logit), 1)`` on the result). Build it from
    raw network outputs with :meth:`from_logits`, which applies the mixture
    every time a distribution is made from logits -- also after a
    ``stop_gradient`` (dreamerv3_spec 1.8). All reductions over the latent
    axis are sums (``Independent(..., 1)``).
    """

    logits: jax.Array

    @classmethod
    def from_logits(cls, raw_logits: jax.Array, unimix: float) -> OneHot:
        """Mix ``softmax(raw_logits)`` with the uniform distribution.

        ``p = (1 - unimix) softmax(l) + unimix / C`` and the stored logits are
        ``log p`` (``2411f7d:nets.py:212-216``). Every class keeps probability
        at least ``unimix / C``; ``unimix = 0`` gives the plain softmax.
        """
        raw_logits = jnp.asarray(raw_logits, jnp.float32)
        probs = jax.nn.softmax(raw_logits, -1)
        if unimix:
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            return cls(jnp.log(probs))
        return cls(jax.nn.log_softmax(raw_logits, -1))

    @property
    def probs(self) -> jax.Array:
        """Class probabilities ``[..., S, C]`` (tfp ``probs_parameter``)."""
        return jax.nn.softmax(self.logits, -1)

    def sample(self, noise: jax.Array) -> jax.Array:
        """Straight-through one-hot sample ``[..., S, C]`` for Gumbel ``noise``.

        The class is ``argmax(log p + noise)`` (Gumbel-max, as tfp's JAX
        sampler); the value is ``sg(one_hot) + (p - sg(p))`` with ``p`` the
        **mixed** probabilities (``2411f7d:jaxutils.py:112-116``), so the
        forward value is exactly one-hot and the gradient is that of ``p``.
        """
        index = jnp.argmax(self.logits + noise, -1)
        one_hot = jax.nn.one_hot(index, self.logits.shape[-1], dtype=jnp.float32)
        probs = self.probs
        return jax.lax.stop_gradient(one_hot) + (probs - jax.lax.stop_gradient(probs))

    def kl(self, other: OneHot) -> jax.Array:
        """``KL(self || other)`` summed over latents and classes, shape ``[...]``.

        tfp's categorical KL, ``sum_C softmax(a) (log_softmax(a) -
        log_softmax(b))``, summed over the latent axis by ``Independent``
        (dreamerv3_spec 1.8).
        """
        log_p = jax.nn.log_softmax(self.logits, -1)
        log_q = jax.nn.log_softmax(other.logits, -1)
        kl = jnp.sum(jax.nn.softmax(self.logits, -1) * (log_p - log_q), -1)
        return jnp.sum(kl, -1)

    def entropy(self) -> jax.Array:
        """Entropy summed over latents, shape ``[...]``."""
        log_p = jax.nn.log_softmax(self.logits, -1)
        return -jnp.sum(jnp.exp(log_p) * log_p, (-2, -1))


def bernoulli_loss(logit: jax.Array, target: jax.Array) -> jax.Array:
    """Negative log-likelihood of a soft label ``target`` in ``[0, 1]``.

    ``-(target log sigmoid(l) + (1 - target) log sigmoid(-l))``, written
    ``(1 - c) softplus(l) + c softplus(-l)`` as tfp's ``Bernoulli.log_prob``
    computes it (dreamerv3_spec 1.10, 2.13). Elementwise, no reduction.
    """
    logit = jnp.asarray(logit, jnp.float32)
    target = jax.lax.stop_gradient(jnp.asarray(target, jnp.float32))
    return (1 - target) * jax.nn.softplus(logit) + target * jax.nn.softplus(-logit)


def symlog_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Squared error in symlog space, summed over the last axis.

    ``sum_i e_i`` with ``e_i = (pred_i - symlog(target_i))^2``, where squared
    errors below :data:`SYMLOG_MSE_TOLERANCE` count as 0 and there is no
    1/2 factor (``2411f7d:jaxutils.py:179-207`` ``TransformedMseDist``,
    ``tol=1e-8``, ``agg='sum'``; dreamerv3_spec 1.11, deviations.md section
    1). The prediction lives in symlog space; the target is raw and carries
    no gradient. Shape ``[...]`` for inputs ``[..., F]``.
    """
    target = jax.lax.stop_gradient(symlog(jnp.asarray(target, jnp.float32)))
    distance = (jnp.asarray(prediction, jnp.float32) - target) ** 2
    distance = jnp.where(distance < SYMLOG_MSE_TOLERANCE, 0, distance)
    return jnp.sum(distance, -1)
