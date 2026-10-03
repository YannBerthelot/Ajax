"""Output distributions of the DreamerV3 world model and actor.

The distributions of the paper-era code ``danijar/dreamerv3@2411f7d`` (the
fidelity target; ``docs/world_models/deviations.md`` section 1), all in
float32. World model:

* :class:`OneHot` -- the categorical latents: ``S`` independent categoricals
  over ``C`` classes with a 1 % uniform mixture, sampled as a straight-through
  one-hot (dreamerv3_spec 1.8; ``2411f7d:dreamerv3/nets.py:207-220``,
  ``jaxutils.py:103-121``).
* :func:`bernoulli_loss` -- the continue head, logistic regression on soft
  labels (dreamerv3_spec 1.10; tfp ``Bernoulli(logits).log_prob``).
* :func:`symlog_mse` -- the vector decoder, squared error in symlog space with
  the 2411f7d tolerance (dreamerv3_spec 1.11; ``jaxutils.py:179-207``).

The reward head's and the critic's two-hot distribution is the shared
:class:`ajax.distributional.TwoHot` (``TwoHot.dreamerv3``). Actor
(dreamerv3_spec 3.13, 3.14; ``2411f7d:dreamerv3/nets.py:486-493``, ``:518-529``
= ``29eb964:dreamerv3/nets.py:491-498``, ``:523-534``):

* :class:`BoundedNormal` -- continuous actions (``actor_dist_cont: normal``):
  ``Normal(tanh(m), (maxstd - minstd) sigmoid(s + 2) + minstd)``, independent
  over the action dimensions, *not* tanh-squashed: samples are unbounded and
  log-probabilities carry no Jacobian term.
* :class:`OneHotPolicy` -- discrete actions (``actor_dist_disc: onehot``): a
  categorical over the actions with a 1 % uniform mixture, sampled as a
  straight-through one-hot; log-probability and entropy are those of the
  mixed distribution (deviations.md section 1, "Discrete actor").

Sampling takes its randomness as an explicit argument: :meth:`OneHot.sample`
and :meth:`OneHotPolicy.sample` consume Gumbel noise drawn by
:func:`draw_onehot_noise` and return ``argmax(log p + noise)``, which is how
tfp's JAX backend samples a categorical
(``random_generators._categorical_jax``); :meth:`BoundedNormal.sample`
consumes standard normal noise and returns ``noise * std + mean``, tfp's
``Normal._sample_n``. Production code draws the noise from a key; the parity
tests pass the noise the reference drew.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp

from ajax.distributional import symlog

#: 2411f7d ``TransformedMseDist(tol=1e-8)``: squared errors below it count 0.
SYMLOG_MSE_TOLERANCE = 1e-8
#: tfp's ``Normal`` log-normaliser constant ``0.5 log(2 pi)`` (normal.py:187).
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


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


# ------------------------------------------------------------------ the actor


def draw_normal_noise(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Standard normal noise of ``shape = (..., A)`` for :meth:`BoundedNormal.sample`."""
    return jax.random.normal(key, shape, jnp.float32)


class BoundedNormal(NamedTuple):
    """The continuous actor's distribution: ``Normal(mean, std)`` per dimension.

    ``2411f7d`` ``Dist('normal')`` (``nets.py:486-493``): the mean and
    standard deviation are bounded, ``mean = tanh(m)`` and ``std = (maxstd -
    minstd) sigmoid(s + 2) + minstd``, computed after the float32 cast of the
    two output layers (``nets.py:437-443``); the samples are not
    (dreamerv3_spec 3.13). The environment clips them to the action bounds;
    the replay stores them unclipped and the dynamics bound them with
    ``a / max(1, |a|)``. Log-probabilities and entropies are tfp's, summed
    over the last axis (``tfd.Independent(..., 1)``).

    Attributes:
        mean: ``[..., A]`` in ``(-1, 1)``.
        std: ``[..., A]`` in ``(minstd, maxstd)``.
    """

    mean: jax.Array
    std: jax.Array

    @classmethod
    def from_outputs(
        cls, mean: jax.Array, std: jax.Array, minstd: float, maxstd: float
    ) -> BoundedNormal:
        """From the raw outputs of the actor's ``mean`` and ``std`` layers."""
        mean = jnp.tanh(jnp.asarray(mean, jnp.float32))
        std = jnp.asarray(std, jnp.float32)
        return cls(mean, (maxstd - minstd) * jax.nn.sigmoid(std + 2.0) + minstd)

    def sample(self, noise: jax.Array) -> jax.Array:
        """``noise * std + mean`` for standard normal ``noise [..., A]``
        (tfp ``Normal._sample_n``: ``sampled * scale + loc``). The sample is
        reparameterised; the actor loss differentiates only its
        log-probability at the stop-gradiented sample (REINFORCE)."""
        return noise * self.std + self.mean

    def log_prob(self, x: jax.Array) -> jax.Array:
        """``sum_A log N(x; mean, std)``, shape ``[...]``.

        tfp ``Normal._log_prob``: ``-0.5 ((x / std) - (mean / std))^2 -
        (0.5 log(2 pi) + log std)``.
        """
        z = x / self.std - self.mean / self.std
        log_normalization = jnp.float32(_HALF_LOG_TWO_PI) + jnp.log(self.std)
        return jnp.sum(-0.5 * jnp.square(z) - log_normalization, -1)

    def entropy(self) -> jax.Array:
        """``sum_A (0.5 + 0.5 log(2 pi) + log std)``, shape ``[...]``
        (tfp ``Normal._entropy``; each term in ``[-0.88, 1.42]`` for ``std``
        in ``[0.1, 1]``)."""
        log_normalization = jnp.float32(_HALF_LOG_TWO_PI) + jnp.log(self.std)
        return jnp.sum((0.5 + log_normalization) * jnp.ones_like(self.mean), -1)

    @staticmethod
    def entropy_range(action_dim: int, minstd: float, maxstd: float) -> tuple:
        """The smallest and largest entropy, at ``std = minstd`` and ``maxstd``
        (``nets.py:491-492``, ``minent`` / ``maxent``; for the ``rand``
        metric)."""

        def entropy(std: float) -> float:
            return action_dim * (0.5 + _HALF_LOG_TWO_PI + math.log(std))

        return entropy(minstd), entropy(maxstd)


class OneHotPolicy(NamedTuple):
    """The discrete actor's distribution: one categorical over the actions.

    ``2411f7d`` ``Dist('onehot')`` with ``unimix = 0.01`` (``nets.py:518-529``):
    the probabilities are ``(1 - unimix) softmax(l) + unimix / A`` and
    sampling, log-probability and entropy all use that mixed distribution
    (dreamerv3_spec 3.14; the later code drops the mixture). The sample is a
    straight-through one-hot (``jaxutils.OneHotDist``), which the dynamics
    read as the action; the environment receives its index.

    Attributes:
        logits: ``[..., A]``, the log of the mixed probabilities.
    """

    logits: jax.Array

    @classmethod
    def from_logits(cls, raw_logits: jax.Array, unimix: float) -> OneHotPolicy:
        """Mix ``softmax(raw_logits)`` with the uniform distribution
        (:meth:`OneHot.from_logits`, the same transformation)."""
        return cls(OneHot.from_logits(raw_logits, unimix).logits)

    def sample(self, noise: jax.Array) -> jax.Array:
        """Straight-through one-hot ``[..., A]`` for Gumbel ``noise [..., A]``
        (:meth:`OneHot.sample`)."""
        return OneHot(self.logits).sample(noise)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """``log p(x)`` of a one-hot ``x [..., A]``, shape ``[...]``.

        tfp ``OneHotCategorical._log_prob``: ``-softmax_cross_entropy(x,
        logits)``, i.e. ``sum_A x (logits - logsumexp(logits))`` with the
        zero entries of ``x`` contributing exactly 0.
        """
        log_probs = self.logits - jax.scipy.special.logsumexp(
            self.logits, -1, keepdims=True
        )
        return jnp.sum(jnp.where(x == 0, 0.0, x * log_probs), -1)

    def entropy(self) -> jax.Array:
        """``-sum_A p log p``, shape ``[...]``.

        tfp ``OneHotCategorical._entropy``: ``logsumexp(l) - sum_A l e^{l - m}
        / sum_A e^{l - m}`` with ``m = max l``.
        """
        m = jnp.max(self.logits, -1, keepdims=True)
        x = self.logits - m
        lse = m[..., 0] + jax.scipy.special.logsumexp(x, -1)
        exp_x = jnp.exp(x)
        weighted = jnp.where(exp_x == 0, 0.0, self.logits * exp_x)
        return lse - jnp.sum(weighted, -1) / jnp.sum(exp_x, -1)

    @staticmethod
    def entropy_range(action_dim: int) -> tuple:
        """``(0, log A)`` (``nets.py:527-528``, ``minent`` / ``maxent``)."""
        return 0.0, math.log(action_dim)


#: The actor's distribution, by action space.
Policy = Union[BoundedNormal, OneHotPolicy]


def draw_action_noise(
    key: jax.Array, shape: tuple[int, ...], discrete: bool
) -> jax.Array:
    """Noise of ``shape = (..., A)`` for the actor's samples: Gumbel for a
    discrete actor (:class:`OneHotPolicy`), standard normal for a continuous
    one (:class:`BoundedNormal`)."""
    if discrete:
        return draw_onehot_noise(key, shape)
    return draw_normal_noise(key, shape)
