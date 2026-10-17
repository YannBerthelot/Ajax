"""Policy distributions shared by the agents.

:class:`Squash` bounds any continuous policy to ``(-1, 1)`` with ``tanh``
(:func:`squash` also takes a :class:`Deterministic` one). The functions
below it read any policy, squashed or not, so callers need not check which
they hold.
"""

from typing import Any, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Deterministic:
    """Tiny distrax-compatible wrapper for deterministic policies (TD3's).

    Exposes `.mean()` so existing code (the policy loss, the target action
    computation, the ImitationLoss extension) keeps working without branches.
    A pytree of its action, as distrax distributions are of their
    parameters (a replayed bootstrap gathers its steps, recurrent.actor_dist).
    """

    _action: jax.Array

    def mean(self) -> jax.Array:
        return self._action

    def sample(self, seed=None, sample_shape=()) -> jax.Array:
        del seed, sample_shape
        return self._action

    def sample_and_log_prob(self, seed=None) -> Tuple[jax.Array, jax.Array]:
        del seed
        return self._action, jnp.zeros(self._action.shape[:-1] + (1,))

    def entropy(self) -> jax.Array:
        return jnp.zeros(self._action.shape[:-1])

    def mode(self) -> jax.Array:
        return self._action

    def stddev(self) -> jax.Array:
        return jnp.zeros_like(self._action)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """For BC pretraining: a delta distribution's log-density is +∞ at
        x=action, -∞ elsewhere. We use the negative-MSE proxy so that
        maximizing log_prob is equivalent to MSE-minimizing the actor toward
        the target action. Shape: (..., action_dim) -> (..., action_dim)
        (per-dim, summed by caller)."""
        return -((x - self._action) ** 2)


class Squash(distrax.Transformed):
    """``tanh`` of a policy distribution, its latent: any distrax
    distribution (a ``Block`` spans its event dimensions, if it has any).

    Exact: sampling and log-probabilities. ``log_prob(action)`` inverts
    ``tanh``, finite only while ``|action| < 1``, which float32 loses once
    the latent sample passes about 9; :meth:`log_prob_from_raw` reads the
    latent sample itself and stays finite.

    Not exact, as neither has a closed form: :meth:`mean` is ``tanh`` of the
    latent's mean, the deterministic action, not the mean of the squashed
    action; :meth:`entropy` is sampled.
    """

    def __init__(self, distribution: distrax.Distribution):
        if not jnp.issubdtype(distribution.dtype, jnp.floating):
            raise ValueError(
                "tanh squashes a continuous distribution, not a"
                f" {type(distribution).__name__}."
            )
        bijector = distrax.Tanh()
        event_ndims = len(distribution.event_shape)
        if event_ndims:
            bijector = distrax.Block(bijector, event_ndims)
        super().__init__(distribution=distribution, bijector=bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

    def log_prob_from_raw(self, raw: jax.Array) -> jax.Array:
        """The log-probability of ``tanh(raw)``, from the latent sample
        ``raw`` (no ``arctanh``, so finite at saturation)."""
        return self.distribution.log_prob(raw) - self.bijector.forward_log_det_jacobian(
            raw
        )

    def sample_with_raw(self, seed: jax.Array):
        """``(action, log_prob, raw)``: ``raw`` the latent sample, the
        action ``tanh(raw)``."""
        raw = self.distribution.sample(seed=seed)
        return self.bijector.forward(raw), self.log_prob_from_raw(raw), raw

    def entropy(self, seed: Optional[jax.Array] = None) -> jax.Array:
        """The squashed action's entropy, estimated as brax does: the
        latent's entropy plus the ``tanh`` log-det-Jacobian at one latent
        sample drawn with ``seed``. Unbiased, and its gradient reaches the
        mean; the latent's entropy alone grows without bound in its scale."""
        if seed is None:
            raise ValueError(
                "A squashed policy's entropy has no closed form: pass a seed"
                " to sample it."
            )
        raw = self.distribution.sample(seed=seed)
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(raw)


def squash(pi: Any) -> Any:
    """``pi`` bounded by ``tanh``: :class:`Squash` of it, or, for a
    :class:`Deterministic` one, ``tanh`` of its action (a point mass has no
    density to correct)."""
    if isinstance(pi, Deterministic):
        return Deterministic(jnp.tanh(pi.mean()))
    return Squash(pi)


def latent(pi: Any) -> Any:
    """The distribution ``pi`` squashes, or ``pi`` itself."""
    return pi.distribution if isinstance(pi, Squash) else pi


def sample_with_raw(pi: Any, seed: jax.Array):
    """``(action, log_prob, raw)``: ``raw`` is the latent sample, the
    action itself when ``pi`` is not squashed."""
    if isinstance(pi, Squash):
        return pi.sample_with_raw(seed)
    action, log_prob = pi.sample_and_log_prob(seed=seed)
    return action, log_prob, action


def log_prob_from_raw(pi: Any, raw: jax.Array) -> jax.Array:
    """The log-probability of the action whose latent sample is ``raw``."""
    return pi.log_prob_from_raw(raw) if isinstance(pi, Squash) else pi.log_prob(raw)


def entropy(pi: Any, seed: Optional[jax.Array] = None) -> jax.Array:
    """``pi``'s entropy: sampled with ``seed`` when squashed, else exact."""
    return pi.entropy(seed) if isinstance(pi, Squash) else pi.entropy()
