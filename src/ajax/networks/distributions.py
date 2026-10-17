"""Policy distributions shared by the agents.

:class:`Squash` bounds any continuous policy to ``(-1, 1)`` with ``tanh``.
The functions below it read any policy, squashed or not, so callers need
not check which they hold.
"""

from typing import Any, Optional

import distrax
import jax


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
