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


class GuardedTanh(distrax.Tanh):
    """``tanh`` whose log-det-Jacobian is ``log(1 - tanh(x)^2 + eps)``, as
    stable-baselines3's squashed Gaussian (eps 1e-6) and REDQ (core.py:232)
    compute it; distrax's own is exact."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward_log_det_jacobian(self, x: jax.Array) -> jax.Array:
        return jnp.log(1.0 - jnp.tanh(x) ** 2 + self.eps)

    def forward_and_log_det(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return jnp.arctanh(y), -jnp.log(1.0 - y**2 + self.eps)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is GuardedTanh and other.eps == self.eps


class Squash(distrax.Transformed):
    """``tanh`` of a policy distribution, its latent: any distrax
    distribution (a ``Block`` spans its event dimensions, if it has any).
    ``log_det_eps`` > 0 takes :class:`GuardedTanh`'s log-det instead.

    Exact (with ``log_det_eps`` 0): sampling and log-probabilities.
    ``log_prob(action)`` inverts ``tanh``, finite only while ``|action| <
    1``, which float32 loses once the latent sample passes about 9;
    :meth:`log_prob_from_raw` reads the latent sample itself and stays
    finite.

    Not exact, as neither has a closed form: :meth:`mean` is ``tanh`` of the
    latent's mean, the deterministic action, not the mean of the squashed
    action, or with ``sigma_point_mean`` the mean of ``tanh`` at the latent's
    mean plus and minus its standard deviation (RVI-SAC's authors' evaluation
    action, yhisaki/average-reward-drl b5b0d31 modules.py:115-116);
    :meth:`entropy` is sampled.
    """

    def __init__(
        self,
        distribution: distrax.Distribution,
        log_det_eps: float = 0.0,
        sigma_point_mean: bool = False,
    ):
        if not jnp.issubdtype(distribution.dtype, jnp.floating):
            raise ValueError(
                "tanh squashes a continuous distribution, not a"
                f" {type(distribution).__name__}."
            )
        bijector = GuardedTanh(log_det_eps) if log_det_eps else distrax.Tanh()
        event_ndims = len(distribution.event_shape)
        if event_ndims:
            bijector = distrax.Block(bijector, event_ndims)
        super().__init__(distribution=distribution, bijector=bijector)
        self.sigma_point_mean = sigma_point_mean

    def mean(self):
        mean = self.distribution.mean()
        if not self.sigma_point_mean:
            return self.bijector.forward(mean)
        spread = self.distribution.stddev()
        upper, lower = (
            self.bijector.forward(mean + spread),
            self.bijector.forward(mean - spread),
        )
        return (upper + lower) / 2

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


def squash(pi: Any, log_det_eps: float = 0.0, sigma_point_mean: bool = False) -> Any:
    """``pi`` bounded by ``tanh``: :class:`Squash` of it, or, for a
    :class:`Deterministic` one, ``tanh`` of its action (a point mass has no
    density to correct)."""
    if isinstance(pi, Deterministic):
        return Deterministic(jnp.tanh(pi.mean()))
    return Squash(pi, log_det_eps, sigma_point_mean)


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


def recompute_log_prob(
    pi: Any, actions: jax.Array, raw_actions: Optional[jax.Array]
) -> jax.Array:
    """The current policy's log-probability of the stored actions, shaped
    like :func:`ajax.agents.minibatch.get_actions_and_log_probs`'s. A squashed policy recomputes it from the
    stored pre-tanh sample, never through arctanh, which is unstable as
    ``|action| -> 1`` (:func:`log_prob_from_raw`); an unsquashed one's
    pre-tanh sample is its action."""
    if isinstance(pi, distrax.Categorical):
        return jnp.expand_dims(pi.log_prob(actions.squeeze(-1)), -1)
    if raw_actions is not None:
        return log_prob_from_raw(pi, raw_actions).sum(-1, keepdims=True)
    return pi.log_prob(actions).sum(-1, keepdims=True)


def policy_entropy(pi: Any, rng: Optional[jax.Array] = None) -> jax.Array:
    """The per-sample entropy bonus: the joint action's entropy, summed over
    the action dimensions as CleanRL's ``ppo_continuous_action.py`` and
    baselines' ppo2 sum it (a mean would scale ``ent_coef`` by 1 / d). A
    squashed policy's is sampled with ``rng`` (:meth:`Squash.entropy`)."""
    if isinstance(pi, distrax.Categorical):
        return pi.entropy()
    return entropy(pi, rng).sum(-1)
