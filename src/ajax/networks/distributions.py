import distrax
import jax
import jax.numpy as jnp


class SquashedNormal(distrax.Transformed):
    _tanh_bijector = distrax.Tanh()

    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)

        super().__init__(
            distribution=normal_dist, bijector=SquashedNormal._tanh_bijector
        )

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

    def unsquashed_mean(self):
        return self.distribution.mean()

    def unsquashed_stddev(self):
        return self.distribution.stddev()

    def unsquashed_entropy(self):
        return self.distribution.entropy()

    def log_determinant(self, u):
        """Numerically stable ``log(1 - tanh(u)^2)``: the tanh's log volume change."""
        return -2.0 * (u + jax.nn.softplus(-2.0 * u) - jnp.log(2.0))

    def log_prob_from_raw(self, raw_action):
        """log_prob via the pre-tanh sample (no arctanh, stable at saturation)."""
        return (
            self.distribution.log_prob(raw_action)
            - self.bijector.forward_log_det_jacobian(raw_action)
        ).sum(-1, keepdims=True)

    def effective_entropy(self, key, num_samples=1):
        """The executed (squashed) action's entropy, estimated as brax does:
        the latent Gaussian's analytical entropy plus the tanh log-det-Jacobian
        averaged over ``num_samples`` latent samples drawn with ``key``."""
        latent_h = self.distribution.entropy().sum(-1)

        u = self.distribution.sample(seed=key, sample_shape=(num_samples,))

        correction = self.log_determinant(u).sum(-1).mean(axis=0)

        return latent_h + correction
