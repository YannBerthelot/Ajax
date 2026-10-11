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
        """
        Numerically stable version of log(1 - tanh^2(u)).
        This represents how much the tanh 'squashes' the probability volume.
        """
        return -2.0 * (u + jax.nn.softplus(-2.0 * u) - jnp.log(2.0))

    def log_prob_from_raw(self, raw_action):
        """log_prob via the pre-tanh sample (no arctanh, stable at saturation)."""
        return (
            self.distribution.log_prob(raw_action)
            - self.bijector.forward_log_det_jacobian(raw_action)
        ).sum(-1, keepdims=True)

    def effective_entropy(self, key, num_samples=1):
        """
        Calculates the entropy proxy for the temperature (alpha) loss.
        Combines analytical Gaussian entropy with a sampled volume correction.
        """
        # 1. Analytical entropy of the latent Gaussian (The 'Alive' signal)
        latent_h = self.distribution.entropy().sum(-1)

        # 2. Sample latent values 'u' to calculate the squash correction
        u = self.distribution.sample(seed=key, sample_shape=(num_samples,))

        # 3. Calculate volume change (Log-Det Jacobian)
        correction = self.log_determinant(u).sum(-1).mean(axis=0)

        # Result is scaled to your target_entropy (e.g., -2.0)
        return latent_h + correction
