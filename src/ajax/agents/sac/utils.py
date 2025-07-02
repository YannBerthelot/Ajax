import distrax
import jax.numpy as jnp

# def simplex(a, b, dyna_factor):
#     assert a.shape == b.shape, "Shapes of a and b must match."
#     b = jnp.broadcast_to(b, a.shape)
#     return (1 - dyna_factor) * a + dyna_factor * b


def simplex(a, b, dyna_factor):
    # Use a fused multiply-add to minimize rounding errors:
    return jnp.add(a, dyna_factor * (b - a))


class SquashedNormal(distrax.Transformed):
    _tanh_bijector = distrax.Tanh()

    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)

        super().__init__(
            distribution=normal_dist, bijector=SquashedNormal._tanh_bijector
        )

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

    def entropy(self):
        return self.distribution.entropy()

    def mix_distributions(self, other, dyna_factor: float):
        return MixedDistribution(self, other, dyna_factor)
        if isinstance(other, SquashedNormal):
            return SquashedNormal(
                simplex(self.distribution.loc, other.distribution.loc, dyna_factor),
                simplex(self.distribution.scale, other.distribution.scale, dyna_factor),
            )
        raise TypeError(
            f"Unsupported type for mix_distributions: {type(other)}. Expected int,"
            " float, or SquashedNormal."
        )


class MixedDistribution(distrax.Distribution):
    def __init__(
        self, dist_1: SquashedNormal, dist_2: SquashedNormal, dyna_factor: float
    ):
        self.dist_1 = dist_1
        self.dist_2 = dist_2
        self.dyna_factor = dyna_factor

    def mean(self):
        return simplex(self.dist_1.mean(), self.dist_2.mean(), self.dyna_factor)

    def variance(self):
        return simplex(self.dist_1.variance(), self.dist_2.variance(), self.dyna_factor)

    def entropy(self):
        return simplex(self.dist_1.entropy(), self.dist_2.entropy(), self.dyna_factor)

    @property
    def event_shape(self):
        return self.dist_1.event_shape

    def sample(self, *, seed, sample_shape=()):
        samples_1 = self.dist_1.sample(seed=seed, sample_shape=sample_shape)
        samples_2 = self.dist_2.sample(seed=seed, sample_shape=sample_shape)
        return simplex(samples_1, samples_2, self.dyna_factor)

    def sample_and_log_prob(self, *, seed, sample_shape=()):
        samples_1, log_prob1 = self.dist_1.sample_and_log_prob(
            seed=seed, sample_shape=sample_shape
        )
        samples_2, log_prob2 = self.dist_2.sample_and_log_prob(
            seed=seed, sample_shape=sample_shape
        )
        samples = simplex(samples_1, samples_2, self.dyna_factor)

        mixed_log_prob = simplex(log_prob1, log_prob2, self.dyna_factor)

        return samples, mixed_log_prob

    def log_prob(self, value):
        prob1 = self.dist_1.log_prob(value)
        prob2 = self.dist_2.log_prob(value)
        return simplex(prob1, prob2, self.dyna_factor)

    def _sample_n(self, key, n):
        samples_1 = self.dist_1._sample_n(key, n)
        samples_2 = self.dist_2._sample_n(key, n)
        return simplex(samples_1, samples_2, self.dyna_factor)
