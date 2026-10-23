"""TQC's contracts: the authors' settings, the truncated target's atoms, the
quantile Huber loss as their code computes it, and its refusals. Whether it
learns is the probes' (tests/probing)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TQC
from ajax.agents.TQC.train_TQC import quantile_huber_loss, truncated_quantiles
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.instrumentation import BiasVorePenalty


def test_the_defaults_are_the_authors() -> None:
    """bayesgroup/tqc_pytorch's main.py, trainer.py and structures.py."""
    agent = TQC("Pendulum-v1")
    config = agent.agent_config
    assert (config.num_critics, config.n_quantiles) == (5, 25)
    assert config.top_quantiles_to_drop_per_net == 2
    assert (config.gamma, config.tau, config.learning_starts) == (0.99, 0.005, 257)
    assert config.target_entropy == -1.0  # -dim(A) on Pendulum
    assert agent.network_args.critic_quantiles == 25
    assert agent.network_args.critic_architecture == ("512", "relu") * 3
    assert agent.network_args.actor_architecture == ("256", "relu") * 2
    for optimizer in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (optimizer.learning_rate, optimizer.eps) == (3e-4, 1e-8)
    assert (agent.alpha_args.learning_rate, agent.alpha_args.alpha_init) == (3e-4, 1.0)


def test_the_target_keeps_the_smallest_pooled_quantiles() -> None:
    """Two critics' three quantiles pooled per row and sorted, the two
    largest dropped: the four smallest of the six."""
    quantiles = jnp.array([[[1.0, 5.0, 3.0]], [[2.0, 6.0, 4.0]]])  # (N, B, M)
    np.testing.assert_array_equal(truncated_quantiles(quantiles, 2), [[1, 2, 3, 4]])


def _authors_loss(quantiles: np.ndarray, samples: np.ndarray) -> float:
    """tqc/functions.py's quantile_huber_loss_f, its torch ops in numpy:
    ``quantiles`` (batch, nets, quantiles), ``samples`` (batch, atoms)."""
    delta = samples[:, None, None, :] - quantiles[:, :, :, None]
    huber = np.where(np.abs(delta) > 1, np.abs(delta) - 0.5, delta**2 * 0.5)
    n = quantiles.shape[2]
    tau = np.arange(n, dtype=np.float32) / n + 1 / 2 / n
    return float((np.abs(tau[None, None, :, None] - (delta < 0)) * huber).mean())


def test_the_quantile_huber_loss_is_the_authors() -> None:
    """On random quantiles and atoms (Ajax's critics come first), and on a
    hand value: quantile 1 at fraction 1/2 against atoms 0 and 3 weighs
    the Huber losses 0.5 and 1.5 by 1/2 each."""
    keys = jax.random.split(jax.random.PRNGKey(0))
    quantiles = 2 * jax.random.normal(keys[0], (3, 8, 5))  # (N, B, M)
    samples = 2 * jax.random.normal(keys[1], (8, 13))  # (B, K)
    authors = _authors_loss(
        np.swapaxes(np.asarray(quantiles), 0, 1), np.asarray(samples)
    )
    np.testing.assert_allclose(
        quantile_huber_loss(quantiles, samples), authors, rtol=1e-6
    )
    hand = quantile_huber_loss(jnp.ones((1, 1, 1)), jnp.array([[0.0, 3.0]]))
    np.testing.assert_allclose(hand, 0.5)


def test_it_refuses_discrete_actions_and_dropping_every_quantile() -> None:
    with pytest.raises(ValueError, match="continuous"):
        TQC("CartPole-v1")
    with pytest.raises(ValueError, match="keeps some"):
        TQC("Pendulum-v1", n_quantiles=2, top_quantiles_to_drop_per_net=2)


def test_an_extension_reading_one_target_value_refuses_its_quantiles() -> None:
    """Its critics regress on many target atoms per row: no scalar_target."""
    with pytest.raises(MissingCapability, match="scalar_target"):
        TQC("Pendulum-v1", extensions=(BiasVorePenalty(alpha=0.25),))
