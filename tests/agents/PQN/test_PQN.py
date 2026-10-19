import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DQN.train_DQN import make_huber_td_loss
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.PQN.PQN import PQN


def test_PQN_rejects_indivisible_minibatches():
    """The rollout's n_steps * n_envs rows must split into num_minibatches."""
    with pytest.raises(ValueError, match="must be divisible"):
        PQN(env_id="CartPole-v1", n_envs=1, n_steps=7, num_minibatches=2)
    PQN(env_id="CartPole-v1", n_envs=2, n_steps=7, num_minibatches=2)


def test_pqn_huber_loss_smoke():
    """The Huber td_loss_fn hook composes and runs end-to-end."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        td_loss_fn=make_huber_td_loss(1.0),
    )
    agent.train(seed=42, n_timesteps=256)


def test_pqn_starts_every_dense_as_flax_default():
    """PQN's reference code (mttga/purejaxql @47af6d7, pqn_gymnax.py:52, :56)
    sets no initializer: every Dense is lecun_normal, a normal truncated at
    two standard deviations with variance 1/fan_in, and a zero bias."""
    network = PQNNetwork(input_architecture=("128", "relu", "128", "relu"), n_actions=2)
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))["params"]
    for name, fan_in in (("Dense_0", 4), ("Dense_1", 128), ("Dense_2", 128)):
        kernel = np.asarray(params[name]["kernel"])
        std = 1 / np.sqrt(fan_in)
        np.testing.assert_allclose(kernel.std(), std, rtol=0.2, err_msg=name)
        assert np.abs(kernel).max() <= 2 * std / 0.87962566, name
        np.testing.assert_array_equal(params[name]["bias"], 0.0, err_msg=name)
