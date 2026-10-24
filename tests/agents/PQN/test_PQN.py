import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.critics import make_huber_td_loss
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.PQN.PQN import PQN
from ajax.agents.PQN.train_PQN import half_mse_td_loss
from ajax.networks.networks import MINATAR_CNN, MNIH_CNN


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


def test_pqn_steps_as_its_reference_optimiser():
    """mttga/purejaxql @47af6d7 (pqn_minatar.py:159-162): optax.radam,
    eps 1e-8, after a global-norm clip of 10."""
    agent = PQN(env_id="CartPole-v1")
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.optimizer, args.eps, args.max_grad_norm) == ("radam", 1e-8, 10.0)


def test_pqn_default_loss_is_half_the_squared_error():
    """purejaxql's loss, 0.5 * mean((Q - G)^2) (pqn_minatar.py:285): its
    gradient is the TD error over the batch size, what the clip of 10 sees."""
    q, target = jnp.array([0.0, 1.0]), jnp.array([3.0, -1.0])
    assert half_mse_td_loss(q, target) == pytest.approx(0.5 * (9.0 + 4.0) / 2)
    assert jnp.allclose(jax.grad(half_mse_td_loss)(q, target), (q - target) / 2)


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


def test_pqn_images_build_purejaxql_s_network_and_train_on_minatar():
    """purejaxql's image networks: Mnih et al.'s from 36 px a side, MinAtar's
    below, he_normal with a LayerNorm after every layer, straight into the
    Q head (no MLP stack after them)."""
    want = {(84, 84, 4): MNIH_CNN, (10, 10, 4): MINATAR_CNN}
    for shape, spec in want.items():
        agent = PQN(env_id="CartPole-v1", cnn_image_shape=shape)
        ref = spec._replace(kernel_init="he_normal", layer_norm=True)
        assert agent.network_args.cnn_spec == ref
    net = PQNNetwork(("8", "relu"), 3, cnn_image_shape=(10, 10, 4), cnn_spec=ref)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 400)))["params"]
    assert sorted(params) == ["CNNEncoder_0", "Dense_0"]
    assert params["Dense_0"]["kernel"].shape == (128, 3)
    agent = PQN(env_id="Breakout-MinAtar", cnn_image_shape=(10, 10, 4), n_envs=2)
    agent.train(seed=0, n_timesteps=2 * agent.agent_config.n_steps)
