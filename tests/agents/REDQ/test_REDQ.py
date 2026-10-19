import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.REDQ.REDQ import REDQ
from ajax.networks.networks import get_initialized_actor_critic

NET = ("16", "relu")


@pytest.mark.parametrize(
    "target_entropy_per_dim, expected_sign",
    [
        (-10.0, -1),  # target far below the policy entropy -> alpha shrinks
        (10.0, 1),  # target far above the policy entropy -> alpha grows
    ],
)
def test_alpha_tracks_target_entropy(target_entropy_per_dim, expected_sign):
    """``target_entropy_per_dim`` drives REDQ's temperature update.

    Regression test (as ASAC's): the temperature step hardcoded the target
    to ``-action_dim``, which the default of -1 per dimension coincides
    with, so a non-default value was silently ignored.
    """
    agent = REDQ(
        env_id="fast",
        actor_architecture=NET,
        critic_architecture=NET,
        num_critics=2,
        num_critic_updates=1,
        learning_starts=10,
        batch_size=8,
        buffer_size=128,
        target_entropy_per_dim=target_entropy_per_dim,
    )
    state, _ = agent.train(seed=0, n_timesteps=60)
    log_alpha = float(jnp.asarray(state.alpha.params["log_alpha"]).ravel()[0])
    assert log_alpha != 0.0, "alpha never moved from alpha_init"
    assert jnp.sign(log_alpha) == expected_sign


def test_redq_starts_every_layer_xavier_uniform():
    """watchernyu/REDQ's weights_init_ (redq/algos/core.py:18-22) starts
    every Linear layer xavier_uniform, gain 1, with a zero bias, the mean and
    log-std layers included; each critic of the ensemble is drawn apart."""
    agent = REDQ(env_id="Pendulum-v1", num_critics=4)
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=4,
    )
    actor = actor_state.params["params"]
    ensemble = critic_state.params["params"]["ensemble"]
    critic = jax.tree.map(lambda x: x[0], ensemble)
    obs_dim, act_dim, width = 3, 1, 256  # Pendulum
    layers = {  # name: (params, fan_in, fan_out)
        "actor hidden 0": (actor["encoder"]["network"]["layers_0"], obs_dim, width),
        "actor hidden 1": (actor["encoder"]["network"]["layers_2"], width, width),
        "mean": (actor["mean"], width, act_dim),
        "log_std": (actor["log_std"], width, act_dim),
        "critic hidden 0": (
            critic["encoder"]["network"]["layers_0"],
            obs_dim + act_dim,
            width,
        ),
        "critic hidden 1": (critic["encoder"]["network"]["layers_2"], width, width),
        "Q": (critic["model"], width, 1),
    }
    for name, (layer, fan_in, fan_out) in layers.items():
        bound = np.sqrt(6 / (fan_in + fan_out))
        kernel = np.asarray(layer["kernel"])
        assert bound * 0.5 < np.abs(kernel).max() <= bound, name
        np.testing.assert_allclose(kernel.var(), bound**2 / 3, rtol=0.3, err_msg=name)
        np.testing.assert_array_equal(layer["bias"], 0.0, err_msg=name)
    first = ensemble["encoder"]["network"]["layers_0"]["kernel"]
    assert not np.allclose(first[0], first[1])


def test_recurrent_redq_keeps_its_previous_initialisation():
    """With memory, REDQ keeps Ajax's previous initialisation until Ni et
    al.'s is audited: an orthogonal(0.01) mean and a zero-kernel log-std
    head starting at -1."""
    from ajax.networks.memory import MemoryConfig

    agent = REDQ(env_id="Pendulum-v1", num_critics=2, memory=MemoryConfig("gru", 16))
    actor_state, _ = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
    )
    actor = actor_state.params["params"]
    mean = np.asarray(actor["mean"]["kernel"])
    np.testing.assert_allclose(mean.T @ mean, 1e-4 * np.eye(1), atol=1e-8)
    np.testing.assert_array_equal(actor["log_std"]["kernel"], 0.0)
    np.testing.assert_array_equal(actor["log_std"]["bias"], -1.0)
