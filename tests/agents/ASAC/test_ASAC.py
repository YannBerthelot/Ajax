import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.ASAC import train_ASAC
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.SAC import core
from ajax.networks.networks import get_initialized_actor_critic


@pytest.mark.parametrize(
    "target_entropy_per_dim, expected_sign",
    [
        (-10.0, -1),  # target far below the policy entropy → alpha shrinks
        (10.0, 1),  # target far above the policy entropy → alpha grows
    ],
)
def test_alpha_tracks_target_entropy(target_entropy_per_dim, expected_sign):
    """``target_entropy_per_dim`` drives the temperature update.

    Regression test: the temperature step used to hardcode the target
    to ``-action_dim`` and discard the updated state, so alpha stayed
    frozen at ``alpha_init`` no matter what the user configured.
    """
    agent = ASAC(
        env_id="fast",
        learning_starts=10,
        target_entropy_per_dim=target_entropy_per_dim,
    )
    state, _ = agent.train(seed=0, n_timesteps=60)
    log_alpha = float(jnp.asarray(state.alpha.params["log_alpha"]).ravel()[0])
    assert log_alpha != 0.0, "alpha never moved from alpha_init"
    assert jnp.sign(log_alpha) == expected_sign


def test_temperature_and_theta_read_one_sample_of_the_policy(monkeypatch):
    """One sample of the updated policy feeds both the temperature and the
    reward-rate (theta) steps: the actor runs once for the two."""
    seen = []
    temperature, theta = core.update_temperature, train_ASAC.update_theta

    def temperature_spy(*args):
        seen.append(args[1])  # the log-probabilities
        return temperature(*args)

    def theta_spy(*args):
        seen.append(args[3])
        return theta(*args)

    monkeypatch.setattr(core, "update_temperature", temperature_spy)
    monkeypatch.setattr(train_ASAC, "update_theta", theta_spy)
    net = ("8", "relu")
    agent = ASAC(
        "Pendulum-v1",
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=4,
        batch_size=4,
    )
    agent.train(seed=0, n_timesteps=8)
    # The update is traced twice: once for it, once for the structure of
    # the NaN metrics before learning_starts (TrainLoop.maybe_update).
    assert len(seen) == 4 and seen[0] is seen[1] and seen[2] is seen[3]


def test_asac_starts_every_layer_as_torch_nn_linear():
    """ASAC's reference code builds stable-baselines3 2.4.0's SAC actor and
    plain torch critics (BobMak/asac @4b68862) at torch's default: every
    kernel and bias uniform in +-1/sqrt(fan_in), the heads included."""
    agent = ASAC(env_id="Pendulum-v1")
    actor_state, critic_state = get_initialized_actor_critic(
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
    critic = jax.tree.map(lambda x: x[0], critic_state.params["params"]["ensemble"])
    obs_dim, act_dim = 3, 1  # Pendulum
    layers = {  # name: (params, fan_in)
        "actor hidden 0": (actor["encoder"]["network"]["layers_0"], obs_dim),
        "actor hidden 1": (actor["encoder"]["network"]["layers_2"], 256),
        "mean": (actor["mean"], 256),
        "log_std": (actor["log_std"], 256),
        "critic hidden 0": (
            critic["encoder"]["network"]["layers_0"],
            obs_dim + act_dim,
        ),
        "critic hidden 1": (critic["encoder"]["network"]["layers_2"], 256),
        "Q": (critic["model"], 256),
    }
    for name, (layer, fan_in) in layers.items():
        bound = 1 / np.sqrt(fan_in)
        kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
        assert np.abs(kernel).max() <= bound and np.abs(bias).max() <= bound, name
        assert np.abs(kernel).max() > 0.5 * bound and np.any(bias), name
        if kernel.size >= 256:  # a uniform's variance, bound**2 / 3
            np.testing.assert_allclose(kernel.var(), bound**2 / 3, rtol=0.3)


def test_recurrent_asac_keeps_its_previous_initialisation():
    """With memory, ASAC keeps Ajax's previous initialisation until Ni et
    al.'s is audited: an orthogonal(0.01) mean and a zero-kernel log-std
    head starting at -1."""
    from ajax.networks.memory import MemoryConfig

    agent = ASAC(env_id="Pendulum-v1", memory=MemoryConfig("gru", 16))
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
