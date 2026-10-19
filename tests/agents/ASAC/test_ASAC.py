import jax.numpy as jnp
import pytest

from ajax.agents.ASAC import train_ASAC
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.SAC import core


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
