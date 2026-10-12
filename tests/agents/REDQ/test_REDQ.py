import jax.numpy as jnp
import pytest

from ajax.agents.REDQ.REDQ import REDQ

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
