import jax.numpy as jnp
import pytest

from ajax.agents.ASAC.ASAC import ASAC


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
