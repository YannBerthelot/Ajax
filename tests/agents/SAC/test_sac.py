import jax.numpy as jnp
import pytest

from ajax.agents.SAC.SAC import SAC


def test_skip_critic_pretrain_is_a_transitional_no_op():
    """AjaxExperiments still passes True; False asks for the critic
    pretraining that was removed, so it raises instead of being ignored."""
    SAC(env_id="Pendulum-v1", skip_critic_pretrain=True)
    with pytest.raises(ValueError, match="MCPretrain"):
        SAC(env_id="Pendulum-v1", skip_critic_pretrain=False)


@pytest.mark.parametrize("alpha_init", [1.0, 0.2])
def test_fixed_alpha_keeps_temperature_at_alpha_init(alpha_init):
    """``fixed_alpha=True`` skips the temperature gradient step entirely.

    ``alpha_update_start=0`` and ``learning_starts=10`` guarantee that the
    temperature update *would* run for most of the 60 steps — the sibling
    run below confirms alpha does move once ``fixed_alpha`` is off — so an
    unchanged ``log_alpha`` proves the flag is honoured rather than the
    update never having been reached.
    """
    common = {
        "env_id": "fast",
        "learning_starts": 10,
        "alpha_update_start": 0,
        "alpha_init": alpha_init,
        "alpha_learning_rate": 1e-2,
    }
    expected_log_alpha = jnp.log(alpha_init)

    fixed_state, _ = SAC(**common, fixed_alpha=True).train(seed=42, n_timesteps=60)
    assert jnp.allclose(fixed_state.alpha.params["log_alpha"], expected_log_alpha)

    learned_state, _ = SAC(**common, fixed_alpha=False).train(seed=42, n_timesteps=60)
    assert not jnp.allclose(
        learned_state.alpha.params["log_alpha"], expected_log_alpha
    ), "sanity check: alpha should move when it is not fixed"
