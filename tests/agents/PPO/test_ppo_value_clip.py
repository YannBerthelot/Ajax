"""PPO's clipped value loss, ppo2's (baselines @ea25b9e ppo2/model.py:66-75):

    L_V = vf_coef * 0.5 * mean max((V - R)^2, (V_old + clip(V - V_old, -c, c) - R)^2)

with V the critic's value, V_old the one the rollout was collected with, R the target and c the
clip range. Once V has moved more than c from V_old towards R, the clipped error is the larger one
and does not depend on V: the loss stops pulling V. Moving away from R, the unclipped error is the
larger one and pulls V back. Within c of V_old both are equal. Clipping is the default but with
unroll_length (brax's PPO, which does not clip).

The critic below outputs its last bias for every observation (every other parameter zero), so V is
set exactly and each expected value is worked out by hand.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.PPO.PPO import PPO, compute_value_loss

SMALL: dict[str, Any] = {
    "n_steps": 8,
    "n_envs": 2,
    "n_epochs": 1,
    "num_minibatches": 2,
    "critic_architecture": ("8", "tanh"),
}
C = 0.2


def _constant_critic(value: float) -> Any:
    """A critic whose value is ``value`` everywhere."""
    agent = PPO(env_id="Pendulum-v1", **SMALL)
    _, context = agent.make_context(1_000)
    critic = agent.init(jax.random.PRNGKey(0), context).critic_state

    def set_leaf(path: Any, leaf: jax.Array) -> jax.Array:
        names = [getattr(key, "key", None) for key in path]
        last_bias = "model" in names and names[-1] == "bias"
        return jnp.full_like(leaf, value) if last_bias else jnp.zeros_like(leaf)

    return critic.replace(
        params=jax.tree_util.tree_map_with_path(set_leaf, critic.params)
    )


def _loss_and_slope(value: float, collected: float, target: float, clip: Any) -> tuple:
    """The loss (vf_coef 1) at a critic of value ``value`` and its derivative in that value."""
    critic = _constant_critic(value)
    minibatch = {
        "obs": jnp.zeros((4, 3)),
        "values": jnp.full((4, 1), collected),
    }
    targets = jnp.full((4, 1), target)

    def loss(params: Any) -> jax.Array:
        return compute_value_loss(params, critic, minibatch, targets, clip_range=clip)[
            0
        ]

    value_loss, grads = jax.value_and_grad(loss)(critic.params)
    slope = jax.tree_util.tree_leaves(grads["params"]["ensemble"]["model"]["bias"])[0]
    return float(value_loss), float(slope.squeeze())


def test_a_value_within_the_clip_takes_the_plain_squared_error():
    # V = 0.1, V_old = 0, R = 10: |V - V_old| < C, both errors (10 - 0.1)^2 = 98.01.
    loss, slope = _loss_and_slope(0.1, 0.0, 10.0, C)
    assert loss == pytest.approx(0.5 * 98.01, rel=1e-6)
    assert slope == pytest.approx(0.1 - 10.0, rel=1e-6)  # d/dV 0.5 (V - R)^2


def test_a_value_past_the_clip_towards_its_target_is_no_longer_pulled():
    # V = 0.5, V_old = 0, R = 10: clipped V 0.2, (0.2 - 10)^2 = 96.04 > (0.5 - 10)^2 = 90.25.
    loss, slope = _loss_and_slope(0.5, 0.0, 10.0, C)
    assert loss == pytest.approx(0.5 * 96.04, rel=1e-6)
    assert slope == 0.0


def test_a_value_past_the_clip_away_from_its_target_is_pulled_back():
    # V = -0.5, V_old = 0, R = 10: (-0.5 - 10)^2 = 110.25 > clipped (-0.2 - 10)^2 = 104.04.
    loss, slope = _loss_and_slope(-0.5, 0.0, 10.0, C)
    assert loss == pytest.approx(0.5 * 110.25, rel=1e-6)
    assert slope == pytest.approx(-0.5 - 10.0, rel=1e-6)


def test_without_a_clip_range_the_loss_is_the_plain_squared_error():
    loss, slope = _loss_and_slope(0.5, 0.0, 10.0, None)
    assert loss == pytest.approx(0.5 * 90.25, rel=1e-6)
    assert slope == pytest.approx(0.5 - 10.0, rel=1e-6)


def test_ppo_clips_its_value_loss_as_its_reference_does():
    assert PPO(env_id="Pendulum-v1", **SMALL).clips_value_loss()
    brax = {**SMALL, "unroll_length": 4}
    assert not PPO(env_id="Pendulum-v1", **brax).clips_value_loss()
    assert not PPO(
        env_id="Pendulum-v1", clip_value_loss=False, **SMALL
    ).clips_value_loss()


def test_brax_minibatches_refuse_a_clipped_value_loss():
    with pytest.raises(ValueError, match="clip_value_loss"):
        PPO(env_id="Pendulum-v1", clip_value_loss=True, unroll_length=4, **SMALL)
