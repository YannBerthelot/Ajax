"""ExpertValueMetrics: SAC's critic at the expert's actions, as metrics of
its evaluations instead of a critic pass at every update."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.buffers.utils import get_batch_from_buffer
from ajax.extensions.expert import ExpertValueMetrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import predict_value

LOGGED = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
KEYS = ("value/q_expert_mean", "value/q_gap")


class _HalfTorque:
    """A stateful expert (SAC's convention)."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0), state


def _sac(*extensions: Any, expert: Any = None) -> SAC:
    return SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        expert_policy=expert or _HalfTorque(),
        expert_buffer_n_steps=0,
        extensions=extensions,
    )


def test_the_metrics_are_logged_only_with_the_extension() -> None:
    _, rows = _sac().train(seed=0, n_timesteps=64, logging_config=LOGGED)
    assert not {*KEYS, "value/phi_star_q_gap_ood"} & set(rows)
    _, rows = _sac(ExpertValueMetrics()).train(
        seed=0, n_timesteps=64, logging_config=LOGGED
    )
    for key in KEYS:  # the evaluations at 32 and 64 steps
        assert np.isfinite(rows[key][0, :2]).all(), key
    assert "value/phi_star_q_gap_ood" not in rows  # no φ* fitted


def test_the_metrics_read_the_critic_at_the_stored_expert_actions() -> None:
    """Their definitions on the replay batch the key draws; a φ* equal to the
    critic disagrees with it nowhere."""
    agent = _sac()
    state, _ = agent.train(seed=0, n_timesteps=64)
    state = jax.tree.map(lambda x: x[0], state)
    critic = state.critic_state
    ext = ExpertValueMetrics().bind_to_agent(
        buffer=agent.buffer, expert_policy=agent.expert_policy
    )
    key = jax.random.PRNGKey(3)
    with_phi = state.replace(expert_critic_params=critic.params)
    got = ext.eval_metrics(with_phi, None, key, None)  # type: ignore[arg-type]
    buffer_state = state.collector_state.buffer_state
    batch = get_batch_from_buffer(agent.buffer, buffer_state, key)
    obs, actions, a_expert = batch.obs, batch.action, batch.a_expert
    # The expert's torque on Pendulum's [-2, 2], stored in agent units.
    np.testing.assert_allclose(a_expert, jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0) / 2)

    def q_min(a: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, a], axis=-1)
        return jnp.min(predict_value(critic, critic.params, x), axis=0)

    np.testing.assert_allclose(got[KEYS[0]], q_min(a_expert).mean(), rtol=1e-6)
    want = (q_min(a_expert) - q_min(actions)).mean()
    np.testing.assert_allclose(got[KEYS[1]], want, rtol=1e-5, atol=1e-6)
    assert float(got["value/phi_star_q_gap_ood"]) == 0.0


def test_it_needs_an_agent_with_an_expert_and_a_replay() -> None:
    with pytest.raises(ValueError, match="expert_policy"):
        ExpertValueMetrics().bind_to_agent(buffer=None, expert_policy=None)
