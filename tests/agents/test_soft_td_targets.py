"""The soft TD targets' reading of the reward (SAC, REDQ, ASAC): each is
``c r + ...``, the reward scaled once by the agent's reward_scale ``c``, and
ASAC's less its average reward ``theta``, whatever the critics' values."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import ASAC, REDQ, SAC
from ajax.agents.ASAC.train_ASAC import compute_asac_td_target
from ajax.agents.REDQ.train_REDQ import compute_redq_td_target
from ajax.agents.SAC.core import compute_td_target

KEY = jax.random.PRNGKey(0)
ROWS = 4


@functools.cache
def _state(name: str):
    net = ("8", "relu")
    agent = {"SAC": SAC, "REDQ": REDQ, "ASAC": ASAC}[name](
        "Pendulum-v1", actor_architecture=net, critic_architecture=net
    )
    return jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])


def _target(name: str, rewards, reward_scale: float, theta: float = 0.0, dones=None):
    state = _state(name)
    common = {
        "actor_state": state.actor_state,
        "next_observations": jnp.ones((ROWS, 3)),
        "dones": jnp.zeros((ROWS, 1)) if dones is None else dones,
        "rewards": rewards,
        "alpha": jnp.asarray(0.1),
        "reward_scale": reward_scale,
    }
    if name == "SAC":
        return compute_td_target(
            critic_state=state.critic_state,
            gamma=0.9,
            rng=KEY,
            recurrent=False,
            **common,
        )
    if name == "REDQ":
        return compute_redq_td_target(
            critic_states=state.critic_state,
            rng=KEY,
            gamma=0.9,
            subset_size=2,
            **common,
        )[0]
    return compute_asac_td_target(
        critic_states=state.critic_state, rng=KEY, theta=theta, **common
    )[0]


@pytest.mark.parametrize("name", ["SAC", "REDQ", "ASAC"])
def test_the_target_scales_the_reward_once(name: str) -> None:
    """One more unit of reward at reward_scale 2 raises the target by 2: the
    same key draws the same next actions, so nothing else moves."""
    rewards = jnp.linspace(-1.0, 1.0, ROWS)[:, None]
    raised = _target(name, rewards + 1.0, 2.0) - _target(name, rewards, 2.0)
    np.testing.assert_allclose(raised, 2.0, atol=1e-5)


def test_asacs_target_subtracts_its_average_reward() -> None:
    """The differential target is r - theta + ... (Adamczyk et al., 2025)."""
    rewards = jnp.zeros((ROWS, 1))
    raised = _target("ASAC", rewards, 1.0, theta=1.0) - _target("ASAC", rewards, 1.0)
    np.testing.assert_allclose(raised, -1.0, atol=1e-5)


def test_asacs_target_bootstraps_through_every_end() -> None:
    """The differential target has no (1 - d) mask: ASAC charges a
    termination its penalty in the reward instead (train_ASAC's update)."""
    rewards = jnp.zeros((ROWS, 1))
    ended = _target("ASAC", rewards, 1.0, dones=jnp.ones((ROWS, 1)))
    np.testing.assert_allclose(ended, _target("ASAC", rewards, 1.0), atol=1e-6)
