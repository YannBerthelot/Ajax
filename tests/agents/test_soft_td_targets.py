"""The soft TD targets (SAC, REDQ, ASAC): each is ``c r + ...``, the reward
scaled once by the agent's reward_scale ``c``, and ASAC's less its average
reward ``theta``, whatever the critics' values; each cuts the bootstrap at a
termination; the next action's log-probability is the joint one; the target
critics read the agent's observation normalisation."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import ASAC, REDQ, SAC
from ajax.agents.ASAC.train_ASAC import compute_asac_td_target
from ajax.agents.REDQ.train_REDQ import compute_redq_td_target
from ajax.agents.SAC.SAC import compute_td_target, sample_next_actions
from ajax.networks.networks import action_value_input, predict_value
from tests.environments import paid_action

KEY = jax.random.PRNGKey(0)
ROWS = 4


@functools.cache
def _state(name: str, normalize_observations: bool = False):
    net = ("8", "relu")
    agent = {"SAC": SAC, "REDQ": REDQ, "ASAC": ASAC}[name](
        "Pendulum-v1",
        actor_architecture=net,
        critic_architecture=net,
        normalize_observations=normalize_observations,
    )
    return jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])


def _target(
    name: str, rewards, reward_scale: float, theta: float = 0.0, dones=None, state=None
):
    state = _state(name) if state is None else state
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


def test_asacs_target_drops_the_next_value_at_a_termination() -> None:
    """As the authors' code (BobMak/asac@4b68862, darer/ASAC.py:179-205), a
    termination's target is r - theta - Q_target(0, 0): no next value, the
    origin shift on every row."""
    critic = _state("ASAC").critic_state
    zero = action_value_input(jnp.zeros((1, 3)), jnp.zeros((1, 1)))
    origin = jnp.mean(predict_value(critic, critic.target_params, zero))
    rewards = jnp.linspace(-1.0, 1.0, ROWS)[:, None]
    ended = _target("ASAC", rewards, 2.0, theta=0.5, dones=jnp.ones((ROWS, 1)))
    np.testing.assert_allclose(ended, 2.0 * rewards - 0.5 - origin, atol=1e-5)


@pytest.mark.parametrize("name", ["SAC", "REDQ"])
def test_a_termination_cuts_the_bootstrap(name: str) -> None:
    """r + gamma (1 - d)(...): at d = 1 the target is the scaled reward."""
    rewards = jnp.linspace(-1.0, 1.0, ROWS)[:, None]
    ended = _target(name, rewards, 2.0, dones=jnp.ones((ROWS, 1)))
    np.testing.assert_allclose(ended, 2.0 * rewards, atol=1e-6)


def test_the_next_actions_log_probability_is_the_joint_one() -> None:
    """On a 2-D box the entropy term reads log pi(a'|s') summed over the
    dimensions: one column, as the target's."""
    env, params = paid_action.make((-1.0, 1.0), length=8, dim=2)
    agent = SAC(env, env_params=params, actor_architecture=("8", "relu"))
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    actions, log_probs = sample_next_actions(
        state.actor_state, jnp.ones((ROWS, 3)), KEY
    )
    assert actions.shape == (ROWS, 2) and log_probs.shape == (ROWS, 1)


def test_the_target_critics_read_the_agents_observation_normalisation() -> None:
    """With normalize_observations the target critics see s' normalised by
    the agent's statistics: statistics of 100 observations of mean 5 and
    variance 4 move the target off the one at the empty statistics (no
    normalisation before the first observation)."""
    state = _state("SAC", normalize_observations=True)
    info = state.critic_state.obs_norm_info
    var = jnp.full_like(info.var, 4.0)
    shifted = info.replace(mean=info.mean + 5.0, var=var, count=info.count + 100)
    moved = state.replace(
        critic_state=state.critic_state.replace(obs_norm_info=shifted)
    )
    rewards = jnp.zeros((ROWS, 1))
    plain, shifted_target = (
        _target("SAC", rewards, 1.0, state=s) for s in (state, moved)
    )
    assert np.abs(np.asarray(shifted_target - plain)).max() > 1e-3
