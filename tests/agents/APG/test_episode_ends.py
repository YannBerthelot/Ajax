"""APG's closed loop at episode ends, on Pendulum with transition gradients
and two systems whose episodes end at the time limits 3 and 4, so their
ends fall apart, over a horizon of 10:

(i) no gradient crosses an episode end: the gradient of every reward with
respect to every action of an earlier episode (or of another env), and to
the controller's carry entering any step of one, is exactly 0; within an
episode both flow;
(ii) the controller restarts its carry where ``rollout.resets`` flags an
episode start, for every memory kind with the PID head and the PID head
alone: each episode's actions are the controller's run from a fresh carry
over that episode's observations.

The rollout's reset flags themselves are closed_loop_rollout's contract
(tests/environments/test_differentiable.py).
"""

from types import SimpleNamespace
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax import make as make_gymnax_env

from ajax.agents.APG.networks import Controller, PIDHeadConfig
from ajax.agents.APG.train_APG import make_policy_step
from ajax.environments.differentiable import (
    closed_loop_rollout,
    with_transition_gradients,
)
from ajax.environments.system_class import broadcast_env_params
from ajax.environments.utils import agent_action_to_env
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig
from ajax.state import LoadedTrainState

HORIZON, LENGTHS = 10, (3, 4)
B = len(LENGTHS)
KEY = jax.random.PRNGKey(0)


def _setup(memory: Optional[MemoryConfig]) -> tuple:
    """The env, its two systems, the controller (memory, PID head), its
    parameters and fresh carry."""
    env, params = make_gymnax_env("Pendulum-v1")
    env = with_transition_gradients(env)
    params = broadcast_env_params(params, B)
    params = params.replace(max_steps_in_episode=jnp.array(LENGTHS))
    # Non-zero integral and derivative gains: the head's carry reaches the action.
    pid = PIDHeadConfig(ki_init=0.2, kd_init=0.2)
    net = Controller(("8", "tanh"), 1, memory=memory, pid=pid)
    carry = net.initialize_carry(KEY, B)
    obs = jnp.zeros((1, B, 3))
    weights = net.init(KEY, obs, hidden_state=carry, done=jnp.zeros((1, B), bool))
    return env, params, net, weights, carry


def _actor(net: Controller) -> LoadedTrainState:
    """A stand-in actor state: make_policy_step reads only its ``apply``."""
    return cast(LoadedTrainState, SimpleNamespace(apply=net.apply))


def _episodes(resets: np.ndarray) -> np.ndarray:
    """``(T, B)``: the index of the episode each step belongs to, per env."""
    return np.cumsum(resets, axis=0)


def test_no_gradient_crosses_an_episode_end() -> None:
    env, params, net, weights, carry = _setup(MemoryConfig(kind="gru", hidden_size=8))
    policy = make_policy_step(_actor(net), weights, True)

    def rewards(d_action: jax.Array, d_carry: Any) -> tuple:
        """The rollout's rewards with ``d_action[t]`` added to the action of
        step ``t`` and ``d_carry[t]`` to the carry entering it (both 0)."""

        def step(state, obs, resets):
            t, inner = state
            inner = jax.tree.map(lambda c, d: c + d[t], inner, d_carry)
            action, inner = policy(inner, obs, resets)
            action = agent_action_to_env(action + d_action[t], env, params)
            return action, (t + 1, inner)

        rollout, _ = closed_loop_rollout(step, (0, carry), KEY, env, params, HORIZON)
        return rollout.reward, rollout

    d_action = jnp.zeros((HORIZON, B, 1))
    d_carry = jax.tree.map(lambda c: jnp.zeros((HORIZON, *c.shape)), carry)
    grads, rollout = jax.jacrev(rewards, argnums=(0, 1), has_aux=True)(
        d_action, d_carry
    )
    # (t', b', t, b): reward t' of env b' by the action or carry of t, b.
    by_action = np.abs(np.asarray(grads[0]))[..., 0]
    by_carry = np.concatenate(
        [
            np.abs(np.asarray(g)).reshape(HORIZON, B, HORIZON, B, -1)
            for g in jax.tree.leaves(grads[1])
        ],
        axis=-1,
    ).sum(-1)
    resets = np.asarray(rollout.resets)
    ends = [list(np.flatnonzero(np.asarray(rollout.done)[:, b])) for b in range(B)]
    assert ends == [[2, 5, 8], [3, 7]], ends
    episode = _episodes(resets)
    # The carry entering a reset step still holds the previous episode's.
    carry_episode = episode - resets
    t_, b_, t, b = np.indices(by_action.shape)
    later = t_ >= t
    same = (b_ == b) & later
    action_ok = same & (episode[t_, b_] == episode[t, b])
    carry_ok = same & (episode[t_, b_] == carry_episode[t, b])
    assert not by_action[~action_ok].any(), np.argwhere(by_action * ~action_ok)
    assert not by_carry[~carry_ok].any(), np.argwhere(by_carry * ~carry_ok)
    # Within an episode the gradient flows, through the env and the carry.
    assert by_action[action_ok & (t_ > t)].all()
    assert by_carry[carry_ok].all()


CONTROLLERS = [
    pytest.param(MemoryConfig(kind=k, hidden_size=8), id=k) for k in MEMORY_KINDS
]
CONTROLLERS += [pytest.param(None, id="pid")]


@pytest.mark.parametrize("memory", CONTROLLERS)
def test_the_controller_restarts_its_carry_at_each_episode_start(
    memory: Optional[MemoryConfig],
) -> None:
    env, params, net, weights, carry = _setup(memory)
    policy = make_policy_step(_actor(net), weights, True)
    rollout, _ = closed_loop_rollout(policy, carry, KEY, env, params, HORIZON)
    obs, action = np.asarray(rollout.obs), np.asarray(rollout.action)[..., 0]
    resets = np.asarray(rollout.resets)
    assert resets[1:].any(), "no episode ended"
    episode, checked = _episodes(resets), 0
    for b in range(B):
        for e in np.unique(episode[:, b]):
            steps = np.flatnonzero(episode[:, b] == e)
            fresh = net.initialize_carry(KEY, 1)
            done = jnp.zeros((len(steps), 1), bool).at[0].set(True)
            x = jnp.asarray(obs[steps, b])[:, None]
            pi, _ = net.apply(weights, x, hidden_state=fresh, done=done)
            np.testing.assert_allclose(
                action[steps, b], np.asarray(pi.mean())[:, 0, 0], rtol=1e-5, atol=1e-6
            )
            checked += len(steps) > 1 and e > 1
    assert checked, "no later episode ran past its first step"
