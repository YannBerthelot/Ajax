"""The ImitationLoss extension: its term and its wiring into four actor losses.

TD3, REDQ, ASAC and APO put the actor's mean action (``pi_mean``) and the
raw observations into their actor-loss batch. One actor update per agent,
run eagerly on a tiny network, checks that the extension's term changes the
actor gradient and has no effect at zero weight or when absent.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.ASAC import train_ASAC
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.REDQ import train_REDQ
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.TD3 import train_TD3
from ajax.agents.TD3.TD3 import TD3
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.extensions.expert import ImitationLoss

NET = ("8", "relu")
BATCH = 4


@dataclasses.dataclass(frozen=True)
class ConstantExpert:
    action: float

    def __call__(self, obs: jax.Array) -> jax.Array:
        return jnp.full(obs.shape[:-1] + (1,), self.action, jnp.float32)


EXPERT = ConstantExpert(0.9)
KEY = jax.random.PRNGKey(0)
CTX = ExtensionContext(step=jnp.asarray(0), rng=KEY)


def test_term_is_the_scaled_squared_distance_to_the_expert() -> None:
    pi_mean = jnp.array([[0.1], [-0.3]])
    obs = jnp.zeros((2, 3))
    batch = {"pi_mean": pi_mean, "raw_observations": obs}
    term = ImitationLoss(expert_policy=EXPERT, coef=2.0).actor_loss(
        None, (), batch, CTX
    )
    expected = 2.0 * np.mean((np.array([0.1, -0.3]) - 0.9) ** 2) / 4.0
    np.testing.assert_allclose(float(term), expected, rtol=1e-6)
    # Without raw observations the expert reads the agent's observations.
    batch = {"pi_mean": pi_mean, "raw_observations": None, "observations": obs}
    again = ImitationLoss(expert_policy=EXPERT, coef=2.0).actor_loss(
        None, (), batch, CTX
    )
    np.testing.assert_allclose(float(again), expected, rtol=1e-6)


def test_agent_without_policy_mean_raises() -> None:
    batch = {"observations": jnp.zeros((2, 3))}
    with pytest.raises(ValueError, match="pi_mean"):
        ImitationLoss(expert_policy=EXPERT).actor_loss(None, (), batch, CTX)


def _ext_state(state: Any, stack: ExtensionStack) -> Any:
    """What make_train does before training: one ext_state per extension."""
    return stack.fold_init_states(state, KEY)


def _obs(agent: Any) -> jax.Array:
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    return jax.random.normal(jax.random.PRNGKey(1), (BATCH, *obs_shape))


def _td3_update(stack: ExtensionStack) -> Any:
    agent = TD3("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    state = train_TD3.init_TD3(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.buffer,
    )
    state = _ext_state(state, stack)
    obs = _obs(agent)
    state, _ = train_TD3.update_policy(state, obs, obs, stack, total_timesteps=1)
    return state.actor_state.params


def _sac_family_update(agent_cls: type, module: Any, init: Callable) -> Callable:
    def update(stack: ExtensionStack) -> Any:
        agent = agent_cls(
            "Pendulum-v1", actor_architecture=NET, critic_architecture=NET
        )
        state = _ext_state(init(agent), stack)
        obs = _obs(agent)
        out = module.update_policy(state, obs, obs, stack, total_timesteps=1)
        return out[0].actor_state.params

    return update


def _init_redq(agent: REDQ) -> Any:
    return train_REDQ.init_REDQ(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
        number_of_critics=agent.agent_config.num_critics,
    )


def _init_asac(agent: ASAC) -> Any:
    return train_ASAC.init_ASAC(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
    )


def _apo_update(stack: ExtensionStack) -> Any:
    agent = APO(
        "Pendulum-v1", n_envs=1, actor_architecture=NET, critic_architecture=NET
    )
    state = train_APO.init_APO(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    state = _ext_state(state, stack)
    obs = _obs(agent)
    state, _ = train_APO.update_policy(
        state,
        obs,
        actions=jnp.zeros((BATCH, 1)),
        gae=jnp.ones((BATCH, 1)),
        log_probs=jnp.zeros((BATCH, 1)),
        clip_coef=0.2,
        ent_coef=0.0,
        extension_stack=stack,
        total_timesteps=1,
        raw_observations=obs,
    )
    return state.actor_state.params


UPDATES: dict[str, Callable[[ExtensionStack], Any]] = {
    "TD3": _td3_update,
    "REDQ": _sac_family_update(REDQ, train_REDQ, _init_redq),
    "ASAC": _sac_family_update(ASAC, train_ASAC, _init_asac),
    "APO": _apo_update,
}


def _stack(coef: float) -> ExtensionStack:
    return ExtensionStack((ImitationLoss(expert_policy=EXPERT, coef=coef),))


def _same(a: Any, b: Any) -> bool:
    leaves = zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    return all(np.allclose(x, y, rtol=1e-6, atol=1e-8) for x, y in leaves)


@pytest.mark.parametrize("agent", sorted(UPDATES))
def test_imitation_term_steers_the_actor_update(agent: str) -> None:
    update = UPDATES[agent]
    absent = update(ExtensionStack())
    assert _same(update(_stack(0.0)), absent), "zero weight must change nothing"
    assert not _same(update(_stack(100.0)), absent), "the term must reach the actor"
