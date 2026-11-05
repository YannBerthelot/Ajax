"""TD3's critics read actions in the env's units, as sfujim/TD3's (34770cc)
read max_action * tanh: Ajax's agents act in [-1, 1], and a critic trained on
those takes Adam steps of another size on its action inputs."""

from __future__ import annotations

from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.PQN.PQN import PQN
from ajax.agents.recurrent import q_values
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.environments.system_class import broadcast_env_params
from ajax.environments.utils import critic_action_units
from ajax.extensions.expert import td3bc_lambda
from ajax.networks.deterministic import get_initialized_td3_actor_critic
from ajax.networks.networks import predict_value
from tests.agents.TD3.test_td3_metrics import KEY, NET, _batch


def _critic(agent: TD3, units: tuple) -> Any:
    """TD3's critic, reading actions in the units ``(offsets, scales)``."""
    _, critic = get_initialized_td3_actor_critic(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    return critic.replace(action_units=units)


def test_the_units_are_the_env_map_without_its_clip_or_none():
    def units(agent):
        return critic_action_units(agent.env_args.env, agent.env_args.env_params)

    assert units(TD3("Pendulum-v1")) == ((0.0,), (2.0,))  # torque in [-2, 2]
    assert units(TD3("MountainCarContinuous-v0")) is None  # already [-1, 1]
    assert units(PQN("CartPole-v1")) is None  # discrete


@pytest.mark.parametrize(
    "low, high, several",
    [
        ((-3.0, -3.0), (5.0, 5.0), False),
        ((-1.0, 0.0), (1.0, 2.0), True),
        ((-1.0, -2.0), (1.0, 2.0), True),
    ],
    ids=["one-box", "two-centres", "two-half-ranges"],
)
def test_per_env_boxes_must_be_one_box(low, high, several):
    """Catches a critic reading one env's units for another's rows, which
    share its batch: a box differing in its centre alone or its half-range
    alone is refused, and envs sharing a box are not."""
    env, params = gymnax.make("MountainCarContinuous-v0")
    per_env = broadcast_env_params(params, 2).replace(
        min_action=jnp.array(low), max_action=jnp.array(high)
    )

    def refused() -> bool:
        try:
            critic_action_units(env, per_env)
        except ValueError:
            return True
        return False

    assert refused() == several


def test_the_critic_reads_the_mapped_actions():
    agent = TD3("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    critic, batch = _critic(agent, units=((0.5,), (2.0,))), _batch(agent)
    plain = critic.replace(action_units=None)
    np.testing.assert_allclose(
        q_values(critic, critic.params, batch.obs, batch.action),
        q_values(plain, plain.params, batch.obs, 0.5 + 2.0 * batch.action),
        rtol=1e-6,
    )


def test_td3_sets_its_critics_units_and_sac_does_not():
    td3, _ = TD3(
        "Pendulum-v1",
        actor_architecture=NET,
        critic_architecture=NET,
        learning_starts=4,
    ).train(seed=[0], n_timesteps=8)
    sac, _ = SAC(
        "Pendulum-v1",
        actor_architecture=NET,
        critic_architecture=NET,
        learning_starts=4,
    ).train(seed=[0], n_timesteps=8)
    assert td3.critic_state.action_units == ((0.0,), (2.0,))
    assert sac.critic_state.action_units is None
    assert jax.tree.structure(td3.critic_state) != jax.tree.structure(
        td3.critic_state.replace(action_units=None)
    )


def test_every_query_of_the_critic_reads_the_mapped_actions():
    """``predict_value``, which the extensions and the evaluation call with
    ``concat(obs, action)``, maps the action slice too: TD3+BC's lambda is
    one such query."""
    agent = TD3("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    critic, batch = _critic(agent, units=((0.5,), (2.0,))), _batch(agent)
    plain = critic.replace(action_units=None)
    x = jnp.concatenate([batch.obs, batch.action], axis=-1)
    mapped = jnp.concatenate([batch.obs, 0.5 + 2.0 * batch.action], axis=-1)
    np.testing.assert_allclose(
        predict_value(critic, critic.params, x),
        predict_value(plain, plain.params, mapped),
        rtol=1e-6,
    )
    q1 = q_values(critic, critic.params, batch.obs, batch.action)[0]
    np.testing.assert_allclose(
        td3bc_lambda(critic, batch.obs, batch.action, 2.5),
        2.5 / jnp.mean(jnp.abs(q1)),
        rtol=1e-6,
    )
