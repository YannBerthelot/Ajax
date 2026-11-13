"""Each network's hidden layers after its trunk (NetworkConfig.head_architecture).

Contract (the owner's request, 2026-11-13; PureJaxRL @31756b1 ppo_rnn.py:51-79,
whose actor and critic share an embedding and a GRU, then each read the GRU
through a Dense(128) + ReLU of its own before its output layer):

- every network's output layers (the policy's, the value's, TD3's action)
  read the trunk's features (the encoder's, then the memory's) through the
  head layers;
- the head layers start with ``head_kernel_init``, else the encoder's
  kernel init;
- each network, and each critic of an ensemble, has its own head layers,
  also when the actor and the critic share their encoder and memory;
- ``()``, the default, reads the trunk's features linearly.
"""

from __future__ import annotations

from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.networks.deterministic import get_initialized_td3_actor_critic
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig

OPTIMIZER = OptimizerConfig(learning_rate=1e-3)
HEAD = ("3", "relu")


@pytest.fixture
def pendulum() -> EnvironmentConfig:
    """Three observation features, one action."""
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(env=env, env_params=env_params, n_envs=1, continuous=True)


@pytest.mark.parametrize(
    "head_architecture, head_kernel_init, expected",
    [
        # Constant kernels, zero biases, an all-ones observation of 3 features:
        # each of the encoder's 2 units is 0.5 * 3 = 1.5, so the output layer
        # (kernel 1.0) reads them as 1.5 + 1.5 = 3.0 ...
        ((), None, 3.0),
        # ... or reads 3 head units, each 0.5 * (1.5 + 1.5) = 1.5 with the
        # encoder's init: 3 * 1.5 = 4.5 ...
        (HEAD, None, 4.5),
        # ... or each 0.25 * 3.0 = 0.75 with their own: 3 * 0.75 = 2.25.
        (HEAD, "0.25", 2.25),
    ],
)
def test_the_output_layers_read_the_trunk_through_the_head_layers(
    pendulum, head_architecture, head_kernel_init, expected
):
    network_config = NetworkConfig(
        actor_architecture=("2",),
        critic_architecture=("2",),
        encoder_kernel_init="0.5",
        mean_kernel_init="1.0",
        value_kernel_init="1.0",
        log_std_state_independent=True,
        head_architecture=head_architecture,
        head_kernel_init=head_kernel_init,
    )
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        pendulum,
        OPTIMIZER,
        OPTIMIZER,
        network_config,
        continuous=True,
        num_critics=1,
    )
    observation = jnp.ones((1, 3))
    policy = actor_state.apply_fn(actor_state.params, observation)
    value = critic_state.apply_fn(critic_state.params, observation)
    np.testing.assert_allclose(policy.mean(), [[expected]], rtol=1e-6)
    np.testing.assert_allclose(value, [[[expected]]], rtol=1e-6)


def _parameter_count(params: Any) -> int:
    return sum(leaf.size for leaf in jax.tree.leaves(params))


def _build(env_config: EnvironmentConfig, head_architecture: tuple, case: dict):
    """The actor's and the critic's states for ``case`` with these head layers."""
    network_config = NetworkConfig(
        actor_architecture=("5",),
        critic_architecture=case.get("critic_architecture", ("5",)),
        memory=case.get("memory"),
        share_encoder=case.get("share_encoder", False),
        head_architecture=head_architecture,
    )
    key = jax.random.PRNGKey(0)
    if case.get("td3"):
        return get_initialized_td3_actor_critic(
            key, env_config, OPTIMIZER, OPTIMIZER, network_config, num_critics=2
        )
    return get_initialized_actor_critic(
        key,
        env_config,
        OPTIMIZER,
        OPTIMIZER,
        network_config,
        continuous=True,
        action_value=case.get("action_value", False),
        num_critics=case.get("num_critics", 2),
    )


GRU = MemoryConfig("gru", 6)


# A head layer as wide as the F features it reads leaves the output layers
# as they were and adds F * F + F parameters to its network. F is the
# memory's 6 units after a memory, else the last hidden layer's 5 units; for
# a recurrent Q-critic, the 6 units of the layers the action joins after the
# memory (critic_architecture), which read before its value layer.
@pytest.mark.parametrize(
    "case, width, actor_added, critic_added",
    [
        ({"memory": GRU}, 6, 6 * 6 + 6, 2 * (6 * 6 + 6)),
        ({"memory": GRU, "share_encoder": True, "num_critics": 1}, 6, 42, 42),
        ({"action_value": True}, 5, 5 * 5 + 5, 2 * (5 * 5 + 5)),
        (
            {"memory": GRU, "td3": True, "critic_architecture": ("6",)},
            6,
            6 * 6 + 6,
            2 * (6 * 6 + 6),
        ),
    ],
    ids=["recurrent", "shared-recurrent", "q-critic", "recurrent-td3"],
)
def test_each_network_has_its_own_head_layers_after_its_trunk(
    pendulum, case, width, actor_added, critic_added
):
    plain_actor, plain_critic = _build(pendulum, (), case)
    actor, critic = _build(pendulum, (str(width), "relu"), case)
    assert (
        _parameter_count(actor.params) - _parameter_count(plain_actor.params)
        == actor_added
    )
    assert (
        _parameter_count(critic.params) - _parameter_count(plain_critic.params)
        == critic_added
    )
    # A shared trunk stays the critic's alone: no head layer joins it.
    assert _parameter_count(actor.shared_params) == _parameter_count(
        plain_actor.shared_params
    )
