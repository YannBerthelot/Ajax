"""SAC-specific training tests.

Tests functions and mechanics unique to SAC: temperature (alpha) loss and
update, and soft target-network updates. Shared behaviors (loss shapes,
agent-state updates, training loop, make_train) are covered by the probing
suite (``tests/probing``) and the tiny runs of
``tests/agents/test_agent_contract.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState

from ajax.agents.SAC.core import (
    create_alpha_train_state,
    temperature_loss_fn,
    update_target_networks,
    update_temperature,
)
from ajax.agents.SAC.train_SAC import init_SAC
from ajax.buffers.utils import get_buffer
from ajax.state import (
    AlphaConfig,
    BufferConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.utils import compare_frozen_dicts


@pytest.fixture
def buffer(env_config):
    return get_buffer(
        **to_state_dict(
            BufferConfig(buffer_size=1000, batch_size=32, n_envs=env_config.n_envs)
        )
    )


@pytest.fixture
def SAC_state(env_config, buffer):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)

    return init_SAC(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=buffer,
    )


def test_create_alpha_train_state():
    learning_rate = 3e-4
    alpha_init = 1.0

    train_state = create_alpha_train_state(
        learning_rate=learning_rate, alpha_init=alpha_init
    )

    assert isinstance(train_state, TrainState)
    assert jnp.isclose(train_state.params["log_alpha"], jnp.log(alpha_init))
    assert train_state.tx is not None


@pytest.mark.parametrize(
    "log_alpha, target_entropy, log_probs, grad",
    [
        # grad = mean(-log_probs) - target_entropy: positive (alpha shrinks)
        # while the entropy is above the target, negative below it
        (0.0, -1.0, [-0.5, -1.5, -1.0], 1.0 + 1.0),
        (-1.0, -2.0, [-1.0, -2.0, -1.5], 1.5 + 2.0),
        (0.5, 1.0, [-0.1, -0.2], 0.15 - 1.0),
    ],
    ids=["above-target", "above-target-2", "below-target"],
)
def test_temperature_loss_function(log_alpha, target_entropy, log_probs, grad):
    """``log_alpha * mean(-log_probs - target_entropy)``, the bracket held
    constant: its gradient in ``log_alpha`` is the bracket."""
    (loss, aux), grads = jax.value_and_grad(temperature_loss_fn, has_aux=True)(
        FrozenDict({"log_alpha": jnp.array(log_alpha)}),
        log_probs=jnp.array(log_probs),
        target_entropy=target_entropy,
    )
    aux = to_state_dict(aux)
    assert isinstance(grads, FrozenDict)
    np.testing.assert_allclose(grads["log_alpha"], grad, rtol=1e-6)
    np.testing.assert_allclose(loss, log_alpha * grad, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(aux["alpha"], np.exp(log_alpha), rtol=1e-6)
    np.testing.assert_allclose(aux["log_alpha"], log_alpha)


def test_update_temperature(env_config, SAC_state):
    log_probs = jnp.ones((env_config.n_envs, 1)) * -0.5
    target_entropy = jnp.array(-1.0)

    original_alpha_params = SAC_state.alpha.params

    updated_state, aux = update_temperature(
        agent_state=SAC_state,
        log_probs=log_probs,
        target_entropy=target_entropy,
    )
    aux = to_state_dict(aux)
    assert not compare_frozen_dicts(updated_state.alpha.params, original_alpha_params)
    assert "alpha" in aux
    assert aux["alpha"] > 0


def test_update_target_networks(env_config, SAC_state):
    tau = 0.05

    critic_state = SAC_state.critic_state
    shifted_params = FrozenDict(
        jax.tree_util.tree_map(lambda x: x + 1, critic_state.params)
    )
    critic_state = critic_state.replace(params=shifted_params)
    SAC_state = SAC_state.replace(critic_state=critic_state)

    original_params = SAC_state.critic_state.params
    original_target_params = SAC_state.critic_state.target_params

    updated_state = update_target_networks(SAC_state, tau=tau)

    assert compare_frozen_dicts(updated_state.critic_state.params, original_params)
    assert not compare_frozen_dicts(
        updated_state.critic_state.target_params, original_target_params
    )

    def validate_soft_update(old_target, new_target, current, tau):
        expected_target = jax.tree_util.tree_map(
            lambda old, cur: tau * cur + (1 - tau) * old,
            old_target,
            current,
        )
        return compare_frozen_dicts(new_target, expected_target)

    for key in original_target_params.keys():
        old_target = original_target_params[key]
        new_target = updated_state.critic_state.target_params[key]
        current = original_params[key]
        assert validate_soft_update(
            old_target, new_target, current, tau
        ), f"Soft update computation is incorrect for key: {key}"
