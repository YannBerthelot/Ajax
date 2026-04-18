"""SAC-specific training tests.

Tests functions and mechanics unique to SAC: temperature (alpha) loss and
update, and soft target-network updates. Shared behaviors (loss shapes,
agent-state updates, training loop, make_train) are covered by the probing
suite (``tests/agents/test_probing.py``) and the per-agent smoke tests
(``test_sac.py``).
"""
import gymnax
import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState

from ajax.agents.SAC.train_SAC import (
    create_alpha_train_state,
    init_SAC,
    temperature_loss_function,
    update_target_networks,
    update_temperature,
)
from ajax.buffers.utils import get_buffer
from ajax.state import (
    AlphaConfig,
    BufferConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.utils import compare_frozen_dicts


@pytest.fixture
def fast_env_config():
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(
        env=env,
        env_params=None,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture
def gymnax_env_config():
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture(params=["fast_env_config", "gymnax_env_config"])
def env_config(request, fast_env_config, gymnax_env_config):
    return fast_env_config if request.param == "fast_env_config" else gymnax_env_config


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
        lstm_hidden_size=None,
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
    "log_alpha_init, target_entropy, corrected_log_probs",
    [
        (0.0, -1.0, jnp.array([-0.5, -1.5, -1.0])),
        (-1.0, -2.0, jnp.array([-1.0, -2.0, -1.5])),
    ],
)
def test_temperature_loss_function(log_alpha_init, target_entropy, corrected_log_probs):
    log_alpha_params = FrozenDict({"log_alpha": jnp.array(log_alpha_init)})

    loss, aux = temperature_loss_function(
        log_alpha_params=log_alpha_params,
        corrected_log_probs=corrected_log_probs,
        effective_target_entropy=target_entropy,
    )
    aux = to_state_dict(aux)
    assert jnp.isfinite(loss)
    assert "alpha" in aux
    assert "log_alpha" in aux
    assert aux["alpha"] > 0


@pytest.mark.parametrize(
    "log_alpha_init, target_entropy, corrected_log_probs",
    [
        (0.0, -1.0, jnp.array([-0.5, -1.5, -1.0])),
        (-1.0, -2.0, jnp.array([-1.0, -2.0, -1.5])),
    ],
)
def test_temperature_loss_function_with_value_and_grad(
    log_alpha_init, target_entropy, corrected_log_probs
):
    log_alpha_params = FrozenDict({"log_alpha": jnp.array(log_alpha_init)})

    def loss_fn(log_alpha_params):
        loss, _ = temperature_loss_function(
            log_alpha_params=log_alpha_params,
            corrected_log_probs=corrected_log_probs,
            effective_target_entropy=target_entropy,
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(log_alpha_params)

    assert jnp.isfinite(loss)
    assert isinstance(grads, FrozenDict)
    assert all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    )


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_temperature(env_config, SAC_state):
    log_probs = jnp.ones((env_config.n_envs, 1)) * -0.5
    effective_target_entropy = jnp.array(-1.0)

    original_alpha_params = SAC_state.alpha.params

    updated_state, aux = update_temperature(
        agent_state=SAC_state,
        log_probs=log_probs,
        effective_target_entropy=effective_target_entropy,
    )
    aux = to_state_dict(aux)
    assert not compare_frozen_dicts(updated_state.alpha.params, original_alpha_params)
    assert "alpha" in aux
    assert aux["alpha"] > 0


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
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
        assert validate_soft_update(old_target, new_target, current, tau), (
            f"Soft update computation is incorrect for key: {key}"
        )
