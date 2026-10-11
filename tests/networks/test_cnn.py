"""The CNN encoder path: every network reads the flat observation as
``(H*W*C + trailing dims)``, the trailing dims being the env's own
(``NetworkConfig.cnn_extra_obs_dim``) and those an agent appends at run time
(the train fraction, UDRL's command)."""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.DQN.networks import get_initialized_q_network
from ajax.agents.UDRL.UDRL import UDRL
from ajax.environments.create import build_env_from_id
from ajax.networks.networks import (
    CNNEncoder,
    CNNSpec,
    get_initialized_actor_critic,
)
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig

ARCH = ("8", "relu")
SPEC = CNNSpec(channels=(4,), kernel_sizes=(2,), strides=(1,), feature_dim=8)
OPTIMIZER = OptimizerConfig(learning_rate=1e-3)
BATCH = 3


@pytest.fixture
def cartpole():
    """CartPole's 4 observation dims, read as a tiny image."""
    env, env_params = build_env_from_id("CartPole-v1")
    return EnvironmentConfig(env=env, env_params=env_params, n_envs=1, continuous=False)


def _conv_params(params) -> list:
    return [
        path
        for path, _ in jax.tree_util.tree_leaves_with_path(params)
        if "Conv" in jax.tree_util.keystr(path)
    ]


@pytest.mark.parametrize(
    "image_shape, env_extra, appended, max_timesteps",
    [
        ((2, 2, 1), 0, 0, None),  # the whole observation is the image
        ((1, 3, 1), 1, 0, None),  # PPO-style: one trailing env dim
        ((2, 2, 1), 0, 2, None),  # UDRL-style: the command appended
        ((2, 2, 1), 0, 0, 100),  # the train fraction appended
        ((1, 3, 1), 1, 2, 100),  # all three
    ],
)
def test_actor_and_state_value_critic_read_the_same_layout(
    cartpole, image_shape, env_extra, appended, max_timesteps
):
    actor_state, critic_state = get_initialized_actor_critic(
        key=jax.random.PRNGKey(0),
        env_config=cartpole,
        actor_optimizer_config=OPTIMIZER,
        critic_optimizer_config=OPTIMIZER,
        network_config=NetworkConfig(
            actor_architecture=ARCH,
            critic_architecture=ARCH,
            cnn_image_shape=image_shape,
            cnn_extra_obs_dim=env_extra,
            cnn_spec=SPEC,
        ),
        num_critics=1,
        extra_obs_dim=appended,
        max_timesteps=max_timesteps,
    )
    assert _conv_params(actor_state.params) and _conv_params(critic_state.params)
    width = 4 + appended + (max_timesteps is not None)
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, width))
    logits = actor_state.apply_fn(actor_state.params, obs).logits
    assert logits.shape == (BATCH, 2) and jnp.all(jnp.isfinite(logits))
    values = critic_state.apply_fn(critic_state.params, obs)
    assert values.shape == (1, BATCH, 1) and jnp.all(jnp.isfinite(values))


def test_cnn_encoder_rejects_a_miscounted_input_width():
    """With trailing dims the encoder slices the image off the front: a
    miscounted width must raise rather than read the wrong columns."""
    encoder = CNNEncoder(image_shape=(1, 3, 1), extra_obs_dim=2, channels=(4,))
    with pytest.raises(ValueError, match="trailing dims"):
        encoder.init(jax.random.PRNGKey(0), jnp.ones((BATCH, 4)))


def test_a_q_network_reads_the_env_trailing_dims(cartpole):
    q_state = get_initialized_q_network(
        key=jax.random.PRNGKey(0),
        env_config=cartpole,
        optimizer_config=OPTIMIZER,
        network_config=NetworkConfig(
            actor_architecture=ARCH,
            critic_architecture=ARCH,
            cnn_image_shape=(1, 3, 1),
            cnn_extra_obs_dim=1,
            cnn_spec=SPEC,
        ),
        n_actions=2,
    )
    assert _conv_params(q_state.params)
    q_values = q_state.apply_fn(q_state.params, jnp.ones((BATCH, 4))).q_values
    assert q_values.shape == (BATCH, 2) and jnp.all(jnp.isfinite(q_values))


def test_udrl_trains_a_cnn_policy_on_image_and_command():
    agent = UDRL(
        "CartPole-v1",
        actor_architecture=ARCH,
        n_steps=8,
        batch_size=8,
        n_updates_per_iter=1,
        buffer_capacity=8,
        cnn_image_shape=(2, 2, 1),
    )
    state, _ = agent.train(seed=0, n_timesteps=16, num_episode_test=1)
    params = jax.tree.map(lambda leaf: leaf[0], state.actor_state.params)
    assert _conv_params(params)
    command_obs = jnp.ones((BATCH, 4 + 2))  # the image, then (d_r, d_h)
    logits = state.actor_state.apply_fn(params, command_obs).logits
    assert logits.shape == (BATCH, 2) and jnp.all(jnp.isfinite(logits))
