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
from ajax.extensions.observe import TrainingProgressInput
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
    "image_shape, env_extra, appended, progress",
    [
        ((2, 2, 1), 0, 0, False),  # the whole observation is the image
        ((1, 3, 1), 1, 0, False),  # PPO-style: one trailing env dim
        ((2, 2, 1), 0, 2, False),  # UDRL-style: the command appended
        ((2, 2, 1), 0, 0, True),  # an observation feature appended
        ((1, 3, 1), 1, 2, True),  # all three
    ],
)
def test_actor_and_state_value_critic_read_the_same_layout(
    cartpole, image_shape, env_extra, appended, progress
):
    """A feature's columns (TrainingProgressInput's one) join after the
    convolutions, with the env's trailing dims and UDRL's command."""
    features = (TrainingProgressInput(),) if progress else ()
    actor_state, critic_state = get_initialized_actor_critic(
        key=jax.random.PRNGKey(0),
        env_config=cartpole.replace(observers=features),
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
    )
    assert _conv_params(actor_state.params) and _conv_params(critic_state.params)
    width = 4 + appended + progress
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, width))
    logits = actor_state.apply_fn(actor_state.params, obs).logits
    assert logits.shape == (BATCH, 2) and jnp.all(jnp.isfinite(logits))
    values = critic_state.apply_fn(critic_state.params, obs)
    assert values.shape == (1, BATCH, 1) and jnp.all(jnp.isfinite(values))


def test_cnn_encoder_rejects_a_miscounted_input_width():
    """With trailing dims the encoder slices the image off the front: a
    miscounted width must raise rather than read the wrong columns."""
    encoder = CNNEncoder(image_shape=(1, 3, 1), extra_obs_dim=2, spec=SPEC)
    with pytest.raises(ValueError, match="trailing dims"):
        encoder.init(jax.random.PRNGKey(0), jnp.ones((BATCH, 4)))


@pytest.mark.parametrize("progress", [False, True])
def test_a_q_network_reads_the_env_trailing_dims(cartpole, progress):
    """DQN's and PQN's network: the env's trailing dims and a feature's
    column join after the convolutions."""
    features = (TrainingProgressInput(),) if progress else ()
    q_state = get_initialized_q_network(
        key=jax.random.PRNGKey(0),
        env_config=cartpole.replace(observers=features),
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
    obs = jnp.ones((BATCH, 4 + progress))
    q_values = q_state.apply_fn(q_state.params, obs).q_values
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


def _encoder_params(spec: CNNSpec, image_shape: tuple) -> dict:
    x = jnp.zeros((1, image_shape[0] * image_shape[1] * image_shape[2]))
    return CNNEncoder(image_shape, spec=spec).init(jax.random.PRNGKey(0), x)["params"]


def test_the_default_spec_is_flax_s_plain_conv_and_dense():
    """CNNSpec() builds what flax's nn.Conv and nn.Dense do at their own
    defaults (lecun_normal, zero biases, SAME padding), bitwise."""
    import flax.linen as nn

    class Plain(nn.Module):
        @nn.compact
        def __call__(self, x):
            img = x.reshape(-1, 10, 10, 4)
            for c, k in ((16, 4), (32, 3)):
                img = nn.relu(nn.Conv(c, (k, k), strides=(2, 2))(img))
            return nn.relu(nn.Dense(128)(img.reshape(img.shape[0], -1)))

    x = jnp.zeros((1, 400))
    want = Plain().init(jax.random.PRNGKey(0), x)["params"]
    got = _encoder_params(CNNSpec(), (10, 10, 4))
    assert jax.tree.structure(got) == jax.tree.structure(want)
    assert all(jax.tree.leaves(jax.tree.map(lambda a, b: (a == b).all(), got, want)))


def test_valid_padding_and_layer_norm_build_mnih_s_stack_with_norms():
    """VALID padding on 84x84x4 leaves Mnih et al.'s 7x7x64 = 3136 features
    for the 512 projection; layer_norm adds one LayerNorm per layer."""
    spec = CNNSpec((32, 64, 64), (8, 4, 3), (4, 2, 1), 512, "VALID", True)
    params = _encoder_params(spec, (84, 84, 4))
    assert params["Dense_0"]["kernel"].shape == (3136, 512)
    assert sorted(k for k in params if k.startswith("LayerNorm")) == [
        f"LayerNorm_{i}" for i in range(4)
    ]


def test_torch_linear_draws_every_kernel_and_bias_within_one_over_root_fan_in():
    """Torch7's and torch's default: uniform in +-1/sqrt(fan_in), the fan-in
    of a convolution its kernel area times its input channels."""
    spec = CNNSpec((16,), (3,), (1,), 128, "VALID", kernel_init="torch_linear")
    params = _encoder_params(spec, (10, 10, 4))
    for name, fan_in in (("Conv_0", 3 * 3 * 4), ("TorchLinear_0", 8 * 8 * 16)):
        bound = 1 / fan_in**0.5
        for leaf in ("kernel", "bias"):
            x = params[name][leaf]
            assert 0.5 * bound < jnp.abs(x).max() <= bound, (name, leaf)
