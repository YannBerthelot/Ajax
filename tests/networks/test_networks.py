import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from gymnax import EnvParams

from ajax.environments.create import build_env_from_id
from ajax.environments.utils import get_state_action_shapes
from ajax.networks.networks import (
    Actor,
    Critic,
    Encoder,
    MultiCritic,
    get_initialized_actor_critic,
    predict_value,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)


@pytest.fixture
def sample_input():
    return jnp.ones((4, 8))  # Example input with batch size 4 and feature size 8


@pytest.fixture
def encoder_architecture():
    return ["16", "relu", "8"]


def test_encoder(sample_input, encoder_architecture):
    encoder = Encoder(input_architecture=encoder_architecture)
    params = encoder.init(jax.random.PRNGKey(0), sample_input)
    output = encoder.apply(params, sample_input)
    assert output.shape == (4, 8)  # Ensure output shape matches expected dimensions


@pytest.fixture
def actor_architecture():
    return ["16", "relu"]


def test_actor_continuous(sample_input, actor_architecture):
    actor = Actor(
        action_dim=4,
        input_architecture=actor_architecture,
        continuous=True,
    )
    params = actor.init(jax.random.PRNGKey(0), sample_input)
    distribution = actor.apply(params, sample_input)
    assert distribution.mean().shape == (4, 4)  # Ensure mean shape matches action_dim
    assert distribution.stddev().shape == (
        4,
        4,
    )  # Ensure stddev shape matches action_dim


def test_actor_discrete(sample_input, actor_architecture):
    actor = Actor(
        action_dim=4,
        input_architecture=actor_architecture,
        continuous=False,
    )
    params = actor.init(jax.random.PRNGKey(0), sample_input)
    distribution = actor.apply(params, sample_input)
    assert distribution.logits.shape == (4, 4)  # Ensure logits shape matches action_dim


@pytest.fixture
def critic_architecture():
    return ["16", "relu"]


def test_critic(sample_input, critic_architecture):
    critic = Critic(
        input_architecture=critic_architecture,
    )
    action_input = jnp.ones(
        (4, 2)
    )  # Example action input with batch size 4 and action size 2
    params = critic.init(
        jax.random.PRNGKey(0), jnp.hstack([sample_input, action_input])
    )
    output = critic.apply(params, jnp.hstack([sample_input, action_input]))
    assert output.shape == (4, 1)  # Ensure output shape matches expected dimensions


@pytest.fixture
def multi_critic_architecture():
    return ["16", "relu"]


@pytest.mark.parametrize("num_critics", [1, 3])  # Test with num=1 and num=3
def test_multi_critic(sample_input, multi_critic_architecture, num_critics):
    multi_critic = MultiCritic(
        input_architecture=multi_critic_architecture,
        num=num_critics,
    )
    action_input = jnp.ones(
        (4, 2)
    )  # Example action input with batch size 4 and action size 2
    params = multi_critic.init(
        jax.random.PRNGKey(0), jnp.hstack([sample_input, action_input])
    )
    output = multi_critic.apply(params, jnp.hstack([sample_input, action_input]))
    assert output.shape == (
        num_critics,
        4,
        1,
    )  # Ensure output shape matches (num, batch_size, 1)


@pytest.fixture
def real_env_config():
    env_name = "CartPole-v1"
    env, env_params = build_env_from_id(env_name)
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture
def mock_env_config():
    class MockEnv:
        def __init__(self, action_space_shape, action_space_n=None):
            self.action_space_shape = action_space_shape
            self.action_space_n = action_space_n

        def action_space(self, params):
            if self.action_space_n:
                return type("Discrete", (), {"n": self.action_space_n})
            return type("Box", (), {"shape": self.action_space_shape})

    mock_env = MockEnv(action_space_shape=(4,), action_space_n=None)
    mock_env_params = EnvParams()  # Example empty EnvParams

    return EnvironmentConfig(
        env=mock_env,
        env_params=mock_env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture
def fast_env_config():
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(
        env=env,
        env_params=None,
        n_envs=1,
        continuous=True,
    )


def test_get_initialized_actor_critic(
    real_env_config, actor_architecture, critic_architecture
):
    key = jax.random.PRNGKey(0)
    optimizer_config = OptimizerConfig(
        **{"learning_rate": 0.001}
    )  # Example optimizer config
    network_config = NetworkConfig(
        **{
            "actor_architecture": actor_architecture,
            "critic_architecture": critic_architecture,
            "lstm_hidden_size": None,  # Example non-recurrent configuration
        }
    )
    num_critics = 2
    actor_state, critic_state = get_initialized_actor_critic(
        key=key,
        env_config=real_env_config,
        actor_optimizer_config=optimizer_config,
        critic_optimizer_config=optimizer_config,
        network_config=network_config,
        continuous=False,
        action_value=True,
        num_critics=num_critics,
    )

    # Ensure actor and critic states are correctly initialized
    assert isinstance(actor_state, LoadedTrainState)
    assert isinstance(critic_state, LoadedTrainState)

    observation_shape, action_shape = get_state_action_shapes(real_env_config.env)
    init_obs = jnp.zeros((real_env_config.n_envs, *observation_shape))
    init_action = jnp.zeros((real_env_config.n_envs, *action_shape))

    # Validate actor state
    actor_output = actor_state.apply_fn(actor_state.params, init_obs)

    assert actor_output.mode().shape == (
        real_env_config.n_envs,
    )  # Ensure mean shape matches action_dim

    # Validate critic state
    critic_output = critic_state.apply_fn(
        critic_state.params, jnp.hstack([init_obs, init_action])
    )
    assert critic_output.shape == (
        num_critics,
        real_env_config.n_envs,
        1,
    )  # Ensure output shape matches (num, batch_size, 1)


def test_get_initialized_actor_critic_continuous(
    fast_env_config, actor_architecture, critic_architecture
):
    key = jax.random.PRNGKey(1)
    optimizer_config = OptimizerConfig(
        **{"learning_rate": 0.001}
    )  # Example optimizer config
    network_config = NetworkConfig(
        **{
            "actor_architecture": actor_architecture,
            "critic_architecture": critic_architecture,
            "lstm_hidden_size": None,  # Example non-recurrent configuration
        }
    )
    num_critics = 2
    actor_state, critic_state = get_initialized_actor_critic(
        key=key,
        env_config=fast_env_config,
        actor_optimizer_config=optimizer_config,
        critic_optimizer_config=optimizer_config,
        network_config=network_config,
        continuous=True,
        action_value=False,
        num_critics=num_critics,
    )

    # Ensure actor and critic states are correctly initialized
    assert isinstance(actor_state, LoadedTrainState)
    assert isinstance(critic_state, LoadedTrainState)

    observation_shape, action_shape = get_state_action_shapes(fast_env_config.env)
    init_obs = jnp.zeros((fast_env_config.n_envs, *observation_shape))

    # Validate actor state
    actor_output = actor_state.apply_fn(actor_state.params, init_obs)

    assert actor_output.mean().shape == (
        fast_env_config.n_envs,
        *action_shape,
    )  # Ensure mean shape matches action_dim
    assert actor_output.stddev().shape == (
        fast_env_config.n_envs,
        *action_shape,
    )  # Ensure stddev shape matches action_dim

    # Validate critic state
    critic_output = critic_state.apply_fn(critic_state.params, init_obs)

    assert critic_output.shape == (
        num_critics,
        fast_env_config.n_envs,
        1,
    )  # Ensure output shape matches (num, batch_size, 1)


def test_predict_value(real_env_config, actor_architecture, critic_architecture):
    key = jax.random.PRNGKey(0)
    optimizer_config = OptimizerConfig(learning_rate=0.001)
    network_config = NetworkConfig(
        actor_architecture=actor_architecture,
        critic_architecture=critic_architecture,
        lstm_hidden_size=None,
    )
    num_critics = 2
    actor_state, critic_state = get_initialized_actor_critic(
        key=key,
        env_config=real_env_config,
        actor_optimizer_config=optimizer_config,
        critic_optimizer_config=optimizer_config,
        network_config=network_config,
        continuous=False,
        action_value=True,
        num_critics=num_critics,
    )

    observation_shape, action_shape = get_state_action_shapes(real_env_config.env)
    init_obs = jnp.ones((real_env_config.n_envs, *observation_shape))
    init_action = jnp.ones((real_env_config.n_envs, *action_shape))
    input_data = jnp.hstack([init_obs, init_action])

    # Predict value using the critic
    predicted_value = predict_value(critic_state, critic_state.params, input_data)

    assert predicted_value.shape == (
        num_critics,
        real_env_config.n_envs,
        1,
    ), "Shape mismatch in predicted value."
    assert jnp.all(
        jnp.isfinite(predicted_value)
    ), "Predicted value contains invalid values."
    assert not jnp.allclose(
        predicted_value[0], predicted_value[1]
    ), "Predicted values for both critics are identical."


def test_encoder_penultimate_normalization():
    input_architecture = ["4", "relu", "2"]
    encoder = Encoder(
        input_architecture=input_architecture, penultimate_normalization=True
    )

    key = jax.random.PRNGKey(0)
    input_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    params = encoder.init(key, input_data)
    output = encoder.apply(params, input_data)

    # Check that the output is normalized to a unit vector
    norms = jnp.linalg.norm(output, axis=1)
    assert jnp.allclose(norms, 1.0), "Output is not normalized to a unit vector"


def test_encoder_penultimate_normalization_gradients():
    input_architecture = ["4", "relu", "2"]
    encoder = Encoder(
        input_architecture=input_architecture, penultimate_normalization=True
    )

    key = jax.random.PRNGKey(0)
    input_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    params = encoder.init(key, input_data)

    def loss_fn(params, input_data):
        output = encoder.apply(params, input_data)
        return jnp.sum(output)  # Example loss function

    grads = jax.grad(loss_fn)(params, input_data)

    # Check that gradients exist for the parameters of the encoder
    assert grads is not None, "Gradients are not flowing through the encoder."
    assert "params" in grads, "Gradients for parameters are missing."
    assert all(
        jnp.any(jnp.abs(g) > 0) for g in jax.tree_util.tree_leaves(grads["params"])
    ), "Some gradients are zero, indicating no flow through _l2_normalize."
