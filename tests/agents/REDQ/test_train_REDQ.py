import gymnax
import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.REDQ.state import REDQConfig, REDQState
from ajax.agents.REDQ.train_REDQ import (
    init_REDQ,
    make_train,
    policy_loss_function,
    training_iteration,
    update_agent,
    update_policy,
    update_value_functions,
    value_loss_function,
)
from ajax.buffers.utils import get_buffer, init_buffer
from ajax.environments.utils import get_state_action_shapes
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
def REDQ_state(env_config, buffer):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    number_of_critics = 5

    return init_REDQ(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=buffer,
        number_of_critics=number_of_critics,
    )


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_init_REDQ(REDQ_state):
    assert isinstance(REDQ_state, REDQState), "Returned object is not an REDQState."
    assert REDQ_state.actor_state is not None, "Actor state is not initialized."
    assert REDQ_state.critic_state is not None, "Critic state is not initialized."
    assert isinstance(
        REDQ_state.alpha.params, FrozenDict
    ), "Alpha state is not initialized correctly."
    assert REDQ_state.collector_state is not None, "Collector state is not initialized."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_value_loss_function(env_config, REDQ_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    # Mock inputs for the value loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    next_observations = jnp.zeros((env_config.n_envs, *observation_shape))
    actions = jnp.zeros((env_config.n_envs, *action_shape))
    rewards = jnp.ones((env_config.n_envs, 1))
    dones = jnp.zeros((env_config.n_envs, 1))
    gamma = 0.99
    alpha = jnp.array(0.1)

    # Call the value loss function
    loss, aux = value_loss_function(
        critic_params=REDQ_state.critic_state.params,
        critic_states=REDQ_state.critic_state,
        rng=rng,
        actor_state=REDQ_state.actor_state,
        actions=actions,
        observations=observations,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        alpha=alpha,
        recurrent=False,
        reward_scale=1.0,
    )
    aux = to_state_dict(aux)
    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert "critic_loss" in aux, "Auxiliary outputs are missing 'critic_loss'."
    assert aux["critic_loss"] >= 0, "Critic loss should be non-negative."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_value_loss_function_with_value_and_grad(env_config, REDQ_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    # Mock inputs for the value loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    next_observations = jnp.zeros((env_config.n_envs, *observation_shape))
    actions = jnp.zeros((env_config.n_envs, *action_shape))
    rewards = jnp.ones((env_config.n_envs, 1))
    dones = jnp.zeros((env_config.n_envs, 1))
    gamma = 0.99
    alpha = jnp.array(0.1)

    # Define a wrapper for value_loss_function
    def loss_fn(critic_params):
        loss, _ = value_loss_function(
            critic_params=critic_params,
            critic_states=REDQ_state.critic_state,
            rng=rng,
            actor_state=REDQ_state.actor_state,
            actions=actions,
            observations=observations,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            gamma=gamma,
            alpha=alpha,
            recurrent=False,
            reward_scale=1.0,
        )
        return loss

    # Compute gradients using jax.value_and_grad
    loss, grads = jax.value_and_grad(loss_fn)(REDQ_state.critic_state.params)

    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
    assert all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    ), "Gradients contain invalid values."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_policy_loss_function(env_config, REDQ_state):
    observation_shape, _ = get_state_action_shapes(env_config.env)

    # Mock inputs for the policy loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    dones = jnp.zeros((env_config.n_envs, 1))
    alpha = jnp.array(0.1)

    # Call the policy loss function
    loss, aux = policy_loss_function(
        actor_params=REDQ_state.actor_state.params,
        actor_state=REDQ_state.actor_state,
        critic_states=REDQ_state.critic_state,
        observations=observations,
        dones=dones,
        recurrent=False,
        alpha=alpha,
        rng=rng,
    )
    aux = to_state_dict(aux)
    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert "policy_loss" in aux, "Auxiliary outputs are missing 'policy_loss'."
    assert "log_pi" in aux, "Auxiliary outputs are missing 'log_pi'."
    assert "q_mean" in aux, "Auxiliary outputs are missing 'q_mean'."
    assert aux["policy_loss"] <= 0, "Policy loss should be negative."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_policy_loss_function_with_value_and_grad(env_config, REDQ_state):
    observation_shape, _ = get_state_action_shapes(env_config.env)

    # Mock inputs for the policy loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    dones = jnp.zeros((env_config.n_envs, 1))
    alpha = jnp.array(0.1)

    # Define a wrapper for policy_loss_function
    def loss_fn(actor_params):
        loss, _ = policy_loss_function(
            actor_params=actor_params,
            actor_state=REDQ_state.actor_state,
            critic_states=REDQ_state.critic_state,
            observations=observations,
            dones=dones,
            recurrent=False,
            alpha=alpha,
            rng=rng,
        )
        return loss

    # Compute gradients using jax.value_and_grad
    loss, grads = jax.value_and_grad(loss_fn)(REDQ_state.actor_state.params)

    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
    assert all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    ), "Gradients contain invalid values."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_value_functions(env_config, REDQ_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    # Mock inputs for the update_value_functions function
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    next_observations = jnp.zeros((env_config.n_envs, *observation_shape))
    actions = jnp.zeros((env_config.n_envs, *action_shape))
    rewards = jnp.ones((env_config.n_envs, 1))
    dones = jnp.zeros((env_config.n_envs, 1))
    gamma = 0.99
    reward_scale = 1.0
    subset_size = 2

    # Save the original target_params for comparison
    original_target_params = REDQ_state.critic_state.target_params

    # Call the update_value_functions function
    updated_state, aux = update_value_functions(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=dones,
        agent_state=REDQ_state,
        recurrent=False,
        rewards=rewards,
        gamma=gamma,
        reward_scale=reward_scale,
        subset_size=subset_size,
    )
    aux = to_state_dict(aux)
    # Validate that only critic_state.params has changed
    assert not compare_frozen_dicts(
        updated_state.critic_state.params, REDQ_state.critic_state.params
    ), "critic_state.params should have been updated."

    assert compare_frozen_dicts(
        updated_state.critic_state.target_params, original_target_params
    ), "critic_state.target_params should not have changed."
    # Validate auxiliary outputs
    assert "critic_loss" in aux, "Auxiliary outputs are missing 'critic_loss'."
    assert aux["critic_loss"] >= 0, "Critic loss should be non-negative."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_policy(env_config, REDQ_state):
    observation_shape, _ = get_state_action_shapes(env_config.env)

    # Mock inputs for the update_policy function
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    dones = jnp.zeros((env_config.n_envs, 1))

    # Save the original actor params for comparison
    original_actor_params = REDQ_state.actor_state.params

    # Call the update_policy function
    updated_state, aux = update_policy(
        observations=observations,
        done=dones,
        agent_state=REDQ_state,
        recurrent=False,
        raw_observations=observation_shape,
    )
    aux = to_state_dict(aux)
    # Validate that only actor_state.params has changed
    assert not compare_frozen_dicts(
        updated_state.actor_state.params, original_actor_params
    ), "actor_state.params should have been updated."

    # Validate auxiliary outputs
    assert "policy_loss" in aux, "Auxiliary outputs are missing 'policy_loss'."
    assert aux["policy_loss"] <= 0, "Policy loss should be negative."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_agent(env_config, REDQ_state):
    # Mock inputs for the update_agent function
    buffer = get_buffer(buffer_size=100, batch_size=32, n_envs=env_config.n_envs)
    gamma = 0.99
    tau = 0.005
    action_dim = 1  # Example action dimension
    recurrent = False

    # Initialize buffer state
    buffer_state = init_buffer(buffer, env_config)
    REDQ_state = REDQ_state.replace(
        collector_state=REDQ_state.collector_state.replace(buffer_state=buffer_state)
    )

    # Call the update_agent function
    updated_state, _ = update_agent(
        agent_state=REDQ_state,
        _=None,
        buffer=buffer,
        recurrent=recurrent,
        gamma=gamma,
        action_dim=action_dim,
        tau=tau,
    )

    # Validate that the state has been updated
    assert updated_state is not None, "Updated state should not be None."
    assert updated_state.rng is not None, "Updated RNG should not be None."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_agent_with_scan(env_config, REDQ_state):
    # Mock inputs for the update_agent function
    buffer = get_buffer(buffer_size=100, batch_size=32, n_envs=env_config.n_envs)
    gamma = 0.99
    tau = 0.005
    action_dim = 1  # Example action dimension
    recurrent = False

    # Initialize buffer state
    buffer_state = init_buffer(buffer, env_config)
    REDQ_state = REDQ_state.replace(
        collector_state=REDQ_state.collector_state.replace(buffer_state=buffer_state)
    )
    num_critic_updates = 2

    update_agent_scan = partial(
        update_agent,
        buffer=buffer,
        recurrent=recurrent,
        gamma=gamma,
        action_dim=action_dim,
        tau=tau,
        num_critic_updates=num_critic_updates,
    )

    # Run the scan
    final_state, _ = jax.lax.scan(update_agent_scan, REDQ_state, None, length=5)

    # Validate the final state
    assert final_state is not None, "Final state should not be None."
    assert final_state.rng is not None, "Final RNG should not be None."


def tree_equal(a, b):
    return jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), a, b)
    )


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_training_iteration_with_scan(env_config, REDQ_state):
    buffer = get_buffer(buffer_size=100, batch_size=32, n_envs=env_config.n_envs)
    gamma = 0.99
    tau = 0.005
    action_dim = 1
    recurrent = False
    agent_config = REDQConfig(
        gamma=gamma,
        tau=tau,
        target_entropy=-1.0,
        learning_starts=5,
        num_critics=3,
        subset_size=2,
        num_critic_updates=2,
    )
    log_frequency = 10

    # Initialize buffer state
    buffer_state = init_buffer(buffer, env_config)
    REDQ_state = REDQ_state.replace(
        collector_state=REDQ_state.collector_state.replace(buffer_state=buffer_state)
    )

    # Define a partial function for training_iteration
    training_iteration_scan = partial(
        training_iteration,
        env_args=env_config,
        mode="gymnax" if env_config.env_params else "brax",
        recurrent=recurrent,
        buffer=buffer,
        agent_config=agent_config,
        action_dim=action_dim,
        log_frequency=log_frequency,
        total_timesteps=5,
    )

    # Run multiple training iterations using jax.lax.scan
    final_state, _ = jax.lax.scan(training_iteration_scan, REDQ_state, None, length=5)

    timesteps = (
        final_state.collector_state.buffer_state.experience["action"].sum(-1) != 0
    ).sum()
    assert timesteps == 5, "Buffer step count should be 5."

    # Check that the buffer state has been updated
    assert not (tree_equal(final_state.collector_state.buffer_state, buffer_state))

    # Validate the final state
    assert isinstance(
        final_state, REDQState
    ), "Final state should be of type REDQState."
    assert final_state.rng is not None, "Final RNG should not be None."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_make_train(env_config):
    """Test the make_train function."""
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    buffer = get_buffer(
        **to_state_dict(
            BufferConfig(buffer_size=1000, batch_size=32, n_envs=env_config.n_envs)
        )
    )
    gamma = 0.99
    tau = 0.005
    agent_config = REDQConfig(
        gamma=gamma,
        tau=tau,
        target_entropy=-1.0,
        learning_starts=5,
        num_critics=3,
        subset_size=2,
        num_critic_updates=2,
    )
    total_timesteps = 1000

    # Create the train function
    train_fn = make_train(
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        buffer=buffer,
        agent_config=agent_config,
        total_timesteps=total_timesteps,
        alpha_args=alpha_args,
        num_episode_test=2,
    )

    # Run the train function
    final_state, _ = train_fn(key)

    # Validate the final state
    assert isinstance(
        final_state, REDQState
    ), "Final state should be of type REDQState."
    assert final_state.rng is not None, "Final RNG should not be None."
    assert final_state.actor_state is not None, "Actor state should not be None."
    assert final_state.critic_state is not None, "Critic state should not be None."
    assert (
        final_state.collector_state is not None
    ), "Collector state should not be None."
    assert final_state.alpha is not None, "Alpha state should not be None."
