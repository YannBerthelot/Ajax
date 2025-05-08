import gymnax
import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.AVG.state import AVGConfig, AVGState
from ajax.agents.AVG.train_AVG import (
    alpha_loss_function,
    create_alpha_train_state,
    init_AVG,
    make_train,
    policy_loss_function,
    training_iteration,
    update_agent,
    update_AVG_values,
    update_policy,
    update_temperature,
    update_value_functions,
    value_loss_function,
)
from ajax.environments.interaction import Transition, get_pi
from ajax.environments.utils import get_state_action_shapes
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)


@pytest.fixture
def fast_env_config():
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(
        env=env,
        env_params=None,
        num_envs=1,
        continuous=True,
    )


@pytest.fixture
def gymnax_env_config():
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        num_envs=1,
        continuous=True,
    )


@pytest.fixture(params=["fast_env_config", "gymnax_env_config"])
def env_config(request, fast_env_config, gymnax_env_config):
    return fast_env_config if request.param == "fast_env_config" else gymnax_env_config


@pytest.fixture
def avg_state(env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
        penultimate_normalization=True,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)

    avg_state = init_AVG(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
    )
    obs_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )
    transition = Transition(
        obs=jnp.ones((env_config.num_envs, *obs_shape)),
        action=jnp.ones((env_config.num_envs, *action_shape)),
        next_obs=jnp.ones((env_config.num_envs, *obs_shape)),
        reward=jnp.ones((env_config.num_envs, 1)),
        terminated=jnp.ones((env_config.num_envs, 1)),
        truncated=jnp.ones((env_config.num_envs, 1)),
    )
    collector_state = avg_state.collector_state.replace(rollout=transition)
    avg_state = avg_state.replace(collector_state=collector_state)
    return avg_state


def test_create_alpha_train_state():
    learning_rate = 3e-4
    alpha_init = 1.0

    train_state = create_alpha_train_state(
        learning_rate=learning_rate, alpha_init=alpha_init
    )

    assert isinstance(train_state, TrainState), "Returned object is not a TrainState."
    assert jnp.isclose(
        train_state.params["log_alpha"], jnp.log(alpha_init)
    ), "log_alpha initialization is incorrect."
    assert train_state.tx is not None, "Optimizer transaction is not initialized."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_init_AVG(avg_state):
    assert isinstance(avg_state, AVGState), "Returned object is not an AVGState."
    assert avg_state.actor_state is not None, "Actor state is not initialized."
    assert avg_state.critic_state is not None, "Critic state is not initialized."
    assert isinstance(
        avg_state.alpha.params, FrozenDict
    ), "Alpha state is not initialized correctly."
    assert avg_state.collector_state is not None, "Collector state is not initialized."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_value_loss_function(env_config, avg_state):
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the value loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.num_envs, *observation_shape))
    next_observations = jnp.zeros((env_config.num_envs, *observation_shape))
    actions = jnp.zeros((env_config.num_envs, *action_shape))
    rewards = jnp.ones((env_config.num_envs, 1))
    dones = jnp.zeros((env_config.num_envs, 1))
    gamma = 0.99
    alpha = jnp.array(0.1)

    # Call the value loss function
    loss, aux = value_loss_function(
        critic_params=avg_state.critic_state.params,
        critic_states=avg_state.critic_state,
        rng=rng,
        actor_state=avg_state.actor_state,
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
def test_value_loss_function_with_value_and_grad(env_config, avg_state):
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the value loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.num_envs, *observation_shape))
    next_observations = jnp.zeros((env_config.num_envs, *observation_shape))
    actions = jnp.zeros((env_config.num_envs, *action_shape))
    rewards = jnp.ones((env_config.num_envs, 1))
    dones = jnp.zeros((env_config.num_envs, 1))
    gamma = 0.99
    alpha = jnp.array(0.1)

    # Define a wrapper for value_loss_function
    def loss_fn(critic_params):
        loss, _ = value_loss_function(
            critic_params=critic_params,
            critic_states=avg_state.critic_state,
            rng=rng,
            actor_state=avg_state.actor_state,
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
    loss, grads = jax.value_and_grad(loss_fn)(avg_state.critic_state.params)

    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
    assert all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    ), "Gradients contain invalid values."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_policy_loss_function(env_config, avg_state):
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the policy loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.ones((env_config.num_envs, *observation_shape))
    # actions = jnp.ones((env_config.num_envs, *action_shape))
    dones = jnp.zeros((env_config.num_envs, 1))
    alpha = jnp.array(0.1)
    pi, _ = get_pi(
        actor_state=avg_state.actor_state,
        actor_params=avg_state.actor_state.params,
        obs=observations,
        done=dones,
        recurrent=False,
    )
    actions = pi.sample(seed=rng)

    # Call the policy loss function
    loss, aux = policy_loss_function(
        actor_params=avg_state.actor_state.params,
        actor_state=avg_state.actor_state,
        critic_states=avg_state.critic_state,
        actions=actions,
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
    assert "q_min" in aux, "Auxiliary outputs are missing 'q_min'."
    assert aux["policy_loss"] <= 0, "Policy loss should be negative."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_policy_loss_function_with_value_and_grad(env_config, avg_state):
    observation_shape, _ = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the policy loss function
    rng = jax.random.PRNGKey(1)
    observations = jnp.zeros((env_config.num_envs, *observation_shape))
    dones = jnp.zeros((env_config.num_envs, 1))
    alpha = jnp.array(0.1)
    pi, _ = get_pi(
        actor_state=avg_state.actor_state,
        actor_params=avg_state.actor_state.params,
        obs=observations,
        done=dones,
        recurrent=False,
    )
    actions = pi.sample(seed=rng)

    # Define a wrapper for policy_loss_function
    def loss_fn(actor_params):
        loss, _ = policy_loss_function(
            actor_params=actor_params,
            actor_state=avg_state.actor_state,
            critic_states=avg_state.critic_state,
            actions=actions,
            observations=observations,
            dones=dones,
            recurrent=False,
            alpha=alpha,
            rng=rng,
        )
        return loss

    # Compute gradients using jax.value_and_grad
    loss, grads = jax.value_and_grad(loss_fn)(avg_state.actor_state.params)

    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
    assert all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    ), "Gradients contain invalid values."


@pytest.mark.parametrize(
    "log_alpha_init, target_entropy, corrected_log_probs",
    [
        (0.0, -1.0, jnp.array([-0.5, -1.5, -1.0])),
        (-1.0, -2.0, jnp.array([-1.0, -2.0, -1.5])),
    ],
)
def test_alpha_loss_function(log_alpha_init, target_entropy, corrected_log_probs):
    log_alpha_params = FrozenDict({"log_alpha": jnp.array(log_alpha_init)})

    # Call the alpha loss function
    loss, aux = alpha_loss_function(
        log_alpha_params=log_alpha_params,
        corrected_log_probs=corrected_log_probs,
        target_entropy=target_entropy,
    )
    aux = to_state_dict(aux)
    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert "alpha_loss" in aux, "Auxiliary outputs are missing 'alpha_loss'."
    assert "alpha" in aux, "Auxiliary outputs are missing 'alpha'."
    assert "log_alpha" in aux, "Auxiliary outputs are missing 'log_alpha'."
    assert aux["alpha"] > 0, "Alpha should be positive."


@pytest.mark.parametrize(
    "log_alpha_init, target_entropy, corrected_log_probs",
    [
        (0.0, -1.0, jnp.array([-0.5, -1.5, -1.0])),
        (-1.0, -2.0, jnp.array([-1.0, -2.0, -1.5])),
    ],
)
def test_alpha_loss_function_with_value_and_grad(
    log_alpha_init, target_entropy, corrected_log_probs
):
    log_alpha_params = FrozenDict({"log_alpha": jnp.array(log_alpha_init)})

    # Define a wrapper for alpha_loss_function
    def loss_fn(log_alpha_params):
        loss, _ = alpha_loss_function(
            log_alpha_params=log_alpha_params,
            corrected_log_probs=corrected_log_probs,
            target_entropy=target_entropy,
        )
        return loss

    # Compute gradients using jax.value_and_grad
    loss, grads = jax.value_and_grad(loss_fn)(log_alpha_params)

    # Validate the outputs
    assert jnp.isfinite(loss), "Loss contains invalid values."
    assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
    assert all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    ), "Gradients contain invalid values."


def compare_frozen_dicts(dict1: FrozenDict, dict2: FrozenDict) -> bool:
    """
    Compares two FrozenDicts to check if they are equal.

    Args:
        dict1 (FrozenDict): The first FrozenDict.
        dict2 (FrozenDict): The second FrozenDict.

    Returns:
        bool: True if the FrozenDicts are equal, False otherwise.
    """
    for key in dict1.keys():
        if key not in dict2:
            return False
        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, FrozenDict) and isinstance(value2, FrozenDict):
            if not compare_frozen_dicts(value1, value2):
                return False
        elif not jnp.allclose(value1, value2):
            return False
    return True


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_value_functions(env_config, avg_state):
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the update_value_functions function
    observations = jnp.zeros((env_config.num_envs, *observation_shape))
    next_observations = jnp.zeros((env_config.num_envs, *observation_shape))
    actions = jnp.zeros((env_config.num_envs, *action_shape))
    rewards = jnp.ones((env_config.num_envs, 1))
    dones = jnp.zeros((env_config.num_envs, 1))
    gamma = 0.99
    reward_scale = 1.0

    # Save the original target_params for comparison
    original_target_params = avg_state.critic_state.target_params

    # Call the update_value_functions function
    updated_state, aux = update_value_functions(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=dones,
        agent_state=avg_state,
        recurrent=False,
        rewards=rewards,
        gamma=gamma,
        reward_scale=reward_scale,
    )
    aux = to_state_dict(aux)
    # Validate that only critic_state.params has changed
    assert not compare_frozen_dicts(
        updated_state.critic_state.params, avg_state.critic_state.params
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
def test_update_policy(env_config, avg_state):
    observation_shape, _ = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the update_policy function
    observations = jnp.zeros((env_config.num_envs, *observation_shape))
    dones = jnp.zeros((env_config.num_envs, 1))
    pi, _ = get_pi(
        actor_state=avg_state.actor_state,
        actor_params=avg_state.actor_state.params,
        obs=observations,
        done=dones,
        recurrent=False,
    )
    actions = pi.sample(seed=jax.random.PRNGKey(0))

    # Save the original actor params for comparison
    original_actor_params = avg_state.actor_state.params

    # Call the update_policy function
    updated_state, aux = update_policy(
        actions=actions,
        observations=observations,
        done=dones,
        agent_state=avg_state,
        recurrent=False,
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
def test_update_temperature(env_config, avg_state):
    observation_shape, _ = get_state_action_shapes(
        env_config.env, env_config.env_params
    )

    # Mock inputs for the update_temperature function
    observations = jnp.zeros((env_config.num_envs, *observation_shape))
    dones = jnp.zeros((env_config.num_envs, 1))
    target_entropy = -1.0

    # Save the original alpha params for comparison
    original_alpha_params = avg_state.alpha.params

    # Call the update_temperature function
    updated_state, aux = update_temperature(
        agent_state=avg_state,
        observations=observations,
        dones=dones,
        target_entropy=target_entropy,
        recurrent=False,
    )
    aux = to_state_dict(aux)
    # Validate that only alpha.params has changed
    assert not compare_frozen_dicts(
        updated_state.alpha.params, original_alpha_params
    ), "alpha.params should have been updated."

    # Validate auxiliary outputs
    assert "alpha_loss" in aux, "Auxiliary outputs are missing 'alpha_loss'."
    assert "alpha" in aux, "Auxiliary outputs are missing 'alpha'."
    assert aux["alpha"] > 0, "Alpha should be positive."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_agent(env_config, avg_state):
    # Mock inputs for the update_agent function
    gamma = 0.99
    action_dim = 1  # Example action dimension
    recurrent = False

    # Call the update_agent function
    updated_state, _ = update_agent(
        agent_state=avg_state,
        _=None,
        recurrent=recurrent,
        gamma=gamma,
        action_dim=action_dim,
    )

    # Validate that the state has been updated
    assert updated_state is not None, "Updated state should not be None."
    assert updated_state.rng is not None, "Updated RNG should not be None."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_agent_with_scan(env_config, avg_state):
    # Mock inputs for the update_agent function
    gamma = 0.99
    action_dim = 1  # Example action dimension
    recurrent = False

    update_agent_scan = partial(
        update_agent,
        recurrent=recurrent,
        gamma=gamma,
        action_dim=action_dim,
    )

    # Run the scan
    final_state, _ = jax.lax.scan(update_agent_scan, avg_state, None, length=5)

    # Validate the final state
    assert final_state is not None, "Final state should not be None."
    assert final_state.rng is not None, "Final RNG should not be None."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_training_iteration_with_scan(env_config, avg_state):
    gamma = 0.99
    action_dim = 1
    recurrent = False
    agent_args = AVGConfig(gamma=gamma, target_entropy=-1.0, learning_starts=5)
    log_frequency = 10

    # Define a partial function for training_iteration
    training_iteration_scan = partial(
        training_iteration,
        env_args=env_config,
        mode="gymnax" if env_config.env_params else "brax",
        recurrent=recurrent,
        agent_args=agent_args,
        action_dim=action_dim,
        log_frequency=log_frequency,
        total_timesteps=5,
    )

    # Run multiple training iterations using jax.lax.scan
    final_state, _ = jax.lax.scan(training_iteration_scan, avg_state, None, length=5)

    # Validate the final state
    assert isinstance(final_state, AVGState), "Final state should be of type AVGState."
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
    agent_args = AVGConfig(gamma=0.99, target_entropy=-1.0)
    total_timesteps = 1000
    # Create the train function
    train_fn = make_train(
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        agent_args=agent_args,
        total_timesteps=total_timesteps,
        alpha_args=alpha_args,
        num_episode_test=2,
    )

    # Run the train function
    final_state = train_fn(key)

    # Validate the final state
    assert isinstance(final_state, AVGState), "Final state should be of type AVGState."
    assert final_state.rng is not None, "Final RNG should not be None."
    assert final_state.actor_state is not None, "Actor state should not be None."
    assert final_state.critic_state is not None, "Critic state should not be None."
    assert (
        final_state.collector_state is not None
    ), "Collector state should not be None."
    assert final_state.alpha is not None, "Alpha state should not be None."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_AVG_values(env_config, avg_state):
    # Mock inputs for the update_AVG_values function
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )
    rollout = Transition(
        obs=jnp.ones((env_config.num_envs, *observation_shape)),
        action=jnp.ones((env_config.num_envs, *action_shape)),
        next_obs=jnp.ones((env_config.num_envs, *observation_shape)),
        reward=jnp.array([[1.0]]),
        terminated=jnp.array([[0.0]]),
        truncated=jnp.array([[0.0]]),
    )
    agent_args = AVGConfig(gamma=0.99, target_entropy=-1.0)

    # Call the update_AVG_values function
    updated_state = update_AVG_values(avg_state, rollout, agent_args)

    # Validate the updated state
    assert updated_state is not None, "Updated state should not be None."
    assert (
        updated_state.reward.count[0] > avg_state.reward.count[0]
    ), "Reward count should have been incremented."
    assert (
        updated_state.gamma.count[0] > avg_state.gamma.count[0]
    ), "Gamma count should have been incremented."
    assert not (
        updated_state.G_return.count[0] > avg_state.G_return.count[0]
    ), "G_return count should not have been incremented."
    assert jnp.allclose(
        updated_state.reward.mean, jnp.array([[1.0]])
    ), "Reward mean should match the rollout reward."
    assert jnp.allclose(
        updated_state.gamma.mean, jnp.array([[0.99]])
    ), "Gamma mean should match the agent_args gamma."
    assert jnp.allclose(
        updated_state.G_return.value, jnp.array([[1.0]])
    ), "G_return value should match the cumulative reward."

    assert jnp.allclose(
        updated_state.G_return.mean, jnp.array([[0.0]])
    ), "G_return mean should not match the cumulative reward."


@pytest.mark.parametrize(
    "env_config", ["fast_env_config", "gymnax_env_config"], indirect=True
)
def test_update_AVG_values_terminal(env_config, avg_state):
    # Mock inputs for the update_AVG_values function
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env, env_config.env_params
    )
    rollout = Transition(
        obs=jnp.ones((env_config.num_envs, *observation_shape)),
        action=jnp.ones((env_config.num_envs, *action_shape)),
        next_obs=jnp.ones((env_config.num_envs, *observation_shape)),
        reward=jnp.array([[1.0]]),
        terminated=jnp.array([[1.0]]),
        truncated=jnp.array([[0.0]]),
    )
    agent_args = AVGConfig(gamma=0.99, target_entropy=-1.0)

    # Call the update_AVG_values function
    updated_state = update_AVG_values(avg_state, rollout, agent_args)

    # Validate the updated state
    assert updated_state is not None, "Updated state should not be None."
    assert (
        updated_state.reward.count[0] > avg_state.reward.count[0]
    ), "Reward count should have been incremented."
    assert (
        updated_state.gamma.count[0] > avg_state.gamma.count[0]
    ), "Gamma count should have been incremented."
    assert (
        updated_state.G_return.count[0] > avg_state.G_return.count[0]
    ), "G_return count should have been incremented."
    assert jnp.allclose(
        updated_state.reward.mean, jnp.array([[1.0]])
    ), "Reward mean should match the rollout reward."
    assert jnp.allclose(
        updated_state.gamma.mean, jnp.array([[0.0]])
    ), "Gamma mean should match the agent_args gamma."
    assert jnp.allclose(
        updated_state.G_return.mean, jnp.array([[1.0]])
    ), "G_return mean should match the cumulative reward."
