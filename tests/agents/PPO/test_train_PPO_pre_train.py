import gymnax
import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.cloning import (
    CloningConfig,
)
from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.train_PPO_pre_train import (
    init_PPO,
    make_train,
    policy_loss_function,
    training_iteration,
    update_agent,
    update_policy,
    update_value_functions,
)
from ajax.agents.PPO.utils import get_minibatches_from_batch
from ajax.environments.utils import get_state_action_shapes
from ajax.state import (
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.utils import compare_frozen_dicts


@pytest.fixture
def ant_env_config():
    env = create_brax_env("ant", batch_size=1)
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


@pytest.fixture
def discrete_gymnax_env_config():
    env, env_params = gymnax.make("CartPole-v1")
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture(
    params=["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"]
)
def env_config(request, ant_env_config, gymnax_env_config, discrete_gymnax_env_config):
    config_dict = {
        "ant_env_config": ant_env_config,
        "gymnax_env_config": gymnax_env_config,
        "discrete_gymnax_env_config": discrete_gymnax_env_config,
    }
    return config_dict[request.param]


@pytest.fixture
def PPO_state(env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
    )

    return init_PPO(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
    )


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_init_PPO(PPO_state):
    assert isinstance(PPO_state, PPOState), "Returned object is not an SACState."
    assert PPO_state.actor_state is not None, "Actor state is not initialized."
    assert PPO_state.critic_state is not None, "Critic state is not initialized."
    assert PPO_state.collector_state is not None, "Collector state is not initialized."


# @pytest.mark.parametrize(
#     "env_config",
#     ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
#     indirect=True,
# )
# def test_value_loss_function(env_config, PPO_state):
#     observation_shape, action_shape = get_state_action_shapes(env_config.env)

#     # Mock inputs for the value loss function
#     observations = jnp.zeros((env_config.n_envs, *observation_shape))
#     rewards = jnp.ones((env_config.n_envs, 1))
#     dones = jnp.zeros((env_config.n_envs, 1))
#     gamma = 0.99

#     # Call the value loss function
#     loss, aux = value_loss_function(
#         critic_params=PPO_state.critic_state.params,
#         critic_states=PPO_state.critic_state,
#         observations=observations,
#         dones=dones,
#         recurrent=False,
#         value_targets=rewards + gamma * jnp.ones((env_config.n_envs, 1)),  # Mock target
#     )
#     aux = to_state_dict(aux)
#     # Validate the outputs
#     assert jnp.isfinite(loss), "Loss contains invalid values."
#     assert "critic_loss" in aux, "Auxiliary outputs are missing 'critic_loss'."
#     assert aux["critic_loss"] >= 0, "Critic loss should be non-negative."


# @pytest.mark.parametrize(
#     "env_config",
#     ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
#     indirect=True,
# )
# def test_value_loss_function_with_value_and_grad(env_config, PPO_state):
#     observation_shape, action_shape = get_state_action_shapes(env_config.env)

#     # Mock inputs for the value loss function
#     observations = jnp.zeros((env_config.n_envs, *observation_shape))
#     rewards = jnp.ones((env_config.n_envs, 1))
#     dones = jnp.zeros((env_config.n_envs, 1))
#     gamma = 0.99

#     # Define a wrapper for value_loss_function
#     def loss_fn(critic_params):
#         loss, _ = value_loss_function(
#             critic_params=critic_params,
#             critic_states=PPO_state.critic_state,
#             observations=observations,
#             dones=dones,
#             recurrent=False,
#             value_targets=rewards + gamma * jnp.ones((env_config.n_envs, 1)),
#         )
#         return loss

#     # Compute gradients using jax.value_and_grad
#     loss, grads = jax.value_and_grad(loss_fn)(PPO_state.critic_state.params)

#     # Validate the outputs
#     assert jnp.isfinite(loss), "Loss contains invalid values."
#     assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
#     assert all(
#         jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
#     ), "Gradients contain invalid values."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_policy_loss_function(env_config, PPO_state):
    observation_shape, _ = get_state_action_shapes(env_config.env)

    # Mock inputs for the policy loss function
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    dones = jnp.zeros((env_config.n_envs, 1))
    actions = jnp.zeros((env_config.n_envs, 1))
    log_probs = jnp.zeros((env_config.n_envs, 1))
    gae = jnp.ones((env_config.n_envs, 1))

    # Call the policy loss function
    for advantage_normalization in [True, False]:
        loss, aux = policy_loss_function(
            actor_params=PPO_state.actor_state.params,
            actor_state=PPO_state.actor_state,
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            gae=gae,
            dones=dones,
            recurrent=False,
            clip_coef=0.2,
            ent_coef=0.1,
            advantage_normalization=advantage_normalization,
        )
        aux = to_state_dict(aux)
        # Validate the outputs
        assert jnp.isfinite(loss), "Loss contains invalid values."

        for auxiliary_value in [
            "policy_loss",
            "log_probs",
            "old_log_probs",
            "clip_fraction",
            "entropy",
        ]:
            assert (
                auxiliary_value in aux
            ), f"Auxiliary outputs are missing '{auxiliary_value}'."

        assert aux["policy_loss"] <= 0, "Policy loss should be negative."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_policy_loss_function_with_value_and_grad(env_config, PPO_state):
    observation_shape, _ = get_state_action_shapes(env_config.env)

    # Mock inputs for the policy loss function
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    dones = jnp.zeros((env_config.n_envs, 1))
    actions = jnp.zeros((env_config.n_envs, 1))
    log_probs = jnp.zeros((env_config.n_envs, 1))
    gae = jnp.ones((env_config.n_envs, 1))

    # Define a wrapper for policy_loss_function
    for advantage_normalization in [True, False]:

        def loss_fn(actor_params):
            loss, _ = policy_loss_function(
                actor_params=PPO_state.actor_state.params,
                actor_state=PPO_state.actor_state,
                observations=observations,
                actions=actions,
                log_probs=log_probs,
                gae=gae,
                dones=dones,
                recurrent=False,
                clip_coef=0.2,
                ent_coef=0.1,
                advantage_normalization=advantage_normalization,  # noqa: B023
            )
            return loss

        # Compute gradients using jax.value_and_grad
        loss, grads = jax.value_and_grad(loss_fn)(PPO_state.actor_state.params)

        # Validate the outputs
        assert jnp.isfinite(loss), "Loss contains invalid values."
        assert isinstance(grads, FrozenDict), "Gradients are not a FrozenDict."
        assert all(
            jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
        ), "Gradients contain invalid values."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_update_value_functions(env_config, PPO_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    # Mock inputs for the update_value_functions function
    observations = jnp.zeros((env_config.n_envs, *observation_shape))
    rewards = jnp.ones((env_config.n_envs, 1))
    dones = jnp.zeros((env_config.n_envs, 1))
    gamma = 0.99
    value_targets = rewards + gamma * jnp.ones((env_config.n_envs, 1))

    # Call the update_value_functions function
    updated_state, aux = update_value_functions(
        observations=observations,
        value_targets=value_targets,
        dones=dones,
        agent_state=PPO_state,
        recurrent=False,
    )
    aux = to_state_dict(aux)
    # Validate that only critic_state.params has changed
    assert not compare_frozen_dicts(
        updated_state.critic_state.params, PPO_state.critic_state.params
    ), "critic_state.params should have been updated."

    # Validate auxiliary outputs
    assert "critic_loss" in aux, "Auxiliary outputs are missing 'critic_loss'."
    assert aux["critic_loss"] >= 0, "Critic loss should be non-negative."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_update_policy(env_config, PPO_state):
    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    # Mock inputs for the update_policy function
    observations = jnp.ones((env_config.n_envs, *observation_shape))
    dones = jnp.zeros((env_config.n_envs, 1))
    actions = jnp.ones((env_config.n_envs, *action_shape))
    log_probs = jnp.ones((env_config.n_envs, 1))
    gae = jnp.ones((env_config.n_envs, 1))

    # Save the original actor params for comparison
    original_actor_params = PPO_state.actor_state.params

    # Call the update_policy function
    updated_state, aux = update_policy(
        agent_state=PPO_state,
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        gae=gae,
        clip_coef=0.2,
        ent_coef=0.1,
        advantage_normalization=True,
        done=dones,
        recurrent=False,
        raw_observations=observations,
    )
    aux = to_state_dict(aux)
    # Validate that only actor_state.params has changed
    assert not compare_frozen_dicts(
        updated_state.actor_state.params, original_actor_params
    ), "actor_state.params should have been updated."

    # Validate auxiliary outputs
    assert "policy_loss" in aux, "Auxiliary outputs are missing 'policy_loss'."
    # assert aux["policy_loss"] <= 0, "Policy loss should be negative."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_update_agent(env_config, PPO_state):
    # Mock inputs for the update_agent function
    recurrent = False
    n_steps = 32
    agent_config = PPOConfig(
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,
        n_steps=n_steps,
        batch_size=16,
        n_epochs=3,
        normalize_advantage=True,
        gae_lambda=0.95,
    )
    observation_shape, action_shape = get_state_action_shapes(env_config.env)
    observations = jnp.ones((n_steps, env_config.n_envs, *observation_shape))
    rewards = jnp.ones((n_steps, env_config.n_envs, 1))
    terminated = jnp.zeros((n_steps, env_config.n_envs, 1))
    truncated = jnp.zeros((n_steps, env_config.n_envs, 1))
    actions = jnp.ones((n_steps, env_config.n_envs, *action_shape))
    log_probs = jnp.ones((n_steps, env_config.n_envs, 1))
    value_targets = jnp.ones((n_steps, env_config.n_envs, 1))
    gae = jnp.ones((n_steps, env_config.n_envs, 1))

    transition = Transition(
        obs=observations,
        action=actions,
        reward=rewards,
        terminated=terminated,
        truncated=truncated,
        next_obs=observations,
        log_prob=log_probs,
    )

    batch = (
        transition.obs,
        (
            jnp.expand_dims(transition.action, axis=-1)
            if jnp.ndim(transition.action)
            < 3  # discrete case without trailing dimension
            else transition.action
        ),
        transition.terminated,
        transition.truncated,
        value_targets,
        gae,
        (
            jnp.expand_dims(transition.log_prob, axis=-1)
            if jnp.ndim(transition.log_prob)
            < 3  # discrete case without trailing dimension
            else transition.log_prob.sum(-1, keepdims=True)
        ),
        observations,  # raw observations
    )
    shuffle_key = jax.random.PRNGKey(0)
    num_minibatches = agent_config.batch_size // 2
    shuffled_batch = get_minibatches_from_batch(
        batch, rng=shuffle_key, num_minibatches=num_minibatches
    )

    # Call the update_agent function
    updated_state, _ = update_agent(
        agent_state=PPO_state,
        _=None,
        agent_config=agent_config,
        shuffled_batch=shuffled_batch,
        recurrent=recurrent,
    )

    # Validate that the state has been updated
    assert updated_state is not None, "Updated state should not be None."
    assert updated_state.rng is not None, "Updated RNG should not be None."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_update_agent_with_scan(env_config, PPO_state):
    # Mock inputs for the update_agent function
    recurrent = False
    n_steps = 32

    agent_config = PPOConfig(
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,
        n_steps=n_steps,
        batch_size=16,
        n_epochs=3,
        normalize_advantage=True,
        gae_lambda=0.95,
    )
    observation_shape, action_shape = get_state_action_shapes(env_config.env)
    observations = jnp.ones((n_steps, env_config.n_envs, *observation_shape))
    rewards = jnp.ones((n_steps, env_config.n_envs, 1))
    terminated = jnp.zeros((n_steps, env_config.n_envs, 1))
    truncated = jnp.zeros((n_steps, env_config.n_envs, 1))
    actions = jnp.ones((n_steps, env_config.n_envs, *action_shape))
    log_probs = jnp.ones((n_steps, env_config.n_envs, 1))
    value_targets = jnp.ones((n_steps, env_config.n_envs, 1))
    gae = jnp.ones((n_steps, env_config.n_envs, 1))

    transition = Transition(
        obs=observations,
        action=actions,
        reward=rewards,
        terminated=terminated,
        truncated=truncated,
        next_obs=observations,
        log_prob=log_probs,
    )

    batch = (
        transition.obs,
        (
            jnp.expand_dims(transition.action, axis=-1)
            if jnp.ndim(transition.action)
            < 3  # discrete case without trailing dimension
            else transition.action
        ),
        transition.terminated,
        transition.truncated,
        value_targets,
        gae,
        (
            jnp.expand_dims(transition.log_prob, axis=-1)
            if jnp.ndim(transition.log_prob)
            < 3  # discrete case without trailing dimension
            else transition.log_prob.sum(-1, keepdims=True)
        ),
        observations,  # raw observations
    )
    shuffle_key = jax.random.PRNGKey(0)
    num_minibatches = agent_config.batch_size // 2
    shuffled_batch = get_minibatches_from_batch(
        batch, rng=shuffle_key, num_minibatches=num_minibatches
    )

    update_agent_scan = partial(
        update_agent,
        agent_config=agent_config,
        shuffled_batch=shuffled_batch,
        recurrent=recurrent,
    )

    # Run the scan
    final_state, _ = jax.lax.scan(update_agent_scan, PPO_state, None, length=5)

    # Validate the final state
    assert final_state is not None, "Final state should not be None."
    assert final_state.rng is not None, "Final RNG should not be None."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
)
def test_training_iteration_with_scan(env_config, PPO_state):
    recurrent = False
    n_steps = 32
    agent_config = PPOConfig(
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,
        n_steps=n_steps,
        batch_size=16,
        n_epochs=3,
        normalize_advantage=True,
        gae_lambda=0.95,
    )
    log_frequency = 10

    # Define a partial function for training_iteration
    training_iteration_scan = partial(
        training_iteration,
        env_args=env_config,
        mode="gymnax" if env_config.env_params else "brax",
        recurrent=recurrent,
        agent_config=agent_config,
        log_frequency=log_frequency,
        total_timesteps=5,
        n_steps=n_steps,
        total_n_updates=3,
    )

    # Run multiple training iterations using jax.lax.scan
    final_state, _ = jax.lax.scan(training_iteration_scan, PPO_state, None, length=5)

    # Validate the final state
    assert isinstance(final_state, PPOState), "Final state should be of type SACState."
    assert final_state.rng is not None, "Final RNG should not be None."


@pytest.mark.parametrize(
    "env_config",
    ["ant_env_config", "gymnax_env_config", "discrete_gymnax_env_config"],
    indirect=True,
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
    n_steps = 32
    agent_config = PPOConfig(
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,
        n_steps=n_steps,
        batch_size=16,
        n_epochs=3,
        normalize_advantage=True,
        gae_lambda=0.95,
    )
    mode = "gymnax" if env_config.env_params else "brax"

    total_timesteps = 1000

    def expert_policy(x):
        return (
            jnp.ones((env_config.env.action_size,))
            if mode == "brax"
            else env_config.env.action_space(env_config.env_params).sample(key)
        )

    cloning_args = CloningConfig(pre_train_n_steps=100)

    # Create the train function
    train_fn = make_train(
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        agent_config=agent_config,
        total_timesteps=total_timesteps,
        num_episode_test=2,
        expert_policy=expert_policy,
        cloning_args=cloning_args,
    )

    # Run the train function
    final_state, _ = train_fn(key)

    # Validate the final state
    assert isinstance(final_state, PPOState), "Final state should be of type SACState."
    assert final_state.rng is not None, "Final RNG should not be None."
    assert final_state.actor_state is not None, "Actor state should not be None."
    assert final_state.critic_state is not None, "Critic state should not be None."
    assert (
        final_state.collector_state is not None
    ), "Collector state should not be None."
