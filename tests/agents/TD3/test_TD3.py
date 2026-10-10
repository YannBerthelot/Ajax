import jax
import pytest

from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.TD3 import TD3
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType


def test_TD3_initialization():
    """TD3 builds the expected sub-configs."""
    env_id = "Pendulum-v1"
    agent = TD3(env_id=env_id)
    for expected_attr, expected_type in zip(
        (
            "env_args",
            "actor_optimizer_args",
            "critic_optimizer_args",
            "network_args",
            "buffer",
        ),
        (
            EnvironmentConfig,
            OptimizerConfig,
            OptimizerConfig,
            NetworkConfig,
            BufferType,
        ),
    ):
        assert hasattr(agent, expected_attr)
        assert isinstance(getattr(agent, expected_attr), expected_type)


def test_TD3_initialization_with_discrete_env():
    """TD3 must reject discrete action spaces."""
    with pytest.raises(ValueError, match="TD3 only supports continuous action spaces."):
        TD3(env_id="CartPole-v1")


@pytest.mark.parametrize(
    "env_id, seeds, n_envs",
    [
        ["fast", 42, 1],
        ["fast", [42, 43], 2],
    ],
)
def test_td3_train_smoke(env_id, seeds, n_envs):
    """Train a few steps to check the full pipeline executes end-to-end."""
    n_timesteps = 50
    learning_starts = 10
    agent = TD3(
        env_id=env_id,
        learning_starts=learning_starts,
        n_envs=n_envs,
        # Tiny networks for speed
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        batch_size=8,
        buffer_size=128,
    )
    agent.train(seed=seeds, n_timesteps=n_timesteps)


def test_td3_networks_are_plain_mlps():
    """Fujimoto's actor and critic are plain MLPs (sfujim/TD3, TD3.py): only
    Dense kernels and biases, no LayerNorm on the critic's encoder output."""
    agent = TD3(env_id="Pendulum-v1")
    nets = get_initialized_td3_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    for net in nets:
        paths = [
            [getattr(k, "key", k) for k in path]
            for path, _ in jax.tree_util.tree_flatten_with_path(net.params)[0]
        ]
        assert {path[-1] for path in paths} == {"kernel", "bias"}, paths
        assert not any("norm" in str(k).lower() for p in paths for k in p), paths
