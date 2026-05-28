import pytest

from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import DuelingQNetwork
from ajax.agents.DQN.train_DQN import compute_double_dqn_td_target, make_huber_td_loss
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType


def test_DQN_initialization():
    """DQN builds the expected sub-configs on a discrete env."""
    agent = DQN(env_id="CartPole-v1")
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


def test_DQN_initialization_with_continuous_env():
    """DQN must reject continuous action spaces."""
    with pytest.raises(ValueError, match="DQN only supports discrete action spaces."):
        DQN(env_id="Pendulum-v1")


@pytest.mark.parametrize(
    "seeds, n_envs",
    [
        [42, 1],
        [[42, 43], 2],
    ],
)
def test_dqn_train_smoke(seeds, n_envs):
    """Train a few steps to check the full pipeline executes end-to-end."""
    n_timesteps = 200
    agent = DQN(
        env_id="CartPole-v1",
        learning_starts=32,
        n_envs=n_envs,
        # Tiny network for speed.
        architecture=("16", "relu"),
        batch_size=8,
        buffer_size=256,
        target_update_interval=10,
    )
    agent.train(seed=seeds, n_timesteps=n_timesteps)


def test_dqn_variants_smoke():
    """Double DQN + Dueling network + Huber loss compose and run end-to-end."""
    agent = DQN(
        env_id="CartPole-v1",
        learning_starts=32,
        n_envs=1,
        architecture=("16", "relu"),
        batch_size=8,
        buffer_size=256,
        target_update_interval=10,
        td_target_fn=compute_double_dqn_td_target,
        td_loss_fn=make_huber_td_loss(1.0),
        q_network_cls=DuelingQNetwork,
    )
    agent.train(seed=42, n_timesteps=200)
