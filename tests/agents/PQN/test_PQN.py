import pytest

from ajax.agents.DQN.train_DQN import make_huber_td_loss
from ajax.agents.PQN.PQN import PQN
from ajax.agents.PQN.state import PQNConfig
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig


def test_PQN_initialization():
    """PQN builds the expected sub-configs on a discrete env."""
    agent = PQN(env_id="CartPole-v1")
    for expected_attr, expected_type in zip(
        (
            "env_args",
            "actor_optimizer_args",
            "critic_optimizer_args",
            "network_args",
            "agent_config",
        ),
        (
            EnvironmentConfig,
            OptimizerConfig,
            OptimizerConfig,
            NetworkConfig,
            PQNConfig,
        ),
    ):
        assert hasattr(agent, expected_attr)
        assert isinstance(getattr(agent, expected_attr), expected_type)
    # PQN is on-policy: no replay buffer.
    assert not hasattr(agent, "buffer")


def test_PQN_initialization_with_continuous_env():
    """PQN must reject continuous action spaces."""
    with pytest.raises(ValueError, match="PQN only supports discrete action spaces."):
        PQN(env_id="Pendulum-v1")


def test_PQN_rejects_indivisible_minibatches():
    """n_steps must be divisible by num_minibatches."""
    with pytest.raises(ValueError, match="must be divisible"):
        PQN(env_id="CartPole-v1", n_steps=7, num_minibatches=2)


@pytest.mark.parametrize(
    "seeds, n_envs",
    [
        [42, 2],
        [[42, 43], 4],
    ],
)
def test_pqn_train_smoke(seeds, n_envs):
    """Train a few steps to check the full pipeline executes end-to-end."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=n_envs,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
    )
    agent.train(seed=seeds, n_timesteps=256)


def test_pqn_huber_loss_smoke():
    """The Huber td_loss_fn hook composes and runs end-to-end."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        td_loss_fn=make_huber_td_loss(1.0),
    )
    agent.train(seed=42, n_timesteps=256)
