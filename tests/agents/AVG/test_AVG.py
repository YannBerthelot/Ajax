import pytest

from ajax.agents.AVG.AVG import AVG
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig


@pytest.mark.parametrize(
    "env_id",
    [
        "Pendulum-v1",
        "fast",
    ],
)
def test_avg_initialization(env_id):
    """Test AVG agent initialization with default parameters."""

    avg_agent = AVG(env_id=env_id)

    for expected_attr, expected_type in zip(
        (
            "env_args",
            "actor_optimizer_args",
            "critic_optimizer_args",
            "network_args",
            "alpha_args",
        ),
        (
            EnvironmentConfig,
            OptimizerConfig,
            OptimizerConfig,
            NetworkConfig,
            AlphaConfig,
        ),
    ):
        assert hasattr(avg_agent, expected_attr)
        assert isinstance(getattr(avg_agent, expected_attr), expected_type)


def test_avg_initialization_with_discrete_env():
    """Test AVG agent initialization fails with a discrete environment."""
    env_id = "CartPole-v1"
    with pytest.raises(ValueError, match="AVG only supports continuous action spaces."):
        AVG(env_id=env_id)


@pytest.mark.parametrize(
    "env_id, seeds, n_envs",
    [
        ["fast", 42, 1],
        ["fast", [42, 43], 2],
    ],
)
def test_avg_train_all_modes(env_id, seeds, n_envs):
    n_timesteps = 50  # keep small for speed
    learning_starts = 10

    avg_agent = AVG(env_id=env_id, learning_starts=learning_starts, n_envs=n_envs)
    avg_agent.train(seed=seeds, n_timesteps=n_timesteps)


def test_avg_keeps_its_network_optimizer_and_env_settings():
    """AVG builds on ActorCritic, then sets what the paper uses: squashed
    actions with penultimate normalisation, Adam with ``beta_1 = 0``,
    normalised observations, no memory."""
    agent = AVG(env_id="Pendulum-v1", beta_1=0.0, beta_2=0.99)
    assert agent.network_args.squash and agent.network_args.penultimate_normalization
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.beta_1, args.beta_2) == (0.0, 0.99)
    assert "NormalizeVecObservation" in type(agent.env_args.env._env).__name__
    assert not AVG.supports_memory
    assert agent.get_make_train().keywords["alpha_args"] is agent.alpha_args
