import pytest

from ajax.agents.DQN.train_DQN import make_huber_td_loss
from ajax.agents.PQN.PQN import PQN


def test_PQN_rejects_indivisible_minibatches():
    """The rollout's n_steps * n_envs rows must split into num_minibatches."""
    with pytest.raises(ValueError, match="must be divisible"):
        PQN(env_id="CartPole-v1", n_envs=1, n_steps=7, num_minibatches=2)
    PQN(env_id="CartPole-v1", n_envs=2, n_steps=7, num_minibatches=2)


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
