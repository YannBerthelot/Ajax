from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import DuelingQNetwork
from ajax.agents.DQN.train_DQN import compute_double_dqn_td_target, make_huber_td_loss


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
