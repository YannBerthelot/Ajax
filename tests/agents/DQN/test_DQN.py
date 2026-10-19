import jax
import pytest

from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import (
    DuelingQNetwork,
    QNetwork,
    get_initialized_q_network,
)
from ajax.agents.DQN.train_DQN import compute_double_dqn_td_target, make_huber_td_loss
from tests.agents.init_checks import assert_torch_linear


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


@pytest.mark.parametrize("cls", [QNetwork, DuelingQNetwork])
def test_dqn_starts_every_layer_as_torch7_nn_linear(cls):
    """google-deepmind/dqn never re-initialises its Torch7 layers, whose
    nn.Linear default draws kernel and bias uniform in +-1/sqrt(fan_in):
    the MLP encoder and every Q head, the dueling ones too."""
    agent = DQN(env_id="CartPole-v1", architecture=("64", "relu", "64", "relu"))
    q_state = get_initialized_q_network(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.network_args,
        n_actions=2,
        q_network_cls=cls,
    )
    params = q_state.params["params"]
    heads = ("value_head", "advantage_head") if cls is DuelingQNetwork else ("head",)
    layers = {  # name: (params, fan_in); CartPole's 4 observations
        "hidden 0": (params["encoder"]["network"]["layers_0"], 4),
        "hidden 1": (params["encoder"]["network"]["layers_2"], 64),
    } | {head: (params[head], 64) for head in heads}
    for name, (layer, fan_in) in layers.items():
        assert_torch_linear(name, layer, fan_in)
