import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.critics import compute_huber_1_td_loss, make_huber_td_loss
from ajax.agents.DQN.DQN import DQN, compute_double_dqn_td_target
from ajax.buffers.utils import get_batch_from_buffer
from ajax.extensions.base import ExtensionContext
from ajax.extensions.expert import ExpertValueMetrics, ExpertWarmup
from ajax.extensions.pretrain import ExpertPrefill
from ajax.networks.networks import MINATAR_CNN, MNIH_CNN, CNNEncoder, CNNSpec
from ajax.networks.q_networks import (
    DuelingQNetwork,
    GreedyQPolicy,
    QNetwork,
    get_initialized_q_network,
    predict_q,
)
from ajax.networks.utils import TORCH_LINEAR
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
    agent.train(seed=42, n_timesteps=128)


@pytest.mark.parametrize("batch_size", [32, 64])
def test_dqn_steps_as_its_reference_optimiser(batch_size):
    """google-deepmind/dqn @9d9b1d1 (NeuralQLearner.lua:265-277, run_gpu:25):
    centered RMSProp at lr 2.5e-4, both averages at 0.95, eps 0.01 on a
    gradient summed over the batch, so 0.01 / batch_size**2 on a mean."""
    agent = DQN(env_id="CartPole-v1", batch_size=batch_size)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.optimizer, args.learning_rate, args.beta_2) == (
            "centered_rmsprop",
            2.5e-4,
            0.95,
        )
        assert args.eps == pytest.approx(0.01 / batch_size**2)


def test_the_default_loss_clips_the_td_error_at_one():
    """Huber 1 (NeuralQLearner.lua:224-226, clip_delta=1): its gradient is
    the TD error clipped to [-1, 1]."""
    q, target = jnp.array([0.0, 0.0, 0.0]), jnp.array([3.0, -0.5, -2.0])
    grad = jax.grad(compute_huber_1_td_loss)(q, target) * q.size
    assert jnp.allclose(grad, jnp.array([-1.0, 0.5, 1.0]))


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
    }
    layers.update({head: (params[head], 64) for head in heads})
    for name, (layer, fan_in) in layers.items():
        assert_torch_linear(name, layer, fan_in)


def test_dqn_images_build_its_reference_network_at_torch7_s_default():
    """Images of 36 px a side and more get Mnih et al.'s network, smaller
    ones MinAtar's, both at Torch7's default; a given cnn_spec is kept."""
    want = {(84, 84, 4): MNIH_CNN, (36, 40, 1): MNIH_CNN, (35, 84, 1): MINATAR_CNN}
    for shape, spec in want.items():
        agent = DQN(env_id="CartPole-v1", cnn_image_shape=shape)
        assert agent.network_args.cnn_spec == spec._replace(kernel_init=TORCH_LINEAR)
    mine = CNNSpec(channels=(4,), kernel_sizes=(2,), strides=(1,), feature_dim=8)
    agent = DQN(env_id="CartPole-v1", cnn_image_shape=(10, 10, 4), cnn_spec=mine)
    assert agent.network_args.cnn_spec == mine
    encoder = CNNEncoder(
        (10, 10, 4), spec=MINATAR_CNN._replace(kernel_init=TORCH_LINEAR)
    )
    params = encoder.init(jax.random.PRNGKey(0), jnp.zeros((1, 400)))["params"]
    assert_torch_linear("conv", params["Conv_0"], 3 * 3 * 4)
    assert_torch_linear("projection", params["TorchLinear_0"], 8 * 8 * 16)
    assert params["TorchLinear_0"]["kernel"].shape == (8 * 8 * 16, 128)


def test_dqn_trains_minatar_s_network_on_minatar_and_refuses_tinier_images():
    """MinAtar's 10 x 10 x 4 observations train through MinAtar's network;
    an image smaller than its 3 x 3 kernel asks for a cnn_spec."""
    agent = DQN(
        env_id="Breakout-MinAtar",
        cnn_image_shape=(10, 10, 4),
        learning_starts=16,
        batch_size=4,
        buffer_size=64,
    )
    agent.train(seed=0, n_timesteps=32)
    with pytest.raises(ValueError, match="pass a cnn_spec"):
        DQN(env_id="CartPole-v1", cnn_image_shape=(2, 2, 1))


def test_the_greedy_policy_takes_the_largest_q_value():
    """mode = argmax_a Q(s, a): the action DQN and PQN act and evaluate with."""
    q = jnp.array([[0.1, 0.9, 0.3], [2.0, -1.0, 1.0], [0.0, 0.0, 5.0]])
    assert GreedyQPolicy(q).mode().tolist() == [1, 0, 2]


def push_toward_lean(obs: jax.Array) -> jax.Array:
    """CartPole's expert: push toward the pole's lean, an int (n, 1) column."""
    return (obs[..., 2:3] > 0).astype(jnp.int32)


EXPERT_RUN = {"env_id": "CartPole-v1", "n_envs": 2, "buffer_size": 200}
EXPERT_RUN.update({"batch_size": 16, "learning_starts": 32})
EXPERT_RUN.update({"architecture": ("16", "relu"), "expert_policy": push_toward_lean})


def test_the_expert_fills_the_prefill_and_the_warm_up_rows_with_its_index():
    """ExpertPrefill's 16 rows per env and the warm-up's (ExpertWarmup taking
    every step) store the expert's index and are flagged as its; the
    policy's rows after learning_starts are not."""
    extensions = (ExpertPrefill(n_steps=16), ExpertWarmup(fraction=1.0))
    agent = DQN(**EXPERT_RUN, extensions=extensions)
    state, _ = agent.train(seed=0, n_timesteps=64)
    rows = jax.tree.map(lambda x: np.asarray(x[0]), state.collector_state.buffer_state)
    written = int(rows.current_index)
    assert written == 16 + 32  # the prefill, then 32 steps per env
    flags = rows.experience["is_expert"][:, :written, 0]
    np.testing.assert_array_equal(flags[:, :32], 1.0)
    np.testing.assert_array_equal(flags[:, 32:], 0.0)
    index = np.asarray(push_toward_lean(rows.experience["raw_obs"]))[:, :written]
    np.testing.assert_array_equal(rows.experience["action"][:, :32], index[:, :32])
    np.testing.assert_array_equal(rows.experience["a_expert"][:, :written], index)


def test_expert_value_metrics_read_the_q_network_at_the_stored_indices():
    """On DQN, value/q_expert_mean is the Q-network's value at the expert's
    stored index and value/q_gap that minus its value at the stored action,
    each a column of the Q table, on the replay batch the extension draws."""
    agent = DQN(**EXPERT_RUN, extensions=(ExpertValueMetrics(),))
    state, _ = agent.train(seed=0, n_timesteps=64)
    state = jax.tree.map(lambda x: x[0], state)
    _, context = agent.make_context(64)
    metrics = context.extensions.extensions[0]
    key = jax.random.PRNGKey(1)
    context = ExtensionContext(step=0, rng=key, total_steps=64)
    got = metrics.eval_metrics(state, None, key, context)
    batch = get_batch_from_buffer(
        metrics.buffer, state.collector_state.buffer_state, key
    )
    q = state.actor_state
    table = np.asarray(predict_q(q, q.params, batch.obs))
    rows = np.arange(len(table))
    q_expert = table[rows, np.asarray(batch.a_expert, int)[:, 0]]
    q_taken = table[rows, np.asarray(batch.action, int)[:, 0]]
    np.testing.assert_allclose(got["value/q_expert_mean"], q_expert.mean(), rtol=1e-5)
    np.testing.assert_allclose(
        got["value/q_gap"], (q_expert - q_taken).mean(), rtol=1e-5, atol=1e-6
    )
