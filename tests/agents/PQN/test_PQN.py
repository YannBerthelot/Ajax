import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax.agents.PQN.PQN as PQN_MODULE
from ajax.agents.critics import make_huber_td_loss
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.networks import PQNNetwork
from ajax.agents.PQN.PQN import PQN, compute_half_mse_td_loss
from ajax.extensions.expert import JSRLCurriculum
from ajax.networks.networks import MINATAR_CNN, MNIH_CNN


def test_PQN_rejects_indivisible_minibatches():
    """The rollout's n_steps * n_envs rows must split into num_minibatches."""
    with pytest.raises(ValueError, match="must be divisible"):
        PQN(env_id="CartPole-v1", n_envs=1, n_steps=7, num_minibatches=2)
    PQN(env_id="CartPole-v1", n_envs=2, n_steps=7, num_minibatches=2)


def test_the_rate_falls_linearly_to_zero_over_the_runs_gradient_steps():
    """lr_linear_decay (purejaxql's LR_LINEAR_DECAY, pqn_minatar.py:140-147):
    a run of 8 updates, each 2 epochs of 2 minibatches, takes 32 gradient
    steps; the rate is whole at the first, halved after 16 and gone after 32.
    A decay counted in updates would be gone after 8."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        n_steps=4,
        n_epochs=2,
        num_minibatches=2,
        learning_rate=1e-3,
    )
    _, context = agent.make_context(64)
    rate = agent.decayed_optimizer_args(context).learning_rate
    rates = [float(rate(step)) for step in (0, 16, 32)]
    np.testing.assert_allclose(rates, [1e-3, 5e-4, 0.0], atol=1e-9)


def test_pqn_keeps_the_executed_actions_it_learns_from():
    """PQN's Q-learning stays correct on actions its policy did not choose,
    so it fits Q where the steps acted: its rollout keeps the executed
    actions, the policy's column without an action extension. PPO, whose
    update needs the policy's own actions, keeps none."""
    small = {"env_id": "CartPole-v1", "n_envs": 2, "n_steps": 8, "n_epochs": 1}
    agent = PQN(
        **small,
        architecture=("16", "relu"),
        num_minibatches=2,
        expose_recent_rollout=True,
    )
    state, _ = agent.train(seed=0, n_timesteps=32)
    rollout = state.last_rollout
    assert rollout.executed_action.dtype == rollout.action.dtype
    np.testing.assert_array_equal(rollout.executed_action, rollout.action)
    ppo = PPO(**small, batch_size=8, expose_recent_rollout=True)
    state, _ = ppo.train(seed=0, n_timesteps=32)
    assert state.last_rollout.executed_action is None


def test_pqn_fits_q_where_the_expert_acted(monkeypatch):
    """Under JSRL handing its expert (index 0) every step, PQN's TD steps
    and its extensions' targets get index 0 throughout: the executed column,
    not the policy's epsilon-greedy choice (uniform at first, so 1 on about
    half the steps)."""
    seen = []
    step = PQN_MODULE.q_gradient_step

    def recording(agent_state, observations, actions, *args):
        jax.debug.callback(lambda a: seen.append(np.asarray(a)), actions)
        return step(agent_state, observations, actions, *args)

    monkeypatch.setattr(PQN_MODULE, "q_gradient_step", recording)
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        n_steps=8,
        n_epochs=1,
        num_minibatches=2,
        architecture=("16", "relu"),
        expert_policy=lambda obs: jnp.zeros(obs.shape[:-1] + (1,), jnp.int32),
        extensions=(
            JSRLCurriculum(episode_length=1000, decay_frac=1e6, schedule="time"),
        ),
    )
    agent.train(seed=0, n_timesteps=32)
    jax.effects_barrier()
    assert seen
    for actions in seen:
        np.testing.assert_array_equal(actions, 0)


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


def test_pqn_steps_as_its_reference_optimiser():
    """mttga/purejaxql @47af6d7 (pqn_minatar.py:159-162): optax.radam,
    eps 1e-8, after a global-norm clip of 10."""
    agent = PQN(env_id="CartPole-v1")
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.optimizer, args.eps, args.max_grad_norm) == ("radam", 1e-8, 10.0)


def test_pqn_default_loss_is_half_the_squared_error():
    """purejaxql's loss, 0.5 * mean((Q - G)^2) (pqn_minatar.py:285): its
    gradient is the TD error over the batch size, what the clip of 10 sees."""
    q, target = jnp.array([0.0, 1.0]), jnp.array([3.0, -1.0])
    assert compute_half_mse_td_loss(q, target) == pytest.approx(0.5 * (9.0 + 4.0) / 2)
    assert jnp.allclose(jax.grad(compute_half_mse_td_loss)(q, target), (q - target) / 2)


def test_pqn_starts_every_dense_as_flax_default():
    """PQN's reference code (mttga/purejaxql @47af6d7, pqn_gymnax.py:52, :56)
    sets no initializer: every Dense is lecun_normal, a normal truncated at
    two standard deviations with variance 1/fan_in, and a zero bias."""
    network = PQNNetwork(input_architecture=("128", "relu", "128", "relu"), n_actions=2)
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))["params"]
    for name, fan_in in (("Dense_0", 4), ("Dense_1", 128), ("Dense_2", 128)):
        kernel = np.asarray(params[name]["kernel"])
        std = 1 / np.sqrt(fan_in)
        np.testing.assert_allclose(kernel.std(), std, rtol=0.2, err_msg=name)
        assert np.abs(kernel).max() <= 2 * std / 0.87962566, name
        np.testing.assert_array_equal(params[name]["bias"], 0.0, err_msg=name)


def test_pqn_normalises_after_every_hidden_dense():
    """PQN's defining layer: LayerNorm after each hidden Dense (Gallici et
    al. 2024; purejaxql's pqn_gymnax.py). At initialisation the biases are
    zero, so a Dense scales with its input and the LayerNorm after it
    removes the scale: Q(10 x) = Q(x). Without the LayerNorms the Q-values
    would be 10 times larger. LayerNorm's epsilon (1e-6 under the root)
    leaves a relative difference of about 1e-6 here; 1e-4 allows float32."""
    network = PQNNetwork(input_architecture=("16", "relu", "16", "relu"), n_actions=3)
    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    params = network.init(jax.random.PRNGKey(0), obs)
    q_values = network.apply(params, obs).q_values
    scaled = network.apply(params, 10.0 * obs).q_values
    assert float(jnp.abs(q_values).max()) > 1e-2  # not trivially equal at zero
    np.testing.assert_allclose(scaled, q_values, rtol=1e-4, atol=1e-5)


def test_pqn_images_build_purejaxql_s_network_and_train_on_minatar():
    """purejaxql's image networks: Mnih et al.'s from 36 px a side, MinAtar's
    below, he_normal with a LayerNorm after every layer, straight into the
    Q head (no MLP stack after them)."""
    want = {(84, 84, 4): MNIH_CNN, (10, 10, 4): MINATAR_CNN}
    for shape, spec in want.items():
        agent = PQN(env_id="CartPole-v1", cnn_image_shape=shape)
        ref = spec._replace(kernel_init="he_normal", layer_norm=True)
        assert agent.network_args.cnn_spec == ref
    net = PQNNetwork(("8", "relu"), 3, cnn_image_shape=(10, 10, 4), cnn_spec=ref)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 400)))["params"]
    assert sorted(params) == ["CNNEncoder_0", "Dense_0"]
    assert params["Dense_0"]["kernel"].shape == (128, 3)
    agent = PQN(env_id="Breakout-MinAtar", cnn_image_shape=(10, 10, 4), n_envs=2)
    agent.train(seed=0, n_timesteps=2 * agent.n_steps)
