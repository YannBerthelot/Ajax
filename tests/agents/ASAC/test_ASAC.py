import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.ASAC import train_ASAC
from ajax.agents.ASAC.ASAC import ASAC
from ajax.networks.distributions import GuardedTanh
from ajax.networks.networks import get_initialized_actor_critic, predict_value
from tests.conformance import kit
from tests.environments.identity_env import IdentityEnv
from tests.probing import envs


@pytest.mark.parametrize(
    "target_entropy_per_dim, expected_sign",
    [
        (-10.0, -1),  # target far below the policy entropy → alpha shrinks
        (10.0, 1),  # target far above the policy entropy → alpha grows
    ],
)
def test_alpha_tracks_target_entropy(target_entropy_per_dim, expected_sign):
    """``target_entropy_per_dim`` drives the temperature update.

    Regression test: the temperature step used to hardcode the target
    to ``-action_dim`` and discard the updated state, so alpha stayed
    frozen at ``alpha_init`` no matter what the user configured.
    """
    agent = ASAC(
        env_id="fast",
        learning_starts=10,
        target_entropy_per_dim=target_entropy_per_dim,
        alpha_init=1.0,
        fixed_alpha=False,
    )
    state, _ = agent.train(seed=0, n_timesteps=60)
    log_alpha = float(jnp.asarray(state.alpha.params["log_alpha"]).ravel()[0])
    assert log_alpha != 0.0, "alpha never moved from alpha_init"
    assert jnp.sign(log_alpha) == expected_sign


def test_temperature_stays_at_alpha_init_by_default():
    """The authors' published runs fix the temperature (1/beta = 0.2,
    BobMak/asac @4b68862 hparams/*/asac.yaml): no step moves it, even with a
    target entropy that would."""
    agent = ASAC(env_id="fast", learning_starts=10, target_entropy_per_dim=10.0)
    state, _ = agent.train(seed=0, n_timesteps=60)
    np.testing.assert_array_equal(state.alpha.params["log_alpha"], jnp.log(0.2))


def test_adam_eps_is_the_references():
    """The reference's torch.optim.Adam keeps torch's eps, 1e-8 (BobMak/asac
    @4b68862 darer/ASAC.py:87-92), not Ajax's shared 1e-5."""
    agent = ASAC(env_id="fast")
    assert agent.actor_optimizer_args.eps == agent.critic_optimizer_args.eps == 1e-8


def test_tau_theta_alone_moves_theta_and_the_penalty():
    """``tau_theta`` is the rate of the ``theta`` and penalty averages
    (BobMak/asac @4b68862, darer/ASAC.py:188, :213), apart from ``tau``,
    the target critics' rate: at 0 both averages stay at 0."""
    net = ("8", "relu")
    kw = {"actor_architecture": net, "critic_architecture": net, "batch_size": 4}
    agent = ASAC("Pendulum-v1", learning_starts=4, tau_theta=0.0, **kw)
    state, _ = agent.train(seed=0, n_timesteps=16)
    assert not np.any(state.theta) and not np.any(state.episode_termination_penalty)
    moved = ASAC("Pendulum-v1", learning_starts=4, **kw).train(seed=0, n_timesteps=16)
    assert np.all(moved[0].theta != 0.0)


def test_temperature_and_theta_read_the_actors_own_sample(monkeypatch):
    """The temperature and the reward-rate (theta) steps read the sample
    the actor's loss drew, before its step, as the authors' code
    (BobMak/asac @4b68862, darer/ASAC.py:146-213): no other draw."""
    seen, drawn = [], []
    temperature, theta = train_ASAC.update_temperature, train_ASAC.update_theta
    policy = train_ASAC.update_policy

    def policy_spy(*args, **kwargs):
        out = policy(*args, **kwargs)
        drawn.append(out[2])  # its sample's log-probabilities
        return out

    def temperature_spy(*args):
        seen.append(args[1])  # the log-probabilities
        return temperature(*args)

    def theta_spy(*args):
        seen.append(args[3])
        return theta(*args)

    monkeypatch.setattr(train_ASAC, "update_policy", policy_spy)
    monkeypatch.setattr(train_ASAC, "update_temperature", temperature_spy)
    monkeypatch.setattr(train_ASAC, "update_theta", theta_spy)
    net = ("8", "relu")
    agent = ASAC(
        "Pendulum-v1",
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=4,
        batch_size=4,
    )
    agent.train(seed=0, n_timesteps=8)
    # The update is traced twice: once for it, once for the structure of
    # the NaN metrics before learning_starts (TrainLoop.maybe_update).
    assert len(seen) == 4 and len(drawn) == 2
    assert seen[0] is seen[1] is drawn[0] and seen[2] is seen[3] is drawn[1]


def test_a_termination_drops_the_next_value_and_keeps_the_shift():
    """``r - theta + (1 - done) (min_i Q'_i(s', a') - alpha log pi(a'|s'))
    - Q'(0, 0)``, as the authors' code (darer/ASAC.py:179-205)."""
    net = ("8", "relu")
    agent = ASAC("Pendulum-v1", actor_architecture=net, critic_architecture=net)
    state = kit.capture_step(agent, 64).fresh(jax.random.PRNGKey(0))
    next_obs = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    rewards = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    dones = jnp.array([[1.0], [0.0], [1.0], [0.0]])

    def target(d):
        return train_ASAC.compute_asac_td_target(
            state.actor_state,
            state.critic_state,
            jax.random.PRNGKey(2),
            next_obs,
            d,
            rewards,
            0.5,
            jnp.asarray(0.2),
            1.0,
        )[0]

    critic = state.critic_state
    origin = predict_value(critic, critic.target_params, jnp.zeros((1, 4))).mean()
    ended, going = target(dones), target(jnp.zeros_like(dones))
    np.testing.assert_allclose(ended[0::2], rewards[0::2] - 0.5 - origin, rtol=1e-6)
    np.testing.assert_allclose(ended[1::2], going[1::2], rtol=1e-6)
    assert np.all(np.abs(going[0::2] - ended[0::2]) > 1e-6)


def test_the_entropy_terms_are_relative_to_a_uniform_policy_over_the_actions():
    """With nA = 2 action dimensions, the target's and theta's entropy terms
    read alpha (log pi - log(1 / nA)) (darer/ASAC.py:54, :182, :191); the
    policy's log-prob subtracts log(1 - tanh^2 + 1e-6), as the reference's
    stable-baselines3 actor."""
    spec = envs.Spec("two", lambda st, a, k: (st, 0.0, False), box=(-1.0, 1.0, 2))
    net = ("8", "relu")
    env, params = spec.make()
    agent = ASAC(
        env, env_params=params, actor_architecture=net, critic_architecture=net
    )
    state = kit.capture_step(agent, 64).fresh(jax.random.PRNGKey(0))
    next_obs = jax.random.normal(jax.random.PRNGKey(1), (4, 1))
    dones, rewards = jnp.array([[1.0], [0.0], [0.0], [1.0]]), jnp.ones((4, 1))

    def target(alpha):
        args = (state.actor_state, state.critic_state, jax.random.PRNGKey(2), next_obs)
        return train_ASAC.compute_asac_td_target(
            *args, dones, rewards, 0.5, jnp.asarray(alpha), 1.0
        )

    (soft, log_pi), (hard, _) = target(0.2), target(0.0)
    want = -0.2 * (log_pi + np.log(2.0)) * (1.0 - dones)
    np.testing.assert_allclose(soft - hard, want, rtol=1e-5, atol=1e-6)

    log_probs = jnp.array([[-1.0], [0.5]])
    moved = train_ASAC.update_theta(state, 1.0, jnp.ones((2, 1)), log_probs, 2)
    want_theta = 1.0 - 0.2 * (np.mean([-1.0, 0.5]) + np.log(2.0))
    np.testing.assert_allclose(moved.theta, want_theta, rtol=1e-6)

    pi = state.actor_state.apply_fn(state.actor_state.params, next_obs)
    assert isinstance(pi.bijector, GuardedTanh) and pi.bijector.eps == 1e-6


def test_asac_starts_every_layer_as_torch_nn_linear():
    """ASAC's reference code builds stable-baselines3 2.4.0's SAC actor and
    plain torch critics (BobMak/asac @4b68862) at torch's default: every
    kernel and bias uniform in +-1/sqrt(fan_in), the heads included."""
    agent = ASAC(env_id="Pendulum-v1")
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
    )
    actor = actor_state.params["params"]
    critic = jax.tree.map(lambda x: x[0], critic_state.params["params"]["ensemble"])
    obs_dim, act_dim = 3, 1  # Pendulum
    layers = {  # name: (params, fan_in)
        "actor hidden 0": (actor["encoder"]["network"]["layers_0"], obs_dim),
        "actor hidden 1": (actor["encoder"]["network"]["layers_2"], 256),
        "mean": (actor["mean"], 256),
        "log_std": (actor["log_std"], 256),
        "critic hidden 0": (
            critic["encoder"]["network"]["layers_0"],
            obs_dim + act_dim,
        ),
        "critic hidden 1": (critic["encoder"]["network"]["layers_2"], 256),
        "Q": (critic["model"], 256),
    }
    for name, (layer, fan_in) in layers.items():
        bound = 1 / np.sqrt(fan_in)
        kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
        assert np.abs(kernel).max() <= bound and np.abs(bias).max() <= bound, name
        assert np.abs(kernel).max() > 0.5 * bound and np.any(bias), name
        if kernel.size >= 256:  # a uniform's variance, bound**2 / 3
            np.testing.assert_allclose(kernel.var(), bound**2 / 3, rtol=0.3)


def test_recurrent_asac_keeps_its_previous_initialisation():
    """With memory, ASAC keeps Ajax's previous initialisation until Ni et
    al.'s is audited: an orthogonal(0.01) mean and a zero-kernel log-std
    head starting at -1."""
    from ajax.networks.memory import MemoryConfig

    agent = ASAC(env_id="Pendulum-v1", memory=MemoryConfig("gru", 16))
    actor_state, _ = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
    )
    actor = actor_state.params["params"]
    mean = np.asarray(actor["mean"]["kernel"])
    np.testing.assert_allclose(mean.T @ mean, 1e-4 * np.eye(1), atol=1e-8)
    np.testing.assert_array_equal(actor["log_std"]["kernel"], 0.0)
    np.testing.assert_array_equal(actor["log_std"]["bias"], -1.0)


def test_a_terminated_rows_reward_is_charged_the_penalty_for_the_critics_alone(
    monkeypatch,
):
    """ASAC's critics learn r - p d: each replayed row's reward less the
    termination penalty p (its running value, updated on the batch first)
    where the row terminated; theta tracks the rewards themselves (BobMak/asac
    @4b68862, darer/ASAC.py:179-205). The identity task terminates some
    episodes; p_0 = 1 and tau_theta = 1 make p the batch's own."""
    seen, theta_rewards = [], []
    update, update_theta = train_ASAC.update_value_functions, train_ASAC.update_theta

    def spy(agent_state, transition, rewards, *args, **kwargs):
        jax.debug.callback(
            lambda *v: seen.append(tuple(map(np.asarray, v))),
            rewards,
            transition.reward,
            transition.terminated,
            agent_state.episode_termination_penalty,
        )
        return update(agent_state, transition, rewards, *args, **kwargs)

    def theta_spy(agent_state, tau, rewards, *args, **kwargs):
        jax.debug.callback(lambda r: theta_rewards.append(np.asarray(r)), rewards)
        return update_theta(agent_state, tau, rewards, *args, **kwargs)

    monkeypatch.setattr(train_ASAC, "update_value_functions", spy)
    monkeypatch.setattr(train_ASAC, "update_theta", theta_spy)
    env = IdentityEnv()
    kw = {"learning_starts": 16, "batch_size": 16, "p_0": 1.0, "tau_theta": 1.0}
    agent = ASAC(env, env_params=env.default_params, n_envs=4, **kw)
    agent.train(seed=0, n_timesteps=96)
    jax.effects_barrier()
    assert any(t.any() and p != 0 for _, _, t, p in seen), "nothing was charged"
    assert len(theta_rewards) == len(seen)
    scale = agent.agent_config.reward_scale
    for (charged, reward, terminated, penalty), theta in zip(
        seen, theta_rewards, strict=True
    ):
        np.testing.assert_allclose(charged, reward - penalty * terminated, rtol=1e-6)
        np.testing.assert_allclose(theta, reward * scale, rtol=1e-6)
