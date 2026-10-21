import jax
import numpy as np

from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.TD3.train_TD3 import make_default_action_pipeline
from ajax.environments.interaction import get_pi


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


def test_adam_eps_is_the_references():
    """The reference's torch.optim.Adam keeps torch's eps, 1e-8 (sfujim/TD3
    @34770cc TD3.py:83, :87), not Ajax's shared 1e-5."""
    agent = TD3(env_id="Pendulum-v1")
    assert agent.actor_optimizer_args.eps == agent.critic_optimizer_args.eps == 1e-8


def test_td3_starts_every_layer_as_torch_nn_linear():
    """sfujim/TD3 (camera-ready bc98e62, as at 34770cc) never re-initialises
    its nn.Linear layers: torch's default, every kernel and bias uniform in
    +-1/sqrt(fan_in), the actor's last layer included."""
    agent = TD3(env_id="Pendulum-v1")
    actor_state, critic_state = get_initialized_td3_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    actor = actor_state.params["params"]
    critic = jax.tree.map(lambda x: x[0], critic_state.params["params"]["ensemble"])
    obs_dim, act_dim = 3, 1  # Pendulum
    layers = {  # name: (params, fan_in); TD3's 256-256 hidden layers
        "actor hidden 0": (actor["hidden"]["layers_0"], obs_dim),
        "actor hidden 1": (actor["hidden"]["layers_2"], 256),
        "actor head": (actor["head"], 256),
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


def test_recurrent_td3_keeps_its_previous_initialisation():
    """With memory, TD3 keeps Ajax's previous initialisation until Ni et
    al.'s is audited: orthogonal(1.0) hidden layers and DDPG's uniform(+-3e-3)
    last layer with a zero bias."""
    from ajax.networks.memory import MemoryConfig

    agent = TD3(env_id="Pendulum-v1", memory=MemoryConfig("gru", 16))
    actor_state, _ = get_initialized_td3_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    actor = actor_state.params["params"]
    hidden = np.asarray(actor["hidden"]["layers_2"]["kernel"])
    np.testing.assert_allclose(hidden.T @ hidden, np.eye(256), atol=1e-4)
    head = np.asarray(actor["head"]["kernel"])
    assert 1e-3 < np.abs(head).max() <= 3e-3
    np.testing.assert_array_equal(actor["head"]["bias"], 0.0)


def test_the_behaviour_policy_adds_its_exploration_noise():
    """clip(mu(s) + N(0, sigma^2), -1, 1) after the warm-up (Fujimoto et al.,
    2018): at sigma = 0.3 the executed action is off the actor's mean by
    about sigma, and at sigma = 0 it is the mean."""
    agent = TD3(env_id="Pendulum-v1", n_envs=64)
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    actor = state.actor_state
    mean = get_pi(actor, actor.params, state.collector_state.last_obs)[0].mean()

    def act(sigma: float):
        pipeline = make_default_action_pipeline(False, sigma)
        return pipeline(state, None, keys[0], False, keys[1], keys[2]).env_action

    np.testing.assert_allclose(act(0.0), np.clip(mean, -1.0, 1.0), atol=1e-6)
    spread = np.std(np.asarray(act(0.3) - mean))
    assert 0.2 < spread < 0.4, spread


def test_the_target_noise_clip_reaches_the_update():
    assert (
        TD3(env_id="Pendulum-v1", target_noise_clip=0.2).agent_config.target_noise_clip
        == 0.2
    )
