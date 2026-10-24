import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brax.envs import create as create_brax_env
from flax import linen as nn

from ajax.environments.create import build_env_from_id
from ajax.environments.utils import get_state_action_shapes
from ajax.networks.memory import MemoryConfig, init_carry
from ajax.networks.networks import (
    Actor,
    Critic,
    Encoder,
    MultiCritic,
    Squashed,
    get_initialized_actor_critic,
    get_initialized_critic,
    init_network_state,
    predict_value,
)
from ajax.state import (
    LOG_STD_BOUNDS,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)


@pytest.fixture
def sample_input():
    return jnp.ones((4, 8))  # Example input with batch size 4 and feature size 8


#: The actor's and the critics' hidden layers.
ARCH = ["16", "relu"]


def test_encoder(sample_input):
    encoder = Encoder(input_architecture=["16", "relu", "8"])
    params = encoder.init(jax.random.PRNGKey(0), sample_input)
    output = encoder.apply(params, sample_input)
    assert output.shape == (4, 8)


def test_actor_continuous(sample_input):
    actor = Actor(action_dim=4, input_architecture=ARCH, continuous=True)
    params = actor.init(jax.random.PRNGKey(0), sample_input)
    distribution = actor.apply(params, sample_input)
    assert distribution.mean().shape == (4, 4)
    assert distribution.stddev().shape == (4, 4)


def test_actor_discrete(sample_input):
    actor = Actor(action_dim=4, input_architecture=ARCH, continuous=False)
    params = actor.init(jax.random.PRNGKey(0), sample_input)
    distribution = actor.apply(params, sample_input)
    assert distribution.logits.shape == (4, 4)


def test_critic(sample_input):
    critic = Critic(input_architecture=ARCH)
    x = jnp.hstack([sample_input, jnp.ones((4, 2))])  # an action of size 2
    params = critic.init(jax.random.PRNGKey(0), x)
    assert critic.apply(params, x).shape == (4, 1)


@pytest.mark.parametrize("num_critics", [1, 3])
def test_multi_critic(sample_input, num_critics):
    multi_critic = MultiCritic(input_architecture=ARCH, num=num_critics)
    x = jnp.hstack([sample_input, jnp.ones((4, 2))])  # an action of size 2
    params = multi_critic.init(jax.random.PRNGKey(0), x)
    assert multi_critic.apply(params, x).shape == (num_critics, 4, 1)


@pytest.mark.parametrize("n_quantiles", [1, 5])
def test_a_critics_value_is_the_mean_of_its_quantiles(sample_input, n_quantiles):
    """A distributional critic (TQC's) has ``n_quantiles`` outputs per
    member: what every value query reads is their mean, and
    ``predict_value(quantiles=True)`` the outputs themselves; with one, the
    value is the output."""
    critic = MultiCritic(input_architecture=ARCH, num=3, n_quantiles=n_quantiles)
    x = jnp.hstack([sample_input, jnp.ones((4, 2))])  # an action of size 2
    state = init_network_state(x, critic, jax.random.PRNGKey(0), optax.sgd(0.0))
    atoms = predict_value(state, state.params, x, quantiles=True)
    values = predict_value(state, state.params, x)
    assert atoms.shape == (3, 4, n_quantiles) and values.shape == (3, 4, 1)
    np.testing.assert_allclose(values, atoms.mean(-1, keepdims=True), rtol=1e-6)
    if n_quantiles == 1:
        np.testing.assert_array_equal(values, atoms)


@pytest.mark.parametrize("multi", [False, True])
def test_the_layernorm_token_adds_a_layernorm_where_it_stands(multi):
    """RLPD's critic MLP (Ball et al. 2023): each ``"layernorm"`` is its own
    LayerNorm (a scale and a bias) right after its Dense, in a Critic and
    in each member of a MultiCritic."""
    arch = ("16", "layernorm", "relu", "16", "layernorm", "relu")
    critic = MultiCritic(arch, num=2) if multi else Critic(arch)
    params = critic.init(jax.random.PRNGKey(0), jnp.ones((4, 10)))["params"]
    layers = (params["ensemble"] if multi else params)["encoder"]["network"]
    dense, norm = ["bias", "kernel"], ["bias", "scale"]
    expected = {
        "layers_0": dense,
        "layers_1": norm,
        "layers_3": dense,
        "layers_4": norm,
    }
    assert {name: sorted(leaves) for name, leaves in layers.items()} == expected


@pytest.fixture
def real_env_config():
    env, env_params = build_env_from_id("CartPole-v1")
    return EnvironmentConfig(env=env, env_params=env_params, n_envs=1, continuous=False)


@pytest.fixture
def fast_env_config():
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(env=env, env_params=None, n_envs=1, continuous=True)


def _init_ac(env_config, continuous, action_value, key=0):
    """Two critics and an actor of :data:`ARCH` on ``env_config``, and its
    observation and action shapes."""
    optimizer_config = OptimizerConfig(learning_rate=0.001)
    actor_state, critic_state = get_initialized_actor_critic(
        key=jax.random.PRNGKey(key),
        env_config=env_config,
        actor_optimizer_config=optimizer_config,
        critic_optimizer_config=optimizer_config,
        network_config=NetworkConfig(actor_architecture=ARCH, critic_architecture=ARCH),
        continuous=continuous,
        action_value=action_value,
        num_critics=2,
    )
    assert isinstance(actor_state, LoadedTrainState)
    assert isinstance(critic_state, LoadedTrainState)
    return actor_state, critic_state, get_state_action_shapes(env_config.env)


def test_get_initialized_actor_critic(real_env_config):
    """A discrete actor and two Q critics on CartPole: the shapes of their
    outputs, and ``predict_value`` of the two critics (on non-zero inputs:
    zero-initialised biases make them agree on zero ones), finite and
    distinct."""
    actor_state, critic_state, (obs_shape, action_shape) = _init_ac(
        real_env_config, continuous=False, action_value=True
    )
    n = real_env_config.n_envs
    init_obs = jnp.zeros((n, *obs_shape))
    init_action = jnp.zeros((n, *action_shape))
    assert actor_state.apply_fn(actor_state.params, init_obs).mode().shape == (n,)
    critic_output = critic_state.apply_fn(
        critic_state.params, jnp.hstack([init_obs, init_action])
    )
    assert critic_output.shape == (2, n, 1)

    ones = jnp.hstack([jnp.ones((n, *obs_shape)), jnp.ones((n, *action_shape))])
    predicted = predict_value(critic_state, critic_state.params, ones)
    assert predicted.shape == (2, n, 1), "Shape mismatch in predicted value."
    assert jnp.all(jnp.isfinite(predicted)), "Predicted value is not finite."
    assert not jnp.allclose(predicted[0], predicted[1]), "The critics are identical."


def test_the_builder_initialises_as_the_network_config_says(fast_env_config):
    """The initializers and the log-std head set on NetworkConfig reach the
    actor and the critic (PPO's, SAC's and UDRL's build through here)."""
    optimizer_config = OptimizerConfig(learning_rate=0.001)
    network_config = NetworkConfig(
        actor_architecture=ARCH,
        critic_architecture=ARCH,
        encoder_kernel_init="zeros",
        actor_bias_init="0.25",
        mean_kernel_init="zeros",
        critic_bias_init="0.5",
        log_std_state_independent=True,
        log_std_init=0.3,
        log_std_bounds=(-1.0, 1.0),
        tanh_mean=True,
    )
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        fast_env_config,
        optimizer_config,
        optimizer_config,
        network_config,
        continuous=True,
        num_critics=1,
    )
    actor, critic = actor_state.params["params"], critic_state.params["params"]
    for encoder in (actor["encoder"], critic["ensemble"]["encoder"]):
        assert not jnp.any(encoder["network"]["layers_0"]["kernel"])
    assert not jnp.any(actor["mean"]["kernel"])
    np.testing.assert_allclose(actor["mean"]["bias"], 0.25)
    np.testing.assert_allclose(actor["log_std_param"], 0.3)
    np.testing.assert_allclose(critic["ensemble"]["model"]["bias"], 0.5)
    beyond = {
        "params": {**actor, "log_std_param": jnp.full_like(actor["mean"]["bias"], 3.0)}
    }
    obs = jnp.zeros((1, get_state_action_shapes(fast_env_config.env)[0][-1]))
    pi = actor_state.apply_fn(beyond, obs)
    np.testing.assert_allclose(pi.stddev(), np.exp(1.0), rtol=1e-6)
    np.testing.assert_allclose(pi.mean(), np.tanh(0.25), rtol=1e-6)


def test_get_initialized_actor_critic_continuous(fast_env_config):
    actor_state, critic_state, (obs_shape, action_shape) = _init_ac(
        fast_env_config, continuous=True, action_value=False, key=1
    )
    n = fast_env_config.n_envs
    init_obs = jnp.zeros((n, *obs_shape))
    actor_output = actor_state.apply_fn(actor_state.params, init_obs)
    assert actor_output.mean().shape == (n, *action_shape)
    assert actor_output.stddev().shape == (n, *action_shape)
    critic_output = critic_state.apply_fn(critic_state.params, init_obs)
    assert critic_output.shape == (2, n, 1)


def test_a_standalone_critic_is_feedforward_even_with_memory(real_env_config):
    """``get_initialized_critic`` builds a feedforward MultiCritic, so its
    state must not be initialised as recurrent when the config sets memory."""
    critic_state = get_initialized_critic(
        key=jax.random.PRNGKey(0),
        env_config=real_env_config,
        critic_optimizer_config=OptimizerConfig(learning_rate=0.001),
        network_config=NetworkConfig(
            actor_architecture=ARCH,
            critic_architecture=ARCH,
            memory=MemoryConfig(kind="gru", hidden_size=4),
        ),
    )
    assert not critic_state.recurrent and critic_state.hidden_state is None
    obs_shape, action_shape = get_state_action_shapes(real_env_config.env)
    x = jnp.ones((3, obs_shape[-1] + action_shape[-1]))
    assert predict_value(critic_state, critic_state.params, x).shape == (2, 3, 1)


def test_encoder_penultimate_normalization():
    input_architecture = ["4", "relu", "2"]
    encoder = Encoder(
        input_architecture=input_architecture, penultimate_normalization=True
    )

    key = jax.random.PRNGKey(0)
    input_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    params = encoder.init(key, input_data)
    output = encoder.apply(params, input_data)

    # Check that the output is normalized to a unit vector
    norms = jnp.linalg.norm(output, axis=1)
    assert jnp.allclose(norms, 1.0), "Output is not normalized to a unit vector"


def test_encoder_penultimate_normalization_gradients():
    input_architecture = ["4", "relu", "2"]
    encoder = Encoder(
        input_architecture=input_architecture, penultimate_normalization=True
    )

    key = jax.random.PRNGKey(0)
    input_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    params = encoder.init(key, input_data)

    def loss_fn(params, input_data):
        output = encoder.apply(params, input_data)
        return jnp.sum(output)  # Example loss function

    grads = jax.grad(loss_fn)(params, input_data)

    # Check that gradients exist for the parameters of the encoder
    assert grads is not None, "Gradients are not flowing through the encoder."
    assert "params" in grads, "Gradients for parameters are missing."
    assert all(
        jnp.any(jnp.abs(g) > 0) for g in jax.tree_util.tree_leaves(grads["params"])
    ), "Some gradients are zero: none flow through the L2 normalisation."


def test_penultimate_normalization_with_memory_keeps_envs_apart(real_env_config):
    """With memory the encoder sees time-major (T, B, F) features. Each
    env's are normalised on their own, so an env's output does not depend
    on which other envs share its batch."""
    memory = MemoryConfig(kind="gru", hidden_size=4)
    optimizer_config = OptimizerConfig(learning_rate=0.001)
    actor_state, _ = get_initialized_actor_critic(
        key=jax.random.PRNGKey(0),
        env_config=real_env_config,
        actor_optimizer_config=optimizer_config,
        critic_optimizer_config=optimizer_config,
        network_config=NetworkConfig(
            actor_architecture=ARCH,
            critic_architecture=ARCH,
            memory=memory,
            penultimate_normalization=True,
        ),
        continuous=False,
        action_value=False,
    )
    obs_shape, _ = get_state_action_shapes(real_env_config.env)
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, 2, *obs_shape))

    def logits(obs):
        batch = obs.shape[1]
        dist, _ = actor_state.apply_fn(
            actor_state.params,
            obs,
            hidden_state=init_carry(memory, jax.random.PRNGKey(2), batch),
            done=jnp.zeros((3, batch), dtype=bool),
        )
        return dist.logits

    assert jnp.allclose(logits(obs)[:, :1], logits(obs[:, :1]), atol=1e-6)


def _soft_nets(agent_cls, init, **n_critics):
    def build(a, key):
        state = init(
            key,
            a.env_args,
            a.actor_optimizer_args,
            a.critic_optimizer_args,
            a.network_args,
            a.alpha_args,
            a.buffer,
            **n_critics,
        )
        return state.actor_state.params, state.critic_state.params

    agent = agent_cls("Pendulum-v1", buffer_size=64, batch_size=8, learning_starts=8)
    return agent, build


def _sac_nets():
    from ajax.agents.SAC.SAC import SAC
    from ajax.agents.SAC.train_SAC import init_SAC

    return _soft_nets(SAC, init_SAC, num_critics=2)


def _redq_nets():
    from ajax.agents.REDQ.REDQ import REDQ
    from ajax.agents.REDQ.train_REDQ import init_REDQ

    return _soft_nets(REDQ, init_REDQ, number_of_critics=10)


def _ppo_nets():
    from ajax.agents.PPO.PPO import PPO

    def build(a, key):
        state = a.init(key, a.context(1)[1])
        return state.actor_state.params, state.critic_state.params

    return PPO("Pendulum-v1"), build


def _dqn_nets():
    from ajax.agents.DQN.DQN import DQN
    from ajax.agents.DQN.train_DQN import init_DQN

    def build(a, key):
        state = init_DQN(key, a.env_args, a.critic_optimizer_args, a.network_args, 2)
        return state.actor_state.params

    return DQN("CartPole-v1"), build


@pytest.mark.parametrize("nets", [_sac_nets, _redq_nets, _ppo_nets, _dqn_nets])
def test_paper_agents_build_plain_mlps_unless_layer_norm_is_opted_in(nets):
    """SAC, REDQ, PPO and DQN use plain MLPs in their papers: by default
    their networks hold Dense kernels and biases only (and PPO's log-std
    parameter); the network option ``encoder_layer_norm`` adds the
    encoder's output LayerNorm."""
    agent, build = nets()

    def leaf_names():
        params = build(agent, jax.random.PRNGKey(0))
        paths = jax.tree_util.tree_flatten_with_path(params)[0]
        return {"/".join(str(getattr(k, "key", k)) for k in p) for p, _ in paths}

    plain = leaf_names()
    leaves = {name.rsplit("/", 1)[-1] for name in plain} - {"log_std_param"}
    assert leaves == {"kernel", "bias"}, plain
    assert not any("norm" in name for name in plain), plain

    agent.network_args = agent.network_args.replace(encoder_layer_norm=True)
    normed = leaf_names()
    assert plain < normed and all("/norm/" in n for n in normed - plain), normed


@pytest.mark.parametrize("state_independent", [False, True])
def test_the_actor_std_starts_at_exp_log_std_init(state_independent):
    """Either log-std head, a Dense with a zero kernel over the features
    (SAC's) or one parameter per action dimension (PPO's), starts every
    std at exp(log_std_init), whatever the input."""
    actor = Actor(
        action_dim=3,
        input_architecture=ARCH,
        continuous=True,
        log_std_state_independent=state_independent,
        log_std_init=-0.5,
    )
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = actor.init(jax.random.PRNGKey(0), obs)["params"]
    head, other = ("log_std_param", "log_std")
    if not state_independent:
        head, other = other, head
    assert head in params and other not in params
    std = actor.apply({"params": params}, obs).stddev()
    np.testing.assert_allclose(std, np.full((4, 3), np.exp(-0.5)), rtol=1e-6)


@pytest.mark.parametrize("bounds", [LOG_STD_BOUNDS, (-1.0, 0.5), None])
@pytest.mark.parametrize("state_independent", [False, True])
def test_log_std_bounds_clip_either_log_std_head(state_independent, bounds):
    """The state-independent parameter and SAC's Dense head are both clipped
    to ``log_std_bounds``, or not at all when None."""
    actor = Actor(
        action_dim=2,
        input_architecture=ARCH,
        continuous=True,
        log_std_state_independent=state_independent,
        log_std_bounds=bounds,
    )
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = actor.init(jax.random.PRNGKey(0), obs)["params"]
    log_std = jnp.array([3.0, -25.0])  # beyond either bound
    if state_independent:
        params = {**params, "log_std_param": log_std}
    else:
        params = {**params, "log_std": {**params["log_std"], "bias": log_std}}
    std = actor.apply({"params": params}, obs).stddev()
    want = np.clip(np.asarray(log_std), *(bounds or (-np.inf, np.inf)))
    np.testing.assert_allclose(std, np.broadcast_to(np.exp(want), (4, 2)), rtol=1e-5)


def test_a_log_variance_head_takes_the_root_after_the_clip():
    """``log_variance_head`` reads the head as a log-variance, as AVG's code
    (avg.py:70-72): std = exp(clip(out) / 2), the bounds bounding the
    log-variance."""
    kw = {"action_dim": 2, "input_architecture": ARCH, "continuous": True}
    actor = Actor(log_variance_head=True, log_std_bounds=(-20.0, 2.0), **kw)
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = actor.init(jax.random.PRNGKey(0), obs)["params"]
    head = jnp.array([3.0, -1.0])  # the first beyond the bound
    params = {**params, "log_std": {**params["log_std"], "bias": head}}
    std = actor.apply({"params": params}, obs).stddev()
    want = np.exp(np.clip(np.asarray(head), -20.0, 2.0) / 2)
    np.testing.assert_allclose(std, np.broadcast_to(want, (4, 2)), rtol=1e-5)


@pytest.mark.parametrize("bounds", [(2.0, -20.0), (1.0, 1.0)])
def test_the_network_refuses_log_std_bounds_out_of_order(bounds):
    """Where the network is built, not only in an agent's constructor: a
    network config, replaced or new, and an Actor refuse swapped or equal
    bounds, which would clip every log-std to the high one."""
    config = NetworkConfig(actor_architecture=ARCH, critic_architecture=ARCH)
    with pytest.raises(ValueError, match="low < high"):
        config.replace(log_std_bounds=bounds)
    actor = Actor(
        action_dim=2, input_architecture=ARCH, continuous=True, log_std_bounds=bounds
    )
    with pytest.raises(ValueError, match="low < high"):
        actor.init(jax.random.PRNGKey(0), jnp.zeros((1, 8)))


def test_jax_rebuilds_a_network_config_around_any_leaves():
    """tree.map rebuilds the config through its bounds check, which reads
    the static bounds, not the new leaves."""
    config = NetworkConfig(actor_architecture=ARCH, critic_architecture=ARCH)
    rebuilt = jax.tree.map(lambda _: None, config)
    assert rebuilt.log_std_bounds == LOG_STD_BOUNDS


def test_tanh_mean_bounds_the_mean_not_the_samples():
    """``tanh_mean`` (APO's reference head) passes the mean layer through
    tanh and leaves the Gaussian unsquashed: a sample beyond [-1, 1] has the
    Normal's log-probability around tanh of the layer. Its parameters are
    the tanh-squashed head's, whose deterministic action, tanh of the same
    layer, it keeps."""
    kw = {"action_dim": 2, "input_architecture": ARCH, "continuous": True}
    kw |= {"log_std_state_independent": True, "log_std_init": 0.0}
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = Actor(**kw).init(jax.random.PRNGKey(0), obs)
    squashed = Squashed(Actor(**kw))
    init = squashed.init(jax.random.PRNGKey(0), obs)
    assert jax.tree.structure(init) == jax.tree.structure(params)
    mean = {**params["params"]["mean"], "bias": jnp.array([3.0, -3.0])}
    params = {"params": {**params["params"], "mean": mean}}
    layer = Actor(**kw).apply(params, obs).mean()
    pi = Actor(tanh_mean=True, **kw).apply(params, obs)
    assert type(pi) is distrax.Normal
    np.testing.assert_allclose(pi.mean(), np.tanh(layer), rtol=1e-6)
    np.testing.assert_allclose(squashed.apply(params, obs).mean(), pi.mean(), 1e-6)
    x = jnp.full((4, 2), 1.7)
    want = jax.scipy.stats.norm.logpdf(x, np.tanh(layer), 1.0)
    np.testing.assert_allclose(pi.log_prob(x), want, rtol=1e-5)


def test_mean_kernel_init_sets_the_initial_mean_head(sample_input):
    """The mean head starts orthogonal with gain 0.01 (a mean near 0);
    ``mean_kernel_init`` replaces it, by name or as an initializer."""

    def kernel(**kwargs):
        actor = Actor(action_dim=3, input_architecture=ARCH, continuous=True, **kwargs)
        params = actor.init(jax.random.PRNGKey(0), sample_input)
        return np.asarray(params["params"]["mean"]["kernel"])

    default = kernel()
    np.testing.assert_allclose(default.T @ default, 1e-4 * np.eye(3), atol=1e-9)
    lecun = np.abs(kernel(mean_kernel_init="lecun_uniform"))
    assert 0.05 < lecun.max() <= np.sqrt(3 / 16)  # fan_in 16
    np.testing.assert_array_equal(kernel(mean_kernel_init=nn.initializers.ones), 1.0)


def test_logits_head_starts_orthogonal_with_gain_001(sample_input):
    """The discrete head starts orthogonal with gain 0.01, PPO's reference
    init, so the initial policy is near uniform."""
    actor = Actor(action_dim=3, input_architecture=ARCH, continuous=False)
    params = actor.init(jax.random.PRNGKey(0), sample_input)
    kernel = np.asarray(params["params"]["model"]["layers_0"]["kernel"])
    np.testing.assert_allclose(kernel.T @ kernel, 1e-4 * np.eye(3), atol=1e-9)
    probs = actor.apply(params, sample_input).probs
    np.testing.assert_allclose(probs, 1 / 3, atol=0.01)


@pytest.mark.parametrize("continuous", [False, True])
def test_the_builder_sets_every_output_layers_kernel(
    continuous, real_env_config, fast_env_config
):
    """logits_kernel_init, log_std_kernel_init and value_kernel_init on
    NetworkConfig reach the logits, the state-dependent log-std and the
    value layers."""
    optimizer_config = OptimizerConfig(learning_rate=0.001)
    network_config = NetworkConfig(
        actor_architecture=ARCH,
        critic_architecture=ARCH,
        logits_kernel_init="zeros",
        log_std_kernel_init="1.0",
        value_kernel_init="zeros",
    )
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        fast_env_config if continuous else real_env_config,
        optimizer_config,
        optimizer_config,
        network_config,
        continuous=continuous,
        num_critics=1,
    )
    actor, critic = actor_state.params["params"], critic_state.params["params"]
    if continuous:
        np.testing.assert_array_equal(actor["log_std"]["kernel"], 1.0)
    else:
        assert not jnp.any(actor["model"]["layers_0"]["kernel"])
    assert not jnp.any(critic["ensemble"]["model"]["kernel"])


def test_output_layers_keep_their_defaults_or_start_as_torch_nn_linear(
    sample_input,
):
    """By default the logits start orthogonal with gain 0.01, the value layer
    with gain 1 and the log-std head with a zero kernel and log_std_init as
    its bias; "torch_linear" draws each kernel and bias in
    +-1/sqrt(fan_in)."""

    def actor_params(continuous, **kwargs):
        actor = Actor(
            action_dim=3, input_architecture=ARCH, continuous=continuous, **kwargs
        )
        return actor.init(jax.random.PRNGKey(0), sample_input)["params"]

    def value_params(**kwargs):
        critic = Critic(input_architecture=ARCH, **kwargs)
        return critic.init(jax.random.PRNGKey(0), sample_input)["params"]["model"]

    logits = np.asarray(actor_params(False)["model"]["layers_0"]["kernel"])
    np.testing.assert_allclose(logits.T @ logits, 1e-4 * np.eye(3), atol=1e-9)
    log_std = actor_params(True)["log_std"]
    assert not jnp.any(log_std["kernel"])
    np.testing.assert_allclose(log_std["bias"], -1.0)
    value = np.asarray(value_params()["kernel"])
    np.testing.assert_allclose(value.T @ value, [[1.0]], atol=1e-5)

    torch_layers = [
        actor_params(False, logits_kernel_init="torch_linear")["model"]["layers_0"],
        actor_params(True, log_std_kernel_init="torch_linear", log_std_init=None)[
            "log_std"
        ],
        value_params(value_kernel_init="torch_linear"),
    ]
    bound = 1 / np.sqrt(16)  # fan_in: ARCH's 16 units
    for layer in torch_layers:
        for name in ("kernel", "bias"):
            w = np.abs(np.asarray(layer[name]))
            assert 0.2 * bound < w.max() <= bound, name


@pytest.mark.parametrize(
    "state_independent, log_std_init, log_std_kernel_init",
    [
        (False, -1.0, "torch_linear"),  # the bias set twice
        (False, None, None),  # the bias set by nothing
        (False, None, "zeros"),
        (True, None, None),  # the parameter needs a start
        (True, 0.0, "zeros"),  # the parameter has no kernel
    ],
)
def test_the_log_std_head_start_is_set_exactly_once(
    sample_input, state_independent, log_std_init, log_std_kernel_init
):
    actor = Actor(
        action_dim=3,
        input_architecture=ARCH,
        continuous=True,
        log_std_state_independent=state_independent,
        log_std_init=log_std_init,
        log_std_kernel_init=log_std_kernel_init,
    )
    with pytest.raises(ValueError, match="log_std_init"):
        actor.init(jax.random.PRNGKey(0), sample_input)
