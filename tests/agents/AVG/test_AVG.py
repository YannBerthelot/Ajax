import jax
import numpy as np
import pytest

from ajax.agents.AVG.AVG import AVG
from ajax.agents.AVG.state import AVGConfig
from ajax.networks.networks import get_initialized_actor_critic


def test_avg_keeps_its_network_optimizer_and_env_settings():
    """AVG builds on ActorCritic, then sets what the paper uses: squashed
    actions with penultimate normalisation, Adam with ``beta_1 = 0`` and
    torch's eps (gauthamvasan/avg @6b6822c avg.py:116-117), normalised
    observations, no memory."""
    agent = AVG(env_id="Pendulum-v1", beta_1=0.0, beta_2=0.99)
    assert agent.network_args.squash and agent.network_args.penultimate_normalization
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.beta_1, args.beta_2, args.eps) == (0.0, 0.99, 1e-8)
    assert "NormalizeVecObservation" in type(agent.env_args.env._env).__name__
    assert not AVG.supports_memory
    assert agent.get_make_train().keywords["alpha_args"] is agent.alpha_args


# Each call passes a keyword the signature no longer has, on purpose.
REMOVED = {
    "alpha_learning_rate": lambda: AVG("Pendulum-v1", alpha_learning_rate=1e-3),  # type: ignore[call-arg]
    "target_entropy_per_dim": lambda: AVG("Pendulum-v1", target_entropy_per_dim=-1),  # type: ignore[call-arg]
    "target_entropy": lambda: AVGConfig(gamma=0.99, target_entropy=-1.0),  # type: ignore[call-arg]
}


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_avg_takes_no_temperature_keywords(name):
    """AVG keeps its entropy coefficient at alpha_init (its reference code's
    fixed coefficient), so it has no keyword that would tune it."""
    with pytest.raises(TypeError, match=name):
        REMOVED[name]()
    assert AVG("Pendulum-v1").alpha_args.learning_rate == 0.0


def _is_orthogonal(kernel) -> bool:
    """Columns (or rows, when the layer widens less than it reads)
    orthonormal: an orthogonal matrix of gain 1."""
    k = np.asarray(kernel)
    gram = k.T @ k if k.shape[0] >= k.shape[1] else k @ k.T
    return np.allclose(gram, np.eye(len(gram)), atol=1e-4)


def test_avg_starts_every_layer_as_its_reference_code():
    """gauthamvasan/avg's orthogonal_weight_init (avg.py:16-20) starts every
    Linear layer orthogonal with gain 1 and a zero bias, the mean and the
    log-std heads included (:59-62) and the Q layer (:97-98)."""
    agent = AVG(env_id="Pendulum-v1")
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=1,
    )
    actor = actor_state.params["params"]
    critic = jax.tree.map(lambda x: x[0], critic_state.params["params"]["ensemble"])
    layers = {
        "actor hidden 0": actor["encoder"]["network"]["layers_0"],
        "actor hidden 1": actor["encoder"]["network"]["layers_2"],
        "mean": actor["mean"],
        "log_std": actor["log_std"],
        "critic hidden 0": critic["encoder"]["network"]["layers_0"],
        "critic hidden 1": critic["encoder"]["network"]["layers_2"],
        "Q": critic["model"],
    }
    for name, layer in layers.items():
        assert _is_orthogonal(layer["kernel"]), name
        np.testing.assert_array_equal(layer["bias"], 0.0, err_msg=name)
