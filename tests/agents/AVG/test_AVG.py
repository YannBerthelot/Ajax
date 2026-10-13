import pytest

from ajax.agents.AVG.AVG import AVG
from ajax.agents.AVG.state import AVGConfig


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
