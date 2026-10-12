import warnings

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


INERT = {
    "alpha_learning_rate": lambda: AVG("Pendulum-v1", alpha_learning_rate=1e-3),
    "target_entropy_per_dim": lambda: AVG("Pendulum-v1", target_entropy_per_dim=-1),
    "target_entropy": lambda: AVGConfig(gamma=0.99, target_entropy=-1.0),
}


@pytest.mark.parametrize("name", sorted(INERT))
def test_the_inert_temperature_keywords_warn(name):
    """AVG keeps its entropy coefficient at alpha_init (its reference code's
    fixed coefficient), so the keywords that would tune it do nothing:
    setting one warns, and leaving them out stays quiet."""
    with pytest.warns(FutureWarning, match=f"{name} has no effect"):
        INERT[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        AVG("Pendulum-v1")
        AVGConfig(gamma=0.99)
