import numpy as np

from ajax.agents.AVG.AVG import AVG


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


def test_avg_keeps_its_temperature_at_alpha_init():
    """AVG keeps its entropy coefficient at alpha_init, as its reference code
    does (gauthamvasan/avg, avg.py): eight updates leave the temperature
    exactly where initialisation put it. Both values come out of ``train``,
    so the bitwise check never meets a log computed elsewhere, which may
    differ in the last bit."""
    alpha_init = 0.5  # not the default, so a dropped alpha_init shows
    agent = AVG(
        "Pendulum-v1",
        alpha_init=alpha_init,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
    )

    def log_alpha(n_timesteps: int) -> np.ndarray:
        state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
        return np.asarray(state.alpha.params["log_alpha"])

    initial, trained = log_alpha(0), log_alpha(8)
    np.testing.assert_allclose(np.exp(initial), alpha_init, rtol=1e-6)
    np.testing.assert_array_equal(trained, initial)
