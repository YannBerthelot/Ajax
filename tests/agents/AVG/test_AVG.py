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
