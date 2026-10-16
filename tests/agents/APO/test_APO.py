import pytest

from ajax.agents.APO.APO import APO


def test_APO_rejects_minibatch_sizes_that_do_not_divide():
    """Without num_minibatches the rollout splits into max(n_steps,
    batch_size) // min(...) minibatches: the two must divide one another."""
    with pytest.raises(ValueError, match="must divide"):
        APO(env_id="Pendulum-v1", n_steps=48, batch_size=32)
    APO(env_id="Pendulum-v1", n_steps=48, batch_size=32, num_minibatches=2)


def test_APO_clips_its_log_std_to_its_configured_bounds():
    """APO's default bounds, SAC's [-20, 2], are recorded in its run config
    and reach its network, as does None."""
    agent = APO(env_id="Pendulum-v1")
    assert agent.config["log_std_bounds"] == (-20.0, 2.0)
    assert agent.network_args.log_std_bounds == (-20.0, 2.0)
    unclipped = APO(env_id="Pendulum-v1", log_std_bounds=None)
    assert unclipped.network_args.log_std_bounds is None
