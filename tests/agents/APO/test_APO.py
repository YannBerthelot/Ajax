import pytest

from ajax.agents.APO.APO import APO


def test_APO_rejects_minibatch_sizes_that_do_not_divide():
    """Without num_minibatches the rollout splits into max(n_steps,
    batch_size) // min(...) minibatches: the two must divide one another."""
    with pytest.raises(ValueError, match="must divide"):
        APO(env_id="Pendulum-v1", n_steps=48, batch_size=32)
    APO(env_id="Pendulum-v1", n_steps=48, batch_size=32, num_minibatches=2)
