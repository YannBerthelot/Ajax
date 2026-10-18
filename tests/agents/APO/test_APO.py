import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.APO.train_APO import init_APO
from ajax.state import LOG_STD_BOUNDS


def test_APO_rejects_minibatch_sizes_that_do_not_divide():
    """Without num_minibatches the rollout splits into max(n_steps,
    batch_size) // min(...) minibatches: the two must divide one another."""
    with pytest.raises(ValueError, match="must divide"):
        APO(env_id="Pendulum-v1", n_steps=48, batch_size=32)
    APO(env_id="Pendulum-v1", n_steps=48, batch_size=32, num_minibatches=2)


def test_APO_records_and_builds_its_log_std_bounds():
    """APO's default, no clip (its reference code's), is recorded in its run
    config and reaches its network config, as do explicit bounds;
    test_networks.py tests the clip."""
    agent = APO(env_id="Pendulum-v1")
    assert agent.config["log_std_bounds"] is None
    assert agent.network_args.log_std_bounds is None
    clipped = APO(env_id="Pendulum-v1", log_std_bounds=LOG_STD_BOUNDS)
    assert clipped.config["log_std_bounds"] == LOG_STD_BOUNDS
    assert clipped.network_args.log_std_bounds == LOG_STD_BOUNDS


def test_APO_builds_its_reference_codes_actor_head():
    """APO's initial actor is xtma/apo's (rlpyt's MujocoFfModel): an
    unsquashed Gaussian, std 1 in every dimension, whose mean is tanh of the
    mean layer; a large layer output (bias 5) stays inside [-1, 1]."""
    agent = APO(env_id="Pendulum-v1", n_envs=2)
    args = (agent.actor_optimizer_args, agent.critic_optimizer_args)
    state = init_APO(jax.random.PRNGKey(0), agent.env_args, *args, agent.network_args)
    actor = state.actor_state
    params = actor.params["params"]
    np.testing.assert_array_equal(params["log_std_param"], 0.0)
    mean = {**params["mean"], "bias": jnp.full_like(params["mean"]["bias"], 5.0)}
    obs = jnp.zeros((3, 3))
    pi = actor.apply_fn({"params": {**params, "mean": mean}}, obs)
    assert type(pi) is distrax.Normal
    np.testing.assert_allclose(pi.stddev(), 1.0)
    np.testing.assert_allclose(pi.mean(), np.tanh(5.0), rtol=1e-6)
