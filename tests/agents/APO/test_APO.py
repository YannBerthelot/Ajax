import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.APO.train_APO import init_APO
from ajax.agents.PPO.PPO import PPO
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
    mean layer; a large layer output (kernel 0, bias 5) stays inside
    [-1, 1]."""
    agent = APO(env_id="Pendulum-v1", n_envs=2)
    args = (agent.actor_optimizer_args, agent.critic_optimizer_args)
    state = init_APO(jax.random.PRNGKey(0), agent.env_args, *args, agent.network_args)
    actor = state.actor_state
    params = actor.params["params"]
    np.testing.assert_array_equal(params["log_std_param"], 0.0)
    mean = {
        "kernel": jnp.zeros_like(params["mean"]["kernel"]),
        "bias": jnp.full_like(params["mean"]["bias"], 5.0),
    }
    obs = jnp.zeros((3, 3))
    pi = actor.apply_fn({"params": {**params, "mean": mean}}, obs)
    assert type(pi) is distrax.Normal
    np.testing.assert_allclose(pi.stddev(), 1.0)
    np.testing.assert_allclose(pi.mean(), np.tanh(5.0), rtol=1e-6)


def test_APO_builds_its_reference_codes_hidden_layers():
    """APO's actor and critic are xtma/apo's MLPs (rlpyt's MujocoFfModel):
    two hidden layers of 64 tanh units each, starting as torch's nn.Linear
    (kernel and bias uniform in +-1/sqrt(fan_in)). The networks' outputs
    equal the mean and the value computed by hand from their parameters.
    PPO keeps its own default."""
    agent = APO(env_id="Pendulum-v1", n_envs=2)
    args = (agent.actor_optimizer_args, agent.critic_optimizer_args)
    state = init_APO(jax.random.PRNGKey(0), agent.env_args, *args, agent.network_args)
    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 3))

    def hidden(params):
        h, fan_in = obs, 3
        for name in ("layers_0", "layers_2"):
            layer = params["encoder"]["network"][name]
            for w in (layer["kernel"], layer["bias"]):
                assert w.shape[-1] == 64
                assert 0.8 / np.sqrt(fan_in) < np.abs(w).max() <= 1 / np.sqrt(fan_in)
            h, fan_in = jnp.tanh(h @ layer["kernel"] + layer["bias"]), 64
        return h

    actor = state.actor_state.params["params"]
    pi = state.actor_state.apply_fn({"params": actor}, obs)
    mean = jnp.tanh(hidden(actor) @ actor["mean"]["kernel"] + actor["mean"]["bias"])
    np.testing.assert_allclose(pi.mean(), mean, rtol=1e-5, atol=1e-6)
    critic = jax.tree.map(lambda x: x[0], state.critic_state.params["params"])
    critic = critic["ensemble"]  # one critic
    value = hidden(critic) @ critic["model"]["kernel"] + critic["model"]["bias"]
    out = state.critic_state.apply_fn(state.critic_state.params, obs)
    np.testing.assert_allclose(out[0], value, rtol=1e-5, atol=1e-6)
    ppo = PPO(env_id="Pendulum-v1")
    assert ppo.config["actor_architecture"] == ("128", "tanh", "128", "tanh")
    assert ppo.network_args.encoder_kernel_init is None
