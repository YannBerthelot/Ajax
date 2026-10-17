import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.extensions.expert import JSRLCurriculum
from ajax.networks.networks import get_initialized_actor_critic
from tests.agents.test_option_smoke import Expert, stateless_expert


def test_skip_critic_pretrain_is_a_transitional_no_op():
    """AjaxExperiments still passes True; False asks for the critic
    pretraining that was removed, so it raises instead of being ignored."""
    SAC(env_id="Pendulum-v1", skip_critic_pretrain=True)
    with pytest.raises(ValueError, match="MCPretrain"):
        SAC(env_id="Pendulum-v1", skip_critic_pretrain=False)


def test_the_gain_policy_refuses_the_experts_action_in_the_observation():
    """With use_pid_policy the actor's output sets the expert's gains, so the
    expert's action cannot be in the observation the actor reads."""
    kw = {"expert_policy": Expert(), "use_pid_policy": True}
    SAC(env_id="Pendulum-v1", augment_obs_with_expert_state=True, **kw)
    with pytest.raises(ValueError, match="augment_obs_with_expert_action"):
        SAC(env_id="Pendulum-v1", augment_obs_with_expert_action=True, **kw)


def test_fixed_alpha_keeps_temperature_at_alpha_init():
    """``fixed_alpha=True`` skips the temperature gradient step entirely.

    ``alpha_update_start=0`` and ``learning_starts=10`` guarantee that the
    temperature update *would* run for most of the 60 steps — the sibling
    run below confirms alpha does move once ``fixed_alpha`` is off — so an
    unchanged ``log_alpha`` proves the flag is honoured rather than the
    update never having been reached. ``alpha_init=0.2``, not 1.0: a
    ``log_alpha`` reset to zero would pass at 1.0.
    """
    alpha_init = 0.2
    common = {
        "env_id": "fast",
        "learning_starts": 10,
        "alpha_update_start": 0,
        "alpha_init": alpha_init,
        "alpha_learning_rate": 1e-2,
    }
    expected_log_alpha = jnp.log(alpha_init)

    fixed_state, _ = SAC(**common, fixed_alpha=True).train(seed=42, n_timesteps=60)
    assert jnp.allclose(fixed_state.alpha.params["log_alpha"], expected_log_alpha)

    learned_state, _ = SAC(**common, fixed_alpha=False).train(seed=42, n_timesteps=60)
    assert not jnp.allclose(
        learned_state.alpha.params["log_alpha"], expected_log_alpha
    ), "sanity check: alpha should move when it is not fixed"


def test_jsrl_curriculum_counts_the_episode_steps_it_reads():
    """JSRLCurriculum reads each env's step within its episode, which SAC
    counts because the extension declares it reads it: no SAC flag is
    needed. The deprecated ``jsrl_curriculum`` flag warns, and leaving it
    out stays quiet."""
    kw = {"expert_policy": stateless_expert, "expert_buffer_n_steps": 0}
    kw |= {"learning_starts": 8, "batch_size": 8, "buffer_size": 64}
    jsrl = (JSRLCurriculum(episode_length=8, decay_frac=0.5),)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        agent = SAC(env_id="Pendulum-v1", extensions=jsrl, **kw)
    state, _ = agent.train(seed=0, n_timesteps=12)
    np.testing.assert_array_equal(state.collector_state.step_in_episode, [[12]])
    plain, _ = SAC(env_id="Pendulum-v1", **kw).train(seed=0, n_timesteps=12)
    assert plain.collector_state.step_in_episode is None
    with pytest.warns(FutureWarning, match="jsrl_curriculum"):
        SAC(env_id="Pendulum-v1", jsrl_curriculum=True, extensions=jsrl, **kw)


def test_sac_starts_every_layer_xavier_uniform():
    """haarnoja/sac (tf.contrib's xavier_initializer, uniform at
    tensorflow 1.4) and softlearning (Keras Dense's glorot_uniform) start
    every layer Glorot-uniform with a zero bias, the policy's output
    included."""
    agent = SAC(env_id="Pendulum-v1")
    actor_state, critic_state = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
    )
    actor = actor_state.params["params"]
    critic = jax.tree.map(lambda x: x[0], critic_state.params["params"]["ensemble"])
    obs_dim, act_dim, width = 3, 1, 256  # Pendulum
    layers = {  # name: (params, fan_in, fan_out)
        "actor hidden 0": (actor["encoder"]["network"]["layers_0"], obs_dim, width),
        "actor hidden 1": (actor["encoder"]["network"]["layers_2"], width, width),
        "mean": (actor["mean"], width, act_dim),
        "log_std": (actor["log_std"], width, act_dim),
        "critic hidden 0": (
            critic["encoder"]["network"]["layers_0"],
            obs_dim + act_dim,
            width,
        ),
        "critic hidden 1": (critic["encoder"]["network"]["layers_2"], width, width),
        "Q": (critic["model"], width, 1),
    }
    for name, (layer, fan_in, fan_out) in layers.items():
        bound = np.sqrt(6 / (fan_in + fan_out))
        kernel = np.asarray(layer["kernel"])
        assert bound * 0.5 < np.abs(kernel).max() <= bound, name
        np.testing.assert_allclose(kernel.var(), bound**2 / 3, rtol=0.3, err_msg=name)
        np.testing.assert_array_equal(layer["bias"], 0.0, err_msg=name)


def test_recurrent_sac_keeps_its_previous_initialisation():
    """With memory, SAC keeps Ajax's previous initialisation until Ni et
    al.'s is audited: orthogonal(1.0) hidden layers, an orthogonal(0.01)
    mean and a zero-kernel log-std head starting at -1."""
    from ajax.networks.memory import MemoryConfig

    agent = SAC(env_id="Pendulum-v1", memory=MemoryConfig("gru", 16))
    actor_state, _ = get_initialized_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
    )
    actor = actor_state.params["params"]
    hidden = np.asarray(actor["encoder"]["network"]["layers_2"]["kernel"])
    np.testing.assert_allclose(hidden.T @ hidden, np.eye(256), atol=1e-4)
    mean = np.asarray(actor["mean"]["kernel"])
    np.testing.assert_allclose(mean.T @ mean, 1e-4 * np.eye(1), atol=1e-8)
    np.testing.assert_array_equal(actor["log_std"]["kernel"], 0.0)
    np.testing.assert_array_equal(actor["log_std"]["bias"], -1.0)
