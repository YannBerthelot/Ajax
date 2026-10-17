"""End-to-end tests for the UDRL agent."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.UDRL.train_UDRL import init_UDRL
from ajax.agents.UDRL.UDRL import UDRL
from ajax.environments.utils import get_state_action_shapes


def test_UDRL_train_updates_actor_params():
    """A short run moves the actor's parameters from the run's own initial
    ones (gradients flow), and the actor's observation carries the
    (d_r, d_h) command after the env's own columns."""
    agent = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=8, batch_size=8)
    state, _ = agent.train(seed=42, n_timesteps=32)
    # The run initialises on the first half of its seed's key (TrainLoop's
    # fresh_state); train(n_timesteps=0) would still run one iteration.
    initial = init_UDRL(
        key=jax.random.split(jax.random.PRNGKey(42))[0],
        env_args=agent.env_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        network_args=agent.network_args,
        agent_config=agent.agent_config,
    )

    def any_diff(a, b):
        leaves_a = jax.tree_util.tree_leaves(a)
        leaves_b = jax.tree_util.tree_leaves(b)
        return any(
            not jnp.allclose(la, lb) for la, lb in zip(leaves_a, leaves_b, strict=False)
        )

    assert any_diff(state.actor_state.params, initial.actor_state.params)
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    assert state.collector_state.last_obs.shape[-1] == obs_shape[0] + 2


def test_UDRL_logs_its_training_metrics_and_evaluation_without_a_backend():
    """With a logging config and no backend, ``train`` returns a row per
    log (every 32 env steps: 4 iterations of 8) of the training metrics and
    the shared evaluation of its policy; without one, nothing."""
    from ajax.logging.wandb_logging import LoggingConfig

    agent = UDRL(env_id="CartPole-v1", n_envs=1, n_steps=8, batch_size=8)
    config = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, rows = agent.train(seed=[0, 1], n_timesteps=64, logging_config=config)
    assert rows["timestep"].tolist() == [[32, 64], [32, 64]]
    finite = ("Train/actor_loss", "Eval/episodic mean reward")
    other = ("Train/episodic mean reward", "Eval/mean episodic length")
    assert set(finite + other) <= set(rows)
    for key in finite:
        assert jnp.isfinite(rows[key]).all(), key
    assert agent.train(seed=0, n_timesteps=64)[1] is None


def test_UDRL_logs_the_returns_of_whole_episodes():
    """CartPole pays 1 a step, so an episode returns its length; with 4-step
    segments its episodes span several. The logged training return is the
    house one (each env's mean of its last 10 whole episodes, averaged over
    envs), read here from the buffer, which holds the whole run."""
    from ajax.logging.wandb_logging import LoggingConfig

    agent = UDRL(env_id="CartPole-v1", n_envs=2, n_steps=4, batch_size=8)
    config = LoggingConfig(config={}, log_frequency=256, use_wandb=False)
    state, rows = agent.train(seed=[0], n_timesteps=256, logging_config=config)
    dones = np.asarray(state.buffer.dones)[0, :32, :, :, 0]  # (slot, t, env)
    lengths = [
        np.diff(np.flatnonzero(dones[:, :, env].reshape(-1)), prepend=-1)
        for env in range(2)
    ]
    means = [x[-10:].mean() for x in lengths]
    assert min(x.min() for x in lengths) > 4  # every episode spans segments
    np.testing.assert_allclose(
        rows["Train/episodic mean reward"][0, -1], np.mean(means)
    )


def test_UDRL_refuses_n_epochs():
    """UDRL takes ``n_updates_per_iter`` gradient steps per iteration
    (Algorithm 3); the ``n_epochs`` it once accepted and ignored is gone."""
    with pytest.raises(TypeError, match="n_epochs"):
        UDRL(env_id="Pendulum-v1", n_epochs=2)


def test_UDRL_gamma_discounts_only_its_diagnostic():
    """As udrl.md says: commands, top-K statistics and training targets sum
    rewards undiscounted, as in the paper, so gamma leaves training as it
    is and changes only the logged returns-to-go (Train/mean_rtg)."""
    from ajax.logging.wandb_logging import LoggingConfig

    logging = LoggingConfig(config={}, log_frequency=16, use_wandb=False)
    runs = {}
    for gamma in (1.0, 0.5):
        agent = UDRL(
            env_id="CartPole-v1", n_envs=1, n_steps=8, batch_size=8, gamma=gamma
        )
        runs[gamma] = agent.train(seed=0, n_timesteps=32, logging_config=logging)
    (plain, plain_logs), (discounted, discounted_logs) = runs[1.0], runs[0.5]
    for a, b in zip(
        jax.tree.leaves(plain.actor_state.params),
        jax.tree.leaves(discounted.actor_state.params),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    rtg = (
        np.asarray(plain_logs["Train/mean_rtg"]),
        np.asarray(discounted_logs["Train/mean_rtg"]),
    )
    assert np.isfinite(rtg[0]).any() and (rtg[1] < rtg[0])[np.isfinite(rtg[0])].all()
