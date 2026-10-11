"""End-to-end tests for the UDRL agent."""

import jax
import jax.numpy as jnp

from ajax.agents.UDRL.UDRL import UDRL


def test_UDRL_train_updates_actor_params():
    """A short training run should change actor parameters (sanity that gradients flow)."""
    agent = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=8, batch_size=8)
    state, _ = agent.train(seed=42, n_timesteps=32)
    # Compare to freshly-initialised params at the same seed.
    fresh = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=8, batch_size=8)
    fresh_state, _ = fresh.train(seed=42, n_timesteps=0)

    def any_diff(a, b):
        leaves_a = jax.tree_util.tree_leaves(a)
        leaves_b = jax.tree_util.tree_leaves(b)
        return any(
            not jnp.allclose(la, lb) for la, lb in zip(leaves_a, leaves_b, strict=False)
        )

    assert any_diff(state.actor_state.params, fresh_state.actor_state.params)


def test_UDRL_obs_dim_is_augmented():
    """Actor's last_obs in the collector should have width = env obs_dim + command_dim (=2)."""
    from ajax.environments.utils import get_state_action_shapes

    agent = UDRL(env_id="Pendulum-v1", n_envs=1, n_steps=4, batch_size=4)
    state, _ = agent.train(seed=42, n_timesteps=0)

    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    raw_obs_dim = obs_shape[0]
    assert state.collector_state.last_obs.shape[-1] == raw_obs_dim + 2


def test_UDRL_logs_its_training_metrics_without_a_backend():
    """With a logging config and no backend, ``train`` returns a row per
    log (every 32 env steps: 4 iterations of 8) of the training metrics;
    without one, nothing."""
    from ajax.logging.wandb_logging import LoggingConfig

    agent = UDRL(env_id="CartPole-v1", n_envs=1, n_steps=8, batch_size=8)
    config = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, rows = agent.train(seed=[0, 1], n_timesteps=64, logging_config=config)
    assert rows["timestep"].tolist() == [[32, 64], [32, 64]]
    assert {"Train/actor_loss", "Train/episodic mean reward"} <= set(rows)
    assert jnp.isfinite(rows["Train/actor_loss"]).all()
    assert agent.train(seed=0, n_timesteps=64)[1] is None
