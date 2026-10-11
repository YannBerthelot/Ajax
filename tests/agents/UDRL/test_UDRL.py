"""End-to-end tests for the UDRL agent."""

import jax
import jax.numpy as jnp

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
