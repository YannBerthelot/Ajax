"""The presets of ``benchmarks/agent_bench.py`` time training, not only
collection: each one updates within the bench's run. Read from the agents'
configs; the bench itself is not run here (its timings need a quiet
machine)."""

import agent_bench
import jax.numpy as jnp
import pytest

from ajax.agents.loop import TrainLoop


def updates_in_run(agent, n_envs: int, timesteps: int) -> int:
    """The updates a run of ``timesteps`` makes, from the agent's config."""
    schedule = getattr(agent, "schedule", None)
    if schedule is not None and hasattr(schedule, "total_updates"):  # DreamerV3
        return schedule.total_updates(timesteps // n_envs)
    if schedule is not None:  # TD-MPC2's seed phase, burst, then per tick
        ticks = jnp.arange(schedule.num_ticks(timesteps))
        return int(schedule.n_updates(ticks).sum())
    config = agent.agent_config
    if hasattr(config, "learning_starts"):
        # One update per iteration (n_envs steps) once the collected steps
        # reach learning_starts (TrainLoop.maybe_update).
        iterations = timesteps // n_envs
        first = max(1, -(-config.learning_starts // n_envs))
        return max(0, iterations - first + 1)
    if hasattr(config, "n_steps"):  # one update per rollout
        loop = TrainLoop.create(agent.env_args, timesteps, num_episode_test=0)
        return loop.n_rollouts(config.n_steps)
    # APG: one update per closed-loop rollout of ``horizon`` steps
    # (train_APG.make_train's num_updates).
    return max(timesteps // (n_envs * config.horizon), 1)


@pytest.mark.parametrize("name", sorted(agent_bench.AGENTS))
def test_every_preset_updates_within_the_run(name):
    env_id, kwargs = agent_bench.AGENTS[name]
    agent = agent_bench._build(name, env_id, kwargs, agent_bench.N_ENVS)
    n_updates = updates_in_run(agent, agent_bench.N_ENVS, agent_bench.TIMESTEPS)
    assert n_updates > 0, f"{name} would time collection only"
