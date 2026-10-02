"""End-to-end regression: SAC must be able to train on a wrapped
mujoco_playground env without the vmap-rank crash we saw on CheetahRun.

The bug (before this test): ``envs.py`` in the AjaxExperiments repo returned
raw mujoco_playground envs to the SAC training loop. Ajax's collection path
calls ``maybe_vmap(raw._get_obs, vmap_on)`` which requires a rank-1 batch
axis on the env state; raw Playground envs have rank-0 state and crash
immediately at ``interaction.py:398`` with:

    ValueError: vmap was requested to map its argument along axis 0, which
    implies that its rank should be at least 1, but is only 0

The fix was to go through ``ajax.environments.create.build_env_from_id``
which applies EpisodeWrapper + VmapWrapper + FinalObsWrapper +
BraxAutoResetWrapper + BatchRngWrapper. This test ensures that path
continues to work for one Playground env so the regression doesn't sneak
back in.
"""

import pytest


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)


@pytest.mark.slow
@requires_playground
def test_sac_trains_on_playground_env_smoke():
    """Build CheetahRun via ``build_env_from_id`` and run a few SAC training
    steps. The assertion is simply "does not raise" — covering the vmap-rank
    regression. Kept tiny (300 steps, 1 seed) so runtime stays under 30 s on
    CPU."""
    from ajax.agents.SAC.SAC import SAC
    from ajax.environments.create import build_env_from_id

    env, env_params = build_env_from_id("CheetahRun", n_envs=1, episode_length=200)
    agent = SAC(env_id=env, env_params=env_params)
    # Returns (state, metrics); we only care that it completes.
    result = agent.train(seed=[0], n_timesteps=300, num_episode_test=1)
    assert result is not None


@requires_playground
def test_playground_env_has_ajax_wrapper_stack():
    """``build_env_from_id`` must return a fully-wrapped env. Missing any
    wrapper in the stack manifests as cryptic vmap/rank errors deep inside
    SAC training; checking the outer type is a fast fail."""
    from ajax.environments.create import build_env_from_id
    from ajax.wrappers import BatchRngWrapper

    env, _ = build_env_from_id("CheetahRun", n_envs=1, episode_length=200)
    assert isinstance(env, BatchRngWrapper), (
        "Expected the outer wrapper to be BatchRngWrapper so SAC's "
        "unbatched-rng convention holds; got "
        f"{type(env).__name__}."
    )
    assert getattr(env, "_ajax_env_id", None) == "CheetahRun"


@pytest.mark.slow
@requires_playground
def test_sac_trains_from_fresh_initial_states():
    """With ``fresh_reset=True`` the episodes SAC stores in its replay buffer
    start from distinct observations. (Playground's cached auto-reset, the
    default, restarts every episode of an env from the same one; see
    tests/environments/test_fresh_auto_reset.py.)"""
    import numpy as np

    from ajax.agents.SAC.SAC import SAC
    from ajax.environments.create import build_env_from_id

    n_envs, episode_length, steps_per_env = 2, 5, 20  # 4 episodes per env
    env, env_params = build_env_from_id(
        "CartpoleBalance",
        n_envs=n_envs,
        episode_length=episode_length,
        fresh_reset=True,
    )
    agent = SAC(
        env_id=env,
        env_params=env_params,
        n_envs=n_envs,
        learning_starts=10,
        batch_size=8,
        buffer_size=100,
    )
    state, _ = agent.train(
        seed=[0], n_timesteps=steps_per_env * n_envs, num_episode_test=1
    )

    # Buffer leaves are (seed, env, time, ...); one seed here.
    experience = state.collector_state.buffer_state.experience
    obs = np.asarray(experience["obs"])[0, :, :steps_per_env]
    done = (
        np.asarray(experience["terminated"])[0, :, :steps_per_env, 0]
        + np.asarray(experience["truncated"])[0, :, :steps_per_env, 0]
    ) > 0
    for env_idx in range(n_envs):
        # An episode starts at t=0 and right after every done transition.
        starts = np.flatnonzero(np.concatenate([[True], done[env_idx, :-1]]))
        assert len(starts) == steps_per_env // episode_length
        start_obs = np.round(obs[env_idx, starts], 6)
        assert len(np.unique(start_obs, axis=0)) == len(starts), start_obs
