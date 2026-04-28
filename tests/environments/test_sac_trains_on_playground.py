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

    env, env_params = build_env_from_id(
        "CheetahRun", n_envs=1, episode_length=200
    )
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
