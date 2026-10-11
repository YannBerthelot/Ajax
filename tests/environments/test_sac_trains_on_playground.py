"""SAC end to end on mujoco_playground envs built by ``build_env_from_id``.

Regression: raw playground envs have rank-0 states, and SAC's collection
vmaps their observation function over the env axis
(``ajax.environments.interaction.get_raw_obs``), so SAC crashed at once
with

    ValueError: vmap was requested to map its argument along axis 0, which
    implies that its rank should be at least 1, but is only 0

until envs went through ``build_env_from_id``, which applies EpisodeWrapper
+ VmapWrapper + FinalObsWrapper + an auto-reset wrapper + BatchRngWrapper.
Both trainings below go through that stack.
"""

import pytest

#: 2-dim actions: the only multi-dimensional playground action space SAC
#: trains on here (CartpoleBalance's is 1-dim), with a small MJX model.
MULTI_DIM_ENV = "PointMass"


@pytest.mark.slow
def test_sac_updates_on_a_multi_dimensional_playground_env():
    """A tiny SAC past its learning starts (critic, policy, temperature):
    gradient updates run on 2-dim actions stored in its replay, and the
    parameters stay finite."""
    import jax
    import numpy as np

    from ajax.agents.SAC.SAC import SAC
    from ajax.environments.create import build_env_from_id

    env, env_params = build_env_from_id(MULTI_DIM_ENV, n_envs=1, episode_length=20)
    net = ("16", "relu")
    agent = SAC(
        env_id=env,
        env_params=env_params,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=10,
        policy_update_start=10,
        alpha_update_start=10,
        batch_size=8,
        buffer_size=100,
    )
    state, _ = agent.train(seed=[0], n_timesteps=40, num_episode_test=1)
    action = np.asarray(state.collector_state.buffer_state.experience["action"])
    assert action.shape[-1] == env.action_size == 2
    for train_state in (state.actor_state, state.critic_state):
        assert np.all(np.asarray(train_state.step) > 0)
    for leaf in jax.tree.leaves((state.actor_state.params, state.critic_state.params)):
        assert np.all(np.isfinite(leaf))


def test_playground_env_has_ajax_wrapper_stack():
    """``build_env_from_id`` must return a fully-wrapped env. Missing any
    wrapper in the stack manifests as cryptic vmap/rank errors deep inside
    SAC training; checking the outer type is a fast fail."""
    from ajax.environments.create import build_env_from_id
    from ajax.wrappers import BatchRngWrapper

    env, _ = build_env_from_id(MULTI_DIM_ENV, n_envs=1, episode_length=200)
    assert isinstance(env, BatchRngWrapper), (
        "Expected the outer wrapper to be BatchRngWrapper so SAC's "
        "unbatched-rng convention holds; got "
        f"{type(env).__name__}."
    )
    assert getattr(env, "_ajax_env_id", None) == MULTI_DIM_ENV


@pytest.mark.slow
def test_sac_trains_from_fresh_initial_states():
    """The episodes SAC stores in its replay buffer start from distinct
    observations with the default env stack. (Playground's cached
    auto-reset, ``fresh_reset=False``, restarts every episode of an env from
    the same one; see tests/environments/test_fresh_auto_reset.py.)"""
    import numpy as np

    from ajax.agents.SAC.SAC import SAC
    from ajax.environments.create import build_env_from_id

    n_envs, episode_length, steps_per_env = 2, 5, 20  # 4 episodes per env
    env, env_params = build_env_from_id(
        "CartpoleBalance", n_envs=n_envs, episode_length=episode_length
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
