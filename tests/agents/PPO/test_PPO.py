import jax
import numpy as np
import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.agents.PPO.train_PPO import _force_reset, _normalisation_info
from ajax.environments.create import strip_ajax_wrappers
from ajax.environments.interaction import reset
from ajax.networks.memory import MemoryConfig


def test_PPO_train_playground_smoke():
    """Smoke test: PPO trains briefly on a playground env (covers playground
    code path end-to-end including seed vmap and eval)."""
    agent = PPO(env_id="PendulumSwingup", n_envs=2)
    agent.train(seed=42, n_timesteps=50)


@pytest.mark.parametrize("env_id, mode", [("Pendulum-v1", "gymnax"), ("fast", "brax")])
def test_a_forced_reset_keeps_the_normaliser_statistics(env_id, mode):
    """num_resets_per_eval resets the envs, not the normaliser's running
    statistics (each gymnax env restarted its own from the reset): the reset
    observation is normalised with the statistics before it."""
    agent = PPO(
        env_id=env_id,
        n_envs=2,
        n_steps=8,
        normalize_observations=True,
        normalize_rewards=True,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=32)[0])
    saved = _normalisation_info(state.collector_state.env_state, mode)
    after = _force_reset(state, agent.env_args, mode).collector_state
    kept = _normalisation_info(after.env_state, mode)
    jax.tree.map(np.testing.assert_array_equal, kept, saved)

    key = jax.random.split(state.rng)[0]
    keys = jax.random.split(key, 2) if mode == "gymnax" else key
    task = strip_ajax_wrappers(agent.env_args.env)[0]
    raw, _ = reset(keys, task, mode, agent.env_args.env_params)
    std = np.sqrt(np.asarray(saved.obs.var) + 1e-8).mean(axis=0)
    want = (raw - np.asarray(saved.obs.mean).mean(axis=0)) / std
    np.testing.assert_allclose(after.last_obs, want.reshape(raw.shape), atol=1e-4)


def test_a_forced_reset_marks_an_episode_start_for_a_recurrent_policy():
    """A forced reset flags every env truncated, the flag a recurrent actor
    and critic restart their carry on (collection and the rollout's first
    reset flag). With ``num_evals=2`` the one reset falls on the last
    update; the run stays finite."""
    agent = PPO(
        env_id="Pendulum-v1",
        n_envs=2,
        n_steps=8,
        num_minibatches=2,
        bptt_length=4,
        memory=MemoryConfig(kind="gru", hidden_size=8),
        num_resets_per_eval=1,
        num_evals=2,
    )
    state, _ = agent.train(seed=0, n_timesteps=48)
    np.testing.assert_array_equal(state.collector_state.last_truncated, 1.0)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert np.isfinite(leaf).all()


def test_ppo_defaults_to_its_reference_log_std():
    """One learned log-std per action dimension, starting at 0 and never
    clipped, as Baselines, CleanRL and PureJaxRL learn it."""
    args = PPO("Pendulum-v1").network_args
    assert args.log_std_state_independent and args.log_std_init == 0.0
    assert args.log_std_bounds is None
