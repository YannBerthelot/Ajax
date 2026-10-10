"""Environment plumbing for the world-model agents: action repeat, episode
length in agent steps, the eval rebuild, and the action-bound mapping."""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments import spaces

from ajax.agents.base import ActorCritic
from ajax.environments import create
from ajax.environments.create import (
    build_env_from_id,
    prepare_env,
    strip_ajax_wrappers,
)
from ajax.environments.system_class import broadcast_env_params
from ajax.environments.utils import (
    agent_action_to_env,
    agent_episode_length,
    env_action_repeat,
)
from ajax.evaluate import setup_environment


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)


# ---------------------------------------------------------------------------
# Construction rules
# ---------------------------------------------------------------------------


def test_gymnax_envs_reject_action_repeat():
    with pytest.raises(NotImplementedError, match="gymnax"):
        build_env_from_id("Pendulum-v1", action_repeat=2)
    with pytest.raises(NotImplementedError, match="gymnax"):
        prepare_env("Pendulum-v1", action_repeat=2)
    env, params = gymnax.make("Pendulum-v1")
    with pytest.raises(NotImplementedError, match="gymnax"):
        setup_environment(env, params, 2, None, 0.99, action_repeat=2)


def test_prebuilt_envs_reject_action_repeat():
    env, _ = gymnax.make("CartPole-v1")
    with pytest.raises(ValueError, match="built from an id"):
        prepare_env(env, action_repeat=2)
    # the default is unchanged
    assert prepare_env(env)[0] is env
    # a prebuilt env that already repeats is refused too, even with the
    # default action_repeat: the agent would record a repeat of 1 and the
    # eval rebuild would not match the env
    repeating, _ = build_env_from_id(
        "fast", n_envs=2, episode_length=10, action_repeat=2
    )
    with pytest.raises(ValueError, match="repeats each action 2 times"):
        prepare_env(repeating, n_envs=2)
    with pytest.raises(ValueError, match="built from an id"):
        ActorCritic(env_id=repeating, n_envs=2)
    unrepeated, _ = build_env_from_id("fast", n_envs=2, episode_length=10)
    prepared = prepare_env(unrepeated, n_envs=2)[0]  # only its action clip on top
    assert strip_ajax_wrappers(prepared) == (unrepeated, {})


def test_actor_critic_builds_the_env_with_action_repeat():
    agent = ActorCritic(env_id="fast", n_envs=1, episode_length=10, action_repeat=2)
    assert agent.env_args.action_repeat == 2
    assert env_action_repeat(agent.env_args.env) == 2
    assert agent_episode_length(agent.env_args.env, None, 2) == 5
    assert ActorCritic(env_id="fast", n_envs=1).env_args.action_repeat == 1
    with pytest.raises(NotImplementedError, match="gymnax"):
        ActorCritic(env_id="Pendulum-v1", n_envs=1, action_repeat=2)


def test_action_repeat_must_be_a_positive_int():
    with pytest.raises(ValueError, match="positive"):
        build_env_from_id("fast", action_repeat=0)


def test_registered_builders_get_action_repeat_only_when_above_one(monkeypatch):
    calls = []

    def legacy(n_envs, episode_length):
        calls.append(("legacy", n_envs, episode_length))
        return "legacy-env"

    def modern(n_envs, episode_length, action_repeat=1):
        calls.append(("modern", n_envs, episode_length, action_repeat))
        return create._build_brax_env("fast", n_envs, episode_length, action_repeat)

    def ignores_it(n_envs, episode_length, **extra):  # takes it, drops it
        return create._build_brax_env("fast", n_envs, episode_length)

    def hardcoded(n_envs, episode_length):  # a fixed repeat of its own
        return create._build_brax_env("fast", n_envs, episode_length, 2)

    monkeypatch.setitem(create._BRAX_BUILDERS, "ajax_test_legacy", legacy)
    monkeypatch.setitem(create._PLAYGROUND_BUILDERS, "ajax_test_modern", modern)
    monkeypatch.setitem(create._BRAX_BUILDERS, "ajax_test_ignores", ignores_it)
    monkeypatch.setitem(create._BRAX_BUILDERS, "ajax_test_hardcoded", hardcoded)

    # repeat 1: legacy builders are called exactly as before
    assert build_env_from_id("ajax_test_legacy", n_envs=2, episode_length=7) == (
        "legacy-env",
        None,
    )
    assert calls[-1] == ("legacy", 2, 7)
    # repeat > 1: a builder that cannot take it raises ...
    with pytest.raises(ValueError, match="does not accept"):
        build_env_from_id("ajax_test_legacy", episode_length=8, action_repeat=2)
    # ... one that can, receives it
    env, _ = build_env_from_id(
        "ajax_test_modern", n_envs=3, episode_length=8, action_repeat=2
    )
    assert calls[-1] == ("modern", 3, 8, 2)
    assert env_action_repeat(env) == 2
    # the built env decides: a builder that accepts the keyword but does not
    # apply it raises; one whose env already repeats as asked is accepted
    with pytest.raises(ValueError, match="repeats each action 1 times"):
        build_env_from_id("ajax_test_ignores", episode_length=8, action_repeat=2)
    env, _ = build_env_from_id("ajax_test_hardcoded", episode_length=8, action_repeat=2)
    assert env_action_repeat(env) == 2


# ---------------------------------------------------------------------------
# Episode length in agent steps
# ---------------------------------------------------------------------------


def test_agent_episode_length_gymnax():
    env, params = gymnax.make("Pendulum-v1")
    assert agent_episode_length(env, params, 1) == 200
    assert agent_episode_length(env, None, 1) == 200  # the env's own params
    assert env_action_repeat(env) == 1
    # per-env (batched) params: one length shared by every system
    batched = broadcast_env_params(params, 3)
    assert agent_episode_length(env, batched, 1) == 200
    disagreeing = batched.replace(max_steps_in_episode=jnp.array([200, 100, 200]))
    with pytest.raises(ValueError, match="disagree"):
        agent_episode_length(env, disagreeing, 1)
    env, params = gymnax.make("CartPole-v1")
    assert agent_episode_length(env, params) == 500


def test_agent_episode_length_brax_counts_agent_steps():
    env, _ = build_env_from_id("fast", episode_length=10, action_repeat=2)
    assert env.episode_length == 10  # simulator steps
    assert agent_episode_length(env, None, 2) == 5
    # the repeat must be the env's own
    with pytest.raises(ValueError, match="repeats each action 2 times"):
        agent_episode_length(env, None, 1)
    with pytest.raises(ValueError, match="positive"):
        agent_episode_length(env, None, 0)
    env, _ = build_env_from_id("fast", episode_length=11, action_repeat=2)
    with pytest.raises(ValueError, match="multiple"):
        agent_episode_length(env, None, 2)


@requires_playground
def test_agent_episode_length_playground_dmc_protocol():
    """The paper DMC protocol: 1000 simulator steps at repeat 2 -> T = 500."""
    env, _ = build_env_from_id("CartpoleBalance", episode_length=1000, action_repeat=2)
    assert agent_episode_length(env, None, 2) == 500


# ---------------------------------------------------------------------------
# Repeat semantics and the eval rebuild
# ---------------------------------------------------------------------------


def rollout(env, n_steps, action, key=0):
    state = jax.jit(env.reset)(jax.random.PRNGKey(key))
    step = jax.jit(env.step)
    rewards, dones = [], []
    for _ in range(n_steps):
        state = step(state, action)
        rewards.append(np.asarray(state.reward))
        dones.append(np.asarray(state.done))
    return state, np.stack(rewards), np.stack(dones)


def check_repeat_and_rebuild(env_id, n_envs, episode_length, action):
    """Repeat 2 vs repeat 1 (summed rewards, same states, T agent steps) and
    the eval rebuild with the training repeat and length (same rollout)."""
    repeated, _ = build_env_from_id(
        env_id, n_envs=n_envs, episode_length=episode_length, action_repeat=2
    )
    single, _ = build_env_from_id(env_id, n_envs=n_envs, episode_length=episode_length)
    T = agent_episode_length(repeated, None, 2)
    assert T == episode_length // 2
    state_r, rewards_r, dones_r = rollout(repeated, T, action)
    state_1, rewards_1, _ = rollout(single, 2 * T, action)
    # one agent step == two simulator steps with the same action
    np.testing.assert_allclose(
        rewards_r, rewards_1.reshape(T, 2, n_envs).sum(1), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        state_r.info["final_obs"], state_1.info["final_obs"], rtol=1e-5, atol=1e-6
    )
    # the episode lasts T agent steps: truncated on the last one only
    assert not dones_r[:-1].any() and dones_r[-1].all()
    assert np.asarray(state_r.info["truncation"]).all()

    eval_env, mode, _ = setup_environment(
        repeated,
        None,
        n_envs,
        None,
        0.99,
        action_repeat=2,
        episode_length=repeated.episode_length,
    )
    assert mode == "brax"
    assert eval_env.episode_length == episode_length and eval_env.action_repeat == 2
    assert agent_episode_length(eval_env, None, 2) == T
    _, rewards_eval, dones_eval = rollout(eval_env, T, action)
    np.testing.assert_allclose(rewards_eval, rewards_r, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(dones_eval, dones_r)


def test_brax_action_repeat_and_eval_rebuild():
    # "fast": deterministic, never terminates, reward = position (grows)
    check_repeat_and_rebuild("fast", 2, 10, jnp.ones((2, 1)))
    # the defaults keep the historical rebuild: 1000 steps, no repeat
    env, _ = build_env_from_id("fast", episode_length=10, action_repeat=2)
    legacy, _, _ = setup_environment(env, None, 2, None, 0.99)
    assert legacy.action_repeat == 1 and legacy.episode_length == 1000


@requires_playground
def test_playground_action_repeat_and_eval_rebuild():
    check_repeat_and_rebuild("CartpoleBalance", 3, 10, jnp.full((3, 1), 0.3))
    env, _ = build_env_from_id("CartpoleBalance", episode_length=10, action_repeat=2)
    legacy, _, _ = setup_environment(env, None, 3, None, 0.99)
    # native length from the env's config, no repeat
    assert legacy.action_repeat == 1 and legacy.episode_length == 1000


# ---------------------------------------------------------------------------
# Action bounds
# ---------------------------------------------------------------------------


def test_actions_map_to_pendulum_bounds():
    env, params = gymnax.make("Pendulum-v1")  # torque in [-2, 2]
    a = jnp.array([[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]])
    expected = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [2.0]])
    np.testing.assert_allclose(agent_action_to_env(a, env, params), expected)
    # also with params traced inside jit (bounds unknown at trace time)
    mapped = jax.jit(lambda p, a: agent_action_to_env(a, env, p))(params, a)
    np.testing.assert_allclose(mapped, expected)


def test_unit_bounds_only_clip_and_discrete_actions_pass_through():
    """[-1, 1] bounds (gymnax MountainCarContinuous, every brax env): the
    identity on in-range actions, out-of-range ones clipped (DreamerV3's
    ClipAction)."""
    a = jnp.array([[1.5], [-0.25], [-3.0]])
    clipped = np.array([[1.0], [-0.25], [-1.0]])
    env, params = gymnax.make("MountainCarContinuous-v0")
    np.testing.assert_array_equal(agent_action_to_env(a, env, params), clipped)
    env, _ = build_env_from_id("fast")
    np.testing.assert_array_equal(agent_action_to_env(a, env, None), clipped)
    env, params = gymnax.make("CartPole-v1")
    discrete = jnp.array([0, 1])
    assert agent_action_to_env(discrete, env, params) is discrete


def test_per_env_params_map_actions_to_each_systems_bounds():
    """Batched gymnax params (one system per env): env e's action is mapped
    with its own bounds, not broadcast against the others'."""
    env, params = gymnax.make("Pendulum-v1")
    params = broadcast_env_params(params, 3).replace(
        max_torque=jnp.array([1.0, 2.0, 3.0])
    )
    a = jnp.array([[1.0], [1.0], [-0.5]])
    expected = np.array([[1.0], [2.0], [-1.5]])
    np.testing.assert_allclose(agent_action_to_env(a, env, params), expected)
    mapped = jax.jit(lambda p, a: agent_action_to_env(a, env, p))(params, a)
    np.testing.assert_allclose(mapped, expected)
    with pytest.raises(ValueError, match="Per-env params for 3 envs"):
        agent_action_to_env(a[:2], env, params)


class _HalfBoundedEnv:
    """A gymnax-like env whose second action dimension is unbounded."""

    def action_space(self, params=None):
        return spaces.Box(
            low=jnp.array([0.0, -jnp.inf]), high=jnp.array([4.0, jnp.inf]), shape=(2,)
        )


def test_unbounded_action_dimensions_are_only_clipped():
    a = jnp.array([[0.5, 0.3], [-3.0, -7.0]])
    out = agent_action_to_env(a, _HalfBoundedEnv(), None)
    np.testing.assert_allclose(out, [[3.0, 0.3], [0.0, -1.0]])
