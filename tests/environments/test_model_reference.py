"""Tests for the model-reference tracking wrapper and its building blocks."""

import jax
import jax.numpy as jnp
import pytest
from gymnax import make as make_gymnax_env

from ajax.environments.differentiable import (
    closed_loop_rollout,
    with_transition_gradients,
)
from ajax.environments.interaction import reset, step
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.system_class import UniformPerturbation

N = 50


def angle(obs):
    """Pendulum angle recovered from (cos, sin)."""
    return jnp.arctan2(obs[1], obs[0])


@pytest.fixture
def tracking_pendulum():
    env, params = make_gymnax_env("Pendulum-v1")
    reference = StepReference(
        horizon=N, min_value=-1.0, max_value=1.0, min_duration=5, max_duration=15
    )
    model = LinearReferenceModel.first_order()
    wrapped = ModelReferenceWrapper(env, reference, model, output_fn=angle)
    return wrapped, params


# ------------------------------------------------------------ StepReference


def test_step_reference_validation():
    with pytest.raises(ValueError, match="horizon"):
        StepReference(0, 0.0, 1.0, 1, 2)
    with pytest.raises(ValueError, match="min_duration"):
        StepReference(10, 0.0, 1.0, 5, 2)


def test_step_reference_shape_range_and_segment_lengths():
    ref = StepReference(
        horizon=200,
        min_value=2.0,
        max_value=3.0,
        min_duration=4,
        max_duration=9,
        n_outputs=2,
    )
    seq = ref.sample(jax.random.PRNGKey(0))
    assert seq.shape == (200, 2)
    assert jnp.all(seq >= 2.0) and jnp.all(seq <= 3.0)
    # segment lengths (excluding the possibly truncated last one) within bounds
    for ch in range(2):
        change = jnp.flatnonzero(jnp.diff(seq[:, ch]) != 0) + 1
        starts = jnp.concatenate([jnp.zeros(1, jnp.int32), change])
        lengths = jnp.diff(starts)
        assert len(lengths) > 3
        assert jnp.all(lengths >= 4) and jnp.all(lengths <= 9)


def test_step_reference_is_deterministic_under_key():
    ref = StepReference(N, -1.0, 1.0, 5, 15)
    assert jnp.array_equal(
        ref.sample(jax.random.PRNGKey(1)), ref.sample(jax.random.PRNGKey(1))
    )
    assert not jnp.array_equal(
        ref.sample(jax.random.PRNGKey(1)), ref.sample(jax.random.PRNGKey(2))
    )


# ----------------------------------------------------- LinearReferenceModel


def test_first_order_model_matches_paper_recurrence_and_unit_dc_gain():
    m = LinearReferenceModel.first_order()
    x = jnp.zeros(1)
    r = jnp.ones(1)
    x1 = m.step(x, r)
    assert jnp.allclose(x1, 0.7143)
    assert jnp.allclose(m.output(x1, r), 0.5669 * 0.7143 + 0.2914)
    for _ in range(200):
        x = m.step(x, r)
    assert jnp.allclose(m.output(x, r), 1.0, atol=1e-4)  # unit DC gain


def test_model_init_state_reproduces_initial_output():
    m = LinearReferenceModel.first_order(n_outputs=2)
    y0 = jnp.array([0.3, -0.7])
    r0 = jnp.array([1.0, 2.0])
    x0 = m.init_state(y0, r0)
    assert jnp.allclose(m.output(x0, r0), y0, atol=1e-6)


# ------------------------------------------------------ ModelReferenceWrapper


def test_wrapper_rejects_output_mismatch():
    env, _ = make_gymnax_env("Pendulum-v1")
    ref = StepReference(N, -1.0, 1.0, 5, 15, n_outputs=2)
    with pytest.raises(ValueError, match="n_outputs"):
        ModelReferenceWrapper(env, ref, LinearReferenceModel.first_order())


def test_wrapper_reset_observation_layout(tracking_pendulum):
    env, params = tracking_pendulum
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    assert obs.shape == (2,)  # [e, u_prev]
    assert env.observation_space(params).shape == (2,)
    raw = env._env.get_obs(state.env_state)
    assert jnp.allclose(obs[0], state.reference[0, 0] - angle(raw))
    assert obs[1] == 0.0 and state.t == 0
    # desired output starts at the plant's output
    assert jnp.allclose(
        env.model.output(state.model_state, state.reference[0]), angle(raw), atol=1e-6
    )


def test_wrapper_with_raw_obs_appends_plant_observation():
    env, params = make_gymnax_env("Pendulum-v1")
    ref = StepReference(N, -1.0, 1.0, 5, 15)
    w = ModelReferenceWrapper(
        env,
        ref,
        LinearReferenceModel.first_order(),
        output_fn=angle,
        include_raw_obs=True,
    )
    obs, state = w.reset(jax.random.PRNGKey(0), params)
    assert obs.shape == (5,) and w.observation_space(params).shape == (5,)
    assert jnp.allclose(obs[2:], env.get_obs(state.env_state))


def test_wrapper_step_bookkeeping_and_reward(tracking_pendulum):
    env, params = tracking_pendulum
    key = jax.random.PRNGKey(3)
    obs0, s0 = env.reset(key, params)
    action = jnp.array([0.7])
    obs1, s1, reward, term, trunc, info = env.step(key, s0, action, params)
    raw1 = env._env.get_obs(s1.env_state)
    y1 = angle(raw1)
    x1 = env.model.step(s0.model_state, s0.reference[0])
    y_d1 = env.model.output(x1, s0.reference[1])
    assert s1.t == 1
    assert jnp.allclose(s1.model_state, x1)
    assert jnp.allclose(reward, -((y_d1 - y1) ** 2).sum())
    assert jnp.allclose(obs1, jnp.array([s0.reference[1, 0] - y1, 0.7]))
    assert jnp.array_equal(s1.reference, s0.reference)
    assert jnp.allclose(info["y_desired"], y_d1) and jnp.allclose(info["y"], y1)
    # get_obs reproduces the observation returned by step
    assert jnp.allclose(env.get_obs(s1, params), obs1)


def test_wrapper_resamples_task_when_plant_episode_ends(tracking_pendulum):
    env, params = tracking_pendulum
    params = params.replace(max_steps_in_episode=2)
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key, params)
    ref0 = s.reference
    for i in range(2):
        _, s, _, _, trunc, _ = env.step(
            jax.random.fold_in(key, i), s, jnp.zeros(1), params
        )
    assert bool(trunc)
    assert s.t == 0 and jnp.all(s.u_prev == 0)
    assert not jnp.array_equal(s.reference, ref0)


def test_wrapper_holds_last_reference_past_horizon(tracking_pendulum):
    env, params = tracking_pendulum
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key, params)
    s = s.replace(t=jnp.asarray(N - 1, jnp.int32))
    _, s, _, _, _, _ = env.step(key, s, jnp.zeros(1), params)
    assert s.t == N - 1


def test_wrapper_vmaps_with_batched_params(tracking_pendulum):
    env, params = tracking_pendulum
    B = 5
    batched = UniformPerturbation(params, fields=("m",), scale=0.2).sample(
        jax.random.PRNGKey(1), B
    )
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    obs, state = reset(keys, env, "gymnax", batched)
    assert obs.shape == (B, 2)
    obs2, state2, reward, term, trunc, _ = step(
        keys, state, jnp.zeros((B, 1)), env, "gymnax", batched
    )
    assert obs2.shape == (B, 2) and reward.shape == (B,)
    assert jnp.all(state2.t == 1)


def test_tracking_objective_is_differentiable_end_to_end(tracking_pendulum):
    env, params = tracking_pendulum
    env = with_transition_gradients(env)
    B, T = 3, 20

    def objective(gain):
        def policy(carry, obs, resets):
            return gain * obs[:, :1], carry  # proportional on the tracking error

        rollout, _ = closed_loop_rollout(
            policy, None, jax.random.PRNGKey(0), env, params, T, n_envs=B
        )
        return rollout.reward.sum()

    g = jax.grad(objective)(jnp.asarray(0.5))
    assert jnp.isfinite(g) and g != 0
