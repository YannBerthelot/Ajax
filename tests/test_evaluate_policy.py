"""evaluate_policy: the stateful-policy evaluation of the agents that own
their collection (DESIGN §5.4)."""

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.environments.create import build_env_from_id, prepare_env
from ajax.environments.interaction import get_pi
from ajax.environments.system_class import broadcast_env_params
from ajax.environments.utils import agent_episode_length
from ajax.evaluate import evaluate, evaluate_policy
from ajax.state import EnvironmentConfig, LoadedTrainState


def _playground_available():
    try:
        import mujoco_playground  # noqa: F401

        return True
    except ImportError:
        return False


requires_playground = pytest.mark.skipif(
    not _playground_available(), reason="mujoco_playground not installed"
)


class TinyActor(nn.Module):
    """Feedforward actor: Categorical (discrete) or Normal with a tanh mean."""

    n_out: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        h = nn.Dense(self.n_out)(obs)
        if self.discrete:
            return distrax.Categorical(logits=h)
        return distrax.Normal(jnp.tanh(h), jnp.ones_like(h))


def actor_state_for(obs_dim, n_out, discrete):
    module = TinyActor(n_out=n_out, discrete=discrete)
    params = module.init(jax.random.PRNGKey(1), jnp.zeros((1, obs_dim)))
    if discrete:
        # a slight state dependence on top of "push right": CartPole
        # episodes terminate early and at different steps
        params = jax.tree.map(lambda p: 0.05 * p, params)
        bias = params["params"]["Dense_0"]["bias"]
        params["params"]["Dense_0"]["bias"] = bias.at[1].add(1.0)
    return LoadedTrainState.create(
        apply_fn=module.apply, params=params, tx=optax.sgd(0.0)
    )


def deterministic_policy(actor_state, discrete):
    """The actor's mode / mean, as evaluate() acts, in the PolicyFn protocol."""

    def policy(carry, obs, is_first, key):
        del is_first, key
        pi, _ = get_pi(actor_state, actor_state.params, obs)
        return (pi.mode() if discrete else pi.mean()), carry, None

    return policy


@pytest.mark.parametrize(
    "env_id",
    [
        "CartPole-v1",
        "fast",
        pytest.param("CartpoleBalance", marks=requires_playground),
    ],
)
def test_evaluate_policy_matches_evaluate_for_a_feedforward_actor(env_id):
    num_episodes, key = 3, jax.random.PRNGKey(4)
    if env_id == "CartPole-v1":
        env, env_params = build_env_from_id(env_id)
        obs_dim, n_out, discrete = 4, 2, True
    else:
        # evaluate() rebuilds brax / playground envs at their native length
        # (1000 steps); bound its scan at the training length instead.
        env, env_params = build_env_from_id(env_id, episode_length=20)
        obs_dim, n_out, discrete = env.observation_size, env.action_size, False
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=not discrete
    )
    T = agent_episode_length(env, env_params, 1)
    actor_state = actor_state_for(obs_dim, n_out, discrete)

    mean_return, mean_length = jax.jit(
        lambda k: evaluate_policy(
            env_args,
            deterministic_policy(actor_state, discrete),
            lambda n: None,
            num_episodes,
            k,
        )
    )(key)
    ref_return, _, _, _, ref_length, _ = evaluate(
        env,
        actor_state=actor_state,
        num_episodes=num_episodes,
        rng=key,
        env_params=env_params,
        max_eval_steps=T,
    )
    np.testing.assert_allclose(mean_return, ref_return, rtol=1e-5)
    np.testing.assert_allclose(mean_length, ref_length, rtol=1e-5)
    if env_id == "CartPole-v1":
        assert float(mean_length) < T  # terminating episodes were masked


def ramp_policy(carry, obs, is_first, key):
    """Stateful: counts steps since is_first, ramps the action from -1 to 1."""
    del obs, key
    carry = jnp.where(is_first, 0, carry) + 1
    return jnp.clip(0.01 * carry - 1.0, -1.0, 1.0)[:, None], carry, None


def test_evaluate_policy_threads_a_stateful_policy_and_maps_bounds():
    """Pendulum (torque in [-2, 2], 200 steps) against a hand-rolled loop: a
    zero carry from init_carry_fn, is_first on the first step only, the
    carry threaded through the episode, actions mapped to the bounds."""
    env, env_params = gymnax.make("Pendulum-v1")
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=True
    )
    num_episodes, key = 2, jax.random.PRNGKey(0)
    mean_return, mean_length = jax.jit(
        lambda k: evaluate_policy(
            env_args,
            ramp_policy,
            lambda n: jnp.full((n,), 1000, jnp.int32),
            num_episodes,
            k,
        )
    )(key)

    _, reset_key = jax.random.split(key)
    _, states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, num_episodes), env_params
    )

    def step(states, t):
        torque = 2.0 * jnp.clip(0.01 * (t + 1) - 1.0, -1.0, 1.0)
        actions = jnp.full((num_episodes, 1), torque)
        _, states, reward, _, _, _ = jax.vmap(env.step, in_axes=(None, 0, 0, None))(
            jax.random.PRNGKey(0), states, actions, env_params
        )
        return states, reward

    _, rewards = jax.lax.scan(step, states, jnp.arange(200))
    assert float(mean_length) == 200.0
    np.testing.assert_allclose(mean_return, rewards.sum(0).mean(), rtol=1e-5)


@requires_playground
def test_evaluate_policy_uses_the_training_action_repeat_and_length():
    num_episodes, key = 2, jax.random.PRNGKey(2)
    env, _ = build_env_from_id(
        "CartpoleBalance", n_envs=num_episodes, episode_length=10, action_repeat=2
    )
    env_args = EnvironmentConfig(
        env=env, env_params=None, n_envs=num_episodes, continuous=True, action_repeat=2
    )

    def constant(carry, obs, is_first, key):
        return jnp.full((obs.shape[0], 1), 0.25), carry, None

    mean_return, mean_length = jax.jit(
        lambda k: evaluate_policy(env_args, constant, lambda n: None, num_episodes, k)
    )(key)
    assert float(mean_length) == 5.0  # 10 simulator steps at repeat 2
    # the training env, reset with the same key, earns the same return
    _, reset_key = jax.random.split(key)
    state = jax.jit(env.reset)(reset_key)
    step = jax.jit(env.step)
    total = 0.0
    for _ in range(5):
        state = step(state, jnp.full((num_episodes, 1), 0.25))
        total = total + np.asarray(state.reward)
    np.testing.assert_allclose(mean_return, total.mean(), rtol=1e-5)


def test_evaluate_policy_refuses_what_it_cannot_rebuild():
    """A normalising training env (the rebuild is the raw env) and per-env
    params (``num_episodes`` envs cannot be rebuilt from them)."""

    def policy(carry, obs, is_first, key):
        return jnp.ones(obs.shape[0], jnp.int32), carry, None

    env, env_params, _, _ = prepare_env(
        "CartPole-v1", n_envs=2, normalize_obs=True, normalize_reward=True, gamma=0.99
    )
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=2, continuous=False
    )
    with pytest.raises(ValueError, match="normalisation"):
        evaluate_policy(env_args, policy, lambda n: None, 2, jax.random.PRNGKey(0))
    env, env_params = gymnax.make("CartPole-v1")
    env_args = EnvironmentConfig(
        env=env,
        env_params=broadcast_env_params(env_params, 2),
        n_envs=2,
        continuous=False,
    )
    with pytest.raises(ValueError, match="one system"):
        evaluate_policy(env_args, policy, lambda n: None, 2, jax.random.PRNGKey(0))


def test_evaluate_rebuilds_the_eval_env_with_the_training_action_repeat():
    """The house evaluate() (every existing agent) reproduces the training
    env's action repeat: brax "fast" never terminates and evaluate()
    rebuilds it at its native 1000 simulator steps, so an episode lasts
    1000 agent steps at repeat 1 and 500 at repeat 2."""
    for repeat, expected_length in ((1, 1000.0), (2, 500.0)):
        env, _ = build_env_from_id("fast", episode_length=10, action_repeat=repeat)
        actor_state = actor_state_for(env.observation_size, env.action_size, False)
        *_, length, _ = evaluate(
            env,
            actor_state=actor_state,
            num_episodes=2,
            rng=jax.random.PRNGKey(0),
            env_params=None,
        )
        assert float(length) == expected_length
