import warnings
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brax.envs import create as create_brax_env
from gymnax import make as make_gymnax_env
from gymnax.environments.classic_control.pendulum import EnvParams as PendulumParams
from gymnax.environments.environment import EnvState as GymnaxEnvState

import ajax.environments.interaction as step_module
from ajax.agents.SAC.SAC import SAC
from ajax.buffers.utils import get_buffer
from ajax.controller import ActingState
from ajax.environments.create import with_episode_keys
from ajax.environments.interaction import (
    collect_experience,
    get_action_and_new_agent_state,
    get_pi,
    init_collector_state,
    init_rolling_mean,
    reset,
    step,
)
from ajax.extensions.expert import ExpertGains, ExpertWarmup
from ajax.networks.distributions import Squash
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    LoadedTrainState,
)
from ajax.wrappers import ObservationNoiseWrapper

n_envs = 4


@pytest.fixture
def gymnax_env():
    env, env_params = make_gymnax_env("CartPole-v1")
    return env, env_params


@pytest.fixture
def brax_env():
    # Create a Brax environment with 4 parallel environments
    env = create_brax_env("fast", batch_size=n_envs)
    return env


@pytest.fixture
def mock_actor_state_continuous():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: distrax.Normal(
            jnp.zeros(n_envs), jnp.ones(n_envs)
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


@pytest.fixture
def mock_actor_state_continuous_squashed():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: Squash(
            distrax.Normal(jnp.zeros((obs.shape[0], 1)), jnp.ones((obs.shape[0], 1)))
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


@pytest.fixture
def mock_actor_state_discrete():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: distrax.Categorical(
            probs=jnp.ones((n_envs, 4)) / 4
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


@pytest.fixture
def mock_recurrent_actor_state():
    """Fixture to create a mock recurrent actor state."""
    tx = optax.adam(learning_rate=0.001)

    def apply_fn(params, obs, hidden_state=None, done=None):
        # Simulate a recurrent network output. Real recurrent networks
        # consume time-major (T, B, obs) and return a distribution with
        # (T, B, ...) batch shape — mirror that so the single-step time
        # axis handling in get_action_and_new_agent_state is exercised.
        new_hidden_state = hidden_state + 1 if hidden_state is not None else None
        pi = distrax.Categorical(probs=jnp.ones((*obs.shape[:-1], 4)) / 4)

        return pi, new_hidden_state

    return LoadedTrainState.create(
        apply_fn=apply_fn,
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
        hidden_state=jnp.zeros((n_envs, 2)),  # Initialize hidden state
    )


@pytest.fixture(params=["gymnax", "brax"])
def env_and_shape(request):
    """``(env, env_params, mode, reset keys, observation shape)``: gymnax
    CartPole reset per env, or brax ``fast`` batched over its envs."""
    key = jax.random.PRNGKey(0)
    if request.param == "gymnax":
        env, env_params = request.getfixturevalue("gymnax_env")
        keys = jax.random.split(key, n_envs)
        return env, env_params, "gymnax", keys, env.obs_shape
    env = request.getfixturevalue("brax_env")
    return env, None, "brax", key, (env.observation_size,)


def test_reset_then_scanned_steps_keep_the_batch_shapes(env_and_shape):
    """``reset`` and ``step`` (under ``jax.lax.scan``) of either API: every
    output has one row per env."""
    env, env_params, mode, rng, obs_shape = env_and_shape
    obs, env_state = reset(rng, env, mode=mode, env_params=env_params)
    assert obs.shape == (n_envs, *obs_shape)
    if mode == "gymnax":
        assert isinstance(env_state, GymnaxEnvState)
    else:
        assert hasattr(env_state, "obs")

    def scan_step(carry, _):
        rng, env_state = carry
        action = jnp.zeros((n_envs,))  # Dummy actions
        obs, env_state, reward, terminated, truncated, info = step(
            rng, env_state, action, env, mode, env_params
        )
        return (rng, env_state), (obs, reward, terminated, truncated)

    _, (obs_seq, reward_seq, terminated_seq, truncated_seq) = jax.lax.scan(
        scan_step, (rng, env_state), None, length=10
    )
    assert obs_seq.shape == (10, n_envs, *obs_shape)
    assert reward_seq.shape == (10, n_envs)
    assert terminated_seq.shape == (10, n_envs)
    assert truncated_seq.shape == (10, n_envs)


@pytest.mark.parametrize(
    "mock_actor_state, expected_distribution",
    [
        ("mock_actor_state_discrete", distrax.Categorical),
        ("mock_actor_state_continuous", distrax.Normal),
        ("mock_actor_state_continuous_squashed", Squash),
    ],
)
def test_get_pi(mock_actor_state, expected_distribution, request):
    """Test policy distribution generation for different actor states."""
    obs = jnp.array([[1.0, 2.0]])
    mock_actor_state = request.getfixturevalue(mock_actor_state)
    pi, new_actor_state = get_pi(
        actor_state=mock_actor_state,
        actor_params=mock_actor_state.params,
        obs=obs,
        recurrent=False,
    )
    assert isinstance(pi, expected_distribution)
    assert new_actor_state.hidden_state is None


def test_get_pi_recurrent(mock_recurrent_actor_state):
    """Test policy distribution generation in recurrent mode."""
    obs = jnp.array([[1.0, 2.0]])
    done = jnp.array([0.0])
    mock_recurrent_actor_state = mock_recurrent_actor_state.replace(
        hidden_state=jnp.zeros((1, 2))
    )
    pi, new_actor_state = get_pi(
        actor_state=mock_recurrent_actor_state,
        actor_params=mock_recurrent_actor_state.params,
        obs=obs,
        done=done,
        recurrent=True,
    )
    assert isinstance(pi, distrax.Distribution)
    assert new_actor_state.hidden_state is not None
    assert new_actor_state.hidden_state.shape == (1, 2)


def _agent_state(env, env_params, actor_state, reset_key, rng, eval_rng):
    """CartPole's reset observation and an agent state collecting from it."""
    reset_keys = jax.random.split(reset_key, n_envs)
    obs, env_state = reset(reset_keys, env, mode="gymnax", env_params=env_params)
    collector_state = CollectorState(
        rng=rng,
        _env_state=env_state,
        last_obs=obs,
        last_terminated=jnp.zeros((n_envs,)),
        last_truncated=jnp.zeros((n_envs,)),
        episodic_return_state=init_rolling_mean(
            10,
            last_return=jnp.zeros((n_envs,)),
            cumulative_reward=jnp.zeros((n_envs,)),
        ),
    )
    return obs, BaseAgentState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=actor_state,
        collector_state=collector_state,
    )


@pytest.mark.parametrize(
    "mock_actor_state_key,recurrent",
    [
        ("mock_actor_state_discrete", False),
        ("mock_actor_state_continuous", False),
        ("mock_recurrent_actor_state", True),
    ],
)
def test_get_action_and_new_agent_state(
    mock_actor_state_key, recurrent, request, gymnax_env
):
    """Test action generation and agent state update for discrete, continuous, and recurrent agents."""
    env, env_params = gymnax_env
    mock_actor_state = request.getfixturevalue(mock_actor_state_key)

    reset_key, rng, eval_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    if recurrent:
        mock_actor_state = mock_actor_state.replace(hidden_state=jnp.zeros((n_envs, 2)))
    obs, agent_state = _agent_state(
        env, env_params, mock_actor_state, reset_key, rng, eval_rng
    )

    action, log_probs, raw_action, new_agent_state = get_action_and_new_agent_state(
        rng, agent_state, obs, done=jnp.zeros((n_envs,)), recurrent=recurrent
    )
    assert raw_action.shape == action.shape

    assert action.shape[0] == obs.shape[0]
    assert action.shape[0] == log_probs.shape[0]

    if recurrent:
        assert new_agent_state.actor_state.hidden_state.shape == (n_envs, 2)

    assert jnp.array_equal(new_agent_state.eval_rng, agent_state.eval_rng)


def test_get_action_and_new_agent_state_recurrent_without_done(
    mock_recurrent_actor_state, gymnax_env
):
    """Ensure AssertionError is raised when `done` is missing in recurrent mode."""
    env, env_params = gymnax_env
    reset_key, rng = jax.random.split(jax.random.PRNGKey(0))
    obs, agent_state = _agent_state(
        env, env_params, mock_recurrent_actor_state, reset_key, rng, rng
    )
    with pytest.raises(AssertionError):
        get_action_and_new_agent_state(rng, agent_state, obs, recurrent=True)


def test_init_collector_state(env_and_shape):
    env, env_params, mode, _, obs_shape = env_and_shape
    buffer = get_buffer(buffer_size=10, batch_size=2, n_envs=n_envs)
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=n_envs, continuous=mode == "brax"
    )
    collector_state = init_collector_state(
        jax.random.PRNGKey(0), env_args, mode=mode, buffer=buffer
    )
    assert collector_state.last_obs.shape == (n_envs, *obs_shape)
    assert collector_state.last_terminated.shape == (n_envs,)
    assert collector_state.buffer_state is not None
    assert collector_state.timestep == 0


# --------------------------------------------------------------------------
# Gymnax >= 1.0 terminal-transition metadata
# --------------------------------------------------------------------------


def test_get_final_obs_prefers_env_reported_final_observation():
    """The pre-reset obs must win over the (post-reset) fallback."""
    fallback = jnp.array([9.0, 9.0])
    final = jnp.array([1.0, 2.0])

    # gymnax >= 1.0 naming
    assert jnp.array_equal(
        step_module.get_final_obs({"final_observation": final}, fallback), final
    )
    # Ajax's brax/playground FinalObsWrapper naming
    assert jnp.array_equal(
        step_module.get_final_obs({"final_obs": final}, fallback), final
    )
    # env exposes none of them -> post-step obs
    assert jnp.array_equal(step_module.get_final_obs({}, fallback), fallback)


def test_step_gymnax_truncation_is_not_termination():
    """A time-limit hit must report truncated, not terminated.

    Conflating the two costs the ``V(s_T)`` bootstrap on every truncated
    episode. Pendulum never terminates naturally, so a short time limit
    isolates truncation.
    """
    env, _ = make_gymnax_env("Pendulum-v1")
    env_params = PendulumParams(max_steps_in_episode=3)
    rng = jax.random.split(jax.random.PRNGKey(0), n_envs)
    _, env_state = reset(rng, env, mode="gymnax", env_params=env_params)
    action = jnp.zeros((n_envs, 1))

    for _ in range(3):
        obsv, env_state, _, terminated, truncated, info = step(
            rng, env_state, action, env, mode="gymnax", env_params=env_params
        )

    assert jnp.all(truncated == 1.0)
    assert jnp.all(terminated == 0.0)
    # The env auto-reset, so the returned obs is the fresh episode's -- the
    # terminal one has to come from info.
    final_obs = step_module.get_final_obs(info, obsv)
    assert not jnp.allclose(final_obs, obsv)


def test_a_gymnax_step_warns_of_nothing_and_flags_in_float32():
    """Contract: a step under JAX's default config (no float64) raises no
    warning, and its terminated and truncated flags are float32, the replay
    buffer's schema. A first run used to warn that float64 was unavailable.
    """
    env, env_params = make_gymnax_env("CartPole-v1")
    rng = jax.random.split(jax.random.PRNGKey(0), n_envs)
    _, env_state = reset(rng, env, mode="gymnax", env_params=env_params)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, _, terminated, truncated, _ = step(
            rng, env_state, jnp.zeros(n_envs), env, mode="gymnax", env_params=env_params
        )

    assert terminated.dtype == jnp.float32
    assert truncated.dtype == jnp.float32


def test_bootstrap_obs_keeps_the_final_obs_where_the_time_limit_alone_ended():
    """Rows: running, truncated, terminated, both (gymnax flags a termination
    on the limit's step as both): the final obs on the truncated row only."""
    next_obs, final_obs = jnp.zeros((4, 2)), jnp.ones((4, 2))
    terminated = jnp.array([0.0, 0.0, 1.0, 1.0])
    truncated = jnp.array([0.0, 1.0, 0.0, 1.0])
    got = step_module.bootstrap_obs(next_obs, final_obs, terminated, truncated)
    assert jnp.array_equal(got[:, 0], jnp.array([0.0, 1.0, 0.0, 0.0]))
    column = step_module.bootstrap_obs(
        next_obs, final_obs, terminated[:, None], truncated[:, None]
    )
    assert jnp.array_equal(column, got)


def test_replayed_next_obs_is_one_step_on_from_obs_through_time_limits():
    """Every replay row's (obs, next_obs) is one pendulum step (theta' =
    theta + thdot' dt), the time-limited rows too, whose next row holds the
    reset observation instead."""
    env, env_params = make_gymnax_env("Pendulum-v1")
    env_params = env_params.replace(max_steps_in_episode=3)
    net = ("8", "relu")
    agent = SAC(
        env_id=env,
        env_params=env_params,
        n_envs=2,
        actor_architecture=net,
        critic_architecture=net,
        learning_starts=100,
        batch_size=4,
        buffer_size=40,
    )
    state, _ = agent.train(seed=0, n_timesteps=14)  # 7 rows per env
    rows = state.collector_state.buffer_state.experience
    obs, next_obs, truncated = (
        np.asarray(rows[k])[0, :, :7] for k in ("obs", "next_obs", "truncated")
    )

    def step_gap(o: np.ndarray, o_next: np.ndarray) -> np.ndarray:
        theta, theta_next = (np.arctan2(x[..., 1], x[..., 0]) for x in (o, o_next))
        gap = theta_next - theta - o_next[..., 2] * env_params.dt
        return np.abs(np.angle(np.exp(1j * gap)))

    cut = truncated[..., 0] > 0
    assert cut[:, [2, 5]].all() and cut.sum() == 4
    assert step_gap(obs, next_obs).max() < 1e-4
    assert (step_gap(obs[:, :-1], obs[:, 1:])[cut[:, :-1]] > 1e-2).all()


# --------------------------------------------------------------------------
# collect_experience, directly
# --------------------------------------------------------------------------


def _constant_actor(distribution):
    """An actor whose policy is ``distribution(batch size)``, whatever its
    input."""
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: distribution(obs.shape[0]),
        params={},
        tx=optax.sgd(0.0),
    )


#: The vanilla collectors (no action pipeline): PPO's, discrete actions and
#: no replay buffer; the off-policy agents', squashed actions into a buffer.
COLLECTORS = {
    "CartPole-v1": (lambda n: distrax.Categorical(logits=jnp.zeros((n, 2))), False),
    "Pendulum-v1": (
        lambda n: Squash(distrax.Normal(jnp.zeros((n, 1)), jnp.ones((n, 1)))),
        True,
    ),
}


@pytest.mark.parametrize("env_id", sorted(COLLECTORS))
def test_collect_experience_steps_every_env_once_per_call(env_id):
    """Five steps of two envs whose episodes last three: the timestep counts
    env steps; each transition's next_obs is the next step's obs, except at
    the time limit, where it is the final observation and the next obs the
    reset one; the flags mark that step; the replay buffer, where there is
    one, holds the transitions in order; and the episodic return is NaN
    until an episode ends, then the ended episodes' mean."""
    distribution, buffered = COLLECTORS[env_id]
    env, params = make_gymnax_env(env_id)
    params = params.replace(max_steps_in_episode=3)
    env_args = EnvironmentConfig(
        env=env, env_params=params, n_envs=2, continuous=buffered
    )
    buffer = get_buffer(buffer_size=16, batch_size=2, n_envs=2) if buffered else None
    key = jax.random.PRNGKey(0)
    actor = _constant_actor(distribution)
    state = BaseAgentState(
        rng=key,
        eval_rng=key,
        actor_state=actor,
        critic_state=actor,
        collector_state=init_collector_state(key, env_args, "gymnax", buffer),
    )
    collect = partial(
        collect_experience,
        recurrent=False,
        mode="gymnax",
        env_args=env_args,
        buffer=buffer,
    )

    def run(state, length):
        return jax.lax.scan(collect, state, None, length=length)

    state, first = run(state, 2)
    assert jnp.isnan(state.collector_state.episodic_mean_return)
    state, limit = run(state, 1)
    # The time limit truncates every episode at the third step.
    np.testing.assert_array_equal(state.collector_state.last_truncated, [1, 1])
    np.testing.assert_array_equal(state.collector_state.last_terminated, [0, 0])
    state, last = run(state, 2)
    rollout = jax.tree.map(lambda *x: jnp.concatenate(x), first, limit, last)
    collector = state.collector_state

    assert collector.timestep == 5 * 2
    ended = np.asarray(rollout.terminated + rollout.truncated)[..., 0] > 0
    np.testing.assert_array_equal(ended.sum(axis=1), [0, 0, 2, 0, 0])
    obs, next_obs = np.asarray(rollout.obs), np.asarray(rollout.next_obs)
    np.testing.assert_array_equal(next_obs[[0, 1, 3]], obs[[1, 2, 4]])
    assert (np.abs(next_obs[2] - obs[3]).max(axis=-1) > 0).all()
    np.testing.assert_array_equal(collector.last_obs, next_obs[-1])
    # Summed step by step there, by an XLA reduction here: equal to float32
    # rounding, not to the last ulp.
    np.testing.assert_allclose(
        collector.episodic_mean_return,
        rollout.reward[:3].sum(axis=0).mean(),
        rtol=1e-6,
    )
    if not buffered:
        return
    assert (np.abs(np.asarray(rollout.action)) <= 1).all()
    assert int(collector.buffer_state.current_index) == 5
    rows = collector.buffer_state.experience
    for field in ("obs", "next_obs", "action", "reward", "truncated"):
        np.testing.assert_array_equal(
            np.asarray(rows[field])[:, :5],
            np.swapaxes(getattr(rollout, field), 0, 1),
            err_msg=field,
        )


def test_every_envs_wrappers_draw_on_a_step_key_of_their_own():
    """Contract (env_keys: one key per gymnax env, at every step): the
    collector hands each env a step key of its own, so a wrapper drawing on
    it, here sensor noise above the episode-keyed resets, draws differently
    in every env. The noise is the observation less the state's noiseless
    one (get_obs)."""
    env, params = make_gymnax_env("Pendulum-v1")
    noisy = with_episode_keys(ObservationNoiseWrapper(env, sigma=1.0))
    env_args = EnvironmentConfig(
        env=noisy, env_params=params, n_envs=4, continuous=True
    )
    key = jax.random.PRNGKey(0)
    actor = _constant_actor(COLLECTORS["Pendulum-v1"][0])
    state = BaseAgentState(
        rng=key,
        eval_rng=key,
        actor_state=actor,
        critic_state=actor,
        collector_state=init_collector_state(key, env_args, "gymnax", None),
    )
    collect = partial(
        collect_experience, recurrent=False, mode="gymnax", env_args=env_args
    )
    for _ in range(3):
        state, _ = collect(state, None)
        collector = state.collector_state
        clean = jax.vmap(noisy.get_obs)(collector.env_state)
        noise = np.asarray(collector.last_obs - clean)
        assert len(np.unique(noise, axis=0)) == 4, noise


def test_every_env_runs_episodes_of_its_own():
    """Contract (env_keys: one key per gymnax env, at reset too): the
    collector resets each env on a key of its own, so with resets keyed by
    episode every env's episodes are its own: Pendulum's random starts
    differ across envs in the first episode and in the next."""
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(max_steps_in_episode=2)
    env_args = EnvironmentConfig(
        env=with_episode_keys(env), env_params=params, n_envs=4, continuous=True
    )
    key = jax.random.PRNGKey(0)
    actor = _constant_actor(COLLECTORS["Pendulum-v1"][0])
    state = BaseAgentState(
        rng=key,
        eval_rng=key,
        actor_state=actor,
        critic_state=actor,
        collector_state=init_collector_state(key, env_args, "gymnax", None),
    )
    collect = partial(
        collect_experience, recurrent=False, mode="gymnax", env_args=env_args
    )
    starts = [state.collector_state.last_obs]
    for _ in range(2):  # the second step truncates and resets every env
        state, _ = collect(state, None)
    starts.append(state.collector_state.last_obs)
    for obs in starts:
        assert len(np.unique(np.asarray(obs), axis=0)) == 4, obs


def test_the_collector_counts_each_envs_steps_in_its_episode_only_when_asked():
    """count_episode_steps keeps step_in_episode, each env's steps in its
    episode: 2 after two steps, back to 0 where the time limit (3) ended
    it; without it the collector keeps none."""
    distribution, _ = COLLECTORS["Pendulum-v1"]
    env, params = make_gymnax_env("Pendulum-v1")
    params = params.replace(max_steps_in_episode=3)
    env_args = EnvironmentConfig(env=env, env_params=params, n_envs=2, continuous=True)
    key = jax.random.PRNGKey(0)
    assert init_collector_state(key, env_args, "gymnax").step_in_episode is None
    actor = _constant_actor(distribution)
    collector = init_collector_state(key, env_args, "gymnax", count_episode_steps=True)
    state = BaseAgentState(
        rng=key,
        eval_rng=key,
        actor_state=actor,
        critic_state=actor,
        collector_state=collector,
    )
    collect = partial(
        collect_experience, recurrent=False, mode="gymnax", env_args=env_args
    )
    counts = []
    for _ in range(4):
        state, _ = collect(state, None)
        counts.append(state.collector_state.step_in_episode.tolist())
    assert counts == [[1, 1], [2, 2], [0, 0], [1, 1]]
    assert state.collector_state.step_in_episode.dtype == jnp.int32


@pytest.mark.parametrize("bonus", [-3.0, 0.0, 2.0])
def test_a_value_box_entry_ends_the_agents_episode_whatever_its_bonus(bonus):
    """Entering the value box pays the expert's value as a bonus and ends
    the agent's episode, the bonus standing in for the bootstrap: on a cost
    task the expert's value is 0 or negative, and its entry ends the
    episode as a positive one does. Only the entering env is cut."""
    distribution, _ = COLLECTORS["Pendulum-v1"]
    env, params = make_gymnax_env("Pendulum-v1")
    env_args = EnvironmentConfig(env=env, env_params=params, n_envs=2, continuous=True)
    key = jax.random.PRNGKey(0)
    actor = _constant_actor(distribution)
    state = BaseAgentState(
        rng=key,
        eval_rng=key,
        actor_state=actor,
        critic_state=actor,
        collector_state=init_collector_state(key, env_args, "gymnax", None),
    )
    entered = jnp.array([[1.0], [0.0]])

    def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        policy = step_module.policy_pipeline(
            agent_state,
            acting,
            raw_obs,
            rng,
            uniform,
            mix_key,
            action_key,
            recurrent=False,
            n_envs=2,
        )
        return policy._replace(
            in_value_box=entered, entry_bonus=bonus * entered, box_entry=entered
        )

    collect = partial(
        collect_experience, recurrent=False, mode="gymnax", env_args=env_args
    )
    _, plain = collect(state, None)
    _, boxed = collect(state, None, action_pipeline=pipeline)
    np.testing.assert_array_equal(plain.terminated[:, 0], [0, 0])
    np.testing.assert_array_equal(boxed.terminated[:, 0], [1, 0])
    np.testing.assert_allclose(
        boxed.reward[:, 0], plain.reward[:, 0] + bonus * entered[:, 0]
    )


@pytest.mark.parametrize("env_id", sorted(COLLECTORS))
def test_an_explore_policy_acts_after_the_warm_up_and_uniform_during_it(env_id):
    """explore_pipeline steps and stores explore's action (a discrete one as
    a column) with zero log-probs and no expert; during the warm-up the env
    steps a uniform action instead, which the replay stores, or its warm-up
    chooser's."""
    distribution, continuous = COLLECTORS[env_id]
    env, params = make_gymnax_env(env_id)
    env_args = EnvironmentConfig(
        env=env, env_params=params, n_envs=2, continuous=continuous
    )
    buffer = get_buffer(buffer_size=16, batch_size=2, n_envs=2)
    key = jax.random.PRNGKey(0)
    actor = _constant_actor(distribution)
    state = BaseAgentState(
        rng=key,
        eval_rng=key,
        actor_state=actor,
        critic_state=actor,
        collector_state=init_collector_state(key, env_args, "gymnax", buffer),
    )
    chosen = 0.5 if continuous else 1

    def explore(pi, key, step):
        return jnp.full(pi.sample(seed=key).shape, chosen) + 0 * step

    pipeline = step_module.explore_pipeline(
        explore, recurrent=False, n_actions=None if continuous else 2
    )
    collect = partial(
        collect_experience,
        recurrent=False,
        mode="gymnax",
        env_args=env_args,
        buffer=buffer,
        action_pipeline=pipeline,
    )

    def stored(state):
        # The replay is [env, row, ...]: the first row of each env.
        return np.asarray(state.collector_state.buffer_state.experience["action"])[:, 0]

    acted, transition = collect(state, None)
    np.testing.assert_array_equal(stored(acted), [[chosen], [chosen]])
    np.testing.assert_array_equal(transition.action, [[chosen], [chosen]])
    np.testing.assert_array_equal(transition.log_prob, 0.0)
    np.testing.assert_array_equal(acted.collector_state.last_expert_frac, 0.0)
    warm, transition = collect(state, None, uniform=True)
    np.testing.assert_array_equal(transition.action, [[chosen], [chosen]])
    assert (stored(warm) != chosen).any() and (np.abs(stored(warm)) <= 1).all()
    # A warm-up chooser taking every warm-up step hands it to the expert,
    # whose action the replay stores, flagged; a later step is not flagged.
    expert_choice = 0.25 if continuous else 0

    def expert(obs):  # a discrete action's index as an (n, 1) column
        return jnp.full(obs.shape[:1] + (1,), expert_choice)

    chooser = step_module.explore_pipeline(
        explore,
        recurrent=False,
        n_actions=None if continuous else 2,
        warmups=(ExpertWarmup(fraction=1.0),),
    )
    expert_kwargs = step_module.expert_collect_kwargs(expert, chooser, False, 2)
    for uniform, flag, action in ((True, 1.0, expert_choice), (False, 0.0, chosen)):
        acted, _ = collect(state, None, uniform=uniform, **expert_kwargs)
        np.testing.assert_array_equal(stored(acted), [[action], [action]])
        np.testing.assert_array_equal(acted.collector_state.last_expert_frac, flag)


class _Pipelines:
    """A gain expert (one learnable gain; its own call adds 10 to its state,
    a call with gains 1) and a pipeline choosing a fixed action, for the
    expert collection wrapper."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([1.0])

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return self.anchor_gains[0] * obs[..., :1], state + 10.0

    def step_with_gains(self, state, obs, gains):
        return gains[0] * obs[..., :1], state + 1.0

    @staticmethod
    def chosen(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        """Reads the expert's action off the acting state, as SAC's does."""
        assert acting.expert_action is not None
        action = jnp.full((2, 1), 0.5)
        return step_module.ActionPipelineResult(env_action=action, policy_action=action)


def test_the_collector_calls_the_expert_once_and_actuates_the_chosen_action():
    """The expert is called before the pipeline (its action and next state
    stored); gain mode's actuate executes the chosen action through it,
    stores the chosen one and keeps the state the actuation leaves."""
    expert, obs = _Pipelines(), jnp.full((2, 1), 2.0)
    acting = ActingState(
        obs=obs, done=jnp.zeros(2), timestep=0, expert_state=jnp.zeros((2, 1))
    )
    args = (None, acting, obs, jax.random.PRNGKey(0), False, None, None)
    for actuate in (None, ExpertGains()):
        kwargs = step_module.expert_collect_kwargs(
            expert, _Pipelines.chosen, False, 2, actuate
        )
        result = kwargs["action_pipeline"](*args)
        np.testing.assert_array_equal(result.a_expert, 2.0)  # the expert's own
        if actuate is None:
            np.testing.assert_array_equal(result.new_expert_state, 10.0)
            np.testing.assert_array_equal(result.env_action, 0.5)
            assert result.buffer_action is None  # the executed action is stored
        else:
            # gains anchor * 10**0.5 on obs 2.0; the replay keeps the gains' action.
            np.testing.assert_array_equal(result.new_expert_state, 1.0)
            np.testing.assert_allclose(result.env_action, 2.0 * 10**0.5, rtol=1e-6)
            np.testing.assert_array_equal(result.buffer_action, 0.5)
