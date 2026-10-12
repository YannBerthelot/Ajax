import copy
import pickle

import jax
import jax.numpy as jnp
import pytest
from brax.envs.base import Env as BraxEnv
from brax.envs.base import State
from brax.envs.wrappers import training as brax_training
from flax import struct
from gymnax import make as make_gymnax_env
from gymnax.environments import environment as gymnax_environment
from gymnax.environments.classic_control.pendulum import EnvParams as PendulumParams

from ajax.environments.utils import get_state_action_shapes
from ajax.wrappers import (
    AutoResetWrapper,
    ClipAction,
    ClipActionBrax,
    EnvNormalizationInfo,
    FlattenObservationWrapper,
    NormalizationInfo,
    NormalizeVecObservationBrax,
    NormalizeVecObservationGymnax,
    TerminatedTruncatedWrapper,
)


class MockGymnaxEnv:
    """Mock Gymnax environment for testing."""

    def __init__(self):
        self.observation_space = lambda params: jnp.array([0.0, 0.0])
        self.action_space = lambda params: jnp.array([-1.0, 1.0])
        self.default_params = None  # Mock params, can be set as needed

    def reset(self, key, params=None):
        obs = jnp.array([1.0, -1.0])
        state = MockGymnaxState(step_count=0)
        return obs, state

    def step(self, key, state, action, params=None):
        obs = jnp.array([1.0, -1.0]) * state.step_count
        reward = jnp.array([2.0])
        terminated = state.step_count >= 5
        truncated = jnp.zeros_like(terminated)
        info = {}
        state = MockGymnaxState(step_count=state.step_count + 1)
        return obs, state, reward, terminated, truncated, info


class MockBraxEnv:
    """Mock Brax environment for testing."""

    def __init__(self):
        self.observation_size = 2
        self.action_size = 1

    def reset(self, key):
        return State(
            obs=jnp.array([[1.0, -1.0]]),
            reward=jnp.array([0.0]),
            done=False,
            metrics={},
            pipeline_state=None,
            info={},
        )

    def step(self, state, action):
        obs = jnp.array([[1.0, -1.0]])
        reward = 1.0
        done = reward >= 5.0
        return State(
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            pipeline_state=None,
            info={},
        )


@struct.dataclass
class MockGymnaxState:
    """Mock Brax state for testing."""

    step_count: jnp.ndarray


class BatchedMockGymnaxEnv:
    """Mock Gymnax environment for batched testing."""

    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.observation_space = lambda params: jnp.array([0.0, 0.0])
        self.action_space = lambda params: jnp.array([-1.0, 1.0])
        self.default_params = None  # Mock params, can be set as needed

    def reset(self, key, params=None):
        obs = jnp.tile(jnp.array([1.0, -1.0]), (self.batch_size, 1))
        state = BatchedMockGymnaxState(step_count=jnp.zeros(self.batch_size))
        return obs, state

    def step(self, key, state, action, params=None):
        obs = jnp.tile(jnp.array([1.0, -1.0]) * action, (self.batch_size, 1))
        reward = jnp.ones(self.batch_size) * 2.0
        terminated = state.step_count >= 5
        truncated = jnp.zeros_like(terminated)
        info = {}
        state = BatchedMockGymnaxState(step_count=state.step_count + 1)
        return obs, state, reward, terminated, truncated, info


class BatchedMockBraxEnv:
    """Mock Brax environment for batched testing."""

    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.observation_size = 2
        self.action_size = 1

    def reset(self, key):
        obs = jnp.tile(jnp.array([1.0, -1.0]), (self.batch_size, 1))
        return State(
            obs=obs,
            reward=jnp.zeros(self.batch_size),
            done=jnp.zeros(self.batch_size, dtype=bool),
            metrics={},
            pipeline_state=None,
            info={},
        )

    def step(self, state, action):
        obs = jnp.tile(jnp.array([1.0, -1.0]), (self.batch_size, 1)) * action
        reward = jnp.ones(self.batch_size)
        done = reward >= 5.0
        return State(
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            pipeline_state=None,
            info={},
        )


@struct.dataclass
class BatchedMockGymnaxState:
    """Mock Gymnax state for batched testing."""

    step_count: jnp.ndarray


@pytest.fixture
def gymnax_env():
    return MockGymnaxEnv(), None, "gymnax"


@pytest.fixture
def brax_env():
    return MockBraxEnv(), None, "brax"


@pytest.fixture
def batched_gymnax_env():
    return BatchedMockGymnaxEnv(), None, "gymnax"


@pytest.fixture
def batched_brax_env():
    return BatchedMockBraxEnv(), None, "brax"


def _reset(env, key, params, mode):
    """``(obs, state)`` of either API's reset."""
    if mode == "gymnax":
        return env.reset(key, params)
    state = env.reset(key)
    return state.obs, state


def _step(env, key, state, action, params, mode):
    """``(obs, state, reward, done)`` of either API's step."""
    if mode == "gymnax":
        obs, state, reward, terminated, truncated, _ = env.step(
            key=key, state=state, action=action, params=params
        )
        return obs, state, reward, jnp.logical_or(terminated, truncated)
    state = env.step(state=state, action=action)
    return state.obs, state, state.reward, state.done


def _norm(state, mode):
    """The normaliser's statistics carried in ``state``."""
    return (
        state.info["normalization_info"] if mode == "brax" else state.normalization_info
    )


NORMALIZERS = {
    "gymnax": NormalizeVecObservationGymnax,
    "brax": NormalizeVecObservationBrax,
}


def test_flatten_observation_wrapper(gymnax_env):
    env, env_params, _ = gymnax_env
    wrapped_env = FlattenObservationWrapper(env)
    key = jax.random.PRNGKey(0)
    obs, state = wrapped_env.reset(key, env_params)
    assert obs.ndim == 1  # Observation should be flattened
    obs, *_ = wrapped_env.step(key, state, 0, env_params)
    assert obs.ndim == 1  # Observation should remain flattened


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, lambda env: pickle.loads(pickle.dumps(env))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_gymnax_wrapper_copies_and_pickles(duplicate):
    """A wrapped env duplicates like the raw env: same type, same reset."""
    env, params = make_gymnax_env("Pendulum-v1")
    wrapped_env = ClipAction(FlattenObservationWrapper(env))
    duplicated = duplicate(wrapped_env)
    assert type(duplicated) is ClipAction
    assert type(duplicated._env) is FlattenObservationWrapper
    key = jax.random.PRNGKey(0)
    assert jnp.array_equal(
        duplicated.reset(key, params)[0], wrapped_env.reset(key, params)[0]
    )


@pytest.mark.parametrize(
    "wrapper,env_fixture",
    [(ClipAction, "gymnax_env"), (ClipActionBrax, "brax_env")],
)
def test_clip_action(wrapper, env_fixture, request):
    env, env_params, mode = request.getfixturevalue(env_fixture)
    wrapped_env = wrapper(env, low=-0.5, high=0.5)
    key = jax.random.PRNGKey(0)
    _, state = _reset(wrapped_env, key, env_params, mode)
    out = _step(wrapped_env, key, state, jnp.array(1.0), env_params, mode)
    bound = _step(wrapped_env, key, state, jnp.array(0.5), env_params, mode)
    assert jnp.allclose(out[0], bound[0])
    assert out[2] == bound[2]


@pytest.mark.parametrize("env_fixture", ["batched_gymnax_env", "batched_brax_env"])
def test_normalize_vec_observation_2(env_fixture, request):
    env, env_params, mode = request.getfixturevalue(env_fixture)
    wrapped_env = NORMALIZERS[mode](env, gamma=0.99)
    key = jax.random.PRNGKey(0)
    obs, state = _reset(wrapped_env, key, env_params, mode)
    norm_info = _norm(state, mode).obs
    assert jnp.all(norm_info.count == 2)  # batch size
    assert jnp.allclose(norm_info.mean, jnp.array([[1.0, -1.0], [1.0, -1.0]]))
    assert jnp.allclose(obs, jnp.zeros(2))  # normalized should be 0
    for count in (4, 6):  # batch size x 2, x 3
        obs, state, *_ = _step(wrapped_env, key, state, 1, env_params, mode)
        norm_info = _norm(state, mode).obs
        assert jnp.all(norm_info.count == count)
        assert jnp.allclose(norm_info.mean, jnp.array([[1.0, -1.0], [1.0, -1.0]]))
        # The fixture env emits constant obs across batches -- correct Welford
        # M2 stays 0 (no variance to accumulate). Previous online_normalize
        # patched zero-M2 to 1 via ``replace_zeros_with_ones`` to dodge
        # div-by-zero; the patch was removed to match brax
        # acme.running_statistics. std is now clipped to [1e-6, 1e6] at use
        # time so the normalised obs stays finite.
        assert jnp.allclose(norm_info.mean_2, 0)
        assert jnp.allclose(obs, jnp.zeros((2, 2)))


def _statistics(count, mean, mean_2, **extra):
    return NormalizationInfo(
        count=count, mean=mean, mean_2=mean_2, var=mean_2 / count, **extra
    )


def _shared(batch_size):
    """Statistics shared by the envs, tiled to one row per env."""

    def tiled(x):
        return jnp.tile(x, reps=(batch_size, 1))

    return EnvNormalizationInfo(
        obs=_statistics(
            tiled(jnp.ones(1)), tiled(jnp.array([1, -1])), tiled(jnp.array([0.5, 0.5]))
        ),
        reward=_statistics(
            tiled(jnp.ones(1)),
            tiled(jnp.array([2.0])),
            tiled(jnp.array([0.5])),
            returns=jnp.array([0.5]),
        ),
    )


#: Given statistics: one env's, one row per env of a batch, shared ones.
GIVEN = {
    "one_env": lambda batch_size: EnvNormalizationInfo(
        obs=_statistics(
            jnp.array([[10]]), jnp.array([[1.0, -1.0]]), jnp.array([[0.5, 0.5]])
        ),
        reward=_statistics(jnp.array([[10]]), jnp.array([2.0]), jnp.array([0.5])),
    ),
    "per_env": lambda batch_size: EnvNormalizationInfo(
        obs=_statistics(
            jnp.array([[10], [20]]),
            jnp.array([[1.0, -1.0]]),
            jnp.array([[0.5, 0.5], [1.0, 1.0]]),
        ),
        reward=_statistics(jnp.array([[10], [20]]), jnp.array([2.0]), jnp.array([0.5])),
    ),
    "shared": _shared,
}


@pytest.mark.parametrize("mode", ["gymnax", "brax"])
@pytest.mark.parametrize("given", list(GIVEN))
def test_eval_mode_normalises_with_the_given_statistics(given, mode, request):
    """``train=False`` keeps the statistics it is given, at their shapes, and
    presents ``(obs - mean) / sqrt(var + 1e-8)``."""
    batched = "" if given == "one_env" else "batched_"
    env, env_params, _ = request.getfixturevalue(f"{batched}{mode}_env")
    key = jax.random.PRNGKey(0)
    raw_obs, _ = _reset(env, key, env_params, mode)
    norm_info = GIVEN[given](raw_obs.shape[0])
    wrapped_env = NORMALIZERS[mode](env, norm_info=norm_info, train=False, gamma=0.99)
    obs, state = _reset(wrapped_env, key, env_params, mode)
    kept = _norm(state, mode).obs
    for field in ("count", "mean", "mean_2", "var"):
        assert getattr(kept, field).shape == getattr(norm_info.obs, field).shape
    assert jnp.allclose(kept.mean, norm_info.obs.mean)
    assert jnp.allclose(kept.var, norm_info.obs.var)
    expected = (raw_obs - norm_info.obs.mean) / jnp.sqrt(norm_info.obs.var + 1e-8)
    assert jnp.allclose(obs, expected)


@pytest.mark.parametrize("mode", ["gymnax", "brax"])
def test_train_mode_updates_shared_statistics_per_env(mode, request):
    env, env_params, _ = request.getfixturevalue(f"batched_{mode}_env")
    key = jax.random.PRNGKey(0)
    raw_obs, _ = _reset(env, key, env_params, mode)
    batch_size, obs_size = raw_obs.shape
    norm_info = _shared(batch_size)
    wrapped_env = NORMALIZERS[mode](env, norm_info=norm_info, train=True, gamma=0.99)

    def check_shapes(statistics):
        assert statistics.count.shape == (batch_size, 1)
        for field in ("mean", "mean_2", "var"):
            assert getattr(statistics, field).shape == (batch_size, obs_size)

    _, state = _reset(wrapped_env, key, env_params, mode)
    check_shapes(_norm(state, mode).obs)
    action = 7 if mode == "gymnax" else jnp.ones_like(raw_obs) * 7
    _, state, *_ = _step(wrapped_env, key, state, action, env_params, mode)
    updated = _norm(state, mode).obs
    check_shapes(updated)
    assert jnp.any(updated.count > norm_info.obs.count), "count not incremented"
    assert not jnp.allclose(updated.mean, norm_info.obs.mean), "mean not updated"
    assert not jnp.allclose(updated.var, norm_info.obs.var), "variance not updated"


# --------------------------------------------------------------------------
# Gymnax >= 1.0 six-value step API
#
# The wrapper tests above run against hand-written mocks, so they cannot
# catch a drift between Ajax's gymnax-side wrappers and gymnax's own step
# contract. These exercise the wrappers against real gymnax environments.
# --------------------------------------------------------------------------


@pytest.fixture
def real_gymnax_env():
    """A stock gymnax env with a short time limit (fast truncation)."""
    env, _ = make_gymnax_env("Pendulum-v1")
    return env, PendulumParams(max_steps_in_episode=3)


def test_gymnax_env_step_returns_six_values(real_gymnax_env):
    """Guards the assumption every gymnax-side wrapper is written against."""
    env, params = real_gymnax_env
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)
    out = env.step(key, state, jnp.array([0.0]), params)
    assert len(out) == 6
    assert "final_observation" in out[-1]


def test_normalize_wrapper_on_real_gymnax_env(real_gymnax_env):
    """NormalizeVecObservationGymnax must survive a real six-value step."""
    env, params = real_gymnax_env
    wrapped_env = NormalizeVecObservationGymnax(env, gamma=0.99)
    key = jax.random.PRNGKey(0)
    obs, state = wrapped_env.reset(key, params)

    obs, state, reward, terminated, truncated, info = wrapped_env.step(
        key=key, state=state, action=jnp.array([0.0]), params=params
    )
    assert obs.shape == env.obs_shape
    assert state.normalization_info.obs is not None
    assert not bool(terminated)


def test_two_normalisers_of_one_env_build_states_of_one_class(real_gymnax_env):
    """A state one normaliser built is another's of the same env: a new
    agent instance resumes the state an earlier one returned."""
    env, params = real_gymnax_env
    states = [
        NormalizeVecObservationGymnax(env).reset(jax.random.PRNGKey(0), params)[1]
        for _ in range(2)
    ]
    assert jax.tree.structure(states[0]) == jax.tree.structure(states[1])


@struct.dataclass
class _LegacyState(gymnax_environment.EnvState):
    pass


class _LegacyDoneEnv(gymnax_environment.Environment):
    """Env whose ``step_env`` still folds the time limit into its done flag.

    Mirrors third-party gymnax envs that have not migrated to the 1.0 split.
    It never terminates on its own, so every terminal flag it reports is the
    time limit. gymnax's base ``step`` auto-resets it, so the state returned
    at the time limit is already the next episode's (``time == 0``).
    """

    @property
    def default_params(self):
        return gymnax_environment.EnvParams(max_steps_in_episode=2)

    def reset_env(self, key, params):
        return jnp.zeros(2), _LegacyState(time=0)

    def step_env(self, key, state, action, params):
        state = _LegacyState(time=state.time + 1)
        done = state.time >= params.max_steps_in_episode
        return jnp.zeros(2), state, jnp.array(1.0), done, {}


def test_terminated_truncated_wrapper_unfolds_time_limit():
    """The wrapper must move a time-limit `done` out of `terminated`."""
    env = _LegacyDoneEnv()
    params = env.default_params
    wrapped_env = TerminatedTruncatedWrapper(env)
    key = jax.random.PRNGKey(0)
    _, state = wrapped_env.reset(key, params)

    _, state, _, terminated, truncated, _ = wrapped_env.step(key, state, 0, params)
    assert not bool(terminated)
    assert not bool(truncated)

    _, state, _, terminated, truncated, _ = wrapped_env.step(key, state, 0, params)
    assert int(state.time) == 0  # auto-reset already happened
    # Time limit reached: truncated, and crucially NOT terminated -- otherwise
    # the agent would drop the V(s_T) bootstrap.
    assert bool(truncated)
    assert not bool(terminated)


def test_initial_state_wrapper_overrides_reset_state():
    from gymnax import make as _make

    from ajax.wrappers import InitialStateWrapper

    env, params = _make("Pendulum-v1")

    def pin(key, state, params):
        del params
        theta = jax.random.uniform(key, minval=-0.1, maxval=0.1)
        return state.replace(theta=theta, theta_dot=jnp.asarray(0.0))

    wrapped = InitialStateWrapper(env, pin)
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    obs, states = jax.vmap(wrapped.reset, in_axes=(0, None))(keys, params)
    assert jnp.all(jnp.abs(states.theta) <= 0.1)
    assert jnp.all(states.theta_dot == 0.0)
    # observation is recomputed from the overridden state
    assert jnp.allclose(obs[:, 0], jnp.cos(states.theta))
    assert jnp.allclose(obs[:, 2], 0.0)
    # stepping still works through the wrapper and keeps the env's own dynamics
    obs2, s2, r, term, trunc, info = wrapped.step(
        keys[0], jax.tree.map(lambda x: x[0], states), jnp.zeros(1), params
    )
    assert obs2.shape == (3,) and jnp.isfinite(r)


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "fast"])
def test_normaliser_presents_the_final_observation_normalised(env_id):
    """Off an episode end the pre-reset observation is the step's own: the
    normaliser presents it as it presents the observation (it is the
    transition's next observation) and keeps the raw one."""
    import numpy as np

    from ajax.environments.create import prepare_env
    from ajax.environments.interaction import (
        get_final_obs,
        get_raw_final_obs,
        reset,
        step,
    )
    from ajax.environments.utils import get_env_type

    env, params, *_ = prepare_env(env_id, normalize_obs=True)
    mode = get_env_type(env)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 1) if mode == "gymnax" else key
    _, state = reset(keys, env, mode, params)
    obs, *_, info = step(keys, state, jnp.full((1, 1), 0.5), env, mode, params)
    np.testing.assert_allclose(get_final_obs(info, None), obs, rtol=1e-6)
    assert not np.allclose(get_raw_final_obs(info, None), obs)


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "fast"])
def test_reward_normaliser_keeps_the_raw_reward(env_id):
    """The normaliser hands the reward divided by its running std and keeps
    the env's own in info (a brax state carries it from the reset on, so its
    info keeps one structure through a scan)."""
    import numpy as np

    from ajax.environments.create import prepare_env
    from ajax.environments.interaction import reset, step
    from ajax.environments.utils import get_env_type
    from ajax.wrappers import RAW_REWARD_KEY

    env, params, *_ = prepare_env(env_id, normalize_reward=True)
    mode = get_env_type(env)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 1) if mode == "gymnax" else key
    _, state = reset(keys, env, mode, params)
    action = jnp.full((1, 1), 0.5)

    def one(state, _):
        _, state, reward, *_, info = step(keys, state, action, env, mode, params)
        return state, (reward, info[RAW_REWARD_KEY])

    state, (normed, raw) = jax.lax.scan(one, state, length=3)
    norm = (
        state.info["normalization_info"] if mode == "brax" else state.normalization_info
    )
    std = np.sqrt(np.asarray(norm.reward.var).mean() + 1e-8)
    np.testing.assert_allclose(normed[-1], raw[-1] / std, rtol=1e-5)
    assert not np.allclose(normed, raw)


@pytest.mark.parametrize("gamma", [None, 0.9])
def test_reward_normaliser_divides_by_the_chosen_statistics(gamma):
    """By default the running std of the single-step reward; with ``gamma``,
    of SB3's discounted return of the episode so far, restarted after the
    episode's end. Replayed in numpy on CartPole under random actions."""
    import numpy as np

    from ajax.environments.create import prepare_env
    from ajax.environments.interaction import reset, step
    from ajax.wrappers import RAW_REWARD_KEY

    env, params, *_ = prepare_env("CartPole-v1", normalize_reward=True, gamma=gamma)
    key = jax.random.PRNGKey(0)
    _, state = reset(jax.random.split(key, 1), env, "gymnax", params)

    def one(state, key):
        action = jax.random.randint(key, (1,), 0, 2)
        out = step(jax.random.split(key, 1), state, action, env, "gymnax", params)
        _, state, normed, terminated, truncated, info = out
        done = jnp.logical_or(terminated, truncated)
        kept = state.normalization_info.reward.returns.reshape(-1)
        return state, (normed, info[RAW_REWARD_KEY], done, kept)

    keys = jax.random.split(key, 100)
    _, (normed, raw, done, kept) = jax.lax.scan(one, state, keys)
    ret, count, mean, m2 = 0.0, 0, 0.0, 0.0
    for t in range(100):
        ret = float(raw[t, 0]) + (gamma or 0.0) * ret
        count, delta = count + 1, ret - mean
        mean += delta / count
        m2 += delta * (ret - mean)
        std = np.clip(np.sqrt(m2 / count + 1e-8), 1e-6, 1e6)
        np.testing.assert_allclose(normed[t, 0], raw[t, 0] / std, rtol=1e-4)
        ret *= 1.0 - float(done[t, 0])
        if gamma is not None:
            np.testing.assert_allclose(kept[t, 0], ret, rtol=1e-5)
    assert done.sum() >= 2


class _ResetRecorder(BraxEnv):
    """A toy brax env counting the resets that run: its callback fires when
    a reset executes, never when it is only traced (``jax.eval_shape``)."""

    def __init__(self):
        self.runs: list = []

    def reset(self, rng):
        jax.debug.callback(lambda: self.runs.append(1))
        x = jax.random.uniform(rng, (3,))
        return State(pipeline_state=x, obs=x, reward=jnp.zeros(()), done=jnp.zeros(()))

    def step(self, state, action):
        return state

    @property
    def observation_size(self):
        """As brax's ``PipelineEnv`` defines it: by running the reset."""
        return self.reset(jax.random.PRNGKey(0)).obs.shape[-1]

    @property
    def action_size(self):
        return 1

    @property
    def backend(self):
        return "toy"


def test_the_wrappers_read_shapes_without_running_the_reset():
    """Building the brax stack (auto-reset, normaliser) and reading
    its shapes trace the reset, never run it: on brax physics a run takes
    seconds."""
    env = _ResetRecorder()
    stack = brax_training.VmapWrapper(
        brax_training.EpisodeWrapper(env, 10, action_repeat=1), batch_size=4
    )
    stack = AutoResetWrapper(stack)
    stack = NormalizeVecObservationBrax(stack)
    shapes = get_state_action_shapes(stack)
    jax.effects_barrier()
    assert env.runs == []
    assert stack.n_envs == 4 and stack.batch_size == 4
    assert shapes == ((3,), (1,))
