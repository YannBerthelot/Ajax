"""Resets keyed by episode.

Contract, from the RNG-stability plan's step 8 (owner-approved 2026-10-25,
after PEGASUS's fixed scenarios, Ng and Jordan 2000) and the owner's
decisions of 2026-10-27:
- env i's k-th episode starts the same in every variant of a run, whatever
  the actions, updates and extensions before it and the other envs' episodes;
  a gymnax env's steps in it draw alike too; so do the wrappers that draw an
  episode's start (InitialStateWrapper, ModelReferenceWrapper; owner, Q3);
- it is on by default, also for an env one built; episode_keys=False keeps
  the env's own auto-reset (gymnax's step), as runs before it did;
- an env overriding gymnax's step or reset is refused, the error naming
  episode_keys=False (owner, Q2);
- an env's episode counter advances only where its episode ends.
Expected values are relations between runs (metamorphic), the env's own
auto-reset (gymnax), or worked by hand; none recomputes a key.
"""

from typing import Any, Callable

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.wrappers import training as brax_training
from flax import struct
from gymnax.environments import environment, spaces

import ajax
from ajax.environments.create import build_env_from_id, prepare_env, with_episode_keys
from ajax.environments.interaction import reset, step
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.row_collector import collect_row, init_row_collector_state
from ajax.environments.utils import env_keys
from ajax.extensions.expert import ExpertWarmup
from ajax.state import EnvironmentConfig
from ajax.wrappers import (
    BatchRngWrapper,
    FinalObsWrapper,
    FreshAutoResetWrapper,
    InitialStateWrapper,
)
from tests.environments.test_brax_auto_reset import _DriftEnv

# Starts and draws compare exactly: each is a uniform draw, an integer hash of
# its key turned into a float, the same bits in any program on one machine.
N = 2  # envs


@struct.dataclass
class _State(environment.EnvState):
    start: jax.Array
    total: jax.Array


class _Pushed(environment.Environment):
    """Starts at a uniform draw and ends once its actions sum to 1, so the
    actions decide where episodes end; pays a draw on its step key."""

    default_params = environment.EnvParams(max_steps_in_episode=50)

    def reset_env(self, key: Any, params: Any) -> Any:
        state = _State(time=0, start=jax.random.uniform(key), total=jnp.float32(0.0))
        return self.get_obs(state), state

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        total = state.total + jnp.reshape(action, ())
        state = _State(time=state.time + 1, start=state.start, total=total)
        return self.get_obs(state), state, jax.random.uniform(key), total >= 1.0, {}

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> Any:
        return jnp.stack([state.start, state.total])

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (1,), jnp.float32)

    def observation_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(0.0, 2.0, (2,), jnp.float32)


def _restart(key: jax.Array, state: Any, params: Any) -> Any:
    return state.replace(start=jax.random.uniform(key))


def _start_obs(obs: Any, state: Any) -> Any:
    return obs[:, 0]


def _reference(obs: Any, state: Any) -> Any:
    return state.reference


TASKS: dict[str, tuple[Callable[[], Any], Callable]] = {
    "plain": (_Pushed, _start_obs),
    "initial_state": (lambda: InitialStateWrapper(_Pushed(), _restart), _start_obs),
    "model_reference": (
        lambda: ModelReferenceWrapper(
            _Pushed(),
            StepReference(8, -1.0, 1.0, 1, 2),
            LinearReferenceModel.first_order(),
        ),
        _reference,
    ),
}


def _episodes(
    env: Any, push: float, step_seed: int, read: Callable = _start_obs
) -> list:
    """Each env's episodes in order: (what read sees at its start, the
    rewards of its steps), pushing push a step on keys of step_seed."""
    params = env.default_params
    obs, state = reset(
        env_keys(jax.random.PRNGKey(0), "gymnax", N), env, "gymnax", params
    )
    starts = np.asarray(read(obs, state))
    out: list = [[(starts[i].tolist(), [])] for i in range(N)]
    for key in jax.random.split(jax.random.PRNGKey(step_seed), 12):
        keys, actions = env_keys(key, "gymnax", N), jnp.full((N, 1), push)
        obs, state, reward, ended, cut, _ = step(
            keys, state, actions, env, "gymnax", params
        )
        starts = np.asarray(read(obs, state))
        for i in range(N):
            out[i][-1][1].append(float(reward[i]))
            if ended[i] or cut[i]:
                out[i].append((starts[i].tolist(), []))
    return out


def _starts(episodes: list) -> list:
    return [[start for start, _ in env] for env in episodes]


@pytest.mark.parametrize("task", TASKS)
def test_env_i_episode_k_starts_alike_whatever_the_actions_and_step_keys(task):
    make, read = TASKS[task]
    env = prepare_env(make())[0]  # on by default
    halves = _episodes(env, 0.5, step_seed=1, read=read)  # episodes of 2 steps
    quarters = _episodes(env, 0.25, step_seed=2, read=read)  # of 4
    for i in range(N):
        assert _starts(halves)[i][:3] == _starts(quarters)[i][:3]
        assert len({str(s) for s in _starts(quarters)[i]}) == len(quarters[i])
    assert _starts(halves)[0][0] != _starts(halves)[1][0]
    assert len(halves[0]) > len(quarters[0])  # the episodes ended at other steps


def test_a_gymnax_envs_step_t_of_episode_k_draws_alike_whatever_the_actions():
    env = prepare_env(_Pushed())[0]
    halves, quarters = _episodes(env, 0.5, 1), _episodes(env, 0.25, 2)
    for i in range(N):
        for (_, rewards), (_, other) in zip(
            halves[i][:3], quarters[i][:3], strict=False
        ):
            assert rewards[:2] == other[:2]  # the draws of steps 0 and 1


def test_counters_advance_only_at_episode_ends():
    """Pushed 0.4 a step, an episode ends on its third step (1.2 >= 1)."""
    env, params = prepare_env(_Pushed())[0], _Pushed.default_params
    _, state = env.reset(jax.random.PRNGKey(0), params)
    seen = []
    for t in range(6):
        _, state, *_ = env.step(jax.random.PRNGKey(t), state, jnp.array([0.4]), params)
        seen.append((int(state.episode_index), int(state.episode_step)))
    assert seen == [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]


@pytest.mark.parametrize("off", ["built", "marked"])
def test_off_steps_as_the_envs_own_auto_reset(off):
    """gymnax's own CartPole, stepped on the same keys, is the reference."""
    raw, params = gymnax.make("CartPole-v1")
    if off == "built":
        env = build_env_from_id("CartPole-v1", episode_keys=False)[0]
    else:
        env = with_episode_keys(gymnax.make("CartPole-v1")[0], on=False)
    env = prepare_env(env)[0]
    mine, theirs = (
        env.reset(jax.random.PRNGKey(0), params),
        raw.reset(jax.random.PRNGKey(0), params),
    )
    for t in range(30):  # pushed left: the pole falls, and CartPole resets
        key = jax.random.PRNGKey(t)
        jax.tree.map(np.testing.assert_array_equal, mine, theirs)
        mine, theirs = (
            env.step(key, mine[1], 0, params),
            raw.step(key, theirs[1], 0, params),
        )


def test_an_env_overriding_gymnax_step_is_refused_naming_the_way_out():
    deep_sea, params = gymnax.make(
        "DeepSea-bsuite"
    )  # keeps bsuite's metrics across resets
    with pytest.raises(ValueError, match="episode_keys=False"):
        prepare_env(deep_sea, env_params=params)
    prepare_env(with_episode_keys(deep_sea, on=False), env_params=params)


def test_static_row_collector_starts_alike_whatever_the_actions():
    """Static resets fall on ticks the schedule fixes (owner, Q4)."""
    raw, params = gymnax.make("Pendulum-v1")
    env = prepare_env(raw)[0]
    env_args = EnvironmentConfig(env=env, env_params=params, n_envs=N, continuous=True)

    def starts(push: float) -> np.ndarray:
        def policy(carry, obs, is_first, key):
            return jnp.full((N, 1), push), carry, None

        state, firsts = init_row_collector_state(jax.random.PRNGKey(0), env_args), []
        for tick in jnp.arange(12):
            state, row = collect_row(
                state,
                tick,
                policy,
                env_args=env_args,
                reset_mode="static",
                episode_length=3,
            )
            firsts.append(np.asarray(row.obs)[np.asarray(row.is_first)])
        return np.concatenate(firsts)

    np.testing.assert_array_equal(starts(-1.0), starts(1.0))


def test_episode_k_starts_alike_with_or_without_an_extension_and_whatever_the_updates():
    net = {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")}
    base = {"env_params": _Pushed.default_params, "n_envs": N, "learning_starts": 8}
    base.update({"batch_size": 4, "buffer_size": 128, "policy_update_start": 0, **net})
    expert = {"expert_policy": lambda obs: jnp.full(obs.shape[:-1] + (1,), 0.3)}
    variants = (
        {},
        {**expert, "extensions": (ExpertWarmup(fraction=0.5),)},
        {"learning_starts": 16},  # fewer updates, a longer warm-up
    )
    starts, rows = [], []
    for options in variants:
        state, _ = ajax.SAC(_Pushed(), **{**base, **options}).train(
            seed=0, n_timesteps=64
        )
        obs = np.asarray(state.collector_state.buffer_state.experience["obs"])
        obs = obs.reshape(N, -1, 2)[:, :32]  # 32 steps of each env
        firsts = [
            np.flatnonzero(obs[i, :, 1] == 0.0) for i in range(N)
        ]  # nothing pushed
        starts.append([obs[i, first, 0].tolist() for i, first in enumerate(firsts)])
        rows.append([first.tolist() for first in firsts])
    for i in range(N):
        common = min(len(run[i]) for run in starts)
        assert common >= 4 and all(
            run[i][:common] == starts[0][i][:common] for run in starts
        )
    assert (
        rows[1] != rows[0] and rows[2] != rows[0]
    )  # the variants end episodes elsewhere


def test_fresh_reset_starts_episode_k_alike_whatever_any_envs_actions():
    env = brax_training.VmapWrapper(brax_training.EpisodeWrapper(_DriftEnv(), 50, 1))
    env = BatchRngWrapper(FreshAutoResetWrapper(FinalObsWrapper(env)), n_envs=N)

    def starts(pushes: tuple) -> list:
        """Each env's starts, env i pushed pushes[i] a step."""
        state = env.reset(jax.random.PRNGKey(0))
        out = [[float(state.obs[i, 0])] for i in range(N)]
        actions = jnp.repeat(jnp.asarray(pushes)[:, None], 3, axis=1)
        for _ in range(24):
            state = env.step(state, actions)
            for i in np.flatnonzero(state.done):
                out[i].append(float(state.obs[i, 0]))
        return out

    one, other = starts((0.0, 1.0)), starts((1.0, 0.0))  # the envs swap paces
    for i in range(N):
        assert one[i][:3] == other[i][:3] and len(set(one[i])) == len(one[i])
    assert len(one[1]) > len(one[0])  # the faster env ended more episodes
