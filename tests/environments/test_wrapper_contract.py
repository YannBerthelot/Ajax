"""The contract every Ajax env wrapper keeps, found by base class (an Ajax
subclass of GymnaxWrapper or of brax's Wrapper without a row below fails):

(a) copy, deepcopy and pickle give a wrapper of the same type that
reproduces the original's reset and step;
(b) on the identity task (tests/environments/identity_env.py), whose
episodes terminate at t = 3 or are truncated at t = 5, it reports the
flags each step raised, alone and in create.py's stacks;
(c) its final observation in ``info`` (what a collector bootstraps a time
limit on) is as wide as the observation and, where no episode ended,
equal to it: a wrapper that transforms observations transforms it too.

A gymnax wrapper wraps the identity env; a brax one, a stand-in for it
under the brax or playground stack it needs (EpisodeWrapper's time limit,
VmapWrapper, FinalObsWrapper and an auto-reset).
"""

import copy
import importlib
import pickle
import pkgutil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import Env, State
from brax.envs.wrappers import training as brax_training
from mujoco_playground._src.mjx_env import State as PlaygroundState

import ajax.environments
import ajax.wrappers as W
from ajax.environments import interaction
from ajax.environments.create import add_ajax_wrappers
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.utils import env_keys, get_env_type
from tests.environments.identity_env import LIMIT, TERMINAL, IdentityEnv, ends

N_ENVS, N_STEPS = 8, 12


class BraxIdentity(Env):
    """The identity task on brax: [t, c] in the state's ``pipeline_state``
    (``data`` on playground), terminating at TERMINAL when c > 0; the
    stack's EpisodeWrapper truncates at LIMIT. Its metrics keep the [t, c]
    a step was taken from."""

    def __init__(self, playground: bool = False):
        self.playground = playground

    def _state(self, x, before, reward, done, info):
        out = {"obs": x, "reward": reward, "done": done, "info": info}
        metrics = {"t": before[0], "c": before[1]}
        if self.playground:
            return PlaygroundState(data=x, metrics=metrics, **out)
        return State(pipeline_state=x, metrics=metrics, **out)

    def reset(self, rng):
        x = jnp.stack([0.0, jax.random.uniform(rng, minval=-1.0, maxval=1.0)])
        return self._state(x, x, jnp.zeros(()), jnp.zeros(()), {})

    def step(self, state, action):
        before = state.data if self.playground else state.pipeline_state
        x = before + jnp.array([1.0, 0.0])
        done = ((x[1] > 0) & (x[0] >= TERMINAL)).astype(jnp.float32)
        return self._state(x, before, action[0], done, state.info)

    observation_size = 2
    action_size = 1
    backend = "identity"


def brax_stack():
    """``_build_brax_env``'s stack."""
    env = brax_training.EpisodeWrapper(BraxIdentity(), LIMIT, action_repeat=1)
    env = brax_training.VmapWrapper(env, batch_size=N_ENVS)
    return W.AutoResetWrapper(W.FinalObsWrapper(env))


def playground_stack():
    """``_build_playground_env``'s stack."""
    env = brax_training.EpisodeWrapper(BraxIdentity(True), LIMIT, action_repeat=1)
    env = W.FreshAutoResetWrapper(W.FinalObsWrapper(brax_training.VmapWrapper(env)))
    return W.BatchRngWrapper(env, n_envs=N_ENVS)


def redraw_c(key, state, params):
    return state.replace(c=jax.random.uniform(key, minval=-1.0, maxval=1.0))


ROWS = {
    W.FlattenObservationWrapper: lambda: W.FlattenObservationWrapper(IdentityEnv()),
    W.ClipAction: lambda: W.ClipAction(IdentityEnv()),
    W.InitialStateWrapper: lambda: W.InitialStateWrapper(IdentityEnv(), redraw_c),
    W.NormalizeVecObservationGymnax: lambda: W.NormalizeVecObservationGymnax(
        IdentityEnv(), gamma=0.9
    ),
    W.TerminatedTruncatedWrapper: lambda: W.TerminatedTruncatedWrapper(IdentityEnv()),
    W.EpisodeKeyedAutoResetWrapper: lambda: W.EpisodeKeyedAutoResetWrapper(
        IdentityEnv()
    ),
    W.ObservationNoiseWrapper: lambda: W.ObservationNoiseWrapper(IdentityEnv(), 0.1),
    ModelReferenceWrapper: lambda: ModelReferenceWrapper(
        IdentityEnv(),
        StepReference(LIMIT, -1.0, 1.0, 1, 2),
        LinearReferenceModel.first_order(),
    ),
    W.AutoResetWrapper: brax_stack,
    W.FinalObsWrapper: brax_stack,
    W.FreshAutoResetWrapper: playground_stack,
    W.BatchRngWrapper: playground_stack,
    W.ClipActionBrax: lambda: W.ClipActionBrax(brax_stack()),
    W.NormalizeVecObservationBrax: lambda: W.NormalizeVecObservationBrax(
        brax_stack(), gamma=0.9
    ),
    W.ObservationNoiseWrapperBrax: lambda: W.ObservationNoiseWrapperBrax(
        brax_stack(), 0.1
    ),
}
_NORMALISED: dict[str, Any] = {
    "normalize_obs": True,
    "normalize_reward": True,
    "gamma": 0.9,
}
STACKS = {
    "gymnax": lambda: add_ajax_wrappers(
        W.FlattenObservationWrapper(IdentityEnv()), **_NORMALISED
    ),
    "brax": lambda: add_ajax_wrappers(brax_stack(), **_NORMALISED),
    "playground": lambda: add_ajax_wrappers(playground_stack(), **_NORMALISED),
}
BUILDERS = {cls.__name__: build for cls, build in ROWS.items()}
BUILDERS.update(STACKS)


def ajax_wrappers() -> set:
    """Every subclass of the two wrapper bases defined in ajax.wrappers or
    ajax.environments."""
    for module in pkgutil.iter_modules(ajax.environments.__path__):
        importlib.import_module(f"ajax.environments.{module.name}")
    found, todo = set(), [W.GymnaxWrapper, W.BraxWrapper]
    while todo:
        subclasses = todo.pop().__subclasses__()
        todo += subclasses
        found.update({c for c in subclasses if c.__module__.startswith("ajax.")})
    return found


def test_every_wrapper_has_a_row():
    assert ajax_wrappers() == set(ROWS)


def _run(env, n_steps: int):
    """Reset and ``n_steps`` steps (action 0.5) on fixed keys: every step's
    ``(obs, state, reward, terminated, truncated, info)``."""
    mode = get_env_type(env)
    params = getattr(env, "default_params", None) if mode == "gymnax" else None
    keys = jax.random.split(jax.random.PRNGKey(0), n_steps + 1)
    out = [interaction.reset(env_keys(keys[0], mode, N_ENVS), env, mode, params)]
    for key in keys[1:]:
        key = env_keys(key, mode, N_ENVS)
        action = jnp.full((N_ENVS, 1), 0.5)
        out.append(interaction.step(key, out[-1][1], action, env, mode, params))
    return mode, out


def _pickled(env):
    return pickle.loads(pickle.dumps(env))


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, _pickled],
    ids=["copy", "deepcopy", "pickle"],
)
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_wrapper_duplicates(name, duplicate):
    env = BUILDERS[name]()
    twin = duplicate(env)
    assert type(twin) is type(env)
    for ours, theirs in zip(_run(env, 1)[1], _run(twin, 1)[1], strict=True):
        for a, b in zip(jax.tree.leaves(ours), jax.tree.leaves(theirs), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_wrapper_reports_the_flags_each_step_raised(name):
    mode, steps = _run(BUILDERS[name](), N_STEPS)
    seen = np.zeros(2, bool)
    for _, state, _, terminated, truncated, info in steps[1:]:
        if mode == "brax":  # from the [t, c] the stand-in stepped from
            expected = ends(np.stack([state.metrics["t"], state.metrics["c"]], -1))
        else:
            expected = info["terminated"], info["truncated"]
        np.testing.assert_array_equal(terminated > 0, expected[0])
        np.testing.assert_array_equal(truncated > 0, expected[1])
        seen |= [np.any(expected[0]), np.any(expected[1])]
    assert seen.all(), "no termination or no truncation in the run"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_wrapper_reports_its_own_final_observation(name):
    _, steps = _run(BUILDERS[name](), N_STEPS)
    for obs, _, _, terminated, truncated, info in steps[1:]:
        final = interaction.get_final_obs(info, None)
        assert final is not None, "no final observation in info"
        assert final.shape == obs.shape
        going = ~((terminated > 0) | (truncated > 0))
        np.testing.assert_allclose(final[going], obs[going], rtol=1e-6)
