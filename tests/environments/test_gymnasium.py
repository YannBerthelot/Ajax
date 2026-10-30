"""Gymnasium environments, stepped in Python (ajax.environments.gymnasium).

Contract (the owner's request of 2026-10-29, the approved plan, Gymnasium's
``Env`` API and GymnasiumEnv's docstrings):

- ``"gymnasium:<id>"`` builds the Gymnasium environment ``<id>``; an id
  without the prefix never does (no fallback) and raises no warning.
- Through Ajax's reset and step, each copy behaves as
  ``gymnasium.make(id)`` reset with ``seed=jax.random.bits(key, uint32)`` of
  its reset key: the same observations, rewards, terminated and truncated
  flags; at an episode end the final observation is in
  ``info["final_observation"]`` and the observation returned is the next
  episode's first (Gymnasium's own reset, unseeded).
- Evaluation's environment (``evaluate.setup_environment``) steps copies of its
  own: the training copies' episodes go on where they were.
- Building one warns (``PythonEnvironmentWarning``) and names the closest
  JAX tasks of the comparison table, which are registered in their backends.
- Refused, each with an inherent reason: gradients through the env (APG), a
  resume from a saved state, action repeat, spaces Ajax's networks cannot
  take, an env without a time limit.
- Agents train on it end to end, the seeds one after another, each seed the
  run it would be alone (slow tier).
"""

import warnings

import brax.envs
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground import registry as playground_registry

from ajax import APG, PPO, SAC
from ajax.environments.create import build_env_from_id
from ajax.environments.gymnasium import (
    COUNTERPARTS,
    GymnasiumEnv,
    PythonEnvironmentWarning,
)
from ajax.environments.interaction import reset, step
from ajax.evaluate import setup_environment
from ajax.logging.wandb_logging import LoggingConfig

# CartPole pushed one way terminates within a dozen steps; alternating
# actions keeps it up past a 12-step limit, which then truncates.
LIMIT = 12
ACTIONS = {
    "always right": [1] * 40,
    "alternating": [0, 1] * 20,
    "always left": [0] * 40,
}


class _Toy(gymnasium.Env):
    """A one-step task whose spaces the tests choose."""

    def __init__(self, observation_space=None, action_space=None):
        self.observation_space = observation_space or gymnasium.spaces.Box(-1, 1, (2,))
        self.action_space = action_space or gymnasium.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def _make_toy(**spaces) -> gymnasium.Env:
    return _Toy(**spaces)


gymnasium.register("AjaxTest/Toy-v0", entry_point=_make_toy, max_episode_steps=5)
gymnasium.register("AjaxTest/NoTimeLimit-v0", entry_point=_make_toy)


def _gymnasium_rollout(seed: int, actions: list) -> dict:
    """The reference: Gymnasium's own CartPole under the same actions."""
    env = gymnasium.make("CartPole-v1", max_episode_steps=LIMIT)
    obs, _ = env.reset(seed=seed)
    out: dict[str, list] = {
        "obs": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "final": [],
    }
    for action in actions:
        final, reward, terminated, truncated, _ = env.step(action)
        obs = env.reset()[0] if terminated or truncated else final
        for key, value in zip(
            out, (obs, reward, terminated, truncated, final), strict=True
        ):
            out[key].append(value)
    return {key: np.asarray(value) for key, value in out.items()}


def _ajax_rollout(env, keys, steps: int, first: int = 0, state=None) -> tuple:
    """Ajax's reset (unless given a state) then ``steps`` steps of ACTIONS
    from ``first``, through the functions every agent collects with."""
    params = env.default_params
    if state is None:
        _, state = reset(keys, env, "gymnax", params)
    actions = jnp.asarray(list(ACTIONS.values()), jnp.float32)
    out: dict[str, list] = {
        "obs": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "final": [],
    }
    for t in range(first, first + steps):
        obs, state, reward, terminated, truncated, info = step(
            keys, state, actions[:, t], env, "gymnax", params
        )
        values = (obs, reward, terminated, truncated, info["final_observation"])
        for key, value in zip(out, values, strict=True):
            out[key].append(np.asarray(value))
    return state, {key: np.stack(value, axis=1) for key, value in out.items()}


def _seeds(keys) -> list:
    return [int(jax.random.bits(key, dtype=jnp.uint32)) for key in keys]


def _assert_copies_follow_gymnasium(ajax_out: dict, keys, first: int = 0) -> None:
    for i, (seed, actions) in enumerate(
        zip(_seeds(keys), ACTIONS.values(), strict=True)
    ):
        expected = _gymnasium_rollout(seed, actions)
        steps = ajax_out["obs"].shape[1]
        for key, value in ajax_out.items():
            # Gymnasium's CartPole computes in float64 and returns float32
            # observations; Ajax keeps float32 throughout, so they are equal.
            np.testing.assert_array_equal(
                value[i],
                expected[key][first : first + steps],
                err_msg=f"copy {i}: {key}",
            )


def test_copies_reproduce_gymnasium_transitions_across_episode_ends():
    env = GymnasiumEnv("CartPole-v1", max_episode_steps=LIMIT)
    keys = jax.random.split(jax.random.PRNGKey(0), len(ACTIONS))

    _, out = _ajax_rollout(env, keys, steps=30)

    # The rollout crosses both kinds of episode end.
    assert out["terminated"].any() and out["truncated"].any()
    _assert_copies_follow_gymnasium(out, keys)


def test_evaluation_steps_copies_of_its_own():
    env = GymnasiumEnv("CartPole-v1", max_episode_steps=LIMIT)
    keys = jax.random.split(jax.random.PRNGKey(1), len(ACTIONS))
    state, before = _ajax_rollout(env, keys, steps=4)

    eval_env, mode, _ = setup_environment(env, env.default_params, 3, None, 0.99)
    eval_keys = jax.random.split(jax.random.PRNGKey(2), 3)
    _ajax_rollout(eval_env, eval_keys, steps=6)
    _, after = _ajax_rollout(env, keys, steps=20, first=4, state=state)

    assert mode == "gymnax"
    _assert_copies_follow_gymnasium(before, keys)
    _assert_copies_follow_gymnasium(after, keys, first=4)


def test_the_prefix_is_the_only_way_in():
    with pytest.warns(PythonEnvironmentWarning):
        env, params = build_env_from_id("gymnasium:CartPole-v1")
    assert isinstance(env, GymnasiumEnv)
    assert params.max_steps_in_episode == 500  # CartPole-v1's registered limit

    with warnings.catch_warnings():
        warnings.simplefilter("error", PythonEnvironmentWarning)
        env, _ = build_env_from_id("CartPole-v1")
    assert not isinstance(env, GymnasiumEnv)


def test_the_warning_names_the_closest_jax_tasks():
    with pytest.warns(PythonEnvironmentWarning) as caught:
        GymnasiumEnv("Hopper-v5")
    message = str(caught[0].message)
    for env_id in ('"hopper"', '"HopperHop"', '"HopperStand"'):
        assert env_id in message
    assert "Hopper-v5" in message


def test_an_env_without_a_counterpart_gets_no_suggestion():
    with pytest.warns(PythonEnvironmentWarning) as caught:
        GymnasiumEnv("AjaxTest/Toy-v0")
    message = str(caught[0].message)
    for backend in ("gymnax", "Brax", "Playground"):
        assert backend not in message


@pytest.mark.parametrize(
    "name,backend,env_id",
    [
        (name, counterpart.backend, env_id)
        for name, (_, counterparts) in COUNTERPARTS.items()
        for counterpart in counterparts
        for env_id in counterpart.env_ids
    ],
)
def test_every_suggested_task_is_registered(name, backend, env_id):
    registered = {
        "gymnax": gymnax.registered_envs,
        # brax lists its environments only in this private registry.
        "Brax": brax.envs._envs,
        "MuJoCo Playground": playground_registry.ALL_ENVS,
    }
    assert env_id in registered[backend], f"{name}: {backend} {env_id}"


@pytest.mark.parametrize("compared_with", [entry[0] for entry in COUNTERPARTS.values()])
def test_every_compared_task_is_registered_in_gymnasium(compared_with):
    assert compared_with in gymnasium.registry


def test_gradients_through_the_env_are_refused():
    with (
        pytest.warns(PythonEnvironmentWarning),
        pytest.raises(ValueError, match="transition gradients"),
    ):
        APG("gymnasium:Pendulum-v1")


def test_resuming_from_a_saved_state_is_refused():
    with pytest.warns(PythonEnvironmentWarning):
        agent = SAC("gymnasium:Pendulum-v1", learning_starts=100)
    state, _ = agent.train(seed=0, n_timesteps=4)
    with pytest.raises(ValueError, match="cannot resume"):
        agent.train(seed=0, n_timesteps=4, initial_state=state)


def test_action_repeat_is_refused():
    with pytest.raises(NotImplementedError, match="action_repeat"):
        build_env_from_id("gymnasium:Pendulum-v1", action_repeat=2)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "observation_space": gymnasium.spaces.Dict(
                    {"x": gymnasium.spaces.Box(0, 1)}
                )
            },
            "observations",
        ),
        ({"action_space": gymnasium.spaces.MultiDiscrete([2, 3])}, "actions"),
        ({"action_space": gymnasium.spaces.Box(-1, 1, (2, 2))}, "actions"),
    ],
)
def test_spaces_the_networks_cannot_take_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GymnasiumEnv("AjaxTest/Toy-v0", **kwargs)


def test_an_env_without_a_time_limit_is_refused():
    with pytest.raises(ValueError, match="time limit"):
        GymnasiumEnv("AjaxTest/NoTimeLimit-v0")


def _train(agent_class, env_id: str, seeds: list, **kwargs) -> tuple:
    with pytest.warns(PythonEnvironmentWarning):
        agent = agent_class(env_id, n_envs=2, **kwargs)
    logging = LoggingConfig(log_frequency=256, backends=())
    return agent.train(
        seed=seeds, n_timesteps=512, logging_config=logging, num_episode_test=2
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "agent_class,env_id,kwargs",
    [
        # 2 x 64-step rollouts, so that 512 steps reach the evaluations.
        (PPO, "gymnasium:CartPole-v1", {"n_steps": 64, "batch_size": 64}),
        (SAC, "gymnasium:Pendulum-v1", {"learning_starts": 128}),
    ],
)
def test_agents_train_seed_by_seed_as_alone(agent_class, env_id, kwargs):
    both, both_out = _train(agent_class, env_id, [0, 1], **kwargs)
    alone = [_train(agent_class, env_id, [seed], **kwargs)[0] for seed in (0, 1)]

    assert np.isfinite(np.asarray(both_out["Eval/episodic mean reward"])).all()
    # The states' arrays: their static parts hold each agent's own optimiser
    # functions.
    for i, single in enumerate(alone):
        leaves = zip(jax.tree.leaves(both), jax.tree.leaves(single), strict=True)
        for leaf_of_both, leaf_alone in leaves:
            np.testing.assert_array_equal(leaf_of_both[i], leaf_alone[0])


@pytest.mark.slow
def test_sac_trains_on_a_mujoco_task():
    _, out = _train(SAC, "gymnasium:HalfCheetah-v5", [0], learning_starts=128)
    assert np.isfinite(np.asarray(out["Eval/episodic mean reward"])).all()
