"""The "mujoco:<id>" environments: Gymnasium's MuJoCo tasks on MJX, from the
gymnasium-mjx package (the gymnasium-mjx extra).

Contract (Ajax's side only; the tasks' equivalence with Gymnasium is
gymnasium-mjx's own test suite):

- "mujoco:<id>" builds the package's task with Gymnasium's time limit (50
  steps for Reacher, 100 for Pusher, 1000 for the others) unless the caller
  gives one, and the evaluation rebuild keeps both;
- an agent's action of -1, 0 and 1 drives each motor at the low end, the
  centre and the high end of Gymnasium's action space (oracle:
  gymnasium.make(id).action_space), whatever does the mapping;
- an unknown id is refused;
- without the package, building one raises an ImportError that names the
  extra to install.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mujoco_playground")
pytest.importorskip("gymnasium_mjx")
gymnasium = pytest.importorskip("gymnasium")

from ajax.environments.create import build_env_from_id  # noqa: E402
from ajax.environments.utils import agent_action_to_env  # noqa: E402
from ajax.evaluate import setup_environment  # noqa: E402


@pytest.mark.parametrize(
    "task_id, steps",
    [("Reacher-v5", 50), ("Pusher-v5", 100), ("Swimmer-v5", 1000), ("Ant-v5", 1000)],
)
def test_an_id_builds_with_gymnasiums_time_limit(task_id, steps):
    env, env_params = build_env_from_id(f"mujoco:{task_id}")

    assert env.episode_length == steps
    assert env_params is None


def test_a_given_episode_length_wins():
    env, _ = build_env_from_id("mujoco:Reacher-v5", episode_length=20)

    assert env.episode_length == 20


def test_the_evaluation_rebuild_keeps_the_task_and_its_time_limit():
    env, env_params = build_env_from_id("mujoco:Reacher-v5")

    evaluation_env, _, _ = setup_environment(
        env, env_params, num_episodes=2, norm_info=None, gamma=0.99
    )

    assert type(evaluation_env.unwrapped.task).__name__ == "Reacher"
    assert evaluation_env.episode_length == 50


# InvertedPendulum's motor takes +-3, so a missing mapping shows; gymnasium-mjx's
# own tests check every task's bounds (the Humanoids' +-0.4 included), and its
# adapter does the mapping for all of them. Not Pusher: at the floors (MuJoCo
# 3.7) its first step compiles for about 30 minutes.
@pytest.mark.parametrize("task_id", ["InvertedPendulum-v5"])
def test_an_agents_action_drives_the_motors_across_gymnasiums_action_space(task_id):
    space = gymnasium.make(task_id).action_space
    env, env_params = build_env_from_id(f"mujoco:{task_id}", n_envs=3)
    agent_actions = jnp.stack(
        [-jnp.ones(space.shape), jnp.zeros(space.shape), jnp.ones(space.shape)]
    )

    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 3))
    state = env.step(state, agent_action_to_env(agent_actions, env, env_params))

    ctrl = np.asarray(state.data.ctrl)
    centre = (space.low + space.high) / 2
    np.testing.assert_allclose(ctrl[0], space.low, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ctrl[1], centre, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ctrl[2], space.high, rtol=0, atol=1e-6)


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError, match="No MuJoCo task 'Hopper-v4'"):
        build_env_from_id("mujoco:Hopper-v4")


def test_without_the_package_the_error_names_the_extra(monkeypatch):
    # None in sys.modules makes `import gymnasium_mjx` raise ImportError, as
    # when the package is not installed.
    monkeypatch.setitem(sys.modules, "gymnasium_mjx", None)

    with pytest.raises(ImportError, match=r'pip install "ajax-rl\[gymnasium-mjx\]"'):
        build_env_from_id("mujoco:Hopper-v5")
