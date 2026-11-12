"""Gymnasium's v5 MuJoCo tasks on MJX ("mujoco:<id>").

Contract (from the owner's request: Gymnasium's tasks, exactly, in JAX, the
same results up to an epsilon; the oracle is Gymnasium itself, run on
MuJoCo's C library):

- one step from any state Gymnasium reaches gives Gymnasium's observation,
  reward and termination, in float64 and in float32 (what training runs);
- without contacts, the same holds over many steps from the same start
  under the same actions (with contacts, tiny differences grow along a
  trajectory, so only the single step is a contract there);
- a reset draws Gymnasium's initial-state distribution;
- "mujoco:<id>" builds the task with Gymnasium's time limit (50 steps for
  Reacher, 100 for Pusher, 1000 for the others) unless the caller gives one,
  keeps it in the evaluation rebuild, and an unknown id is refused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mujoco_playground")
gymnasium = pytest.importorskip("gymnasium")

from ajax.environments.create import build_env_from_id  # noqa: E402
from ajax.environments.mujoco.tasks import TASKS, make  # noqa: E402
from ajax.evaluate import setup_environment  # noqa: E402

STEPS = 40
# Both sides run MuJoCo's equations in float64; only the order of floating
# point operations differs (XLA against C), about 1e-15 per operation. The
# pendulums amplify it as they fall, and 60 steps showed at most 9e-12:
# 1e-9 leaves that margin while a wrong term (a weight, a stale state, a
# missing clip) moves values by far more.
TOLERANCE = 1e-9


def gymnasium_steps(task_id, n_steps, solve_to_convergence=False):
    """(qpos, qvel, action, observation, reward, terminated) of each of
    n_steps random steps in Gymnasium, restarting after a termination."""
    reference = gymnasium.make(task_id).unwrapped
    if solve_to_convergence:
        # From MuJoCo 3.11, C's Newton solver stops once its Newton decrement
        # or duality gap is below opt.tolerance (commits 1e66efd11,
        # c69ef0308), within 1e-8 of the solution but not on it; MJX solves
        # on. At tolerance 0, C reaches MJX's values (Reacher: 4e-15 apart,
        # against 1.9e-8 at the default), so the float64 check stays a check
        # of the translation.
        reference.model.opt.tolerance = 0.0
    reference.reset(seed=0)
    random = np.random.default_rng(0)
    rows = []
    for episode in range(1, n_steps + 1):
        qpos, qvel = reference.data.qpos.copy(), reference.data.qvel.copy()
        action = random.uniform(reference.action_space.low, reference.action_space.high)
        observation, reward, terminated, _, _ = reference.step(action)
        rows.append((qpos, qvel, action, observation, reward, terminated))
        if terminated:
            reference.reset(seed=episode)
    return rows


ONE_STEP_TOLERANCE = {
    # A single step from the same state differs only by the order of
    # floating-point operations, and by the contact solver stopping on its
    # tolerance: at most 1.5e-10 measured (HalfCheetah, values up to ~10).
    "float64": 1e-8,
    # float32 (what training runs) carries 7 digits: 1e-6 on values of
    # ~10 for smooth steps, and the contact solver stops earlier, so
    # HalfCheetah and Walker2d reach 1.4e-4 (measured over 100 steps).
    "float32": 1e-3,
}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_one_step_from_gymnasiums_states_gives_gymnasiums_values(task_id, dtype):
    rows = gymnasium_steps(
        task_id, n_steps=100, solve_to_convergence=dtype == "float64"
    )
    tolerance = ONE_STEP_TOLERANCE[dtype]

    with jax.enable_x64(dtype == "float64"):
        task = make(task_id)
        template = task.reset(jax.random.key(0))

        @jax.jit
        def one_step(qpos, qvel, action):
            state = template.replace(data=task.set_state(qpos, qvel))
            return task.step(state, action)

        for qpos, qvel, action, observation, reward, terminated in rows:
            state = one_step(*(jnp.asarray(value) for value in (qpos, qvel, action)))
            np.testing.assert_allclose(state.obs, observation, rtol=0, atol=tolerance)
            np.testing.assert_allclose(state.reward, reward, rtol=0, atol=tolerance)
            assert bool(state.done) == terminated


HEALTH_CASES = [
    # hopper_v5: z > 0.7, |angle| < 0.2, every state entry but x and z
    # within 100 (here the thigh's velocity).
    ("Hopper-v5", {"z": 0.71}, True),
    ("Hopper-v5", {"z": 0.69}, False),
    ("Hopper-v5", {"angle": 0.19}, True),
    ("Hopper-v5", {"angle": -0.21}, False),
    ("Hopper-v5", {"qvel3": 99.0}, True),
    ("Hopper-v5", {"qvel3": 101.0}, False),
    # walker2d_v5: 0.8 < z < 2, |angle| < 1, no state range.
    ("Walker2d-v5", {"z": 0.81}, True),
    ("Walker2d-v5", {"z": 0.79}, False),
    ("Walker2d-v5", {"z": 2.01}, False),
    ("Walker2d-v5", {"angle": 0.99}, True),
    ("Walker2d-v5", {"angle": 1.01}, False),
    ("Walker2d-v5", {"qvel3": 1000.0}, True),
]


@pytest.mark.parametrize("task_id, change, healthy", HEALTH_CASES)
def test_health_follows_gymnasiums_ranges(task_id, change, healthy):
    task = make(task_id)
    qpos = (
        task.init_qpos.at[1]
        .set(change.get("z", 1.25))
        .at[2]
        .set(change.get("angle", 0.0))
    )
    qvel = task.init_qvel.at[3].set(change.get("qvel3", 0.0))

    assert bool(task.is_healthy(task.set_state(qpos, qvel))) == healthy


CONTACT_FREE = [
    "InvertedDoublePendulum-v5",
    "InvertedPendulum-v5",
    "Reacher-v5",
    "Swimmer-v5",
]


@pytest.mark.parametrize("task_id", CONTACT_FREE)
def test_a_task_steps_as_gymnasium_does(task_id):
    reference = gymnasium.make(task_id).unwrapped
    first_observation, _ = reference.reset(seed=0)
    start = (reference.data.qpos.copy(), reference.data.qvel.copy())
    actions = np.random.default_rng(0).uniform(
        reference.action_space.low,
        reference.action_space.high,
        (STEPS, reference.action_space.shape[0]),
    )
    expected = [reference.step(action)[:3] for action in actions]

    with jax.enable_x64(True):
        task = make(task_id)
        reset = task.reset(jax.random.key(0))
        # The reference's reset draw, placed in the task's state.
        data = task.set_state(jnp.asarray(start[0]), jnp.asarray(start[1]))
        state = reset.replace(data=data, obs=task.observe(data))
        step = jax.jit(task.step)
        observed = []
        for action in actions:
            state = step(state, jnp.asarray(action))
            observed.append((state.obs, state.reward, state.done))

        assert reset.obs.shape == first_observation.shape
        np.testing.assert_allclose(
            task.observe(data), first_observation, rtol=0, atol=TOLERANCE
        )
    for (observation, reward, done), (gym_obs, gym_reward, gym_terminated) in zip(
        observed, expected, strict=False
    ):
        np.testing.assert_allclose(observation, gym_obs, rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(reward, gym_reward, rtol=0, atol=TOLERANCE)
        assert bool(done) == gym_terminated


@pytest.mark.parametrize(
    "task_id", ["InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
)
def test_the_comparison_reaches_terminations(task_id):
    """Guards the test above: under its actions the pendulums fall within
    its horizon, so the termination rule is compared, not only skipped."""
    reference = gymnasium.make(task_id).unwrapped
    reference.reset(seed=0)
    actions = np.random.default_rng(0).uniform(
        reference.action_space.low,
        reference.action_space.high,
        (STEPS, reference.action_space.shape[0]),
    )

    terminations = [reference.step(action)[2] for action in actions]

    assert any(terminations)


SAMPLES = 4096


def resets(task_id):
    task = make(task_id)
    keys = jax.random.split(jax.random.key(1), SAMPLES)
    data = jax.vmap(task.reset)(keys).data
    return task, np.asarray(data.qpos), np.asarray(data.qvel)


def test_reacher_draws_its_goal_uniformly_in_the_disc_of_radius_0_2():
    """reacher_v5: arm angles uniform within 0.1, goal uniform in the open
    disc of radius 0.2, the goal's velocity zero. Uniform in a disc of
    radius R: mean 0, E[r^2] = R^2 / 2 = 0.02, Var[r^2] = R^4 / 12."""
    _, qpos, qvel = resets("Reacher-v5")
    goal = qpos[:, -2:]
    squared_radius = np.sum(goal**2, axis=1)

    assert np.all(np.abs(qpos[:, :2]) <= 0.1)
    assert np.all(squared_radius < 0.04)
    assert np.all(qvel[:, -2:] == 0)
    assert np.all(np.abs(qvel[:, :2]) <= 0.005)
    # Four standard errors: a false alarm in about 1 run of 16,000.
    np.testing.assert_allclose(goal.mean(axis=0), 0, atol=4 * 0.1 / np.sqrt(SAMPLES))
    np.testing.assert_allclose(
        squared_radius.mean(), 0.02, atol=4 * np.sqrt(0.2**4 / 12 / SAMPLES)
    )


def test_pusher_places_its_object_away_from_the_goal():
    """pusher_v5: the object uniform in [-0.3, 0] x [-0.2, 0.2], redrawn
    until more than 0.17 from the goal at the origin; the arm unperturbed,
    the object and goal still."""
    task, qpos, qvel = resets("Pusher-v5")
    place = qpos[:, -4:-2]

    assert np.all(np.linalg.norm(place, axis=1) > 0.17)
    assert np.all((place[:, 0] >= -0.3) & (place[:, 0] <= 0))
    assert np.all(np.abs(place[:, 1]) <= 0.2)
    np.testing.assert_array_equal(qpos[:, -2:], 0)
    np.testing.assert_array_equal(
        qpos[:, :7], np.broadcast_to(task.init_qpos[:7], (SAMPLES, 7))
    )
    np.testing.assert_array_equal(qvel[:, -4:], 0)
    # The rectangle less the disc is symmetric in y: the mean of y is 0.
    np.testing.assert_allclose(place[:, 1].mean(), 0, atol=4 * 0.2 / np.sqrt(SAMPLES))


def test_inverted_double_pendulum_draws_normal_velocities_of_scale_0_1():
    """inverted_double_pendulum_v5: positions uniform within 0.1 of the
    initial ones, velocities normal with standard deviation 0.1."""
    task, qpos, qvel = resets("InvertedDoublePendulum-v5")

    assert np.all(np.abs(qpos - np.asarray(task.init_qpos)) <= 0.1)
    # The sample standard deviation's standard error is about sigma / sqrt(2n).
    np.testing.assert_allclose(
        qvel.std(axis=0), 0.1, atol=4 * 0.1 / np.sqrt(2 * SAMPLES)
    )


@pytest.mark.parametrize(
    "task_id, steps", [("Reacher-v5", 50), ("Pusher-v5", 100), ("Swimmer-v5", 1000)]
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

    assert type(evaluation_env.unwrapped).__name__ == "Reacher"
    assert evaluation_env.episode_length == 50


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError, match="No MuJoCo task 'Hopper-v4'"):
        build_env_from_id("mujoco:Hopper-v4")
