"""Training on MuJoCo Warp physics (``physics="warp"``), on an NVIDIA GPU.

Contract (from the task: Ajax's agents train on Warp on a GPU; Warp is a port
of MuJoCo, so C MuJoCo is its reference):

- through Ajax's environment stack, auto-resets included, every transition
  is C MuJoCo's from the state before it, within float32 rounding;
- each family of agents trains on Warp, its run ending with finite
  evaluations: SAC (off-policy), PPO and APO (on-policy), TD-MPC2 and
  DreamerV3 (model-based);
- the contact buffer, which every world of a batch shares, holds a
  benchmark's batch, and a batch too big for it stops the run.

Large tests: they need a GPU, which CI's runners lack, and are skipped
without one. On a GPU machine: ``uv run pytest
tests/environments/test_warp_physics.py``.
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from ajax import APO, PPO, SAC
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.environments.create import build_env_from_id
from ajax.environments.mujoco import tasks
from ajax.logging import LoggingConfig
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY

pytest.importorskip("mujoco_playground")
pytestmark = [
    pytest.mark.large,
    pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="MuJoCo Warp needs an NVIDIA GPU"
    ),
]

#: Three episodes per env, the second and third started by the auto-reset.
N_ENVS, EPISODE_LENGTH, N_STEPS = 2, 5, 15
#: Warp against C MuJoCo over one transition from the same state, measured on
#: an L40S over these rollouts: at most 4e-7 on CheetahRun and 4e-6 on
#: HalfCheetah-v5 (float32 against float64), in joint positions and in
#: velocities times the step's duration (how far they move a joint in it).
#: 1e-4 is above that, and far below what a corrupted state does to a step.
TOLERANCE = 1e-4


def warp_rollout(task: str) -> tuple:
    """A fixed-action rollout through Ajax's stack on Warp, vmapped over two
    seeds as agents run: the raw env, the actions, the physics state (joint
    positions, velocities, solver warm start) it starts from and after each
    step, and the episode ends."""
    env, _ = build_env_from_id(
        task, n_envs=N_ENVS, episode_length=EPISODE_LENGTH, physics="warp"
    )
    shape = (N_STEPS, N_ENVS, env.action_size)
    actions = 0.5 * jax.random.uniform(jax.random.PRNGKey(1), shape, minval=-1)

    def physics(state) -> tuple:
        return state.data.qpos, state.data.qvel, state.data.qacc_warmstart

    def run(seed: jax.Array) -> tuple:
        def body(state, action):
            state = env.step(state, action)
            return state, (physics(state), state.done)

        first = env.reset(jax.random.PRNGKey(seed))
        return physics(first), jax.lax.scan(body, first, actions)[1]

    rollout = jax.jit(jax.vmap(run))(jnp.arange(2))
    first, (path, done) = jax.tree.map(np.asarray, rollout)
    return env.unwrapped, np.asarray(actions), first, path, done


def c_mujoco_replay(raw, actions, first, path, done) -> tuple:
    """Each transition that does not end an episode (its state is the next
    episode's reset), as C MuJoCo steps it from the state before it, and as
    the rollout reported it: joint positions and velocities times the step's
    duration, stacked."""
    model = raw.mj_model
    data = mujoco.MjData(model)
    expected, reported = [], []
    for seed in range(done.shape[0]):
        for env_index in range(N_ENVS):
            before = [leaf[seed, env_index] for leaf in first]
            for t in range(N_STEPS):
                after = [leaf[seed, t, env_index] for leaf in path]
                if not done[seed, t, env_index]:
                    data.qpos[:], data.qvel[:], data.qacc_warmstart[:] = before
                    data.ctrl[:] = actions[t, env_index]
                    mujoco.mj_step(model, data, nstep=raw.n_substeps)
                    expected.append(np.concatenate([data.qpos, data.qvel * raw.dt]))
                    reported.append(np.concatenate([after[0], after[1] * raw.dt]))
                before = after
    return np.stack(expected), np.stack(reported)


@pytest.mark.parametrize("task", ["CheetahRun", "mujoco:HalfCheetah-v5"])
def test_every_transition_on_warp_is_c_mujocos(task: str):
    """Catches the auto-reset leaving Warp a corrupted state, its contact
    buffer above all (kept for the whole batch, it is merged by no env):
    each step the stack reports, after a reset or not, is C MuJoCo's step
    from the state before it. Both tasks touch the ground from the first
    step, so the buffer is in use."""
    raw, actions, first, path, done = warp_rollout(task)

    expected, reported = c_mujoco_replay(raw, actions, first, path, done)

    assert done.sum() == 2 * N_ENVS * (N_STEPS // EPISODE_LENGTH)
    assert len(expected) == 2 * N_ENVS * N_STEPS - done.sum()
    np.testing.assert_allclose(reported, expected, rtol=0, atol=TOLERANCE)


def hopper_rollout(n_seeds: int, n_envs: int, steps: int) -> jax.Array:
    """Random actions on mujoco:Hopper-v5 through Ajax's stack on Warp, vmapped
    over seeds; the most contact slots the batch used at each step."""
    env, _ = build_env_from_id("mujoco:Hopper-v5", n_envs=n_envs, physics="warp")
    shape = (steps, n_envs, env.action_size)

    def run(seed: jax.Array) -> jax.Array:
        def body(state, action):
            state = env.step(state, action)
            warp = state.data._impl
            return state, jnp.maximum(warp.nacon, warp.ncollision)

        reset_key, action_key = jax.random.split(jax.random.PRNGKey(seed))
        actions = jax.random.uniform(action_key, shape, minval=-1)
        return jax.lax.scan(body, env.reset(reset_key), actions)[1]

    return jax.block_until_ready(jax.jit(jax.vmap(run))(jnp.arange(n_seeds)))


def test_the_contact_buffer_holds_a_thousand_worlds():
    """Catches a buffer sized for one world: MuJoCo Warp's default, 48
    contacts for the whole batch, overflowed at 47 Hopper worlds."""
    used = hopper_rollout(n_seeds=10, n_envs=100, steps=200)

    assert int(used.max()) <= tasks.WARP_CONTACTS


def test_a_batch_too_big_for_the_contact_buffer_stops_the_run(monkeypatch):
    """Catches dropped contacts going on unnoticed: at MuJoCo Warp's default
    of 48 contacts, 60 Hopper worlds overflow on their first steps."""
    monkeypatch.setattr(tasks, "WARP_CONTACTS", 48)

    with pytest.raises(jax.errors.JaxRuntimeError, match="Warp dropped contacts"):
        hopper_rollout(n_seeds=1, n_envs=60, steps=20)


N_TIMESTEPS = 1024
AGENTS = {
    "SAC": (SAC, "mujoco:Hopper-v5", {"learning_starts": 256}),
    "PPO": (PPO, "mujoco:Hopper-v5", {"n_steps": 64}),
    "APO": (APO, "mujoco:Hopper-v5", {"n_steps": 64}),
    # TD-MPC2 is benchmarked on tasks without terminations (CheetahRun is one).
    "TDMPC2": (TDMPC2, "CheetahRun", {**TDMPC2_TINY, "seed_steps": 256}),
    "DreamerV3": (DreamerV3, "CheetahRun", DREAMER_TINY),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(AGENTS))
def test_an_agent_trains_on_warp_physics(name: str):
    """Catches an agent family whose training or evaluation does not run on
    Warp (before the fix, every agent failed in the auto-reset on its first
    step)."""
    cls, task, kwargs = AGENTS[name]
    env, _ = build_env_from_id(task, n_envs=4, physics="warp")
    agent = cls(env, n_envs=4, **kwargs)

    _, evaluations = agent.train(
        seed=[0, 1],
        n_timesteps=N_TIMESTEPS,
        num_episode_test=2,
        logging_config=LoggingConfig(log_frequency=N_TIMESTEPS, backends=()),
    )

    returns = np.asarray(evaluations["Eval/episodic mean reward"])
    assert returns.shape[0] == 2
    assert np.all(np.isfinite(returns[:, -1]))
