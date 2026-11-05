"""TargetGym's PID experts drive an Ajax agent through their batched form.

Contract (docs/guide/expert-guidance.md): an agent's stateful expert answers
``init_state(n_envs)`` and ``expert(state, obs) -> (action, state)``.
TargetGym's ``make_expert`` returns a user-facing object of another form
(``reset``, ``step``); ``REGISTRY[name].make_pid_controller().batched()`` is
the adapter. The realistic bug: a TargetGym release drops or renames that
form, and the benchmark's arms stop running. Runs with the ``plane`` group.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.buffers import dataset as ds
from ajax.extensions.expert import JSRLCurriculum, ResidualPolicy
from ajax.extensions.pretrain import PriorData

target_gym = pytest.importorskip("target_gym")
registry = pytest.importorskip("target_gym.registry")

TASK = "four_tank"
EPISODE = 12  # steps: the rollouts' cost, and whole episodes in 32 rows
SHORT: dict[str, Any] = {"learning_starts": 4, "batch_size": 4, "buffer_size": 64}


def test_the_batched_pid_is_a_stateful_expert_of_one_action_per_env() -> None:
    env, params = target_gym.make(TASK)
    expert = registry.REGISTRY[TASK].make_pid_controller().batched()
    n_actions = env.action_space(params).shape[0]
    obs = jnp.stack([env.reset(jax.random.PRNGKey(i), params)[0] for i in range(3)])
    action, state = expert(expert.init_state(3), obs)
    assert action.shape == (3, n_actions)
    jax.tree.map(lambda x: np.testing.assert_equal(x.shape[0], 3), state)


def test_an_agent_trains_with_it_and_stores_its_actions() -> None:
    env, params = target_gym.make(TASK)
    expert = registry.REGISTRY[TASK].make_pid_controller().batched()
    agent = SAC(env, env_params=params, n_envs=2, expert_policy=expert, **SHORT)
    state, _ = agent.train(seed=0, n_timesteps=16)
    replay = state.collector_state.buffer_state
    assert replay is not None
    a_expert = np.asarray(replay.experience["a_expert"])[..., :8, :]
    assert np.isfinite(a_expert).all() and np.abs(a_expert).max() <= 1.0 + 1e-6
    assert a_expert.std() > 0  # the PID's own actions, not an empty column


def _agent(extensions: tuple, expert: bool = True, **kwargs: Any) -> Any:
    env, params = target_gym.make(TASK, max_steps_in_episode=EPISODE)
    if expert:
        kwargs["expert_policy"] = _pid()
    return SAC(
        env,
        env_params=params,
        n_envs=1,
        extensions=extensions,
        **{**SHORT, "buffer_size": 512, **kwargs},
    )


def _pid() -> Any:
    return registry.REGISTRY[TASK].make_pid_controller().batched()


@pytest.mark.parametrize(
    "extension",
    [
        JSRLCurriculum(eval_every=8, eval_episodes=2, moving_average=2),
        JSRLCurriculum(schedule="random"),
        ResidualPolicy(zero_start=True),
        ResidualPolicy(zero_start=True, critic_reads_residual=True),
    ],
    ids=["jsrl", "jsrl-random", "residual", "residual-reads-residual"],
)
def test_each_expert_arm_runs_on_a_real_task_with_its_pid(extension) -> None:
    """The arms' expert paths meet a stateful PID, a task's params and its own
    episode length here, as no stand-in env shows."""
    state, _ = _agent((extension,)).train(seed=0, n_timesteps=16)
    leaves = jax.tree.leaves(state.actor_state.params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)


def test_the_gate_reads_the_pids_return_on_a_real_task() -> None:
    """A gate left at its first stage evaluates the expert alone: its best is
    the PID's own (finite) return, not a NaN or an untouched infinity."""
    never = JSRLCurriculum(
        threshold=float("inf"), eval_every=8, eval_episodes=2, moving_average=2
    )
    state, _ = _agent((never,)).train(seed=0, n_timesteps=16)
    jsrl_state = jax.tree.map(lambda x: x[0], state.ext_state[0])
    assert int(jsrl_state.stage) == 0
    assert int(jsrl_state.evaluations) >= 1
    assert np.isfinite(float(jsrl_state.best))


def test_offline_updates_train_on_rows_recorded_from_the_pid() -> None:
    """A recorded-data arm holds no live expert: its replay has no expert
    columns for a dataset to lack."""
    plain = _agent((), expert=False, learning_starts=0)
    rows = ds.record(_pid(), plain.env_args, jax.random.PRNGKey(0), 32)
    pretrained = _agent((PriorData(rows, 4),), expert=False, learning_starts=0)
    state = pretrained.train(seed=0, n_timesteps=0)[0]
    untouched = plain.train(seed=0, n_timesteps=0)[0]
    moved = jax.tree.map(
        lambda a, b: not np.array_equal(a, b),
        state.actor_state.params,
        untouched.actor_state.params,
    )
    assert all(jax.tree.leaves(moved))
