"""Jump-start RL (Uchendu et al. 2023, arXiv 2204.02372).

Contract: the guide (the agent's expert) acts each episode's first h steps
and the learner the others. JSRL-Curriculum starts h at the task's horizon H
and lowers it a stage at a time when an evaluation of the combined policy
passes; JSRL-Random draws h from 0 to H each episode; the time schedule is
Ajax's own. Realistic bugs named per test: a horizon that is not the task's,
a stage moving without a passing evaluation, an evaluation of the learner
alone (not the combined policy), an evaluation stealing training draws.
Sections and tables below are the paper's.
"""

import itertools
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.extensions.expert import (
    JSRLCurriculum,
    JSRLState,
    combined_policy_return,
    next_stage,
)
from tests.environments import paid_action
from tests.environments.identity_env import LIMIT, IdentityEnv

SMALL: dict[str, Any] = {
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
    "batch_size": 4,
    "buffer_size": 256,
    "learning_starts": 8,
    "policy_update_start": 0,
    "alpha_update_start": 0,
}
TICKS = 64


def _expert(obs: jax.Array) -> jax.Array:
    return jnp.ones(obs.shape[:-1] + (1,))


def _run(jsrl: JSRLCurriculum, env: Any = None, **kwargs: Any) -> tuple:
    agent = SAC(
        env or IdentityEnv(),
        n_envs=2,
        expert_policy=_expert,
        extensions=(jsrl,),
        **{**SMALL, **kwargs},
    )
    state, _ = agent.train(seed=0, n_timesteps=TICKS)
    return agent, jax.tree.map(lambda x: x[0], state)


def _bound(agent: Any) -> Any:
    """The agent's JSRLCurriculum, bound to it."""
    return agent.make_context(1)[0].extensions.extensions[0]


def _state(best: float = -jnp.inf, stage: int = 0, window: int = 1) -> JSRLState:
    return JSRLState(
        key=jax.random.PRNGKey(0),
        stage=jnp.int32(stage),
        best=jnp.float32(best),
        recent=jnp.zeros(window),
        evaluations=jnp.int32(0),
    )


# -- when a stage moves --------------------------------------------------------
@pytest.mark.parametrize(
    ("best", "value", "moves"),
    [
        (-jnp.inf, -500.0, True),  # the guide's own first evaluation passes
        (-100.0, -104.0, True),  # within 5% of a negative best
        (-100.0, -106.0, False),  # past 5% of a negative best
        (100.0, 96.0, True),  # within 5% of a positive best
        (100.0, 94.0, False),
        (-100.0, -50.0, True),  # better than the best
    ],
)
def test_a_stage_moves_within_tolerance_of_the_best(best, value, moves) -> None:
    after = next_stage(_state(best), jnp.float32(value), 10, 0.05)
    assert int(after.stage) == int(moves)
    assert float(after.best) == max(best, value)


def test_the_paper_compares_moving_averages_not_single_evaluations() -> None:
    """A.4, a horizon of 3 and a tolerance of 0: evaluations 10, 10, 4, 16
    average 10, 10, 8, 10, so the third stops and the best stays 10. A rule on
    the single evaluation would end at a best of 16."""
    state = _state(window=3)
    moved = []
    for value in (10.0, 10.0, 4.0, 16.0):
        before = int(state.stage)
        state = next_stage(state, jnp.float32(value), 10, 0.0)
        moved.append(int(state.stage) > before)
    assert moved == [True, True, False, True]
    assert float(state.best) == 10.0


def test_an_absolute_threshold_replaces_the_tolerance() -> None:
    state = _state(best=100.0)
    below = next_stage(state, jnp.float32(99.0), 10, 0.05, threshold=120.0)
    above = next_stage(state, jnp.float32(99.0), 10, 0.05, threshold=90.0)
    assert (int(below.stage), int(above.stage)) == (0, 1)


def test_the_last_stage_stays() -> None:
    after = next_stage(_state(stage=10), jnp.float32(0.0), 10, 0.05)
    assert int(after.stage) == 10


# -- the horizon -------------------------------------------------------------
def test_the_horizon_is_read_from_the_task() -> None:
    """A default of 1000 on a task of 5 steps would hand the guide every step
    of every episode for most of the curriculum."""
    agent, _ = _run(JSRLCurriculum(schedule="time"))
    assert _bound(agent).episode_length == LIMIT


def test_stage_zero_hands_the_guide_whole_episodes() -> None:
    """No evaluation passes threshold inf, so the first stage holds and the
    expert acts every step after the warm-up."""
    never = JSRLCurriculum(threshold=float("inf"), eval_every=8, eval_episodes=2)
    _, state = _run(never)
    rows = state.collector_state.buffer_state.experience
    past_warm_up = np.asarray(rows["is_expert"])[:, 4 : TICKS // 2]
    assert past_warm_up.min() == 1.0
    assert int(state.ext_state[0].stage) == 0


def test_passing_evaluations_walk_the_curriculum_down() -> None:
    """Threshold -inf passes each evaluation, one per 8 steps from step 8 on:
    stage min(n, k), h = H (1 - k / n)."""
    always = JSRLCurriculum(
        threshold=-float("inf"), eval_every=8, eval_episodes=2, n_stages=5
    )
    _, state = _run(always)
    assert int(state.ext_state[0].stage) == 5
    stages = JSRLCurriculum(episode_length=LIMIT, n_stages=5)
    np.testing.assert_allclose(
        [stages.stage_horizon(jnp.int32(k)) for k in (0, 1, 5)], [5.0, 4.0, 0.0]
    )


# -- the evaluation -----------------------------------------------------------
def test_the_evaluation_rolls_the_guide_in_then_the_learner() -> None:
    """On a task paying the executed action, 4 steps long, an expert acting 1
    and a learner acting a: guide_steps h return h + (4 - h) a. A bug
    evaluating the learner alone gives 4 a at every h."""
    env, params = paid_action.make(length=4)
    agent = SAC(
        env,
        env_params=params,
        n_envs=1,
        expert_policy=_expert,
        extensions=(JSRLCurriculum(schedule="random"),),
        **SMALL,
    )
    out = agent.train(seed=0, n_timesteps=8)
    state = jax.tree.map(lambda x: x[0], out[0])
    a = float(agent.predict(out, paid_action.OBS[None])[0, 0])
    jsrl = _bound(agent)
    for h in (0, 2, 4):
        value = combined_policy_return(
            agent.env_args,
            jsrl.expert_policy,
            state,
            jnp.int32(h),
            3,
            jax.random.PRNGKey(1),
        )
        np.testing.assert_allclose(value, h + (4 - h) * a, rtol=1e-5)


class Counting:
    """A stateful expert acting its step count in the episode: 1, 2, 3, ..."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return state + 1.0, state + 1.0


def test_the_evaluation_tracks_a_stateful_guide_along_the_episode() -> None:
    """A PID's memory runs from the episode's start: the guide's first two
    steps act 1 then 2, on a task paying the action, bounded at 2. A guide
    called from a fresh state each step would act 1 twice."""
    env, params = paid_action.make(length=2)
    agent = SAC(
        env,
        env_params=params,
        n_envs=1,
        expert_policy=Counting(),
        extensions=(JSRLCurriculum(schedule="random"),),
        **SMALL,
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=8)[0])
    jsrl = _bound(agent)
    value = combined_policy_return(
        agent.env_args,
        jsrl.expert_policy,
        state,
        jnp.int32(2),
        3,
        jax.random.PRNGKey(1),
    )
    np.testing.assert_allclose(value, 3.0, rtol=1e-5)


def test_the_evaluation_moves_no_training_draw() -> None:
    """Evaluating often or rarely leaves training as it was, with no stage
    moving: the rollout has its envs and keys, and the expert's return does
    not reach the replay."""
    never: dict[str, Any] = {"threshold": float("inf"), "eval_episodes": 2}
    _, often = _run(JSRLCurriculum(eval_every=2, **never))
    _, rarely = _run(JSRLCurriculum(eval_every=10_000, **never))
    jax.tree.map(
        np.testing.assert_array_equal,
        often.actor_state.params,
        rarely.actor_state.params,
    )


# -- JSRL-Random --------------------------------------------------------------
def _guided_steps(rows: dict, env: int, ticks: int) -> list[int]:
    """Per finished episode of one env, the expert steps it opened with;
    raises if the expert acted after the learner did."""
    t = np.asarray(rows["obs"])[env, :ticks, 0]
    flags = np.asarray(rows["is_expert"])[env, :ticks, 0]
    starts = np.flatnonzero(t == 0)
    guided = []
    for start, end in itertools.pairwise(starts):
        episode = flags[start:end]
        assert (np.diff(episode) <= 0).all(), episode  # guide first, then learner
        guided.append(int(episode.sum()))
    return guided


def test_jsrl_random_hands_over_once_each_episode() -> None:
    """The guide opens each episode and the learner takes over once; h varies
    between episodes. A bug redrawing h each step would interleave them."""
    agent = SAC(
        IdentityEnv(),
        n_envs=2,
        expert_policy=_expert,
        extensions=(JSRLCurriculum(schedule="random"),),
        **{**SMALL, "learning_starts": 0, "buffer_size": 512},
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=240)[0])
    rows = state.collector_state.buffer_state.experience
    guided = _guided_steps(rows, 0, 120) + _guided_steps(rows, 1, 120)
    assert len(guided) >= 40
    assert len(set(guided)) >= 3


def _horizons(steps: list[int], tick: int, key: jax.Array) -> np.ndarray:
    collector = SimpleNamespace(
        step_in_episode=jnp.array(steps), timestep=jnp.int32(tick * len(steps))
    )
    random = JSRLCurriculum(episode_length=100, n_stages=4, schedule="random")
    return np.asarray(random.episode_horizons(collector, key))


def test_jsrl_randoms_h_holds_through_an_episode() -> None:
    """Env 0 starts at tick 10: its h at tick 13, three steps in, is the
    same. Env 1, whose episode started later, may draw another."""
    key = jax.random.PRNGKey(0)
    first = _horizons([0, 0], 10, key)
    later = _horizons([3, 1], 13, key)
    assert first[0] == later[0]


def test_jsrl_randoms_h_is_uniform_on_the_curriculums_sequence() -> None:
    """Sec. 4.2: h uniform on {H_1, ..., H_n}, here 100, 75, 50, 25, 0 (A.4).
    2400 envs each draw once: 480 of each expected, a binomial standard
    deviation of sqrt(2400 * 0.2 * 0.8) = 20, so 5 of them (100) hold for a
    right draw on any seed. A draw on every integer leaves no step of 25, an
    exclusive bound empties an end."""
    drawn = _horizons([0] * 2400, 0, jax.random.PRNGKey(1))
    values, counts = np.unique(drawn, return_counts=True)
    np.testing.assert_array_equal(values, [0.0, 25.0, 50.0, 75.0, 100.0])
    np.testing.assert_allclose(counts, 480, atol=100)


# -- what the configuration refuses ---------------------------------------------
def test_decay_frac_belongs_to_the_time_schedule() -> None:
    """A caller of the old form must not silently get another schedule."""
    with pytest.raises(ValueError, match="decay_frac"):
        JSRLCurriculum(decay_frac=0.5)


def test_an_unknown_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="schedule"):
        JSRLCurriculum(schedule="linear")
