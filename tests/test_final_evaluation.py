"""The study's final evaluation (ajax.log.final_evaluation).

Contract (the expert-RL benchmark plan, Phase 3's evaluation, and the brief
of this work; nothing here is read off the code):

- final_evaluation(agent, state, task, seed_numbers, expert) returns what
  TargetGym's seed_scores reads, (seeds, episodes): each episode's mean of
  -reward over all its steps, and its trip count. A diverged policy costs
  inf; a diverged expert stays NaN, which the score refuses.
- Seed number i has its own fixed episode set, from the task and i alone:
  every arm and the expert meet the same initial states and the same
  disturbance draws; another seed number or task meets others. Neither the
  training progress, the method nor the state's own keys reach it, and an arm
  sampling its action moves none of the environment's draws.
- The episode sets stay outside the expert-tuning seeds 0-9, 100-109, 200-209.
- The expert's costs are a function of the episode set: an arm reuses what a
  cache holds.
- A hybrid arm is scored twice: with the expert, and its actor alone.

Oracles: the arms act differently, so equal draws are read through a task
whose cost reads only its draws, whatever the actions (reading_the_draws):
two rollouts cost the same exactly when they met the same draws. Expected
values otherwise come by hand (a cost of 1 a step over a known length) or
from a second agent known to act as the first (actor-only against a plain
agent on the same actor). Rollouts compiled apart may round apart (XLA fuses
differently, RTOL), so those compare within RTOL; a different draw differs by
orders of magnitude more.
"""

from functools import partial
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import SAC
from ajax.agents.loop import fresh_state
from ajax.controller import Controller
from ajax.environments.create import with_episode_keys
from ajax.evaluate import evaluate, expert_episodes, policy_episodes
from ajax.extensions.expert import ExpertGains, ResidualPolicy
from ajax.extensions.target_mods import IBRL
from ajax.log import (
    FinalEvaluation,
    actor_only,
    final_evaluation,
    final_evaluation_key,
    final_expert_episodes,
)
from tests.environments.identity_env import IdentityEnv

#: float32 sums of the same draws, compiled apart: see the module docstring.
RTOL = 1e-5
EPISODES = 3
NETS: dict[str, Any] = {
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
    "learning_starts": 8,
    "batch_size": 4,
}


class IntegratingExpert:
    """A stateful expert in env units, as a PID is: its action is what it
    integrated of the first observation column, clipped into [-1, 1]."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        state = state + 0.1 * obs[..., :1]
        return jnp.clip(state, -1.0, 1.0), state


def must_not_run(obs: jax.Array) -> jax.Array:
    raise AssertionError("the expert ran although the cache held its costs")


def reading_the_draws(env_class: type, initial_reading: Any) -> type:
    """env_class whose cost reads its draws alone, whatever the actions: the
    initial state on the first step, and each step's normal draw, on the key
    and in the form a TargetGym plant draws its turbulence."""

    class DrawReadout(env_class):  # type: ignore[valid-type, misc]
        def step_env(self, key, state, action, params):
            obs, new_state, _, terminated, info = super().step_env(
                key, state, action, params
            )
            draw = jax.random.normal(jax.random.fold_in(key, state.time))
            start = jnp.where(state.time == 0, initial_reading(state), 0.0)
            return obs, new_state, -(start + draw), terminated, info

    return DrawReadout


def pendulum_task() -> tuple[Any, Any]:
    env = reading_the_draws(Pendulum, lambda state: state.theta + state.theta_dot)()
    return env, env.default_params.replace(max_steps_in_episode=16)


def pendulum_on_step_keys() -> tuple[Any, Any]:
    """The same task without episode keys, whose plant draws on the rollout's
    step keys, as a wrapper over it does on every stack: only there can an
    action's draw shift a step key's."""
    env, params = pendulum_task()
    return with_episode_keys(env, on=False), params


def turbine_task() -> tuple[Any, Any]:
    """TargetGym's wind turbine: its wind and target drawn at reset, its
    turbulence at every step."""
    target_gym = pytest.importorskip("target_gym.registry")
    spec = target_gym.REGISTRY["wind_turbine"]
    reading = lambda state: state.v_wind + state.target_power  # noqa: E731
    env = reading_the_draws(type(spec.make_env()), reading)()
    return env, spec.make_test_params().replace(max_steps_in_episode=16)


def sac(env: Any, params: Any = None, **kwargs: Any) -> SAC:
    return SAC(env, env_params=params, n_envs=1, **NETS, **kwargs)


def fresh_states(agent: SAC, seeds: list) -> Any:
    """Each seed's state as initialised: the evaluation reads no more of a
    state than a trained one has, and no training compiles."""
    loop, context = agent.make_context(1)
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])

    def one_seed(key: jax.Array) -> Any:
        init = partial(agent.init, context=context)
        return fresh_state(init, loop.extensions, 1, key, env_args=agent.env_args)

    return jax.vmap(one_seed)(keys)


def costs_of(
    agent: SAC,
    state: Any,
    expert: Any = None,
    task: str = "task",
    seed_numbers: tuple = (3, 7),
    expert_cache: Any = None,
) -> FinalEvaluation:
    expert = expert or IntegratingExpert()
    return final_evaluation(
        agent, state, task, seed_numbers, expert, expert_cache, EPISODES
    )


@pytest.mark.parametrize("task", [pendulum_task, pendulum_on_step_keys, turbine_task])
def test_every_arm_and_the_expert_meet_the_same_draws(task):
    """A rollout drawing its initial states or its disturbances from the
    method, its seed or anything but the episode key would compare a policy
    and the expert on different episodes: the score's pairing would be void."""
    env, params = task()
    plain, hybrid = (
        sac(env, params),
        sac(
            env,
            params,
            expert_policy=IntegratingExpert(),
            extensions=(ResidualPolicy(scale=1.0),),
        ),
    )

    first = costs_of(plain, fresh_states(plain, [0, 1]))
    second = costs_of(hybrid, fresh_states(hybrid, [5, 6]))

    assert first.policy.cost_per_step.shape == (2, EPISODES)
    np.testing.assert_allclose(
        first.policy.cost_per_step, second.policy.cost_per_step, rtol=RTOL
    )
    np.testing.assert_allclose(
        first.policy.cost_per_step, first.expert.cost_per_step, rtol=RTOL
    )
    np.testing.assert_allclose(
        second.policy.cost_per_step, second.actor_only.cost_per_step, rtol=RTOL
    )


def test_each_seed_number_and_each_task_has_its_own_episodes():
    """One shared episode set would hide how much the expert's cost varies
    between episodes from the seed-level statistics."""
    env, params = pendulum_task()
    agent = sac(env, params)
    state = fresh_states(agent, [0, 1])

    result = costs_of(agent, state).policy.cost_per_step
    other_task = costs_of(agent, state, task="other task").policy.cost_per_step

    assert not np.isclose(result[0], result[1], rtol=1e-3).any()
    assert not np.isclose(result, other_task, rtol=1e-3).any()


def test_training_progress_and_the_states_keys_change_nothing():
    """The in-training evaluation folds the timestep into a key off eval_rng:
    done here, an arm's episodes would depend on when it stopped and on its
    training seed."""
    env, params = gymnax.make("Pendulum-v1")
    agent = sac(env, params)
    state = fresh_states(agent, [0, 1])
    collector = state.collector_state
    later = state.replace(
        collector_state=collector.replace(timestep=collector.timestep + 900_000),
        n_logs=state.n_logs + 90,
        eval_rng=state.eval_rng + 1,
        rng=state.rng + 1,
    )

    np.testing.assert_array_equal(
        costs_of(agent, state).policy.cost_per_step,
        costs_of(agent, later).policy.cost_per_step,
    )


@pytest.mark.parametrize("task", [pendulum_task, pendulum_on_step_keys])
def test_an_arm_sampling_its_actions_moves_no_draw(task):
    """IBRL's soft choice samples in evaluation. A sample split from the
    rollout's key would shift every later disturbance of that arm alone."""
    env, params = task()
    plain, expert = sac(env, params), IntegratingExpert()
    sampling = sac(
        env,
        params,
        expert_policy=expert,
        extensions=(IBRL(expert_policy=expert, soft_choice=True),),
    )

    deterministic = costs_of(plain, fresh_states(plain, [0, 1]))
    stochastic = costs_of(sampling, fresh_states(sampling, [0, 1]))

    np.testing.assert_allclose(
        deterministic.policy.cost_per_step, stochastic.policy.cost_per_step, rtol=RTOL
    )


class Hand(IdentityEnv):
    """Hand-derivable episodes: 3 steps where c > 0, else 5, a reward of
    reward(c) a step, and a trip on every even step: 1 in a short episode,
    2 in a long one."""

    def __init__(self, reward: Any):
        super().__init__()
        self.reward = reward

    def step_env(self, key, state, action, params):
        obs, new_state, _, terminated, _ = super().step_env(key, state, action, params)
        info = {"tripped": new_state.time % 2 == 0}
        return obs, new_state, self.reward(state.c), terminated, info


def constant_expert(obs: jax.Array) -> jax.Array:
    return jnp.zeros((*obs.shape[:-1], 1))


def hand_episodes(reward: Any) -> FinalEvaluation:
    env = Hand(reward)
    agent = sac(env, env.default_params)
    state = fresh_states(agent, [0, 1])
    return final_evaluation(agent, state, "hand", [3, 7], constant_expert, None, 16)


def pay_two(c: jax.Array) -> jax.Array:
    return jnp.full_like(c, -2.0)


def test_a_cost_is_the_mean_of_minus_the_reward_over_the_episodes_own_steps():
    """seed_scores compares per-step costs with a per-step ceiling: a summed
    cost sits an episode length off, a horizon-long mean under-reads an
    episode that ended early. Hand-derived: -2 a step costs 2 at any length."""
    result = hand_episodes(pay_two)

    assert result.policy.cost_per_step.shape == (2, 16)
    np.testing.assert_array_equal(result.policy.cost_per_step, np.full((2, 16), 2.0))
    np.testing.assert_array_equal(result.expert.cost_per_step, np.full((2, 16), 2.0))


def test_trips_are_counted_per_episode_to_its_end():
    """The score's clip hides how many trips an episode took. Hand-derived:
    a trip every even step gives 1 in 3 steps and 2 in 5; a count running on
    past the episode's end would read 2 in both."""
    result = hand_episodes(pay_two)

    assert set(np.unique(result.policy.trips)) == {1.0, 2.0}
    np.testing.assert_array_equal(result.policy.trips, result.expert.trips)


def test_an_env_reporting_no_trips_has_an_unknown_count():
    """A zero would read as a plant that never trips, and a renamed signal
    would pass as a perfect trip record."""
    env, params = gymnax.make("Pendulum-v1")
    agent = sac(env, params)

    result = costs_of(agent, fresh_states(agent, [0, 1]))

    assert np.isnan(result.policy.trips).all()
    assert np.isnan(result.expert.trips).all()


def test_a_diverged_policy_costs_infinity_and_a_diverged_expert_stays_nan():
    """seed_scores raises on a NaN cost and takes inf as a diverged policy,
    which then scores the floor; a NaN expert is a broken baseline it must
    refuse, not a floor to beat."""
    result = hand_episodes(lambda c: jnp.where(c > 0, jnp.nan, -2.0))

    diverged = result.expert.cost_per_step != result.expert.cost_per_step
    assert diverged.any() and not diverged.all()
    assert (result.policy.cost_per_step[diverged] == np.inf).all()
    assert (result.policy.cost_per_step[~diverged] == 2.0).all()


def test_the_expert_runs_once_for_an_episode_set():
    """Rerunning the expert for each of 12 arms repeats a fixed result, and a
    cache that served another set's costs would pair the wrong episodes."""
    env, params = pendulum_task()
    plain = sac(env, params)
    state = fresh_states(plain, [0, 1])
    cache: dict = {}

    first = costs_of(plain, state, expert_cache=cache)
    again = costs_of(plain, state, must_not_run, expert_cache=cache)
    swapped = costs_of(
        plain, state, must_not_run, expert_cache=cache, seed_numbers=(7, 3)
    )

    np.testing.assert_array_equal(
        first.expert.cost_per_step, again.expert.cost_per_step
    )
    np.testing.assert_array_equal(
        first.expert.cost_per_step[::-1], swapped.expert.cost_per_step
    )
    with pytest.raises(AssertionError, match="the expert ran"):
        costs_of(plain, state, must_not_run, expert_cache=cache, seed_numbers=(3, 9))


def test_the_expert_table_can_be_filled_before_any_arm():
    """A sweep runs the expert once, up front, in one process; the arms of
    every process then find its costs."""
    env, params = pendulum_task()
    agent = sac(env, params)
    cache: dict = {}

    filled = final_expert_episodes(
        env, params, IntegratingExpert(), "task", (3, 7), cache, EPISODES
    )
    found = costs_of(
        agent, fresh_states(agent, [0, 1]), must_not_run, expert_cache=cache
    )

    assert filled.cost_per_step.shape == (2, EPISODES)
    np.testing.assert_array_equal(found.expert.cost_per_step, filled.cost_per_step)


EXPERT = IntegratingExpert()


@pytest.mark.parametrize(
    "extension",
    [ResidualPolicy(scale=1.0), IBRL(expert_policy=EXPERT)],
    ids=["residual", "ibrl"],
)
def test_a_hybrid_arm_is_scored_with_the_expert_and_actor_only(extension):
    """A hybrid's actor can look good only because the expert stays in the
    loop. Oracle: the actor alone costs what a plain agent holding the same
    actor costs on the same episodes, unlike the hybrid."""
    env, params = gymnax.make("Pendulum-v1")
    hybrid = sac(env, params, expert_policy=EXPERT, extensions=(extension,))
    state = fresh_states(hybrid, [0, 1])

    result = costs_of(hybrid, state)
    alone = costs_of(sac(env, params), state)

    assert result.actor_only.cost_per_step.shape == (2, EPISODES)
    np.testing.assert_allclose(
        result.actor_only.cost_per_step, alone.policy.cost_per_step, rtol=RTOL
    )
    assert alone.actor_only is None
    assert not np.isclose(
        result.policy.cost_per_step, result.actor_only.cost_per_step, rtol=1e-3
    ).all()


def test_gain_mode_has_no_actor_only_table():
    """Under gain mode the actor's actions are gains: stepped into the plant
    alone, they would score nonsense, or fail on their width."""
    assert actor_only(Controller(transforms=(ExpertGains(),))) is None


def test_final_evaluation_acts_as_the_in_training_one():
    """The final path reuses evaluate's rollout: their means agree, for the
    policy and the expert, on one key (Pendulum's 200 steps per episode)."""
    env, params = gymnax.make("Pendulum-v1")
    agent = sac(env, params)
    state = jax.tree.map(lambda leaf: leaf[0], fresh_states(agent, [0]))
    key = jax.random.PRNGKey(11)
    expert = IntegratingExpert()

    policy = policy_episodes(env, state.actor_state, EPISODES, key, params)
    reference = expert_episodes(env, expert, EPISODES, key, params)
    outputs = evaluate(
        env, state.actor_state, EPISODES, key, params, eval_expert_policy=expert
    )

    np.testing.assert_allclose(-200 * policy.cost_per_step.mean(), outputs[0], RTOL)
    np.testing.assert_allclose(-200 * reference.cost_per_step.mean(), outputs[5], RTOL)


def key_data(key: jax.Array) -> tuple:
    return tuple(np.asarray(key).tolist())


def test_episode_sets_stay_off_the_seeds_the_experts_were_tuned_on():
    """An expert scored on its tuning episodes looks better than it is."""
    reserved = [*range(10), *range(100, 110), *range(200, 210)]
    tuned_on = {key_data(jax.random.PRNGKey(seed)) for seed in reserved}

    keys = {key_data(final_evaluation_key("hvac", i)) for i in range(40)}

    assert len(keys) == 40
    assert keys.isdisjoint(tuned_on)


def test_characterization_a_seed_numbers_episode_key_stays_put():
    """Pins current behaviour, not verified against a spec: the key of one
    set, as first derived. One process runs per GPU, and expert costs are
    cached: a name hashed with Python's salted hash, or a changed derivation,
    would leave arms on unmatched episodes. A deliberate change re-records it,
    and every cached expert cost with it."""
    assert key_data(final_evaluation_key("hvac", 3)) == (837563822, 4008161446)


def test_a_trained_run_evaluates_from_trains_own_return():
    """The harness's call on train's (state, out): 50 episodes per seed, the
    study's set; fewer would quietly widen every seed's score."""
    env, params = gymnax.make("Pendulum-v1")
    agent = sac(env, params)
    trained = agent.train(seed=[0, 1], n_timesteps=16)

    result = final_evaluation(agent, trained, "task", [0, 1], IntegratingExpert())

    assert result.policy.cost_per_step.shape == (2, 50)
    assert result.policy.trips.shape == (2, 50)
    assert result.expert.cost_per_step.shape == (2, 50)
    assert result.actor_only is None
