"""Noise levels through Ajax: a run's sensors, its expert's, and what it trains on.

Contract (the study's brief, on TargetGym's sensor noise):

- A run's params reach everything that reads the plant: the agent, the expert
  in training, and both evaluations. Oracle: a plant told level 2 through its
  params must run as a plant whose default is level 2, so any reader falling
  back on default params shows.
- Two levels read differently on one set of keys, and meet the same plant.
- A single state stays single, whatever vectors it holds (a per-episode key).
- Expert records do not cross params, and each level can have its own gains.
- Under MeasuredRewardWrapper, training is training where the measured reward
  is the reward, and everything reported stays the true reward. An env that
  reports none is refused.

Costs from programs compiled apart may round apart (RTOL, as in
tests/test_final_evaluation.py); a different draw differs by far more.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs import create as create_brax_env
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import SAC, UDRL
from ajax.environments.interaction import get_raw_obs, holds_many_envs
from ajax.evaluate import evaluate, expert_episodes
from ajax.extensions.expert import ResidualPolicy
from ajax.extensions.observe import ExpertActionInput
from ajax.extensions.pretrain import ExpertPrefill
from ajax.log import final_evaluation, final_expert_episodes
from ajax.logging import LoggingConfig
from ajax.wrappers import MeasuredRewardWrapper

RTOL = 1e-5
EPISODES = 4


@struct.dataclass
class PlantState(environment.EnvState):
    plant: jax.Array
    sensor_key: jax.Array


@struct.dataclass
class PlantParams(environment.EnvParams):
    noise: float = 1.0
    max_steps_in_episode: int = 8


LEVEL_ONE, LEVEL_TWO = PlantParams(noise=1.0), PlantParams(noise=2.0)


class Plant(environment.Environment):
    """A scalar plant read through a sensor, as TargetGym's noise works: the
    reading is a pure function of the state and the params, on a key the
    state holds, and it never reaches the dynamics. The true reward reads the
    plant, the measured one the sensor."""

    def __init__(self, default_noise: float = 1.0, scored_on_readings: bool = False):
        self.default_noise = default_noise
        self.scored_on_readings = scored_on_readings

    @property
    def default_params(self) -> PlantParams:
        return PlantParams(noise=self.default_noise)

    def reset_env(self, key, params):
        sensor_key = jax.random.key_data(jax.random.fold_in(key, 1))
        state = PlantState(time=0, plant=jax.random.normal(key), sensor_key=sensor_key)
        return self.get_obs(state, params), state

    def get_obs(self, state, params=None, key=None):
        params = self.default_params if params is None else params
        draw = jax.random.normal(jax.random.fold_in(state.sensor_key, state.time))
        return (state.plant + params.noise * draw)[None].astype(jnp.float32)

    def step_env(self, key, state, action, params):
        drift = 0.1 * jax.random.normal(key)
        plant = 0.9 * state.plant + 0.1 * jnp.reshape(action, ()) + drift
        new = PlantState(state.time + 1, plant, state.sensor_key)
        obs = self.get_obs(new, params)
        measured = -(obs[0] ** 2)
        reward = measured if self.scored_on_readings else -(plant**2)
        return obs, new, reward, jnp.array(False), {"measured_reward": measured}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(-1e3, 1e3, (1,), jnp.float32)

    def action_space(self, params=None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), jnp.float32)


class ReadingExpert:
    """A stateful expert in env units, as a PID is, acting on what it reads."""

    def __init__(self, gain: float = -0.5):
        self.gain = gain

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.clip(self.gain * obs[..., :1], -1.0, 1.0), state + obs[..., :1]


def zero_action(obs: jax.Array) -> jax.Array:
    return jnp.zeros((*obs.shape[:-1], 1))


def must_not_run(obs: jax.Array) -> jax.Array:
    raise AssertionError("the expert ran although the cache held its records")


def sac(env: Any, params: Any, **kwargs: Any) -> SAC:
    return SAC(
        env,
        env_params=params,
        n_envs=1,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=8,
        batch_size=4,
        **kwargs,
    )


def test_two_levels_read_differently_and_meet_the_same_plant():
    """A level reaching the dynamics would move the true cost of an expert
    that ignores its sensors; a level reaching nothing would leave a reading
    expert's cost alone."""
    env, key = Plant(), jax.random.PRNGKey(0)

    def costs(expert: Any, params: PlantParams) -> np.ndarray:
        return np.asarray(expert_episodes(env, expert, 8, key, params).cost_per_step)

    deaf = costs(zero_action, LEVEL_ONE), costs(zero_action, LEVEL_TWO)
    reading = costs(ReadingExpert(), LEVEL_ONE), costs(ReadingExpert(), LEVEL_TWO)

    np.testing.assert_allclose(*deaf, rtol=RTOL)
    assert not np.isclose(*reading, rtol=1e-3).any()


def one_run(env: Plant) -> tuple:
    """A short run at level 2 whose expert acts, feeds the actor's input and
    fills the replay: what training, the in-training evaluation and the final
    evaluation each read of the plant."""
    expert = ReadingExpert()
    agent = sac(
        env,
        LEVEL_TWO,
        expert_policy=expert,
        extensions=(
            ExpertPrefill(n_steps=4),
            ExpertActionInput(),
            ResidualPolicy(scale=1.0),
        ),
    )
    logging = LoggingConfig(log_frequency=8, backends=())
    state, evaluations = agent.train(
        seed=0, n_timesteps=16, num_episode_test=2, logging_config=logging
    )
    final = final_evaluation(agent, state, "task", [3], expert, None, EPISODES)
    assert final.actor_only is not None
    return (
        jax.tree.leaves(state.actor_state.params),
        np.asarray(evaluations["Eval/episodic mean reward"]),
        np.asarray(evaluations["Eval/episodic mean expert reward"]),
        final.policy.cost_per_step,
        final.expert.cost_per_step,
        final.actor_only.cost_per_step,
    )


def test_every_reader_takes_the_runs_level():
    """evaluate.py read the expert's observation under default params: at any
    level but the default, the expert read other sensors than the agent, in
    evaluation alone. The two plants differ in their default and nothing else
    the run is told, so a stray default shows here."""
    told, default = one_run(Plant(1.0)), one_run(Plant(2.0))

    for ran, expected in zip(told[1:], default[1:], strict=True):
        np.testing.assert_allclose(ran, expected, rtol=RTOL)
    for ran, expected in zip(told[0], default[0], strict=True):
        np.testing.assert_allclose(ran, expected, rtol=RTOL)


def test_a_reading_expert_is_evaluated_at_the_runs_level():
    """The same in-training path alone, cheaply: its expert's return at level
    2 is not its return at level 1."""
    env = Plant()
    agent = sac(env, LEVEL_TWO)
    context = agent.make_context(1)[1]
    actor = agent.init(jax.random.PRNGKey(0), context).actor_state
    expert = ReadingExpert()

    def expert_return(params: PlantParams) -> float:
        outputs = evaluate(
            env,
            actor,
            EPISODES,
            jax.random.PRNGKey(1),
            params,
            eval_expert_policy=expert,
        )
        return float(outputs[5])

    assert expert_return(LEVEL_ONE) != pytest.approx(expert_return(LEVEL_TWO), rel=1e-3)


def udrl_run(env: Plant) -> list:
    agent = UDRL(
        env,
        env_params=LEVEL_TWO,
        n_envs=1,
        expert_policy=ReadingExpert(),
        extensions=(ResidualPolicy(scale=1.0),),
        actor_architecture=("8", "relu"),
        n_steps=8,
        batch_size=8,
        n_updates_per_iter=4,
        buffer_capacity=8,
    )
    state, _ = agent.train(seed=0, n_timesteps=16)
    return jax.tree.leaves(state.actor_state.params)


def test_udrl_collects_with_its_expert_at_the_runs_level():
    """UDRL builds its expert's reading itself, and read it under default
    params: at any other level its expert saw other sensors than the agent.
    The residual makes the expert act: alone, UDRL calls it and uses nothing."""
    told, default = udrl_run(Plant(1.0)), udrl_run(Plant(2.0))

    for ran, expected in zip(told, default, strict=True):
        np.testing.assert_allclose(ran, expected, rtol=RTOL)


def test_a_single_state_holding_a_vector_stays_single():
    """The largest leaf rank read TargetGym's two-word sensor_key as a batch
    axis, so a single state was vmapped and failed on its scalar time."""
    env, key = Plant(), jax.random.PRNGKey(0)
    _, single = env.reset(key, LEVEL_TWO)
    reset = jax.vmap(env.reset, in_axes=(0, None))
    _, batch = reset(jax.random.split(key, 3), LEVEL_TWO)

    assert get_raw_obs(single, env, "gymnax", LEVEL_TWO).shape == (1,)
    assert get_raw_obs(batch, env, "gymnax", LEVEL_TWO).shape == (3, 1)


class Nested(NamedTuple):
    env_state: Any


@pytest.mark.parametrize("shape, many", [((), False), ((3,), True)])
def test_a_state_without_a_scalar_field_falls_back_on_its_ranks(shape, many):
    """A wrapper's state may nest the env's, with no time of its own."""
    assert holds_many_envs(Nested(jnp.zeros(shape)), "gymnax") == many


def test_expert_records_do_not_cross_levels():
    """A cache keyed by task and seed number alone would hand a run at level
    2 the expert's costs at level 1, without a word."""
    env, cache = Plant(), {}

    def records(expert: Any, params: PlantParams) -> Any:
        return final_expert_episodes(
            env, params, expert, "task", (3, 7), cache, EPISODES
        )

    one = records(ReadingExpert(), LEVEL_ONE)
    two = records(ReadingExpert(), LEVEL_TWO)
    again = records(must_not_run, LEVEL_ONE)

    assert not np.isclose(one.cost_per_step, two.cost_per_step, rtol=1e-3).any()
    np.testing.assert_array_equal(again.cost_per_step, one.cost_per_step)


def test_each_level_can_have_its_own_gains():
    """Gains are stored per level. Two experts of one class must run as
    themselves, not as a compiled program kept from the first."""
    env = Plant()

    def costs(gain: float) -> np.ndarray:
        records = final_expert_episodes(
            env, LEVEL_TWO, ReadingExpert(gain), "task", (3,), None, EPISODES
        )
        return records.cost_per_step

    assert not np.isclose(costs(-0.2), costs(-0.9), rtol=1e-3).any()


def measured_run(env: Any, **kwargs: Any) -> tuple:
    expert = ReadingExpert()
    agent = sac(
        env,
        LEVEL_TWO,
        expert_policy=expert,
        extensions=(ExpertPrefill(n_steps=4),),
        normalize_rewards=True,
        **kwargs,
    )
    return agent, agent.train(seed=0, n_timesteps=16)[0]


def test_training_on_the_measured_reward_is_training_where_it_is_the_reward():
    """The oracle is a plant scored on its readings: the wrapper must give
    the same replay rows, normaliser and networks. A swap made above the
    normaliser, or in collection alone, would leave the prefill or the
    statistics on the true reward."""
    wrapped = measured_run(MeasuredRewardWrapper(Plant()))[1]
    scored = measured_run(Plant(scored_on_readings=True))[1]

    stored = jax.tree.leaves(wrapped.collector_state.buffer_state)
    expected = jax.tree.leaves(scored.collector_state.buffer_state)
    for ran, reference in zip(stored, expected, strict=True):
        np.testing.assert_allclose(ran, reference, rtol=RTOL)
    networks = zip(
        jax.tree.leaves(wrapped.actor_state.params),
        jax.tree.leaves(scored.actor_state.params),
        strict=True,
    )
    for ran, reference in networks:
        np.testing.assert_allclose(ran, reference, rtol=RTOL)


def test_the_replay_holds_the_measured_reward_and_not_the_true_one():
    """Hand-derived from the plant: a row's measured reward is minus its next
    reading squared. Its true reward reads the plant, not the sensor."""
    state = measured_run(MeasuredRewardWrapper(Plant()))[1]
    rows = state.collector_state.buffer_state.experience

    held = np.flatnonzero(np.asarray(rows["reward"]).reshape(-1))
    rewards = np.asarray(rows["reward"]).reshape(-1)[held]
    readings = np.asarray(rows["next_obs"])[..., 0].reshape(-1)[held]

    np.testing.assert_allclose(rewards, -(readings**2), rtol=RTOL)


def test_what_a_measured_run_reports_stays_true():
    """Scores read the true cost. An evaluation keeping the wrapper would
    score a policy on its own sensors' noise."""
    agent, state = measured_run(MeasuredRewardWrapper(Plant()))
    truth = sac(Plant(), LEVEL_TWO, expert_policy=ReadingExpert())
    on_readings = sac(Plant(scored_on_readings=True), LEVEL_TWO)
    expert = ReadingExpert()

    reported = final_evaluation(agent, state, "task", [3], expert, None, EPISODES)
    true = final_evaluation(truth, state, "task", [3], expert, None, EPISODES)
    measured = final_evaluation(on_readings, state, "task", [3], expert, None, EPISODES)

    np.testing.assert_allclose(
        reported.policy.cost_per_step, true.policy.cost_per_step, rtol=RTOL
    )
    assert not np.isclose(
        reported.policy.cost_per_step, measured.policy.cost_per_step, rtol=1e-3
    ).all()


@pytest.mark.parametrize(
    "env",
    [Pendulum(), create_brax_env("fast", batch_size=1)],
    ids=["gymnax", "brax"],
)
def test_an_env_reporting_no_measured_reward_is_refused(env):
    """Silently training on the true reward would be the study's control,
    filed under the arm that reads its sensors."""
    with pytest.raises(ValueError, match="measured_reward"):
        MeasuredRewardWrapper(env)
