"""What a first run shows and returns.

Contract (docs/getting-started/quickstart.md, ``ActorCritic.train``,
``ajax.logging.progress``):
  - a train call given no logging_config evaluates about ten times, prints
    a first line, one line per evaluation and a last line, and returns the
    evaluations, each after a whole number of the agent's iterations; logging_config=None evaluates nothing, prints nothing and
    returns None; a LoggingConfig of the caller's prints one line, when the
    run starts, saying how often it evaluates and where it logs (owner,
    2026-11-27: such a run was silent until it returned);
  - from a run's second evaluation, a line ends with the call's time left,
    at the pace of the last interval between two evaluations; members
    trained one after another count the runs still to come;
  - the last line gives the layout and shape of the returned returns, a
    row per seed (per run in a sweep), with one seed too, and printing the
    evaluations opens on the same layout;
  - the evaluations are the dict of logged names, with ``returns`` (members
    x evaluations) and ``timesteps`` (one per evaluation);
  - a printed evaluation is one line for all members trained side by side
    (their mean return and its range), once each member's has come;
  - ``agent.evaluate(state, episodes, seed)`` is the return of fresh
    episodes of each member's policy, members x episodes, the same for the
    same seed, on a Gymnasium environment too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO
from ajax.agents.sweep import by_configuration
from ajax.evaluate import policy_returns
from ajax.logging import LoggingConfig, TensorBoardBackend
from ajax.logging.progress import (
    Evaluations,
    ProgressPrinter,
    default_log_frequency,
    describe_duration,
    run_logging,
)

RETURN = "Eval/episodic mean reward"


@pytest.mark.parametrize(
    "n_timesteps, frequency",
    # A tenth of the call, rounded up: 1,001 / 10 = 100.1.
    [(40_960, 4_096), (1_000, 100), (1_001, 101), (5, 1), (1, 1)],
)
def test_a_default_run_evaluates_every_tenth_of_its_steps(n_timesteps, frequency):
    assert default_log_frequency(n_timesteps) == frequency


@pytest.mark.parametrize(
    "n_timesteps, iteration_steps, frequency",
    [
        (40_960, 1_024, 4_096),  # a tenth is 4 rollouts already
        (32_768, 1_024, 4_096),  # a tenth, 3,277, is 3.2 rollouts: 4
        (4_096, 2_048, 2_048),  # less than a rollout: one
        (100, 8, 16),  # a tenth, 10, is 1.25 iterations: 2
        (3_000, 1, 300),
    ],
)
def test_the_default_spacing_is_whole_iterations(
    n_timesteps, iteration_steps, frequency
):
    assert default_log_frequency(n_timesteps, iteration_steps) == frequency


def test_evaluations_name_the_returns_and_the_steps_and_stay_a_dict():
    table = Evaluations(
        {
            RETURN: np.array([[1.0, 2.0], [3.0, 4.0]]),
            "timestep": np.array([[10, 20], [10, 20]]),
        }
    )
    np.testing.assert_array_equal(table.returns, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(table.timesteps, [10, 20])
    assert table[RETURN] is table.returns
    doubled = jax.tree.map(lambda leaf: 2 * leaf, table)
    assert isinstance(doubled, Evaluations)
    np.testing.assert_array_equal(doubled.timesteps, [20, 40])


def printer_with_a_clock(labels, together=True):
    lines, now = [], [0.0]
    printer = ProgressPrinter(
        "PPO", labels, 200, together, write=lines.append, clock=lambda: now[0]
    )
    return printer, lines, now


def test_members_trained_together_print_one_line_once_all_have_come():
    printer, lines, now = printer_with_a_clock(["seed 0", "seed 1"])
    printer.start()
    now[0] = 2.5
    printer({"timestep": np.array(100), RETURN: np.array(10.0)}, 1)
    assert len(lines) == 1  # the first line only: seed 0 has not come
    printer({"timestep": np.array(100), RETURN: np.array(30.0)}, 0)
    # Mean of 30 and 10 is 20; 2.5 s after the start.
    assert lines[1] == (
        "        100 / 200 steps | return 20.0"
        " (min 10.0, max 30.0 over 2 seeds) | 2.5 s"
    )
    now[0] = 4.0
    printer.close()
    assert lines[2] == "Done in 4.0 s."


def test_members_trained_one_after_another_print_a_line_each():
    printer, lines, _ = printer_with_a_clock(["seed 7", "seed 9"], together=False)
    printer({"timestep": np.array(200), RETURN: np.array(5.0)}, 1)
    assert lines == ["        200 / 200 steps | return 5.0 | seed 9 | 0.0 s"]


def tiny_ppo() -> PPO:
    return PPO("CartPole-v1", n_envs=2, n_steps=8, n_epochs=1)


def test_a_run_given_no_logging_evaluates_ten_times_and_prints_each(capsys):
    _, evaluations = tiny_ppo().train(seed=[0, 1], n_timesteps=160)
    printed = capsys.readouterr().out.splitlines()
    # 160 steps: an evaluation every 16.
    np.testing.assert_array_equal(evaluations.timesteps, np.arange(16, 161, 16))
    assert evaluations.returns.shape == (2, 10)
    assert len(printed) == 12
    assert printed[0].startswith("PPO, 2 run(s) of 160 steps")
    assert [line.split("/")[0].strip() for line in printed[1:11]] == [
        str(step) for step in range(16, 161, 16)
    ]
    assert all("over 2 seeds" in line for line in printed[1:11])
    assert printed[11].startswith("Done in ")


def test_a_default_run_evaluates_after_whole_rollouts():
    # 16 steps per rollout; a tenth of 96 is 9.6 steps: every rollout.
    _, evaluations = tiny_ppo().train(seed=0, n_timesteps=96)
    np.testing.assert_array_equal(evaluations.timesteps, np.arange(16, 97, 16))


def test_logging_config_none_is_silent_and_evaluates_nothing(capsys):
    _, evaluations = tiny_ppo().train(seed=0, n_timesteps=32, logging_config=None)
    assert evaluations is None
    assert capsys.readouterr().out == ""


def test_a_logging_config_of_the_callers_prints_its_start_line_only(capsys):
    logging = LoggingConfig(log_frequency=16, backends=())
    _, evaluations = tiny_ppo().train(seed=0, n_timesteps=32, logging_config=logging)
    np.testing.assert_array_equal(evaluations.timesteps, [16, 32])
    assert capsys.readouterr().out.splitlines() == [
        "PPO, 1 run(s) of 32 steps: compiling, then an evaluation every 16 steps,"
        " logged to memory only (train returns the evaluations). Nothing more"
        " prints until train returns."
    ]


def test_the_start_line_of_a_logged_run_names_its_tensorboard_folder(capsys):
    logging = LoggingConfig(log_frequency=1_000, backends=(TensorBoardBackend("runs"),))
    run_logging(logging, "SAC", [0, 1], 3_000, 3_000, together=True)
    assert "an evaluation every 1,000 steps, logged to TensorBoard in runs/tensorboard." in (
        capsys.readouterr().out
    )


def test_a_line_after_a_runs_first_evaluation_gives_the_time_left():
    """The pace is the last interval's: 2 s for 100 steps, then 1 s."""
    lines, now = [], [0.0]
    printer = ProgressPrinter(
        "PPO", ["seed 0"], 400, True, write=lines.append, clock=lambda: now[0]
    )
    for step, elapsed in [(100, 5.0), (200, 7.0), (300, 8.0), (400, 9.0)]:
        now[0] = elapsed
        printer({"timestep": np.array(step), RETURN: np.array(1.0)}, 0)
    assert lines == [
        "        100 / 400 steps | return 1.0 | 5.0 s",
        # 0.02 s a step, 200 steps left.
        "        200 / 400 steps | return 1.0 | 7.0 s, about 4 s left",
        # 0.01 s a step, 100 steps left.
        "        300 / 400 steps | return 1.0 | 8.0 s, about 1 s left",
        "        400 / 400 steps | return 1.0 | 9.0 s",
    ]


def test_members_trained_one_after_another_count_the_runs_to_come():
    """Seed 0's runs of 300 steps, evaluated every 100: after its second
    evaluation, its last 100 steps and seed 1's 300, at 0.01 s a step."""
    lines, now = [], [0.0]
    printer = ProgressPrinter(
        "PPO", ["seed 0", "seed 1"], 300, False, write=lines.append, clock=lambda: now[0]
    )
    for step, elapsed in [(100, 1.0), (200, 2.0)]:
        now[0] = elapsed
        printer({"timestep": np.array(step), RETURN: np.array(1.0)}, 0)
    assert lines[-1].endswith("| seed 0 | 2.0 s, about 4 s left")


@pytest.mark.parametrize(
    "seconds, words",
    [(45, "45 s"), (130, "2 min"), (5_400, "1 h 30 min"), (7_500, "2 h 05 min")],
)
def test_a_duration_reads_in_seconds_minutes_or_hours(seconds, words):
    assert describe_duration(seconds) == words


def test_an_unknown_logging_word_is_refused():
    with pytest.raises(ValueError, match="pass a LoggingConfig"):
        tiny_ppo().train(seed=0, n_timesteps=32, logging_config="verbose")


def test_evaluate_returns_each_members_fresh_episodes():
    """CartPole pays 1 per step and ends at 500 steps at the latest: a return
    is its episode's length, a whole number from 1 to 500. An agent trained
    for 16 steps does not balance the pole for 500."""
    agent = tiny_ppo()
    state, _ = agent.train(seed=[0, 1, 2], n_timesteps=16, logging_config=None)
    returns = np.asarray(agent.evaluate(state, episodes=4, seed=3))
    assert returns.shape == (3, 4)
    np.testing.assert_array_equal(returns, np.round(returns))
    assert returns.min() >= 1 and returns.max() < 500
    again = np.asarray(agent.evaluate((state, None), episodes=4, seed=3))
    np.testing.assert_array_equal(again, returns)
    other = np.asarray(agent.evaluate(state, episodes=4, seed=4))
    assert not np.array_equal(other, returns)

    one_member = jax.tree.map(lambda leaf: leaf[0], state)
    parts = agent.policy_parts()(one_member)
    key = jax.random.split(jax.random.PRNGKey(3), 3)[0]
    episode_returns, lengths = policy_returns(agent.env_args, parts, 4, key)
    np.testing.assert_array_equal(episode_returns, lengths)
    np.testing.assert_array_equal(episode_returns, returns[0])
    assert jnp.all(lengths >= 1)


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore::ajax.environments.gymnasium.PythonEnvironmentWarning"
)
def test_evaluate_runs_on_a_gymnasium_environment():
    """As on gymnax's CartPole: a return is its episode's length, a whole
    number from 1 to 500, and 16 steps of training do not reach 500."""
    agent = PPO("gymnasium:CartPole-v1", n_envs=2, n_steps=8, n_epochs=1)
    state, _ = agent.train(seed=[0, 1], n_timesteps=16, logging_config=None)
    returns = np.asarray(agent.evaluate(state, episodes=3, seed=3))
    assert returns.shape == (2, 3)
    np.testing.assert_array_equal(returns, np.round(returns))
    assert returns.min() >= 1 and returns.max() < 500
    again = np.asarray(agent.evaluate(state, episodes=3, seed=3))
    np.testing.assert_array_equal(again, returns)


def test_a_sweep_prints_a_line_per_configuration_under_each_evaluation():
    lines = []
    printer = ProgressPrinter(
        "PPO",
        ["seed 0", "seed 1", "seed 0", "seed 1"],
        200,
        write=lines.append,
        clock=lambda: 0.0,
        groups=["lr=0.1", "lr=0.1", "lr=0.2", "lr=0.2"],
    )
    for member, score in enumerate([10.0, 30.0, 50.0, 90.0]):
        printer({"timestep": np.array(100), RETURN: np.array(score)}, member)
    # All four: mean 45, from 10 to 90. lr=0.1: mean 20; lr=0.2: mean 70.
    assert lines == [
        "        100 / 200 steps | return 45.0"
        " (min 10.0, max 90.0 over 4 runs) | 0.0 s",
        "              lr=0.1: return 20.0 (min 10.0, max 30.0 over 2 seeds)",
        "              lr=0.2: return 70.0 (min 50.0, max 90.0 over 2 seeds)",
    ]


def test_seeds_of_one_configuration_print_no_configuration_line():
    lines = []
    printer = ProgressPrinter(
        "PPO", ["seed 0", "seed 1"], 200, write=lines.append, groups=["", ""]
    )
    printer({"timestep": np.array(100), RETURN: np.array(1.0)}, 0)
    printer({"timestep": np.array(100), RETURN: np.array(3.0)}, 1)
    assert len(lines) == 1
    assert "over 2 seeds" in lines[0]


def test_rows_group_by_configuration_in_the_sweeps_order():
    members = [
        {"seed": 1, "tau": 0.5},
        {"seed": 2, "tau": 0.5},
        {"seed": 1, "tau": 0.1},
        {"seed": 2, "tau": 0.1},
    ]
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    grouped = by_configuration(members, values)
    assert [configuration for configuration, _ in grouped] == [
        {"tau": 0.5},
        {"tau": 0.1},
    ]
    np.testing.assert_array_equal(grouped[0][1], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(grouped[1][1], [[5.0, 6.0], [7.0, 8.0]])


def test_rows_that_do_not_match_the_members_are_refused():
    with pytest.raises(ValueError, match="a row per member"):
        by_configuration([{"seed": 0}], np.zeros((2, 3)))


def test_a_default_sweep_prints_and_returns_each_configuration(capsys):
    agent = tiny_ppo()
    _, evaluations = agent.train(
        seed=[0, 1], n_timesteps=32, sweep={"ent_coef": [0.0, 0.01]}
    )
    printed = capsys.readouterr().out.splitlines()
    # Two evaluations (every 16 steps), each a line and one per configuration.
    assert len(printed) == 1 + 2 * 3 + 1
    assert "over 4 runs" in printed[1]
    assert printed[2].strip().startswith("ent_coef=0: return")
    assert printed[3].strip().startswith("ent_coef=0.01: return")
    grouped = agent.by_configuration(evaluations.returns)
    assert [configuration for configuration, _ in grouped] == [
        {"ent_coef": 0.0},
        {"ent_coef": 0.01},
    ]
    np.testing.assert_array_equal(grouped[1][1], evaluations.returns[2:])


def test_an_agent_prints_as_the_call_that_builds_it():
    agent = PPO("CartPole-v1", n_envs=2, n_steps=8, actor_architecture=("8", "tanh"))
    shown = repr(agent)
    assert shown.startswith("PPO(env_id='CartPole-v1'")
    assert "n_envs=2" in shown
    assert "actor_architecture=('8', 'tanh')" in shown
    assert "gamma" not in shown  # left at its default
    rebuilt = eval(shown, {"PPO": PPO})
    assert rebuilt.config == agent.config


def test_the_closing_line_gives_the_layout_of_the_returns_for_one_seed(capsys):
    """One seed has its row too (docs/guide/training.md, "Seeds"): 160
    steps are 10 evaluations, so the returns are 1 by 10, and the last
    line says so."""
    _, evaluations = tiny_ppo().train(seed=0, n_timesteps=160)
    last = capsys.readouterr().out.splitlines()[-1]
    assert evaluations.returns.shape == (1, 10)
    assert last.startswith("Done in ")
    assert "a row per seed and a column per evaluation: shape (1, 10)." in last


def test_printing_the_evaluations_opens_on_their_layout():
    table = Evaluations(
        {
            "timestep": np.array([[100, 200, 300]] * 2),
            RETURN: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        }
    )
    shown = repr(table)
    assert shown.startswith(
        "Evaluations: evaluations.returns has a row per seed and a column per"
        " evaluation: shape (2, 3)."
    )
    assert RETURN in shown and "timestep" in shown
    # The arrays stay a dict's.
    assert dict(table)[RETURN].shape == (2, 3)


def test_a_sweeps_closing_line_counts_runs():
    """A sweep's rows are its runs, a configuration with a seed each."""
    lines: list = []
    printer = ProgressPrinter(
        "SAC",
        ["seed 0", "seed 0"],
        200,
        write=lines.append,
        clock=lambda: 0.0,
        groups=["tau=0.1", "tau=0.2"],
    )
    printer.close(Evaluations({RETURN: np.zeros((2, 4))}))
    assert lines[-1].endswith(
        "a row per run and a column per evaluation: shape (2, 4)."
    )
