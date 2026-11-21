"""What a first run shows and returns.

Contract (docs/getting-started/quickstart.md, ``ActorCritic.train``,
``ajax.logging.progress``):
  - a train call given no logging_config evaluates about ten times, prints
    a first line, one line per evaluation and a last line, and returns the
    evaluations; logging_config=None evaluates nothing, prints nothing and
    returns None; a LoggingConfig of the caller's prints nothing;
  - the evaluations are the dict of logged names, with ``returns`` (members
    x evaluations) and ``timesteps`` (one per evaluation);
  - a printed evaluation is one line for all members trained side by side
    (their mean return and its range), once each member's has come;
  - ``agent.evaluate(state, episodes, seed)`` is the return of fresh
    episodes of each member's policy, members x episodes, the same for the
    same seed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import PPO
from ajax.evaluate import policy_returns
from ajax.logging import LoggingConfig
from ajax.logging.progress import (
    Evaluations,
    ProgressPrinter,
    default_log_frequency,
)

RETURN = "Eval/episodic mean reward"


@pytest.mark.parametrize(
    "n_timesteps, frequency",
    # A tenth of the call, rounded up: 1,001 / 10 = 100.1.
    [(40_960, 4_096), (1_000, 100), (1_001, 101), (5, 1), (1, 1)],
)
def test_a_default_run_evaluates_every_tenth_of_its_steps(n_timesteps, frequency):
    assert default_log_frequency(n_timesteps) == frequency


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


def test_logging_config_none_is_silent_and_evaluates_nothing(capsys):
    _, evaluations = tiny_ppo().train(seed=0, n_timesteps=32, logging_config=None)
    assert evaluations is None
    assert capsys.readouterr().out == ""


def test_a_logging_config_of_the_callers_prints_nothing(capsys):
    logging = LoggingConfig(log_frequency=16, backends=())
    _, evaluations = tiny_ppo().train(seed=0, n_timesteps=32, logging_config=logging)
    np.testing.assert_array_equal(evaluations.timesteps, [16, 32])
    assert capsys.readouterr().out == ""


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
