"""``ajax.compare``: AdaStop's error rate, power and determinism on synthetic
scores, the training rounds it asks for, and a tiny comparison of two agents."""

import matplotlib
import numpy as np
import pytest

import ajax
from ajax.compare import AdaStop, auc, compare, final_return, last_third

matplotlib.use("Agg")


def run(test, scores):
    """Feed each agent's ``(K, n)`` scores interim by interim; the runs each used."""
    used = dict.fromkeys(test.names, 0)
    while not test.finished:
        active = test.active
        for a in active:
            used[a] += test.n
        test.update({a: scores[a][test.k] for a in active})
    return used


@pytest.mark.parametrize("names", [("a", "b"), ("a", "b", "c")])
def test_family_wise_error_stays_at_alpha_under_the_null(names):
    # adastop 0.1.3 compares |T| with signed quantiles: about 2 alpha, a
    # failure here. Three agents vs a baseline measure 0.054 over 20,000
    # replicates: the paper's shared permutation is exact for two agents only.
    rng, reps, alpha = np.random.default_rng(0), 1500, 0.05
    rejected = 0
    for r in range(reps):
        pairs = [(names[0], b) for b in names[1:]]
        test = AdaStop(names, alpha=alpha, B=1000, comparisons=pairs, seed=r)
        run(test, {a: rng.normal(size=(5, 5)) for a in names})
        rejected += any(d != "equal" for d in test.decisions.values())
    assert rejected / reps <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)


def test_shifts_are_found_early_with_their_sign_and_the_rest_left_equal():
    rng = np.random.default_rng(1)
    scores = {a: rng.normal(size=(5, 5)) for a in "ab"} | {
        "c": rng.normal(3, size=(5, 5))
    }
    test = AdaStop("abc", B=2000)
    used = run(test, scores)
    assert test.decisions == {
        ("a", "b"): "equal",
        ("a", "c"): "smaller",
        ("b", "c"): "smaller",
    }
    assert used["c"] < 25 == used["a"]


def test_scores_tied_up_to_rounding_are_not_a_difference():
    # One run at 0.6 among 0.1s: every permutation gives the same |T|, but
    # summed in another order its last bits differ (adastop rejects on these).
    test = AdaStop("ab", n=4, K=2, B=2000)
    run(test, {"a": [[0.1] * 4, [0.1] * 3 + [0.6]], "b": np.full((2, 4), 0.1)})
    assert test.decisions == {("a", "b"): "equal"}


def test_decisions_repeat_for_a_seed():
    scores = {
        a: np.random.default_rng(i).normal(i / 2, size=(5, 5))
        for i, a in enumerate("abc")
    }

    def decide(seed):
        test = AdaStop("abc", B=500, seed=seed)
        return run(test, scores), test.decisions

    assert decide(3) == decide(3)


def test_scores_read_the_last_evaluation_the_last_third_or_the_mean():
    curves = np.array([[0.0, 1, 2, 3, 4, 5], [6, 6, 6, 6, 6, 0]])
    assert final_return(curves).tolist() == [5, 0]
    assert last_third(curves).tolist() == [4.5, 3]
    assert auc(curves).tolist() == [2.5, 5]


class Constant:
    """Trains nothing: every evaluation of every seed returns ``value``."""

    calls: list = []

    def __init__(self, value):
        self.value = value

    def train(self, seed, **kwargs):
        Constant.calls.append(list(seed))
        return None, {"Eval/episodic mean reward": np.full((len(seed), 2), self.value)}


def test_compare_trains_new_seeds_for_undecided_agents_only():
    # a = b throughout; c's runs score apart from the start, decided at the
    # second interim (one at the first is too few permutations).
    factories = {
        a: lambda v=v: Constant(v) for a, v in (("a", 1.0), ("b", 1.0), ("c", 9.0))
    }
    Constant.calls.clear()
    result = compare(factories, 10, 5, baseline="a", n=3, K=3, B=500, seed=7)
    assert result.decisions == {("a", "b"): "equal", ("a", "c"): "smaller"}
    assert result.seeds == {
        "a": [7, 8, 9, 10, 11, 12, 13, 14, 15],
        "b": [16, 17, 18, 19, 20, 21, 22, 23, 24],
        "c": [25, 26, 27, 28, 29, 30],
    }
    assert len(Constant.calls) == 8 and result.scores["c"].tolist() == [9.0] * 6
    assert result.plot().get_title() == "a = b, a < c"


def test_two_agents_compare_end_to_end():
    result = compare(
        {
            "fast": lambda: ajax.DQN("CartPole-v1", learning_rate=1e-3),
            "slow": lambda: ajax.DQN("CartPole-v1", learning_rate=1e-4),
        },
        n_timesteps=1_000,
        eval_every=500,
        n=2,
        K=2,
        B=200,
        num_episode_test=2,
    )
    # Two runs a side over two interims cannot reject anything (the smallest
    # level they reach is 2/36): this checks the wiring to real training.
    assert result.decisions == {("fast", "slow"): "equal"}
    assert result.seeds == {"fast": [0, 1, 2, 3], "slow": [4, 5, 6, 7]}
    assert all(np.isfinite(s).all() and len(s) == 4 for s in result.scores.values())
