"""``ajax.compare``: AdaStop's error rate, power and determinism on synthetic
scores, the training rounds it asks for, and a tiny comparison of two agents;
the same for :func:`estimate`'s confidence sequences, and their coverage."""

import matplotlib
import numpy as np
import pytest

import ajax
from ajax.compare import (
    AdaStop,
    auc,
    compare,
    difference,
    estimate,
    final_return,
    half_width,
    last_third,
    resolution,
    threshold,
)

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
    scores = {a: rng.normal(size=(5, 5)) for a in "ab"}
    scores.update({"c": rng.normal(3, size=(5, 5))})
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


class Draws:
    """Trains nothing: each seed evaluates once per key, to a draw from a
    generator seeded by the run seed."""

    calls: list = []

    def __init__(self, **draws):
        self.draws = draws

    def train(self, seed, **kwargs):
        Draws.calls.append(list(seed))
        rngs = [np.random.default_rng(s) for s in seed]
        return None, {
            k: np.array([[d(g) for g in rngs]]).T for k, d in self.draws.items()
        }


DISTRIBUTIONS = {  # name: (draw, true bounds, mean)
    "bernoulli": (lambda g: g.binomial(1, 0.3), (0, 1), 0.3),
    "skewed return": (lambda g: 500 * g.beta(5, 1), (0, 500), 500 * 5 / 6),
    "uniform cost": (lambda g: g.uniform(-3254, -3000), (-3254, 0), -3127.0),
}


def last(key):
    return lambda evals: evals[key][:, -1]


def test_intervals_cover_every_mean_together_with_probability_one_minus_delta():
    # Never stopping early, a final interval misses its mean iff one at some
    # seed count did (they only shrink): the anytime guarantee, three
    # metrics under one union bound.
    reps, delta, missed = 150, 0.2, 0
    agent = Draws(**{k: d for k, (d, _, _) in DISTRIBUTIONS.items()})
    metrics = {k: (last(k), b, half_width(0)) for k, (_, b, _) in DISTRIBUTIONS.items()}
    for r in range(reps):
        result = estimate(
            lambda: agent, 1, 1, metrics, delta=delta, max_seeds=16, seed=16 * r
        )
        means = {k: m for k, (_, _, m) in DISTRIBUTIONS.items()}
        missed += any(
            not lo <= means[k] <= hi for k, (lo, hi) in result.interval.items()
        )
    assert missed / reps <= delta + 3 * np.sqrt(delta * (1 - delta) / reps)


@pytest.mark.parametrize("bad", [[0.5, 1.5], [np.nan, 0.5], [0.5]])
def test_scores_outside_their_bounds_or_one_short_are_refused(bad):
    metric = (lambda evals: np.array(bad), (0, 1), half_width(0.1))
    with pytest.raises(ValueError, match="one score per seed in"):
        estimate(lambda: Draws(), 1, 1, {"x": metric}, n=2)


def test_estimates_repeat_for_a_seed():
    metrics = {"x": (last("x"), (0, 1), half_width(0.2))}
    a, b = (
        estimate(lambda: Draws(x=lambda g: g.uniform()), 1, 1, metrics, n=4, seed=3)
        for _ in "ab"
    )
    assert (a.interval, a.mean, a.seeds) == (b.interval, b.mean, b.seeds)


def test_rounds_stop_when_every_rule_holds_or_at_max_seeds():
    agent = Draws(solved=lambda g: 1.0, x=lambda g: g.uniform())
    solved = (last("solved"), (0, 1), threshold(0.5))
    Draws.calls.clear()
    result = estimate(lambda: agent, 1, 1, {"solved": solved}, n=4)
    # 16 successes put the 95% lower bound at 0.63, 12 not above 0.5.
    assert result.met == {"solved": True} and Draws.calls[-1] == [12, 13, 14, 15]
    assert result.interval["solved"][0] > 0.5 and result.mean == {"solved": 1.0}
    Draws.calls.clear()
    never = (last("x"), (0, 1), half_width(0))
    result = estimate(
        lambda: agent, 1, 1, {"solved": solved, "x": never}, n=4, max_seeds=10
    )
    assert Draws.calls == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert result.met == {"solved": False, "x": False}


def test_one_agent_is_estimated_end_to_end():
    final = lambda e: final_return(e["Eval/episodic mean reward"])  # noqa: E731
    result = estimate(
        lambda: ajax.DQN("CartPole-v1", learning_rate=1e-3),
        n_timesteps=1_000,
        eval_every=500,
        metrics={"r": (final, (0, 500), half_width(0))},
        n=2,
        max_seeds=4,
        num_episode_test=2,
    )
    lo, hi = result.interval["r"]
    assert result.seeds == [0, 1, 2, 3] and 0 <= lo <= hi <= 500
    assert result.scores["r"].shape == (4,) and not result.met["r"]


def test_difference_interval_covers_the_true_gap():
    # Percentile bootstrap intervals run a little short at 16 samples.
    rng, covered = np.random.default_rng(0), 0
    for i in range(300):
        a, b = rng.normal(1.0, 1.0, 16), rng.normal(0.0, 2.0, 16)
        gap, (low, high) = difference(a, b, B=2_000, seed=i)
        assert low < gap < high and gap == a.mean() - b.mean()
        covered += low <= 1.0 <= high
    assert 0.88 <= covered / 300 <= 0.98
    assert difference(a, b, seed=3) == difference(a, b, seed=3)


def test_resolution_is_the_gap_adastop_detects_with_the_asked_power():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=32), rng.normal(size=32)
    gap = resolution(a, b, n=8, K=4, reps=200, B=2_000)
    # A fixed-size test of 32 against 32 at 80% power needs 0.70 standard
    # deviations; AdaStop's four interims spend alpha, so a bit more.
    sd = np.sqrt((a.var() + b.var()) / 2)
    assert 0.6 < gap / sd < 1.0
    hits = 0
    for r in range(400):  # the power at that gap, for normal scores so spread
        test = AdaStop(["a", "b"], n=8, K=4, B=2_000, seed=r)
        while not test.finished:
            test.update({"a": rng.normal(0, sd, 8), "b": rng.normal(gap, sd, 8)})
        hits += test.decisions[("a", "b")] != "equal"
    assert 0.7 <= hits / 400 <= 0.95


def test_resolution_is_deterministic_and_scales_with_the_scores():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=16), rng.exponential(size=16)
    gap = resolution(a, b, n=4, K=4, reps=50, B=1_000, seed=5)
    assert gap == resolution(a, b, n=4, K=4, reps=50, B=1_000, seed=5)
    scaled = resolution(10 * a, 10 * b, n=4, K=4, reps=50, B=1_000, seed=5)
    assert scaled == pytest.approx(10 * gap, rel=1e-6)
    # Without spread, any gap is seen at once.
    assert resolution(np.ones(8), np.ones(8), n=4, K=2, reps=20, B=500) == 0.0
    with pytest.raises(ValueError, match="never reach"):
        resolution(a, b, n=2, K=2, reps=10, B=500)
