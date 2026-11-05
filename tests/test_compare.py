"""``ajax.compare``: AdaStop's error rate, power and determinism on synthetic
scores, the training rounds it asks for, and a tiny comparison of two agents;
the same for :func:`estimate`'s confidence sequences, and their coverage."""

import inspect
import itertools
from functools import partial

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


def test_the_published_defaults_stay_as_they_are():
    """Pins the published defaults: AdaStop's 10,000 permutations and
    estimate's 64-seed cap; changing one changes every call that leaves it
    out, a breaking change."""
    assert inspect.signature(AdaStop).parameters["B"].default == 10_000
    assert inspect.signature(estimate).parameters["max_seeds"].default == 64


def test_decisions_repeat_for_a_seed():
    scores = {
        a: np.random.default_rng(i).normal(i / 2, size=(5, 5))
        for i, a in enumerate("abc")
    }

    def decide(seed):
        test = AdaStop("abc", B=500, seed=seed)
        return run(test, scores), test.decisions

    assert decide(3) == decide(3)


def enumerated_p_value(differences):
    """The exact two-sided sign-flip p-value, over every one of the 2^n sign
    patterns: the share as extreme as the observed sum."""
    d = np.asarray(differences, float)
    flips = np.array(list(itertools.product([-1.0, 1.0], repeat=d.size)))
    return (np.abs(flips @ d) >= abs(d.sum()) - 1e-12).mean()


@pytest.mark.parametrize(
    "differences, alpha, verdict",
    [
        # 1 + 2 + 3 + 4 + 5 = 15 is the largest sum: 2 of 32 patterns reach it.
        ([1, 2, 3, 4, 5], 0.10, "larger"),
        ([1, 2, 3, 4, 5], 0.05, "equal"),
        ([-1, -2, -3, -4, -5], 0.10, "smaller"),
        # 13 falls one flip short of the largest: 4 of 32 reach it.
        ([5, 4, 3, 2, -1], 0.15, "larger"),
        ([5, 4, 3, 2, -1], 0.10, "equal"),
        ([0, 0, 0, 0, 0], 0.50, "equal"),
    ],
)
def test_two_agents_decide_as_the_enumerated_sign_flip_test(
    differences, alpha, verdict
):
    """Flipping the pooled scores, or dropping the identity, leaves this test
    at another level than the textbook one on the same differences."""
    p_value = enumerated_p_value(differences)
    assert (p_value <= alpha) == (verdict != "equal") or p_value == 1.0

    test = AdaStop("ab", n=5, K=1, alpha=alpha, B=200_000, paired=True)
    test.update({"a": differences, "b": np.zeros(5)})

    assert test.decisions[("a", "b")] == verdict


def test_comparisons_share_their_sign_flips():
    """Two identical comparisons are one question: under shared flips the
    largest of them is that one statistic, and decides as it does alone
    (2 of 32 patterns reach 15, within alpha 0.1). Flips drawn per comparison
    would raise the boundary for nothing, and turn this into "equal"."""
    test = AdaStop(
        "abc",
        n=5,
        K=1,
        alpha=0.1,
        B=200_000,
        comparisons=[("a", "b"), ("a", "c")],
        paired=True,
    )

    test.update({"a": [1, 2, 3, 4, 5], "b": np.zeros(5), "c": np.zeros(5)})

    assert test.decisions == {("a", "b"): "larger", ("a", "c"): "larger"}


def test_paired_evidence_accumulates_across_interims():
    """Hand-derived with [1, 2, 3], a quarter of whose sign patterns reach 6:
    one interim cannot reject at alpha 0.2, and two reach 12, which 2 of 64
    patterns do. A test reading the last interim alone stays "equal"."""
    test = AdaStop("ab", n=3, K=2, alpha=0.2, B=100_000, paired=True)
    test.update({"a": [1, 2, 3], "b": np.zeros(3)})
    assert test.decisions[("a", "b")] == "continue"

    test.update({"a": [1, 2, 3], "b": np.zeros(3)})
    assert test.decisions[("a", "b")] == "larger"


def test_what_a_row_shares_leaves_a_paired_comparison():
    """Pooled scores would carry the seed's effect into the statistic: a
    large row effect would then hide the gap a paired test exists to find."""
    gap, effect = np.array([1.0, 2, 3, 4, 5]), np.array([9.0, -40, 3, 70, -11])
    plain = AdaStop("ab", n=5, K=1, alpha=0.1, B=20_000, paired=True)
    shared = AdaStop("ab", n=5, K=1, alpha=0.1, B=20_000, paired=True)

    plain.update({"a": gap, "b": np.zeros(5)})
    shared.update({"a": effect + gap, "b": effect})

    assert plain.decisions == shared.decisions == {("a", "b"): "larger"}


def test_correlated_pairs_gain_power():
    """Rows sharing 90% of their variance leave differences of variance 0.2
    instead of 2, so a one-sigma gap over 10 rows stands at about 7 standard
    errors paired and 2.2 unpaired. Losing the pairing loses that margin."""
    rng, reps = np.random.default_rng(5), 200
    hits = {True: 0, False: 0}
    for r in range(reps):
        shared = np.sqrt(0.9) * rng.normal(size=(2, 5))
        noise = np.sqrt(0.1) * rng.normal(size=(2, 2, 5))
        scores = {"a": shared + noise[0] + 1.0, "b": shared + noise[1]}
        for paired in hits:
            test = AdaStop("ab", n=5, K=2, alpha=0.05, B=1000, seed=r, paired=paired)
            run(test, scores)
            hits[paired] += test.decisions[("a", "b")] == "larger"
    assert hits[True] / reps >= 0.95
    assert hits[False] / reps <= 0.75


@pytest.mark.parametrize("names", ["ab", "abcd"])
def test_paired_family_wise_error_stays_at_alpha_under_the_null(names):
    """A statistic left unflipped, or flips drawn apart from the observed
    row, inflates the error. The rows share 60% of their variance, which
    pairing must cancel, not count. Tolerance: 3 binomial standard errors of
    the replicates; 12 agents measure 0.049 over 3,000 replicates."""
    rng, reps, alpha = np.random.default_rng(6), 1500, 0.05
    rejected = 0
    for r in range(reps):
        shared = np.sqrt(0.6) * rng.normal(size=(5, 5))
        test = AdaStop(names, alpha=alpha, B=1000, seed=r, paired=True)
        run(
            test,
            {a: shared + np.sqrt(0.4) * rng.normal(size=(5, 5)) for a in names},
        )
        rejected += any(d != "equal" for d in test.decisions.values())
    assert rejected / reps <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)


def test_an_agent_that_stopped_leaves_the_others_rows_aligned():
    """Rows go by interim, so c stopping early must change nothing for a and
    b: the pair decides as it would without c. A row index counted per agent
    would shift a's rows against b's from then on."""
    rng = np.random.default_rng(7)
    scores = {a: rng.normal(size=(3, 6)) for a in "ab"}
    scores["a"] = scores["a"] + np.array([[0.0], [1.0], [1.0]])
    scores["c"] = rng.normal(50, size=(3, 6))
    with_c = AdaStop("abc", n=6, K=3, alpha=0.2, B=5000, paired=True)
    without_c = AdaStop("ab", n=6, K=3, alpha=0.2, B=5000, paired=True)

    used = run(with_c, scores)
    run(without_c, scores)

    # All six rows one sign: 2 of 64 patterns, within the first 0.2 / 3.
    assert used["c"] == 6 < used["a"]
    assert with_c.decisions[("a", "b")] == without_c.decisions[("a", "b")]


@pytest.mark.parametrize(
    "scores",
    [
        {"a": np.zeros(4), "b": np.zeros(3)},
        {"a": np.zeros(4), "b": np.zeros(1)},  # would broadcast
        {"a": np.zeros(3), "b": np.zeros(3)},
    ],
)
def test_paired_rows_of_unequal_lengths_are_refused(scores):
    """A difference of unequal rows pairs the wrong seeds, or broadcasts one
    score across them all."""
    test = AdaStop("ab", n=4, K=2, paired=True)
    with pytest.raises(ValueError):
        test.update(scores)


def test_characterization_the_default_test_stays_as_it_was():
    """Pins current behaviour, not verified against a spec: the unpaired test
    before paired mode existed. Forty permutations on a close call turn on the
    draws themselves, so verdicts over 64 seeds move with the default stream;
    an option drawing from it first fails here."""
    a, b = np.random.default_rng(3).normal(size=(2, 5))
    verdicts = ""
    for seed in range(64):
        test = AdaStop("ab", n=5, K=1, alpha=0.1, B=40, seed=seed)
        test.update({"a": a + 2.7, "b": b})
        verdicts += test.decisions[("a", "b")][0]

    x = np.random.default_rng(3).normal(size=16)
    y = np.random.default_rng(4).normal(0.5, 2.0, size=16)
    gap, interval = difference(x, y, B=2000, seed=5)

    assert (
        verdicts == "lleleeleeleeeeleelleeeleeeleeelllelleelllllellelleellelleeelelle"
    )
    assert gap == pytest.approx(-0.903848501620633, rel=1e-9)
    low, high = interval
    assert (low, high) == pytest.approx((-2.1558344823027142, 0.31378846390516485))


def test_paired_difference_drops_what_the_rows_share():
    """Hand-derived: b plus a constant has no spread in its differences, so
    the interval closes on the gap. Resampling each sample alone would leave
    the rows' own spread in it."""
    b = np.array([-3.0, 7.0, 0.5, 12.0, -8.0, 2.0])

    gap, interval = difference(b + 1.5, b, paired=True)

    assert gap == pytest.approx(1.5)
    assert interval == pytest.approx((1.5, 1.5))
    assert np.diff(difference(b + 1.5, b)[1]) > 1.0


def test_paired_resolution_keeps_what_the_rows_share():
    """Rows sharing 90% of their variance detect far smaller gaps. Replicates
    drawing each sample alone would break the pairs and report the unpaired
    test's resolution for a paired one."""
    rng = np.random.default_rng(8)
    shared = rng.normal(size=32)
    a = np.sqrt(0.9) * shared + np.sqrt(0.1) * rng.normal(size=32)
    b = np.sqrt(0.9) * shared + np.sqrt(0.1) * rng.normal(size=32)

    settings = {"n": 8, "K": 2, "reps": 40, "B": 500}
    paired = resolution(a, b, paired=True, **settings)
    unpaired = resolution(a, b, **settings)

    assert paired < unpaired / 2


@pytest.mark.parametrize("function", [difference, partial(resolution, n=2, K=1)])
def test_paired_samples_of_unequal_lengths_are_refused(function):
    """Their rows cannot be one seed number each."""
    with pytest.raises(ValueError, match="differ in length"):
        function(np.zeros(4), np.zeros(5), paired=True)


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


def test_paired_agents_train_the_same_seeds_at_each_interim():
    """Rows pair only if seed number i is one seed in every agent: disjoint
    seeds would pair strangers. An agent that stopped keeps a prefix of them."""
    agents = {"a": lambda: Constant(0.0), "b": lambda: Constant(0.0)}
    agents["c"] = lambda: Constant(99.0)

    result = compare(agents, 8, 8, seed=100, n=3, K=2, alpha=0.6, B=500, paired=True)

    # c differs in every row: 2 of 8 sign patterns, within the first 0.6 / 2.
    assert result.seeds["c"] == [100, 101, 102]
    assert result.seeds["a"] == result.seeds["b"] == [100, 101, 102, 103, 104, 105]


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


def test_the_metrics_share_delta():
    """Catches each interval spending the whole delta, which voids the level
    they promise together: a second metric widens the first's on its same
    scores."""
    draw = {"x": lambda g: g.uniform(), "y": lambda g: g.uniform()}
    x, y = ((last(k), (0, 1), half_width(0)) for k in draw)
    size = {"n": 32, "max_seeds": 128}  # few scores leave both at the whole range
    alone = estimate(lambda: Draws(**draw), 1, 1, {"x": x}, **size)
    shared = estimate(lambda: Draws(**draw), 1, 1, {"x": x, "y": y}, **size)
    assert np.ptp(shared.interval["x"]) > np.ptp(alone.interval["x"])


def test_half_width_bounds_half_the_interval():
    """Catches the rule reading the interval's whole width: (0, 2) lies within
    1 of its middle."""
    assert half_width(1.0)(0.0, 2.0)
    assert not half_width(1.0)(0.0, 2.5)


def test_an_interim_of_the_wrong_size_is_refused():
    """Catches a short interim reaching the permutations: one pooled score
    would broadcast over a pair's four signs and pass as a whole interim."""
    test = AdaStop(["a", "b"], n=2, K=2, B=16)
    with pytest.raises(ValueError, match="needs 2 scores"):
        test.update({"a": [1.0], "b": []})


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
