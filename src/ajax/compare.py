"""Compare agents with AdaStop: train seeds in rounds until the evidence decides.

AdaStop (Mathieu, Della Vecchia, Shilova, Centa de Medeiros, Kohler, Maillard
and Preux, "AdaStop: adaptive statistical testing for sound comparisons of
Deep RL agents", TMLR 2024, arXiv:2306.10882) is a permutation-based group
sequential test with a step-down procedure. At each of up to ``K`` interims,
every agent still in an undecided comparison adds ``n`` runs. A comparison
stops once its two score distributions are shown to differ. The chance of
declaring any difference where there is none (the family-wise error) stays at
most ``alpha``. A comparison still undecided after ``K`` interims is "equal":
not shown to differ, which does not prove the agents equal.

docs/guide/comparing-agents.md lists the differences from the paper and from
the authors' ``adastop`` package.

:func:`estimate` measures one agent instead: it trains seeds until a
confidence sequence pins down each chosen metric.
"""

import dataclasses
import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ajax.logging.wandb_logging import LoggingConfig


def final_return(curves: np.ndarray) -> np.ndarray:
    """Each seed's last evaluation; ``curves`` is ``(seeds, evaluations)``."""
    return curves[:, -1]


def last_third(curves: np.ndarray) -> np.ndarray:
    """Each seed's mean over the last third of its evaluations (at least one)."""
    return curves[:, -max(1, curves.shape[1] // 3) :].mean(axis=1)


def auc(curves: np.ndarray) -> np.ndarray:
    """Each seed's area under its evaluation curve, per evaluation: the mean."""
    return curves.mean(axis=1)


class AdaStop:
    """AdaStop's test (the paper's Algorithm 3), fed one interim at a time.

    Args:
        names: the agents.
        n: the runs each agent adds per interim.
        K: the most interims.
        alpha: the family-wise error rate.
        B: the Monte Carlo permutations.
        comparisons: the pairs of names to test; every pair when None.
        seed: draws the permutations, independently of the scores.
    """

    def __init__(
        self,
        names: Sequence[str],
        n: int = 5,
        K: int = 5,
        alpha: float = 0.05,
        B: int = 10_000,
        comparisons: Optional[Sequence[tuple[str, str]]] = None,
        seed: int = 0,
    ):
        self.names, self.n, self.K, self.k = list(names), n, K, 0
        self.comparisons = list(comparisons or itertools.combinations(self.names, 2))
        self.decisions = dict.fromkeys(self.comparisons, "continue")
        # How many of the B permutations may have crossed a boundary by each
        # interim: the spending alpha * k / K of Eq. 10, on the 1 / B grid.
        self._spent = np.floor(alpha * B * np.arange(K + 1) / K + 1e-9).astype(int)
        # One permutation per interim, shared by every comparison (Appendix
        # D.1), as signs over a pair's 2n pooled scores: +1 counts for the
        # first agent. Permutation 0 is the identity, the observed statistic.
        identity = np.repeat([1.0, -1.0], n)
        draws = np.broadcast_to(identity, (B, K, 2 * n))
        self._signs = np.random.default_rng(seed).permuted(draws, axis=-1)
        self._signs[0] = identity
        # Eq. 8 before the absolute value, per comparison, interim, permutation,
        # and the sum of |scores| behind it, the scale of its rounding.
        self._sums = np.zeros((len(self.comparisons), K, B))
        self._size = np.zeros(len(self.comparisons))

    @property
    def active(self) -> set[str]:
        """The agents in an undecided comparison: the next interim's runs."""
        return {
            a for pair, d in self.decisions.items() if d == "continue" for a in pair
        }

    @property
    def finished(self) -> bool:
        return self.k == self.K or not self.active

    def update(self, scores: Mapping[str, Any]) -> dict:
        """Run interim ``k`` on the ``n`` new scores of every active agent;
        ``decisions[(a, b)]`` becomes "larger" (a's mean is larger),
        "smaller", or, after the last interim, "equal"."""
        k = self.k
        undecided = [
            c
            for c, pair in enumerate(self.comparisons)
            if self.decisions[pair] == "continue"
        ]
        for c in undecided:
            pooled = np.concatenate(
                [np.asarray(scores[a], float) for a in self.comparisons[c]]
            )
            if pooled.shape != (2 * self.n,):
                raise ValueError(f"interim {k} needs {self.n} scores per agent")
            signed = (self._signs[:, k] * pooled).sum(axis=1)
            self._sums[c, k] = signed + (self._sums[c, k - 1] if k else 0)
            self._size[c] += np.abs(pooled).sum()
        # Step-down: reject the largest statistic while it crosses the
        # boundary of the comparisons left, recomputed after each rejection.
        while undecided:
            observed = self._sums[undecided, k, 0]
            top = int(np.argmax(np.abs(observed)))
            if abs(observed[top]) <= self._boundary(undecided, k):
                break
            pair = self.comparisons[undecided.pop(top)]
            self.decisions[pair] = "larger" if observed[top] > 0 else "smaller"
        self.k += 1
        if self.k == self.K:
            for c in undecided:
                self.decisions[self.comparisons[c]] = "equal"
        return self.decisions

    def _boundary(self, comparisons: list[int], k: int) -> float:
        """Eqs. 10-11 for ``comparisons``: interim k's boundary over the
        permutations that stayed at or under every earlier boundary."""
        # |.| before the max, as in Eqs. 8-9: adastop 0.1.3 takes the signed
        # max here, which about doubles its error rate.
        stat = np.abs(self._sums[comparisons, : k + 1]).max(axis=0)
        # Statistics equal in exact arithmetic can differ in their last bits
        # (summed in another order); within tol they tie, as in scipy's
        # permutation_test, or rounding alone could reject.
        tol = 1e-9 * self._size[comparisons].max()
        alive, crossed = np.ones(stat.shape[1], bool), 0
        for m in range(k + 1):
            values = np.sort(stat[m, alive])
            # The smallest value that at most the level left exceed; level
            # that ties left unspent carries over, as in the authors' code.
            i = len(values) - 1 - (self._spent[m + 1] - crossed)
            bound = (values[i] if i >= 0 else 0.0) + tol
            crossed += int((stat[m, alive] > bound).sum())
            alive &= stat[m] <= bound
        return float(bound)


@dataclasses.dataclass(frozen=True)
class Comparison:
    """What :func:`compare` decided, and on which runs.

    ``decisions[(a, b)]`` is "larger", "smaller" or "equal", a against b;
    ``seeds[a]`` and ``scores[a]`` are a's runs in training order.
    """

    decisions: dict
    seeds: dict
    scores: dict

    def plot(self, ax: Any = None) -> Any:
        """Each agent's scores, one box per agent, the decisions as the title
        (needs matplotlib)."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots()[1]
        names = list(self.scores)
        ax.boxplot([self.scores[a] for a in names])
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([f"{a}\n{len(self.scores[a])} seeds" for a in names])
        sign = {"larger": ">", "smaller": "<", "equal": "="}
        ax.set_title(
            ", ".join(f"{a} {sign[d]} {b}" for (a, b), d in self.decisions.items())
        )
        return ax


def compare(
    agents: Mapping[str, Callable[[], Any]],
    n_timesteps: int,
    eval_every: int,
    score: Callable[[np.ndarray], np.ndarray] = final_return,
    *,
    baseline: Optional[str] = None,
    num_episode_test: int = 10,
    seed: int = 0,
    **adastop: Any,
) -> Comparison:
    """Train and score agents in AdaStop's interims until it decides.

    At each interim, every agent still in an undecided comparison trains
    ``n`` new seeds in one ``train`` call, evaluated every ``eval_every``
    steps; ``score`` turns the ``(seeds, evaluations)`` returns into one
    score per seed. At most ``n * K`` runs per agent; the paper advises
    ``n * K >= 30``.

    Args:
        agents: a factory per name, building the agent to train.
        n_timesteps: each run's training budget.
        eval_every: steps between evaluations.
        score: :func:`final_return`, :func:`last_third`, :func:`auc` or
            another map from evaluation curves to per-seed scores.
        baseline: compare it with each other agent; every pair when None.
        num_episode_test: episodes per evaluation.
        seed: the permutations' seed and the first run seed. Agents train
            disjoint seeds, so their runs are independent.
        **adastop: :class:`AdaStop`'s ``n``, ``K``, ``alpha`` and ``B``.
    """
    names = list(agents)
    pairs = [(baseline, a) for a in names if a != baseline] if baseline else None
    test = AdaStop(names, comparisons=pairs, seed=seed, **adastop)
    n, K = test.n, test.K
    logging = LoggingConfig(config={}, log_frequency=eval_every, use_wandb=False)
    seeds: dict = {a: [] for a in names}
    scores: dict = {a: [] for a in names}
    while not test.finished:
        new = {}
        for i, name in enumerate(names):
            if name in test.active:
                run_seeds = [seed + (i * K + test.k) * n + r for r in range(n)]
                _, evals = agents[name]().train(
                    seed=run_seeds,
                    n_timesteps=n_timesteps,
                    num_episode_test=num_episode_test,
                    logging_config=logging,
                )
                new[name] = score(np.asarray(evals["Eval/episodic mean reward"]))
                seeds[name] += run_seeds
                scores[name] += new[name].tolist()
        test.update(new)
    return Comparison(
        dict(test.decisions), seeds, {a: np.asarray(s) for a, s in scores.items()}
    )


def half_width(width: float) -> Callable[[float, float], bool]:
    """A rule for :func:`estimate`: the interval is at most ``2 * width`` wide."""
    return lambda lower, upper: (upper - lower) / 2 <= width


def threshold(value: float) -> Callable[[float, float], bool]:
    """A rule for :func:`estimate`: the interval lies above or below ``value``,
    so the mean is decided against it."""
    return lambda lower, upper: lower > value or upper < value


@dataclasses.dataclass(frozen=True)
class Estimate:
    """What :func:`estimate` measured, and on which runs.

    Per metric: ``mean`` is the scores' sample mean; ``interval`` the
    ``(lower, upper)`` confidence interval when it stopped (adastop's "est"
    is its midpoint, not the mean); ``met`` whether its rule held then;
    ``scores`` one per seed, in training order. ``seeds`` are the runs.
    """

    mean: dict
    interval: dict
    met: dict
    scores: dict
    seeds: list


def estimate(
    agent: Callable[[], Any],
    n_timesteps: int,
    eval_every: int,
    metrics: Mapping[str, tuple[Callable, tuple[float, float], Callable]],
    *,
    delta: float = 0.05,
    n: int = 8,
    max_seeds: int = 64,
    num_episode_test: int = 10,
    seed: int = 0,
) -> Estimate:
    """Train one agent ``n`` seeds per ``train`` call until every metric's
    rule holds or ``max_seeds`` have trained.

    Each metric's mean gets the hedged betting confidence sequence of
    Waudby-Smith and Ramdas ("Estimating means of bounded random variables
    by betting", JRSS-B, Section 4.4): ``adastop``'s ``Bounded_Confseq``,
    from the ``compare`` extra. It holds at every seed count at once, so
    stopping on what the scores show keeps its level. The metrics split
    ``delta`` evenly (a union bound), so all intervals hold together with
    probability at least ``1 - delta``. ``n_timesteps``, ``eval_every`` and
    ``num_episode_test`` are as in :func:`compare`.

    Args:
        agent: builds the agent to train.
        metrics: per name, ``(score, (low, high), rule)``. ``score`` maps
            ``train``'s evaluations, each ``(seeds, evaluations)``, to one
            value per seed, which ``low`` and ``high`` must bound for every
            possible run, not only those seen. ``rule`` is :func:`half_width`,
            :func:`threshold` or another test of ``(lower, upper)``.
        delta: the chance that any interval misses its mean.
        seed: the first run seed; the runs are ``seed, seed + 1, ...``.
    """
    from adastop.confidence_sequences import Bounded_Confseq

    logging = LoggingConfig(config={}, log_frequency=eval_every, use_wandb=False)
    sequences = {
        name: Bounded_Confseq(delta=delta / len(metrics), bounds=bounds)
        for name, (_, bounds, _) in metrics.items()
    }
    seeds: list = []
    scores: dict = {name: [] for name in metrics}
    met = dict.fromkeys(metrics, False)
    while not all(met.values()) and len(seeds) < max_seeds:
        first = seed + len(seeds)
        run_seeds = list(range(first, first + min(n, max_seeds - len(seeds))))
        _, evals = agent().train(
            seed=run_seeds,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            logging_config=logging,
        )
        evals = {key: np.asarray(value) for key, value in evals.items()}
        for name, (score, (low, high), rule) in metrics.items():
            new = np.asarray(score(evals), float)
            # observe() does not check: a score out of bounds would void the
            # level silently.
            inside = (low <= new) & (new <= high)
            if new.shape != (len(run_seeds),) or not inside.all():
                raise ValueError(
                    f"{name} needs one score per seed in [{low}, {high}]: {new}"
                )
            for x in new:
                sequences[name].observe(x)
            scores[name] += new.tolist()
            met[name] = rule(*_interval(sequences[name]))
        seeds += run_seeds
    return Estimate(
        {name: float(np.mean(s)) for name, s in scores.items()},
        {name: _interval(cs) for name, cs in sequences.items()},
        met,
        {name: np.asarray(s) for name, s in scores.items()},
        seeds,
    )


def _interval(sequence: Any) -> tuple[float, float]:
    return float(sequence.stats["lb"][-1]), float(sequence.stats["ub"][-1])
