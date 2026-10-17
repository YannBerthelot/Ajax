# Comparing agents

Is one agent better than another, and how many seeds does it take to say
so? `ajax.compare` answers with AdaStop (Mathieu et al., "AdaStop: adaptive
statistical testing for sound comparisons of Deep RL agents", TMLR 2024,
[arXiv:2306.10882](https://arxiv.org/abs/2306.10882)). It trains a few seeds
of each agent, tests, and trains more only while the answer is unclear.

```python
import ajax
from ajax.compare import compare, last_third

result = compare(
    {"SAC": lambda: ajax.SAC("Pendulum-v1"), "TD3": lambda: ajax.TD3("Pendulum-v1")},
    n_timesteps=20_000,
    eval_every=1_000,
    score=last_third,
)
result.decisions  # {("SAC", "TD3"): "equal"}, say
result.seeds  # the seeds each agent trained
result.plot()  # each agent's scores, the decisions as the title
```

At each of up to `K` interims, every agent still in an undecided comparison
trains `n` new seeds in one `train` call, so all of them run in one compiled
program. Every pair of agents is compared, or with `baseline="SAC"` only SAC
against each other agent. `AdaStop` itself takes scores from anywhere, one
interim at a time, for runs recorded elsewhere.

## What it decides

- **"larger" or "smaller"**: the two agents' scores do not come from the same
  distribution, and the first agent's mean is larger (or smaller).
- **"equal"**: no difference was shown after `n * K` runs each. This does not
  show that the agents are equal: a small difference can need more runs than
  that.

The chance of reporting any difference between agents that do not differ
(the family-wise error) is at most `alpha`, whatever the number of
comparisons. That holds exactly for two agents. With three or more it rests
on the paper's shared permutations, which are an approximation there: over
20,000 simulated comparisons of equal agents at `alpha = 0.05`, two agents
gave 0.044 and three or four gave 0.054 to 0.056.

## What a verdict leaves open

Two numbers say how much a verdict tells, and belong next to it:

```python
from ajax.compare import difference, resolution

a, b = result.scores["SAC"], result.scores["TD3"]
gap, (low, high) = difference(a, b)  # mean(a) - mean(b), 95% bootstrap interval
smallest = resolution(a, b, n=5, K=5)  # the gap detected 80% of the time
```

`difference` estimates the gap between the means. `resolution` is the
smallest gap that AdaStop, with the same `n`, `K` and `alpha`, detects with
probability `power` (0.8 by default) for scores spread like these. It
simulates, as the paper's power study does, but resamples the observed
scores instead of assuming a distribution. An "equal" rules out gaps much
larger than the resolution; it says nothing about smaller ones. On normal
scores with 32 runs each (`n = 8`, `K = 4`) the resolution is about 0.75
standard deviations.

## The score

The test compares one number per run, and the choice of that number changes
the question:

- `final_return`: the last evaluation, or how the run ends.
- `last_third`: the mean over the last third of the evaluations, or how
  steadily it stays there.
- `auc`: the mean over all evaluations, or how fast it learns.

They can disagree on the same runs. Choose the score before seeing the
results: picking it afterwards voids the error guarantee.

## Cost

At most `n * K` runs per agent: 25 with the defaults `n = K = 5`. A
comparison stops early only on a clear difference; agents that differ
little, or not at all, use every run. The paper advises `n * K >= 30` for
the means to be well estimated. Very few runs cannot reject anything: with
`n = 2` and `K = 2` no outcome is extreme enough at `alpha = 0.05`. The test
itself takes milliseconds; the training is the cost.

The [reference comparison](../../benchmarks/reference/README.md) reports an
AdaStop verdict on its recorded runs.

## Differences from the paper and the authors' package

The authors' `adastop` package (0.1.3) compares the observed difference with
quantiles of the signed permuted differences, which about doubles the error
rate (0.097 at `alpha = 0.05` over 1,000 simulations of equal agents). It can
also reject on rounding alone, when every permutation gives the same
statistic. `ajax.compare` follows the paper instead, with these choices:

- Absolute values on both sides of the test, as in the paper's Eqs. 8-9.
- Statistics equal up to rounding tie.
- Level that ties leave unspent at an interim carries to the next, as in the
  authors' package.
- Permutations are always Monte Carlo (`B` draws, the identity first); the
  paper enumerates them when there are fewer than `B`.
- No early acceptance (the paper's Appendix F, which gives no error bound
  for it), as in the package's default.
- Every agent adds the same `n` runs per interim, as in the paper.

## One agent: how sure is a number?

With nothing to compare, the question is how well one agent does: its
success rate, say. `estimate` trains `n` seeds per `train` call until each
chosen metric is known well enough, or `max_seeds` have trained:

```python
from ajax.compare import estimate, final_return, half_width, threshold


def final(evals):  # evals: train's evaluations, each (seeds, evaluations)
    return final_return(evals["Eval/episodic mean reward"])


result = estimate(
    lambda: ajax.PPO("CartPole-v1"),
    n_timesteps=100_000,
    eval_every=10_000,
    metrics={  # name: (per-seed score, its true bounds, when to stop)
        "solved": (lambda evals: final(evals) >= 475, (0, 1), threshold(0.8)),
        "final return": (final, (0, 500), half_width(25)),
    },
    n=16,
    max_seeds=128,
)
result.interval  # {"solved": (0.83, 1.0), "final return": (452.1, 500.0)}, say
result.mean, result.met, result.seeds, result.scores
```

`threshold(t)` stops once the interval lies above or below `t`, deciding
whether the mean is above `t`; `half_width(w)` once it is at most `2 * w`
wide. The intervals are the hedged betting confidence sequence of
Waudby-Smith and Ramdas ("Estimating means of bounded random variables by
betting", JRSS-B, Section 4.4), the authors' `Bounded_Confseq` from the
`adastop` package: install the `compare` extra (`uv sync --extra compare`,
or depend on `ajax[compare]`). It comes from adastop's main branch until a
release includes it.

- **Valid at every seed count at once**: with probability at least
  `1 - delta` the interval holds the true mean after every round, not only
  after a number of seeds fixed in advance, so stopping once the scores look
  good enough keeps the guarantee.
- **All metrics together**: `delta` is split evenly between them (a union
  bound), so each extra metric widens every interval a little.
- **Only within true bounds**: `(low, high)` must bound every score a run
  could get, not only those seen: `(0, 500)` for CartPole's return, about
  `(-3254, 0)` for Pendulum's. A score outside them is refused.
- `mean` is the sample mean; the interval need not be centred on it.

The cost is many seeds. A metric that is 1 on every seed has a 95% lower
bound of 0.63 after 16 seeds and 0.92 after 64. On the reference
comparison's recorded runs, 16 seeds left PPO on CartPole a success rate
(final return of at least 475) of 0.88 within `[0.45, 1]` and a final
return of 468 within `[275, 500]`, and SAC on Pendulum a final return of
-146 within `[-1359, 0]`. So decide against a threshold rather than aim for
a width; prefer metrics whose bounds are close to what runs reach, such as
success rates or KPI fractions, to returns; and train many seeds per round,
which share one compiled program.
