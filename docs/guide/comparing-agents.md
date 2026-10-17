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
