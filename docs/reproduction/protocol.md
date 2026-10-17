# Reproduction protocol

Version 1, 2026-10-17. Every batch of runs records the version it followed;
a change to this page that changes what is run or how it is judged is a
new version.

Does each Ajax agent learn like the implementation it reproduces? The
protocol trains an Ajax agent and a reference implementation of the same
algorithm on the same environment, budget and evaluation, and lets AdaStop
(see [Comparing agents](../guide/comparing-agents.md)) decide how many
seeds the answer takes. Everything is re-run: no number is taken from a
paper. The harness lives in
[`benchmarks/reference`](../../benchmarks/reference/README.md).

## Choosing the reference

The reference is a library's implementation of the algorithm
(Stable-Baselines3, Brax), or the authors' code where no library has it
(purejaxql for PQN). Both sides train on the same JAX environment, the
closest one to the task the reference was written for: gymnax's ports of
Gymnasium's CartPole and Pendulum and of MinAtar, or Brax's own tasks.
Stable-Baselines3 steps a gymnax environment through a small `VecEnv`
adapter (`refcmp_sb3.gymnax_vec_env`), so it runs the same environment code
as Ajax. Ajax itself only trains on JAX environments.

## What a comparison fixes before it runs

Each comparison in
[`refcmp_specs.py`](../../benchmarks/reference/refcmp_specs.py), listed in
full under [Specifications](specs.md), is written and committed before its
runs:

- the two implementations: the reference's source, and the uv project
  pinning its libraries, `envs/<library>/` with its lockfile; the
  vendored purejaxql is pinned to a commit;
- the JAX environment, and how it differs from the original task;
- every setting of the reference, and the Ajax keyword holding each one
  (`renames`, `scale`); what no Ajax option can match is listed under
  `unmatched`;
- the budget, the evaluation (the deterministic policy plays
  `n_eval_episodes` episodes every `eval_every` steps) and the score, the
  mean return of the last evaluation;
- AdaStop's `n` (seeds per side and interim: 4), `K` (the most interims:
  24, so at most 96 seeds per side) and `alpha` (the family-wise error:
  0.05);
- the margin within which the two sides count as similar.

Two Ajax arms can be compared: `matched`, set to the reference's
conventions, and `defaults`, Ajax as shipped. The protocol's verdicts are
on the matched arm.

A spec is not edited once results exist: a change becomes a new
`version` of the comparison. Every record keeps the version and the hash
of the spec it ran, and the tests check that the recorded results of the
current version ran the current spec, from a clean tree.

## Running

```bash
uv run python benchmarks/reference/refcmp_protocol.py run PQN
```

With `--cache DIR` and an empty directory, the same command trains every
seed again instead of reusing the recorded ones: a reproduction from
scratch, whose report (`report PQN --cache DIR`) can be set beside the
published one.

At each interim, every side still in an undecided comparison trains `n`
new seeds: Ajax and purejaxql in one program, Stable-Baselines3 and Brax
one process per seed, all on the CPU, each in its own uv project. Every
batch is appended to `benchmarks/reference/results/protocol/<comparison>.jsonl`
with:

- the per-seed evaluation curves;
- its speed: wall, CPU and compile seconds, environment steps per second,
  and the machine;
- a manifest: the protocol's version, the spec's version and hash, the
  runner script, Ajax's commit and whether the tree was clean, the
  lockfile's hash, the libraries' and Python's versions, the platform, the
  seeds and the resolved settings.

A later run reuses each recorded seed while its fingerprint, the hash of
its settings and of the code it ran, is unchanged: a reference trains once,
and an Ajax arm again after a change to Ajax.

## Reading a verdict

Each comparison registers a margin with its spec: how far apart the two
mean scores may be and still count as similar. It is 5% of the score's
range (±25 on CartPole's 0 to 500), or 5% of the reference's mean score
when the score has no fixed range. The verdict sets the 95% bootstrap
interval of the gap, Ajax's mean score minus the reference's, against it:

- **similar**: the interval lies inside the margin, whether or not
  AdaStop detected a difference;
- **different**: the interval lies entirely outside it;
- **inconclusive**: anything else.

AdaStop decides when to stop. It stops as soon as it detects a difference
(a clear one ends a comparison after a few seeds); otherwise it goes on to
its cap of `n * K` seeds per side. A small cap can therefore only make a
verdict inconclusive, never similar. Next to the verdict stand AdaStop's
own decision and its resolution: the smallest gap its budget detects 80%
of the time, simulated from the observed spread of the scores
(`ajax.compare.resolution`). With RL's spread from seed to seed, a margin
of 5% is about the tightest that 96 seeds per side can confirm: on CartPole
with a few collapsed seeds, the interval at 96 seeds is about ±19.

## Tracing a number

Every published number leads back to what produced it. A row of
`runs.parquet` names its comparison, side, seed and batch; the batch's
record in `batches.jsonl` gives the spec's version and hash, the runner
script, Ajax's commit (whose `refcmp_specs.py` holds the spec, and whose
`envs/<library>/uv.lock` pins the reference), the lockfile's hash and the
resolved settings. The verdicts in `summary.jsonl`, and the results page
drawn from the same records, are computed from those rows.

## The data

`refcmp_publish.py dataset` writes every record as a Hugging Face dataset:
`runs.parquet` holds one row per seed and evaluation, `batches.jsonl` each
batch's speed and manifest, `summary.jsonl` the verdicts and `specs.json`
the comparisons' settings. With the `reproduction` extra:

```python
from ajax.reproduction import load

runs = load("runs", "<dataset repository>")
```
