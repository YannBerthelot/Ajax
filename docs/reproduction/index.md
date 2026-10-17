# Reproduction

Does each Ajax agent learn like the implementation it reproduces? The
reproduction protocol trains an Ajax agent and a reference implementation
of the same algorithm on the same environment, budget and evaluation, and
lets AdaStop (see [Comparing agents](../guide/comparing-agents.md)) decide
how many seeds the answer takes. Everything is re-run: no number is taken
from a paper. The harness lives in
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

## What a comparison records

Each comparison in
[`refcmp_specs.py`](../../benchmarks/reference/refcmp_specs.py) is set
before its runs:

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
- AdaStop's `n` (seeds per side and interim), `K` (the most interims) and
  `alpha` (the family-wise error).

Two Ajax arms can be compared: `matched`, set to the reference's
conventions, and `defaults`, Ajax as shipped. The protocol's verdicts are
on the matched arm.

## Running

```bash
uv run python benchmarks/reference/refcmp_protocol.py run PQN
```

At each interim, every side still in an undecided comparison trains `n`
new seeds: Ajax and purejaxql in one program, Stable-Baselines3 and Brax
one process per seed, all on the CPU, each in its own uv project. Every
batch is appended to `benchmarks/reference/results/protocol/<comparison>.jsonl`
with:

- the per-seed evaluation curves;
- its speed: wall, CPU and compile seconds, environment steps per second,
  and the machine;
- a manifest: Ajax's commit and whether the tree was clean, the lockfile's
  hash, the libraries' and Python's versions, the platform, the seeds and
  the resolved settings.

A later run reuses each recorded seed while its fingerprint, the hash of
its settings and of the code it ran, is unchanged: a reference trains once,
and an Ajax arm again after a change to Ajax.

## Reading a verdict

AdaStop says "Ajax larger", "Ajax smaller", or, when no difference was
shown after `n * K` seeds, "equal". Two numbers stand next to it:

- the difference between the mean scores, Ajax's minus the reference's,
  with its 95% bootstrap interval;
- the resolution: the smallest gap AdaStop's budget detects 80% of the
  time, simulated from the observed spread of the scores
  (`ajax.compare.resolution`).

An "equal" rules out gaps well above the resolution, not smaller ones.

## The data

`refcmp_publish.py dataset` writes every record as a Hugging Face dataset:
`runs.parquet` holds one row per seed and evaluation, `batches.jsonl` each
batch's speed and manifest, `summary.jsonl` the verdicts and `specs.json`
the comparisons' settings. With the `reproduction` extra:

```python
from ajax.reproduction import load

runs = load("runs", "<dataset repository>")
```
