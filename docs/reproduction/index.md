# Reproduction

Does each Ajax agent learn like the implementation it reproduces? Each
comparison trains Ajax and a reference implementation (Stable-Baselines3,
Brax, or the authors' code) on the same JAX environment and budget, and
AdaStop decides how many seeds the answer takes.

- [Protocol](protocol.md): how comparisons are set up, run and judged,
  and what each run records.
- [Specifications](specs.md): every comparison's settings, written before
  its runs.
- [Results](results.md): the verdicts, speeds and learning curves.

The harness is [`benchmarks/reference`](../../benchmarks/reference/README.md),
run on demand, not in CI.

The runs are published as a Hugging Face dataset, which
`ajax.reproduction.load` reads (the `reproduction` extra).
