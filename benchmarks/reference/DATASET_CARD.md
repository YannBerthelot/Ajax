---
license: mit
pretty_name: Ajax reference comparisons
configs:
  - config_name: runs
    data_files: runs.parquet
  - config_name: batches
    data_files: batches.jsonl
  - config_name: summary
    data_files: summary.jsonl
---

# Ajax reference comparisons

Training runs of [Ajax](https://github.com/YannBerthelot/Ajax)'s agents and
of reference implementations of the same algorithms (Stable-Baselines3,
Brax, purejaxql), trained by Ajax's reproduction protocol on the same JAX
environments, budgets and evaluations. AdaStop decides how many seeds each
side trains. The protocol, the settings of every comparison and the
commands that reproduce these runs are described in
[Ajax's documentation](https://yannberthelot.github.io/Ajax/reproduction/).

## Files

- `runs.parquet`: one row per seed and evaluation: `comparison`, `side`
  ("reference" or an Ajax arm), `implementation`, `fingerprint` (the hash
  of the settings and code a run used), `batch`, `seed`, `step`, `return`
  (the mean return of the evaluation's episodes).
- `batches.jsonl`: one record per batch of seeds trained together: its
  speed (wall, CPU and compile seconds, environment steps per second) and
  manifest (Ajax commit, lockfile hash, library versions, platform,
  hardware, resolved settings).
- `summary.jsonl`: per comparison, AdaStop's decision, the seeds it took,
  the difference in mean final return with its 95% interval, the smallest
  gap the budget resolves, and each side's speed.
- `specs.json`: each comparison's settings, as `refcmp_specs.py` holds them.

## Loading

```python
from ajax.reproduction import load  # pip install "ajax[reproduction]"

runs = load("runs", "<this repository>")
```

or with `pandas.read_parquet` on `runs.parquet`.
