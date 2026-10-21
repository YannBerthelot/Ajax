# Installation

Ajax installs from source with [uv](https://docs.astral.sh/uv/). It needs
Python 3.12 or newer.

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
uv sync
```

`uv sync` creates `.venv` with the versions in `uv.lock`, which CI tests.
Prefix commands with `uv run`, or activate the environment with
`source .venv/bin/activate`. To check the install:

```bash
uv run python -c "import ajax; print(ajax.__version__)"
```

## With pip

pip installs Ajax from its repository:

```bash
pip install "ajax @ git+https://github.com/YannBerthelot/Ajax.git"
```

pip does not read `uv.lock`, so it takes the newest versions that
`pyproject.toml` allows. A weekly CI job tests those too, and another the
oldest versions allowed
([Dependencies and releases](../developer/contributing.md#dependencies-and-releases)).

## CPU or GPU

Ajax installs JAX for the CPU. CUDA comes only from its `cuda` extra (Linux
only); without it, JAX runs on the CPU even on a machine with an NVIDIA GPU.
To use the GPU:

```bash
uv sync --extra cuda
```

A project that depends on Ajax asks for `ajax[cuda]` instead.

To keep a run on the CPU on a machine with a GPU, set `JAX_PLATFORMS=cpu`.
The repository's `make` targets go further and hide the GPU from CUDA as well,
so tests never compete with an experiment running on it:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python my_script.py
```

[Performance](../guide/performance.md) says when each device pays off.

## Dependency groups

`uv sync` installs the runtime and two groups by default. The others are
opt-in.

| Group | Default | What it adds |
| --- | --- | --- |
| `dev` | yes | pytest, coverage, pre-commit, ruff, mypy, the probing environments the tests train on, and the `playground` extra |
| `wandb` | yes | the `wandb` extra (below) |
| `plane` | no | [TargetGym](https://github.com/YannBerthelot/TargetGym), for the Plane3DCircle scenario of the speed benchmark (`benchmarks/perf_bench.py`) |
| `docs` | no | MkDocs and its plugins, to build this site |

```bash
uv sync --group plane         # add an opt-in group
uv sync --no-default-groups   # the runtime alone
```

The `docs` group builds the site without the runtime
(`uv sync --only-group docs`): the API pages are read from the source, not
imported.

## Extras

| Extra | What it adds |
| --- | --- |
| `cuda` | JAX with CUDA, on Linux ([above](#cpu-or-gpu)) |
| `brax` | [Brax](https://github.com/google/brax), for Brax environments such as `"ant"` |
| `playground` | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and Brax, for Playground environments such as `"CartpoleBalance"` |
| `compare` | the `adastop` package, whose confidence sequence [`ajax.compare.estimate`](../guide/comparing-agents.md#one-agent-how-sure-is-a-number) uses; AdaStop itself needs no extra |
| `reproduction` | Hugging Face Hub, pandas and pyarrow, for `ajax.reproduction.load` ([Reproduction](../reproduction/index.md)) |
| `wandb` | Weights & Biases, for [logging](../guide/logging.md) with `use_wandb=True`; the default `wandb` group installs it |

```bash
uv sync --extra compare
```

Brax and MuJoCo Playground come only with their extras: a project that
uses their environments asks for `ajax[playground]`, or `ajax[brax]` for
Brax alone. Without them, building a Brax or Playground environment raises
an error that names the extra. The `dev` group installs both, so `uv sync`
in this repository has them.

The `compare` extra takes `adastop` from a commit of its main branch, named
in `[tool.uv.sources]`, until a release includes the confidence sequence.
pip does not read that table: with pip, install `adastop` from that commit
yourself.

## Using Ajax from another project

Every runtime dependency of Ajax, gymnax 1.0.1 included, comes from PyPI,
so a project that depends on Ajax declares only where Ajax comes from:

```toml
[project]
dependencies = ["ajax"]  # "ajax[cuda]" for CUDA, "ajax[playground]" for Brax and Playground envs

[tool.uv.sources]
ajax = { git = "https://github.com/YannBerthelot/Ajax.git" }
```

uv resolves that project's own versions within Ajax's bounds, not those of
Ajax's `uv.lock`.

## The compile cache

`import ajax` turns on JAX's persistent compile cache, in
`~/.cache/ajax/jax_compile_cache`. A second process that runs the same
program loads it from disk instead of compiling it again. Two environment
variables control it:

| Variable | Effect |
| --- | --- |
| `AJAX_JAX_COMPILE_CACHE_DIR` | puts the cache somewhere else |
| `AJAX_NO_COMPILE_CACHE=1` | turns the cache off and leaves the disk untouched |

[Performance](../guide/performance.md#the-compile-cache) explains what it
saves and when it does not apply.

## Weights & Biases

A [`LoggingConfig`](../guide/logging.md) logs to Weights & Biases unless you
pass `use_wandb=False`. Log in once with `uv run wandb login` before the first
run that uses it. Runs that keep their results in memory, as in the
[quickstart](quickstart.md), need no account.

## Working on Ajax

After `uv sync`, run `uv run pre-commit install` once, so ruff and mypy check
every commit. `make docs` builds this site as CI does
(`uv run --group docs mkdocs build --strict`: a broken link or a docstring
naming a parameter the code does not have fails it), and `make docs-serve`
serves it at `http://127.0.0.1:8000`, rebuilt on save.
[Contributing](../developer/contributing.md) covers the rest.
