# Installation

Ajax installs from source with [uv](https://docs.astral.sh/uv/). It needs
Python 3.11, 3.12 or 3.13.

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
uv sync
```

`uv sync` creates `.venv` with the locked dependencies. Prefix commands with
`uv run`, or activate the environment with `source .venv/bin/activate`. To
check the install:

```bash
uv run python -c "import ajax; print(ajax.__version__)"
```

## CPU or GPU

On Apple-silicon Macs, Ajax installs JAX for the CPU. Everywhere else it
installs `jax[cuda]`, which runs on an NVIDIA GPU when there is one and
falls back to the CPU when there is not.

To keep a run on the CPU on a machine with a GPU, set `JAX_PLATFORMS=cpu`.
The repository's `make` targets go further and hide the GPU from CUDA as well,
so tests never compete with an experiment running on it:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python my_script.py
```

[Performance](../guide/performance.md) says when each device pays off.

## Dependency groups

`uv sync` installs the runtime and three groups by default. The others are
opt-in.

| Group | Default | What it adds |
| --- | --- | --- |
| `dev` | yes | pytest, coverage, pre-commit, ruff, mypy, and the probing environments the tests train on |
| `render` | yes | `imageio[ffmpeg]`, to write videos |
| `wandb` | yes | Weights & Biases, for [logging](../guide/logging.md) |
| `plane` | no | [TargetGym](https://github.com/YannBerthelot/TargetGym), for the Plane3DCircle benchmark scenarios |
| `docs` | no | MkDocs and its plugins, to build this site |

```bash
uv sync --group plane         # add an opt-in group
uv sync --no-default-groups   # the runtime alone
```

## Using Ajax from another project

Ajax needs gymnax 1.0 (its step returns `terminated` and `truncated`
apart). The gymnax 1.0.0 release caps JAX below 0.7 and gymnasium below 1.2,
which no resolver can satisfy next to distrax, so Ajax's `pyproject.toml`
takes gymnax from a fork whose only change lifts those two caps. uv applies
`[tool.uv.sources]` from the project you sync, not from its dependencies, so
a project that depends on Ajax declares the same source:

```toml
[project]
dependencies = ["ajax"]

[tool.uv.sources]
ajax = { git = "https://github.com/YannBerthelot/Ajax.git" }
gymnax = { git = "https://github.com/YannBerthelot/gymnax.git", branch = "chore/relax-runtime-pins" }
```

With pip, install that gymnax branch first, then Ajax from its repository.

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
every commit. `make docs-serve` serves this site at `http://127.0.0.1:8000`
and rebuilds it on save. [Contributing](../developer/contributing.md) covers
the rest.
