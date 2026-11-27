# Installation

This page installs Ajax, the extras some environments and tools need, and
CUDA for a GPU. Ajax is on PyPI as `ajax-rl` and imports as `ajax`. It needs
Python 3.12 or newer:

```bash
pip install ajax-rl
```

That is enough for the [quickstart](quickstart.md): gymnax's environments,
such as `"CartPole-v1"`, come with Ajax.

## Extras

Other environments and some tools come with an extra, such as
`pip install "ajax-rl[playground]"`. Without it, building such an
environment raises an error that names the extra.

| Extra | What it adds |
| --- | --- |
| `cuda` | JAX with CUDA, on Linux ([below](#cpu-or-gpu)) |
| `brax` | [Brax](https://github.com/google/brax), for Brax environments such as `"ant"` |
| `playground` | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and Brax, for Playground environments such as `"CartpoleBalance"` |
| `gymnasium` | [Gymnasium](https://gymnasium.farama.org) with MuJoCo, for [Gymnasium environments](../guide/environments.md#gymnasium-environments) such as `"gymnasium:HalfCheetah-v5"`, which step in Python |
| `gymnasium-mjx` | [gymnasium-mjx](https://github.com/YannBerthelot/gymnasium-mjx) and the `playground` extra, for [Gymnasium's MuJoCo tasks in JAX](../guide/environments.md#gymnasiums-mujoco-tasks-in-jax) such as `"mujoco:HalfCheetah-v5"` |
| `render` | matplotlib and Gymnasium's classic-control drawings (pygame), for [`agent.render`](../guide/deploy.md#watch-the-policy) on gymnax tasks |
| `compare` | the `adastop` package, whose confidence sequence [`ajax.compare.estimate`](../guide/comparing-agents.md#one-agent-how-sure-is-a-number) uses; AdaStop itself needs no extra |
| `reproduction` | Hugging Face Hub, pandas and pyarrow, for `ajax.reproduction.load` ([Reproduction](../reproduction/index.md)) |
| `manifolds` | umap-learn, M-PHATE and matplotlib, for embedding a recorded run ([Representations during training](../guide/representations.md#embedding-a-run)) |
| `wandb` | Weights & Biases, for [logging](../guide/logging.md#backends) with `WandbBackend` |

Until an adastop release includes the confidence sequence, a checkout's
`uv sync` takes adastop from the commit that
[`pyproject.toml`](../../pyproject.toml) names in `[tool.uv.sources]`. pip
does not read that table: with pip, install adastop from that commit
yourself.

## CPU or GPU

Ajax installs JAX for the CPU: without the `cuda` extra, JAX runs on the CPU
even on a machine with an NVIDIA GPU. On Linux, the extra adds CUDA:

```bash
pip install "ajax-rl[cuda]"   # or, in a checkout: uv sync --extra cuda
```

To keep a run on the CPU of a machine with a GPU, set `JAX_PLATFORMS=cpu`;
`CUDA_VISIBLE_DEVICES=""` hides the GPU from CUDA as well:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python my_script.py
```

[Performance](../guide/performance.md#cpu-or-gpu) says when a GPU pays off.

## From source

pip installs main, the development version, from the repository:

```bash
pip install "ajax-rl @ git+https://github.com/YannBerthelot/Ajax.git"
```

pip takes the newest versions of the dependencies that `pyproject.toml`
allows; a weekly CI job tests those, and another the oldest allowed. For a
checkout, [uv](https://docs.astral.sh/uv/) installs the versions in
`uv.lock`, which CI tests:

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
uv sync
uv run python -c "import ajax; print(ajax.__version__)"
```

`uv sync` creates `.venv`: prefix commands with `uv run`, or activate it
with `source .venv/bin/activate`. It also installs the `dev` group, with
the test tools and the extras the tests need;
[Contributing](../developer/contributing.md#development-setup) lists the
groups and the checks a change runs.

## Using Ajax from another project

A project that depends on Ajax names `ajax-rl` with the extras it needs:

```toml
[project]
dependencies = ["ajax-rl"]  # "ajax-rl[cuda]" for CUDA, "ajax-rl[playground]" for Brax and Playground envs

[tool.uv.sources]
ajax-rl = { git = "https://github.com/YannBerthelot/Ajax.git" }  # optional: main instead of a release
```

Its installer picks versions within Ajax's bounds; Ajax's `uv.lock` applies
only to a checkout of Ajax. A project that declared `ajax` from git or a
path renames it `ajax-rl`; the import stays `import ajax`.

## The compile cache

`import ajax` turns on JAX's persistent compile cache, in
`~/.cache/ajax/jax_compile_cache`: a second process that runs the same
program loads it from disk instead of compiling it again.

| Variable | Effect |
| --- | --- |
| `AJAX_JAX_COMPILE_CACHE_DIR` | puts the cache somewhere else |
| `AJAX_NO_COMPILE_CACHE=1` | turns the cache off and leaves the disk untouched |

[Performance](../guide/performance.md#the-compile-cache) explains what it
saves and when it does not apply.

## Weights & Biases

Logging to Weights & Biases needs the `wandb` extra, a `WandbBackend` in the
[`LoggingConfig`](../guide/logging.md#backends), and a login once, with
`wandb login`, before the first run. A run that keeps its evaluations in
memory, as in the quickstart, or writes TensorBoard files needs no account.

## Next

The [quickstart](quickstart.md) trains a first agent, reads its results,
acts with it and saves it.
