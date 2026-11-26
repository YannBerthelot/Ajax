"""The scripts of examples/ and the notebook of notebooks/ run.

Contract: every script of examples/ has a ``main`` taking ``n_timesteps``
and runs to its end; the notebook's code cells run in order, as a reader
runs them (its shell and ``%`` lines aside: they install the package).
"Runs without raising" is the contract of an example; what the calls return
is tested where the calls are. They train, so they run in the slow tier, as
the documentation's pages that train do (tests/test_docs_examples.py); the
scripts are given few steps, the notebook runs word for word.
"""

from __future__ import annotations

import importlib.util
import json

import matplotlib
import pytest

from tests.doc_examples import ROOT

EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
# Enough steps for each script to reach its last line: one evaluation, and
# an update for SAC after its 1,000 warm-up steps.
STEPS = {
    "train_ppo_cartpole": 4_096,
    "sac_pendulum_tensorboard": 1_200,
    "save_and_act": 1_024,
    "hyperparameter_sweep": 1_200,
}


def test_every_example_is_listed_with_its_steps_and_in_the_readme():
    names = [script.stem for script in EXAMPLES]
    assert sorted(STEPS) == names
    readme = (ROOT / "examples" / "README.md").read_text()
    assert [name for name in names if f"`{name}.py`" not in readme] == []


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda script: script.stem)
def test_an_example_runs(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # what it writes stays out of the checkout
    spec = importlib.util.spec_from_file_location(f"example_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(n_timesteps=STEPS[script.stem])


def code_cells(notebook) -> list[str]:
    """The notebook's code cells, without their shell and magic lines."""
    cells = json.loads(notebook.read_text())["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    return [
        "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        for source in sources
    ]


@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda notebook: notebook.stem)
def test_a_notebook_s_cells_run_in_order(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # plt.show() must open no window, which would wait for it to close;
    # forced, since an earlier test may have imported matplotlib already.
    matplotlib.use("Agg", force=True)
    namespace: dict = {"__name__": "__notebook__"}
    for index, code in enumerate(code_cells(notebook)):
        exec(compile(code, f"{notebook.name}, cell {index}", "exec"), namespace)
