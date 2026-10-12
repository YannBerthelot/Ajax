"""Shared by the validation-script tests: a stand-in for a killed process,
a run directory written as the scripts write theirs, and a ``--smoke`` entry
of ``paper_protocol.py`` killed after its first chunk's save then resumed
from disk by the CLI."""

from __future__ import annotations

import json
import os

import paper_protocol
import pytest
from wm_runs import (
    CURVE_FILE,
    RUN_FILE,
    STATE_FILE,
    jsonable,
    read_records,
    run_single_task,
)


class Interrupted(Exception):
    """Stands for a killed process."""


def write_run_dir(directory, spec: dict, records) -> None:
    """A run directory: ``run.json`` holding ``spec`` and one
    ``curve.jsonl`` line per record."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, RUN_FILE), "w") as f:
        json.dump({"spec": jsonable(spec)}, f)
    with open(os.path.join(directory, CURVE_FILE), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def records(run_dir: str) -> list[dict]:
    return read_records(os.path.join(run_dir, CURVE_FILE))


def run_resumed(name: str, tmp_path) -> list[dict]:
    """Kill the run after its first chunk's save, resume it with the CLI;
    its curve."""
    spec = paper_protocol.smoke_runs()[name]
    split = str(tmp_path / "split")
    first = f": {spec.chunk}/{spec.budget}"

    def kill_after_first_chunk(message: str) -> None:
        if first in message:
            raise Interrupted

    with pytest.raises(Interrupted):
        run_single_task(
            spec, os.path.join(split, name), save_every_s=0, log=kill_after_first_chunk
        )
    assert os.path.isfile(os.path.join(split, name, STATE_FILE))
    assert len(records(os.path.join(split, name))) == 1
    assert paper_protocol.main(["--smoke", "--only", name, "--out", split]) == 0
    assert not os.path.exists(os.path.join(split, name, STATE_FILE))
    return records(os.path.join(split, name))
