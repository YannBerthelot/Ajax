"""Shared by the paper-protocol smoke tests: run a ``--smoke`` entry of
``paper_protocol.py`` killed after its first chunk's save then resumed from
disk by the CLI, and (DreamerV3) uninterrupted through the CLI too."""

from __future__ import annotations

import os

import paper_protocol
import pytest
from wm_runs import CURVE_FILE, STATE_FILE, read_records, run_single_task


class Interrupted(Exception):
    """Stands for a killed process."""


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


def run_twice(name: str, tmp_path) -> tuple[list[dict], list[dict]]:
    """The curves of the uninterrupted run and of :func:`run_resumed`."""
    whole = str(tmp_path / "whole")
    assert paper_protocol.main(["--smoke", "--only", name, "--out", whole]) == 0
    assert not os.path.exists(os.path.join(whole, name, STATE_FILE))
    return records(os.path.join(whole, name)), run_resumed(name, tmp_path)
