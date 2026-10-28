"""The fake wandb module, for the tests of what runs send to W&B."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from tests.logging import fake_wandb


@pytest.fixture
def wandb_calls(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """The file fake_wandb records into, in the caller and in the worker."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "wandb.py").write_text("from tests.logging.fake_wandb import *  # noqa\n")
    record = str(tmp_path / "wandb_calls.jsonl")
    monkeypatch.setenv(fake_wandb.LOG_ENV, record)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.syspath_prepend(str(shim))
    return record
