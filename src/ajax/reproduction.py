"""The reference comparisons' published data (docs/reproduction/index.md).

Every run of the reproduction protocol, Ajax's and the reference
implementations', as tables: ``runs`` has one row per seed and evaluation,
``batches`` each batch's speed and manifest, ``summary`` each comparison's
verdict. Needs the ``reproduction`` extra (``pip install "ajax[reproduction]"``).
"""

import pathlib
from typing import Any, Optional

TABLES = ("runs", "batches", "summary")


def load(table: str, source: str, revision: Optional[str] = None) -> Any:
    """One table as a pandas ``DataFrame``.

    Args:
        table: "runs", "batches" or "summary".
        source: a Hugging Face dataset repository, or a local directory
            holding one (``refcmp_publish.py dataset`` writes it).
        revision: the dataset's commit, branch or tag; the latest when None.
    """
    import pandas as pd

    if table not in TABLES:
        raise ValueError(f"table is one of {TABLES}, not {table!r}")
    path = pathlib.Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = pathlib.Path(
            snapshot_download(source, repo_type="dataset", revision=revision)
        )
    if table == "runs":
        return pd.read_parquet(path / "runs.parquet")
    return pd.read_json(path / f"{table}.jsonl", lines=True)
