"""The command line every runner shares: train seeds, print one JSON line.

usage: python <runner> NAME SIDE FIRST_SEED N_SEEDS

NAME is a comparison of refcmp_specs and SIDE "reference" or an Ajax arm.
The line holds the per-seed evaluation curves, the resolved settings, the
libraries' versions and the seconds JAX spent tracing and compiling;
refcmp_protocol.py adds the rest of the record. Imports nothing from Ajax.
"""

import importlib.metadata
import inspect
import json
import pathlib
import platform
import sys

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from agent_bench import compile_spans, covered_seconds
from refcmp_specs import COMPARISONS


def resolved(fn, kwargs: dict) -> dict:
    """``fn``'s keyword defaults that are plain values, under ``kwargs``."""
    plain = (bool, int, float, str, type(None))
    params = inspect.signature(fn).parameters.items()
    defaults = {k: p.default for k, p in params if isinstance(p.default, plain)}
    defaults.update(kwargs)
    return defaults


def main(run, config, libraries) -> None:
    """``run(spec, side, seeds)`` gives the curves, ``config(spec, side)``
    the settings; ``libraries`` are the distributions whose versions count."""
    name, side, first, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    spec, seeds = COMPARISONS[name], list(range(first, first + n))
    spans = compile_spans()
    curve = run(spec, side, seeds)
    versions = {lib: importlib.metadata.version(lib) for lib in libraries}
    record = {
        "curve": [[round(float(x), 4) for x in row] for row in curve],
        "config": config(spec, side),
        "versions": {**versions, "python": platform.python_version()},
        "compile_s": round(covered_seconds(spans), 2),
    }
    print(json.dumps(record, default=str))
