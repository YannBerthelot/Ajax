"""The validation scripts of ``benchmarks/world_models/`` (M9) import each
other as sibling modules (they run as scripts); their directory goes on
``sys.path`` for these tests."""

import os
import sys

SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "benchmarks",
    "world_models",
)
sys.path.insert(0, os.path.normpath(SCRIPTS))
