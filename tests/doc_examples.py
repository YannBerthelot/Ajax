"""The README's and the documentation's Python examples, run as a reader
runs them: the one place that says which blocks run, and how.

A ``python`` block runs unless its first line is ``# doc: skip``. Each README
block runs on its own; a docs page's blocks run in order in one namespace.
docs/world_models/ quotes the reference implementations' code: not examples.
tests/test_readme.py and tests/test_docs_examples.py take their cases from
here. As a script, the Release workflow's installed-package gate runs it
against the built wheel; it imports only the standard library:

    python tests/doc_examples.py [README.md] [docs]

Each example runs in a temporary directory; a failure makes the exit status
non-zero, and the failing pages and blocks are named.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import re
import sys
import tempfile
import traceback
import types
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
SKIP = "# doc: skip"


def runnable(code: str) -> bool:
    return not code.lstrip().startswith(SKIP)


def runnable_blocks(page: pathlib.Path) -> list[str]:
    return [code for code in BLOCK.findall(page.read_text()) if runnable(code)]


def readme_blocks(readme: pathlib.Path) -> list[tuple[int, str]]:
    """Each runnable block, with its place among all the README's blocks:
    its test is ``block<place>``."""
    blocks = enumerate(BLOCK.findall(readme.read_text()))
    return [(place, code) for place, code in blocks if runnable(code)]


def example_pages(docs: pathlib.Path) -> list[pathlib.Path]:
    return [
        page
        for page in sorted(docs.rglob("*.md"))
        if "world_models" not in page.relative_to(docs).parts and runnable_blocks(page)
    ]


def run_readme_block(code: str, readme: pathlib.Path) -> None:
    exec(compile(code, str(readme), "exec"), {"__name__": "__readme__"})


def run_page(page: pathlib.Path) -> None:
    """The page's blocks in order, in a module of their own, as a script
    is: a dataclass finds its module's names (string annotations) through
    sys.modules."""
    module = types.ModuleType("__docs__")
    sys.modules[module.__name__] = module
    try:
        for index, code in enumerate(runnable_blocks(page)):
            exec(compile(code, f"{page}, block {index}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)


def examples(source: pathlib.Path) -> list[tuple[str, Callable[[], None]]]:
    """A docs directory's pages, or a README's blocks, each named."""
    if source.is_dir():
        return [
            (
                page.relative_to(source.parent).as_posix(),
                functools.partial(run_page, page),
            )
            for page in example_pages(source)
        ]
    return [
        (
            f"{source.name} block{place}",
            functools.partial(run_readme_block, code, source),
        )
        for place, code in readme_blocks(source)
    ]


def main(sources: list[pathlib.Path]) -> int:
    spec = importlib.util.find_spec("ajax")
    print(f"ajax from {spec.origin if spec else None}")
    home = os.getcwd()
    failed = []
    for source in sources:
        for name, run in examples(source.resolve()):
            print(f"running {name}", flush=True)
            # A scratch directory, so nothing an example writes lands in the checkout.
            with tempfile.TemporaryDirectory() as scratch:
                os.chdir(scratch)
                try:
                    run()
                except Exception:
                    traceback.print_exc()
                    failed.append(name)
                finally:
                    os.chdir(home)
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    # A script's own directory comes first on sys.path: here tests/, whose
    # tests/logging package would hide the standard library's logging,
    # which JAX imports.
    here = pathlib.Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if pathlib.Path(entry).resolve() != here]
    arguments = [pathlib.Path(argument) for argument in sys.argv[1:]]
    sys.exit(main(arguments or [ROOT / "README.md", ROOT / "docs"]))
