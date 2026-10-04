"""run_ajax.py with one patch: ajax TwoHot.decode replaced by the reference's
literal two-hot expectation (29eb964 jaxutils.py:226-243, mirror-pair sum, bins
symexp(linspace(-20, 0, 128)) built in float32 under jit; deviation D22), from
refdecode.py in this directory. The Ajax tree is not edited: the
method is replaced at import time. Same CLI as run_ajax.py; writes PATCH.txt
into --out."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import refdecode

refdecode.install()

import run_ajax  # noqa: E402

if __name__ == "__main__":
    code = run_ajax.main(sys.argv[1:])
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
    (out / "PATCH.txt").write_text(
        "TwoHot.decode -> reference literal pair sum (run_ajax_refdecode.py)\n"
    )
    sys.exit(code)
