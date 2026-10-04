# Contains code adapted from danijar/dreamerv3 at commit 29eb964
# (29eb964e2918a3f4db04086f7f51b60388e97f3d): ``mean`` below replaces
# TwoHotDist.mean of dreamerv3/jaxutils.py:226-250, keeping its attribute
# names and odd-n set-up (lines 233-235).
# Modified: the expectation is Ajax's exact mirror-difference form instead of
# the reference's mirror-pair sum (deviation D22).
#
# Copyright (c) 2023 Danijar Hafner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The license text is also in LICENSE.dreamerv3 in this directory.
"""run_reference.py with one patch: the reference's TwoHotDist.mean replaced by
Ajax's exact mirror-difference form (deviation D22), sum_j (p_j - p_mirror(j)) b_j
over the upper half. Equal in exact arithmetic (the reference's bins are exactly
antisymmetric with b_m = 0, nets.py:466-470); removes the FMA-contraction noise
of the literal pair sum under jit (DIAGNOSIS.md in this directory). The
reference checkout is not edited: the method is replaced at import time.
Same CLI as run_reference.py; writes PATCH.txt into --out."""

# ruff: noqa: I001
# Import order matters and is kept as in the runs: jaxutils (jax, tfp and
# dreamerv3.main, with its sys.path inserts) is imported before run_reference,
# whose module body sets T_START, so build_and_compile_s and wall_s exclude
# those imports, as in the committed timing.json.
import pathlib
import sys

from dreamerv3 import jaxutils

import run_reference


def mean(self):
    n = self.logits.shape[-1]
    assert n % 2 == 1, n
    m = (n - 1) // 2
    upper = self.bins[..., m + 1 :]
    diff = self.probs[..., m + 1 :] - self.probs[..., :m][..., ::-1]
    return self.transbwd((diff * upper).sum(-1))


jaxutils.TwoHotDist.mean = mean

if __name__ == "__main__":
    code = run_reference.main(sys.argv[1:])
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
    (out / "PATCH.txt").write_text(
        "TwoHotDist.mean -> exact mirror-difference form (run_reference_exact.py)\n"
    )
    sys.exit(code)
