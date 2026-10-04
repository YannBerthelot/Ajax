# Contains code adapted from danijar/dreamerv3 at commit 29eb964
# (29eb964e2918a3f4db04086f7f51b60388e97f3d): ``mean`` below replaces
# TwoHotDist.mean of dreamerv3/jaxutils.py:226-250, keeping its attribute
# names and odd-n set-up (lines 233-235).
# Modified: the expectation is Ajax's exact mirror-difference form instead of
# the reference's mirror-pair sum (deviation D22).
# replay_init wraps embodied.replay.Replay.__init__ (embodied/replay/replay.py)
# to pass a per-run sampler seed; the reference code itself is not edited.
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
"""Round 3 reference run: run_reference.py with two patches, both matching Ajax's
documented choices (docs/world_models/deviations.md), applied at import time
without editing the reference checkout:

1. D22: TwoHotDist.mean -> exact mirror-difference form (as run_reference_exact.py).
2. D2: the replay sampler is seeded per run. Stock make_replay (main.py:160-187)
   never passes a seed, so Replay(seed=0) -> selectors.Uniform(0) samples every
   run with the same index stream (replay.py:20,27; selectors.py:29-32). Here
   Replay's default seed becomes [run seed, 0x5EED] (a stream distinct from the
   agent's numpy Generator(seed), jaxagent.py:39).

Same CLI as run_reference.py; writes PATCH.txt into --out."""

# ruff: noqa: I001
# Import order matters and is kept as in the runs: jaxutils (jax, tfp and
# dreamerv3.main, with its sys.path inserts) is imported before run_reference,
# whose module body sets T_START, so build_and_compile_s and wall_s exclude
# those imports, as in the committed timing.json.
import pathlib
import sys

from dreamerv3 import jaxutils

import embodied
import run_reference

SEED = int(sys.argv[sys.argv.index("--seed") + 1])
SAMPLER_SEED = [SEED, 0x5EED]


def mean(self):
    n = self.logits.shape[-1]
    assert n % 2 == 1, n
    m = (n - 1) // 2
    upper = self.bins[..., m + 1 :]
    diff = self.probs[..., m + 1 :] - self.probs[..., :m][..., ::-1]
    return self.transbwd((diff * upper).sum(-1))


jaxutils.TwoHotDist.mean = mean

_replay_init = embodied.replay.Replay.__init__
REPLAYS = []


def replay_init(self, *args, seed=0, **kwargs):
    assert seed == 0 and "selector" not in kwargs, (seed, kwargs.keys())
    _replay_init(self, *args, seed=SAMPLER_SEED, **kwargs)
    REPLAYS.append(self)


embodied.replay.Replay.__init__ = replay_init

if __name__ == "__main__":
    code = run_reference.main(sys.argv[1:])
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
    assert len(REPLAYS) == 1, len(REPLAYS)
    (out / "PATCH.txt").write_text(
        "TwoHotDist.mean -> exact mirror-difference form; "
        f"replay sampler seed {SAMPLER_SEED} (run_reference_round3.py)\n"
    )
    sys.exit(code)
