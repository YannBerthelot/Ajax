"""The PyTorch originals' records (benchmarks/reference/conformance_*.py), and Ajax's update compared
with them: what their conformance tests share.

A recorder writes, per recorded state, ``in.*`` (the state), the update's own draws, and the state
after one update from it in float32 and in float64 (``f32.*``, ``f64.*``), whose difference is the
rounding the check tolerates. CI keeps the records while the vendored originals, the recorders and the
torch env's lock stay the same (``AJAX_CONFORMANCE_RECORDS``, under a cache key that hashes them);
unset, every session records them.
"""

import os
import pathlib
import subprocess

import numpy as np

from tests.conformance import submaps

REPO = pathlib.Path(__file__).parents[2]
CPU = os.environ | {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
KEPT = os.environ.get("AJAX_CONFORMANCE_RECORDS")
TORCH_ENV = "benchmarks/reference/envs/torch"  # CPU-only PyTorch, gymnax, jax


def recorded(
    name: str, recorder: str, envs: tuple[str, ...], scratch: pathlib.Path
) -> pathlib.Path:
    """``recorder``'s records (benchmarks/reference/) on ``envs``: CI's kept ones when whole, else
    recorded now in the torch env, under the kept directory or else in ``scratch``."""
    out = REPO / KEPT / name if KEPT else scratch
    marker = out / "envs"  # written last: unfinished records are never taken
    if marker.is_file() and marker.read_text() == " ".join(envs):
        return out
    for env_id in envs:
        script = f"benchmarks/reference/{recorder}"
        command = ["uv", "run", "--project", TORCH_ENV, "--locked", "python", script]
        subprocess.run([*command, env_id, str(out)], cwd=REPO, env=CPU, check=True)
    marker.write_text(" ".join(envs))
    return out


def reports(records: pathlib.Path, ours: dict, env_id: str, update: dict) -> list:
    """Per record of ``env_id``, ``update``'s state (by record name, in the original's names) against
    the original's float32 one, on the change from ``in.``, within 10 times the two sides' float32
    rounding (``ours``: Ajax's float32 and float64 runs, :func:`submaps.rounding_null`)."""
    out = []
    for path in sorted(records.glob(f"{env_id}_*.npz")):
        ref = dict(np.load(path))
        a32, a64, new = (d[path.stem] for d in (ours["f32"], ours["f64"], update))
        old = {k: ref[f"in.{k}"] for k in a32}
        r32 = {k: ref[f"f32.{k}"] for k in a32}
        nulls = submaps.rounding_null(
            (r32, {k: ref[f"f64.{k}"] for k in a32}), (a32, a64), old=old
        )
        out.append(submaps.compare(new, r32, nulls, old=old, context=path.stem))
    assert out, f"no record for {env_id}"
    return out
