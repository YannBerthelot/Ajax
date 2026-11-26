"""A test process releases the programs JAX compiled for a file once it moves
on to the next (tests/conftest.py, pytest_runtest_teardown)."""

import os
import subprocess
import sys
import textwrap


def test_the_next_file_builds_again_what_the_last_one_compiled(tmp_path) -> None:
    """Two files call one jitted function: with the programs released at the
    file boundary, the second traces it again, so a trace counter reads 2.
    Catches the release no longer running (a renamed or shadowed hook), which
    shows only as workers growing until a 16 GB runner dies (compatibility,
    lowest, shard 4, 2026-11-26)."""
    (tmp_path / "shared.py").write_text(
        textwrap.dedent(
            """
            import jax

            traces = []

            @jax.jit
            def step(x):
                traces.append(1)
                return x + 1
            """
        )
    )
    (tmp_path / "test_a.py").write_text(
        "import shared\n\ndef test_a():\n    shared.step(1)\n"
    )
    (tmp_path / "test_b.py").write_text(
        "import shared\n\ndef test_b():\n    shared.step(1)\n    assert len(shared.traces) == 2\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), repo]))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += ["-p", "no:xdist", "-p", "tests.conftest", "test_a.py", "test_b.py"]
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True, capture_output=True)
