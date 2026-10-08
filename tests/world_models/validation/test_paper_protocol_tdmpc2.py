"""``paper_protocol.py --smoke`` for TD-MPC2, resumed from disk (M9; tiny
CPU run on playground CartpoleBalance: plumbing only, no learning claim).

That a resumed TD-MPC2 run, restored from a checkpoint into a new agent's
skeleton, computes what the uninterrupted run computes is pinned by
``tests/agents/TDMPC2/test_TDMPC2.py``; here, the script's records.
"""

from __future__ import annotations

import numpy as np
import paper_report
import pytest

from .smoke_helpers import run_resumed

pytestmark = pytest.mark.slow

NAME = "tdmpc2-cartpole-balance"


def test_smoke_run_resumes_from_disk(tmp_path):
    records = run_resumed(NAME, tmp_path)
    # Two chunks of 2 episodes (T = 20 agent steps), each evaluated on 2
    # whole episodes of the planner in eval_mode, for both seeds.
    assert [r["progress"] for r in records] == [40, 80]
    assert [r["env_frames"] for r in records] == [80, 160]
    for record in records:
        assert record["metric"] == "eval_return" and record["seeds"] == [0, 1]
        assert record["unit"] == "agent_steps"
        assert record["env_frames"] == 2 * record["progress"]
        assert record["length"] == [20.0, 20.0] and record["episodes"] == [2, 2]
        assert np.all(np.isfinite(record["value"]))
    # UTD 1 after the 20 seed steps' burst (20 updates): one per env step.
    assert [r["n_updates"] for r in records] == [[39, 39], [79, 79]]
    # The run.json the CLI wrote is its registry entry's: on protocol.
    runs = paper_report.read_runs(str(tmp_path / "split"))
    assert runs["TDMPC2", "cartpole-balance"]["off_protocol"] == []
