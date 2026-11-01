"""Which slow-tier runs tested a commit, read off their shard jobs, not off
the run's conclusion: a run whose shards were skipped still concludes
"success", and must never count as green.

    python3 .github/scripts/slow_runs.py history
        prints last=SHA and last_green=SHA: the last daily or by-hand run on
        main whose shards ran, and the last whose shards all passed.
    python3 .github/scripts/slow_runs.py green-at SHA
        prints green=true when a run at SHA had all its shards pass.

Both read the GitHub API through gh, for $GITHUB_REPOSITORY.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Iterable

#: The shard jobs' names, "Slow (shard 1)" to "Slow (shard 6)" (slow.yml).
#: Skipped, the matrix is not expanded: "Slow (shard ${{ matrix.shard }})".
SHARD = "Slow (shard "


def outcome(jobs: Iterable[dict[str, Any]]) -> str:
    """ "passed" when the run's shard jobs ran and all succeeded, "failed"
    when one ran and did not succeed (failed, timed out or cancelled), "not
    run" when none ran."""
    shards = [job for job in jobs if job["name"].startswith(SHARD)]
    ran = [job for job in shards if job["conclusion"] not in ("skipped", None)]
    if not ran:
        return "not run"
    if len(ran) == len(shards) and all(job["conclusion"] == "success" for job in ran):
        return "passed"
    return "failed"


def last_tested(runs: Iterable[tuple[str, str]]) -> tuple[str, str]:
    """From runs (sha, outcome), newest first: the sha of the last one whose
    shards ran, and of the last one whose shards passed ("" for none)."""
    last, last_green = "", ""
    for sha, result in runs:
        if result != "not run" and not last:
            last = sha
        if result == "passed":
            last_green = sha
            break
    return last, last_green


def _api(path: str) -> Any:
    out = subprocess.run(
        ["gh", "api", path], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout)


def _outcome_of(run_id: int) -> str:
    repository = os.environ["GITHUB_REPOSITORY"]
    return outcome(
        _api(f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
    )


def _runs(query: str) -> list[dict[str, Any]]:
    repository = os.environ["GITHUB_REPOSITORY"]
    path = f"repos/{repository}/actions/workflows/slow.yml/runs?status=completed&per_page=50&{query}"
    return _api(path)["workflow_runs"]


def main(argv: list[str]) -> None:
    if argv[:1] == ["history"]:
        runs = [
            run
            for run in _runs("branch=main")
            if run["event"] in ("schedule", "workflow_dispatch")
        ]
        # Lazily, newest first: the API is asked for jobs only until both are found.
        last, last_green = last_tested(
            (run["head_sha"], _outcome_of(run["id"])) for run in runs
        )
        print(f"last={last}")
        print(f"last_green={last_green}")
    elif argv[:1] == ["green-at"] and len(argv) == 2:
        green = any(
            _outcome_of(run["id"]) == "passed" for run in _runs(f"head_sha={argv[1]}")
        )
        print(f"green={'true' if green else 'false'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
