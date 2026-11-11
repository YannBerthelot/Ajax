"""``benchmarks/suite/publish.py``, which writes the agent benchmark's tables
from its records.

Contract: a cell is the return at the last evaluation of each seed's latest
attempt, shown as mean ± sample standard deviation, and only once every seed
of the run reached the step budget; seeds run in parts are gathered. The
README and the benchmark page carry exactly what the records give.
"""

import json

import publish


def record(seeds, attempt, start, end, returns, budget=4):
    return {
        "seeds": seeds,
        "attempt": attempt,
        "start": start,
        "end": end,
        "budget": budget,
        "eval_steps": list(range(start + 2, end + 1, 2)),
        "returns": returns,
    }


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_published_tables_match_the_records():
    """Catches records added or changed without rerunning publish.py, and
    hand edits to the generated tables."""
    runs = publish.read_runs()
    page = publish.PAGE.read_text()
    readme = publish.README.read_text()
    assert (
        publish.replace_block(page, "benchmark-tables", publish.page_block(runs))
        == page
    )
    summary = publish.readme_block(runs)
    assert publish.replace_block(readme, "benchmark-summary", summary) == readme


def test_score_uses_the_latest_attempt_and_gathers_seeds_run_in_parts(tmp_path):
    """Catches scoring an abandoned first attempt, or dropping the seeds
    that ran in a second file."""
    write(
        tmp_path / "SAC" / "seeds-0-0.jsonl",
        [
            record([0], 0, 0, 2, [[100.0]]),  # abandoned at step 2 of 4
            record([0], 1, 0, 2, [[0.0]]),
            record([0], 1, 2, 4, [[1.0]]),
        ],
    )
    write(tmp_path / "SAC" / "seeds-1-1.jsonl", [record([1], 0, 0, 4, [[5.0, 3.0]])])

    run = publish.read_run("SAC", tmp_path)

    assert run.final.tolist() == [1.0, 3.0]
    # mean 2, sample standard deviation sqrt(((1-2)^2 + (3-2)^2) / 1) = 1.41
    assert publish.table_body(["T"], ["SAC"], {("T", "SAC"): run}, bold=True) == (
        "| T | 2.0 ± 1.4 |"
    )


def test_unfinished_run_has_no_cell(tmp_path):
    """Catches showing a run in progress, whose last evaluation is not the
    end of training."""
    write(
        tmp_path / "SAC" / "seeds-0-9.jsonl", [record([0, 1], 0, 0, 2, [[1.0], [2.0]])]
    )

    assert publish.read_run("SAC", tmp_path) is None


def test_bold_marks_the_best_mean_of_each_row(tmp_path):
    """Catches bolding the wrong agent, e.g. the lowest or the last."""
    write(
        tmp_path / "SAC" / "seeds-0-1.jsonl",
        [record([0, 1], 0, 0, 4, [[0.0, 2.0], [0.0, 4.0]])],
    )
    write(
        tmp_path / "TD3" / "seeds-0-1.jsonl",
        [record([0, 1], 0, 0, 4, [[0.0, 5.0], [0.0, 7.0]])],
    )
    cell = {("T", a): publish.read_run(a, tmp_path) for a in ("SAC", "TD3")}

    assert publish.table_body(["T"], ["SAC", "TD3"], cell, bold=True) == (
        "| T | 3.0 ± 1.4 | **6.0 ± 1.4** |"
    )
