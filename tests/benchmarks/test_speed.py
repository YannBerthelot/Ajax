"""``benchmarks/speed/speed_figure.py``, which writes the speed comparison's
tables and the README's sentence from its records.

Contract (benchmarks/speed/README.md): a cell shows the latest record of its
algorithm, task, library, seed count and device; a library that runs one
process per seed shows ten seeds at the faster of its ten-process run and its
one-seed speed (its seeds one after another); Ajax's factor is its speed over
the fastest other library's in the same cell; a family's headline spans its
ten-seed CPU cells' factors, lowest to highest.
"""

import json

import speed_figure


def record(library, seeds, speed, how, algorithm="SAC", device="cpu"):
    return {
        "algorithm": algorithm,
        "task": "Pendulum-v1",
        "library": library,
        "seeds": seeds,
        "device": device,
        "env_steps_per_s": speed,
        "how": how,
    }


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_docs_and_readme_match_the_records():
    """Catches records added or changed without rerunning speed_figure.py,
    and hand edits to the generated tables or sentence."""
    speeds = speed_figure.read()
    page = speed_figure.PAGE.read_text()
    block = speed_figure.replace_block(
        page, "speed-by-family", speed_figure.table(speeds)
    )
    settings = speed_figure.settings_block()
    assert speed_figure.replace_block(block, "speed-settings", settings) == page
    readme = speed_figure.README.read_text()
    sentence = speed_figure.sentence(speeds)
    assert speed_figure.replace_block(readme, "speed-by-family", sentence) == readme


def test_ten_processes_slower_than_one_seed_show_one_after_another(tmp_path):
    """Catches crediting a process-per-seed library with its ten-process run
    when its seeds one after another (its one-seed speed) are faster, which
    would overstate Ajax's factor."""
    path = tmp_path / "speed.jsonl"
    write(
        path,
        [
            record("SB3", 1, 100.0, "processes"),
            record("SB3", 10, 80.0, "processes"),
            record("Ajax", 10, 400.0, "program"),
        ],
    )
    row = speed_figure.read(path)[("SAC", "Pendulum-v1", 10, "cpu")]
    assert row["SB3"] == (100.0, "one after another")
    assert speed_figure.factor(row) == (4.0, "SB3")


def test_the_latest_record_of_a_cell_replaces_the_earlier(tmp_path):
    """Catches a rerun of a cell that leaves the stale record in the table."""
    path = tmp_path / "speed.jsonl"
    write(
        path,
        [
            record("Ajax", 1, 50.0, "program"),
            record("SBX", 1, 100.0, "processes"),
            record("Ajax", 1, 300.0, "program"),
        ],
    )
    row = speed_figure.read(path)[("SAC", "Pendulum-v1", 1, "cpu")]
    assert speed_figure.factor(row) == (3.0, "SBX")


def test_a_family_headline_spans_its_ten_seed_cpu_cells(tmp_path):
    """Catches a headline drawn from one-seed or GPU cells, or from one task."""
    path = tmp_path / "speed.jsonl"
    write(
        path,
        [
            record("Ajax", 10, 300.0, "program"),
            record("SBX", 10, 100.0, "processes"),
            record("Ajax", 10, 100.0, "program", algorithm="TD3"),
            record("SBX", 10, 200.0, "processes", algorithm="TD3"),
            record("Ajax", 1, 1000.0, "program"),  # one seed: not in the headline
            record("SBX", 1, 1.0, "processes"),
            record("Ajax", 10, 9000.0, "program", device="gpu"),
            record("Brax", 10, 1.0, "program", device="gpu"),
        ],
    )
    heads = speed_figure.headlines(speed_figure.read(path))
    assert heads == {"SAC and TD3": (0.5, 3.0)}
