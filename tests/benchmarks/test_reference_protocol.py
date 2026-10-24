"""``benchmarks/reference/refcmp_protocol.py``: AdaStop trains only the
seeds it lacks and reuses recorded runs while their fingerprint holds; a
batch records its curves, speed and manifest. No comparison trains here: a
stub runner stands in for the libraries."""

import dataclasses
import json
import re
import sys

import numpy as np
import pytest
import refcmp_protocol as protocol
import refcmp_publish
import refcmp_worker
from refcmp_specs import COMPARISONS, spec_sha256

from ajax import reproduction


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "CACHE", tmp_path)
    return tmp_path


def fake_train(calls, final):
    def train(name, side, seeds):
        calls.append((side, seeds))
        curve = [[0.0, final[side]] for _ in seeds]
        speed = {"wall_s": 1.0, "cpu_s": 2.0, "compile_s": 0.5, "env_steps_per_s": 10}
        return {
            "side": side,
            "spec_version": COMPARISONS[name].version,
            "implementation": protocol.implementation(name, side),
            "fingerprint": protocol.fingerprint(name, side),
            "seeds": seeds,
            "steps": [1, 2],
            "curve": curve,
            "speed": speed,
            "manifest": {"hardware": {"cpu": "M", "cores": 2}},
        }

    return train


def test_decide_trains_only_missing_seeds_and_reuses_the_records(cache, monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        protocol, "train", fake_train(calls, {"reference": 500.0, "matched": 9.0})
    )
    test, scores = protocol.decide("PQN")
    n = COMPARISONS["PQN"].adastop[0]
    # Constant scores, one side far below: decided at the second interim,
    # the first whose share of alpha (alpha / K) 4 + 4 permutations can reach.
    assert test.decisions == {("matched", "reference"): "smaller"}
    batches = [list(range(n)), list(range(n, 2 * n))]
    assert sorted(calls) == [
        (side, b) for side in ("matched", "reference") for b in batches
    ]
    assert len((cache / "PQN.jsonl").read_text().splitlines()) == 4
    calls.clear()
    _, keys, _ = protocol.reported("PQN")
    assert protocol.decide("PQN", keys=keys)[1] == scores and not calls
    row = protocol.summary("PQN", reps=10)[0]
    assert (row["decision"], row["seeds"], row["difference"]) == (
        "smaller",
        2 * n,
        -491.0,
    )
    assert (row["margin"], row["verdict"]) == (25.0, "different")
    assert row["speed"]["ajax"]["cpu_s_per_1m_steps"] == pytest.approx(
        1e6 * 2.0 / n / COMPARISONS["PQN"].n_timesteps
    )


def test_a_changed_setting_or_source_retrains(cache, monkeypatch, tmp_path):
    before = protocol.fingerprint("PQN", "matched")
    assert before == protocol.fingerprint("PQN", "matched")
    assert before != protocol.fingerprint("PQN", "reference")
    spec = dataclasses.replace(COMPARISONS["PQN"], matched={"n_envs": 4})
    monkeypatch.setitem(COMPARISONS, "PQN", spec)
    assert protocol.fingerprint("PQN", "matched") != before
    source = tmp_path / "runner.py"
    source.write_text("a")
    monkeypatch.setattr(protocol, "HERE", tmp_path)
    implementation = (None, None, False, ("runner.py",))
    monkeypatch.setitem(protocol.IMPLEMENTATIONS, "ajax", implementation)
    first = protocol.fingerprint("PQN", "matched")
    source.write_text("b")
    assert protocol.fingerprint("PQN", "matched") != first
    with pytest.raises(LookupError, match="no runs"):
        protocol.final_scores("PQN", "matched", [0], key="unrecorded")


STUB = """
import json, sys
seeds = range(int(sys.argv[3]), int(sys.argv[3]) + int(sys.argv[4]))
print("noise")
print(json.dumps({"curve": [[s, 2 * s] for s in seeds], "config": {"a": 1},
                  "versions": {"x": "1"}, "compile_s": 0.25}))
"""


@pytest.mark.parametrize("per_seed", [False, True])
def test_a_batch_records_curves_speed_and_manifest(monkeypatch, tmp_path, per_seed):
    (tmp_path / "stub.py").write_text(STUB)
    monkeypatch.setattr(protocol, "HERE", tmp_path)
    project = protocol.ROOT  # Ajax's uv project runs the stub
    monkeypatch.setitem(
        protocol.IMPLEMENTATIONS, "ajax", (project, "stub.py", per_seed, ())
    )
    record = protocol.train("PQN", "matched", [3, 4])
    assert record["curve"] == [[3, 6], [4, 8]] and record["seeds"] == [3, 4]
    assert record["steps"][-1] == COMPARISONS["PQN"].n_timesteps
    assert record["speed"]["processes"] == (2 if per_seed else 1)
    assert record["speed"]["compile_s"] == 0.25 and record["speed"]["cpu_s"] > 0
    manifest = record["manifest"]
    assert len(manifest["ajax_commit"]) == 40 and manifest["config"] == {"a": 1}
    assert manifest["hardware"]["cores"] > 0 and manifest["jax_platforms"] == "cpu"
    assert len(manifest["lock_sha256"]) == 64


def test_worker_prints_one_record(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner", "PQN", "matched", "5", "2"])
    refcmp_worker.main(
        lambda spec, side, seeds: np.full((len(seeds), 3), float(seeds[0])),
        lambda spec, side: {"side": side, "env": spec.env_id},
        ("numpy",),
    )
    record = json.loads(capsys.readouterr().out)
    assert record["curve"] == [[5.0] * 3] * 2
    assert record["config"] == {"side": "matched", "env": "CartPole-v1"}
    assert set(record["versions"]) == {"numpy", "python"} and record["compile_s"] >= 0
    defaults = refcmp_worker.resolved(lambda a, b=2, c=print: None, {"b": 3})
    assert defaults == {"b": 3}


def test_the_dataset_loads_back_and_the_docs_draw_from_it(cache, monkeypatch, tmp_path):
    finals = {"reference": 500.0, "matched": 9.0}
    monkeypatch.setattr(protocol, "train", fake_train([], finals))
    protocol.decide("PQN")
    out = tmp_path / "dataset"
    refcmp_publish.dataset(out, reps=10)
    runs = reproduction.load("runs", str(out))
    n = COMPARISONS["PQN"].adastop[0]
    assert len(runs) == 2 * 2 * n * 2  # sides x seeds x evaluations
    assert set(runs["return"]) == {0.0, 9.0, 500.0}
    assert set(reproduction.load("batches", str(out))["side"]) == set(finals)
    assert reproduction.load("summary", str(out))["decision"][0] == "smaller"
    assert (out / "README.md").exists() and "PQN" in (out / "specs.json").read_text()
    with pytest.raises(ValueError, match="table is one of"):
        reproduction.load("curves", str(out))
    monkeypatch.setattr(refcmp_publish, "DOCS", tmp_path)
    refcmp_publish.docs(reps=10)
    version = COMPARISONS["PQN"].version
    assert (
        f"| PQN | v{version},  | different | ±25 | 8 | 500 | 9 | -491 ["
        in (tmp_path / "results.md").read_text()
    )
    assert (tmp_path / "PQN.svg").read_text().startswith("<?xml")


def test_recorded_results_ran_the_current_spec_from_a_clean_tree():
    # A spec edited after its results must become a new version instead.
    for path in protocol.CACHE.glob("*.jsonl"):
        spec = COMPARISONS[path.stem]
        for record in protocol.records(path.stem):
            manifest = record["manifest"]
            assert manifest["ajax_clean"], (path.stem, record["seeds"])
            if record["spec_version"] == spec.version:
                assert manifest["spec_sha256"] == spec_sha256(spec), (
                    f"{path.stem} version {spec.version} changed after its"
                    f" results: make it version {spec.version + 1}"
                )


def test_the_docs_state_the_protocol_version_and_the_current_specs():
    docs = protocol.ROOT / "docs" / "reproduction"
    version = re.search(r"^Version (\d+),", (docs / "protocol.md").read_text(), re.M)
    assert int(version.group(1)) == protocol.PROTOCOL_VERSION
    assert (docs / "specs.md").read_text() == refcmp_publish.specs_page(), (
        "run benchmarks/reference/refcmp_publish.py specs"
    )


@pytest.mark.parametrize(
    ("interval", "word"),
    [((-20, 24), "similar"), ((-40, -30), "different"), ((-30, 10), "inconclusive")],
)
def test_the_verdict_sets_the_interval_against_the_margin(interval, word):
    spec = COMPARISONS["PQN"]  # margin 25
    assert protocol.verdict(spec, interval, 500.0) == (25.0, word)
    relative = dataclasses.replace(spec, margin=None, relative_margin=0.05)
    assert protocol.verdict(relative, interval, -500.0) == (25.0, word)
    none = dataclasses.replace(spec, margin=None)
    assert protocol.verdict(none, interval, 1.0)[1] == "no margin registered"


def test_the_records_do_not_count_against_a_clean_tree(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(protocol, "_run", lambda *command: calls.append(command) or "")
    assert protocol.clean_tree()
    assert calls[-1][-1] == ":!benchmarks/reference/results/protocol"
    monkeypatch.setattr(protocol, "CACHE", tmp_path)  # outside the repository
    assert protocol.clean_tree() and calls[-1][-1] == "."


def test_a_record_is_judged_under_the_spec_it_ran(monkeypatch):
    # The committed specs: read back from HEAD, as from any record's commit.
    committed = COMPARISONS["PQN"]
    head = protocol._run("git", "rev-parse", "HEAD")
    record = {"manifest": {"ajax_commit": head, "spec_sha256": spec_sha256(committed)}}
    monkeypatch.setitem(COMPARISONS, "PQN", dataclasses.replace(committed, margin=1.0))
    assert protocol.spec_of("PQN", record).margin == committed.margin
    record["manifest"]["spec_sha256"] = "0" * 64
    with pytest.raises(LookupError, match="has its hash"):
        protocol.spec_of("PQN", record)


def test_deviations_document_the_codes_without_changing_what_a_spec_runs():
    for name, spec in COMPARISONS.items():
        assert spec_sha256(spec) == spec_sha256(
            dataclasses.replace(spec, deviations=())
        ), name
        for d in spec.deviations:
            assert d.kind in ("ajax", "convention", "reference defect"), (name, d.name)
            assert d.check.startswith("tests/conformance/") and "::test_" in d.check


def test_a_subproject_runs_without_the_callers_venv_and_resolution_mode(monkeypatch):
    """compat.yml's lowest job exports UV_RESOLUTION=lowest-direct for Ajax's own lock; a reference
    subproject's uv.lock records its own mode, which `uv run --locked` would otherwise refuse."""
    monkeypatch.setenv("UV_RESOLUTION", "lowest-direct")
    monkeypatch.setenv("VIRTUAL_ENV", "/caller/.venv")
    monkeypatch.setenv("AJAX_KEPT", "1")
    env = protocol.subproject_env(protocol.HERE / "envs" / "torch", JAX_PLATFORMS="cpu")
    assert not set(protocol.CALLER_SETTINGS).intersection(env)
    assert (env["AJAX_KEPT"], env["JAX_PLATFORMS"]) == ("1", "cpu")


def test_ajax_s_own_project_keeps_the_callers_resolution_mode(monkeypatch):
    """The lowest job's lowest-direct sync rewrote Ajax's own uv.lock in that mode, so Ajax's side
    of a comparison runs `uv run --locked` in it too (#214); only the virtualenv goes."""
    monkeypatch.setenv("UV_RESOLUTION", "lowest-direct")
    monkeypatch.setenv("VIRTUAL_ENV", "/caller/.venv")
    env = protocol.subproject_env(protocol.ROOT)
    assert env["UV_RESOLUTION"] == "lowest-direct"
    assert "VIRTUAL_ENV" not in env
