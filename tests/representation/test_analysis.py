"""Reading a run back and embedding it.

Contract (ajax.representation.run_data, .embeddings): a run reads in the order
its probes were taken, each seed apart; a trajectory's principal components
explain it as linear algebra says; a matched random walk takes the trajectory's
step lengths; the k-th principal component of a long high-dimensional random
walk explains 6 / (pi k)^2 of its variance (Antognini and Sohl-Dickstein,
NeurIPS 2018), which is the baseline's whole point; M-PHATE gets at most the
nodes it is allowed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ajax.representation import embeddings, run_data


def write_probe(directory: Path, seed: int, probe: int, **arrays) -> None:
    np.savez(directory / run_data.PROBE_FILE.format(seed=seed, probe=probe), **arrays)


def test_probes_read_in_the_order_they_were_taken(tmp_path):
    """Catches sorting by file name alone beyond the padding's width, which
    would shuffle a long run's trajectory: probe 100000 follows 99999."""
    for probe in (99999, 100000, 2):
        write_probe(tmp_path, 0, probe, step=np.asarray(probe))

    assert run_data.load_steps(tmp_path).tolist() == [2, 99999, 100000]


def test_each_seed_reads_its_own_files(tmp_path):
    """Catches seeds mixed into one trajectory, whose steps would jump between
    two different networks."""
    write_probe(tmp_path, 0, 0, step=np.asarray(0))
    write_probe(tmp_path, 1, 0, step=np.asarray(10))

    assert run_data.load_steps(tmp_path, seed=1).tolist() == [10]
    assert sorted(run_data.probe_paths(tmp_path)) == [0, 1]


def test_an_empty_directory_says_so(tmp_path):
    """Catches a wrong path read as a run of nothing."""
    with pytest.raises(FileNotFoundError, match="no probe file"):
        run_data.probe_paths(tmp_path)


def test_an_ensemble_layer_reads_one_member(tmp_path):
    """Catches the member axis read as states, or members averaged into a
    representation no critic has."""
    write_probe(tmp_path, 0, 0, **{"layers/critic/x": np.arange(24.0).reshape(2, 3, 4)})

    layer = run_data.load_layer(tmp_path, "critic/x", member=1)

    assert layer.shape == (1, 3, 4)
    assert layer[0, 0, 0] == 12.0


def test_parameters_flatten_in_one_order_at_every_probe(tmp_path):
    """Catches a flattening that follows each file's order, which would turn
    a weight trajectory into noise. Names sort a, then b, whatever was saved."""
    write_probe(
        tmp_path,
        0,
        0,
        **{
            "parameters/actor/b": np.array([2.0]),
            "parameters/actor/a": np.array([1.0]),
        },
    )
    write_probe(
        tmp_path,
        0,
        1,
        **{
            "parameters/actor/a": np.array([3.0]),
            "parameters/actor/b": np.array([4.0]),
        },
    )

    assert run_data.load_parameters(tmp_path, "actor").tolist() == [[1, 2], [3, 4]]


def test_metrics_read_per_seed_and_probe(tmp_path):
    """Catches a table with its axes swapped, plotted as probes across seeds."""
    name = "metrics/Representation/actor/x/effective_rank"
    for seed in (0, 1):
        for probe in (0, 1, 2):
            write_probe(tmp_path, seed, probe, **{name: np.asarray(float(probe))})

    table = run_data.load_metrics(tmp_path)[name.removeprefix("metrics/")]

    assert table.shape == (2, 3)
    assert table[1].tolist() == [0.0, 1.0, 2.0]


def test_saved_evaluations_come_back(tmp_path):
    """Catches a run's returns lost, and a run without them failing."""
    assert run_data.load_evaluations(tmp_path) is None

    run_data.save_evaluations(tmp_path, {"timestep": [[1, 2]]})

    assert run_data.load_evaluations(tmp_path)["timestep"].tolist() == [[1, 2]]


@pytest.mark.parametrize(
    "n_items, at_most, expected",
    [(5, 10, [0, 1, 2, 3, 4]), (5, 3, [0, 2, 4]), (101, 2, [0, 100]), (1, 4, [0])],
)
def test_evenly_spaced_keeps_both_ends(n_items, at_most, expected):
    """Catches losing the last probe, where training ended."""
    assert embeddings.evenly_spaced(n_items, at_most).tolist() == expected


@pytest.mark.parametrize(
    "n_probes, n_units, kept_probes, kept_units",
    [
        # All fit: 10 * 64 = 640.
        (10, 64, 10, 64),
        # 20000 // 101 = 198 units a probe, every probe kept.
        (101, 512, 101, 198),
        # 20000 // 1000 = 20 is under the 32 units worth keeping: 32 units,
        # then 20000 // 32 = 625 probes.
        (1000, 512, 625, 32),
    ],
)
def test_fit_to_nodes_stays_within_the_budget(
    n_probes, n_units, kept_probes, kept_units
):
    """Catches an M-PHATE of too many nodes, whose dense kernel would take
    tens of gigabytes, or a sample dropping more than it must."""
    probes, units = embeddings.fit_to_nodes(n_probes, n_units, 20_000)

    assert (len(probes), len(units)) == (kept_probes, kept_units)
    assert len(probes) * len(units) <= 20_000
    assert len(np.unique(units)) == kept_units


def test_a_straight_trajectory_is_one_principal_component():
    """Catches an uncentred PCA, whose first component would point at the
    mean. A line has one direction: PC1 explains all of it, in order."""
    time = np.linspace(0.0, 1.0, 16)[:, None]
    direction = np.array([[3.0, -4.0, 0.0]])

    path, shares = embeddings.principal_path(10.0 + time * direction)

    assert shares[0] == pytest.approx(1.0)
    assert np.all(np.diff(path[:, 0]) > 0) or np.all(np.diff(path[:, 0]) < 0)
    assert np.ptp(path[:, 0]) == pytest.approx(5.0)


def test_a_trajectory_that_never_moved_explains_nothing():
    """Catches a NaN figure for an actor whose updates had not started."""
    path, shares = embeddings.principal_path(np.ones((4, 8)))

    assert shares.tolist() == [0.0, 0.0]
    assert np.abs(path).max() == 0.0


def test_a_matched_random_walk_keeps_the_start_and_step_lengths():
    """Catches a baseline with other steps: PCA's picture of a walk depends
    on them, so it would compare nothing."""
    random = np.random.default_rng(0)
    points = np.cumsum(random.normal(size=(16, 64)) * [[3.0]], axis=0)

    walk = embeddings.matched_random_walk(points)

    np.testing.assert_array_equal(walk[0], points[0])
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(walk, axis=0), axis=1),
        np.linalg.norm(np.diff(points, axis=0), axis=1),
    )
    assert np.abs(walk[-1] - points[-1]).max() > 1.0


def test_pca_of_a_random_walk_explains_what_theory_says():
    """Catches a walk that is not one, or a PCA not over the trajectory: PC1
    and PC2 of a walk explain 6 / pi^2 = .61 and 6 / (2 pi)^2 = .15. With 256
    steps the shares fluctuate; .12 and .06 cover the seed's spread."""
    points = np.zeros((257, 2048))

    walk = embeddings.matched_random_walk(points + np.arange(257)[:, None], seed=1)
    _, shares = embeddings.principal_path(walk)

    assert shares[0] == pytest.approx(6 / np.pi**2, abs=0.12)
    assert shares[1] == pytest.approx(6 / (2 * np.pi) ** 2, abs=0.06)
