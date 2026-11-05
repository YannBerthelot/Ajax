"""The real embedding libraries on a tiny real run.

Contract (the guide, docs/guide/representations.md): ``python -m
ajax.representation RUN FIGURES`` turns a run directory into the figures.
umap-learn and M-PHATE stand in for no double: these tests are slow because
numba compiles them, not because they train.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from ajax import PQN, SAC
from ajax.extensions.representation import RepresentationProbe
from ajax.logging import LoggingConfig
from ajax.representation import embeddings, run_data
from ajax.representation.__main__ import main

pytestmark = pytest.mark.slow

NET = ("8", "relu", "8", "relu")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run")
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
        extensions=(RepresentationProbe(directory, 8, n_states=128),),
    )
    logging = LoggingConfig(log_frequency=8, backends=())
    _, evaluations = agent.train(seed=[0, 1], n_timesteps=64, logging_config=logging)
    run_data.save_evaluations(directory, evaluations)
    return directory


def test_the_command_writes_every_figure(run, tmp_path, monkeypatch):
    """Catches a figure failing on a real run's layers, names or shapes."""
    arguments = ["ajax.representation", str(run), str(tmp_path), "--truth", "theta"]
    monkeypatch.setattr(sys, "argv", arguments)

    main()

    names = {path.name for path in tmp_path.iterdir() if path.stat().st_size > 1000}
    assert names == {
        "metrics.png",
        "manifold_actor.png",
        "manifold_critic.png",
        "units_actor.png",
        "units_critic.png",
        "weights_actor.png",
        "weights_critic.png",
    }


def test_an_agent_without_a_critic_ensemble_gets_its_figures(tmp_path, monkeypatch):
    """Catches a figure assuming a critic's values: PQN keeps its Q-network on
    actor_state, and its run has no values to colour by."""
    directory, figures = tmp_path / "run", tmp_path / "figures"
    agent = PQN(
        "CartPole-v1",
        n_envs=2,
        n_steps=8,
        architecture=NET,
        extensions=(RepresentationProbe(directory, 8, n_states=128),),
    )
    agent.train(seed=[0], n_timesteps=64)
    monkeypatch.setattr(
        sys, "argv", ["ajax.representation", str(directory), str(figures)]
    )

    main()

    names = {path.name for path in figures.iterdir()}
    assert names == {
        "metrics.png",
        "manifold_actor.png",
        "units_actor.png",
        "weights_actor.png",
    }


def test_aligned_umap_keeps_a_state_in_place_while_nothing_changes():
    """Catches probes embedded apart, which is what the alignment is for: on
    identical probes a state stays far closer to itself than the embedding
    is wide, and the embedding has the state's place in every probe."""
    states = np.random.default_rng(0).normal(size=(256, 8))

    embedded = embeddings.aligned_umap(np.stack([states] * 4))

    assert embedded.shape == (4, 256, 2)
    moved = np.linalg.norm(embedded[0] - embedded[3], axis=-1).mean()
    assert moved < 0.2 * np.linalg.norm(np.ptp(embedded[0], axis=0))


def test_mphate_gives_a_point_per_unit_and_probe():
    """Catches the axes swapped: M-PHATE reads (time, units, states)."""
    trace = np.random.default_rng(0).normal(size=(8, 32, 64))

    embedded = embeddings.mphate(trace)

    assert embedded.shape == (8 * 32, 2)
    assert np.isfinite(embedded).all()
