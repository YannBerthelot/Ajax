"""A run's figures: ``python -m ajax.representation RUN_DIRECTORY FIGURES``.

Needs the ``manifolds`` extra. See docs/guide/representations.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ajax.representation import figures, run_data
from ajax.representation.probe_set import PROBE_SET_FILE, ProbeSet


def network_of(layer: str) -> str:
    return layer.split("/")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="the seed index to embed")
    parser.add_argument("--max-probes", type=int, default=12, help="Aligned UMAP's")
    parser.add_argument("--max-states", type=int, default=1024, help="Aligned UMAP's")
    parser.add_argument("--max-nodes", type=int, default=20_000, help="M-PHATE's")
    parser.add_argument(
        "--observation-columns",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="the observation columns to colour the states by",
    )
    parser.add_argument(
        "--truth",
        nargs="*",
        default=[],
        help="the probe set's ground-truth fields to colour the states by",
    )
    arguments = parser.parse_args()
    directory, output = arguments.run_directory, arguments.output
    output.mkdir(parents=True, exist_ok=True)

    states = ProbeSet.load(directory / PROBE_SET_FILE)
    colourings: dict[str, np.ndarray] = {}
    for column in arguments.observation_columns:
        if column < states.observations.shape[1]:
            colourings[f"Observation {column}"] = states.observations[:, column]
    for field in arguments.truth:
        colourings[field] = states.ground_truth[field]

    layers = run_data.layer_names(directory)
    networks = sorted({network_of(layer) for layer in layers})
    figures.metrics_figure(directory, output / "metrics.png")
    for network in networks:
        own_layers = [layer for layer in layers if network_of(layer) == network]
        figures.manifold_figure(
            directory,
            own_layers[-1],
            output / f"manifold_{network}.png",
            colourings,
            arguments.max_probes,
            arguments.max_states,
            arguments.seed,
        )
        figures.units_figure(
            directory,
            own_layers,
            output / f"units_{network}.png",
            arguments.max_nodes,
            arguments.seed,
        )
        figures.weights_figure(
            directory, network, output / f"weights_{network}.png", arguments.seed
        )


if __name__ == "__main__":
    main()
