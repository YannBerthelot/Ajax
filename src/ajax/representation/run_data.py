"""Reading back what a RepresentationProbe wrote, with NumPy alone."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

#: A probe's file in a run directory: the seed's index, then the probe's.
PROBE_FILE = "seed{seed:03d}_probe{probe:05d}.npz"
PROBE_FILE_PATTERN = re.compile(r"seed(\d+)_probe(\d+)\.npz")
#: What :func:`save_evaluations` writes.
EVALUATIONS_FILE = "evaluations.npz"


def probe_paths(directory: str | os.PathLike) -> dict[int, list[Path]]:
    """Each seed's probe files, in the order they were taken."""
    found: dict[int, list[tuple[int, Path]]] = {}
    for path in Path(directory).iterdir():
        named = PROBE_FILE_PATTERN.fullmatch(path.name)
        if named is not None:
            seed, probe = int(named[1]), int(named[2])
            found.setdefault(seed, []).append((probe, path))
    if not found:
        raise FileNotFoundError(f"{directory} holds no probe file")
    return {seed: [p for _, p in sorted(found[seed])] for seed in sorted(found)}


def layer_names(directory: str | os.PathLike) -> list[str]:
    """The layers the run recorded."""
    first = next(iter(probe_paths(directory).values()))[0]
    with np.load(first) as probe:
        return [
            k.removeprefix("layers/") for k in probe.files if k.startswith("layers/")
        ]


def load_steps(directory: str | os.PathLike, seed: int = 0) -> np.ndarray:
    """The env step of each of the seed's probes."""
    steps = []
    for path in probe_paths(directory)[seed]:
        with np.load(path) as probe:
            steps.append(int(probe["step"]))
    return np.asarray(steps)


def load_layer(
    directory: str | os.PathLike, layer: str, seed: int = 0, member: int = 0
) -> np.ndarray:
    """One layer through training, ``(n_probes, n_states, n_units)``:
    ``member``'s, on an ensemble."""
    probes = []
    for path in probe_paths(directory)[seed]:
        with np.load(path) as probe:
            activations = probe[f"layers/{layer}"]
        probes.append(activations[member] if activations.ndim == 3 else activations)
    return np.stack(probes)


def load_values(directory: str | os.PathLike, seed: int = 0) -> np.ndarray | None:
    """The critic's value of each state at each probe, ``(n_probes,
    n_states)``; ``None`` for an agent without a critic ensemble."""
    probes = []
    for path in probe_paths(directory)[seed]:
        with np.load(path) as probe:
            if "values" not in probe.files:
                return None
            probes.append(probe["values"])
    return np.stack(probes)


def load_parameters(
    directory: str | os.PathLike, network: str, seed: int = 0
) -> np.ndarray:
    """A network's parameters through training, flattened,
    ``(n_probes, n_parameters)``."""
    prefix = f"parameters/{network}/"
    rows = []
    for path in probe_paths(directory)[seed]:
        with np.load(path) as probe:
            names = sorted(k for k in probe.files if k.startswith(prefix))
            rows.append(np.concatenate([probe[name].ravel() for name in names]))
    return np.stack(rows)


def load_metrics(directory: str | os.PathLike) -> dict[str, np.ndarray]:
    """Every metric through training, ``(n_seeds, n_probes)``, by the name
    the logger shows it under."""
    columns: dict[str, list] = {}
    for paths in probe_paths(directory).values():
        rows: dict[str, list] = {}
        for path in paths:
            with np.load(path) as probe:
                for key in probe.files:
                    if key.startswith("metrics/"):
                        name = key.removeprefix("metrics/")
                        rows.setdefault(name, []).append(float(probe[key]))
        for name, values in rows.items():
            columns.setdefault(name, []).append(values)
    return {name: np.asarray(seeds) for name, seeds in columns.items()}


def save_evaluations(directory: str | os.PathLike, evaluations: dict) -> None:
    """Keeps ``train``'s evaluations with the probes, so that an analysis can
    set the metrics against the returns."""
    arrays = {name: np.asarray(values) for name, values in evaluations.items()}
    with open(Path(directory) / EVALUATIONS_FILE, "wb") as file:
        np.savez(file, **arrays)  # type: ignore[arg-type]


def load_evaluations(directory: str | os.PathLike) -> dict[str, np.ndarray] | None:
    """What :func:`save_evaluations` kept; ``None`` when it was not called."""
    path = Path(directory) / EVALUATIONS_FILE
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {name: saved[name] for name in saved.files}
