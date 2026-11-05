"""Embedding a run's activations and weights, offline.

umap-learn and M-PHATE come with the ``manifolds`` extra, and only this
module's functions that need them import them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: M-PHATE's default interslice neighbours (Gigante et al., NeurIPS 2019); it
#: needs fewer than the checkpoints it is given.
INTERSLICE_NEIGHBOURS = 25
#: M-PHATE's default number of components it first reduces the states to; it
#: fails on fewer nodes than that.
PCA_COMPONENTS = 100
#: The fewest units per checkpoint worth embedding: below it, checkpoints go.
MIN_UNITS = 32


def evenly_spaced(n_items: int, at_most: int) -> np.ndarray:
    """At most ``at_most`` indices of ``n_items``, first and last included."""
    return np.unique(np.linspace(0, n_items - 1, min(n_items, at_most)).astype(int))


def aligned_umap(activations: np.ndarray, n_neighbours: int = 15, seed: int = 0) -> Any:
    """A layer's states through training, ``(n_probes, n_states, n_units)``,
    as ``(n_probes, n_states, 2)``. Each probe gets an embedding of its own,
    tied to the next by the states they share (Aligned UMAP): a joint
    embedding would have to treat one unit at two checkpoints as one axis."""
    import umap

    same_states = {state: state for state in range(activations.shape[1])}
    model = umap.AlignedUMAP(
        n_neighbors=n_neighbours, random_state=seed, n_components=2
    ).fit(list(activations), relations=[same_states] * (len(activations) - 1))
    return np.stack(model.embeddings_)


def fit_to_nodes(
    n_probes: int, n_units: int, max_nodes: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Which probes and units to keep so that their product, M-PHATE's nodes,
    stays within ``max_nodes``: units go first (they are many and alike),
    checkpoints only below :data:`MIN_UNITS` per checkpoint."""
    kept_units = min(n_units, max(max_nodes // n_probes, MIN_UNITS))
    kept_probes = min(n_probes, max_nodes // kept_units)
    units = np.random.default_rng(seed).choice(n_units, kept_units, replace=False)
    return evenly_spaced(n_probes, kept_probes), np.sort(units)


def mphate(activations: np.ndarray, seed: int = 0) -> Any:
    """Hidden units through training, ``(n_probes, n_units, n_states)``, one
    point per unit and probe, ``(n_probes * n_units, 2)`` (M-PHATE, Gigante
    et al., NeurIPS 2019)."""
    import m_phate

    n_probes, n_units = activations.shape[:2]
    model = m_phate.M_PHATE(
        interslice_knn=min(INTERSLICE_NEIGHBOURS, n_probes - 1),
        n_pca=min(PCA_COMPONENTS, n_probes * n_units - 1),
        random_state=seed,
        verbose=0,
    )
    return model.fit_transform(activations)


def principal_path(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A trajectory on its two principal components, ``(n_points, 2)``, and
    the share of the variance each explains, none for a trajectory that
    never moved (an actor before its first update)."""
    centred = points - points.mean(axis=0)
    left, singular_values, _ = np.linalg.svd(centred, full_matrices=False)
    variances = np.square(singular_values)
    shares = variances / variances.sum() if variances.sum() > 0 else variances
    return left[:, :2] * singular_values[:2], shares[:2]


def matched_random_walk(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random walk from the same start, step for step as long as
    ``points``' steps, in random directions. PCA draws any high-dimensional
    random walk as a smooth curve explaining most of its variance (Antognini
    and Sohl-Dickstein, NeurIPS 2018), so a smooth training curve means
    little until it stands apart from this one."""
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1, keepdims=True)
    random = np.random.default_rng(seed)
    directions = random.normal(size=(len(step_lengths), points.shape[1]))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return np.concatenate(
        [points[:1], points[:1] + np.cumsum(step_lengths * directions, axis=0)]
    )
