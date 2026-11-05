"""A run's figures, drawn with matplotlib (the ``manifolds`` extra)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ajax.representation import embeddings, run_data

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#e1e0d9"
#: Blue, light to dark, for a magnitude; its lightest step clears 2:1 on SURFACE.
SEQUENTIAL = ("#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
#: Blue to red through a neutral grey, for a value with a sign.
DIVERGING = ("#184f95", "#3987e5", "#f0efec", "#e66767", "#a83232")
#: One hue per layer, in a fixed order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
METRICS = ("effective_rank", "cka_previous", "activation_norm", "dead_fraction")
METRIC_TITLES = {
    "effective_rank": "Effective rank",
    "cka_previous": "CKA with the previous probe",
    "activation_norm": "Mean activation norm",
    "dead_fraction": "Dead units",
}
RETURN = "Eval/episodic mean reward"


def colour_map(colours: tuple[str, ...]) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ramp", list(colours))


def new_figure(n_rows: int, n_columns: int, width: float, height: float) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(width, height), squeeze=False, facecolor=SURFACE
    )
    for axis in axes.flat:
        axis.set_facecolor(SURFACE)
        axis.tick_params(colors=MUTED, labelsize=8, length=0)
        for side, spine in axis.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(GRID)
    return figure, axes


def bare(axis: Any) -> None:
    """An embedding's axes mean nothing: no ticks, no frame."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def title(axis: Any, text: str) -> None:
    axis.set_title(text, loc="left", color=INK, fontsize=10)


def save(figure: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def short_name(layer: str) -> str:
    return layer.replace("/ensemble", "").replace("/encoder/network", "")


def add_colour_bar(axis: Any, drawn: Any) -> None:
    bar = axis.figure.colorbar(drawn, ax=axis, fraction=0.04, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    bar.outline.set_visible(False)


def scatter(axis: Any, points: np.ndarray, colour: np.ndarray, ramp: Any) -> Any:
    """Thin points, in random order: drawn by probe, the last would hide the
    others."""
    order = np.random.default_rng(0).permutation(len(points))
    drawn = axis.scatter(
        points[order, 0],
        points[order, 1],
        c=colour[order],
        cmap=ramp,
        s=3,
        linewidths=0,
        rasterized=True,
    )
    bare(axis)
    return drawn


def manifold_figure(
    directory: str | os.PathLike,
    layer: str,
    path: Path,
    colourings: dict[str, np.ndarray],
    max_probes: int = 12,
    max_states: int = 1024,
    seed: int = 0,
) -> None:
    """One layer's states through training, on an Aligned UMAP: the whole run
    coloured by probe, by value, where the agent has a critic, and by
    ``colourings`` (per state, name to values), then six of its probes."""
    activations = run_data.load_layer(directory, layer, seed)
    steps = run_data.load_steps(directory, seed)
    values = run_data.load_values(directory, seed)
    probes = embeddings.evenly_spaced(len(activations), max_probes)
    states = embeddings.evenly_spaced(activations.shape[1], max_states)
    selected = np.asarray(activations[probes][:, states], np.float64)
    embedded = embeddings.aligned_umap(selected, seed=seed)
    n_probes, n_states = embedded.shape[:2]
    flat = embedded.reshape(n_probes * n_states, 2)
    probe_steps = np.repeat(steps[probes], n_states)

    panels = {"Env steps": (probe_steps, SEQUENTIAL)}
    # What the small multiples show a probe by: its own value where it has
    # one, else the probe's position on the first colouring.
    per_probe = np.asarray(next(iter(colourings.values())))[states]
    per_probe = np.tile(per_probe, (n_probes, 1))
    per_probe_name = next(iter(colourings), "")
    if values is not None:
        per_probe = values[probes][:, states]
        per_probe_name = "value"
        panels["Critic's value"] = (per_probe.ravel(), SEQUENTIAL)
    for name, field in colourings.items():
        panels[name] = (np.tile(np.asarray(field)[states], n_probes), DIVERGING)

    n_columns = max(len(panels), 6)
    figure, axes = new_figure(2, n_columns, 2.9 * n_columns, 5.4)
    for axis, (name, (colour, ramp)) in zip(axes[0], panels.items(), strict=False):
        add_colour_bar(axis, scatter(axis, flat, colour, colour_map(ramp)))
        title(axis, name)
    for axis in axes[0][len(panels) :]:
        axis.set_visible(False)
    shown = embeddings.evenly_spaced(n_probes, 6)
    for axis, probe in zip(axes[1], shown, strict=False):
        drawn = scatter(axis, embedded[probe], per_probe[probe], colour_map(SEQUENTIAL))
        drawn.set_clim(per_probe.min(), per_probe.max())
        title(axis, f"Step {steps[probes[probe]]:,}, by {per_probe_name}")
    add_colour_bar(axes[1][len(shown) - 1], drawn)
    for axis in axes[1][len(shown) :]:
        axis.set_visible(False)
    figure.suptitle(f"{layer}, seed {seed}", x=0.01, ha="left", color=MUTED, fontsize=9)
    save(figure, path)


def units_figure(
    directory: str | os.PathLike,
    layers: list[str],
    path: Path,
    max_nodes: int = 20_000,
    seed: int = 0,
) -> None:
    """A network's hidden units through training, on M-PHATE, by env step and
    by layer. Each unit is its activations over the probe set, so two units
    that respond alike sit together whatever their weights."""
    per_layer = [run_data.load_layer(directory, layer, seed) for layer in layers]
    trace = np.concatenate([a.transpose(0, 2, 1) for a in per_layer], axis=1)
    layer_of_unit = np.repeat(
        np.arange(len(layers)), [activations.shape[2] for activations in per_layer]
    )
    steps = run_data.load_steps(directory, seed)
    n_probes, n_units_total = trace.shape[:2]
    probes, units = embeddings.fit_to_nodes(n_probes, n_units_total, max_nodes, seed)
    selected = np.asarray(trace[probes][:, units], np.float64)
    embedded = embeddings.mphate(selected, seed)
    n_units = len(units)

    figure, axes = new_figure(1, 2, 10.0, 4.4)
    by_step = np.repeat(steps[probes], n_units)
    drawn = scatter(axes[0, 0], embedded, by_step, colour_map(SEQUENTIAL))
    add_colour_bar(axes[0, 0], drawn)
    title(
        axes[0, 0], f"M-PHATE of {n_units} units at {len(probes)} probes, by env step"
    )
    layer_of_node = np.tile(layer_of_unit[units], len(probes))
    for index, layer in enumerate(layers):
        chosen = layer_of_node == index
        axes[0, 1].scatter(
            embedded[chosen, 0],
            embedded[chosen, 1],
            s=3,
            linewidths=0,
            color=CATEGORICAL[index % len(CATEGORICAL)],
            label=short_name(layer),
            rasterized=True,
        )
    bare(axes[0, 1])
    legend = axes[0, 1].legend(frameon=False, fontsize=8, markerscale=3)
    for text in legend.get_texts():
        text.set_color(MUTED)
    title(axes[0, 1], "By layer")
    save(figure, path)


def weights_figure(
    directory: str | os.PathLike, network: str, path: Path, seed: int = 0
) -> None:
    """A network's weights through training, on their two principal
    components, beside a random walk of the same step lengths."""
    weights = run_data.load_parameters(directory, network, seed)
    walk = embeddings.matched_random_walk(weights, seed)
    trajectories = {"Training": weights, "Random walk, same step lengths": walk}
    figure, axes = new_figure(1, 2, 9.0, 4.2)
    for axis, (name, trajectory) in zip(axes[0], trajectories.items(), strict=False):
        path_2d, shares = embeddings.principal_path(trajectory)
        axis.plot(path_2d[:, 0], path_2d[:, 1], color=CATEGORICAL[0], linewidth=2)
        axis.plot(*path_2d[0], "o", color=MUTED, markersize=8)
        axis.annotate(
            "start",
            path_2d[0],
            color=MUTED,
            fontsize=8,
            xytext=(6, 6),
            textcoords="offset points",
        )
        axis.grid(color=GRID, linewidth=0.5)
        axis.set_xlabel(f"PC1, {shares[0]:.0%} of variance", color=MUTED, fontsize=8)
        axis.set_ylabel(f"PC2, {shares[1]:.0%}", color=MUTED, fontsize=8)
        title(axis, f"{name}: {network}, seed {seed}")
    save(figure, path)


def plot_across_seeds(axis: Any, steps: Any, seeds: np.ndarray, colour: str) -> None:
    """Each seed thin, their mean thick; ``seeds`` is ``(n_seeds, n_points)``."""
    axis.plot(np.transpose(steps), seeds.T, color=colour, linewidth=1, alpha=0.3)
    mean_steps = steps if np.ndim(steps) == 1 else steps[0]
    # Masked rather than nanmean: the first probe's CKA is NaN on every seed.
    mean = np.ma.masked_invalid(seeds).mean(axis=0).filled(np.nan)
    axis.plot(mean_steps, mean, color=colour, linewidth=2)


def metrics_figure(directory: str | os.PathLike, path: Path) -> None:
    """Every layer's metrics against env steps, one panel per metric and
    layer. The return above, when the run kept its evaluations
    (:func:`ajax.representation.run_data.save_evaluations`), on an axis of
    its own: two scales on one axis would make their crossings look like a
    fact."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    metrics = run_data.load_metrics(directory)
    layers = run_data.layer_names(directory)
    steps = run_data.load_steps(directory)
    evaluations = run_data.load_evaluations(directory)
    returns_shown = int(evaluations is not None)
    n_rows = len(METRICS) + returns_shown
    figure, axes = new_figure(n_rows, len(layers), 3.0 * len(layers), 2.0 * n_rows)
    if evaluations is not None:
        plot_across_seeds(
            axes[0, 0], evaluations["timestep"], evaluations[RETURN], CATEGORICAL[1]
        )
        title(axes[0, 0], "Return")
        for axis in axes[0, 1:]:
            axis.set_visible(False)
    for column, layer in enumerate(layers):
        for row, metric in enumerate(METRICS, start=returns_shown):
            axis = axes[row, column]
            seeds = metrics[f"Representation/{layer}/{metric}"]
            plot_across_seeds(axis, steps, seeds, CATEGORICAL[0])
            heading = METRIC_TITLES[metric]
            title(
                axis,
                heading if row > returns_shown else f"{short_name(layer)}\n{heading}",
            )
            if metric == "dead_fraction":
                axis.set_ylim(bottom=0)
    for axis in axes.flat:
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    for axis in axes[-1]:
        axis.set_xlabel("Env steps", color=MUTED, fontsize=8)
    save(figure, path)
