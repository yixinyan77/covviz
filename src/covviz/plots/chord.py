"""Chord-style visualization for symmetric matrices."""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, pi, sin

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


def plot_chord(
    matrix: np.ndarray,
    *,
    ax=None,
    labels: Sequence[str] | None = None,
    threshold: float = 0.0,
    node_radius: float = 1.0,
    node_size: int = 80,
    positive_color: str = "#2563eb",
    negative_color: str = "#dc2626",
    node_color: str = "#f8fafc",
    ring_color: str = "#334155",
    ring_linewidth: float = 1.4,
    label_color: str = "#111827",
    max_linewidth: float = 5.0,
    alpha: float = 0.68,
    label_offset: float = 1.18,
):
    """Draw a compact chord diagram for a symmetric matrix."""

    if ax is None:
        _, ax = plt.subplots()

    labels = labels or [str(i) for i in range(matrix.shape[0])]
    points = _circle_points(len(labels), node_radius)
    max_weight = _max_offdiag_abs(matrix, threshold)

    ring = Circle(
        (0.0, 0.0),
        radius=node_radius,
        fill=False,
        edgecolor=ring_color,
        linewidth=ring_linewidth,
        alpha=0.85,
        zorder=0,
    )
    ax.add_patch(ring)

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            weight = float(matrix[i, j])
            if abs(weight) <= threshold:
                continue
            linewidth = 0.4 + (abs(weight) / max_weight) * max_linewidth
            patch = FancyArrowPatch(
                points[i],
                points[j],
                connectionstyle="arc3,rad=0.22",
                arrowstyle="-",
                mutation_scale=1,
                linewidth=linewidth,
                color=positive_color if weight >= 0 else negative_color,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(patch)

    xy = np.array(points)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        edgecolors="#334155",
        linewidths=1.0,
        zorder=2,
    )

    label_points = _circle_points(len(labels), label_offset)
    for label, (x, y) in zip(labels, label_points):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=label_color,
            zorder=3,
        )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def _circle_points(count: int, radius: float) -> list[tuple[float, float]]:
    if count == 1:
        return [(0.0, 0.0)]
    return [
        (radius * cos(2 * pi * index / count), radius * sin(2 * pi * index / count))
        for index in range(count)
    ]


def _max_offdiag_abs(matrix: np.ndarray, threshold: float) -> float:
    if matrix.shape[0] < 2:
        return 1.0
    upper = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
    upper = upper[upper > threshold]
    if upper.size == 0:
        return 1.0
    return float(upper.max())
