"""Sparsity-pattern visualization for symmetric matrices."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_sparsity(
    matrix: np.ndarray,
    *,
    ax=None,
    labels: Sequence[str] | None = None,
    threshold: float = 0.0,
    include_diagonal: bool = False,
    positive_color: str = "#2563eb",
    negative_color: str = "#dc2626",
    marker_size: float = 36,
):
    """Draw the thresholded sparsity pattern of a symmetric matrix."""

    if ax is None:
        _, ax = plt.subplots()

    mask = np.abs(matrix) > threshold
    if not include_diagonal:
        np.fill_diagonal(mask, False)

    rows, cols = np.where(mask)
    values = matrix[rows, cols]
    positive = values >= 0

    ax.scatter(
        cols[positive],
        rows[positive],
        s=marker_size,
        marker="s",
        color=positive_color,
        alpha=0.82,
        edgecolors="none",
    )
    ax.scatter(
        cols[~positive],
        rows[~positive],
        s=marker_size,
        marker="s",
        color=negative_color,
        alpha=0.82,
        edgecolors="none",
    )

    size = matrix.shape[0]
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(color="#e5e7eb", linewidth=0.6)

    if labels is not None:
        ticks = np.arange(size)
        ax.set_xticks(ticks, labels=labels, rotation=45, ha="right")
        ax.set_yticks(ticks, labels=labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return ax
