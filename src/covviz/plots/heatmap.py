"""Heatmap visualization for symmetric matrices."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def plot_heatmap(
    matrix: np.ndarray,
    *,
    ax=None,
    labels: Sequence[str] | None = None,
    cmap: str = "coolwarm",
    center: float | None = 0.0,
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    triangle: str | None = None,
    annotate: bool = False,
    fmt: str = ".2g",
    annotation_size: int = 8,
    tick_rotation: float = 45,
    **imshow_kwargs,
):
    """Draw a covariance-style heatmap."""

    if ax is None:
        _, ax = plt.subplots()

    plot_matrix = _mask_triangle(matrix, triangle)
    norm = None
    if center is not None and vmin is not None and vmax is not None and vmin < center < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    image = ax.imshow(
        plot_matrix,
        cmap=cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
        **imshow_kwargs,
    )

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=tick_rotation, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        _annotate_heatmap(ax, plot_matrix, fmt=fmt, size=annotation_size)

    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def _mask_triangle(matrix: np.ndarray, triangle: str | None) -> np.ndarray:
    if triangle is None:
        return matrix
    if triangle not in {"upper", "lower"}:
        raise ValueError("triangle must be one of: None, 'upper', 'lower'.")
    data = matrix.astype(float, copy=True)
    if triangle == "upper":
        data[np.tril_indices_from(data, k=-1)] = np.nan
    else:
        data[np.triu_indices_from(data, k=1)] = np.nan
    return data


def _annotate_heatmap(ax, matrix: np.ndarray, *, fmt: str, size: int) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=size)
