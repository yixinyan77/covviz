"""Contour visualization for symmetric matrices."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def plot_contour(
    matrix: np.ndarray,
    *,
    ax=None,
    labels: Sequence[str] | None = None,
    levels: int | Sequence[float] = 12,
    cmap: str = "coolwarm",
    center: float | None = 0.0,
    filled: bool = True,
    colorbar: bool = True,
    linewidths: float = 0.8,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Draw matrix values as a 2D contour map."""

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    xx, yy = np.meshgrid(x, y)

    norm = None
    if center is not None:
        low = float(np.nanmin(matrix)) if vmin is None else vmin
        high = float(np.nanmax(matrix)) if vmax is None else vmax
        if low < center < high:
            norm = TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)

    contour_func = ax.contourf if filled else ax.contour
    contours = contour_func(
        xx,
        yy,
        matrix,
        levels=levels,
        cmap=cmap,
        norm=norm,
        linewidths=None if filled else linewidths,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
    )

    if filled:
        ax.contour(xx, yy, matrix, levels=levels, colors="#111827", linewidths=0.25, alpha=0.28)

    if labels is not None:
        ax.set_xticks(x, labels=labels, rotation=45, ha="right")
        ax.set_yticks(y, labels=labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.max(), y.min())
    ax.set_aspect("equal")
    ax.tick_params(length=0)

    if colorbar:
        ax.figure.colorbar(contours, ax=ax, fraction=0.046, pad=0.04)
    return ax
