"""Figure and axis layout helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def auto_grid(count: int, ncols: int | str = "auto") -> tuple[int, int]:
    """Return a compact grid shape for ``count`` panels."""

    if count < 1:
        raise ValueError("count must be positive.")
    if ncols == "auto":
        columns = math.ceil(math.sqrt(count))
    else:
        columns = int(ncols)
        if columns < 1:
            raise ValueError("ncols must be positive.")
    rows = math.ceil(count / columns)
    return rows, columns


def auto_figsize(rows: int, cols: int, *, panel_size: float = 4.2) -> tuple[float, float]:
    """Compute a readable default figure size."""

    return cols * panel_size, rows * panel_size


def make_axes(
    count: int,
    *,
    ncols: int | str = "auto",
    figsize: tuple[float, float] | str | None = "auto",
    squeeze: bool = False,
):
    """Create a figure and exactly ``count`` visible axes."""

    rows, cols = auto_grid(count, ncols=ncols)
    if figsize == "auto":
        figsize = auto_figsize(rows, cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=squeeze)
    axes_array = np.atleast_1d(axes).ravel()
    for ax in axes_array[count:]:
        ax.set_visible(False)
    return fig, axes_array[:count]


def apply_titles(axes: Sequence, titles: Sequence[str] | None) -> None:
    if titles is None:
        return
    if len(titles) != len(axes):
        raise ValueError(f"Expected {len(axes)} titles, got {len(titles)}.")
    for ax, title in zip(axes, titles):
        ax.set_title(str(title))
