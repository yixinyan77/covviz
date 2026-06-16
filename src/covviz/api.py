"""Public plotting API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from .layout import apply_titles, make_axes
from .plots.chord import plot_chord
from .plots.contour import plot_contour
from .plots.heatmap import plot_heatmap
from .plots.network import plot_network
from .plots.sparsity import plot_sparsity
from .validation import MatrixData, MatrixInputError, normalize_matrices

PlotFunc = Callable[..., object]

PLOTTERS: dict[str, PlotFunc] = {
    "heatmap": plot_heatmap,
    "chord": plot_chord,
    "contour": plot_contour,
    "network": plot_network,
    "sparsity": plot_sparsity,
}


def plot(
    matrices,
    *,
    kind: str = "heatmap",
    labels: Sequence[object] | Sequence[Sequence[object]] | None = None,
    titles: Sequence[str] | None = None,
    ax=None,
    ncols: int | str = "auto",
    figsize: tuple[float, float] | str | None = "auto",
    check_symmetric: bool = True,
    atol: float = 1e-8,
    share_scale: bool = True,
    **kwargs,
):
    """Plot one or more symmetric matrices.

    Returns ``(fig, axes)`` for both single- and multi-matrix inputs. ``axes`` is
    always a flat list, which makes downstream code predictable.
    """

    normalized = normalize_matrices(
        matrices, labels=labels, check_symmetric=check_symmetric, atol=atol
    )
    plotter = _get_plotter(kind)

    if ax is not None and len(normalized) != 1:
        raise MatrixInputError("Pass ax only when plotting a single matrix.")
    if ax is not None:
        plotter(
            normalized[0].matrix,
            ax=ax,
            labels=normalized[0].labels,
            **_scale_kwargs(normalized, kind, share_scale, kwargs),
            **kwargs,
        )
        if titles:
            ax.set_title(str(titles[0]))
        return ax.figure, [ax]

    return plot_grid(
        normalized,
        kind=kind,
        titles=titles,
        ncols=ncols,
        figsize=figsize,
        share_scale=share_scale,
        **kwargs,
    )


def plot_grid(
    matrices,
    *,
    kind: str = "heatmap",
    titles: Sequence[str] | None = None,
    ncols: int | str = "auto",
    figsize: tuple[float, float] | str | None = "auto",
    share_scale: bool = True,
    **kwargs,
):
    """Plot normalized or raw matrices in an adaptive subplot grid."""

    normalized = _ensure_normalized(matrices)
    plotter = _get_plotter(kind)
    fig, axes = make_axes(len(normalized), ncols=ncols, figsize=figsize)
    shared_kwargs = _scale_kwargs(normalized, kind, share_scale, kwargs)

    for ax, item in zip(axes, normalized):
        plotter(item.matrix, ax=ax, labels=item.labels, **shared_kwargs, **kwargs)

    apply_titles(axes, titles)
    fig.tight_layout()
    return fig, list(axes)


def _get_plotter(kind: str) -> PlotFunc:
    try:
        return PLOTTERS[kind]
    except KeyError as exc:
        options = ", ".join(sorted(PLOTTERS))
        raise MatrixInputError(f"Unknown plot kind {kind!r}. Choose from: {options}.") from exc


def _ensure_normalized(matrices) -> list[MatrixData]:
    if isinstance(matrices, list) and matrices and isinstance(matrices[0], MatrixData):
        return matrices
    return normalize_matrices(matrices)


def _scale_kwargs(
    normalized: Sequence[MatrixData],
    kind: str,
    share_scale: bool,
    user_kwargs: dict,
) -> dict:
    if not share_scale or kind not in {"heatmap", "contour"}:
        return {}
    if "vmin" in user_kwargs or "vmax" in user_kwargs:
        return {}
    max_abs = max(float(abs(item.matrix).max()) for item in normalized)
    return {"vmin": -max_abs, "vmax": max_abs}
