"""Covariance graph visualization inspired by sparse covariance demos."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_covariance_graph(
    matrix: np.ndarray,
    *,
    ax=None,
    labels: Sequence[str] | None = None,
    threshold: float = 0.0,
    layout: str = "spring",
    seed: int = 42,
    k: float | None = 0.7,
    node_size: int = 36,
    node_color: str = "#2563eb",
    positive_edge_color: str = "#2563eb",
    negative_edge_color: str = "#dc2626",
    edge_width: float = 0.55,
    alpha_scale: float = 1.0,
    with_labels: bool = False,
):
    """Draw the covariance graph used in covariance-estimation comparisons.

    Unlike ``plot_network``, this view keeps the original notebook style:
    compact blue nodes, thin signed edges, and edge opacity proportional to the
    absolute matrix entry.
    """

    if ax is None:
        _, ax = plt.subplots()

    labels = labels or [str(i) for i in range(matrix.shape[0])]
    graph = nx.Graph()
    graph.add_nodes_from(labels)

    max_weight = _max_offdiag_abs(matrix, threshold)
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            weight = float(matrix[i, j])
            if abs(weight) <= threshold:
                continue
            graph.add_edge(
                labels[i],
                labels[j],
                weight=weight,
                alpha=min(abs(weight) / max_weight * alpha_scale, 1.0),
            )

    positions = _layout_positions(graph, layout=layout, seed=seed, k=k)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_size,
        node_color=node_color,
        linewidths=0,
        ax=ax,
    )

    for u, v, data in graph.edges(data=True):
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(u, v)],
            width=edge_width,
            edge_color=positive_edge_color if data["weight"] >= 0 else negative_edge_color,
            alpha=data["alpha"],
            ax=ax,
        )

    if with_labels:
        nx.draw_networkx_labels(graph, positions, font_size=8, ax=ax)

    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def _layout_positions(graph: nx.Graph, *, layout: str, seed: int, k: float | None):
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, k=k, weight="weight")
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph, weight="weight")
    raise ValueError("layout must be one of: 'spring', 'circular', 'kamada_kawai'.")


def _max_offdiag_abs(matrix: np.ndarray, threshold: float) -> float:
    if matrix.shape[0] < 2:
        return 1.0
    upper = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
    upper = upper[upper > threshold]
    if upper.size == 0:
        return 1.0
    return float(upper.max())
