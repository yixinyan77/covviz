"""Network visualization for symmetric matrices."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_network(
    matrix: np.ndarray,
    *,
    ax=None,
    labels: Sequence[str] | None = None,
    threshold: float = 0.0,
    absolute_threshold: bool = True,
    layout: str = "spring",
    seed: int = 7,
    node_size: int = 420,
    node_color: str = "#f8fafc",
    positive_edge_color: str = "#2563eb",
    negative_edge_color: str = "#dc2626",
    edge_alpha: float = 0.72,
    max_linewidth: float = 4.0,
    with_labels: bool = True,
):
    """Draw matrix entries as a weighted undirected network."""

    if ax is None:
        _, ax = plt.subplots()

    labels = labels or [str(i) for i in range(matrix.shape[0])]
    graph = _matrix_to_graph(matrix, labels, threshold, absolute_threshold)
    positions = _layout_positions(graph, layout=layout, seed=seed)

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors="#334155",
        linewidths=1.0,
    )

    if graph.number_of_edges():
        weights = np.array([abs(data["weight"]) for _, _, data in graph.edges(data=True)])
        max_weight = weights.max() or 1.0
        widths = 0.5 + (weights / max_weight) * max_linewidth
        positive_edges = [
            (u, v) for u, v, data in graph.edges(data=True) if data["weight"] >= 0
        ]
        negative_edges = [
            (u, v) for u, v, data in graph.edges(data=True) if data["weight"] < 0
        ]
        widths_by_edge = dict(zip(graph.edges(), widths))
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=positive_edges,
            width=[widths_by_edge[edge] for edge in positive_edges],
            edge_color=positive_edge_color,
            alpha=edge_alpha,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=negative_edges,
            width=[widths_by_edge[edge] for edge in negative_edges],
            edge_color=negative_edge_color,
            alpha=edge_alpha,
            style="dashed",
            ax=ax,
        )

    if with_labels:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9)

    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def _matrix_to_graph(
    matrix: np.ndarray,
    labels: Sequence[str],
    threshold: float,
    absolute_threshold: bool,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            weight = float(matrix[i, j])
            score = abs(weight) if absolute_threshold else weight
            if score > threshold:
                graph.add_edge(labels[i], labels[j], weight=weight)
    return graph


def _layout_positions(graph: nx.Graph, *, layout: str, seed: int):
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, weight="weight")
    raise ValueError("layout must be one of: 'spring', 'circular', 'kamada_kawai'.")
