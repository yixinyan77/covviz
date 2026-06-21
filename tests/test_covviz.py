import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import covviz as cv
from covviz.layout import auto_grid
from covviz.validation import MatrixInputError


def spd(size=5):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(size, size))
    return x @ x.T


def test_plot_single_heatmap_returns_flat_axes():
    fig, axes = cv.plot(spd(), kind="heatmap", colorbar=False)

    assert fig is axes[0].figure
    assert len(axes) == 1


def test_plot_multiple_chord_uses_all_axes():
    matrices = [spd(4), spd(4), spd(4)]

    fig, axes = cv.plot(matrices, kind="chord", titles=["a", "b", "c"])

    assert fig is axes[0].figure
    assert len(axes) == 3
    assert [ax.get_title() for ax in axes] == ["a", "b", "c"]


def test_network_plot_accepts_threshold():
    fig, axes = cv.plot(spd(5), kind="network", threshold=0.2)

    assert fig is axes[0].figure


def test_covariance_graph_plot_accepts_threshold():
    fig, axes = cv.plot(spd(5), kind="covariance_graph", threshold=0.2)

    assert fig is axes[0].figure


def test_heatmap_accepts_triangle_and_annotations():
    fig, axes = cv.plot(spd(4), kind="heatmap", triangle="upper", annotate=True, colorbar=False)

    assert fig is axes[0].figure
    assert axes[0].texts


def test_contour_plot_accepts_levels():
    fig, axes = cv.plot(spd(5), kind="contour", levels=8, colorbar=False)

    assert fig is axes[0].figure


def test_sparsity_plot_accepts_threshold():
    fig, axes = cv.plot(spd(5), kind="sparsity", threshold=0.5)

    assert fig is axes[0].figure


def test_covgraph_alias():
    fig, axes = cv.plot(spd(5), kind="covgraph", threshold=0.2)

    assert fig is axes[0].figure


def test_dataframe_input_uses_index_labels():
    labels = ["a", "b", "c"]
    frame = pd.DataFrame(spd(3), index=labels, columns=labels)

    _, axes = cv.plot(frame, kind="heatmap", colorbar=False)

    assert [tick.get_text() for tick in axes[0].get_xticklabels()] == labels


def test_rejects_unknown_kind():
    with pytest.raises(MatrixInputError, match="Unknown plot kind"):
        cv.plot(spd(3), kind="unknown")


def test_rejects_non_symmetric_matrix():
    matrix = np.array([[1.0, 2.0], [3.0, 1.0]])

    with pytest.raises(MatrixInputError, match="symmetric"):
        cv.plot(matrix)


def test_rejects_mismatched_shapes_in_grid():
    with pytest.raises(MatrixInputError, match="share shape"):
        cv.plot([spd(3), spd(4)])


def test_auto_grid_shapes():
    assert auto_grid(1) == (1, 1)
    assert auto_grid(4) == (2, 2)
    assert auto_grid(5) == (2, 3)
