# covviz

`covviz` is a small Python package for adaptive visualization of symmetric
covariance-like matrices. It accepts one matrix or many matrices and returns a
Matplotlib figure with either one panel or an automatically arranged grid.

## Install

From this repository:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import covviz as cv

rng = np.random.default_rng(0)
x = rng.normal(size=(8, 8))
cov = x @ x.T

fig, axes = cv.plot(cov, kind="heatmap")
fig.savefig("covariance-heatmap.png", dpi=200)
```

Multiple matrices are arranged automatically:

```python
matrices = [cov, cov * 0.7, cov * 1.2]

fig, axes = cv.plot(
    matrices,
    kind="chord",
    labels=[f"V{i}" for i in range(8)],
    titles=["A", "B", "C"],
)
```

Network view:

```python
cv.plot(cov, kind="network", threshold=0.25)
```

Annotated upper-triangle heatmap:

```python
cv.plot(cov, kind="heatmap", triangle="upper", annotate=True, colorbar=False)
```

Contour and sparsity views:

```python
cv.plot(cov, kind="contour", levels=16)
cv.plot(cov, kind="sparsity", threshold=0.5)
```

## Plot Types

The current MVP supports:

- `heatmap`: signed matrix heatmap centered on zero by default.
- `chord`: circular chord-style view of off-diagonal relationships.
- `network`: weighted undirected network from off-diagonal relationships.
- `contour`: filled or line contour view of the matrix surface.
- `sparsity`: thresholded signed sparsity pattern.

## Input Rules

Inputs may be:

- A single `numpy.ndarray`.
- A list of arrays with the same shape.
- A pandas `DataFrame` whose index and columns match.

Matrices must be square and symmetric. Non-finite values are rejected. For
covariance matrices this catches common data issues early, before a plot is
created.

## API

```python
fig, axes = cv.plot(
    matrices,
    kind="heatmap",
    labels=None,
    titles=None,
    ncols="auto",
    figsize="auto",
    check_symmetric=True,
    share_scale=True,
)
```

`axes` is always returned as a flat list, even for a single matrix.

## Roadmap

- Add modularity/community visualization as an optional extra.
- Add richer documentation notebooks.
- Publish to PyPI after the first GitHub release.
