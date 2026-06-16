"""Adaptive visualization tools for symmetric covariance-like matrices."""

from .api import plot, plot_grid
from .validation import MatrixInputError, validate_symmetric_matrix

__all__ = [
    "MatrixInputError",
    "plot",
    "plot_grid",
    "validate_symmetric_matrix",
]

__version__ = "0.1.0"
