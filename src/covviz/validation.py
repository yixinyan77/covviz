"""Input validation and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class MatrixInputError(ValueError):
    """Raised when a matrix input cannot be plotted safely."""


@dataclass(frozen=True)
class MatrixData:
    """Normalized representation of one matrix and its labels."""

    matrix: np.ndarray
    labels: list[str]


def _is_dataframe_like(value: object) -> bool:
    return all(hasattr(value, attr) for attr in ("values", "index", "columns"))


def _coerce_labels(labels: Sequence[object] | None, size: int) -> list[str]:
    if labels is None:
        return [str(i) for i in range(size)]
    if len(labels) != size:
        raise MatrixInputError(
            f"Expected {size} labels for a {size}x{size} matrix, got {len(labels)}."
        )
    return [str(label) for label in labels]


def validate_symmetric_matrix(
    matrix: object,
    *,
    labels: Sequence[object] | None = None,
    check_symmetric: bool = True,
    atol: float = 1e-8,
    allow_nan: bool = False,
) -> MatrixData:
    """Validate a single square symmetric matrix.

    Parameters
    ----------
    matrix:
        A numpy-like 2D square array. pandas DataFrames are supported without
        requiring pandas as an installation dependency.
    labels:
        Optional labels for rows and columns. DataFrame labels are used when
        labels is omitted.
    check_symmetric:
        Whether to require ``matrix == matrix.T`` within ``atol``.
    atol:
        Absolute tolerance for symmetry checks.
    allow_nan:
        Whether NaN values are accepted.
    """

    inferred_labels = None
    if _is_dataframe_like(matrix):
        inferred_labels = list(matrix.index)
        columns = list(matrix.columns)
        if inferred_labels != columns:
            raise MatrixInputError(
                "DataFrame input must have identical index and column labels."
            )
        matrix = matrix.values

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise MatrixInputError(f"Expected a 2D matrix, got {array.ndim} dimensions.")
    if array.shape[0] != array.shape[1]:
        raise MatrixInputError(f"Expected a square matrix, got shape {array.shape}.")
    if array.shape[0] == 0:
        raise MatrixInputError("Matrix must contain at least one row and one column.")
    if not allow_nan and np.isnan(array).any():
        raise MatrixInputError("Matrix contains NaN values.")
    if np.isinf(array).any():
        raise MatrixInputError("Matrix contains infinite values.")
    if check_symmetric and not np.allclose(array, array.T, atol=atol, equal_nan=allow_nan):
        raise MatrixInputError("Matrix must be symmetric.")

    chosen_labels = labels if labels is not None else inferred_labels
    return MatrixData(matrix=array, labels=_coerce_labels(chosen_labels, array.shape[0]))


def normalize_matrices(
    matrices: object,
    *,
    labels: Sequence[object] | Sequence[Sequence[object]] | None = None,
    check_symmetric: bool = True,
    atol: float = 1e-8,
) -> list[MatrixData]:
    """Normalize either one matrix or an iterable of matrices."""

    if _looks_like_single_matrix(matrices):
        label_set = labels if _labels_for_single(labels) else None
        return [
            validate_symmetric_matrix(
                matrices, labels=label_set, check_symmetric=check_symmetric, atol=atol
            )
        ]

    if not isinstance(matrices, Iterable):
        raise MatrixInputError("Expected a matrix or an iterable of matrices.")

    matrix_list = list(matrices)
    if not matrix_list:
        raise MatrixInputError("Expected at least one matrix.")

    labels_per_matrix = _expand_labels(labels, len(matrix_list))
    normalized = [
        validate_symmetric_matrix(
            matrix,
            labels=labels_per_matrix[index],
            check_symmetric=check_symmetric,
            atol=atol,
        )
        for index, matrix in enumerate(matrix_list)
    ]

    sizes = {item.matrix.shape for item in normalized}
    if len(sizes) != 1:
        raise MatrixInputError("All matrices in a multi-matrix plot must share shape.")
    return normalized


def _looks_like_single_matrix(value: object) -> bool:
    if _is_dataframe_like(value):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.ndim == 2


def _labels_for_single(labels: object) -> bool:
    if labels is None:
        return False
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return False
    if len(labels) == 0:
        return True
    return not isinstance(labels[0], Sequence) or isinstance(labels[0], (str, bytes))


def _expand_labels(
    labels: Sequence[object] | Sequence[Sequence[object]] | None,
    count: int,
) -> list[Sequence[object] | None]:
    if labels is None:
        return [None] * count
    if _labels_for_single(labels):
        return [labels] * count
    if len(labels) != count:
        raise MatrixInputError(
            f"Expected labels for {count} matrices, got {len(labels)} label sets."
        )
    return list(labels)
