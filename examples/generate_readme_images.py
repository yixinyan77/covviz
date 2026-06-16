"""Generate gallery images used by the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import covviz as cv


def make_covariance(seed: int, size: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(size, size))
    cov = base @ base.T
    scale = np.sqrt(np.diag(cov))
    corr = cov / np.outer(scale, scale)
    corr[np.abs(corr) < 0.18] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(exist_ok=True)

    matrix = make_covariance(11)
    labels = [f"V{i}" for i in range(matrix.shape[0])]
    plot_specs = [
        ("heatmap", {"triangle": "upper", "annotate": True, "colorbar": False}),
        ("chord", {"threshold": 0.2}),
        ("network", {"threshold": 0.22, "layout": "circular"}),
        ("contour", {"levels": 12, "colorbar": False}),
        ("sparsity", {"threshold": 0.2}),
    ]

    for kind, kwargs in plot_specs:
        fig, _ = cv.plot(matrix, kind=kind, labels=labels, figsize=(5, 5), **kwargs)
        fig.savefig(assets / f"{kind}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(17, 3.6))
    for ax, (kind, kwargs) in zip(axes, plot_specs):
        cv.plot(matrix, kind=kind, labels=labels, ax=ax, **kwargs)
        ax.set_title(kind)
    fig.tight_layout()
    fig.savefig(assets / "gallery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
