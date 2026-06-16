import matplotlib.pyplot as plt
import numpy as np

import covviz as cv


def make_covariance(seed: int, size: int = 8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size, size))
    return x @ x.T


labels = [f"V{i}" for i in range(8)]
matrices = [make_covariance(seed) for seed in range(3)]

cv.plot(matrices[0], kind="heatmap", labels=labels)
cv.plot(matrices[0], kind="heatmap", labels=labels, triangle="upper", annotate=True)
cv.plot(matrices, kind="chord", labels=labels, titles=["A", "B", "C"])
cv.plot(matrices[0], kind="network", labels=labels, threshold=1.0)
cv.plot(matrices[0], kind="contour", labels=labels, levels=14)
cv.plot(matrices[0], kind="sparsity", labels=labels, threshold=1.0)

plt.show()
