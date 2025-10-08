import numpy as np
from core.distance_measure import DistanceMeasure


def build_distance_matrix(X: np.ndarray, distance_measure: DistanceMeasure) -> np.ndarray:
    """
    Compute a full NxN distance matrix using the provided DistanceMeasure.

    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features)
        distance_measure (DistanceMeasure): Custom distance measure object
            implementing the `calculate(point1, point2)` method.

    Returns:
        np.ndarray: Symmetric distance matrix (n_samples x n_samples)
    """
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = distance_measure.calculate(X[i], X[j])
            D[i, j] = d
            D[j, i] = d

    return D
