import numpy as np
from core.dataset import Dataset
from core.dr_quality_measure import DRQualityMeasure


class Continuity(DRQualityMeasure):
    """
    Measures how well neighborhood relationships from the original space
    are preserved in the reduced space (complementary to Trustworthiness).

    Range: [0, 1]
    Higher is better.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def evaluate(self, dataset: Dataset, reduced_data: np.ndarray) -> float:
        X = np.asarray(dataset.get_data(), dtype=float)
        Y = np.asarray(reduced_data, dtype=float)

        n = X.shape[0]
        k = self.n_neighbors

        # Get ranks of neighbors in original and reduced spaces
        dist_X = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        dist_Y = np.linalg.norm(Y[:, np.newaxis] - Y[np.newaxis, :], axis=2)

        # argsort gives sorted indices (neighbors)
        ranks_X = np.argsort(dist_X, axis=1)
        ranks_Y = np.argsort(dist_Y, axis=1)

        # Continuity: proportion of shared neighbors in top-k
        continuity_sum = 0
        for i in range(n):
            # Neighbors in reduced space but not in original
            U_k_i = set(ranks_Y[i, 1:k + 1]) - set(ranks_X[i, 1:k + 1])
            continuity_sum += sum((k - (np.where(ranks_X[i] == j)[0][0] - k)) for j in U_k_i)

        # Normalize score
        continuity_score = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * continuity_sum
        return float(max(0, min(1, continuity_score)))  # clamp to [0, 1]
