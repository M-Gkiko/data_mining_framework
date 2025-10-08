import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from core.dataset import Dataset
from core.dr_quality_measure import DRQualityMeasure


class Trustworthiness(DRQualityMeasure):
    """
    Measures how well local neighborhoods are preserved after dimensionality reduction.

    Based on sklearn.manifold.trustworthiness:
        Trustworthiness ∈ [0, 1]
        Higher is better (1.0 = perfect preservation)
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def evaluate(self, dataset: Dataset, reduced_data: np.ndarray) -> float:
        X = np.asarray(dataset.get_data(), dtype=float)
        score = sklearn_trustworthiness(X, reduced_data, n_neighbors=self.n_neighbors)
        return float(score)
