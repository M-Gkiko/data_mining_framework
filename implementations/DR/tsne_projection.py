from typing import Any
import numpy as np
from sklearn.manifold import TSNE
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure
from core.dimensionality_reduction import DimensionalityReduction


class TSNEProjection(DimensionalityReduction):
    """
    Adapter for sklearn's t-SNE restricted to custom DistanceMeasure implementations.

    This version enforces the use of a DistanceMeasure from the framework
    and disallows sklearn's internal metrics.
    """

    def __init__(self, **kwargs: Any):
        # Fixed to precomputed — we only use custom DistanceMeasures
        self.params = {
            "n_components": 2,
            "perplexity": 30,
            "learning_rate": "auto",
            "n_iter": 1000,
            "metric": "precomputed",  # fixed for our framework
            "random_state": 42,
        }
        # Allow other parameters (but not metric)
        if "metric" in kwargs:
            raise ValueError("Custom metric types are not allowed. Use DistanceMeasure instead.")
        self.params.update(kwargs)

        self.model = None
        self.projection = None

    def _build_distance_matrix(self, X: np.ndarray, distance_measure: DistanceMeasure) -> np.ndarray:
        """Compute full NxN distance matrix using the provided DistanceMeasure."""
        n = X.shape[0]
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = distance_measure.calculate(X[i], X[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def fit_transform(
        self, dataset: Dataset, distance_measure: DistanceMeasure | None = None, **kwargs: Any
    ) -> np.ndarray:
        """Perform t-SNE projection using only custom DistanceMeasure objects."""
        self.params.update(kwargs)

        if distance_measure is None:
            raise ValueError("A DistanceMeasure instance must be provided for TSNEProjection.")

        X = np.asarray(dataset.get_data(), dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one sample.")

        # Compute precomputed distance matrix using our DistanceMeasure
        dist_matrix = self._build_distance_matrix(X, distance_measure)

        # Always use precomputed metric
        self.model = TSNE(**self.params)
        self.projection = self.model.fit_transform(dist_matrix)

        return np.asarray(self.projection)
