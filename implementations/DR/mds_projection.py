from typing import Any
import numpy as np
from sklearn.manifold import MDS
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure
from core.dimensionality_reduction import DimensionalityReduction
from utils.distance_utils import build_distance_matrix


class MDSProjection(DimensionalityReduction):
    """
    Adapter for sklearn's Multidimensional Scaling (MDS)
    restricted to custom DistanceMeasure implementations.

    - Uses our DistanceMeasure to build a precomputed distance matrix.
    - Returns a 2D numpy array (rows = samples, columns = components).
    """

    def __init__(self, **kwargs: Any):
        # Default parameters for MDS
        self.params = {
            "n_components": 2,
            "dissimilarity": "precomputed",
            "random_state": 42,
            "max_iter": 300,
            "n_init": 4,
        }

        # Prevent overriding dissimilarity
        if "dissimilarity" in kwargs:
            raise ValueError("Custom 'dissimilarity' not allowed. Use DistanceMeasure instead.")
        self.params.update(kwargs)

        self.model = None
        self.projection = None

    def fit_transform(
        self, dataset: Dataset, distance_measure: DistanceMeasure | None = None, **kwargs: Any
    ) -> np.ndarray:
        """
        Perform MDS projection using only a custom DistanceMeasure.
        """
        self.params.update(kwargs)

        if distance_measure is None:
            raise ValueError("A DistanceMeasure instance must be provided for MDSProjection.")

        X = np.asarray(dataset.get_data(), dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one sample.")

        # Compute distance matrix via shared utility
        dist_matrix = build_distance_matrix(X, distance_measure)

        # Fit and transform with sklearn's MDS
        self.model = MDS(**self.params)
        self.projection = self.model.fit_transform(dist_matrix)

        return np.asarray(self.projection)
