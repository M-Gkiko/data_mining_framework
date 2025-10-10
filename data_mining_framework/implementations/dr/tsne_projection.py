from typing import Any, Optional
import numpy as np
from sklearn.manifold import TSNE
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...core.dimensionality_reduction import DimensionalityReduction
from ...utils.distance_utils import build_distance_matrix


class TSNEProjection(DimensionalityReduction):
    """
    Adapter for sklearn's t-SNE restricted to custom DistanceMeasure implementations.

    - Uses only DistanceMeasure from our framework (no built-in sklearn metrics).
    - Returns a 2D numpy array (rows = samples, columns = components).
    """

    def __init__(self, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any):
        # Store distance measure
        self.distance_measure = distance_measure
        
        # Force custom distance computation
        self.params = {
            "n_components": 2,
            "perplexity": 30,
            "learning_rate": "auto",
            "max_iter": 1000,
            "metric": "precomputed",
            "init": "random",  # Required when using precomputed distance matrix
            "random_state": 42,
        }

        # Prevent overriding metric
        if "metric" in kwargs:
            raise ValueError("Custom 'metric' not allowed. Use DistanceMeasure instead.")
        self.params.update(kwargs)

        self.model = None
        self.projection = None

    def fit_transform(self, dataset: Dataset, **kwargs: Any) -> np.ndarray:
        """
        Perform t-SNE projection using a custom DistanceMeasure.
        """
        self.params.update(kwargs)

        if self.distance_measure is None:
            raise ValueError("A DistanceMeasure instance must be provided for TSNEProjection in constructor.")

        X = np.asarray(dataset.get_data(), dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one sample.")

        # Build pairwise distance matrix using our shared utility
        dist_matrix = build_distance_matrix(X, self.distance_measure)

        # Fit and transform with sklearn's t-SNE
        self.model = TSNE(**self.params)
        self.projection = self.model.fit_transform(dist_matrix)

        return np.asarray(self.projection)
