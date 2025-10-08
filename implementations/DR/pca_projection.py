from typing import Any
import numpy as np
from sklearn.decomposition import PCA
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure
from core.dimensionality_reduction import DimensionalityReduction


class PCAProjection(DimensionalityReduction):
    """
    - Accepts a Dataset and optional DistanceMeasure (for interface consistency).
    - Returns a 2D numpy array (rows = samples, columns = components).
    - Hyperparameters (like n_components) are passed dynamically via kwargs.
    """

    def __init__(self, **kwargs: Any):
        # Default hyperparameters
        self.params = {"n_components": 2}
        self.params.update(kwargs)
        self.model = None
        self.projection = None

    def fit_transform(
        self, dataset: Dataset, distance_measure: "DistanceMeasure | None" = None, **kwargs: Any
    ) -> np.ndarray:
        """
        Reduce the dimensionality of the given dataset using PCA.

        Args:
            dataset (Dataset): The dataset to project.
            distance_measure (DistanceMeasure, optional): Included for
                consistency, but not used by PCA.
            **kwargs: Optional hyperparameters (e.g., n_components).

        Returns:
            np.ndarray: 2D array of projected data (n_samples x n_components).
        """
        # Update parameters dynamically
        self.params.update(kwargs)

        X = dataset.get_data()
        if X is None or len(X) == 0:
            raise ValueError("Dataset is empty or invalid.")

        # Ensure numpy array format
        X = np.asarray(X, dtype=float)

        # Create and fit PCA model
        self.model = PCA(n_components=self.params["n_components"])
        self.projection = self.model.fit_transform(X)

        return np.asarray(self.projection)
