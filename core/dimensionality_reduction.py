from abc import ABC, abstractmethod
from typing import Any, Optional
from .dataset import Dataset
from .distance_measure import DistanceMeasure
import numpy as np


class DimensionalityReduction(ABC):
    """
    Abstract base class for dimensionality reduction / projection techniques.
    """

    @abstractmethod
    def fit_transform(self, dataset: Dataset, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any) -> np.ndarray:
        """
        Reduce the dimensionality of the given dataset.

        Args:
            dataset (Dataset): The input dataset.
            distance_measure (Optional[DistanceMeasure]): An optional distance measure to use.
            **kwargs: Algorithm-specific hyperparameters.

        Returns:
            np.ndarray: The reduced 2D array of data points.
        """
        pass
