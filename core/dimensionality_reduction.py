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
    def __init__(self, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any) -> None:
        """
        Initialize the dimensionality reduction algorithm with configuration parameters.
        
        Args:
            distance_measure (Optional[DistanceMeasure]): The distance measure to use (default: None)
            **kwargs: Additional algorithm-specific hyperparameters
        """
        pass

    @abstractmethod
    def fit_transform(self, dataset: Dataset, **kwargs: Any) -> np.ndarray:
        """
        Reduce the dimensionality of the given dataset.

        Args:
            dataset (Dataset): The input dataset.
            **kwargs: Algorithm-specific hyperparameters.

        Returns:
            np.ndarray: The reduced 2D array of data points.
        """
        pass
