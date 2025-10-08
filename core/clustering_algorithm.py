from abc import ABC, abstractmethod
from typing import List, Optional, Any
import numpy as np
from .dataset import Dataset
from .distance_measure import DistanceMeasure


class ClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms in the data mining framework.
    
    This interface defines the contract that all clustering algorithm implementations
    must follow, enabling the Strategy Pattern for clustering operations.
    """
    
    @abstractmethod
    def fit(self, dataset: Dataset, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any) -> None:
        """
        Fit the clustering algorithm to the given dataset.
        
        Args:
            dataset (Dataset): The dataset to cluster
            distance_measure (Optional[DistanceMeasure]): The distance measure to use (default: None)
            **kwargs: Additional algorithm-specific hyperparameters
            
        Raises:
            ValueError: If dataset is empty or invalid
        """
        pass
    
    @abstractmethod
    def get_labels(self) -> Optional[List[int]]:
        """
        Get the cluster labels for each data point.
        
        Returns:
            Optional[List[int]]: List of cluster labels for each data point,
                               or None if the algorithm hasn't been fitted yet
        """
        pass