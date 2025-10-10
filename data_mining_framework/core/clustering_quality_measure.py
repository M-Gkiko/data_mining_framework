from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from .dataset import Dataset


class ClusteringQualityMeasure(ABC):
    """
    Abstract base class for clustering quality measures in the data mining framework.
    
    This interface defines the contract that all quality measure implementations
    must follow, enabling the Strategy Pattern for clustering evaluation.
    """
    
    @abstractmethod
    def evaluate(self, dataset: Dataset, labels: List[int]) -> float:
        """
        Evaluate the quality of clustering results.
        
        Args:
            dataset (Dataset): The original dataset that was clustered
            labels (List[int]): Cluster labels assigned to each data point
            
        Returns:
            float: Quality score (interpretation depends on the specific measure)
            
        Raises:
            ValueError: If labels length doesn't match dataset size or
                       if labels contain invalid cluster assignments
        """
        pass