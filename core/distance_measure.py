"""
Distance measure abstract base class for the data mining framework.

This module defines the DistanceMeasure interface following the Strategy Pattern,
allowing different distance/similarity metrics to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class DistanceMeasure(ABC):
    """
    Abstract base class for distance measures in the data mining framework.
    
    This interface defines the contract that all distance measure implementations
    must follow, enabling the Strategy Pattern for distance calculations.
    """
    
    @abstractmethod
    def calculate(self, point1: Union[np.ndarray, list], point2: Union[np.ndarray, list]) -> float:
        """
        Calculate the distance between two data points.
        
        Args:
            point1 (Union[np.ndarray, list]): First data point
            point2 (Union[np.ndarray, list]): Second data point
            
        Returns:
            float: Distance value between the two points
            
        Raises:
            ValueError: If points have incompatible dimensions
        """
        pass