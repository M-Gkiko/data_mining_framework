from typing import Union
import numpy as np
from ...core.distance_measure import DistanceMeasure
from scipy.spatial.distance import cityblock


class ManhattanDistance(DistanceMeasure):
    """    
    The Manhattan distance (L1 norm) calculates the sum of absolute differences
    between corresponding coordinates of two points. This is also known as 
    taxicab distance or city block distance.
    
    Formula: d(p1, p2) = Σ|p1[i] - p2[i]| for all dimensions i
    """
    
    def calculate(self, point1: Union[np.ndarray, list], point2: Union[np.ndarray, list]) -> float:
        """
        Calculate the Manhattan distance between two data points.
        
        Args:
            point1 (Union[np.ndarray, list]): First data point
            point2 (Union[np.ndarray, list]): Second data point
            
        Returns:
            float: Manhattan distance value between the two points
            
        Raises:
            ValueError: If points have incompatible dimensions
        """
        # Convert to numpy arrays for consistent processing
        p1 = np.asarray(point1)
        p2 = np.asarray(point2)
        
        # Validate dimensional compatibility
        if p1.shape != p2.shape:
            raise ValueError(
                f"Points must have the same dimensions. "
                f"Point 1 has shape {p1.shape}, Point 2 has shape {p2.shape}"
            )
        
        
        manhattan_dist = cityblock(p1, p2)
        return float(manhattan_dist)
