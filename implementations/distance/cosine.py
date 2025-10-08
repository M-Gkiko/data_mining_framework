import numpy as np
from typing import Union
from core.distance_measure import DistanceMeasure
from scipy.spatial.distance import cosine


class CosineDistance(DistanceMeasure):
    """ 
    Calculates the cosine distance between two vectors, which is 1 minus the cosine similarity.
    Cosine distance = 1 - (A · B) / (||A|| * ||B||)
    
    The cosine distance ranges from 0 to 2:
    - 0: vectors are identical (cosine similarity = 1)
    - 1: vectors are orthogonal (cosine similarity = 0)
    - 2: vectors are opposite (cosine similarity = -1)
    """
    
    def calculate(self, point1: Union[np.ndarray, list], point2: Union[np.ndarray, list]) -> float:
        """
        Calculate cosine distance between two points.
        
        Args:
            point1 (Union[np.ndarray, list]): First data point
            point2 (Union[np.ndarray, list]): Second data point
            
        Returns:
            float: Cosine distance between the points (0 to 2)
            
        Raises:
            ValueError: If points have incompatible dimensions or are zero vectors
        """
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)
        
        if p1.shape != p2.shape:
            raise ValueError(f"Points must have the same dimensions: {p1.shape} vs {p2.shape}")
        
        
        # Handle zero vectors (scipy handles most edge cases)
        try:
            cosine_distance = cosine(p1, p2)
            # scipy returns nan for zero vectors, handle this case
            if np.isnan(cosine_distance):
                if np.array_equal(p1, p2):  # Both are identical (including zero vectors)
                    return 0.0
                else:
                    return 1.0  # One zero, one non-zero
            return float(cosine_distance)
        except Exception as e:
            raise ValueError(f"Error calculating cosine distance: {str(e)}")
