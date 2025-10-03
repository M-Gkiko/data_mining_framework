import numpy as np
from typing import Union
from core.distance_measure import DistanceMeasure


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
        
        dot_product = np.dot(p1, p2)
        
        # Calculate magnitudes (L2 norms)
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        
        # Handle zero vectors
        if norm_p1 == 0 or norm_p2 == 0:
            if np.array_equal(p1, p2): 
                return 0.0
            else:  
                return 1.0
        
        # Calculate cosine similarity
        cosine_similarity = dot_product / (norm_p1 * norm_p2)
        
        # Ensure cosine similarity is in valid range [-1, 1] (handle floating point errors)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        # Return cosine distance (1 - cosine similarity)
        return 1.0 - cosine_similarity
