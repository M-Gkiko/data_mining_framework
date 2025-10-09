from core.distance_measure import DistanceMeasure
from typing import Union
import numpy as np
from scipy.spatial.distance import euclidean


class EuclideanDistance(DistanceMeasure):
    """
    Calculates the Euclidean distance between two points using SciPy.
    """

    def get_name(self) -> str:
        """Return the name of this distance measure."""
        return "Euclidean"

    def calculate(self, point1: Union[np.ndarray, list], point2: Union[np.ndarray, list]) -> float:
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (Union[np.ndarray, list]): First data point.
            point2 (Union[np.ndarray, list]): Second data point.

        Returns:
            float: Euclidean distance between the two points.

        Raises:
            ValueError: If points have incompatible dimensions.
        """
        p1 = np.asarray(point1, dtype=float)
        p2 = np.asarray(point2, dtype=float)

        if p1.shape != p2.shape:
            raise ValueError(f"Points must have the same dimensions: {p1.shape} vs {p2.shape}")

        return float(euclidean(p1, p2))
