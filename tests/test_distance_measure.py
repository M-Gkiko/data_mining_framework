"""
Tests for the DistanceMeasure abstract base class.
"""

import pytest
import unittest
import numpy as np
from core.distance_measure import DistanceMeasure


class TestDistanceMeasure(unittest.TestCase):
    """Test cases for DistanceMeasure abstract base class."""
    
    def test_distance_measure_is_abstract(self):
        """Test that DistanceMeasure cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DistanceMeasure()
    
    def test_distance_measure_interface_methods_exist(self):
        """Test that all required abstract methods are defined."""
        required_methods = ['calculate']
        for method in required_methods:
            self.assertTrue(hasattr(DistanceMeasure, method))
            self.assertTrue(callable(getattr(DistanceMeasure, method)))


class MockDistanceMeasure(DistanceMeasure):
    """Mock implementation for testing purposes (Euclidean distance)."""
    
    def calculate(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        if p1.shape != p2.shape:
            raise ValueError("Points must have the same dimensions")
        
        return np.sqrt(np.sum((p1 - p2) ** 2))


class TestMockDistanceMeasure(unittest.TestCase):
    """Test cases for mock DistanceMeasure implementation."""
    
    def setUp(self):
        self.distance_measure = MockDistanceMeasure()
    
    def test_mock_distance_measure_instantiation(self):
        """Test that mock distance measure can be instantiated."""
        self.assertIsInstance(self.distance_measure, DistanceMeasure)
    
    def test_calculate_same_points(self):
        """Test distance calculation for identical points."""
        point1 = [1, 2, 3]
        point2 = [1, 2, 3]
        distance = self.distance_measure.calculate(point1, point2)
        self.assertAlmostEqual(distance, 0.0)
    
    def test_calculate_different_points(self):
        """Test distance calculation for different points."""
        point1 = [0, 0]
        point2 = [3, 4]
        distance = self.distance_measure.calculate(point1, point2)
        self.assertAlmostEqual(distance, 5.0)
    
    def test_calculate_numpy_arrays(self):
        """Test distance calculation with numpy arrays."""
        point1 = np.array([1, 1])
        point2 = np.array([4, 5])
        distance = self.distance_measure.calculate(point1, point2)
        self.assertAlmostEqual(distance, 5.0)
    
    def test_calculate_incompatible_dimensions(self):
        """Test error handling for incompatible point dimensions."""
        point1 = [1, 2]
        point2 = [1, 2, 3]
        with self.assertRaises(ValueError):
            self.distance_measure.calculate(point1, point2)


if __name__ == '__main__':
    unittest.main()