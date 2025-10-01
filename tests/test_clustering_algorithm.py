"""
Tests for the ClusteringAlgorithm abstract base class.
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock
from core.clustering_algorithm import ClusteringAlgorithm
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure


class TestClusteringAlgorithm(unittest.TestCase):
    """Test cases for ClusteringAlgorithm abstract base class."""
    
    def test_clustering_algorithm_is_abstract(self):
        """Test that ClusteringAlgorithm cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ClusteringAlgorithm()
    
    def test_clustering_algorithm_interface_methods_exist(self):
        """Test that all required abstract methods are defined."""
        required_methods = ['fit', 'get_labels']
        for method in required_methods:
            self.assertTrue(hasattr(ClusteringAlgorithm, method))
            self.assertTrue(callable(getattr(ClusteringAlgorithm, method)))


class MockClusteringAlgorithm(ClusteringAlgorithm):
    """Mock implementation for testing purposes."""
    
    def __init__(self):
        self._labels = None
        self._fitted = False
    
    def fit(self, dataset, distance_measure, **kwargs):
        """Mock fit implementation."""
        if dataset.get_rows() == 0:
            raise ValueError("Dataset is empty")
        
        # Simple mock clustering: assign alternating labels
        num_points = dataset.get_rows()
        self._labels = [i % 2 for i in range(num_points)]
        self._fitted = True
    
    def get_labels(self):
        """Mock get_labels implementation."""
        return self._labels if self._fitted else None


class TestMockClusteringAlgorithm(unittest.TestCase):
    """Test cases for mock ClusteringAlgorithm implementation."""
    
    def setUp(self):
        self.algorithm = MockClusteringAlgorithm()
        self.mock_dataset = Mock(spec=Dataset)
        self.mock_distance = Mock(spec=DistanceMeasure)
    
    def test_mock_clustering_algorithm_instantiation(self):
        """Test that mock clustering algorithm can be instantiated."""
        self.assertIsInstance(self.algorithm, ClusteringAlgorithm)
    
    def test_get_labels_before_fitting(self):
        """Test get_labels returns None before fitting."""
        labels = self.algorithm.get_labels()
        self.assertIsNone(labels)
    
    def test_fit_and_get_labels(self):
        """Test fitting algorithm and retrieving labels."""
        self.mock_dataset.get_rows.return_value = 4
        
        self.algorithm.fit(self.mock_dataset, self.mock_distance)
        labels = self.algorithm.get_labels()
        
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels, [0, 1, 0, 1])  # Alternating pattern
    
    def test_fit_with_empty_dataset(self):
        """Test error handling for empty dataset."""
        self.mock_dataset.get_rows.return_value = 0
        
        with self.assertRaises(ValueError) as context:
            self.algorithm.fit(self.mock_dataset, self.mock_distance)
        
        self.assertIn("Dataset is empty", str(context.exception))
    
    def test_fit_with_kwargs(self):
        """Test fitting with additional keyword arguments."""
        self.mock_dataset.get_rows.return_value = 3
        
        # Should not raise any errors
        self.algorithm.fit(self.mock_dataset, self.mock_distance, 
                          k=2, max_iterations=100)
        
        labels = self.algorithm.get_labels()
        self.assertEqual(len(labels), 3)


if __name__ == '__main__':
    unittest.main()