"""
Tests for the Dataset abstract base class.
"""

import pytest
import unittest
from abc import ABC
from core.dataset import Dataset


class TestDataset(unittest.TestCase):
    """Test cases for Dataset abstract base class."""
    
    def test_dataset_is_abstract(self):
        """Test that Dataset cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Dataset()
    
    def test_dataset_interface_methods_exist(self):
        """Test that all required abstract methods are defined."""
        required_methods = ['get_data', 'get_features', 'get_rows', 'shape']
        for method in required_methods:
            self.assertTrue(hasattr(Dataset, method))
            self.assertTrue(callable(getattr(Dataset, method)))


class MockDataset(Dataset):
    """Mock implementation for testing purposes."""
    
    def __init__(self):
        self._data = [[1, 2], [3, 4], [5, 6]]
        self._features = ['feature1', 'feature2']
    
    def get_data(self):
        return self._data
    
    def get_features(self):
        return self._features
    
    def get_rows(self):
        return len(self._data)
    
    def shape(self):
        return (len(self._data), len(self._features))


class TestMockDataset(unittest.TestCase):
    """Test cases for mock Dataset implementation."""
    
    def setUp(self):
        self.dataset = MockDataset()
    
    def test_mock_dataset_instantiation(self):
        """Test that mock dataset can be instantiated."""
        self.assertIsInstance(self.dataset, Dataset)
    
    def test_get_data(self):
        """Test get_data method."""
        data = self.dataset.get_data()
        expected = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(data, expected)
    
    def test_get_features(self):
        """Test get_features method."""
        features = self.dataset.get_features()
        expected = ['feature1', 'feature2']
        self.assertEqual(features, expected)
    
    def test_get_rows(self):
        """Test get_rows method."""
        rows = self.dataset.get_rows()
        self.assertEqual(rows, 3)
    
    def test_shape(self):
        """Test shape method."""
        shape = self.dataset.shape()
        self.assertEqual(shape, (3, 2))


if __name__ == '__main__':
    unittest.main()