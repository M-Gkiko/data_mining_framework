"""
Tests for the QualityMeasure abstract base class.
"""

import pytest
import unittest
from unittest.mock import Mock
from core.clustering_quality_measure import QualityMeasure
from core.dataset import Dataset


class TestQualityMeasure(unittest.TestCase):
    """Test cases for QualityMeasure abstract base class."""
    
    def test_quality_measure_is_abstract(self):
        """Test that QualityMeasure cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            QualityMeasure()
    
    def test_quality_measure_interface_methods_exist(self):
        """Test that all required abstract methods are defined."""
        required_methods = ['evaluate']
        for method in required_methods:
            self.assertTrue(hasattr(QualityMeasure, method))
            self.assertTrue(callable(getattr(QualityMeasure, method)))


class MockQualityMeasure(QualityMeasure):
    """Mock implementation for testing purposes."""
    
    def evaluate(self, dataset, labels):
        """Mock evaluation implementation."""
        if len(labels) != dataset.get_rows():
            raise ValueError("Labels length doesn't match dataset size")
        
        if any(label < 0 for label in labels):
            raise ValueError("Invalid cluster assignments")
        
        # Simple mock quality: return number of unique clusters
        return len(set(labels))


class TestMockQualityMeasure(unittest.TestCase):
    """Test cases for mock QualityMeasure implementation."""
    
    def setUp(self):
        self.quality_measure = MockQualityMeasure()
        self.mock_dataset = Mock(spec=Dataset)
    
    def test_mock_quality_measure_instantiation(self):
        """Test that mock quality measure can be instantiated."""
        self.assertIsInstance(self.quality_measure, QualityMeasure)
    
    def test_evaluate_valid_labels(self):
        """Test evaluation with valid labels."""
        self.mock_dataset.get_rows.return_value = 4
        labels = [0, 1, 0, 2]
        
        score = self.quality_measure.evaluate(self.mock_dataset, labels)
        self.assertEqual(score, 3)  # 3 unique clusters
    
    def test_evaluate_single_cluster(self):
        """Test evaluation with single cluster."""
        self.mock_dataset.get_rows.return_value = 3
        labels = [0, 0, 0]
        
        score = self.quality_measure.evaluate(self.mock_dataset, labels)
        self.assertEqual(score, 1)  # 1 unique cluster
    
    def test_evaluate_mismatched_lengths(self):
        """Test error handling for mismatched dataset and labels lengths."""
        self.mock_dataset.get_rows.return_value = 3
        labels = [0, 1]  # Only 2 labels for 3 data points
        
        with self.assertRaises(ValueError) as context:
            self.quality_measure.evaluate(self.mock_dataset, labels)
        
        self.assertIn("Labels length doesn't match dataset size", str(context.exception))
    
    def test_evaluate_invalid_labels(self):
        """Test error handling for invalid cluster assignments."""
        self.mock_dataset.get_rows.return_value = 3
        labels = [0, 1, -1]  # Invalid negative label
        
        with self.assertRaises(ValueError) as context:
            self.quality_measure.evaluate(self.mock_dataset, labels)
        
        self.assertIn("Invalid cluster assignments", str(context.exception))
    
    def test_evaluate_empty_labels(self):
        """Test evaluation with empty labels."""
        self.mock_dataset.get_rows.return_value = 0
        labels = []
        
        score = self.quality_measure.evaluate(self.mock_dataset, labels)
        self.assertEqual(score, 0)  # No clusters


if __name__ == '__main__':
    unittest.main()