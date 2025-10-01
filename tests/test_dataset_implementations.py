"""
Unit tests for Dataset implementations.
"""

import pytest
import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from unittest.mock import patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.dataset import Dataset
from implementations.datasets import CSVDataset, NumpyDataset


class TestCSVDataset(unittest.TestCase):
    """Test cases for CSVDataset implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary CSV files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Valid CSV file
        self.valid_csv_path = os.path.join(self.temp_dir, 'valid.csv')
        with open(self.valid_csv_path, 'w') as f:
            f.write('feature1,feature2,feature3\n')
            f.write('1.0,2.0,3.0\n')
            f.write('4.0,5.0,6.0\n')
            f.write('7.0,8.0,9.0\n')
        
        # Empty CSV file
        self.empty_csv_path = os.path.join(self.temp_dir, 'empty.csv')
        with open(self.empty_csv_path, 'w') as f:
            f.write('')
        
        # CSV file with only header
        self.header_only_csv_path = os.path.join(self.temp_dir, 'header_only.csv')
        with open(self.header_only_csv_path, 'w') as f:
            f.write('col1,col2,col3\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_csv_dataset_is_dataset_instance(self):
        """Test that CSVDataset is an instance of Dataset."""
        dataset = CSVDataset(self.valid_csv_path)
        self.assertIsInstance(dataset, Dataset)
    
    def test_csv_dataset_valid_file(self):
        """Test CSVDataset with valid CSV file."""
        dataset = CSVDataset(self.valid_csv_path)
        
        # Test basic functionality
        self.assertEqual(dataset.get_rows(), 3)
        self.assertEqual(dataset.shape(), (3, 3))
        self.assertEqual(dataset.get_features(), ['feature1', 'feature2', 'feature3'])
        
        # Test data retrieval
        data = dataset.get_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (3, 3))
        expected_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(data.values, expected_values)
    
    def test_csv_dataset_file_not_found(self):
        """Test CSVDataset with non-existent file."""
        with self.assertRaises(FileNotFoundError) as context:
            CSVDataset('non_existent_file.csv')
        self.assertIn('CSV file not found', str(context.exception))
    
    def test_csv_dataset_empty_file(self):
        """Test CSVDataset with empty CSV file."""
        with self.assertRaises(pd.errors.EmptyDataError) as context:
            CSVDataset(self.empty_csv_path)
        self.assertIn('Error loading CSV file', str(context.exception))
    
    def test_csv_dataset_header_only_file(self):
        """Test CSVDataset with header-only CSV file."""
        with self.assertRaises(pd.errors.EmptyDataError) as context:
            CSVDataset(self.header_only_csv_path)
        self.assertIn('CSV file is empty', str(context.exception))
    
    def test_csv_dataset_custom_delimiter(self):
        """Test CSVDataset with custom delimiter."""
        # Create semicolon-delimited file
        semicolon_csv_path = os.path.join(self.temp_dir, 'semicolon.csv')
        with open(semicolon_csv_path, 'w') as f:
            f.write('a;b;c\n')
            f.write('1;2;3\n')
            f.write('4;5;6\n')
        
        dataset = CSVDataset(semicolon_csv_path, delimiter=';')
        self.assertEqual(dataset.get_rows(), 2)
        self.assertEqual(dataset.get_features(), ['a', 'b', 'c'])
    
    def test_csv_dataset_no_header(self):
        """Test CSVDataset without header row."""
        dataset = CSVDataset(self.valid_csv_path, header=None)
        features = dataset.get_features()
        self.assertEqual(len(features), 3)
        # Default pandas column names for no header
        self.assertTrue(all(isinstance(f, (int, str)) for f in features))
    
    def test_csv_dataset_data_immutability(self):
        """Test that returned data is a copy (immutable)."""
        dataset = CSVDataset(self.valid_csv_path)
        data1 = dataset.get_data()
        data2 = dataset.get_data()
        
        # Modify one copy
        data1.iloc[0, 0] = 999
        
        # Check that other copy is unchanged
        self.assertNotEqual(data1.iloc[0, 0], data2.iloc[0, 0])
    
    def test_csv_dataset_string_representation(self):
        """Test string representation of CSVDataset."""
        dataset = CSVDataset(self.valid_csv_path)
        str_repr = str(dataset)
        self.assertIn('CSVDataset', str_repr)
        self.assertIn(self.valid_csv_path, str_repr)
        self.assertIn('(3, 3)', str_repr)


class TestNumpyDataset(unittest.TestCase):
    """Test cases for NumpyDataset implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        self.feature_names = ['x', 'y', 'z']
    
    def test_numpy_dataset_is_dataset_instance(self):
        """Test that NumpyDataset is an instance of Dataset."""
        dataset = NumpyDataset(self.valid_data)
        self.assertIsInstance(dataset, Dataset)
    
    def test_numpy_dataset_valid_data(self):
        """Test NumpyDataset with valid numpy array."""
        dataset = NumpyDataset(self.valid_data, self.feature_names)
        
        # Test basic functionality
        self.assertEqual(dataset.get_rows(), 3)
        self.assertEqual(dataset.shape(), (3, 3))
        self.assertEqual(dataset.get_features(), self.feature_names)
        
        # Test data retrieval
        data = dataset.get_data()
        self.assertIsInstance(data, np.ndarray)
        np.testing.assert_array_equal(data, self.valid_data)
    
    def test_numpy_dataset_auto_feature_names(self):
        """Test NumpyDataset with automatically generated feature names."""
        dataset = NumpyDataset(self.valid_data)
        features = dataset.get_features()
        expected_features = ['feature_0', 'feature_1', 'feature_2']
        self.assertEqual(features, expected_features)
    
    def test_numpy_dataset_invalid_input_type(self):
        """Test NumpyDataset with invalid input type."""
        with self.assertRaises(ValueError) as context:
            NumpyDataset([[1, 2, 3], [4, 5, 6]])  # List instead of numpy array
        self.assertIn('Data must be a NumPy array', str(context.exception))
    
    def test_numpy_dataset_invalid_dimensions(self):
        """Test NumpyDataset with invalid array dimensions."""
        # 1D array
        with self.assertRaises(ValueError) as context:
            NumpyDataset(np.array([1, 2, 3, 4, 5]))
        self.assertIn('Data must be 2D array', str(context.exception))
        
        # 3D array
        with self.assertRaises(ValueError) as context:
            NumpyDataset(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
        self.assertIn('Data must be 2D array', str(context.exception))
    
    def test_numpy_dataset_empty_array(self):
        """Test NumpyDataset with empty array."""
        with self.assertRaises(ValueError) as context:
            NumpyDataset(np.array([]).reshape(0, 0))
        self.assertIn('Data array cannot be empty', str(context.exception))
    
    def test_numpy_dataset_mismatched_feature_names(self):
        """Test NumpyDataset with mismatched number of feature names."""
        wrong_feature_names = ['x', 'y']  # Only 2 names for 3 features
        with self.assertRaises(ValueError) as context:
            NumpyDataset(self.valid_data, wrong_feature_names)
        self.assertIn('Number of feature names', str(context.exception))
    
    def test_numpy_dataset_data_immutability(self):
        """Test that returned data is a copy (immutable)."""
        dataset = NumpyDataset(self.valid_data)
        data1 = dataset.get_data()
        data2 = dataset.get_data()
        
        # Modify one copy
        data1[0, 0] = 999
        
        # Check that other copy is unchanged
        self.assertNotEqual(data1[0, 0], data2[0, 0])
        # Check that original is unchanged
        self.assertNotEqual(data1[0, 0], self.valid_data[0, 0])
    
    def test_numpy_dataset_feature_names_immutability(self):
        """Test that returned feature names are a copy."""
        dataset = NumpyDataset(self.valid_data, self.feature_names)
        features1 = dataset.get_features()
        features2 = dataset.get_features()
        
        # Modify one copy
        features1[0] = 'modified'
        
        # Check that other copy is unchanged
        self.assertNotEqual(features1[0], features2[0])
    
    def test_numpy_dataset_string_representation(self):
        """Test string representation of NumpyDataset."""
        dataset = NumpyDataset(self.valid_data)
        str_repr = str(dataset)
        self.assertIn('NumpyDataset', str_repr)
        self.assertIn('(3, 3)', str_repr)
        self.assertIn('float64', str_repr)





class TestDatasetInterfaces(unittest.TestCase):
    """Test that all dataset implementations properly implement the Dataset interface."""
    
    def test_all_implementations_have_required_methods(self):
        """Test that all implementations have the required Dataset methods."""
        # Create instances of each implementation
        valid_data = np.array([[1, 2], [3, 4]])
        
        # Create temporary CSV for testing
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, 'test.csv')
        with open(csv_path, 'w') as f:
            f.write('a,b\n1,2\n3,4\n')
        
        try:
            datasets = [
                CSVDataset(csv_path),
                NumpyDataset(valid_data)
            ]
            
            required_methods = ['get_data', 'get_features', 'get_rows', 'shape']
            
            for dataset in datasets:
                self.assertIsInstance(dataset, Dataset)
                for method in required_methods:
                    self.assertTrue(hasattr(dataset, method))
                    self.assertTrue(callable(getattr(dataset, method)))
                    
                    # Test that methods return expected types
                    if method == 'get_features':
                        result = getattr(dataset, method)()
                        self.assertIsInstance(result, list)
                    elif method == 'get_rows':
                        result = getattr(dataset, method)()
                        self.assertIsInstance(result, int)
                        self.assertGreater(result, 0)
                    elif method == 'shape':
                        result = getattr(dataset, method)()
                        self.assertIsInstance(result, tuple)
                        self.assertEqual(len(result), 2)
                    elif method == 'get_data':
                        result = getattr(dataset, method)()
                        self.assertTrue(isinstance(result, (np.ndarray, pd.DataFrame)))
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main(verbose=2)