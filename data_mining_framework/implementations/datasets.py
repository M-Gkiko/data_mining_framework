import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional
from ..core.dataset import Dataset


class CSVDataset(Dataset):
    """
    Concrete implementation of Dataset for CSV files.
    
    This class provides access to CSV data through the Dataset interface,
    allowing CSV files to be used seamlessly with any clustering algorithm.
    """
    
    def __init__(self, file_path: str, delimiter: str = ',', header: Optional[Union[int, str]] = 0,
                 encoding: str = 'utf-8', **kwargs):
        """
        Initialize CSVDataset with a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            delimiter (str): Field delimiter (default: ',')
            header (Optional[Union[int, str]]): Row number(s) to use as column names (default: 0)
            encoding (str): File encoding (default: 'utf-8')
            **kwargs: Additional arguments passed to pandas.read_csv()
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
            pd.errors.ParserError: If there's an error parsing the CSV
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        self.file_path = file_path
        self.name = os.path.basename(file_path).split('.')[0]  # Extract filename without extension
        self.delimiter = delimiter
        self.header = header
        self.encoding = encoding
        self.kwargs = kwargs
        
        try:
            self._data = pd.read_csv(
                file_path, 
                delimiter=delimiter, 
                header=header,
                encoding=encoding,
                **kwargs
            )
            
            if self._data.empty:
                raise pd.errors.EmptyDataError("CSV file is empty")
                
        except Exception as e:
            raise type(e)(f"Error loading CSV file '{file_path}': {str(e)}")
    
    def get_data(self) -> pd.DataFrame:
        """
        Retrieve the complete dataset as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The full dataset
        """
        return self._data.copy()
    
    def get_features(self) -> List[str]:
        """
        Get the column names from the CSV file.
        
        Returns:
            List[str]: List of column names
        """
        return list(self._data.columns)
    
    def get_rows(self) -> int:
        """
        Get the number of rows in the CSV file.
        
        Returns:
            int: Number of data rows
        """
        return len(self._data)
    
    def get_columns(self) -> int:
        """
        Get the number of columns in the CSV file.
        
        Returns:
            int: Number of columns
        """
        return len(self._data.columns)
    
    def shape(self) -> Tuple[int, int]:
        """
        Get the dimensions of the CSV dataset.
        
        Returns:
            Tuple[int, int]: (number of rows, number of columns)
        """
        return self._data.shape
    
    def __str__(self) -> str:
        return f"CSVDataset(file='{self.file_path}', shape={self.shape()})"
    
    def __repr__(self) -> str:
        return self.__str__()


class NumpyDataset(Dataset):
    """
    Concrete implementation of Dataset for NumPy arrays.
    
    This class wraps NumPy arrays to provide the Dataset interface,
    enabling numpy-based data to work with clustering algorithms.
    """
    
    def __init__(self, data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize NumpyDataset with a NumPy array.
        
        Args:
            data (np.ndarray): The data array (2D expected)
            feature_names (Optional[List[str]]): Names for the features/columns
            
        Raises:
            ValueError: If data is not a 2D array or feature_names length doesn't match
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a NumPy array")
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got {data.ndim}D")
        
        if data.size == 0:
            raise ValueError("Data array cannot be empty")
        
        self._data = data.copy()
        
        if feature_names is None:
            self._feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        else:
            if len(feature_names) != data.shape[1]:
                raise ValueError(f"Number of feature names ({len(feature_names)}) "
                               f"must match number of columns ({data.shape[1]})")
            self._feature_names = list(feature_names)
    
    def get_data(self) -> np.ndarray:
        """
        Retrieve the complete dataset as a NumPy array.
        
        Returns:
            np.ndarray: The full dataset
        """
        return self._data.copy()
    
    def get_features(self) -> List[str]:
        """
        Get the feature names.
        
        Returns:
            List[str]: List of feature names
        """
        return self._feature_names.copy()
    
    def get_rows(self) -> int:
        """
        Get the number of rows in the dataset.
        
        Returns:
            int: Number of data rows
        """
        return self._data.shape[0]
    
    def shape(self) -> Tuple[int, int]:
        """
        Get the dimensions of the dataset.
        
        Returns:
            Tuple[int, int]: (number of rows, number of columns)
        """
        return self._data.shape
    
    def __str__(self) -> str:
        return f"NumpyDataset(shape={self.shape()}, dtype={self._data.dtype})"
    
    def __repr__(self) -> str:
        return self.__str__()