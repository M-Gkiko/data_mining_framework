"""
Dataset abstract base class for the data mining framework.

This module defines the Dataset interface following the Strategy Pattern,
allowing different data sources and formats to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
import numpy as np
import pandas as pd


class Dataset(ABC):
    """
    Abstract base class for datasets in the data mining framework.
    
    This interface defines the contract that all dataset implementations
    must follow, enabling the Strategy Pattern for data handling.
    """
    
    @abstractmethod
    def get_data(self) -> Union[np.ndarray, pd.DataFrame]:
        """
        Retrieve the complete dataset.
        
        Returns:
            Union[np.ndarray, pd.DataFrame]: The full dataset in its native format
        """
        pass
    
    @abstractmethod
    def get_features(self) -> List[str]:
        """
        Get the names/identifiers of all features in the dataset.
        
        Returns:
            List[str]: List of feature names or column identifiers
        """
        pass
    
    @abstractmethod
    def get_rows(self) -> int:
        """
        Get the number of rows (samples) in the dataset.
        
        Returns:
            int: Number of rows/samples in the dataset
        """
        pass
    
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """
        Get the dimensions of the dataset.
        
        Returns:
            Tuple[int, int]: (number of rows, number of columns/features)
        """
        pass