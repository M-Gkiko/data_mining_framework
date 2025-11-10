from typing import Any, Optional
import numpy as np
from ...core.pipeline import PipelineComponent
from ...core.dimensionality_reduction import DimensionalityReduction
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...implementations.datasets import NumpyDataset


class DRAdapter(PipelineComponent):
    """
    Adapter that wraps DimensionalityReduction algorithms for pipeline execution.
    
    Handles Dataset input and produces numpy array output for next pipeline step.
    """
    
    def __init__(self, dr_algorithm: DimensionalityReduction, 
                 distance_measure: Optional[DistanceMeasure] = None,
                 name: str = None, **kwargs):
        """
        Initialize DR adapter.
        
        Args:
            dr_algorithm: The dimensionality reduction algorithm to wrap
            distance_measure: Optional distance measure for the algorithm
            name: Name for this component in the pipeline
            **kwargs: Algorithm-specific parameters
        """
        self.dr_algorithm = dr_algorithm
        self.distance_measure = distance_measure
        self.algorithm_params = kwargs
            
        super().__init__(name)
    
    def execute(self, input_data: Dataset) -> Dataset:
        """
        Execute dimensionality reduction algorithm.
        
        Args:
            input_data: Dataset to reduce
            
        Returns:
            Dataset: Reduced dimensionality data wrapped in NumpyDataset
            
        Raises:
            ValueError: If input data is not a Dataset
            RuntimeError: If DR algorithm execution fails
        """
        if not isinstance(input_data, Dataset):
            raise ValueError(f"DRAdapter expects Dataset, got {type(input_data)}")
        
        try:
            # Execute dimensionality reduction
            reduced_data = self.dr_algorithm.fit_transform(
                input_data, 
                **self.algorithm_params
            )
            
            if not isinstance(reduced_data, np.ndarray):
                raise RuntimeError(f"DR algorithm {self.dr_algorithm.__class__.__name__} must return np.ndarray")
            
            # Wrap the reduced numpy array in a NumpyDataset for next pipeline step
            return NumpyDataset(reduced_data)
            
        except Exception as e:
            raise RuntimeError(f"DR algorithm {self.dr_algorithm.__class__.__name__} failed: {str(e)}") from e