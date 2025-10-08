from typing import Any, Optional
import numpy as np
from core.pipeline import PipelineComponent
from core.dimensionality_reduction import DimensionalityReduction
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure


class DRAdapter(PipelineComponent):
    """
    Adapter that wraps DimensionalityReduction algorithms for pipeline execution.
    
    Handles Dataset input and produces numpy array output for next pipeline step.
    """
    
    def __init__(self, dr_algorithm: DimensionalityReduction, 
                 distance_measure: Optional[DistanceMeasure] = None,
                 name: Optional[str] = None, **kwargs):
        """
        Initialize DR adapter.
        
        Args:
            dr_algorithm: The dimensionality reduction algorithm to wrap
            distance_measure: Optional distance measure for the algorithm
            name: Optional custom name (defaults to algorithm class name)
            **kwargs: Algorithm-specific parameters
        """
        self.dr_algorithm = dr_algorithm
        self.distance_measure = distance_measure
        self.algorithm_params = kwargs
        
        # Use custom name or algorithm class name
        adapter_name = name or f"DR_{dr_algorithm.__class__.__name__}"
        super().__init__(adapter_name)
    
    def execute(self, input_data: Any) -> np.ndarray:
        """
        Execute dimensionality reduction algorithm.
        
        Args:
            input_data: Dataset or numpy array to reduce
            
        Returns:
            np.ndarray: Reduced dimensionality data
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If DR algorithm execution fails
        """
        # Handle different input types
        if isinstance(input_data, Dataset):
            dataset = input_data
        elif isinstance(input_data, np.ndarray):
            # Create temporary Dataset from numpy array (for chained operations)
            dataset = Dataset("temp_dataset", input_data)
        else:
            raise ValueError(f"DRAdapter expects Dataset or np.ndarray, got {type(input_data)}")
        
        try:
            # Execute dimensionality reduction
            reduced_data = self.dr_algorithm.fit_transform(
                dataset, 
                **self.algorithm_params
            )
            
            if not isinstance(reduced_data, np.ndarray):
                raise RuntimeError(f"DR algorithm {self.dr_algorithm.__class__.__name__} must return np.ndarray")
            
            return reduced_data
            
        except Exception as e:
            raise RuntimeError(f"DR algorithm {self.dr_algorithm.__class__.__name__} failed: {str(e)}") from e