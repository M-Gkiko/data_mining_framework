from typing import Any, Dict, Optional
import numpy as np
from ...core.pipeline import PipelineComponent
from ...core.dimensionality_reduction_quality_measure import DRQualityMeasure
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure


class DRQualityAdapter(PipelineComponent):
    """
    Adapter that wraps DR quality measures for pipeline execution.
    
    Evaluates quality of dimensionality reduction results.
    Requires both original dataset and reduced data.
    """
    
    def __init__(self, quality_measure: DRQualityMeasure,
                 original_dataset: Dataset,
                 distance_measure: Optional[DistanceMeasure] = None,
                 name: str = None, **kwargs):
        """
        Initialize DR quality adapter.
        
        Args:
            quality_measure: The DR quality measure to wrap
            original_dataset: Original dataset before dimensionality reduction
            distance_measure: Optional distance measure for the quality measure
            name: Name for this component in the pipeline
            **kwargs: Quality measure specific parameters
        """
        self.quality_measure = quality_measure
        self.original_dataset = original_dataset
        self.distance_measure = distance_measure
        self.measure_params = kwargs
            
        super().__init__(name)
    
    def execute(self, input_data: Any) -> Dict[str, float]:
        """
        Execute DR quality evaluation.
        
        Args:
            input_data: Reduced data (Dataset) from DR adapter
            
        Returns:
            Dict with quality measure name as key and score as value
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If quality evaluation fails
        """
        if not isinstance(input_data, Dataset):
            raise ValueError(f"DRQualityAdapter expects Dataset (reduced data), got {type(input_data)}")
        
        try:
            # Extract numpy array from Dataset for quality evaluation
            reduced_data = input_data.get_data()
            if not isinstance(reduced_data, np.ndarray):
                raise ValueError(f"Dataset must contain numpy array data, got {type(reduced_data)}")
            
            # Evaluate DR quality
            score = self.quality_measure.evaluate(
                self.original_dataset,
                reduced_data,
                **self.measure_params
            )
            
            if not isinstance(score, (int, float)):
                raise RuntimeError(f"DR quality measure {self.quality_measure.__class__.__name__} must return numeric score")
            
            # Return score with measure name as key
            return {self.name: float(score)}
            
        except Exception as e:
            raise RuntimeError(f"DR quality measure {self.quality_measure.__class__.__name__} failed: {str(e)}") from e