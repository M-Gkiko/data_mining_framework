"""
Adapter for clustering quality measures in pipelines.
"""

from typing import Any, Dict, Optional
from core.pipeline import PipelineComponent
from core.clustering_quality_measure import ClusteringQualityMeasure
from core.distance_measure import DistanceMeasure


class ClusteringQualityAdapter(PipelineComponent):
    """
    Adapter that wraps clustering quality measures for pipeline execution.
    
    Evaluates quality of clustering results.
    Expects clustering results dict from ClusteringAdapter.
    """
    
    def __init__(self, quality_measure: ClusteringQualityMeasure,
                 distance_measure: Optional[DistanceMeasure] = None,
                 name: Optional[str] = None, **kwargs):
        """
        Initialize clustering quality adapter.
        
        Args:
            quality_measure: The clustering quality measure to wrap
            distance_measure: Optional distance measure for the quality measure
            name: Optional custom name (defaults to measure class name)
            **kwargs: Quality measure specific parameters
        """
        self.quality_measure = quality_measure
        self.distance_measure = distance_measure
        self.measure_params = kwargs
        
        # Use custom name or quality measure class name
        adapter_name = name or f"ClusteringQuality_{quality_measure.__class__.__name__}"
        super().__init__(adapter_name)
    
    def execute(self, input_data: Any) -> Dict[str, float]:
        """
        Execute clustering quality evaluation.
        
        Args:
            input_data: Clustering results dict from ClusteringAdapter containing:
                       - 'labels': cluster labels
                       - 'algorithm': clustering algorithm instance
                       - 'dataset': dataset used for clustering
                       
        Returns:
            Dict with quality measure name as key and score as value
            
        Raises:
            ValueError: If input data format is invalid
            RuntimeError: If quality evaluation fails
        """
        if not isinstance(input_data, dict):
            raise ValueError(f"ClusteringQualityAdapter expects dict from ClusteringAdapter, got {type(input_data)}")
        
        # Extract required components
        required_keys = ['labels', 'algorithm', 'dataset']
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"ClusteringQualityAdapter missing required keys: {missing_keys}")
        
        labels = input_data['labels']
        clustering_algorithm = input_data['algorithm']
        dataset = input_data['dataset']
        
        try:
            # Evaluate clustering quality - quality measures expect dataset and labels
            score = self.quality_measure.evaluate(
                dataset,
                labels,
                **self.measure_params
            )
            
            if not isinstance(score, (int, float)):
                raise RuntimeError(f"Clustering quality measure {self.quality_measure.__class__.__name__} must return numeric score")
            
            # Return score with measure name as key
            return {self.name: float(score)}
            
        except Exception as e:
            raise RuntimeError(f"Clustering quality measure {self.quality_measure.__class__.__name__} failed: {str(e)}") from e