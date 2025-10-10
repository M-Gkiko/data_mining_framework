from typing import Any, Optional, Dict, Union, Callable, Type
import numpy as np
from ...core.pipeline import PipelineComponent
from ...core.clustering import Clustering
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure


class ClusteringAdapter(PipelineComponent):
    """
    Adapter that wraps Clustering algorithms for pipeline execution.
    
    Handles both Dataset and numpy array inputs (from DR step).
    Returns comprehensive clustering results for quality measures.
    """
    
    def __init__(self, clustering_algorithm: Clustering,
                 distance_measure: Optional[DistanceMeasure] = None,
                 name: str = None,
                 **kwargs):
        """
        Initialize clustering adapter.
        
        Args:
            clustering_algorithm: The clustering algorithm to wrap
            distance_measure: Optional distance measure for the algorithm  
            name: Name for this component in the pipeline
            **kwargs: Algorithm-specific parameters
        """
        self.clustering_algorithm = clustering_algorithm
        self.distance_measure = distance_measure
        self.algorithm_params = kwargs
            
        super().__init__(name)
    
    def execute(self, input_data: Dataset) -> Dict[str, Any]:
        """
        Execute clustering algorithm.
        
        Args:
            input_data: Dataset object (conversion from np.ndarray should happen before this)
            
        Returns:
            Dict containing:
                - 'labels': Cluster labels for each data point
                - 'algorithm': Reference to fitted clustering algorithm
                - 'dataset': Dataset object used for clustering
                - 'original_data': Original input data
                - 'n_clusters': Number of clusters (if available)
                
        Raises:
            ValueError: If input data is not a Dataset
            RuntimeError: If clustering algorithm execution fails
        """
        if not isinstance(input_data, Dataset):
            raise ValueError(f"ClusteringAdapter expects Dataset, got {type(input_data)}")
        
        dataset = input_data
        
        try:
            # Execute clustering (distance_measure is now set in constructor)
            self.clustering_algorithm.fit(
                dataset,
                **self.algorithm_params
            )
            
            # Get clustering results
            labels = self.clustering_algorithm.get_labels()
            if labels is None:
                raise RuntimeError(f"Clustering algorithm {self.clustering_algorithm.__class__.__name__} returned None labels")
            
            # Prepare comprehensive results for next pipeline steps
            results = {
                'labels': labels,
                'algorithm': self.clustering_algorithm,
                'dataset': dataset,
                'original_data': input_data
            }
            
            # Add number of clusters if available
            if hasattr(self.clustering_algorithm, 'get_n_clusters'):
                try:
                    n_clusters = self.clustering_algorithm.get_n_clusters()
                    if n_clusters is not None:
                        results['n_clusters'] = n_clusters
                except:
                    pass  # Some algorithms might not support this
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Clustering algorithm {self.clustering_algorithm.__class__.__name__} failed: {str(e)}") from e