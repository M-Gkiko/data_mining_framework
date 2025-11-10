import numpy as np
from typing import List, Optional, Any
from sklearn.cluster import DBSCAN
from ...core.clustering import Clustering
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...utils.distance_utils import build_distance_matrix


class DBSCANClustering(Clustering):
    """    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
    together points that are closely packed while marking outliers as noise.
    
    Supports both custom distance measures and sklearn's built-in metrics.
    """
    
    def __init__(self, distance_measure: Optional[DistanceMeasure] = None, eps: float = 0.5, min_samples: int = 5, **kwargs: Any):
        """
        Initialize DBSCAN clustering with parameters.
        
        Args:
            distance_measure (Optional[DistanceMeasure]): Custom distance measure to use.
                                                        If provided, uses metric='precomputed'
            eps (float): Maximum distance between two samples for neighborhood
                        (epsilon radius). Default: 0.5
            min_samples (int): Minimum number of samples in neighborhood to
                             form a core point. Default: 5
            **kwargs: Additional parameters including:
                - metric (str): Built-in sklearn metric to use when no distance_measure provided
                              (e.g., 'euclidean', 'manhattan', 'cosine'). Default: 'euclidean'
        """
        self.distance_measure = distance_measure
        
        # Set defaults based on whether custom distance measure is provided
        if distance_measure is not None:
            # Use precomputed with custom distance
            default_metric = "precomputed"
        else:
            # Use built-in sklearn metric
            default_metric = kwargs.pop("metric", "euclidean")
        
        self.params = {
            "eps": eps,
            "min_samples": min_samples,
            "metric": default_metric
        }
        self.params.update(kwargs)
        
        self._dbscan = None
        self._labels = None
        self._fitted = False
    
    def fit(self, dataset: Dataset, **kwargs: Any) -> None:
        """
        Fit the DBSCAN clustering algorithm to the given dataset.
        
        Args:
            dataset (Dataset): The dataset to cluster
            **kwargs: Optional hyperparameters including:
                - eps (float): Maximum distance between samples for neighborhood
                - min_samples (int): Minimum samples in neighborhood for core point
                - metric (str): Distance metric for built-in sklearn metrics
                
        Raises:
            ValueError: If dataset is empty or invalid parameters are provided
        """
        # Allow runtime parameter overrides
        runtime_params = kwargs.copy()
        
        # Handle metric parameter in runtime overrides
        if "metric" in runtime_params and self.distance_measure is not None:
            raise ValueError(
                "Cannot override metric when using custom distance_measure. "
                "Custom distance measures always use metric='precomputed'"
            )
        
        self.params.update(runtime_params)
        
        X = dataset.get_data()
        if X is None:
            raise ValueError("Dataset.get_data() returned None.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one row.")
        
        eps = self.params["eps"]
        min_samples = self.params["min_samples"]
        metric = self.params["metric"]
        
        # Prepare data based on whether using custom distance or built-in metric
        if metric == "precomputed":
            if self.distance_measure is None:
                raise ValueError(
                    "metric='precomputed' requires a distance_measure parameter. "
                    "Either provide distance_measure in constructor or use a built-in metric"
                )
            # Build distance matrix using custom distance measure
            D = build_distance_matrix(X, self.distance_measure)
            self._dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="precomputed"
            )
            self._dbscan.fit(D)
        else:
            # Use built-in sklearn metric directly
            if self.distance_measure is not None:
                raise ValueError(
                    "distance_measure provided but using built-in metric. "
                    "To use custom distance measures, set metric='precomputed' or don't provide metric parameter"
                )
            self._dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric
            )
            self._dbscan.fit(X)
        
        self._labels = self._dbscan.labels_.tolist()
        self._fitted = True
    
    def get_labels(self) -> Optional[List[int]]:
        """
        Get the cluster labels for each data point.
        
        Returns:
            Optional[List[int]]: List of cluster labels for each data point,
                               or None if the algorithm hasn't been fitted yet.
                               Noise points have label -1, clusters labeled 0, 1, 2, ...
        """
        if not self._fitted:
            return None
        
        return self._labels.copy()  
    
    def get_n_clusters(self) -> Optional[int]:
        """
        Get the number of clusters found (excluding noise).
        
        Returns:
            Optional[int]: Number of clusters, or None if not fitted yet
        """
        if not self._fitted:
            return None
        
        unique_labels = set(self._labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        return len(unique_labels)
    
    def get_n_noise_points(self) -> Optional[int]:
        """
        Get the number of noise points found.
        
        Returns:
            Optional[int]: Number of noise points, or None if not fitted yet
        """
        if not self._fitted:
            return None
        
        return self._labels.count(-1)