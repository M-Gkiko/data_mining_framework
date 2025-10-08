import numpy as np
from typing import List, Optional, Any
from sklearn.cluster import DBSCAN
from core.clustering_algorithm import ClusteringAlgorithm
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure


class DBSCANClustering(ClusteringAlgorithm):
    """    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
    together points that are closely packed while marking outliers as noise.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN clustering with parameters.
        
        Args:
            eps (float): Maximum distance between two samples for neighborhood
                        (epsilon radius). Default: 0.5
            min_samples (int): Minimum number of samples in neighborhood to
                             form a core point. Default: 5
        """
        self.eps = eps
        self.min_samples = min_samples
        self._dbscan = None
        self._labels = None
        self._fitted = False
    
    def _build_distance_matrix(self, data: np.ndarray, distance_measure: DistanceMeasure) -> np.ndarray:
        """
        Build pairwise distance matrix using the custom distance measure.
        
        Args:
            data (np.ndarray): Data points array of shape (n_samples, n_features)
            distance_measure (DistanceMeasure): Custom distance measure to use
            
        Returns:
            np.ndarray: Symmetric distance matrix of shape (n_samples, n_samples)
        """
        n_samples = data.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        # Build symmetric distance matrix
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = distance_measure.calculate(data[i], data[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric matrix
        
        return distance_matrix
    
    def fit(self, dataset: Dataset, **kwargs: Any) -> None:
        """
        Fit the DBSCAN clustering algorithm to the given dataset.
        
        Args:
            dataset (Dataset): The dataset to cluster
            **kwargs: Optional hyperparameters including:
                - distance_measure (DistanceMeasure): The distance measure to use
                - eps (float): Maximum distance between samples for neighborhood
                - min_samples (int): Minimum samples in neighborhood for core point
            
        Raises:
            ValueError: If dataset is empty or invalid
        """
        if dataset.get_rows() == 0:
            raise ValueError("Dataset is empty")
        
        # Extract hyperparameters from kwargs
        distance_measure = kwargs.get('distance_measure')
        eps = kwargs.get('eps', self.eps)
        min_samples = kwargs.get('min_samples', self.min_samples)
        
        data = dataset.get_data()
        
        if hasattr(data, 'values'):
            data = data.values
        
        data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array")
        
        # Handle distance measure - use precomputed if provided, otherwise use euclidean
        if distance_measure is not None:
            # Build precomputed distance matrix using custom distance measure
            distance_matrix = self._build_distance_matrix(data, distance_measure)
            
            # Initialize sklearn DBSCAN with precomputed metric
            self._dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'
            )
            
            self._dbscan.fit(distance_matrix)
        else:
            # Use sklearn's default euclidean distance
            self._dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            self._dbscan.fit(data)
        
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
