from typing import List, Optional, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ...core.clustering import Clustering
from ...core.dataset import Dataset
from ...core.distance_measure import DistanceMeasure
from ...utils.distance_utils import build_distance_matrix


class HierarchicalClustering(Clustering):
    """
    Adapter for sklearn's AgglomerativeClustering that follows our
    ClusteringAlgorithm interface.

    Supports both custom distance measures and sklearn's built-in metrics.
    """

    def __init__(self, distance_measure: Optional[DistanceMeasure] = None, **kwargs: Any):
        """
        Initialize Hierarchical clustering with parameters.
        
        Args:
            distance_measure (Optional[DistanceMeasure]): Custom distance measure to use.
                                                        If provided, uses metric='precomputed'
            **kwargs: Additional parameters including:
                - n_clusters (int): Number of clusters to find (default: 2)
                - linkage (str): Linkage criterion ('ward', 'complete', 'average', 'single')
                - metric (str): Built-in sklearn metric to use when no distance_measure provided
                              (e.g., 'euclidean', 'manhattan', 'cosine')
        """
        self.distance_measure = distance_measure
        
        # Set smart defaults based on whether custom distance measure is provided
        if distance_measure is not None:
            # Use precomputed with custom distance - ward not compatible
            default_linkage = "complete"
            default_metric = "precomputed"
        else:
            # Use built-in sklearn metric - ward is compatible
            default_linkage = kwargs.pop("linkage", "ward")
            default_metric = kwargs.pop("metric", "euclidean")
        
        self.params = {
            "n_clusters": 2,
            "linkage": default_linkage,
            "metric": default_metric,
        }
        
        # Back-compat: allow 'affinity' as alias of 'metric'
        if "affinity" in kwargs and "metric" not in kwargs:
            kwargs = {**kwargs, "metric": kwargs.pop("affinity")}
        self.params.update(kwargs)

        self._labels: Optional[List[int]] = None
        self._model: Optional[AgglomerativeClustering] = None

    def fit(self, dataset: Dataset, **kwargs: Any) -> None:
        """
        Fit the hierarchical clustering algorithm to the given dataset.
        
        Args:
            dataset (Dataset): The dataset to cluster
            **kwargs: Optional hyperparameters including:
                - n_clusters (int): Number of clusters to find
                - linkage (str): Linkage criterion
                - metric (str): Distance metric for built-in sklearn metrics
                
        Raises:
            ValueError: If dataset is invalid or incompatible parameters are provided
        """
        # Allow runtime parameter overrides
        runtime_params = kwargs.copy()
        
        # Handle metric parameter in runtime overrides
        if "metric" in runtime_params and self.distance_measure is not None:
            raise ValueError(
                "Cannot override metric when using custom distance_measure. "
                "Custom distance measures always use metric='precomputed'"
            )
        
        # Back-compat: allow 'affinity' as alias of 'metric'
        if "affinity" in runtime_params and "metric" not in runtime_params:
            runtime_params["metric"] = runtime_params.pop("affinity")
        
        self.params.update(runtime_params)

        X = dataset.get_data()
        if X is None:
            raise ValueError("Dataset.get_data() returned None.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Dataset must be a 2D array with at least one row.")

        linkage = self.params["linkage"]
        metric = self.params["metric"]

        # Validate parameter compatibility
        if linkage == "ward":
            if metric != "euclidean":
                raise ValueError(
                    "linkage='ward' can only be used with metric='euclidean'. "
                    f"Got metric='{metric}'"
                )
            if self.distance_measure is not None:
                raise ValueError(
                    "linkage='ward' is incompatible with custom distance measures. "
                    "Use a different linkage or don't provide distance_measure"
                )

        if metric == "precomputed":
            if self.distance_measure is None:
                raise ValueError(
                    "metric='precomputed' requires a distance_measure parameter. "
                    "Either provide distance_measure in constructor or use a built-in metric"
                )
        else:
            if self.distance_measure is not None:
                raise ValueError(
                    "distance_measure provided but using built-in metric. "
                    "To use custom distance measures, set metric='precomputed' or don't provide metric parameter"
                )

        # Prepare data based on whether using custom distance or built-in metric
        if metric == "precomputed":
            # Build distance matrix using custom distance measure
            D = build_distance_matrix(X, self.distance_measure)
            self._model = AgglomerativeClustering(
                n_clusters=self.params["n_clusters"],
                linkage=linkage,
                metric="precomputed",
            )
            self._labels = self._model.fit_predict(D).tolist()
        else:
            # Use built-in sklearn metric directly
            self._model = AgglomerativeClustering(
                n_clusters=self.params["n_clusters"],
                linkage=linkage,
                metric=metric,
            )
            self._labels = self._model.fit_predict(X).tolist()

    def get_labels(self) -> Optional[List[int]]:
        """
        Get the cluster labels for each data point.
        
        Returns:
            Optional[List[int]]: List of cluster labels for each data point,
                               or None if the algorithm hasn't been fitted yet.
        """
        return self._labels.copy() if self._labels is not None else None
    
    def get_n_clusters(self) -> Optional[int]:
        """
        Get the number of clusters found.
        
        Returns:
            Optional[int]: Number of clusters, or None if not fitted yet
        """
        if self._labels is None:
            return None
        
        return len(np.unique(self._labels))